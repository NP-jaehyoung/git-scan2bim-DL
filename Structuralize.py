"""
 25.03.05 벽체 및 구조체 분류 및 구조화
    파이썬 매직 메서드 __getitem__은 “특정 객체에 대해 obj[idx]로 접근할 때 자동 실행”되는 원리
    파이토치 Dataset과 DataLoader를 사용하시려면 반드시 Dataset을 상속받은 클래스 안에 __len__, __getitem__을 정의하고, 그 클래스를 인스턴스화한 후 DataLoader에 전달해야 합니다.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from os import path
from plyfile import PlyData
import open3d as o3d

class S3DISDataset(Dataset):
    def __init__(self, root_path, splits_path, split):
        super().__init__()
        self.root_path = root_path
        with open(path.join(splits_path, split + '.txt'), 'r') as f:
            self.items = [l.strip() for l in f]

    def __len__(self):
        return len(self.items)

    def init_idmap(self):
        idmap = {0: 'unassigned',
                 1: 'ceiling',
                 2: 'floor',
                 3: 'wall',
                 4: 'beam',
                 5: 'column',
                 6: 'window',
                 7: 'door',
                 8: 'table',
                 9: 'chair',
                10: 'sofa',
                11: 'bookcase',
                12: 'board',
                13: 'clutter'}
        idmap = {v:k for k,v in idmap.items()}
        return idmap

    def init_cmap(self):
        cmap = np.array(  [[128, 64,128], # 0: unassigned
                           [244, 35,232], # 1: ceiling
                           [ 70, 70, 70], # 2: floor
                           [102,102,156], # 3: wall
                           [190,153,153], # 4: beam
                           [153,153,153], # 5: column
                           [250,170, 30], # 6: window
                           [220,220,  0], # 7: door
                           [107,142, 35], # 8: table
                           [152,251,152], # 9: chair
                           [ 70,130,180], # 10: sofa
                           [220, 20, 60], # 11: bookcase
                           [  0,  0,142], # 12: board
                           [  0,  0,  0]], dtype=np.uint8) # 13: clutter
        return cmap


    def __getitem__(self, idx):
        fname = path.join(self.root_path, self.items[idx])
        data = PlyData.read(fname)

        # (1) x, y, z 읽기
        xyz = np.array([data['vertex']['x'],
                        data['vertex']['y'],
                        data['vertex']['z']]).T

        # (2) 라벨 읽기
        lab = data['vertex'][['class']].astype(int)
        labels_np = np.array(lab, dtype=np.int32)

        # (3) rgb 읽기
        rgb = np.array([data['vertex']['red'],
                        data['vertex']['green'],
                        data['vertex']['blue']]).T

        return xyz, labels_np, rgb

class floor_structuralize:
    def __init__(self, xyz):
        """
        xyz: torch.Tensor (M,3) or np.ndarray (M,3)
             'floor'로 추정된 점들만 모아놓은 것
        """
        # 1. Tensor -> NumPy
        if isinstance(xyz, torch.Tensor):
            xyz_np = xyz.cpu().numpy()  # shape (M,3)
        else:
            xyz_np = xyz  # 이미 np일 수 있음

        xyz_np = xyz_np.astype(np.float32)  # float으로 변환

        # 2. Open3D point cloud
        self.floor_candidate_pcd = o3d.geometry.PointCloud()
        self.floor_candidate_pcd.points = o3d.utility.Vector3dVector(xyz_np)

        # 3. segment_plane
        plane_model, inliers = self.floor_candidate_pcd.segment_plane(
            distance_threshold=0.15,
            ransac_n=3,
            num_iterations=200
        )
        # inliers: shape (K,)  -> K <= M
        inlier_pcd = self.floor_candidate_pcd.select_by_index(inliers)
        outlier_pcd = self.floor_candidate_pcd.select_by_index(inliers, invert=True)

        # 4. 바닥 평면 인라이어 좌표
        inlier_points = xyz_np[inliers]  # shape=(K,3)

        # 5. z좌표 중앙 구하기
        z_min = inlier_points[:, 2].min() if len(inlier_points) > 0 else 0
        z_max = inlier_points[:, 2].max() if len(inlier_points) > 0 else 0
        z_centroid = (z_min + z_max) / 2.0
        floor_thk = 0.15

        # 6. plane_model -> [a, b, c, d], normal
        normal = plane_model[:3].astype(np.float32)
        normal_len = np.linalg.norm(normal)
        if normal_len > 1e-6:
            normal /= normal_len
        else:
            normal = np.array([0, 0, 1], dtype=np.float32)

        # 위/아래 offset
        floor_upper = z_centroid
        floor_lower = z_centroid - normal[2] * floor_thk

        print(f"plane_model={plane_model}, inliers={len(inliers)}")
        print(f"z_range=({z_min:.3f}, {z_max:.3f}), z_centroid={z_centroid:.3f}")
        print(f"floor_upper={floor_upper:.3f}, floor_lower={floor_lower:.3f}")

        # 4) 평면 인라이어 점들에 대한 OBB
        obb = inlier_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0)
        # 시각화(원한다면)
        o3d.visualization.draw_geometries([inlier_pcd, obb])

        # or AABB
        # aabb = inlier_pcd.get_axis_aligned_bounding_box()
        # aabb.color = (0, 1, 0)

        # etc. 원하는 대로 저장
        self.obb = obb
        self.plane_model = plane_model
        self.inlier_pcd = inlier_pcd
        self.outlier_pcd = outlier_pcd

        # 필요하면 self.* 로 저장

if __name__ == "__main__":
    root_path = "data/S3DIS/S3DIS_labeled/"
    splits_path = "data/S3DIS/S3DIS_labeled/"
    split = "Structuralize"

    dataset = S3DISDataset(root_path, splits_path, split)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    idx_map = dataset.init_idmap()
    floor_idx = idx_map['floor']

    #  enumerate : 순서가 있는 자료형(list, set, tuple, dictionary, string)을 입력으로 받았을 때, 인덱스와 값을 포함하여 리턴
    for i, (xyz, labels, rgb) in enumerate(dataloader):
        # xyz.shape = (1, N, 3)
        # labels.shape = (1, N)
        xyz = xyz.squeeze(dim=0)  # (N,3)
        labels = labels.squeeze(dim=0)  # (N,3)

        # 1) CNN이나 기존 라벨에서 floor 라벨만 추출
        floor_mask = (labels == floor_idx)  # shape=(N,)
        floor_candidate = xyz[floor_mask]    # shape=(M,3)

        print(f"Batch {i}: xyz={xyz.shape}, floor_candidate={floor_candidate.shape}")

        if floor_candidate.shape[0] == 0:
            print("No floor points found in this batch.")
            continue

        # 2) RANSAC 평면 분할
        struct = floor_structuralize(floor_candidate)

        # 필요하면 break
        #break


