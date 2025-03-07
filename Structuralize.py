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
import os
import pandas as pd


def create_axes_at_corners(corners_3d, size=0.1):
    """
    corners_3d: shape=(N, 3)
    size: 각 축의 길이 스케일
    return: list of open3d.geometry.TriangleMesh objects
    """
    axes_list = []
    for corner in corners_3d:
        # 방법1: create_coordinate_frame 후 translate
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        #     size=size
        # )
        # frame.translate(corner)

        # 방법2: origin 인자를 직접 지정
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=size,
            origin=corner
        )
        axes_list.append(frame)
    return axes_list

def normalize_plane(plane_model):
    """
    plane_model = [a, b, c, d]
    returns normalized plane [a', b', c', d']
    such that ||(a',b',c')|| = 1
    """
    a, b, c, d = plane_model
    norm_len = np.sqrt(a*a + b*b + c*c)
    if norm_len < 1e-8:
        # fallback, assume z-up
        return np.array([0., 0., 1., 0.], dtype=np.float32)
    return np.array([a/norm_len, b/norm_len, c/norm_len, d/norm_len], dtype=np.float32)

def get_inclined_floor_corners(plane_model, floor_points):
    floor_points = np.asarray(floor_points.points)  # (N,3)
    """
    기울어진 바닥 평면과 그 위의 점들로부터,
    'AABB 4모서리'를 3D 좌표로 반환한다.

    Parameters:
    -----------
    plane_model : list/tuple [a, b, c, d]
        RANSAC 등으로 추정된 평면 방정식 (a*x + b*y + c*z + d = 0)
    floor_points : np.ndarray (N,3)
        바닥 라벨에 해당하는 점들 (평면 근처라고 가정)

    Returns:
    --------
    corners_3d : np.ndarray (4,3)
        바닥 모서리 4점의 3D 좌표 (시계 or 반시계 순)
    """

    # 1) 평면 법선 정규화
    a, b, c, d = plane_model
    norm_len = np.sqrt(a*a + b*b + c*c)
    if norm_len < 1e-8:
        # 만약 법선 벡터가 0에 가까우면 fallback
        print("Warning: invalid plane model. Returning None.")
        return None

    # 정규화된 법선, d
    n = np.array([a, b, c], dtype=np.float64) / norm_len
    d_ = d / norm_len  # 정규화된 d

    # 2) 평면 위 임의 점 (원점으로부터의 투영)
    # plane eq: n · x + d_ = 0  =>  n · x = -d_
    # x0 = -d_ * n  (이걸 plane origin으로 삼는다)
    x0 = -d_ * n

    # 3) 평면 위의 두 축 u, v
    #  - n과 직교하는 벡터 하나 구해서 -> 그것과 n의 cross로 v를 얻기
    #  - 여기서는 임의로 [1,0,0] 과 cross해보고, 만약 n이 [1,0,0]에 가까우면 다른 벡터를 쓴다.
    def find_orthogonal_vector(n):
        # n과 너무 평행하지 않은 축을 하나 골라 cross
        test_vec = np.array([1,0,0], dtype=np.float64)
        if abs(np.dot(n, test_vec)) > 0.9:
            test_vec = np.array([0,1,0], dtype=np.float64)
        u_ = np.cross(n, test_vec)
        u_norm = np.linalg.norm(u_)
        if u_norm < 1e-8:
            # fallback (n이 [0,0,1] 등인 경우 test_vec cross가 0이 될 수도)
            return np.array([0,1,0], dtype=np.float64)
        return u_ / u_norm

    u = find_orthogonal_vector(n)  # n에 직교
    v = np.cross(n, u)             # n, u에 모두 직교
    v /= np.linalg.norm(v)

    # 4) floor_points를 (u,v) 좌표계로 사영
    #    p' = p - x0;  px = p'·u, py = p'·v
    shifted_pts = floor_points - x0  # (N,3)
    px = np.dot(shifted_pts, u)      # shape=(N,)
    py = np.dot(shifted_pts, v)      # shape=(N,)

    # 5) AABB in 2D (px, py)
    px_min, px_max = px.min(), px.max()
    py_min, py_max = py.min(), py.max()

    # 2D corners (u,v)좌표계
    corners_2d = np.array([
        [px_min, py_min],
        [px_min, py_max],
        [px_max, py_max],
        [px_max, py_min]
    ], dtype=np.float64)

    # 6) 2D -> 3D 역변환
    # corner_3d = x0 + px_i*u + py_i*v
    corners_3d = []
    for (px_i, py_i) in corners_2d:
        corner_3d = x0 + px_i*u + py_i*v
        corners_3d.append(corner_3d)
    corners_3d = np.array(corners_3d)
    return corners_3d



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

        # or AABB
        # aabb = inlier_pcd.get_axis_aligned_bounding_box()
        # aabb.color = (0, 1, 0)

        # etc. 원하는 대로 저장
        self.obb = obb
        self.plane_model = plane_model
        self.inlier_pcd = inlier_pcd
        self.outlier_pcd = outlier_pcd

        # 필요하면 self.* 로 저장

class wall_structuralize:
    def __init__(self, xyz):
        """
        xyz: torch.Tensor (M,3) or np.ndarray (M,3)
             'wall'로 추정된 점들만 모아놓은 것
        """
        # 1. Tensor -> NumPy
        if isinstance(xyz, torch.Tensor):
            xyz_np = xyz.cpu().numpy()  # shape (M,3)
        else:
            xyz_np = xyz  # 이미 np일 수 있음

        xyz_np = xyz_np.astype(np.float32)  # float으로 변환

        # 2. Open3D point cloud
        self.wall_candidate_pcd = o3d.geometry.PointCloud()
        self.wall_candidate_pcd.points = o3d.utility.Vector3dVector(xyz_np)

        # 3. segment_plane
        plane_model, inliers = self.wall_candidate_pcd.segment_plane(
            distance_threshold=0.15,
            ransac_n=3,
            num_iterations=200
        )
        # inliers: shape (K,)  -> K <= M
        inlier_pcd = self.wall_candidate_pcd.select_by_index(inliers)
        outlier_pcd = self.wall_candidate_pcd.select_by_index(inliers, invert=True)

        # 4. 벽체 평면 인라이어 좌표
        inlier_points = xyz_np[inliers]  # shape=(K,3)

        wall_thk = 0.2
        offset = wall_thk / 2.0  # 0.1

        plane_model_n = normalize_plane(plane_model)  # [a', b', c', d']
        n = plane_model_n[:3]  # 법선 (단위벡터)
        d = plane_model_n[3]

        plane_model_plus = plane_model_n.copy()
        plane_model_plus[3] = d - offset

        plane_model_minus = plane_model_n.copy()
        plane_model_minus[3] = d + offset

        print("Central plane:", plane_model_n)
        print("Plane + offset:", plane_model_plus)
        print("Plane - offset:", plane_model_minus)


        # 4) 평면 인라이어 점들에 대한 OBB
        obb = inlier_pcd.get_oriented_bounding_box()
        obb.color = (1, 0, 0)

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
    wall_idx = idx_map['wall']
    ceiling_idx = idx_map['ceiling']

    #  enumerate : 순서가 있는 자료형(list, set, tuple, dictionary, string)을 입력으로 받았을 때, 인덱스와 값을 포함하여 리턴
    for i, (xyz, labels, rgb) in enumerate(dataloader):
        # xyz.shape = (1, N, 3)
        # labels.shape = (1, N)
        xyz = xyz.squeeze(dim=0)  # (N,3)
        labels = labels.squeeze(dim=0)  # (N,3)

        # 1) CNN이나 기존 라벨에서 특정 라벨만 추출
        floor_mask = (labels == floor_idx)  # shape=(N,)
        floor_candidate = xyz[floor_mask]    # shape=(M,3)

        ceiling_mask = (labels == ceiling_idx)  # shape=(N,)
        ceiling_candidate = xyz[ceiling_mask]    # shape=(M,3)

        wall_mask = (labels == wall_idx)  # shape=(N,)
        wall_candidate = xyz[wall_mask]    # shape=(M,3)

        print(f"Batch {i}: xyz={xyz.shape}, floor_candidate={floor_candidate.shape}")

        if floor_candidate.shape[0] == 0:
            print("No floor points found in this batch.")
            continue

        # 2) RANSAC 평면 분할
        floor = floor_structuralize(floor_candidate)
        wall = wall_structuralize(wall_candidate)
        ceiling = floor_structuralize(ceiling_candidate)

        # (3) 모서리 4점 찾기
        corners_3d_floor = get_inclined_floor_corners(floor.plane_model, floor.inlier_pcd)
        # corners_3d => (4,3) numpy array
        corners_3d_wall = get_inclined_floor_corners(wall.plane_model, wall.inlier_pcd)
        corners_3d_ceiling = get_inclined_floor_corners(ceiling.plane_model, ceiling.inlier_pcd)

        # 각 모서리에 축 표시
        corner_axes_floor = create_axes_at_corners(corners_3d_floor, size=0.15)
        corner_axes_wall = create_axes_at_corners(corners_3d_wall, size=0.15)
        corner_axes_ceiling = create_axes_at_corners(corners_3d_ceiling, size=0.15)

        pcd_list  = [floor.inlier_pcd, wall.inlier_pcd, ceiling.inlier_pcd]
        box_list  = [floor.obb, wall.obb, ceiling.obb]
        #o3d.visualization.draw_geometries(pcd_list + box_list + corner_axes_floor + corner_axes_wall + corner_axes_ceiling )

        print("corners_3d_floor=", corners_3d_floor)
        print("corners_3d_wall=", corners_3d_wall)
        print("corners_3d_ceiling=", corners_3d_ceiling)

        folder_path = "./results"
        os.makedirs(folder_path, exist_ok=True)  # 폴더 없으면 생성

        # (이름, 배열) 쌍을 리스트로 묶기
        data_list = [
            ("floor", corners_3d_floor),
            ("wall", corners_3d_wall),
            ("ceiling", corners_3d_ceiling),
        ]

        for name, arr in data_list:
            df = pd.DataFrame(arr)
            save_path = os.path.join(folder_path, f"corners_{name}.csv")
            df.to_csv(save_path, index=False, header=False)
            print(f"Saved '{name}' to {save_path}")


        # 필요하면 break
        #break


