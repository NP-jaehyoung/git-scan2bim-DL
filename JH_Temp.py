# 0. Import Library
"""
25.02.21 작성 RANSAC알고리즘으로 평면을 찾을 때, 큐브의 중간 높이에서 선택하는 경향이 큰 것으로 보임. 이를 해결하기 위한 방안이 필요함(높이를 기준으로 낮은 평면을 선택하는 등)

"""
import numpy as np
import time
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import copy
from os import path

class PlaneRetriever:
    def __init__(self, pcd):
        self.pcd = pcd  # 평면 색칠할 실제 point cloud
        self.cubeedge = len(pcd.points)

        z_min = np.percentile(np.asarray(pcd.points)[:, 2], 1)  # 하위 1% (바닥일 가능성 높은 값)
        z_max = z_min + 0.5  # 바닥 두께 (예: 50cm)

        # 바닥 후보점 추출
        floor_candidate_points = np.asarray(pcd.points)[
            (np.asarray(pcd.points)[:, 2] >= z_min) &
            (np.asarray(pcd.points)[:, 2] <= z_max)
        ]
        self.floor_candidate_pcd = o3d.geometry.PointCloud()
        self.floor_candidate_pcd.points = o3d.utility.Vector3dVector(floor_candidate_points)

        # NumPy 배열 변환
        pcd_np = np.asarray(self.pcd.points)  # 원본 포인트 클라우드 (N,3)
        floor_candidate_pcd_np = np.asarray(self.floor_candidate_pcd.points)  # 바닥 후보 (M,3)

        # --- 튜플 + set 사용해서 바닥 후보점 교집합 구하기 ---
        floor_set = set(tuple(pt) for pt in floor_candidate_pcd_np)  # 바닥 후보점 집합
        # 각 점이 바닥 후보점 집합에 속하는지 여부(True/False) 리스트 생성
        mask_list = [tuple(pt) in floor_set for pt in pcd_np]
        # 리스트를 NumPy 불리언 배열로 변환
        self.mask = np.array(mask_list, dtype=bool)

        self.intersection = pcd_np[self.mask]      # 교집합(바닥 후보에 해당하는 점)
        self.difference = pcd_np[~self.mask]      # 차집합(바닥 후보가 아닌 점)

        # 만약 PointCloud에 컬러 정보가 없다면 예외 처리
        if not self.pcd.has_colors():
            raise ValueError("PointCloud has no color information to revert to.")
        else:
            # 원본 색상 깊은 복사
            self.original_colors = np.array(self.pcd.colors)  # shape = (N, 3)

        print(f"self.cubeedge: {self.cubeedge}")
        print(f"Floor candidate points: {len(self.intersection)}")
        print(f"Non-floor points: {len(self.difference)}")

    def plane_update(self, vis):
        print("plane_updating...")
        # 평면 추출(전체 floor_candidate_pcd에 대해 RANSAC)
        plane_model, inliers = self.floor_candidate_pcd.segment_plane(
            distance_threshold=0.15,
            ransac_n=max(3, int(self.cubeedge * 0.2)),
            num_iterations=200
        )
        print(f"inliers: {len(inliers)}, mask : {np.count_nonzero(self.mask)}")
        # 시각화 업데이트
        inlier_indices = np.where(self.mask)[0]
        colors = np.asarray(self.pcd.colors)
        colors[inlier_indices] = [0.0, 0.0, 1.0] # 파란색
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        # 인라이어(평면에 속한 점) 빨강색으로 표시
        floor_indices = np.where(self.mask)[0] # mask=True인 점들의 원본 인덱스
        inlier_indices_in_pcd = floor_indices[inliers] # RANSAC 결과로 얻은 inliers(이는 floor_candidate_pcd 기준)이 원본 pcd에서 어느 인덱스인지 매핑
        colors = np.asarray(self.pcd.colors) # 원본 pcd.colors에 정확히 매핑된 인덱스를 칠하기
        colors[inlier_indices_in_pcd] = [1.0, 0.0, 0.0]  # 빨간색
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        vis.update_geometry(self.pcd)
        vis.poll_events()
        vis.update_renderer()

        # 다시 원본 컬러로 복원
        self.pcd.colors = o3d.utility.Vector3dVector(self.original_colors)

        print(f"Colored {len(inliers)} inlier points in red.")
        return False  # 콜백 계속 사용

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dset_path", type=str, default="data/s3DIS/S3DIS_labeled/", help='dataset path')
    parser.add_argument("--file_name", type=str, default="Area_1_conferenceRoom_1.ply", help='file name')
    args = parser.parse_args()

    if o3d.core.cuda.is_available():
        device = o3d.core.Device("CUDA:0")
        print("cuda available")
    # 이후 텐서 생성 시 device=device를 사용하여 GPU에서 연산 수행

    # 1. Load Data
    fname = path.join(args.dset_path, args.file_name)
    pcd = o3d.io.read_point_cloud(fname)

    print(f"Points before downsampling: {len(pcd.points)} ")
    pcd = pcd.voxel_down_sample(voxel_size=0.1)
    print(f"Points after downsampling: {len(pcd.points)}")

    # VisualizerWithKeyCallback 생성
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='Plane Color Example', width=800, height=600)

    # 시각화에 우리가 보고 싶은 실제 pcd를 등록
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])  # 회색
    vis.add_geometry(pcd)

    # 좌표축 표시 (옵션)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axes)

    # PlaneRemover 인스턴스
    Retriever = PlaneRetriever(pcd)

    # 스페이스 키 누를 때마다 plane_update 실행
    vis.register_key_callback(ord(" "), Retriever.plane_update)

    print("Press SPACE to segment plane and color inliers red. Press Q/ESC to exit.")

    # 카메라 파라미터 조정 (옵션)
    view_ctl = vis.get_view_control()
    cam_params = view_ctl.convert_to_pinhole_camera_parameters()
    cam_params.intrinsic.width = 800
    cam_params.intrinsic.height = 600
    view_ctl.convert_from_pinhole_camera_parameters(cam_params)

    # 시각화 이벤트 루프
    while True:
        if not vis.poll_events():
            break
        vis.update_renderer()

    # 창 닫기
    vis.destroy_window()


"""
    # ----------------------
    # 4. 바닥 평면 고르기 (수평 + 가장 낮은 z)
    # ----------------------
    lowest_z = float('inf')
    best_floor_index = None

    for i, (plane_model, inliers) in enumerate(candidate_planes):
        a, b, c, d = plane_model
        # 수평성 점검 => c=±1과 가까울수록 수평
        score = abs(abs(c) - 1.0)  # c=±1일 때 0, c=0일 때 1

        # 여기서는 수평판단을 위해 threshold 사용
        # 예: score < 0.2 => |c|>0.8 이상이면 "거의 수평"으로 본다
        if score < 0.2:
            # 이 평면에 속하는 점들의 평균 z
            plane_points = pcd.select_by_index(inliers)
            mean_z = np.asarray(plane_points.points)[:, 2].mean()

            # 가장 낮은 z를 찾는다
            if mean_z < lowest_z:
                lowest_z = mean_z
                best_floor_index = i

    # 5. 실제 바닥 평면 선택
    if best_floor_index is not None:
        floor_model, floor_inliers = candidate_planes[best_floor_index]
        inlier_cloud = pcd.select_by_index(floor_inliers)
        outlier_cloud = pcd.select_by_index(floor_inliers, invert=True)
        print("Selected floor plane model:", floor_model)
        print("Average Z of floor plane:", lowest_z)
    else:
        # "거의 수평" 평면을 하나도 못 찾았을 경우 fallback
        print("Could not find a suitable floor plane (horizontal + lowest). Using the largest plane as fallback.")
        floor_model, floor_inliers = candidate_planes[0]
        inlier_cloud = pcd.select_by_index(floor_inliers)
        outlier_cloud = pcd.select_by_index(floor_inliers, invert=True)

    inlier_cloud.paint_uniform_color([0, 1, 1])
    outlier_cloud.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    # 4. Clustering using DBSCAN -> To further segment objects on the road
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(outlier_cloud.cluster_dbscan(eps=0.4, min_points=50, print_progress=False))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    ## Get label colors
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    ## Colorized objects on the road
    outlier_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #o3d.visualization.draw_geometries([outlier_cloud])

    # 5. Generate 3D Bounding Boxes
    obbs = []  # Oriented Bounding Box의 약어이나, 실제론 get_axis_aligned_bounding_box()를 사용하고 있으니 축 방향 정렬 바운딩 박스(AABB)를 얻고 있음
    # AABB & OBB 차이 :  https://m.blog.naver.com/lyshyn/221033104965
    indexes = pd.Series(range(len(labels))).groupby(labels, sort=False).apply(list).tolist()

    MAX_POINTS = 300
    MIN_POINTS = 40

    ## For each individual object on the road
    for i in range(0, len(indexes)):
        nb_points = len(outlier_cloud.select_by_index(indexes[i]).points)
        # If object size within the criteria, draw bounding box
        if (nb_points > MIN_POINTS and nb_points < MAX_POINTS):
            sub_cloud = outlier_cloud.select_by_index(indexes[i])
            obb = sub_cloud.get_axis_aligned_bounding_box()
            obb.color = (0, 0, 1)
            obbs.append(obb)

    print(f"Number of Bounding Boxes calculated {len(obbs)}")

    ## Combined all visuals: outlier_cloud (objects), obbs (oriented bounding boxes), inlier_cloud (road)
    list_of_visuals = []
    list_of_visuals.append(outlier_cloud)
    list_of_visuals.extend(obbs)
    list_of_visuals.append(inlier_cloud)

    print(type(pcd))
    print(type(list_of_visuals))

    # 좌표축(코디네이트 프레임) 생성
    # inlier_cloud에서 AABB 구하기
    bbox = inlier_cloud.get_axis_aligned_bounding_box()

    # min_bound나 max_bound, 또는 get_center() 등을 활용할 수 있음
    outer_corner = bbox.min_bound  # (x_min, y_min, z_min)

    # 좌표축 생성 (size=1.0, origin=outer_corner)
    axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.0,
        origin=outer_corner
    )

    # 기존 list_of_visuals에 axis_frame 추가
    list_of_visuals.append(axis_frame)

    # 3) 시각화
    #o3d.visualization.draw_geometries(list_of_visuals)

"""


