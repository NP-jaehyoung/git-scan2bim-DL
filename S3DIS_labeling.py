from tqdm import tqdm
import os
import numpy as np
from plyfile import PlyData, PlyElement

# 1) 전역 라벨 딕셔너리 (S3DIS 기준)
LABEL_DICT = {
    0: 'unassigned',
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
    13: 'clutter'
}
# "이름: 숫자" 형태로 뒤집기
LABEL_DICT = {v: k for k, v in LABEL_DICT.items()}


def write_ply(points, save_path):
    """
    points: shape (N, 7)
            각 행이 [x, y, z, r, g, b, class_id]
    """
    vertices = []
    for p in points:
        x, y, z, r, g, b, c = p
        vertices.append((x, y, z, int(r), int(g), int(b), int(c)))

    vertex = np.array(vertices, dtype=[
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
        ('class', 'i4')
    ])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(save_path)


def convert_txt_to_array(txt_path, class_name):
    try:
        # 1) 파일 읽기
        data = np.loadtxt(txt_path, dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] np.loadtxt failed at file: {txt_path}")
        raise e  # 에러를 다시 올려서 중단하거나, 원하는 방식으로 처리

    # 2) 라벨 처리
    try:
        label_id = LABEL_DICT.get(class_name.lower(), 0)
        c_col = np.full((data.shape[0], 1), label_id, dtype=np.int32)
        points_with_label = np.concatenate([data, c_col], axis=1)
    except Exception as e:
        print(f"[ERROR] Label processing failed at file: {txt_path}")
        raise e

    return points_with_label


def convert_only_annotation_txt(root_dir):
    """
    1) root_dir 아래 모든 폴더를 재귀 탐색
    2) 이름이 'Annotations'인 폴더를 찾음
    3) 그 안의 .txt 파일을 모두 읽어 하나로 합침
    4) 합쳐진 결과를 (Area_x, conferenceRoom_y)에 맞춰
       root_dir/S3DIS_labeled/폴더에 저장

    예:
      .../data/S3DIS/Area_1/conferenceRoom_1/Annotations/beam.txt, chair.txt ...
      --> 하나로 합쳐서
      --> .../data/S3DIS/S3DIS_labeled/Area_1_conferenceRoom_1.ply
    """
    # 1) S3DIS_labeled 폴더 생성 (없으면 만들기)
    labeled_dir = os.path.join(root_dir, "S3DIS_labeled")
    os.makedirs(labeled_dir, exist_ok=True)

    # 2) os.walk로 재귀 탐색
    for current_path, dirs, files in os.walk(root_dir):
        if os.path.basename(current_path) == 'Annotations':
            # Annotations 폴더 발견 -> 그 안의 txt 파일들을 모아 tqdm 진행률 표시
            txt_files = [f for f in files if f.endswith('.txt')]

            # 진행 상황 표시
            all_points = []
            for fname in tqdm(txt_files, desc=f"Processing {current_path}", unit="file"):
                txt_path = os.path.join(current_path, fname)

                # 파일명에서 class_name 추출
                base, _ = os.path.splitext(fname)
                base = base.split('_')[0]  # beam_1 -> beam
                label_id = LABEL_DICT.get(base.lower(), 0)
                # 필요하다면:
                # base = base.split('_')[0]  # wall_1 -> "wall"

                # txt -> (N,7) [x,y,z,r,g,b,class]
                array_ = convert_txt_to_array(txt_path, base)
                all_points.append(array_)

            # 3) 하나로 합쳐진 결과가 있으면 ply로 저장
            if len(all_points) > 0:
                merged_points = np.concatenate(all_points, axis=0)

                # 예) .../Area_1/conferenceRoom_1/Annotations
                #  -> parent_dir = "conferenceRoom_1"
                #  -> grandparent_dir = "Area_1"
                parent_dir = os.path.basename(os.path.dirname(current_path))
                grandparent_dir = os.path.basename(
                    os.path.dirname(os.path.dirname(current_path))
                )

                # 저장할 파일 이름 예: "Area_1_conferenceRoom_1.ply"
                ply_name = f"{grandparent_dir}_{parent_dir}.ply"

                # 최종 경로
                save_path = os.path.join(labeled_dir, ply_name)

                # ply로 쓰기
                write_ply(merged_points, save_path)
                tqdm.write(f"[Saved] {save_path} (merged {len(all_points)} txt files)")

    tqdm.write(f"Done converting only 'Annotations' folder .txt files under {root_dir}")


if __name__ == "__main__":
    # 예: data/S3DIS 아래 Area_1, conferenceRoom_1, Annotations, ...
    #     통째로 변환
    convert_only_annotation_txt("data/S3DIS")
