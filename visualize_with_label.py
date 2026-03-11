# visualize_with_label.py

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np

def visualize_with_label(point_cloud, dataset, label_text="Hello, Open3D!"):
    """
    시각화 함수: Open3D GUI를 통해 포인트클라우드와 라벨을 시각화합니다.

    Args:
        point_cloud (o3d.geometry.PointCloud): 시각화할 포인트 클라우드 데이터
        dataset (PCSDataset): 색상 및 ID 맵을 참조할 PCSDataset 객체
        label_text (str): 윈도우 왼쪽 상단에 표시할 텍스트
    """
    # Open3D GUI 초기화
    gui.Application.instance.initialize()

    # 윈도우 생성
    window = gui.Application.instance.create_window("Open3D Label Example", 1280, 640)

    # SceneWidget 생성 및 윈도우에 추가
    scene_widget = gui.SceneWidget()
    scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene = scene
    window.add_child(scene_widget)

    # 포인트클라우드 추가
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    scene.add_geometry("pc", point_cloud, mat)

    # PCSDataset에서 cmap 및 idmap 가져오기
    cmap = dataset.init_cmap()  # 색상 맵 (numpy 배열)
    idmap = dataset.init_idmap()  # ID 맵 (딕셔너리)

    # ID 맵 순서를 기반으로 색상과 이름 정렬
    id_to_name = {v: k for k, v in idmap.items()}  # {ID: Name} 변환
    ordered_labels = [id_to_name[i] for i in range(len(id_to_name))]  # ID 순서대로 이름 정렬

    # Label 추가
    panel = gui.Vert(5, gui.Margins(10, 10, 10, 10))

    # 기본 라벨 추가
    title_label = gui.Label(label_text)
    title_label.text_color = gui.Color(1.0, 0.0, 0.0)  # 빨간색 텍스트
    panel.add_child(title_label)

    # Color map에 따른 레이블 추가
    for idx, label in enumerate(ordered_labels):
        rgb_color = cmap[idx] / 255.0  # 색상 정규화
        item_label = gui.Label(f"{label.capitalize()}")
        item_label.text_color = gui.Color(*rgb_color)
        panel.add_child(item_label)

    window.add_child(panel)

    # Layout 위치 조정
    def on_layout(layout_context):
        content_rect = window.content_rect
        constraints = gui.Widget.Constraints()
        pref_size = panel.calc_preferred_size(layout_context, constraints)
        panel.frame = gui.Rect(content_rect.x + 10, content_rect.y + 10, pref_size.width, pref_size.height)
        scene_widget.frame = gui.Rect(content_rect.x, content_rect.y, content_rect.width, content_rect.height)

    window.set_on_layout(on_layout)

    # 카메라 설정
    bounds = scene.bounding_box
    center = bounds.get_center()
    eye = [center[0], center[1], center[2] + 5.0]
    up = [0, 1, 0]
    scene.camera.look_at(center, eye, up)



    # GUI 실행
    gui.Application.instance.run()
