#
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from tqdm import tqdm

import torch
torch.backends.cudnn.benchmark = True
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from model.bimnet_S3DIS import BIMNet
from dataloaders.S3DISdataset import S3DISDataset
from visualize_with_label import visualize_with_label  # 분리된 함수 import

if __name__ == '__main__':

    cube_edge = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #cuda사용환경에선 cuda / 그 외엔 cpu
    model = BIMNet(num_classes=13)

    #if device == 'cuda':
    #    model.load_state_dict(torch.load("log/train_s3distest/latest.pth"))# epoch: 100 훈련 모델
    #else:
    #    model.load_state_dict(torch.load("log/train_s3distest/latest.pth", map_location=torch.device('cpu')))# epoch: 100 훈련 모델

    if device == 'cuda':
        model.load_state_dict(torch.load("log/train_s3distest/Pretrained_BIM-Net_S3DIS.pth"))# epoch: 100 훈련 모델
    else:
        model.load_state_dict(torch.load("log/train_s3distest/Pretrained_BIM-Net_S3DIS.pth", map_location=torch.device('cpu')))# epoch: 100 훈련 모델

    #model.load_state_dict(torch.load("log/train_s3distest/Pretrained_BIM-Net_S3DIS.pth"), strict=False) # GitHub제공자 훈련 모델// 02.19 BIMNET 채널 수정이 필요함
    model.to(device)

    dset = S3DISDataset(cube_edge=cube_edge,
                        root_path='data/S3DIS/S3DIS_labeled',
                        splits_path='data/S3DIS/S3DIS_labeled',
                      augment=False,
                      split='show')

    ids = np.indices((cube_edge, cube_edge, cube_edge)).reshape(3, -1).T # 24.12.29 추정 : 3D 공간의 각 점의 좌표를 나타냄 / T : Transpose / -1 : 자동계산

    # ids 전체 크기 확인
    print(f"ids shape: {ids.shape}")



    with torch.no_grad():
        for x, y in dset:
            y -= 1 #Unassign=0으로 라벨링 되어있는데  my = y.flatten()>=0으로 0은 false 처리해서 데이터 경량화
            print(f"x: {x.shape}, y: {y.shape}")
            cy = dset.color_label(y).reshape(-1, 3) # 24.12.29 추정 : 각 점(포인트)의 Ground Truth 색상 정보
            my = y.flatten()>=0 # 24.12.29 추정 : 유효한 점인지 여부를 나타내는 불리언 마스크(Boolean Mask).
            print(f"dset.color_label(y): {dset.color_label(y).shape},cy: {cy.shape}, my: {my.shape}, ids: {ids.shape}")

            #unsqueeze(dim)은 지정한 차원에 하나의 차원(크기 1)을 추가합니다.
            #x가 3D 데이터라면, [256, 256, 256] → [1, 256, 256, 256]으로 변환.
            x = x.to(device).unsqueeze(0)
            #squeeze(dim=None)은 크기가 1인 차원을 제거합니다.
            #x를 모델에 입력. 출력 크기: [1, C, 256, 256, 256] (예: C는 클래스 수).
            #출력 텐서의 클래스 축(dim=1)에서 가장 높은 값을 가진 인덱스를 반환. 결과 크기: [1, 256, 256, 256].
            #배치 차원(크기 1)을 제거. 결과 크기: [256, 256, 256]. GPU 텐서를 CPU 텐서로 변환.
            p = model(x).argmax(dim=1).squeeze(0).cpu()
            cp = dset.color_label(p).reshape(-1, 3)

            # 24.12.29 추정 : y : Ground Truth 데이터, 즉 실제 레이블 데이터. pcd: Point Cloud(점 구름). 3D 공간의 점으로 구성된 데이터. 즉 3D 좌표 데이터.
            ypcd = o3d.geometry.PointCloud() # 빈 PointCloud 객체 생성
            ypcd.points = o3d.utility.Vector3dVector(ids[my]) # 좌표 설정
            ypcd.colors = o3d.utility.Vector3dVector(cy[my,:3]) # 색상 설정
            print(f"ypcd.points: {np.asarray(ypcd.points).shape}, ypcd.colors: {np.asarray(ypcd.colors).shape}")

            # 24.12.29 추정 : p: Prediction 데이터, 즉 모델이 예측한 클래스 레이블 데이터. pcd: Point Cloud(점 구름). 3D 공간의 점으로 구성된 데이터. 즉 모델의 예측 결과를 기반으로 생성된 포인트 클라우드
            ppcd = o3d.geometry.PointCloud() # 빈 PointCloud 객체 생성
            ppcd.points = o3d.utility.Vector3dVector(ids[my])  # 좌표 설정 (ypcd와 동일)
            ppcd.colors = o3d.utility.Vector3dVector(cp[my,:3]) # 색상 설정

            # 각각 시각화 (Label 문구를 다르게 설정)
            visualize_with_label(ypcd, dataset= dset, label_text="GroundTruth Data")
            visualize_with_label(ppcd, dataset= dset, label_text="Prediction Data")

            #pcd = o3d.geometry.PointCloud()
            #pcd.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            #pcd.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            #o3d.visualization.draw_geometries([pcd])

            #vis = o3d.visualization.Visualizer()
            #vis.create_window(width=1280, height=640)
            #vis.add_geometry(ypcd)
            #vis.run()

            # 한 루프만 시연 후 종료하고 싶다면 break 추가
            break

            #vis = o3d.visualization.Visualizer()
            #vis.create_window(width=1280, height=640)
            #vis.add_geometry(ppcd)
            #vis.run()

            #vis.destroy_window()
            #vis.destroy_window()


