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


class S3DISDataset(Dataset):
    def __init__(self, root_path, splits_path, split):
        super().__init__()
        self.root_path = root_path
        with open(path.join(splits_path, split + '.txt'), 'r') as f:
            self.items = [l.strip() for l in f]

    def __len__(self):
        return len(self.items)

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

class Structuralize:
    def __init__(self, xyz, labels):
        self.x = xyz
        self.y = labels


if __name__ == "__main__":
    root_path = "data/S3DIS/S3DIS_labeled/"
    splits_path = "data/S3DIS/S3DIS_labeled/"
    split = "Structuralize"

    dataset = S3DISDataset(root_path, splits_path, split)

    # DataLoader로 감싸서 사용
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for i, (xyz, labels, rgb) in enumerate(dataloader):
        print(f"Batch {i}: xyz shape={xyz.shape}, labels shape={labels.shape}, rgb shape={rgb.shape}")
        structuralize(xyz,labels)
        # 필요하면 break
        #break
