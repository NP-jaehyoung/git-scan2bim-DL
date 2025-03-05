import torch

checkpoint_path = "log/train/latest.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')  # CPU로 불러오기

# 체크포인트가 'state_dict' 형태로 저장되어 있는 경우
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print("=== State Dict Keys ===")
    for k in state_dict.keys():
        print(k)
else:
    # 만약 바로 모델 파라미터(OrderedDict)가 저장되어 있다면
    state_dict = checkpoint
    print("=== State Dict Keys ===")
    for k in state_dict.keys():
        print(k)
