import numpy as np #수치 계산 및 배열 처리를 위한 라이브러리.
from tqdm import tqdm #진행 상황(progress bar)을 시각적으로 보여주는 라이브러리.
from tensorboardX import SummaryWriter #TensorBoard를 활용해 모델 학습 과정을 시각화.
from shutil import rmtree #파일 디렉토리를 삭제하는 데 사용.

import torch #PyTorch, 딥러닝 모델 학습을 위한 핵심 라이브러리.
torch.backends.cudnn.benchmark = True # cuDNN 최적화 설정. 입력 크기가 고정된 경우 성능을 높임.
from torch.optim import Adam # Adam 최적화 알고리즘.
from torch.utils.data import DataLoader #데이터셋을 쉽게 로드하고 배치로 처리.
from torch import nn #신경망 구성 요소(레이어, 손실 함수 등).
import argparse #명령줄에서 입력 인자를 처리하기 위한 라이브러리.


from model.bimnet import BIMNet #네트워크 모델 정의 (파일: model/bimnet.py).
from util.losses import ClassWiseCrossEntropyLoss, HNMCrossEntropyLoss #HNMCrossEntropyLoss: 맞춤형 손실 함수 정의 (파일: util/losses.py).
from dataloaders.PCSdataset import PCSDataset #데이터셋 로드 클래스 (파일: dataloaders/PCSdataset.py).
from util.metrics import Metrics #성능 지표를 계산하는 클래스 (파일: util/metrics.py).
from util.common_util import schedule, log_pcs #학습률 스케줄링 및 포인트 클라우드 로깅 함수.

#set seed for reproducibility #랜덤 시드를 고정하여 실험 결과의 재현성을 확보.
seed = 12345
np.random.seed(seed)
torch.manual_seed(seed)


###### VALIDATION
#Validation 수행: 모델 성능 평가.
#입력 매개변수:
#writer: 학습 결과를 TensorBoard에 기록.
#vset, vloader: 검증 데이터셋과 DataLoader.
#epoch: 현재 학습 epoch 번호.
#model: 학습 중인 모델.
#device: GPU 또는 CPU.
#출력: mIoU, 예측값 o, 실제값 y.
def validate(writer, vset, vloader, epoch, model, device): #PA, PP, mIoU
    metric = Metrics(vset.cnames[1:], device=device)
    model.eval()
    with torch.no_grad(): #그래디언트 계산을 비활성화해 메모리 사용을 줄임.
        for x, y in tqdm(vloader, "Validating Epoch %d"%(epoch+1), total=len(vset)):
            x, y = x.to(device), y.to(device, dtype=torch.long)-1 # shift indices 
            o = model(x)
            metric.add_sample(o.argmax(dim=1).flatten(), y.flatten())
            #break
    miou = metric.percent_mIoU()
    acc = metric.percent_acc()
    prec = metric.percent_prec()
    writer.add_scalar('mIoU', miou, epoch)
    writer.add_scalar('PP', prec, epoch)
    writer.add_scalar('PA', acc, epoch)
    writer.add_scalars('IoU', {n:100*v for n,v in zip(metric.name_classes, metric.IoU()) if not torch.isnan(v)}, epoch)
    print(metric)
    model.train()
    return miou, o, y

#파이썬에서 if __name__ == '__main__': 구문은 해당 파일이 직접 실행될 때만 특정 코드를 실행하도록 하는 조건문입니다.
#따라서, 이 구문을 사용하면 모듈을 다른 파일에서 import 할 때는 실행되지 않게 하고, 독립적으로 실행할 때만 실행되는 코드를 분리할 수 있습니다.
if __name__ == '__main__':
    #명령줄 옵션을 정의. 실행 시 사용자가 학습 epoch 수, 배치 크기, 데이터 경로 등을 설정 가능.
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500, help='number of epochs to run')
    parser.add_argument("--batch_size", type=int, default=4, help='batch_size')
    parser.add_argument("--cube_edge", type=int, default=64, help='granularity of voxelization train')
    parser.add_argument("--val_cube_edge", type=int, default=64, help='granularity of voxelization val')
    parser.add_argument("--num_classes", type=int, default=8, help='number of classes to consider')
    parser.add_argument("--dset_path", type=str, default="data/HePIC/", help='dataset path')
    parser.add_argument("--test_name", type=str, default='test', help='optional test name')
    parser.add_argument("--pretrain", type=str, help='pretrained model path')
    parser.add_argument("--loss", choices=['ce','cwce','ohem','mixed'], default='mixed', type=str, help='which loss to use')
    args = parser.parse_args()

    lr0 = 2.5e-4 #학습 초기 단계에서 사용할 시작 학습률입니다.
    lre = 1e-5 #학습 후반부에서 사용할 최종 학습률/ 학습이 진행됨에 따라 학습률을 점차 줄임으로써 모델이 안정적으로 최적화
    eval_every_n_epochs = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logdir = "log/train_pcs" + "_" + args.test_name
    rmtree(logdir, ignore_errors=True)
    writer = SummaryWriter(logdir, flush_secs=.5)

    # Load model
    #모델 초기화: 클래스 수 기반으로 네트워크 정의.
    #사전 학습 모델 로드: args.pretrain 옵션으로 경로 제공 시 모델 가중치 복구.
    model = BIMNet(args.num_classes)
    if args.pretrain:
        new = model.state_dict()
        old = torch.load(args.pretrain)
        for k in new:
            if "out" not in k:
                new[k] = old[k]
        model.load_state_dict(new)
        print("model restored from ", args.pretrain)
    model.to(device)
        
    # Load dataset
    dataset = PCSDataset
    dset = dataset(root_path=args.dset_path, #dset: 학습 데이터셋.
                   #fsl=50,
                   cube_edge=args.cube_edge)
    dloader = DataLoader(dset, #DataLoader가 데이터를 로드할 데이터셋 객체입니다.
                         batch_size=args.batch_size, #한 번에 로드할 데이터의 개수를 설정합니다.
                         shuffle=True, #데이터셋의 샘플 순서를 무작위로 섞을지 여부를 결정
                         num_workers=4, #데이터를 로드할 때 사용할 병렬 처리 작업자(worker) 프로세스의 수를 설정
                         drop_last=True) #데이터셋의 크기가 batch_size로 나누어떨어지지 않을 때, 마지막 미니배치를 버릴지 여부를 결정

    vset = dataset(root_path=args.dset_path, #vset: 검증 데이터셋
                   cube_edge=args.val_cube_edge,
                   augment=False,
                   split='val')
    vloader = DataLoader(vset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=4)


    # set up parameters for training
    steps_per_epoch = len(dset)//args.batch_size
    tot_steps = steps_per_epoch*args.epochs
    #Adam(Adaptive Moment Estimation)
    optim = Adam(model.parameters(), weight_decay=1e-5)
    
    # to visualize point cloud
    pts = 2*torch.from_numpy(np.indices((args.val_cube_edge, args.val_cube_edge, args.val_cube_edge))
                             .reshape(3, -1).T).unsqueeze(0)/args.cube_edge - 1.
    best_miou = 0
    #명령줄 옵션(--loss)에 따라 손실 함수를 선택.
    if args.loss == 'ce':
        loss = nn.CrossEntropyLoss(ignore_index=-1)
    elif args.loss == 'cwce':
        loss = ClassWiseCrossEntropyLoss(ignore_index=-1)
    elif args.loss == 'ohem':
        loss = HNMCrossEntropyLoss(ignore_index=-1)
    elif args.loss == 'mixed':
        loss1 = nn.CrossEntropyLoss(ignore_index=-1, weight=torch.sqrt(
                    torch.tensor(dset.weights[1:], dtype=torch.float32,
                                device=device)))  # weight=torch.tensor(dset.weights, dtype=torch.float32, device=device))
        loss2 = ClassWiseCrossEntropyLoss(ignore_index=-1, 
                    weight=torch.tensor(np.ones_like(dset.weights[1:]), dtype=torch.float32, device=device))
    else:
        raise NotImplementedError

    # TRAINING PHASE #훈련 루프
    for e in range(args.epochs): #Epoch 루프: args.epochs 동안 모델을 학습.
        torch.cuda.empty_cache()  #torch.cuda.empty_cache(): GPU 메모리 캐시를 비움.

        #Evaluate every n epochs #매 n epoch마다 검증 수행.
        if e % eval_every_n_epochs == 0:           
            if e>=0:
                miou, o, y = validate(writer, vset, vloader, e, model, device)
                if miou>best_miou:
                    best_miou = miou
                    torch.save(model.state_dict(), logdir+"/val_best.pth")
                #log_pcs(writer, dset, pts, o, y)
            metrics = Metrics(dset.cnames[1:], device=device)
       
        pbar = tqdm(dloader, total=steps_per_epoch, desc="Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress"%(e+1, args.epochs, 0., 0.))

    #Batch 루프: 데이터로더를 통해 배치 단위로 데이터 가져옴.
    # pbar는 tqbm형태로 묶은 dloader(튜플내 텐서)를 말하며,
    # i는 index를 x는 튜플 첫번째 여기선 3차텐서(좌표값),y는 튜플의 두번째 여기선 1차텐서(라벨)를 의미함
    #각 반복문이 진행되며 각 배치에 해당하는 값이 할당되어 conv수행
        for i, (x, y) in enumerate(pbar):

            step = i+steps_per_epoch*e

            #schedule 함수는 학습률을 lr0에서 lre로 점진적으로 감소시키는 역할을 합니다.
            #매 스텝(step)마다 학습률이 조정되며, 학습이 진행될수록 감소율이 줄어듭니다.
            lam = schedule(0, 1, step, tot_steps, .9) #schedule: 학습률과 손실 가중치 조정.
            lr = schedule(lr0, lre, step, tot_steps, .9)
            optim.param_groups[0]['lr'] = lr #학습률 스케줄링에 따라 옵티마이저(Adam)의 학습률을 업데이트
            optim.zero_grad() #모델의 모든 파라미터에 대해 계산된 그래디언트를 초기화
            
            x, y = x.to(device), y.to(device, dtype=torch.long)-1 # shift indices 
            
            o = model(x) #순전파 및 역전파: model(x) → 손실 계산 → 그래디언트 계산 및 업데이트.
            if args.loss == 'mixed':
                l = loss2(o, y) * (1 - lam) + loss1(o, y) * (lam)
            else:
                l = loss(o, y)
            l.backward() #PyTorch는 자동 미분(Autograd)을 통해 모델의 모든 학습 가능한 파라미터에 대한 그래디언트를 계산

            #모델의 예측값(predicted)과 실제값(ground_truth)을 성능 지표 클래스(metrics)에 전달하여 지표를 업데이트
            metrics.add_sample(o.detach().argmax(dim=1).flatten(), y.flatten()) #

            optim.step()
            miou = metrics.percent_mIoU()
            pbar.set_description("Epoch %d/%d, Loss: %.2f, mIoU: %.2f, Progress"%(e+1, args.epochs, l.item(), miou))
            
            writer.add_scalar('lr', lr, step)
            writer.add_scalar('loss', l.item(), step)
            writer.add_scalar('step_mIoU', miou, step)

        torch.save(model.state_dict(), logdir+"/latest.pth") #매 epoch 후 최신 모델 가중치를 저장.
        
    # EVALUATION 학습 완료 후 최종 검증. 최고 성능을 기록한 모델 저장.
    miou, o, y = validate(writer, vset, vloader, e, model, device)
    if miou>best_miou:
        best_miou = miou
        torch.save(model.state_dict(), logdir+"/val_best.pth")
