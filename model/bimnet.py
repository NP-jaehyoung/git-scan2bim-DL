import torch
from torch import nn
from torch.nn import functional as F

class SepConv(nn.Module):
    def __init__(self, inchs, outchs, kernel, dilation=1, project=True):
        super(SepConv, self).__init__()
        
        self.project = project
        if project:
            self.proj = nn.Conv3d(inchs, outchs, 1, bias=False)
            self.pbn = nn.BatchNorm3d(outchs)
        
        self.cx = nn.Conv3d(outchs, outchs, (kernel,1,1), padding=(dilation*(kernel//2),0,0), dilation=dilation, bias=False)
        self.cy = nn.Conv3d(outchs, outchs, (1,kernel,1), padding=(0,dilation*(kernel//2),0), dilation=dilation, bias=False)
        self.cz = nn.Conv3d(outchs, outchs, (1,1,kernel), padding=(0,0,dilation*(kernel//2)), dilation=dilation, bias=False)
        self.cbn = nn.BatchNorm3d(outchs)
    
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        if self.project:
            x = self.proj(x)
            x0 = self.pbn(x)
        else:
            x0 = x
        
        x = self.cx(x)
        x = self.cy(x)
        x = self.cz(x)
        x = self.cbn(x)+x0
        x = self.relu(x)
        
        return x

class SepInvRes(nn.Module):
    def __init__(self, inchs, midchs):
        super(SepInvRes, self).__init__()
        self.conv0 = nn.Conv3d(inchs, midchs, 1, bias=False)
        self.bn0 = nn.InstanceNorm3d(midchs)
        self.conv1 = SepConv(midchs, midchs, 7, project=False)
        self.bn1 = nn.InstanceNorm3d(midchs)
        self.conv2 = nn.Conv3d(midchs, inchs, 1, bias=False)
        
    def forward(self, x0):
        x = self.conv0(x0)
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)+x0
        return x

class BIMNet(nn.Module):
    #1. class BIMNet(nn.Module):
        # PyTorch의 기본 모듈인 nn.Module을 상속받아 새로운 신경망 클래스를 정의합니다.
        # BIMNet 이라는 이름을 가진 네트워크를 구성할 때, 필요한 레이어와 연산을 정의하고, 순전파 과정을 forward 메서드에서 구현하게 됩니다.
    #2. def __init__(self, num_classes=8):
        # 클래스가 초기화될 때 실행되는 생성자(Constructor)입니다.
        # num_classes는 최종 출력 채널(클래스 수)을 결정하는 인자입니다.
        # 기본값으로 8이 설정되어 있으며, 필요에 따라 변경 가능합니다.
    #3. super(BIMNet, self).__init__()
        # 부모 클래스(nn.Module)의 생성자를 호출하여 모듈 초기화를 수행합니다.

    def __init__(self, num_classes=8):
        super(BIMNet, self).__init__()
        # inch; inputChannel / outchs:outputChannel / kernel:커널
        self.conv0 = SepConv(1, 16, 7) # SepConv : SeperateConvolution ; 분리합성곱
        self.pool = nn.MaxPool3d(3, 2, padding=1) #풀링 커널 크기는 3, 스트라이드는 2, padding=1을 줌으로써 특성 맵의 공간/깊이 차원을 절반 정도로 줄이는 역할을 합니다

        #midchs:middleChannel / BottleNeck구조에서 사용
        self.rn1 = SepInvRes(16, 64) #입력 채널 16, 확장 채널(혹은 내부 채널)이 64로 설정된 Inverted Residual 블록(가정)입니다.
        self.bn1 = nn.InstanceNorm3d(16) # 3D 데이터에 대해 채널별로 정규화를 진행하는 Instance Normalization 레이어입니다.
        self.conv1 = SepConv(16, 8, 7) #분리합성곱 레이어로, 채널을 16에서 8로 줄이고 커널 사이즈는 7입니다.

        # 두 번째 Inverted Residual 블록은 입력 채널 8, 내부 채널 48로 설정되어 있습니다.
        # InstanceNorm3d와 분리합성곱 레이어를 차례로 거치며 채널 수를 다시 8 → 32로 확장합니다.
        self.rn2 = SepInvRes(8, 48)
        self.bn2 = nn.InstanceNorm3d(8)
        self.conv2 = SepConv(8, 32, 7)

        #세 번째 Inverted Residual 블록은 입력 32, 내부 채널 96입니다.
        #정규화와 분리합성곱을 거쳐 채널을 32 → 16으로 만듭니다.
        self.rn3 = SepInvRes(32, 96)
        self.bn3 = nn.InstanceNorm3d(32)
        self.conv3 = SepConv(32, 16, 7)

        #네 번째 Inverted Residual 블록은 입력 16, 내부 채널 64로 설정되어 있습니다.
        #정규화와 분리합성곱을 거치며 채널을 16 → 128로 크게 늘립니다.
        self.rn4 = SepInvRes(16, 64)
        self.bn4 = nn.InstanceNorm3d(16)
        self.conv4 = SepConv(16, 128, 7)

        #최종 출력 레이어.
        #128채널을 num_classes(기본값 8) 채널로 변환하고, 커널 사이즈는 3입니다.
        #보통 세그멘테이션 출력(차원 축소)에 사용합니다.
        self.out = SepConv(128, num_classes, 3)

        #활성화 함수로 ReLU를 사용합니다.
        #inplace=True로 설정할 수도 있는데, 이는 메모리 사용 최적화에 영향을 줍니다.
        self.relu = nn.ReLU()#inplace=True)



    #순전파(forward) 메서드
    #네트워크에 입력 x를 넣었을 때 일어나는 연산 과정을 기술합니다.
    def forward(self, x):
    
        x = self.conv0(x) #입력 x(채널 1짜리 3D 데이터)를 분리합성곱으로 처리하여 채널을 16으로 만듭니다.
        x = self.pool(x) #s2 #맥스풀링을 통해 공간/깊이 차원을 절반(스트라이드=2) 정도로 줄입니다(s2는 stride 2의 의미).
        x = self.relu(x) #ReLU 활성화 함수를 적용합니다.
        
        x = self.rn1(x) # 첫 번째 Inverted Residual 블록을 거칩니다(16 채널 → 내부 64 채널 과정을 수행).
        x = self.bn1(x) #채널 16개에 대해 Instance Normalization 적용.
        x = self.conv1(x) #분리합성곱으로 채널을 8로 만듭니다.
        x = self.pool(x) #s4 #다시 맥스풀링(stride=2)으로 공간/깊이 축소(s4는 총 4배 down-sampling).
        x = self.relu(x) #ReLU 활성화.
        
        x = self.rn2(x) #두 번째 Inverted Residual 블록 처리(8 → 내부 48).
        x = self.bn2(x) #채널 8개에 대해 Instance Normalization 적용.
        x = self.conv2(x) #분리합성곱으로 채널을 8에서 32로 확장.
        x = self.relu(x) # ReLU 활성화.

        x = self.rn3(x) #세 번째 Inverted Residual 블록 처리(32 → 내부 96).
        x = self.bn3(x) #채널 32개에 대해 Instance Normalization 적용.
        x = self.conv3(x) #분리합성곱으로 채널을 32에서 16으로 축소.
        x = self.relu(x) # ReLU 활성화.

        x = self.rn4(x) #네 번째 Inverted Residual 블록 처리(16 → 내부 64).
        x = self.bn4(x) # 채널 16개에 대해 Instance Normalization 적용.
        x = self.conv4(x) #분리합성곱으로 채널을 16에서 128로 확장.
        x = self.relu(x) # ReLU 활성화.
        
        x = self.out(x) #마지막 분리합성곱 레이어로 채널 수를 num_classes로 만듭니다(기본 8).

        #3D trilinear 보간법을 사용하여 특성 맵을 4배 확대합니다(공간/깊이 차원을 원래 크기로 복원하거나 특정 출력 해상도에 맞추기 위해).
        x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=True)

        #return x: 최종 출력(채널 수: num_classes)을 반환합니다.
        return x