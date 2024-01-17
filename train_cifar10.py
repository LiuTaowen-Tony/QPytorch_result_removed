# %% [markdown]
# # CIFAR10 Low Precision Training Example
# In this notebook, we present a quick example of how to simulate training a deep neural network in low precision with QPyTorch.

# %%
# import useful modules
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from torch.optim import SGD
from qtorch import FloatingPoint
from tqdm import tqdm
import math

# %% [markdown]
# We first load the data. In this example, we will experiment with CIFAR10.

# %%
# loading data
ds = torchvision.datasets.CIFAR10
path = os.path.join("./data", "CIFAR10")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_set = ds(path, train=True, download=True, transform=transform_train)
test_set = ds(path, train=False, download=True, transform=transform_test)
loaders = {
        'train': torch.utils.data.DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            test_set,
            batch_size=128,
            num_workers=4,
            pin_memory=True
        )
}

# %% [markdown]
# We then define the quantization setting we are going to use. In particular, here we follow the setting reported in the paper "Training Deep Neural Networks with 8-bit Floating Point Numbers", where the authors propose to use specialized 8-bit and 16-bit floating point format.

# %%
# define two floating point formats
bit_8 = FloatingPoint(exp=5, man=2)
bit_16 = FloatingPoint(exp=6, man=9)

# define quantization functions
weight_quant = quantizer(forward_number=bit_8,
                        forward_rounding="nearest")
grad_quant = quantizer(forward_number=bit_8,
                        forward_rounding="nearest")
momentum_quant = quantizer(forward_number=bit_16,
                        forward_rounding="stochastic")
acc_quant = quantizer(forward_number=bit_16,
                        forward_rounding="stochastic")

# define a lambda function so that the Quantizer module can be duplicated easily
act_error_quant = lambda : Quantizer(forward_number=bit_8, backward_number=bit_8,
                        forward_rounding="nearest", backward_rounding="nearest")

# %% [markdown]
# Next, we define a low-precision ResNet. In the definition, we recursively insert quantization module after every convolution layer. Note that the quantization of weight, gradient, momentum, and gradient accumulator are not handled here.

# %%
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, quant, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.quant = quant()

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv1(out)
        out = self.quant(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv2(out)
        out = self.quant(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out
    
class PreResNet(nn.Module):

    def __init__(self,quant, num_classes=10, depth=20):

        super(PreResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = BasicBlock

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, quant)
        self.layer2 = self._make_layer(block, 32, n, quant, stride=2)
        self.layer3 = self._make_layer(block, 64, n, quant, stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.quant = quant()
        IBM_half = FloatingPoint(exp=6, man=9)
        self.quant_half = Quantizer(IBM_half, IBM_half, "nearest", "nearest")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, quant, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, quant , stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, quant))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant_half(x)
        x = self.conv1(x)
        x = self.quant(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)
        x = self.quant(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.quant_half(x)

        return x

# %%
model = PreResNet(act_error_quant)

# %%
device = 'cpu' # change device to 'cpu' if you want to run this example on cpu
model = model.to(device=device)

# %% [markdown]
# We now use the low-precision optimizer wrapper to help define the quantization of weight, gradient, momentum, and gradient accumulator.

# %%
optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
optimizer = OptimLP(optimizer,
                    weight_quant=weight_quant,
                    grad_quant=grad_quant,
                    momentum_quant=momentum_quant,
                    acc_quant=acc_quant,
                    grad_scaling=1/1000 # do loss scaling
)

# %% [markdown]
# We can reuse common training scripts without any extra codes to handle quantization.

# %%
def run_epoch(loader, model, criterion, optimizer=None, phase="train"):
    assert phase in ["train", "eval"], "invalid running phase"
    loss_sum = 0.0
    correct = 0.0

    if phase=="train": model.train()
    elif phase=="eval": model.eval()

    ttl = 0
    with torch.autograd.set_grad_enabled(phase=="train"):
        for i, (input, target) in tqdm(enumerate(loader), total=len(loader)):
            input = input.to(device=device)
            target = target.to(device=device)
            output = model(input)
            loss = criterion(output, target)
            loss_sum += loss.cpu().item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            ttl += input.size()[0]

            if phase=="train":
                loss = loss * 1000 # do loss scaling
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    correct = correct.cpu().item()
    return {
        'loss': loss_sum / float(ttl),
        'accuracy': correct / float(ttl) * 100.0,
    }

# %% [markdown]
# Begin the training process just as usual. Enjoy!

# %%
for epoch in range(1):
    train_res = run_epoch(loaders['train'], model, F.cross_entropy,
                                optimizer=optimizer, phase="train")
    test_res = run_epoch(loaders['test'], model, F.cross_entropy,
                                optimizer=optimizer, phase="eval")

# %%
train_res

# %%
test_res

# %%



