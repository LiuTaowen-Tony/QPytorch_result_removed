from torch import nn
import torch
import math

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# class Probe(nn.Module):
#     def __init__(self, module: nn.Module):
#         super().__init__()
#         self.act_cache = None
#         self.module = module
#         self.act_no_detach_cache = None

#     def clean(self):
#         self.act_cache = None
#         self.act_no_detach_cache = None

#     def forward(self, x):
#         out = self.module(x)
#         if self.act_no_detach_cache is None and out.requires_grad:
#             out.retain_grad()
#             self.act_no_detach_cache = out
#         if self.act_cache is None and out.requires_grad:
#             self.act_cache = out.detach().clone()
#         return out

#     def probe_weight(self):
#         return self.module.weight.data.detach().clone()

#     def probe_activation(self):
#         temp = self.act_cache
#         self.act_cache = None
#         return temp

#     def probe_grad(self):
#         return self.module.weight.grad.data.detach().clone()

#     def probe_err(self):
#         temp = self.act_no_detach_cache
#         self.act_no_detach_cache = None
#         return temp.grad.data.detach().clone()
    

class Probe(nn.Module):
    def forward(self, x):
        return self.module(x)

    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

class fake_norm(nn.Module):
    def forward(self, x):
        return x
    
expansion = 1

class BasicBlock(nn.Module):

    def __init__(self,inplanes, planes, quant, stride=1, downsample=None, norm=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.bn1 = norm(inplanes)
        self.relu = nn.ReLU()
        self.probe = Probe(conv3x3(inplanes, planes, stride))
        self.bn2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.quant = quant()

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.quant(out)
        out = self.probe(out)
        out = self.quant(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.quant(out)
        out = self.conv2(out)
        out = self.quant(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.quant(out)

        return out
    
class PreResNet(nn.Module):

    def __init__(self,quant, num_classes=10, depth=20, norm=nn.BatchNorm2d):

        super(PreResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        def block(*args):
            return BasicBlock(*args, norm=norm)

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, quant)
        self.layer2 = self._make_layer(block, 32, n, quant, stride=2)
        self.layer3 = self._make_layer(block, 64, n, quant, stride=2)
        self.bn = norm(64 * expansion)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * expansion, num_classes)
        self.quant = quant()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, quant, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, quant , stride, downsample))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, quant))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
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
        x = self.quant(x)

        return x

    # def _apply_to_all_children(self, fn, cond):
    #     def recursive_apply(module):
    #         for child in module.children():
    #             recursive_apply(child)
    #         if cond(module):
    #             fn(module)
    #     recursive_apply(self)

    def get_probes(self):
        probes = {}
        for name, module in self.named_modules():
            if isinstance(module, Probe):
                probes[name] = module
        return probes

    def probe_weight(self):
        return {n:m.probe_weight() for n, m in self.get_probes().items()}

    def probe_activation(self):
        return {n:m.probe_activation() for n, m in self.get_probes().items()}
    
    def probe_grad(self):
        return {n:m.probe_grad() for n, m in self.get_probes().items()}
    
    def probe_err(self):
        return {n:m.probe_err() for n, m in self.get_probes().items()}

    def clean(self):
        for n, m in self.get_probes().items():
            m.clean()
    

