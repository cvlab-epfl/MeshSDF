import functools
import math
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import torch
import pdb

class _BatchInstanceNorm(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(_BatchInstanceNorm, self).__init__(num_features, eps, momentum, affine)
        self.gate = Parameter(torch.Tensor(num_features))
        self.gate.data.fill_(1)
        setattr(self.gate, 'bin_gate', True)

    def forward(self, input):
        self._check_input_dim(input)

        # Batch norm
        if self.affine:
            bn_w = self.weight * self.gate
        else:
            bn_w = self.gate
        out_bn = F.batch_norm(
            input, self.running_mean, self.running_var, bn_w, self.bias,
            self.training, self.momentum, self.eps)

        # Instance norm
        b, c  = input.size(0), input.size(1)
        if self.affine:
            in_w = self.weight * (1 - self.gate)
        else:
            in_w = 1 - self.gate
        input = input.view(1, b * c, *input.size()[2:])
        out_in = F.batch_norm(
            input, None, None, None, None,
            True, self.momentum, self.eps)
        out_in = out_in.view(b, c, *input.size()[2:])
        out_in.mul_(in_w[None, :, None, None])

        return out_bn + out_in


class BatchInstanceNorm1d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(input.dim()))


class BatchInstanceNorm2d(_BatchInstanceNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, normlayer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = normlayer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = normlayer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, normlayer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = normlayer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = normlayer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = normlayer(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, latent_size, depth, basicblock=False, norm_type="bin", inputchannels=4):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        if basicblock:
            block = BasicBlock
        else:
            block = Bottleneck if depth >=44 else BasicBlock

        if norm_type == 'bn':
            from torch.nn import BatchNorm2d
            self.normlayer = functools.partial(BatchNorm2d, affine=True)
        elif norm_type == 'in':
            from torch.nn import InstanceNorm2d
            self.normlayer = functools.partial(InstanceNorm2d, affine=True)
        elif norm_type == 'bin':
            self.normlayer = functools.partial(BatchInstanceNorm2d, affine=True)
        else:
            raise ValueError('Normalization should be either of type')



        self.inplanes = 64
        self.conv1 = nn.Conv2d(inputchannels, 64, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = self.normlayer(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, n)
        self.layer2 = self._make_layer(block, 128, n, stride=2)
        self.layer3 = self._make_layer(block, 256, n, stride=2)
        self.layer4 = self._make_layer(block, 256, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, latent_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, self.normlayer.func):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.normlayer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, normlayer=self.normlayer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, normlayer=self.normlayer))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature_1 = self.conv1(x)
        feature_1 = self.bn1(feature_1)
        feature_1 = self.relu(feature_1)
        feature_2 = self.layer1(feature_1)
        feature_3 = self.layer2(feature_2)
        feature_4 = self.layer3(feature_3)
        feature_out = self.layer4(feature_4)
        x = self.avgpool(feature_out)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
