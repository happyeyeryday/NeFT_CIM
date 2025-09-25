import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Type, List


class BayesianLinear(nn.Module):
    """贝叶斯全连接层"""
    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        
        # 权重的均值和标准差参数
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        
        # 偏置的均值和标准差参数
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))
        
        # 初始化参数
        self.reset_parameters()
        
    def reset_parameters(self):
    # 修改初始化策略
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_out', nonlinearity='relu')
        # 关键修改：将rho初始化改为更大的值
        nn.init.constant_(self.weight_rho, -5)  # 从-5改为-3，增加初始方差
        nn.init.constant_(self.bias_mu, 0)
        nn.init.constant_(self.bias_rho, -5)   # 从-5改为-3
    
    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # 重参数化采样
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        
        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """计算KL散度"""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # 计算权重的KL散度
        weight_kl = self._kl_divergence(self.weight_mu, weight_sigma, 0, self.prior_sigma)
        bias_kl = self._kl_divergence(self.bias_mu, bias_sigma, 0, self.prior_sigma)
        
        return weight_kl + bias_kl
    
    def _kl_divergence(self, mu_q, sigma_q, mu_p, sigma_p):
        """计算两个高斯分布之间的KL散度"""
        kl = torch.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5
        return kl.sum()


class BayesianConv2d(nn.Module):
    """贝叶斯卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, prior_sigma=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.prior_sigma = prior_sigma
        
        # 权重的均值和标准差参数
        self.weight_mu = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_rho = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        
        # 偏置参数（如果需要）
        if bias:
            self.bias_mu = nn.Parameter(torch.zeros(out_channels))
            self.bias_rho = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.weight_rho, -5)
        if self.bias_mu is not None:
            nn.init.constant_(self.bias_mu, 0)
            nn.init.constant_(self.bias_rho, -5)
    
    def forward(self, x):
        # 计算权重的标准差
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        
        # 重参数化采样权重
        weight_eps = torch.randn_like(self.weight_mu)
        weight = self.weight_mu + weight_sigma * weight_eps
        
        # 处理偏置
        bias = None
        if self.bias_mu is not None:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_eps = torch.randn_like(self.bias_mu)
            bias = self.bias_mu + bias_sigma * bias_eps
        
        return F.conv2d(x, weight, bias, self.stride, self.padding)
    
    def kl_divergence(self):
        """计算KL散度"""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_kl = self._kl_divergence(self.weight_mu, weight_sigma, 0, self.prior_sigma)
        
        if self.bias_mu is not None:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_kl = self._kl_divergence(self.bias_mu, bias_sigma, 0, self.prior_sigma)
            return weight_kl + bias_kl
        
        return weight_kl
    
    def _kl_divergence(self, mu_q, sigma_q, mu_p, sigma_p):
        kl = torch.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5
        return kl.sum()


def conv3x3_bayes(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1) -> BayesianConv2d:
    """3x3 贝叶斯卷积"""
    return BayesianConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def conv1x1_bayes(in_planes: int, out_planes: int, stride: int = 1, padding: int = 0) -> BayesianConv2d:
    """1x1 贝叶斯卷积"""
    return BayesianConv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=False)


class BayesianBasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        
        self.conv1 = conv3x3_bayes(in_planes, out_planes, stride=stride)
        self.conv2 = conv3x3_bayes(out_planes, out_planes, stride=1)
        self.unit_conv = conv1x1_bayes(in_planes, out_planes, stride=2) if stride != 1 or in_planes != out_planes else None
        
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn3 = nn.BatchNorm2d(out_planes) if self.unit_conv is not None else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.unit_conv is not None:
            identity = self.unit_conv(x)
            identity = self.bn3(identity)
        
        out += identity
        out = self.relu(out)
        return out
    
    def kl_divergence(self):
        kl = self.conv1.kl_divergence() + self.conv2.kl_divergence()
        if self.unit_conv is not None:
            kl += self.unit_conv.kl_divergence()
        return kl


class BayesianResNet(nn.Module):
    def __init__(self, in_channels, num_classes, block: Type[BayesianBasicBlock], layers: List[int]):
        super().__init__()
        self.conv1 = conv3x3_bayes(in_channels, 64, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.layer1 = self._make_layer(block, 64, 64, layers[0])
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = BayesianLinear(512, num_classes)
    
    def _make_layer(self, block: Type[BayesianBasicBlock], in_planes: int, out_planes: int, blocks: int, stride: int = 1):
        layers = []
        layers.append(block(in_planes, out_planes, stride=stride))
        for _ in range(1, blocks):
            layers.append(block(out_planes, out_planes, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        
        return x
    
    def kl_divergence(self):
        """计算整个网络的KL散度"""
        kl = self.conv1.kl_divergence() + self.fc1.kl_divergence()
        
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                kl += block.kl_divergence()
        
        return kl


def resnet18_bnn(in_channels, num_classes):
    """创建贝叶斯ResNet-18"""
    return BayesianResNet(in_channels, num_classes, BayesianBasicBlock, [2, 2, 2, 2])