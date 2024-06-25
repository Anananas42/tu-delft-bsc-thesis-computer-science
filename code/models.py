import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
from spikingjelly.clock_driven import functional, layer, surrogate, accelerating
from spikingjelly.clock_driven.neuron import BaseNode, LIFNode
from torchvision import transforms
import math

_seed_ = 2020
torch.manual_seed(_seed_)

def sample_log_normal(mean=2.0, std=0, sample_shape=None):
    # https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal-data-with-specified-mean-and-variance.html
    if std == 0:
      if sample_shape is None:
        return mean
      else:
        return torch.full(sample_shape, mean)

    var = std ** 2
    phi = torch.sqrt(torch.tensor([var + mean ** 2]))
    mean_log = torch.log(mean ** 2 / phi)
    sigma_log = torch.sqrt(torch.log(torch.tensor([phi ** 2 / (mean ** 2)])))
    dist = torch.distributions.LogNormal(mean_log.item(), sigma_log.item())
    if sample_shape is None:
      return dist.sample().item()
    return dist.sample(sample_shape)

class PLIFNode(BaseNode):
    def __init__(self, out_shape, init_tau=2.0, tau_std_dev=0, v_threshold=1.0, v_reset=0.0, detach_reset=True, surrogate_function=surrogate.ATan(), monitor_state=False, tau_per_neuron=False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, monitor_state)
        self.tau_per_neuron = tau_per_neuron
        if self.tau_per_neuron:
            init_tau_sample = sample_log_normal(init_tau, tau_std_dev, out_shape)
            init_w = - torch.log(init_tau_sample - 1)
        else:
            init_w = 0.0
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))

    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            self.v += (dv - self.v) * self.w.sigmoid()
        else:
            self.v += (dv - (self.v - self.v_reset)) * self.w.sigmoid()
        return self.spiking()

    def tau(self):
        if self.tau_per_neuron:
            return 1 / self.w.data.sigmoid()
        else:
            return 1 / self.w.data.sigmoid().item()
        

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau()}'

def create_conv_sequential(in_channels, out_channels, number_layer, init_tau, tau_std_dev, use_plif, use_max_pool, alpha_learnable, detach_reset, tau_per_neuron=False):
    # 首层是in_channels-out_channels
    # 剩余number_layer - 1层都是out_channels-out_channels
    conv = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        PLIFNode(out_shape=torch.Size([1, 128, 128, 128]), init_tau=init_tau, tau_std_dev=tau_std_dev, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset, tau_per_neuron=tau_per_neuron) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
        nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
    ]

    for i in range(number_layer - 1):
        conv.extend([
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            PLIFNode(out_shape=torch.Size([1, 128, 64 // (2 ** i), 64 // (2 ** i)]), init_tau=init_tau, tau_std_dev=tau_std_dev, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset, tau_per_neuron=tau_per_neuron) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
        ])
    return nn.Sequential(*conv)


def create_2fc(channels, h, w, dpp, class_num, init_tau, tau_std_dev, use_plif, alpha_learnable, detach_reset, tau_per_neuron=False):
    return nn.Sequential(
        nn.Flatten(),
        layer.Dropout(dpp),
        nn.Linear(channels * h * w, channels * h * w // 4, bias=False),
        PLIFNode(out_shape=torch.Size([1, 512]), init_tau=init_tau, tau_std_dev=tau_std_dev, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset, tau_per_neuron=tau_per_neuron) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
        layer.Dropout(dpp, dropout_spikes=True),
        nn.Linear(channels * h * w // 4, class_num * 10, bias=False),
        PLIFNode(out_shape=torch.Size([1, class_num * 10]), init_tau=init_tau, tau_std_dev=tau_std_dev, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset, tau_per_neuron=tau_per_neuron) if use_plif else LIFNode(tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
    )


class StaticNetBase(nn.Module):
    def __init__(self, T, init_tau, tau_std_dev, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.tau_std_dev = tau_std_dev
        self.use_plif = use_plif
        self.use_max_pool = use_max_pool
        self.alpha_learnable = alpha_learnable
        self.detach_reset = detach_reset
        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.static_conv = None
        self.conv = None
        self.fc = None
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        x = self.static_conv(x)
        out_spikes_counter = self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze(1)
        for t in range(1, self.T):
            out_spikes_counter += self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze(1)

        return out_spikes_counter

class MNISTNet(StaticNetBase):
    def __init__(self, T, init_tau, tau_std_dev, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__(T, init_tau=init_tau, tau_std_dev=tau_std_dev, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.conv = nn.Sequential(
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, dropout_spikes=use_max_pool),
            nn.Linear(128 * 7 * 7, 128 * 4 * 4, bias=False),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            layer.Dropout(0.5, dropout_spikes=True),
            nn.Linear(128 * 4 * 4, 100, bias=False),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset)
        )

class FashionMNISTNet(MNISTNet):
    pass  # 与MNISTNet的结构完全一致

class Cifar10Net(StaticNetBase):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__(T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        self.static_conv = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )

        self.conv = nn.Sequential(
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, dropout_spikes=use_max_pool),
            nn.Linear(256 * 8 * 8, 128 * 4 * 4, bias=False),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset),
            nn.Linear(128 * 4 * 4, 100, bias=False),
            PLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(learnable=alpha_learnable), detach_reset=detach_reset)
        )
def get_transforms(dataset_name):
    transform_train = None
    transform_test = None
    if dataset_name == 'MNIST':
        transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=30, translate=(0.15, 0.15), scale=(0.85, 1.11)),
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.1307, 0.3081),
        ])
    elif dataset_name == 'FashionMNIST':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.2860, 0.3530),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.2860, 0.3530),
        ])
    elif dataset_name == 'CIFAR10':
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
    return transform_train, transform_test

class NeuromorphicNet(nn.Module):
    def __init__(self, T, init_tau, tau_std_dev, use_plif, use_max_pool, alpha_learnable, detach_reset):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.tau_std_dev = tau_std_dev
        self.use_plif = use_plif
        self.use_max_pool = use_max_pool
        self.alpha_learnable = alpha_learnable
        self.detach_reset = detach_reset

        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.conv = None
        self.fc = None
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        out_spikes_counter = self.boost(self.fc(self.conv(x[0])).unsqueeze(1)).squeeze(1)
        for t in range(1, x.shape[0]):
            out_spikes_counter += self.boost(self.fc(self.conv(x[t])).unsqueeze(1)).squeeze(1)
        return out_spikes_counter

class NMNISTNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        w = 34
        h = 34  # 原始数据集尺寸
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >>number_layer, dpp=0.5, class_num=10, init_tau=init_tau, use_plif=use_plif, alpha_learnable=alpha_learnable, detach_reset=detach_reset)


class CIFAR10DVSNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, channels, number_layer, detach_reset):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        w = 128
        h = 128
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, use_plif=use_plif,
                                           use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >> number_layer, dpp=0.5, class_num=10,
                             init_tau=init_tau, use_plif=use_plif, alpha_learnable=alpha_learnable, detach_reset=detach_reset)


class Interpolate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwargs)

class ASLDVSNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, alpha_learnable, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        # input size 256 * 256
        w = 256
        h = 256

        self.conv = nn.Sequential(
            Interpolate(size=256, mode='bilinear'),
        )

class DVS128GestureNet(NeuromorphicNet):
    def __init__(self, T, init_tau, tau_std_dev, use_plif, use_max_pool, alpha_learnable, detach_reset, channels, number_layer, tau_per_neuron=False):
        super().__init__(T=T, init_tau=init_tau, tau_std_dev=tau_std_dev, use_plif=use_plif, use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset)
        w = 128
        h = 128
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, tau_std_dev=tau_std_dev, use_plif=use_plif,
                                           use_max_pool=use_max_pool, alpha_learnable=alpha_learnable, detach_reset=detach_reset, tau_per_neuron=tau_per_neuron)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >> number_layer, dpp=0.5, class_num=11,
                             init_tau=init_tau, tau_std_dev=tau_std_dev, use_plif=use_plif, alpha_learnable=alpha_learnable, detach_reset=detach_reset, tau_per_neuron=tau_per_neuron)