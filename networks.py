import numpy as np
import torch
import torch.nn as nn
import functools
from losses import *
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision import models
from attention import MyAttention
import torch.nn.functional as F
from modules.attention.cbam import ChannelAttention, SpatialAttention, _ChannelAttention, _SpatialAttention
import attention
from trans import base_trans
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange as rearrange
import math
from options import MyOptions
class DropBlock(nn.Module):
    def __init__(self, block_size: int = 5, p: float = 0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x: Tensor) -> float:
        """计算gamma
        Args:
            x (Tensor): 输入张量
        Returns:
            Tensor: gamma
        """

        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Last_Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Last_Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            DropBlock(7, 0.18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # else:
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, 1, keepdim=True)
        x3 = torch.cat((x1, x2), dim=1)
        x4 = torch.sigmoid(self.conv(x3))
        x = x4 * x
        assert len(x.shape) == 4, f"好像乘不了"
        return x


# 先验信息修复网络
class LBPGenerator(nn.Module):
    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 1,
                 bilinear: bool = False,
                 base_c: int = 8,
                 num_block=[1, 2, 3, 4], num_head=[1, 2, 4, 8], factor=2.66):
        super(LBPGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.conv1 = DoubleConv(in_channels, 32)
        self.down1 = Down(32, base_c * 8)
        self.down2 = Down(base_c * 8, base_c * 16)
        self.down3 = Down(base_c * 16, base_c * 32)
        self.down4 = Last_Down(base_c * 32, base_c * 64)
        self.attn = Attention()
        self.conv2 = nn.Sequential(nn.Conv2d(base_c * 64, base_c * 64, kernel_size=3, padding=1, bias=False),
                                   DropBlock(7, 0.9),
                                   nn.BatchNorm2d(base_c * 64),
                                   nn.ReLU(inplace=True))
        self.up1 = Up(base_c * 64, base_c * 32, bilinear)
        self.up2 = Up(base_c * 32, base_c * 16, bilinear)
        self.up3 = Up(base_c * 16, base_c * 8, bilinear)
        self.up4 = Up(base_c * 8, base_c, bilinear)
        self.out_conv = OutConv(8, 1)

    def forward(self, lbp, mask):

        dn1 = torch.cat([lbp, mask], 1)
        x1 = self.conv1(dn1)  # 8
        x2 = self.down1(x1)  # 64
        x3 = self.down2(x2)  # 128
        x4 = self.down3(x3)  # 256
        x5 = self.down4(x4)  # 512
        x6 = self.attn(x5)  # 512
        x7 = self.conv2(x6)  # 512
        x8 = self.up1(x7, x4)  # 256
        x9 = self.up2(x8, x3)  # 128
        x10 = self.up3(x9, x2)  # 64
        x11 = self.up4(x10, x1)  # 8

        logits = self.out_conv(x11) + lbp
        plbp = logits * mask + lbp * (1 - mask)

        return plbp

class ImageGenerator(nn.Module):
    def __init__(self, factor=2.66):
        super(ImageGenerator, self).__init__()
        use = False
        ngf = 64
        self.dn11 = nn.Sequential(
            spectral_norm(nn.Conv2d(5, ngf * 1, kernel_size=4, stride=2, padding=1), use),
        )
        self.dn21 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 1, ngf * 2, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 2),
        )
        self.dn31 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 4),
        )

        self.trand256 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 4, head=8, expansion_factor=factor) for i in range(1)]
        )

        self.dn41 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )

        self.trand512 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 8, head=8, expansion_factor=factor) for i in range(1)]
        )

        self.dn51 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.dn61 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.dn71 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.bottle1 = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.up71 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.up61 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.up51 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 8),
        )
        self.trand512 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 8, head=8, expansion_factor=factor) for i in range(1)]
        )
        self.up41 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 4),
        )
        self.trand256 = nn.Sequential(
            *[TransformerEncoder(in_ch=ngf * 4, head=8, expansion_factor=factor) for i in range(1)]
        )
        self.up31 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 2),
        )
        self.up21 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 2 * 2, ngf * 1, kernel_size=4, stride=2, padding=1), use),
            nn.InstanceNorm2d(ngf * 1),
        )
        self.up11 = nn.Sequential(
            nn.ReLU(True),
            spectral_norm(nn.ConvTranspose2d(ngf * 1 * 2, 3, kernel_size=4, stride=2, padding=1), use),
            nn.Tanh(),
        )


    def forward(self, x, lbp, mask, second=False):
        dn11 = self.dn11(torch.cat([x, lbp, 1 - mask], 1))
        dn21 = self.dn21(dn11)
        dn31 = self.dn31(dn21)
        dn31 = self.trand256(dn31)
        dn41 = self.dn41(dn31)
        dn41 = self.trand512(dn41)
        dn51 = self.dn51(dn41)
        dn61 = self.dn61(dn51)
        dn71 = self.dn71(dn61)
        bottle1 = self.bottle1(dn71)
        up71 = self.up71(torch.cat([bottle1, dn71], 1))
        up61 = self.up61(torch.cat([up71, dn61], 1))
        up51 = self.up51(torch.cat([up61, dn51], 1))
        up51 = self.trand512(up51)
        up41 = self.up41(torch.cat([up51, dn41], 1))
        up41 = self.trand256(up41)
        up31 = self.up31(torch.cat([up41, dn31], 1))
        up21 = self.up21(torch.cat([up31, dn21], 1))
        output = self.up11(torch.cat([up21, dn11], 1)) + x
        output = output * mask + x * (1 - mask)
        return output


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False,
                 use_spectral_norm=True):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), use_spectral_norm),
            nn.LeakyReLU(0.2)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                        kernel_size=kw, stride=2, padding=padw, bias=use_bias), use_spectral_norm),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                    kernel_size=kw, stride=1, padding=padw, bias=use_bias), use_spectral_norm),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), use_spectral_norm)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = functools.partial(SwitchNorm2d)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, mode='min', patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, device=''):
    assert (torch.cuda.is_available())
    net.to(device)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(opt):
    netG = ImageGenerator(opt)
    return init_net(netG, 'normal', 0.02, device=opt.device)


def define_LBP(opt):
    netG = LBPGenerator()
    return init_net(netG, device=opt.device)


def define_D(input_nc, ndf, device=''):
    norm_layer = get_norm_layer(norm_type='instance')
    netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=True, use_spectral_norm=1)
    return init_net(netD, 'normal', 0.02, device)


class TransformerEncoder(nn.Module):
    def __init__(self, in_ch=256, head=4, expansion_factor=2.66):
        super().__init__()
        self.attn = mGAttn(in_ch=in_ch, num_head=head)
        self.feed_forward = FeedForward(dim=in_ch, expansion_factor=expansion_factor)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.feed_forward(x)
        return x

opt = MyOptions().parse()


class FeedForward(nn.Module):
    def __init__(self, dim=64, expansion_factor=2.66):
        super().__init__()
        # print(f"expansion_factor type: {type(expansion_factor)}, value: {expansion_factor}")
        # expansion_factor = opt.expansion_factor  # 这应该是一个数值

        num_ch = int(dim * opt.expansion_factor)
        self.norm = nn.InstanceNorm2d(num_features=dim, track_running_stats=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=num_ch * 2, kernel_size=1, bias=False),
            nn.Conv2d(in_channels=num_ch * 2, out_channels=num_ch * 2, kernel_size=3, stride=1, padding=1,
                      groups=num_ch * 2, bias=False)
        )
        self.linear = nn.Conv2d(in_channels=num_ch, out_channels=dim, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.norm(x)
        x1, x2 = self.conv(out).chunk(2, dim=1)
        out = F.gelu(x1) * x2
        out = self.linear(out)
        out = out + x
        return out

class mGAttn(nn.Module):
    def __init__(self, in_ch=256, num_head=4):
        super().__init__()
        self.head = num_head
        self.query = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.Softplus(),
        )

        self.key = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.Softplus(),
        )

        self.value = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU()
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, padding=0),
            nn.GELU()
        )
        self.output_linear = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
        self.norm = nn.InstanceNorm2d(num_features=in_ch)

    def forward(self, x):
        """
        x: b * c * h * w
        """
        x = self.norm(x)
        Ba, Ca, He, We = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        g = self.gate(x)
        num_per_head = Ca // self.head
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.head)  # B * head * c * N
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.head)  # B * head * c * N
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.head)  # B * head * c * N
        kv = torch.matmul(k, v.transpose(-2, -1))
        z = torch.einsum('bhcn,bhc -> bhn', q, k.sum(dim=-1)) / math.sqrt(num_per_head)
        z = 1.0 / (z + He * We)  # b h n
        out = torch.einsum('bhcn, bhcd-> bhdn', q, kv)
        out = out / math.sqrt(num_per_head)  # b h c n
        out = out + v
        out = out * z.unsqueeze(2)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=He)
        out = out * g      # gate
        out = self.output_linear(out)
        return out
