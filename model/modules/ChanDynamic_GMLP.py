import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from model.gshift_arch import Encoder_shift_block


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


# Layer Norm
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()

        assert LayerNorm_type in ['BiasFree', 'WithBias']
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out[:,:self.channel_in,:,:], out[:,self.channel_in:self.channel_in*4,:,:]
        else:
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups=self.channel_in)


class DynamicDWConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        Block1 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2, groups=channels)
                  for _ in range(3)]
        Block2 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2, groups=channels)
                  for _ in range(3)]
        self.tokernel = nn.Conv2d(channels, kernel_size ** 2 * self.channels, 1, 1, 0)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.Block1 = nn.Sequential(*Block1)
        self.Block2 = nn.Sequential(*Block2)

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.tokernel(self.pool(self.Block2(self.maxpool(self.Block1(self.avgpool(x))))))
        weight = weight.view(b * self.channels, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride,
                     padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])

        return x


# Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv3d(dim, hidden_features*2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias)
        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias)
        self.kerner_conv_channel = DynamicDWConv(hidden_features, 3, 1, hidden_features)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        b = x1.shape[0]
        x1 = rearrange(self.kerner_conv_channel(rearrange(x1, 'b c t h w -> (b t) c h w')), '(b t) c h w -> b c t h w', b=b)
        x = x1 * x2
        x = self.project_out(x)
        return x


class CWGDN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type='BiasFree'):
        super(CWGDN, self).__init__()
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        b = x.shape[0]
        identity = x
        x = rearrange(self.norm2(rearrange(x, 'b t c h w -> (b t) c h w')), '(b t) c h w -> b t c h w', b=b)
        x = rearrange(self.ffn(rearrange(x, 'b t c h w -> b c t h w')), 'b c t h w -> b t c h w')
        return x + identity


class CWGDN1(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type='BiasFree'):
        super(CWGDN1, self).__init__()
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.wave = HaarDownsampling(dim)
        self.x_wave_conv1 = nn.Conv2d(dim * 3, dim * 3, 1, 1, 0, groups=3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.x_wave_conv2 = nn.Conv2d(dim * 3, dim * 3, 1, 1, 0, groups=3)
        self.encoder_level1 = Encoder_shift_block(dim, 3, reduction=4, bias=False)
        # self.encoder_level1_1 = Encoder_shift_block(dim, 3, reduction=4, bias=False)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        b = x.shape[0]
        identity = x
        x = self.norm2(rearrange(x, 'b t c h w -> (b t) c h w'))
        tf_wave_l, tf_wave_h = self.wave(x)
        tf_wave_h = self.x_wave_conv2(self.lrelu(self.x_wave_conv1(tf_wave_h)))
        tf_wave_l = self.encoder_level1(tf_wave_l)
        tf_wave_l = rearrange(self.ffn(rearrange(tf_wave_l, '(b t) c h w -> b c t h w', b=b)),
                              'b c t h w -> (b t) c h w')
        x = rearrange(self.wave(torch.cat([tf_wave_l, tf_wave_h], dim=1), rev=True),
                      '(b t) c h w -> b t c h w', b=b)  # 上采样

        return x + identity