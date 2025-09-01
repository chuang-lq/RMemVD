import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from .modules import *
from .memory_bank import MemoryBank


class Model(nn.Module):
    def __init__(self, para):
        super(Model, self).__init__()

        self.mid_channels = para.mid_channels
        self.mem_every = para.mem_every  # mem_every: 8
        # self.deep_update_every = para.deep_update_every
        self.num_blocks_forward = para.num_blocks_forward
        self.num_blocks_backward = para.num_blocks_backward

        # self.mid_channels = 64
        # self.mem_every = 8
        # self.deep_update_every = 5
        # self.num_blocks_forward = 30
        # self.num_blocks_backward = 15

        # ----------------- Deblurring branch -----------------
        # Downsample Module
        self.n_feats = 16

        self.downsampling = nn.Sequential(
            conv5x5(3, 3, stride=1),
            RDB_DS(in_channels=3, growth_rate=16, num_layer=3, bias=False),
            RDB_DS(in_channels=3 * 4, growth_rate=16, num_layer=3, bias=False)  # b, 48, h/4, w/4
        )

        self.feat_downsample = nn.Sequential(
            conv5x5(3, 4, stride=1),
            RDB_DS(in_channels=4, growth_rate=16, num_layer=3, bias=False),
            RDB_DS(in_channels=4 * 4, growth_rate=int(self.n_feats * 4 / 2), num_layer=3, bias=False)  # b, 64, h/4, w/4
        )

        # Upsample Module
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(self.mid_channels, 2 * self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.ConvTranspose2d(2 * self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            conv5x5(self.n_feats, 3, stride=1)
        )

        # Feature fusion Module
        self.forward_fusion_conv = nn.Sequential(
            nn.Conv2d(self.mid_channels + 48 * 2 + 64 * 2, self.mid_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.forward_resblocks = ResidualBlocksWithoutInputConv(
            self.mid_channels, self.num_blocks_forward)  # 30 for memory network

        self.backward_resblocks = ResidualBlocksWithInputConv(
            self.mid_channels * 2 + 48 * 2, self.mid_channels, self.num_blocks_backward)

        self.fusion = nn.Conv2d(
            self.mid_channels * 2, self.mid_channels, 1, 1, 0)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 5, 1, 2)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # ----------------- Memory branch -----------------
        self.memory = Memory(para, reinforce=False)
        self.memory.mem_bank_forward.training = self.training
        self.memory.mem_bank_backward.training = self.training

    def forward(self, inputs, profile_flag=False):
        if profile_flag:
            return self.profile_forward(inputs)

        self.memory.mem_bank_backward.clear_memory()
        # self.memory.mem_bank_forward.clear_memory()
        self.memory.mem_bank_forward.init_memory()

        n, t, c, h_ori, w_ori = inputs.size()

        outputs = []

        # encode_key
        downsampled_features = []
        fs = []
        keys = []
        s = []
        e = []
        for i in range(0, t):
            downsampled_features.append(self.downsampling(inputs[:, i]))  # b, 48, h/4, w/4
            # f: b, 256, h/16, w/16, key: b, 64, h/16, w/16
            f, key, shrinkage, selection = self.memory.encode_key(downsampled_features[-1])
            keys.append(key)
            s.append(shrinkage)
            e.append(selection)
            fs.append(f)

        h = h_ori // 4
        w = w_ori // 4

        # ------------ backward ------------
        feat_backs = []
        prev_featback = None
        for i in range(t - 1, -1, -1):
            downsampled_feature = downsampled_features[i]
            f = fs[i]
            key_curr = keys[i]
            shrinkage = s[i]
            selection = e[i]

            if i == t - 1:
                if self.memory.mem_bank_backward.get_hidden() is None:
                    self.memory.mem_bank_backward.create_hidden_state(key_curr)
                hidden = self.memory.mem_bank_backward.get_hidden()
                hidden, feat = self.memory.decode(f, f, hidden)
                feat = torch.cat([downsampled_feature,
                                  downsampled_feature.new_zeros(n, self.mid_channels+downsampled_feature.shape[1], h, w), feat], dim=1)
            else:
                memory_readout = self.memory.mem_bank_backward.match_memory(key_curr, selection)
                hidden = self.memory.mem_bank_backward.get_hidden()
                hidden, feat = self.memory.decode(f, memory_readout, hidden)
                feat = torch.cat([downsampled_feature, downsampled_features[i+1], feat, prev_featback], dim=1)

            self.memory.mem_bank_backward.set_hidden(hidden)
            feat = self.backward_resblocks(feat)

            # mem_every
            hidden = self.memory.mem_bank_backward.get_hidden()
            value, _ = self.memory.encode_value(downsampled_feature, f, feat, hidden)
            self.memory.mem_bank_backward.add_memory(key_curr, value=value, shrinkage=shrinkage)

            feat_backs.append(feat)
            prev_featback = feat

        feat_backs = list(reversed(feat_backs))

        # ----------- forward --------------
        hidden_future = self.memory.mem_bank_backward.get_hidden()
        prev_featpre = None
        for i in range(0, t):
            input_curr = inputs[:, i]
            down_feature = downsampled_features[i]
            f = fs[i]
            key_curr = keys[i]
            shrinkage = s[i]
            selection = e[i]
            feat_back = feat_backs[i]

            # Memory from backward
            memory_readout = self.memory.mem_bank_backward.match_memory(key_curr, selection)
            _, feat_future = self.memory.decode(f, memory_readout, hidden_future, h_out=False)

            if i == 0:
                if self.memory.mem_bank_forward.get_hidden() is None:
                    self.memory.mem_bank_forward.create_hidden_state(key_curr)
                hidden = self.memory.mem_bank_forward.get_hidden()
                hidden, feat = self.memory.decode(f, memory_readout, hidden)
                feat = torch.cat([down_feature, feat_future,
                                  down_feature.new_zeros(n, self.mid_channels + down_feature.shape[1], h, w), feat], dim=1)
            else:
                hidden = self.memory.mem_bank_forward.get_hidden()
                memory_readout = self.memory.mem_bank_forward.match_memory(key_curr, selection)
                hidden, feat = self.memory.decode(f, memory_readout, hidden)
                feat = torch.cat([down_feature, feat_future, downsampled_features[i - 1], feat, prev_featpre], dim=1)

            self.memory.mem_bank_forward.set_hidden(hidden)
            feat = self.forward_fusion_conv(feat)
            feat = self.forward_resblocks(feat)

            out = torch.cat([feat_back, feat], dim=1)
            out = self.lrelu(self.fusion(out))
            out = self.upsampling(out)
            out = out + input_curr
            outputs.append(out)

            # # mem_every
            hidden = self.memory.mem_bank_forward.get_hidden()
            value, _ = self.memory.encode_value(down_feature, f, self.feat_downsample(out), hidden)
            self.memory.mem_bank_forward.add_memory(key_curr, value=value, shrinkage=shrinkage)

            prev_featpre = feat

        results = torch.stack(outputs, dim=1)
        return results

    def profile_forward(self, inputs):
        return self.forward(inputs)


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)


class ResidualBlocksWithoutInputConv(nn.Module):
    def __init__(self, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)


class KeyEncoder(nn.Module):
    def __init__(self, in_dim=48):
        super().__init__()
        self.forward_resblocks = ResidualBlocksWithInputConv(in_dim, 48, 2)

        resnet = resnet50(pretrained=True, extra_chan=48-3)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # b, 256, 1/4

    def forward(self, f):
        x = self.forward_resblocks(f)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        return x


class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim=64):
        super().__init__()

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        # shrinkage
        self.s_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        # selection
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x, need_s, need_e):
        shrinkage = self.s_proj(x) ** 2 + 1 if (need_s) else None  # s: [1,+00)
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None  # e: [0,1]

        return self.key_proj(x), shrinkage, selection


class HiddenUpdater(nn.Module):

    def __init__(self, in_dim, hidden_dim, kernel_size, bias):
        """
        Initialize the ConvLSTM cell
        """
        super(HiddenUpdater, self).__init__()
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.padding = "same"
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.conv_gates = nn.Conv2d(
            in_channels=in_dim + hidden_dim,
            out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        self.conv_can = nn.Conv2d(
            in_channels=in_dim + hidden_dim,
            out_channels=self.hidden_dim,  # for candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_x, h_cur):
        """
        :param input_x: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_x, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_x, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = h_cur * (1 - update_gate) + update_gate * cnm
        return h_next


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        if in_dim == out_dim:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, g):
        out_g = self.conv2(self.lrelu(self.conv1(g)))

        if self.downsample is not None:
            g = self.downsample(g)

        return out_g + g


class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.block1 = ResBlock(x_in_dim + g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = ResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):

        g = torch.cat([x, g], dim=1)
        g = self.block1(g)
        r = self.attention(g)
        g = self.block2(g + r)

        return g


class ValueEncoder(nn.Module):
    def __init__(self, in_dim=112, hidden_dim=64):
        super().__init__()
        self.forward_resblocks = ResidualBlocksWithInputConv(in_dim, 32, 2)

        resnet = resnet50(pretrained=True, extra_chan=32-3)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4

        self.fuser = FeatureFusionBlock(256, 256, 256, 256)

        if hidden_dim > 0:
            self.hidden_reinforce = HiddenUpdater(256, hidden_dim, kernel_size=3, bias=True)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat, gt, hidden, is_deep_update=False):
        f = torch.cat([image, gt], 1)

        x = self.forward_resblocks(f)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)  # b, 256, 1/4

        x = self.fuser(image_feat, x)

        if is_deep_update and self.hidden_reinforce is not None:
            hidden = self.hidden_reinforce(x, hidden)

        return x, hidden


class MemoryReinforcingModule(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, size):
        super().__init__()

        self.conv_wxh = nn.Conv2d(in_channels+mid_channels, mid_channels, size, padding=(size // 2))  # Convw×h
        self.conv_1x1 = nn.Conv2d(mid_channels, out_channels, 1)  # Conv1×1

    def forward(self, kQ, prev_feat):

        # Concatenate mt⊔1 with k′Q
        prev_frame_mask = torch.nn.functional.interpolate(prev_feat, kQ.shape[-2:])
        concatenated_features = torch.cat((prev_frame_mask, kQ), dim=1)

        # Apply Convw×h to generate local attention feature Fatt
        local_attention_feature = self.conv_wxh(concatenated_features)

        # Apply Conv1×1 to transform the dimensions of Fatt
        local_attention_feature = self.conv_1x1(local_attention_feature)

        # Apply softmax to normalize local attention weights
        alpha = F.softmax(local_attention_feature, dim=1)

        # Incorporate previous feat
        prev_frame_mask = self.conv_1x1(prev_frame_mask)
        kQ = alpha * prev_frame_mask

        return kQ


class PixelShuffleBlock(torch.nn.Module):
    def __init__(self, channels, bias):
        super(PixelShuffleBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1, stride=1, bias=bias)
        self.conv2 = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1, stride=1, bias=bias)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1, bias=bias)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.shuffle = torch.nn.PixelShuffle(2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.shuffle(x)
        x = self.relu(self.conv2(x))
        x = self.shuffle(x)
        x = self.relu(self.conv3(x))

        return x


class Decoder(nn.Module):  # Decoder
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()

        self.hidden_dim = hidden_dim

        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater(out_dim, hidden_dim, kernel_size=3, bias=True)
            self.fusion = nn.Conv2d(in_dim + out_dim, out_dim, 3, padding=1)
        else:
            self.hidden_update = None

        self.forward_resblocks = ResidualBlocksWithInputConv(in_dim * 2 + hidden_dim, out_dim, 2)

        self.upsample = PixelShuffleBlock(out_dim, False)

    def forward(self, f, memory_readout, hidden_state, h_out=True):
        if self.hidden_update is not None:
            x = torch.cat([f, memory_readout, hidden_state], dim=1)
        else:
            x = torch.cat([f, memory_readout], dim=1)
        x = self.forward_resblocks(x)

        if h_out and self.hidden_update is not None:
            f = self.fusion(torch.cat([f, x], dim=1))
            hidden_state = self.hidden_update(f, hidden_state)

        return hidden_state, self.upsample(x)


class Memory(nn.Module):
    def __init__(self, para, reinforce=False):
        super().__init__()

        self.key_encoder = KeyEncoder(48)  # b, 256, 1/4
        self.key_proj = KeyProjection(256, keydim=64)

        self.value_encoder = ValueEncoder(112, hidden_dim=64)

        self.decoder = Decoder(in_dim=256, out_dim=64, hidden_dim=64)

        self.reinforce = reinforce
        if self.reinforce:
            self.memory_reinforce = MemoryReinforcingModule(256, 64, 256, 3)

        self.mem_bank_forward = MemoryBank(count_usage=True)
        self.mem_bank_backward = MemoryBank(count_usage=True)

    def encode_value(self, image, image_feat, gt, hidden_state, is_deep_update=False):
        if self.reinforce:
            image_feat = self.memory_reinforce(image_feat, gt)
        f, h = self.value_encoder(image, image_feat, gt, hidden_state, is_deep_update=is_deep_update)

        return f, h

    def encode_key(self, frame, need_sk=True, need_ek=True):
        f = self.key_encoder(frame)
        k, shrinkage, selection = self.key_proj(f, need_s=need_sk, need_e=need_ek)
        return f, k, shrinkage, selection

    def decode(self, f, memory_readout, hidden_state, h_out=True):
        hidden_state, f = self.decoder(f, memory_readout, hidden_state, h_out=h_out)
        return hidden_state, f


def cost_profile(model, H, W, seq_length=100):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops / seq_length, params


def feed(model, iter_samples):
    inputs = iter_samples[0]
    outputs = model(inputs)
    return outputs


if __name__ == '__main__':
    x = torch.randn(1, 48, 256, 256)
    y = torch.randn(1, 64, 256, 256)
    # x0 = torch.randn(1, 256, 64, 64)
    # h = torch.randn(1, 64, 64, 64)
    # hidden_update = HiddenUpdater(256, 64, kernel_size=3, bias=True)
    # h = hidden_update(x0, h)
    # print(h.shape)
    # key_encoder = KeyEncoder()
    # value_encoder = ValueEncoder()
    # x1 = key_encoder(x)
    # x2, h = value_encoder(x, y, h, is_deep_update=True)
    # print(x1.shape)
    # print(x2.shape)
    # print(h.shape)
