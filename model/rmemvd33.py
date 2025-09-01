import torch
import torch.nn as nn
from thop import profile
from .modules import *
from .memory_bank3 import MemoryBank


class Model(nn.Module):
    def __init__(self, para):
        super().__init__()

        self.mid_channels = para.mid_channels
        self.mem_every = para.mem_every
        # self.deep_update_every = para.deep_update_every
        self.num_blocks_forward = para.num_blocks_forward
        self.num_blocks_backward = para.num_blocks_backward

        # ----------------- Deblurring branch -----------------
        self.n_feats = 16

        self.downsampling = nn.Sequential(
            nn.Conv3d(3, self.n_feats, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            RRDB(self.n_feats, self.mid_channels, num_RDB=4, growth_rate=16, num_dense_layer=3, bias=False)
        )

        self.feat_downsample = nn.Sequential(
            conv3x3(3, self.n_feats, stride=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            RDBs(self.n_feats, self.mid_channels, num_RDB=4, growth_rate=16, num_layer=3, bias=False)  # b, 64, h/2, w/2
        )

        # SA transformer
        # transformer_scale4 = []
        # for _ in range(2):
        #     transformer_scale4.append(
        #         nn.Sequential(
        #             CWGDN(dim=self.mid_channels, ffn_expansion_factor=4, bias=False, LayerNorm_type='WithBias'))
        #     )
        # self.transformer_scale4 = nn.Sequential(*transformer_scale4)

        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(self.mid_channels, self.n_feats, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            conv5x5(self.n_feats, 3, stride=1)
        )

        self.conv_trans = nn.Conv2d(self.mid_channels * 2, 256, 3, 1, 1)

        # Feature fusion Module
        self.feat_fusion = DTFF(self.mid_channels)

        self.object_transformer = QueryTransformer(
            dim=256, num_heads=8, num_blocks=3, LayerNorm_type='WithBias', ffn_expansion_factor=2, bias=False)

        self.forward_backbone = ResidualBlocksWithInputConv(
            self.mid_channels * 3, self.mid_channels, self.num_blocks_forward)
        self.backward_backbone = ResidualBlocksWithInputConv(
            self.mid_channels * 2, self.mid_channels, self.num_blocks_backward)

        self.tfr = TFR(n_feat=self.mid_channels, kernel_size=3, reduction=4, act=nn.PReLU(),
                       bias=False, scale_unetfeats=int(self.mid_channels / 2), num_cab=5)
        # self.tfr = TFR_UNet(n_feat=self.mid_channels, kernel_size=3, reduction=4, act=nn.PReLU(),
        #                     bias=False, scale_unetfeats=int(self.mid_channels / 2))

        # ----------------- Memory branch -----------------
        self.memory = Memory(para)

    def forward(self, inputs, profile_flag=False):
        if profile_flag:
            return self.profile_forward(inputs)

        self.memory.mem_bank_backward.clear_memory()
        self.memory.mem_bank_forward.init_memory()

        n, t, c, h_ori, w_ori = inputs.size()

        outputs = []

        # feature extraction
        down_feats = self.downsampling(rearrange(inputs, 'b t c h w -> b c t h w')).permute(0, 2, 1, 3, 4).contiguous()
        # down_feats = self.transformer_scale4(down_feats)  # b, t, c, h, w

        prev_feat = None
        encoder_outs = None
        decoder_outs = None
        encoder_outs_list = []
        decoder_outs_list = []

        # ------------ backward ------------
        for i in range(t - 1, -1, -1):
            down_feat = down_feats[:, i]
            key, shrinkage, selection, encoder_outs, semantic_f = self.memory.encode_key(down_feat, encoder_outs, decoder_outs)
            f, f_1, f_2 = encoder_outs

            if i == t - 1:
                if self.memory.mem_bank_backward.get_hidden() is None:
                    self.memory.mem_bank_backward.create_hidden_state(key[0])
                pixel_readout = self.conv_trans(f_2)
                object_readout, pixel_readout = self.object_transformer(semantic_f, pixel_readout)
                hidden = self.memory.mem_bank_backward.get_hidden()
                hidden, decoder_outs = self.memory.decoder(f_2, f_1, f, semantic_f, pixel_readout, object_readout, hidden)
                feat = torch.cat([down_feat, decoder_outs[0]], dim=1)
            else:
                pixel_readout, object_readout = self.memory.mem_bank_backward.match_memory(key[0], selection[0], key[1], selection[1])
                object_readout, pixel_readout = self.object_transformer(object_readout, pixel_readout)
                hidden = self.memory.mem_bank_backward.get_hidden()
                hidden, decoder_outs = self.memory.decoder(f_2, f_1, f, semantic_f, pixel_readout, object_readout, hidden)
                # down_feat = self.feat_fusion(down_feat, down_feats[:, i+1])
                feat = self.feat_fusion(decoder_outs[0], prev_feat)
                feat = torch.cat([self.feat_fusion(down_feat, down_feats[:, i+1]), feat], dim=1)

            self.memory.mem_bank_backward.set_hidden(hidden)
            feat = self.backward_backbone(feat)

            decoder_outs[0] = feat
            encoder_outs_list.append(encoder_outs)
            decoder_outs_list.append(decoder_outs)

            # mem_every
            hidden = self.memory.mem_bank_backward.get_hidden()
            pixel, object_feat, _ = self.memory.encode_value(down_feat, f_2, semantic_f, feat, hidden)
            self.memory.mem_bank_backward.add_memory(key, values=[pixel, object_feat], shrinkage=shrinkage)

            prev_feat = feat

        encoder_outs_list = list(reversed(encoder_outs_list))
        decoder_outs_list = list(reversed(decoder_outs_list))

        # ----------- forward --------------
        encoder_forward_list = []
        decoder_forward_list = []
        hidden_future = self.memory.mem_bank_backward.get_hidden()
        for i in range(0, t):
            inputs_curr = inputs[:, i]
            down_feat = down_feats[:, i]
            key, shrinkage, selection, encoder_outs, semantic_f = self.memory.encode_key(down_feat, encoder_outs, decoder_outs)
            f, f_1, f_2 = encoder_outs

            # Memory from backward
            pixel_readout, object_readout = self.memory.mem_bank_backward.match_memory(key[0], selection[0], key[1], selection[1])
            object_readout, pixel_readout = self.object_transformer(object_readout, pixel_readout)
            # print('pixel_readout: ', torch.max(pixel_readout))
            # print('object_readout: ', torch.max(object_readout))
            _, feat_future = self.memory.decoder(f_2, f_1, f, semantic_f, pixel_readout, object_readout, hidden_future,
                                                 h_out=False)

            if i == 0:
                if self.memory.mem_bank_forward.get_hidden() is None:
                    self.memory.mem_bank_forward.create_hidden_state(key[0])
                pixel_readout = self.conv_trans(f_2)
                object_readout, pixel_readout = self.object_transformer(semantic_f, pixel_readout)
                hidden = self.memory.mem_bank_forward.get_hidden()
                hidden, decoder_outs = self.memory.decoder(f_2, f_1, f, semantic_f, pixel_readout, object_readout, hidden)
                feat = self.feat_fusion(decoder_outs[0], prev_feat)
                feat = torch.cat([down_feat, feat_future[0], feat], dim=1)
            else:
                pixel_readout, object_readout = self.memory.mem_bank_forward.match_memory(key[0], selection[0], key[1], selection[1])
                object_readout, pixel_readout = self.object_transformer(object_readout, pixel_readout)
                hidden = self.memory.mem_bank_forward.get_hidden()
                hidden, decoder_outs = self.memory.decoder(f_2, f_1, f, semantic_f, pixel_readout, object_readout, hidden)
                # down_feat = self.feat_fusion(down_feat, down_feats[:, i-1])
                feat = self.feat_fusion(decoder_outs[0], prev_feat)
                feat = torch.cat([self.feat_fusion(down_feat, down_feats[:, i-1]), feat_future[0], feat], dim=1)

            self.memory.mem_bank_forward.set_hidden(hidden)
            feat = self.forward_backbone(feat)

            decoder_outs[0] = feat
            # decoder_outs[1] = feat_future[1] + decoder_outs[1]
            # decoder_outs[2] = feat_future[2] + decoder_outs[2]
            encoder_forward_list.append(encoder_outs)
            decoder_forward_list.append(decoder_outs)

            for j in range(3):
                encoder_outs_list[i][j] = encoder_outs_list[i][j] + encoder_forward_list[i][j]
                decoder_outs_list[i][j] = decoder_outs_list[i][j] + decoder_forward_list[i][j]

            out = self.tfr(down_feat, encoder_outs_list[i], decoder_outs_list[i])
            out = self.upsampling(out)
            out += inputs_curr
            outputs.append(out)

            # mem_every
            hidden = self.memory.mem_bank_forward.get_hidden()
            pixel, object_feat, _ = self.memory.encode_value(down_feat, f_2, semantic_f, self.feat_downsample(out), hidden)
            self.memory.mem_bank_forward.add_memory(key, values=[pixel, object_feat], shrinkage=shrinkage)

            prev_feat = feat

        results = torch.stack(outputs, dim=1)

        return results

    def profile_forward(self, inputs):
        return self.forward(inputs)


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, num_blocks=15):
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


class ResnetBlock(nn.Module):
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


class KeyEncoder(nn.Module):
    def __init__(self, in_dim=64):
        super().__init__()
        self.key_encoder = Encoder(n_feat=in_dim, kernel_size=3, reduction=4, bias=False)

        # self.transformer = RQBs(img_size=(256, 256), dim=in_dim, depth=2, window_size=8, rpe='v1', coords_lambda=5e-1)
        self.transformer = SCA(dim=in_dim, num_heads=8, num_blocks=3, LayerNorm_type='WithBias',
                               ffn_expansion_factor=2, bias=False)

    def forward(self, x, encoder_outs=None, decoder_outs=None):

        # semantic_x = self.transformer(x)
        x = self.key_encoder(x, encoder_outs=encoder_outs, decoder_outs=decoder_outs)
        semantic_x = self.transformer(x[0])

        return x, semantic_x


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

    def forward(self, x, need_s=True, need_e=True):
        shrinkage = self.s_proj(x) ** 2 + 1 if (need_s) else None  # s: [1,+00)
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None  # e: [0,1]

        return self.key_proj(x), shrinkage, selection


class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.block1 = ResnetBlock(x_in_dim + g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = ResnetBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):

        g = torch.cat([x, g], dim=1)
        g = self.block1(g)
        r = self.attention(g)
        g = self.block2(g + r)

        return g


class HiddenReinforcer(nn.Module):

    def __init__(self, in_dim, hidden_dim, kernel_size):
        """
        Initialize the ConvLSTM cell
        """
        super(HiddenReinforcer, self).__init__()
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.padding = "same"
        self.hidden_dim = hidden_dim

        self.conv_gates = nn.Conv2d(
            in_channels=in_dim + hidden_dim,
            out_channels=2*self.hidden_dim,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding=self.padding
        )

        self.conv_can = nn.Conv2d(
            in_channels=in_dim + hidden_dim,
            out_channels=self.hidden_dim,  # for candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding
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


class HiddenUpdater(nn.Module):

    def __init__(self, in_dim, hidden_dim, kernel_size):
        """
        Initialize the ConvLSTM cell
        """
        super(HiddenUpdater, self).__init__()
        # self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.padding = "same"
        self.hidden_dim = hidden_dim

        self.f_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim + int(in_dim / 2), kernel_size=5, stride=2, padding=2),
            nn.PReLU(),
            nn.Conv2d(in_dim + int(in_dim / 2), in_dim + int(in_dim / 2), kernel_size=3, stride=2, padding=1))
        self.f1_conv = nn.Conv2d(in_dim + int(in_dim / 2), in_dim + int(in_dim / 2), kernel_size=3, stride=2, padding=1)
        self.f2_conv = nn.Conv2d(in_dim * 2, in_dim + int(in_dim / 2), kernel_size=1)

        self.conv_gates = nn.Conv2d(
            in_channels=in_dim + int(in_dim / 2) + hidden_dim,
            out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size,
            padding=self.padding
        )

        self.conv_can = nn.Conv2d(
            in_channels=in_dim + int(in_dim / 2) + hidden_dim,
            out_channels=self.hidden_dim,  # for candidate neural memory
            kernel_size=kernel_size,
            padding=self.padding
        )

    def forward(self, x, h_cur):
        x = self.f_conv(x[0]) + self.f1_conv(x[1]) + self.f2_conv(x[2])

        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([x, reset_gate * h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = h_cur * (1 - update_gate) + update_gate * cnm
        return h_next


class ValueEncoder(nn.Module):
    def __init__(self, in_dim=64, hidden_dim=96):
        super().__init__()

        self.forward_resblocks = ResidualBlocksWithInputConv(in_dim*2, 64, 2)
        resnet = resnet50(pretrained=True, extra_chan=64-3)

        # self.transformer = RQBs(img_size=(256, 256), dim=in_dim, depth=2, window_size=8, rpe='v1', coords_lambda=5e-1)
        self.transformer = SCA(dim=in_dim, num_heads=8, num_blocks=3, LayerNorm_type='WithBias',
                               ffn_expansion_factor=2, bias=False)

        self.body = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,  # 1/2
            resnet.maxpool,
            resnet.layer1,  # 1/4
        )

        self.conv_trans = nn.Conv2d(256, in_dim * 2, kernel_size=3, padding=1)
        self.fuser = FeatureFusionBlock(in_dim * 2, 256, 128, 256)

        # if hidden_dim > 0:
        #     self.fusion = nn.Conv2d(256 * 2, 256, kernel_size=3, padding=1)
        #     self.hidden_reinforce = HiddenReinforcer(256, hidden_dim, kernel_size=3)
        # else:
        #     self.hidden_reinforce = None

    def forward(self, image, image_feat, semantic_feat, gt, hidden_state, is_deep_update=False):

        x = self.forward_resblocks(torch.cat([image, gt], 1))

        f = self.body(x)
        f = self.fuser(image_feat, f)

        semantic_f = self.transformer(x)
        semantic_f = self.fuser(self.conv_trans(semantic_feat), semantic_f)

        # if is_deep_update and self.hidden_reinforce is not None:
        #     x = self.fusion(torch.cat([f, semantic_f], 1))
        #     hidden_state = self.hidden_reinforce(x, hidden_state)

        return f, semantic_f, hidden_state


class MemoryReinforcingModule(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, size):
        super().__init__()

        self.conv_wxh = nn.Conv2d(in_channels+mid_channels, mid_channels, size, padding=(size // 2))  # Convw×h
        self.conv_1x1 = nn.Conv2d(mid_channels, out_channels, 1)  # Conv1×1

    def forward(self, kQ, prev_feat):
        kQ_shape = kQ.shape
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


class MemoryDecoder(nn.Module):  # Decoder
    def __init__(self, n_feat=64, hidden_dim=96):
        super().__init__()
        self.hidden_dim = hidden_dim

        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater(n_feat, hidden_dim, kernel_size=3)
        else:
            self.hidden_update = None

        self.conv_fusion = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv_trans = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.cafm = CAFM(128)
        self.fuser = FeatureFusionBlock(128, 256 + hidden_dim, 128, 128)

        self.decoder = Decoder(n_feat, kernel_size=3, reduction=4, bias=False)

    def forward(self, f_2, f_1, f, semantic_f, pixel_readout, object_readout, hidden_state, h_out=True):
        memory_readout = self.conv_fusion(torch.cat([pixel_readout, object_readout], dim=1))

        if self.hidden_update is not None:
            memory_readout = torch.cat([memory_readout, hidden_state], dim=1)

        f_2, semantic_f = self.cafm(f_2, self.conv_trans(semantic_f))
        f_2 = self.conv_trans(torch.cat([f_2, semantic_f], dim=1))
        f_2 = self.fuser(f_2, memory_readout)
        decoder_outs = self.decoder([f, f_1, f_2])

        if h_out and self.hidden_update is not None:
            hidden_state = self.hidden_update(decoder_outs, hidden_state)

        return hidden_state, decoder_outs


class Memory(nn.Module):
    def __init__(self, para):
        super().__init__()

        self.key_encoder = KeyEncoder(64)
        self.key_proj1 = KeyProjection(128, keydim=64)
        self.key_proj2 = KeyProjection(256, keydim=64)

        self.value_encoder = ValueEncoder(64, hidden_dim=96)

        self.decoder = MemoryDecoder(n_feat=64, hidden_dim=96)

        self.mem_bank_forward = MemoryBank(para=para, count_usage=True)
        self.mem_bank_backward = MemoryBank(para=para, count_usage=True)

    def encode_value(self, image, image_feat, semantic_feat, gt, hidden_state, is_deep_update=False):
        f, semantic_f, h = self.value_encoder(image, image_feat, semantic_feat, gt, hidden_state, is_deep_update=is_deep_update)

        return f, semantic_f, h

    def encode_key(self, frame, encoder_outs=None, decoder_outs=None, need_sk=True, need_ek=True):
        encoder_outs, semantic_f = self.key_encoder(frame, encoder_outs=encoder_outs, decoder_outs=decoder_outs)
        k1, shrinkage1, selection1 = self.key_proj1(encoder_outs[2], need_s=need_sk, need_e=need_ek)
        k2, shrinkage2, selection2 = self.key_proj2(semantic_f, need_s=need_sk, need_e=need_ek)
        return [k1, k2], [shrinkage1, shrinkage2], [selection1, selection2], encoder_outs, semantic_f


def cost_profile(model, H, W, seq_length=5):
    x = torch.randn(1, seq_length, 3, H, W).cuda()
    profile_flag = True
    flops, params = profile(model, inputs=(x, profile_flag), verbose=False)

    return flops / seq_length, params


def feed(model, inputs):
    # inputs = iter_samples[0]
    outputs = model(inputs)
    return outputs


if __name__ == '__main__':
    x = torch.randn(2, 8, 3, 160, 90).cuda()
    # x = x.rot90(1, [3, 4]).flip(1)
    model = Model().cuda()
    x = model(x)
    print(x.shape)
