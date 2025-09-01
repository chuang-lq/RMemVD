from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


def softmax_w_top(x, top):
    top = min(top, x.shape[1])
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = values.exp_()
    x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
    # x.zero_().scatter_(1, indices, x_exp)
    result = torch.zeros_like(x).scatter(1, indices, x_exp)

    return result


class Attention(nn.Module):
    # if tlc = True, patch_size = 32
    def __init__(self, input_dim=256, mid_dim=256, num_heads=8, patch_size=32, attention_dropout=0, bias=False):
        super(Attention, self).__init__()

        self.mid_dim = mid_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        head_dim = mid_dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(input_dim, mid_dim, bias=bias)
        self.k_proj = nn.Linear(input_dim, mid_dim, bias=bias)
        self.v_proj = nn.Linear(input_dim, mid_dim, bias=bias)

        self.norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout = nn.Dropout(attention_dropout)
        self.out_proj = nn.Linear(mid_dim, input_dim)

    def grids(self, x):
        b, t, h, w, c = x.shape
        self.original_size = x.shape

        k1 = k2 = self.patch_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2, :])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.stack(parts, dim=1)  # b, num_patches, t, k1, k2, c
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        b, t, h, w, c = self.original_size
        preds = torch.zeros(self.original_size).to(outs.device)
        count_mt = torch.zeros((b, 1, h, w, 1)).to(outs.device)

        k1 = k2 = self.patch_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[:, :, i:i + k1, j:j + k2, :] += outs[:, cnt]
            count_mt[:, 0, i:i + k1, j:j + k2, 0] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def local_partition(self, x, h_step, w_step, dh, dw):
        b, t, h, w, c = x.size()
        local_x = []
        for i in range(0, h + h_step - dh, h_step):
            top = i
            down = i + dh
            if down > h:
                top = h - dh
                down = h
            for j in range(0, w + w_step - dw, w_step):
                left = j
                right = j + dw
                if right > w:
                    left = w - dw
                    right = w
                local_x.append(x[:, :, top:down, left:right, :])
        local_x = torch.stack(local_x, dim=1)  # b n t dh dw c
        return local_x

    def local_reverse(self, local_x, x, h_step, w_step, dh, dw):
        b, t, h, w, c = x.size()
        x_output = torch.zeros_like(x)
        count = torch.zeros((b, h, w), device=x.device)

        index = 0
        for i in range(0, h + h_step - dh, h_step):
            top = i
            down = i + dh
            if down > h:
                top = h - dh
                down = h
            for j in range(0, w + w_step - dw, w_step):
                left = j
                right = j + dw
                if right > w:
                    left = w - dw
                    right = w
                x_output[:, :, top:down, left:right, :] += local_x[:, index]  # local_x: b n t dh dw c
                count[:, top:down, left:right] += 1
                index += 1
        x_output = x_output / count.unsqueeze(1).unsqueeze(-1)
        return x_output

    def forward(self, x, y=None):  # x: B, HW, CV / B, T, H, W, CV
        # assert len(x.shape) == 3 or len(x.shape) == 5
        b = x.shape[0]
        x = self.norm(x)
        q = self.q_proj(x)
        if y is None:
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            k = self.k_proj(y)
            v = self.v_proj(y)

        if len(x.shape) == 3:
            q, k, v = map(lambda x: rearrange(x, 'b n (head c) -> b head n c', head=self.num_heads), (q, k, v))
        else:
            dh = dw = self.patch_size
            # local_q = self.grids(q)
            # local_k = self.grids(k)
            # local_v = self.grids(v)  # b n t dh dw c
            local_q = self.local_partition(q, dh - dh // 4, dw - dh // 4, dh, dw)
            local_k = self.local_partition(k, dh - dh // 4, dw - dh // 4, dh, dw)
            local_v = self.local_partition(v, dh - dh // 4, dw - dh // 4, dh, dw)  # b n t dh dw c

            q, k, v = map(lambda x: rearrange(x, 'b n t dh dw (head c) -> (b n) head (t dh dw) c',
                                              head=self.num_heads), (local_q, local_k, local_v))

        # match: B, HW, CV @ B, CV, THW
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = scores.softmax(dim=-1)
        attn = self.dropout(attn)

        # B, HW, THW @ B, THW, CV
        output = torch.matmul(attn, v)
        if len(x.shape) == 3:
            output = rearrange(output, 'b head n c -> b n (head c)')
        else:
            output = rearrange(output, '(b n) head (t dh dw) c -> b n t dh dw (head c)', b=b, dh=dh, dw=dw)
            # output = self.grids_inverse(output)  # b t h w c
            output = self.local_reverse(output, x, dh - dh // 4, dw - dh // 4, dh, dw)  # b t h w c
        output = self.out_proj(output)

        return output


class Attention1(nn.Module):
    # semantic memory_bank: enforcing local attention, patch size = 8
    def __init__(self, input_dim=256, mid_dim=256, num_heads=8, patch_size=8, attention_dropout=0, bias=False):
        super(Attention1, self).__init__()

        self.mid_dim = mid_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        head_dim = mid_dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(input_dim, mid_dim, bias=bias)
        self.k_proj = nn.Linear(input_dim, mid_dim, bias=bias)
        self.v_proj = nn.Linear(input_dim, mid_dim, bias=bias)

        self.norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.dropout = nn.Dropout(attention_dropout)
        self.out_proj = nn.Linear(mid_dim, input_dim)

    def local_partition(self, x, h_step, w_step, dh, dw):
        b, t, h, w, c = x.size()
        local_x = []
        for i in range(0, h + h_step - dh, h_step):
            top = i
            down = i + dh
            if down > h:
                top = h - dh
                down = h
            for j in range(0, w + w_step - dw, w_step):
                left = j
                right = j + dw
                if right > w:
                    left = w - dw
                    right = w
                local_x.append(x[:, :, top:down, left:right, :])
        local_x = torch.stack(local_x, dim=1)  # b n t dh dw c
        return local_x

    def local_reverse(self, local_x, x, h_step, w_step, dh, dw):
        b, t, h, w, c = x.size()
        x_output = torch.zeros_like(x)
        count = torch.zeros((b, h, w), device=x.device)

        index = 0
        for i in range(0, h + h_step - dh, h_step):
            top = i
            down = i + dh
            if down > h:
                top = h - dh
                down = h
            for j in range(0, w + w_step - dw, w_step):
                left = j
                right = j + dw
                if right > w:
                    left = w - dw
                    right = w
                x_output[:, :, top:down, left:right, :] += local_x[:, index]  # local_x: b n t dh dw c
                count[:, top:down, left:right] += 1
                index += 1
        x_output = x_output / count.unsqueeze(1).unsqueeze(-1)
        return x_output

    def forward(self, x):  # x: B, T, H, W, CV
        b = x.shape[0]
        x = self.norm(x)

        dh = dw = self.patch_size
        loacl_x = self.local_partition(x, dh - dh // 4, dw - dw // 4, dh, dw)  # b n t dh dw c
        q = self.q_proj(loacl_x)
        k = self.k_proj(loacl_x)
        v = self.v_proj(loacl_x)

        q, k, v = map(lambda x: rearrange(x, 'b n t dh dw (head c) -> (b n) head (t dh dw) c',
                                          head=self.num_heads), (q, k, v))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = rearrange(output, '(b n) head (t dh dw) c -> b n t dh dw (head c)', b=b, dh=dh, dw=dw)
        output = self.local_reverse(output, x, dh - dh // 4, dw - dw // 4, dh, dw)  # b t h w c
        output = self.out_proj(output)

        return output


class MemoryBank:
    def __init__(self, para, count_usage: bool, top_k: Optional[int] = None, tlc_flag=False):
        super().__init__()
        self.count_usage = count_usage
        self.top_k = top_k
        self.tlc_flag = tlc_flag

        self.CK = None
        self.CV = None
        self.H = None
        self.W = None
        self.mem_pixel_cnt = 0
        self.mem_semantic_cnt = 0
        self.mem_every = para.mem_every

        self.mem_k = None
        self.mem_vs = [None for i in range(2)]

        # shrinkage and selection are also single tensors
        self.s = None
        self.e = None

        # usage
        if self.count_usage:
            self.use_count = self.life_count = None

        # The hidden state will be stored in a single tensor
        self.hidden_dim = 96
        self.hidden = None

    def _global_matching(self, mk, ms, qk, qe, return_usage=False):
        # mk: B x CK x [N]    - Memory keys
        # ms: B x  1 x [N]    - Memory shrinkage
        # qk: B x CK x [HW/P] - Query keys
        # qe: B x CK x [HW/P] - Query selection

        B, CK, NE = mk.shape

        if qe is not None:
            # (a-b)^2 = a^2 - 2ab + b^2
            mk = mk.transpose(1, 2)
            a_sq = (mk.pow(2) @ qe)
            two_ab = 2 * (mk @ (qk * qe))
            b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
            similarity = (-a_sq + two_ab - b_sq)
        else:
            # similar to STCN if we don't have the selection term
            a_sq = mk.pow(2).sum(1).unsqueeze(2)
            two_ab = 2 * (mk.transpose(1, 2) @ qk)
            similarity = (-a_sq + two_ab)

        if ms is not None:
            ms = ms.flatten(start_dim=1).unsqueeze(2)  # B, THW, 1
            similarity = similarity * ms / math.sqrt(CK)  # B, THW, HW
        else:
            similarity = similarity / math.sqrt(CK)

        if self.top_k is None:
            maxes = torch.max(similarity, dim=1, keepdim=True)[0]
            x_exp = torch.exp(similarity - maxes)
            x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
            affinity = x_exp / x_exp_sum
        else:
            affinity = softmax_w_top(similarity, top=self.top_k)  # B, THW, HW

        if return_usage:
            return affinity, affinity.sum(dim=-1)

        return affinity

    def _readout(self, affinity, mv):
        # mv: B, CV, THW
        return torch.bmm(mv, affinity)  # B, CV, HW

    def match_memory(self, qk, qe, sf):
        b, c_k, h, w = qk.shape

        qk = qk.flatten(start_dim=2)
        sf = sf.permute(0, 2, 3, 1).contiguous()  # B, H, W, CV
        if qe is not None:
            qe = qe.flatten(start_dim=2)

        mk = self.mem_k
        mvs = self.mem_vs
        ms = self.s

        if self.count_usage:
            affinity, usage = self._global_matching(mk, ms, qk, qe, return_usage=True)
            self.update_usage(usage)
        else:
            affinity = self._global_matching(mk, ms, qk, qe)

        readout_mems = []
        attn = Attention().to(self.mem_vs[1].device)
        if self.tlc_flag:
            sf = sf.view(b, 1, h, w, self.CV)  # B, 1, H, W, CV
            mv1 = mvs[1].permute(0, 2, 1).contiguous().view(b, -1, h, w, self.CV)  # B, T, H, W, CV
        else:
            sf = sf.view(b, -1, self.CV)  # B, HW, CV
            mv1 = mvs[1].permute(0, 2, 1).contiguous()  # B, THW, CV

        readout_mems.append(self._readout(affinity, mvs[0]))
        readout_mems.append(attn(sf, mv1).view(b, -1, self.CV).permute(0, 2, 1).contiguous())  # B, CV, HW

        return [readout_mems[i].view(b, self.CV, h, w) for i in range(2)]

    def add_memory(self, key, values, shrinkage, selection=None):
        if not isinstance(values, list):
            values = [values]
        if self.H is None:
            self.H, self.W = key.shape[-2:]
        key = key.flatten(start_dim=2)  # B, CK, HW
        values = [values[i].flatten(start_dim=2) for i in range(2)]  # B, CV, HW
        if shrinkage is not None:
            shrinkage = shrinkage.flatten(start_dim=2)
        if selection is not None:
            selection = selection.flatten(start_dim=2)

        # b = key.shape[0]
        new_count = torch.zeros((key.shape[0], 1, key.shape[2]), device=key.device, dtype=torch.float32)
        new_life = torch.zeros((key.shape[0], 1, key.shape[2]), device=key.device, dtype=torch.float32) + 1e-7

        if self.mem_k is None:
            self.mem_k = key
            self.mem_vs = values
            self.s = shrinkage
            self.e = selection
            if self.count_usage:
                self.use_count = new_count
                self.life_count = new_life
            self.CK = key.shape[1]
            self.CV = values[0].shape[1]

            self.mem_semantic_cnt += 1
        else:
            if self.mem_pixel_cnt == self.mem_every:
                self.remove_obsolete_feature(n=2)

            if self.mem_semantic_cnt == self.mem_every:
                self.update_object_memory(values[1])  # update object memory: self.mem_vs[1]
            else:
                self.add_object_memory(values[1])  # add object memory: self.mem_vs[1]

            self.mem_k = torch.cat([self.mem_k, key], -1)
            self.mem_vs[0] = torch.cat([self.mem_vs[0], values[0]], -1)  # add pixel memory: self.mem_vs[0]
            # self.mem_vs = [torch.cat([self.mem_vs[i], values[i]], -1) for i in range(2)]
            if shrinkage is not None:
                self.s = torch.cat([self.s, shrinkage], -1)
            if selection is not None:
                self.e = torch.cat([self.e, selection], -1)
            if self.count_usage:
                self.use_count = torch.cat([self.use_count, new_count], -1)
                self.life_count = torch.cat([self.life_count, new_life], -1)

        self.mem_pixel_cnt += 1

    def create_hidden_state(self, sample_key):
        b, _, h, w = sample_key.shape
        if self.hidden is None:
            self.hidden = torch.zeros((b, self.hidden_dim, h, w), device=sample_key.device, dtype=torch.float32)

    def set_hidden(self, hidden_state):
        self.hidden = hidden_state

    def get_hidden(self):
        return self.hidden

    def update_usage(self, usage):
        # increase all life count by 1
        # increase use of indexed elements
        if not self.count_usage:
            return

        self.use_count += usage.view_as(self.use_count)
        self.life_count += 1

    def get_usage(self):
        # return normalized usage
        if not self.count_usage:
            raise RuntimeError('I did not count usage!')
        else:
            usage = self.use_count / self.life_count
            return usage

    def add_object_memory(self, new_feature):
        b = new_feature.shape[0]  # new_feature: B, CV, HW
        model = Attention1(patch_size=8).to(self.mem_vs[1].device)
        mv1 = torch.cat([self.mem_vs[1], new_feature], -1)  # B, CV, THW
        mv1 = mv1.permute(0, 2, 1).contiguous()  # B, THW, CV
        mv1 = mv1.view(b, -1, self.H, self.W, self.CV)  # B, T, H, W, CV
        mv1 = model(mv1)
        self.mem_vs[1] = mv1.view(b, -1, self.CV).permute(0, 2, 1).contiguous()  # B, CV, THW

        # self.mem_vs[1] = torch.cat([self.mem_vs[1], new_feature], -1)
        self.mem_semantic_cnt += 1

    def update_object_memory(self, new_feature):
        self.mem_vs[1] = new_feature
        self.mem_semantic_cnt = 1

    def remove_obsolete_feature(self, n=1):
        # normalize with life duration
        usage = self.get_usage().flatten(start_dim=1)
        b = usage.shape[0]
        HW = self.H * self.W
        _, indices = torch.topk(usage, k=(self.mem_every - n) * HW, dim=-1, largest=True, sorted=True)  # 从大到小排序

        mk = torch.zeros((b, self.CK, (self.mem_every - n) * HW), device=usage.device, dtype=torch.float32)
        mv0 = torch.zeros((b, self.CV, (self.mem_every - n) * HW), device=usage.device, dtype=torch.float32)
        mv1 = self.mem_vs[1]
        # mv1 = torch.zeros((b, self.CV, (self.mem_every - n) * HW), device=usage.device, dtype=torch.float32)
        if self.s is not None:
            ms = torch.zeros((b, 1, (self.mem_every - n) * HW), device=usage.device, dtype=torch.float32)
        if self.e is not None:
            qe = torch.zeros((b, self.CK, (self.mem_every - n) * HW), device=usage.device, dtype=torch.float32)
        use_c = torch.zeros((b, 1, (self.mem_every - n) * HW), device=usage.device, dtype=torch.float32)
        life_c = torch.zeros((b, 1, (self.mem_every - n) * HW), device=usage.device, dtype=torch.float32)

        for i in range(b):
            mk[i] = self.mem_k[i, :, indices[i]]
            mv0[i] = self.mem_vs[0][i, :, indices[i]]
            # mv1[i] = self.mem_vs[1][i, :, indices[i]]
            if self.s is not None:
                ms[i] = self.s[i, :, indices[i]]
            if self.e is not None:
                qe[i] = self.e[i, :, indices[i]]
            use_c[i] = self.use_count[i, :, indices[i]]
            life_c[i] = self.life_count[i, :, indices[i]]

        self.mem_k = mk
        self.mem_vs = [mv0, mv1]
        self.s = ms if self.s is not None else None
        self.e = qe if self.e is not None else None
        self.use_count = use_c
        self.life_count = life_c

        self.mem_pixel_cnt -= n

    def init_memory(self):
        if self.mem_k is not None:
            self.mem_k = self.mem_k.detach()
            self.mem_vs = [self.mem_vs[i].detach() for i in range(2)]
            if self.s is not None:
                self.s = self.s.detach()
            if self.e is not None:
                self.e = self.e.detach()
            if self.hidden is not None:
                self.hidden = self.get_hidden().detach()
            if self.count_usage:
                self.use_count = self.use_count.detach()
                self.life_count = self.life_count.detach()

    def clear_memory(self):
        self.mem_k = None
        self.mem_vs = [None for i in range(2)]
        self.s = None
        self.e = None
        self.mem_pixel_cnt = 0
        self.mem_semantic_cnt = 0
        if self.hidden is not None:
            self.hidden = self.get_hidden().detach()
        if self.count_usage:
            self.use_count = self.life_count = None


if __name__ == '__main__':
    x = torch.randn(2, 5, 128, 128, 64)  # b, t, h, w, c
    y = torch.randn(2, 5, 128, 128, 64)
    model = Attention(64, 64, 4)
    x = model(x, y)
    print(x.shape)
