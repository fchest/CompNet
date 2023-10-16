import os
import json
import torch
import torch.fft
import torch.nn.functional as F
import math
import logging
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
EPSILON = np.finfo(np.float32).eps


def logger_print(log):
    logging.info(log)
    print(log)


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


class ToTensor(object):
    def __call__(self,
                 x,
                 type="float"):
        if type == "float":
            return torch.FloatTensor(x)
        elif type == "int":
            return torch.IntTensor(x)


def pad_to_longest(batch_data):
    """
    pad the waves with the longest length among one batch chunk
    :param batch_data:
    :return:
    """
    mix_wav_batch_list, target_wav_batch_list, wav_len_list = batch_data[0]
    mix_tensor, target_tensor = nn.utils.rnn.pad_sequence(mix_wav_batch_list, batch_first=True), \
                                nn.utils.rnn.pad_sequence(target_wav_batch_list, batch_first=True)  # (B,L,M)
    return mix_tensor, target_tensor, wav_len_list


class BatchInfo(object):
    def __init__(self, feats, labels, frame_mask_list):
        self.feats = feats
        self.labels = labels
        self.frame_mask_list = frame_mask_list



def init_kernel(frame_len, frame_hop, num_fft=None, window="sqrt_hann", is_fft=True):
    """
        return: dft_matrix: (frame_len, fft_size+2)
                kernel: (frame_len,1,frame_len)->just enframe
                        (fft_size+2, 1, frame_len)->stft
    """
    if window != "sqrt_hann":
        raise RuntimeError("Now only support sqrt hanning window in order "
                           "to make signal perfectly reconstructed")
    #fft_size = 2 ** math.ceil(math.log2(frame_len)) if not num_fft else num_fft
    fft_size = num_fft
    window = torch.hann_window(frame_len) ** 0.5
    #S_ = 0.5 * (frame_len * frame_len / frame_hop) ** 0.5
    S_ = 0.5 * (frame_len / frame_hop) ** 0.5
    w = torch.fft.rfft(torch.eye(frame_len) / S_)
    dft_matrix = torch.stack([w.real, w.imag], -1)
    dft_matrix = torch.transpose(dft_matrix, 0, 2) * window
    if not is_fft:
        kernel = torch.reshape(torch.eye(frame_len), (frame_len, 1, frame_len))
    else:
        kernel = torch.reshape(dft_matrix, (fft_size+2, 1, frame_len))
    return dft_matrix.reshape(fft_size+2, frame_len).T, kernel


class STFTBase(nn.Module):
    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 num_fft=None,
                 is_fft=True,
                 device="cpu",
                 ):
        super(STFTBase, self).__init__()
        dft_matrix, K = init_kernel(frame_len, frame_hop, num_fft=num_fft, window=window, is_fft=is_fft)
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.dft_matrix = nn.Parameter(dft_matrix, requires_grad=False).to(device)
        self.K = nn.Parameter(K, requires_grad=False).to(device)
        self.stride = frame_hop
        self.window = window

    def freeze(self): self.K.requires_grad = False
    def unfreeze(self): self.K.requires_grad = True
    def check_nan(self):
        num_nan = torch.sum(torch.isnan(self.K))
        if num_nan:
            raise RuntimeError(
                "detect nan in STFT kernels: {:d}".format(num_nan))
    def extra_repr(self):
        return "window={0}, stride={1}, requires_grad={2}, kernel_size={3[0]}x{3[2]}".format(self.window, self.stride, self.K.requires_grad, self.K.shape)


class ConvEnframe(STFTBase):
    def __init__(self, *args, **kwargs):
        super(ConvEnframe, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
            x: (B, L) or (B, 1, L)
            return:
                    frame_x: (B, T, K)
                    stft_x: (B, 2, T, F)
                    dft_matrix: (K, 2F)
        """
        if x.dim() not in [2, 3]:
            print(x.shape)
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(x.dim()))
        self.check_nan()
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        b_size = x.shape[0]
        frame_x = F.conv1d(x, self.K, stride=self.stride, padding=0).transpose(1, 2)
        stft_x = torch.matmul(frame_x, self.dft_matrix)
        real_x, imag_x = torch.chunk(stft_x, 2, dim=-1)
        stft_x = torch.stack((real_x, imag_x), dim=1)
        return frame_x, stft_x, self.dft_matrix


class ConvOLA(STFTBase):
    def __init__(self, *args, **kwargs):
        super(ConvOLA, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
            x: (B, T, K) or (B, K)
            return:
                    x: (B, L)
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(x.dim()))
        self.check_nan()
        if x.dim() == 2:
            x = torch.unsqueeze(x, dim=1)
        x = F.conv_transpose1d(x.transpose(1, 2), self.K, stride=self.stride, padding=0)
        S = self.frame_len / self.frame_hop
        x /= S
        return x


class STFT(STFTBase):
    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            print(x.shape)
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                x.dim()))
        self.check_nan()
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        b_size = x.shape[0]
        c = F.conv1d(x, self.K, stride=self.stride, padding=0)
        r, i = torch.chunk(c, 2, dim=1)
        m = (r ** 2 + i ** 2) ** 0.5
        p = torch.atan2(i, r)
        return m, p, r, i


class iSTFT(STFTBase):
    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, squeeze=False):
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        self.check_nan()
        if p.dim() == 2:
            p = torch.unsqueeze(p, 0)
            m = torch.unsqueeze(m, 0)
        r = m * torch.cos(p)
        i = m * torch.sin(p)
        c = torch.cat([r, i], dim=1)
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        if squeeze:
            s = torch.squeeze(s, dim=1)
        return s


class TorchSignalToFrames(object):
    def __init__(self, frame_size=320, frame_shift=160):
        super(TorchSignalToFrames, self).__init__()
        self.frame_size = frame_size
        self.frame_shift = frame_shift

    def __call__(self, in_sig):
        sig_len = in_sig.shape[-1]
        #nframes = (sig_len // self.frame_shift)
        nframes = (sig_len - self.frame_size) // self.frame_shift + 1
        a = torch.zeros(tuple(in_sig.shape[:-1]) + (nframes, self.frame_size), device=in_sig.device)
        start = 0
        end = start + self.frame_size
        k = 0
        for i in range(nframes):
            if end < sig_len:
                a[..., i, :] = in_sig[..., start:end]
                k += 1
            else:
                tail_size = sig_len - start
                a[..., i, :tail_size] = in_sig[..., start:]

            start = start + self.frame_shift
            end = start + self.frame_size
        return a


class TorchOLA(nn.Module):
    r"""Overlap and add on gpu using torch tensor"""
    # Expects signal at last dimension
    def __init__(self, frame_shift=160):
        super(TorchOLA, self).__init__()
        self.frame_shift = frame_shift

    def forward(self, inputs):
        nframes = inputs.shape[-2]
        frame_size = inputs.shape[-1]
        frame_step = self.frame_shift
        sig_length = (nframes - 1) * frame_step + frame_size
        sig = torch.zeros(list(inputs.shape[:-2]) + [sig_length], dtype=inputs.dtype, device=inputs.device)
        ones = torch.zeros_like(sig)
        start = 0
        end = start + frame_size
        for i in range(nframes):
            sig[..., start:end] += inputs[..., i, :]
            ones[..., start:end] += 1.
            start = start + frame_step
            end = start + frame_size
        return sig / ones


class Conv1D(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


def json_extraction(file_path, json_path, data_type):
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    file_list = os.listdir(file_path)
    file_num = len(file_list)
    json_list = []

    for i in range(file_num):
        file_name = file_list[i]
        file_name = os.path.splitext(file_name)[0]
        json_list.append(file_name)

    with open(os.path.join(json_path, "{}_files.json".format(data_type)), "w") as f:
        json.dump(json_list, f, indent=4)
    return os.path.join(json_path, "{}_files.json".format(data_type))


def complex_mul(inpt1, inpt2):
    """
    inpt1: (B,2,...) or (...,2)
    inpt2: (B,2,...) or (...,2)
    """
    if inpt1.shape[1] == 2:
        out_r = inpt1[:,0,...]*inpt2[:,0,...] - inpt1[:,-1,...]*inpt2[:,-1,...]
        out_i = inpt1[:,0,...]*inpt2[:,-1,...] + inpt1[:,-1,...]*inpt2[:,0,...]
        return torch.stack((out_r, out_i), dim=1)
    elif inpt1.shape[-1] == 2:
        out_r = inpt1[...,0]*inpt2[...,0] - inpt1[...,-1]*inpt2[...,-1]
        out_i = inpt1[...,0]*inpt2[...,-1] + inpt1[...,-1]*inpt2[...,0]
        return torch.stack((out_r, out_i), dim=-1)
    else:
        raise RuntimeError("Only supports two tensor formats")


def complex_conj(inpt):
    """
    inpt: (B,2,...) or (...,2)
    """
    if inpt.shape[1] == 2:
        inpt_r, inpt_i = inpt[:,0,...], inpt[:,-1,...]
        return torch.stack((inpt_r, -inpt_i), dim=1)
    elif inpt.shape[-1] == 2:
        inpt_r, inpt_i = inpt[...,0], inpt[...,-1]
        return torch.stack((inpt_r, -inpt_i), dim=-1)


def complex_div(inpt1, inpt2):
    """
    inpt1: (B,2,...) or (...,2)
    inpt2: (B,2,...) or (...,2)
    """
    if inpt1.shape[1] == 2:
        inpt1_r, inpt1_i = inpt1[:,0,...], inpt1[:,-1,...]
        inpt2_r, inpt2_i = inpt2[:,0,...], inpt2[:,-1,...]
        denom = torch.norm(inpt2, dim=1)**2.0 + EPSILON
        out_r = inpt1_r * inpt2_r + inpt1_i * inpt2_i
        out_i = inpt1_i * inpt2_r - inpt1_r * inpt2_i
        return torch.stack((out_r/denom, out_i/denom), dim=1)
    elif inpt1.shape[-1] == 2:
        inpt1_r, inpt1_i = inpt1[...,0], inpt1[...,-1]
        inpt2_r, inpt2_i = inpt2[...,0], inpt2[...,-1]
        denom = torch.norm(inpt2, dim=-1)**2.0 + EPSILON
        out_r = inpt1_r * inpt2_r + inpt1_i * inpt2_i
        out_i = inpt1_i * inpt2_r - inpt1_r * inpt2_i
        return torch.stack((out_r/denom, out_i/denom), dim=-1)


class ActiSwitch(nn.Module):
    def __init__(self,
                 acti_type: str,
                 channel_num: int = 0,
                 ):
        super(ActiSwitch, self).__init__()
        self.acti_type = acti_type
        self.channel_num = channel_num

        if acti_type.lower() == "relu":
            self.acti = nn.ReLU()
        elif acti_type.lower() == "prelu":
            self.acti = nn.PReLU(channel_num)
        elif acti_type.lower() == "leakyrelu":
            self.acti = nn.LeakyReLU(0.1)
        elif acti_type.lower() == "elu":
            self.acti = nn.ELU()

    def forward(self, inpt):
        return self.acti(inpt)


class NormSwitch(nn.Module):
    def __init__(self,
                 norm_type: str,
                 format: str,
                 channel_num: int,
                 frequency_num: int = 0,
                 affine: bool = True,
                 ):
        super(NormSwitch, self).__init__()
        self.norm_type = norm_type
        self.format = format
        self.channel_num = channel_num
        self.frequency_num = frequency_num
        self.affine = affine

        if norm_type == "BN":
            if format in ["1D", "1d"]:
                self.norm = nn.BatchNorm1d(channel_num, affine=True)
            elif format in ["2D", "2d"]:
                self.norm = nn.BatchNorm2d(channel_num, affine=True)
        elif norm_type == "IN":
            if format in ["1D", "1d"]:
                self.norm = nn.InstanceNorm1d(channel_num, affine=True)
            elif format in ["2D", "2d"]:
                self.norm = nn.InstanceNorm2d(channel_num, affine=True)
        elif norm_type == "iLN":
            if format in ["1D", "1d"]:
                self.norm = InstantLayerNorm1d(channel_num, affine=True)
            elif format in ["2D", "2d"]:
                self.norm = InstantLayerNorm2d(frequency_num, channel_num, affine=True)
        elif norm_type == "cLN":
            if format in ["1D", "1d"]:
                self.norm = CumulativeLayerNorm1d(channel_num, affine=True)
            elif format in ["2D", "2d"]:
                self.norm = CumulativeLayerNorm2d(frequency_num, channel_num, affine=True)
        else:
            raise RuntimeError("Only BN, IN, iLN and cLN are supported currently")

    def forward(self, inpt):
        return self.norm(inpt)


class CumulativeLayerNorm2d(nn.Module):
    def __init__(self,
                 frequency_num: int,
                 channel_num: int,
                 affine=True,
                 eps=1e-5,
                 ):
        super(CumulativeLayerNorm2d, self).__init__()
        self.frequency_num = frequency_num
        self.channel_num = channel_num
        self.eps = eps
        self.affine = affine

        if affine:
            self.gain = nn.Parameter(torch.ones(1,channel_num,1,frequency_num), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1,channel_num,1,frequency_num), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1,channel_num,1,frequency_num))
            self.bias = Variable(torch.zeros(1,channel_num,1,frequency_num))

    def forward(self, inpt):
        """
        :param inpt: (B,C,T,F)
        :return:
        """
        b_size, channel, seq_len, freq_num = inpt.shape
        step_sum = inpt.sum([1,3], keepdim=True)  # (B,1,T,1)
        step_pow_sum = inpt.pow(2).sum([1,3], keepdim=True)  # (B,1,T,1)
        cum_sum = torch.cumsum(step_sum, dim=-2)  # (B,1,T,1)
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=-2)  # (B,1,T,1)

        entry_cnt = np.arange(channel*freq_num, channel*freq_num*(seq_len+1), channel*freq_num)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1,1,seq_len,1).expand_as(cum_sum)
        cum_mean = cum_sum / entry_cnt
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean) / cum_std
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class CumulativeLayerNorm1d(nn.Module):
    def __init__(self,
                 channel_num,
                 affine=True,
                 eps=1e-5,
                 ):
        super(CumulativeLayerNorm1d, self).__init__()
        self.channel_num = channel_num
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1, channel_num, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, channel_num, 1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, channel_num, 1))
            self.bias = Variable(torch.zeros(1, channel_num, 1))

    def forward(self, inpt):
        # inpt: (B,C,T)
        b_size, channel, seq_len = inpt.shape
        cum_sum = torch.cumsum(inpt.sum(1), dim=1)  # (B,T)
        cum_power_sum = torch.cumsum(inpt.pow(2).sum(1), dim=1)  # (B,T)

        entry_cnt = np.arange(channel, channel*(seq_len+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(inpt.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)  # (B,T)

        cum_mean = cum_sum / entry_cnt  # (B,T)
        cum_var = (cum_power_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)
        cum_std = (cum_var + self.eps).sqrt()

        x = (inpt - cum_mean.unsqueeze(dim=1).expand_as(inpt)) / cum_std.unsqueeze(dim=1).expand_as(inpt)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class InstantLayerNorm2d(nn.Module):
    def __init__(self,
                 frequency_num: int,
                 channel_num: int,
                 affine: bool=True,
                 eps=1e-5,
                 ):
        super(InstantLayerNorm2d, self).__init__()
        self.frequency_num = frequency_num
        self.channel_num = channel_num
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1, channel_num, 1, frequency_num), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, channel_num, 1, frequency_num), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, channel_num, 1, frequency_num))
            self.bias = Variable(torch.zeros(1, channel_num, 1, frequency_num))

    def forward(self, inpt):
        # inpt: (B,C,T,F)
        ins_mean = torch.mean(inpt, dim=[1,3], keepdim=True)  # (B,1,T,1)
        ins_std = (torch.std(inpt, dim=[1,3], keepdim=True) + self.eps).pow(0.5)  # (B,1,T,1)
        x = (inpt - ins_mean) / ins_std
        return x * self.gain.type(x.type()) + self.bias.type(x.type())


class InstantLayerNorm1d(nn.Module):
    def __init__(self,
                 channel_num: int,
                 affine: bool=True,
                 eps=1e-5,
                 ):
        super(InstantLayerNorm1d, self).__init__()
        self.channel_num = channel_num
        self.affine = affine
        self.eps = eps

        if affine:
            self.gain = nn.Parameter(torch.ones(1, channel_num, 1), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(1, channel_num, 1), requires_grad=True)
        else:
            self.gain = Variable(torch.ones(1, channel_num, 1))
            self.bias = Variable(torch.zeros(1, channel_num, 1))

    def forward(self, inpt):
        # inpt: (B,C,T)
        b_size, channel, seq_len = inpt.shape
        ins_mean = torch.mean(inpt, dim=1, keepdim=True)  # (B,1,T)
        ins_std = (torch.var(inpt, dim=1, keepdim=True) + self.eps).pow(0.5)  # (B,1,T)
        x = (inpt - ins_mean) / ins_std
        return x * self.gain.type(x.type()) + self.bias.type(x.type())


# obtain ERB filter-banks
class EquivalentRectangularBandwidth():
    def __init__(self,
                 nfreqs: int = 257,
                 sample_rate: int = 16000,
                 total_erb_bands: int = 64,
                 low_freq: float = 20,
                 max_freq: float = 16000//2,
                 ):
        if not low_freq:
            low_freq = 20
        if not max_freq:
            max_freq = sample_rate // 2
        freqs = np.linspace(0, max_freq, nfreqs)  # 每个STFT频点对应多少Hz
        self.EarQ = 9.265  # _ERB_Q
        self.minBW = 24.7  # minBW
        # 在ERB刻度上建立均匀间隔
        erb_low = self.freq2erb(low_freq)  # 最低 截止频率
        erb_high = self.freq2erb(max_freq)  # 最高 截止频率
        # 在ERB频率上均分为(total_erb_bands +2)个 频带
        erb_lims = np.linspace(erb_low, erb_high, total_erb_bands + 2)
        cutoffs = self.erb2freq(erb_lims)  # 将 ERB频率再转到 hz频率, 在线性频率Hz上找到ERB截止频率对应的频率
        # self.nfreqs  F
        # self.freqs # 每个STFT频点对应多少Hz
        self.filters = self.get_bands(total_erb_bands, nfreqs, freqs, cutoffs)

    def freq2erb(self, frequency):
        """ [Hohmann2002] Equation 16"""
        return self.EarQ * np.log(1 + frequency / (self.minBW * self.EarQ))

    def erb2freq(self, erb):
        """ [Hohmann2002] Equation 17"""
        return (np.exp(erb / self.EarQ) - 1) * self.minBW * self.EarQ

    def get_bands(self,
                  total_erb_bands,
                  nfreqs,
                  freqs,
                  cutoffs):
        """
        获取erb bands、索引、带宽和滤波器形状
        :param erb_bands_num: ERB 频带数
        :param nfreqs: 频点数 F
        :param freqs: 每个STFT频点对应多少Hz
        :param cutoffs: 中心频率 Hz
        :param erb_points: ERB频带界限 列表
        :return:
        """
        cos_filts = np.zeros([nfreqs, total_erb_bands])  # (F, ERB)
        for i in range(total_erb_bands):
            lower_cutoff = cutoffs[i]  # 上限截止频率 Hz
            higher_cutoff = cutoffs[i + 2]  # 下限截止频率 Hz, 相邻filters重叠50%

            lower_index = np.min(np.where(freqs > lower_cutoff))  # 下限截止频率对应的Hz索引 Hz。np.where 返回满足条件的索引
            higher_index = np.max(np.where(freqs < higher_cutoff))  # 上限截止频率对应的Hz索引
            avg = (self.freq2erb(lower_cutoff) + self.freq2erb(higher_cutoff)) / 2
            rnge = self.freq2erb(higher_cutoff) - self.freq2erb(lower_cutoff)
            cos_filts[lower_index:higher_index + 1, i] = np.cos(
                (self.freq2erb(freqs[lower_index:higher_index + 1]) - avg) / rnge * np.pi)  # 减均值，除方差

        # 加入低通和高通，得到完美的重构
        filters = np.zeros([nfreqs, total_erb_bands + 2])  # (F, ERB)
        filters[:, 1:total_erb_bands + 1] = cos_filts
        # 低通滤波器上升到第一个余cos filter的峰值
        higher_index = np.max(np.where(freqs < cutoffs[1]))  # 上限截止频率对应的Hz索引
        filters[:higher_index + 1, 0] = np.sqrt(1 - np.power(filters[:higher_index + 1, 1], 2))
        # 高通滤波器下降到最后一个cos filter的峰值
        lower_index = np.min(np.where(freqs > cutoffs[total_erb_bands]))
        filters[lower_index:nfreqs, total_erb_bands + 1] = np.sqrt(
            1 - np.power(filters[lower_index:nfreqs, total_erb_bands], 2))
        return cos_filts


def sisnr(est, label):
    label_power = np.sum(label**2.0) + 1e-8
    scale = np.sum(est*label) / label_power
    est_true = scale * label
    est_res = est - est_true
    true_power = np.sum(est_true**2.0, axis=0) + 1e-8
    res_power = np.sum(est_res**2.0, axis=0) + 1e-8
    sdr = 10*np.log10(true_power) - 10*np.log10(res_power)
    return sdr

def cal_pesq(id, esti_utts, clean_utts, fs):
    clean_utt, esti_utt = clean_utts[id,:], esti_utts[id,:]
    from pypesq import pesq
    pesq_score = pesq(clean_utt, esti_utt, fs=fs)
    return pesq_score

def cal_stoi(id, esti_utts, clean_utts, fs):
    clean_utt, esti_utt = clean_utts[id,:], esti_utts[id,:]
    from pystoi import stoi
    stoi_score = stoi(clean_utt, esti_utt, fs, extended=True)
    return 100*stoi_score

def cal_sisnr(id, esti_utts, clean_utts, fs):
    clean_utt, esti_utt = clean_utts[id,:], esti_utts[id,:]
    sisnr_score = sisnr(esti_utt, clean_utt)
    return sisnr_score
