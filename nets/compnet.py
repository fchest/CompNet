import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Tuple, List
from torch.autograd import Variable
from utils.util import TorchOLA, TorchSignalToFrames
from nets.module import InterIntraRNN, FrameUNetEncoder, FrameUNetDecoder, TCMList, FreqU2NetEncoder, FreqUNetEncoder
torch_eps = torch.finfo(torch.float32).eps


class CompNet(nn.Module):
    def __init__(self,
                 win_size: int = 320,
                 win_shift: int = 160,
                 fft_num: int = 320,
                 k1: Union[Tuple, List] = (2, 3),
                 k2: Union[Tuple, List] = (2, 3),
                 c: int = 64,
                 embed_dim: int = 64,
                 kd1: int = 5,
                 cd1: int = 64,
                 d_feat: int = 256,
                 hidden_dim: int = 64,
                 hidden_num: int = 2,
                 group_num: int = 2,
                 dilations: Union[Tuple, List] = (1,2,5,9),
                 inter_connect: str = "cat",
                 intra_connect: str = "cat",
                 norm_type: str = "iLN",
                 rnn_type: str = "LSTM",
                 post_type: str = "collaborative",
                 is_dual_rnn: bool = True,
                 is_causal: bool = True,
                 is_u2: bool = True,
                 is_mu_compress: bool = True,
                 ):
        super(CompNet, self).__init__()
        self.win_size = win_size
        self.win_shift = win_shift
        self.fft_num = fft_num
        self.k1 = tuple(k1)
        self.k2 = tuple(k2)
        self.c = c
        self.embed_dim = embed_dim
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.group_num = group_num
        self.dilations = tuple(dilations)
        self.inter_connect = inter_connect
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.rnn_type = rnn_type
        self.post_type = post_type
        self.is_dual_rnn = is_dual_rnn
        self.is_causal = is_causal
        self.is_u2 = is_u2
        self.is_mu_compress = is_mu_compress
        #
        # first simultaneously pre-estimate mag and phase
        pre_kwargs = {
            "k1": k1,
            "c": c,
            "embed_dim": embed_dim,
            "kd1": kd1,
            "cd1": cd1,
            "d_feat": d_feat,
            "hidden_dim": hidden_dim,
            "hidden_num": hidden_num,
            "group_num": group_num,
            "dilations": dilations,
            "inter_connect": inter_connect,
            "norm_type": norm_type,
            "rnn_type": rnn_type,
            "is_dual_rnn": is_dual_rnn,
            "is_causal": is_causal
        }
        self.pre_seperate = TCNN(cin=1, **pre_kwargs)
        post_kwargs = {
            "k1": k1,
            "k2": k2,
            "c": c,
            "kd1": kd1,
            "cd1": cd1,
            "d_feat": d_feat,
            "fft_num": fft_num,
            "dilations": dilations,
            "intra_connect": intra_connect,
            "norm_type": norm_type,
            "is_causal": is_causal,
            "is_u2": is_u2,
            "group_num": group_num
        }

        if self.post_type == "vanilla":
            self.post = VanillaPostProcessing(cin=4, **post_kwargs)
        elif self.post_type == "collaborative":
            self.post = CollaborativePostProcessing(cin=4, **post_kwargs)

        # enframe and ola
        self.enframe = TorchSignalToFrames(frame_size=win_size, frame_shift=win_shift)
        self.ola = TorchOLA(frame_shift=win_shift)

    def forward(self, inpt: Tensor) -> tuple:
        """
            inpt: (B, L)
        """
        #
        frame_inpt = self.enframe(inpt)
        stft_inpt = torch.stft(inpt,
                               self.fft_num,
                               self.win_shift,
                               self.win_size,
                               window=torch.sqrt(torch.hann_window(self.win_size).to(inpt.device)))
        esti_x = self.pre_seperate(frame_inpt)
        esti_wav = self.ola(esti_x)
        esti_stft = torch.stft(esti_wav,
                               self.fft_num,
                               self.win_shift,
                               self.win_size,
                               window=torch.sqrt(torch.hann_window(self.win_size).to(inpt.device)))
        p = 0.5 if self.is_mu_compress else 1.0
        esti_mag, esti_phase = ((torch.norm(esti_stft, dim=-1) + torch_eps) ** p).transpose(-2, -1), \
                               torch.atan2(esti_stft[..., -1], esti_stft[..., 0]).transpose(-2, -1)
        mix_mag, mix_phase = ((torch.norm(stft_inpt, dim=-1) + torch_eps) ** p).transpose(-2, -1), \
                               torch.atan2(stft_inpt[..., -1], stft_inpt[..., 0]).transpose(-2, -1)

        comp_phase = torch.stack((mix_mag*torch.cos(esti_phase), mix_mag*torch.sin(esti_phase)), dim=1)
        comp_mag = torch.stack((esti_mag*torch.cos(mix_phase), esti_mag*torch.sin(mix_phase)), dim=1)

        post_x = self.post(comp_mag, comp_phase)
        return [esti_wav, esti_mag], post_x


class TCNN(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: Union[Tuple, List],
                 c: int,
                 embed_dim: int,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 hidden_dim: int,
                 hidden_num: int,
                 group_num: int,
                 dilations: Union[Tuple, List],
                 inter_connect: str,
                 norm_type: str,
                 rnn_type: str,
                 is_dual_rnn: bool,
                 is_causal: bool = True,
                 ):
        super(TCNN, self).__init__()
        self.cin = cin
        self.k1 = tuple(k1)
        self.c = c
        self.embed_dim = embed_dim
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.group_num = group_num
        self.dilations = dilations
        self.inter_connect = inter_connect
        self.norm_type = norm_type
        self.rnn_type = rnn_type
        self.is_dual_rnn = is_dual_rnn
        self.is_causal = is_causal
        #
        self.en = FrameUNetEncoder(cin=cin,
                                   k1=k1,
                                   c=c,
                                   norm_type=norm_type)
        self.de = FrameUNetDecoder(c=c,
                                   embed_dim=embed_dim,
                                   k1=k1,
                                   inter_connect=inter_connect,
                                   norm_type=norm_type)
        stcns = []
        for i in range(group_num):
            stcns.append(TCMList(kd1, cd1, d_feat, norm_type, dilations, is_causal))
        self.stcns = nn.ModuleList(stcns)
        if is_dual_rnn:
            self.dual_rnn = InterIntraRNN(embed_dim=embed_dim,
                                          hidden_dim=hidden_dim,
                                          hidden_num=hidden_num,
                                          rnn_type=rnn_type,
                                          is_causal=is_causal,
                                          )
        else:
            assert embed_dim == 1, "the embed_dim should be 1 if no dual-rnn is adopted!"

    def forward(self, inpt: Tensor) -> Tensor:
        """
            inpt: (B, T, K) or (B, 1, T, K)
            return:
                    (B, T, K)
        """
        if inpt.ndim == 3:
            inpt = inpt.unsqueeze(dim=1)
        en_x, en_list = self.en(inpt)
        b_size, c, seq_len, k = en_x.shape
        x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        x_acc = Variable(torch.zeros_like(x, device=x.device), requires_grad=True)
        for i in range(self.group_num):
            x = self.stcns[i](x)
            x_acc = x_acc + x
        x = x_acc.clone()
        x = x.view(b_size, c, k, seq_len).transpose(-2, -1)
        embed_x = self.de(x, en_list)
        #
        if self.is_dual_rnn:
            x = self.dual_rnn(embed_x)
        else:
            x = embed_x.squeeze(dim=1)
        return x


class CollaborativePostProcessing(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: Union[Tuple, List],
                 k2: Union[Tuple, List],
                 c: int,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 fft_num: int,
                 dilations: Union[Tuple, List],
                 intra_connect: str,
                 norm_type: str,
                 is_causal: bool,
                 is_u2: bool,
                 group_num: int = 2,
                 ):
        super(CollaborativePostProcessing, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.fft_num = fft_num
        self.dilations = dilations
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.is_causal = is_causal
        self.is_u2 = is_u2
        self.group_num = group_num
        #
        freq_dim = fft_num//2+1
        if self.is_u2:
            self.en = FreqU2NetEncoder(cin=cin, k1=k1, k2=k2, c=c, intra_connect=intra_connect, norm_type=norm_type)
        else:
            self.en = FreqUNetEncoder(cin=cin, k1=k1, c=c, norm_type=norm_type)
        self.gain_in_conv = nn.Conv1d(d_feat+freq_dim, d_feat, kernel_size=1)
        self.resi_in_conv = nn.Conv1d(d_feat+freq_dim*2, d_feat, kernel_size=1)
        stcns_gain, stcns_resi = [], []
        for i in range(group_num):
            stcns_gain.append(TCMList(kd1, cd1, d_feat, norm_type, dilations, is_causal))
            stcns_resi.append(TCMList(kd1, cd1, d_feat, norm_type, dilations, is_causal))
        self.stcns_gain, self.stcns_resi = nn.ModuleList(stcns_gain), nn.ModuleList(stcns_resi)
        self.gain_out_conv = nn.Sequential(
            nn.Conv1d(d_feat, freq_dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.resi_out_conv = nn.Conv1d(d_feat, freq_dim*2, kernel_size=1)
        #
        self.real_rnn = nn.GRU(input_size=freq_dim*2, hidden_size=freq_dim, num_layers=1, batch_first=True)
        self.imag_rnn = nn.GRU(input_size=freq_dim*2, hidden_size=freq_dim, num_layers=1, batch_first=True)
        self.real_decode = nn.Linear(freq_dim, freq_dim, bias=False)
        self.imag_decode = nn.Linear(freq_dim, freq_dim, bias=False)

    def forward(self, comp_mag: Tensor, comp_phase: Tensor) -> Tensor:
        """
            comp_mag: (B, 2, T, F)
            comp_phase: (B, 2, T, F)
        """
        inpt_x = torch.cat((comp_mag, comp_phase), dim=1)
        en_x, _ = self.en(inpt_x)
        b_size, c, seq_len, f = en_x.shape
        en_x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        # gain branch
        gain_branch_inpt = torch.cat((en_x, torch.norm(comp_mag, dim=1).transpose(-2, -1)), dim=1)
        gain_branch_x = self.gain_in_conv(gain_branch_inpt)
        # resi branch
        resi_branch_inpt = torch.cat((en_x, comp_phase.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)), dim=1)
        resi_branch_x = self.resi_in_conv(resi_branch_inpt)
        #
        gain_x, resi_x = gain_branch_x.clone(), resi_branch_x.clone()
        for i in range(self.group_num):
            gain_x = self.stcns_gain[i](gain_x)
            resi_x = self.stcns_resi[i](resi_x)
        gain = self.gain_out_conv(gain_x).transpose(-2, -1)
        com_resi = self.resi_out_conv(resi_x)
        resi_r, resi_i = torch.chunk(com_resi, 2, dim=1)
        resi = torch.stack((resi_r, resi_i), dim=1).transpose(-2, -1)
        # collaboratively recover
        comp_phase = comp_phase * gain.unsqueeze(dim=1)
        comp_mag = comp_mag + resi
        # fusion
        real_x = torch.cat((comp_phase[:,0,...], comp_mag[:,0,...]), dim=-1)
        print("real_x.shape",real_x.shape)
        imag_x = torch.cat((comp_phase[:,-1,...], comp_mag[:,-1,...]), dim=-1)
        print("imag_x.shape",imag_x.shape)

        real_x, imag_x = self.real_decode(self.real_rnn(real_x)[0]), \
                         self.imag_decode(self.imag_rnn(imag_x)[0])
        return torch.stack((real_x, imag_x), dim=1)


class VanillaPostProcessing(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: Union[Tuple, List],
                 k2: Union[Tuple, List],
                 c: int,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 fft_num: int,
                 dilations: Union[Tuple, List],
                 intra_connect: str,
                 norm_type: str,
                 is_causal: bool,
                 is_u2: bool,
                 group_num: int = 2,
                 ):
        super(VanillaPostProcessing, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.fft_num = fft_num
        self.dilations = dilations
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.is_causal = is_causal
        self.is_u2 = is_u2
        self.group_num = group_num
        #
        freq_dim = fft_num//2+1
        if self.is_u2:
            self.en = FreqU2NetEncoder(cin=cin, k1=k1, k2=k2, c=c, intra_connect=intra_connect, norm_type=norm_type)
        else:
            self.en = FreqUNetEncoder(cin=cin, k1=k1, c=c, norm_type=norm_type)
        self.in_conv = nn.Conv1d(d_feat+freq_dim*4, d_feat, kernel_size=1)
        stcns = []
        for i in range(group_num):
            stcns.append(TCMList(kd1, cd1, d_feat, norm_type, dilations, is_causal))
        self.stcns = nn.ModuleList(stcns)
        self.real_conv, self.imag_conv = nn.Conv1d(d_feat, freq_dim, kernel_size=1), \
                                         nn.Conv1d(d_feat, freq_dim, kernel_size=1)

    def forward(self, comp_mag: Tensor, comp_phase: Tensor) -> Tensor:
        """
            comp_mag: (B, 2, T, F)
            comp_phase: (B, 2, T, F)
            return:
                    (B, 2, T, F)
        """
        inpt_x = torch.cat((comp_mag, comp_phase), dim=1)
        en_x, _ = self.en(inpt_x)
        b_size, c, seq_len, f = en_x.shape
        en_x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        en_x = torch.cat((en_x,
                          comp_mag.transpose(-2, -1).contiguous().view(b_size, -1, seq_len),
                          comp_phase.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)), dim=1)
        x = self.in_conv(en_x)
        x_acc = Variable(torch.ones_like(x, device=x.device), requires_grad=True)
        for i in range(self.group_num):
            x = self.stcns[i](x)
            x_acc = x_acc + x
        x_real, x_imag = self.real_conv(x_acc).transpose(-2, -1), self.imag_conv(x_acc).transpose(-2, -1)
        return torch.stack((x_real, x_imag), dim=1)


if __name__ == "__main__":
    net = CompNet(win_size=320,
                  win_shift=160,
                  fft_num=320,
                  k1=(2, 3),
                  k2=(2, 3),
                  c=64,
                  embed_dim=64,
                  kd1=5,
                  cd1=64,
                  d_feat=256,
                  hidden_dim=64,
                  hidden_num=2,
                  group_num=2,
                  dilations=(1,2,5,9),
                  inter_connect="cat",
                  intra_connect="cat",
                  norm_type="iLN",
                  rnn_type="LSTM",
                  post_type="vanilla",
                  is_dual_rnn=True,
                  is_causal=True,
                  is_u2=True,
                  is_mu_compress=True,).cuda()
    x = torch.rand([2, 16000-379]).cuda()
    _, y = net(x)
    print(f"{x.shape}->{y.shape}")
    # from ptflops import get_model_complexity_info
    # get_model_complexity_info(net, (16000,))