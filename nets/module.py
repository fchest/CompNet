import torch
import torch.nn as nn
from utils.util import NormSwitch
from torch import Tensor

frame_encoder_list = [159, 79, 39, 19, 9, 4]
frame_decoder_list = [9, 19, 39, 79, 159, 320]
frequency_encoder_list = [79, 39, 19, 9, 4]
frequency_decoder_list = [9, 19, 39, 79, 161]


class InterIntraRNN(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 hidden_num: int,
                 rnn_type: str,
                 is_causal: bool,
                 ):
        super(InterIntraRNN, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.hidden_num = hidden_num
        self.rnn_type = rnn_type
        self.is_causal = is_causal
        #
        self.norm = nn.LayerNorm([embed_dim])
        p = 2 if not is_causal else 1
        self.intra_rnn = getattr(nn, rnn_type)(input_size=embed_dim, hidden_size=hidden_dim//2, num_layers=hidden_num//2,
                                               batch_first=True, bidirectional=True)
        self.inter_rnn = getattr(nn, rnn_type)(input_size=hidden_dim, hidden_size=hidden_dim//p, num_layers=hidden_num//2,
                                               batch_first=True, bidirectional=not is_causal)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inpt):
        """
            inpt: (B, C, T, K)
            return:
                    x: (B, T, K)
        """
        b_size, embed_dim, seq_len, k = inpt.shape
        inpt = self.norm(inpt.permute(0, 2, 3, 1))
        # intra part
        x = inpt.contiguous().view(b_size*seq_len, k, embed_dim)  # (BT, K, C)
        x, _ = self.intra_rnn(x)
        # inter part
        x = x.view(b_size, seq_len, k, -1).transpose(1, 2).contiguous().view(b_size*k, seq_len, -1)
        x, _ = self.inter_rnn(x)
        x = self.ff(x).squeeze(dim=-1).view(b_size, k, seq_len).transpose(1, 2)
        return x


class FrameUNetEncoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 c: int,
                 norm_type: str,
                 ):
        super(FrameUNetEncoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        stride = (1, 2)
        c_final = 64
        global frame_encoder_list
        unet = []
        unet.append(nn.Sequential(
            GateConv2d(cin, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
            NormSwitch(norm_type, "2D", c, frame_encoder_list[0], affine=True),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
            NormSwitch(norm_type, "2D", c, frame_encoder_list[1], affine=True),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
            NormSwitch(norm_type, "2D", c, frame_encoder_list[2], affine=True),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
            NormSwitch(norm_type, "2D", c, frame_encoder_list[3], affine=True),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
            NormSwitch(norm_type, "2D", c, frame_encoder_list[4], affine=True),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c_final, k1, stride, padding=(0, 0, k1[0] - 1, 0)),
            NormSwitch(norm_type, "2D", c_final, frame_encoder_list[-1], affine=True),
            nn.PReLU(c_final)))
        self.unet = nn.ModuleList(unet)


    def forward(self, x: Tensor) -> tuple:
        """
            x: (B, 1, T, F) or (B, T, F)
        """
        if x.ndim == 3:
            x = x.unsqueeze(dim=1)
        en_list = []
        for i in range(len(self.unet)):
            x = self.unet[i](x)
            en_list.append(x)
        return x, en_list


class FrameUNetDecoder(nn.Module):
    def __init__(self,
                 c: int,
                 embed_dim: int,
                 k1: tuple,
                 inter_connect: str,
                 norm_type: str,
                 ):
        super(FrameUNetDecoder, self).__init__()
        self.k1 = k1
        self.c = c
        self.embed_dim = embed_dim
        self.inter_connect = inter_connect
        self.norm_type = norm_type
        c_begin = 64
        stride = (1, 2)
        global frame_decoder_list
        unet = []
        base_num = 2 if inter_connect == "cat" else 1
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c_begin*base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frame_decoder_list[0]),
                nn.PReLU(c)))
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c*base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frame_decoder_list[1]),
                nn.PReLU(c)
            ))
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c*base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frame_decoder_list[2]),
                nn.PReLU(c)
            ))
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c*base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frame_decoder_list[3]),
                nn.PReLU(c)
            ))
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c*base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frame_decoder_list[4]),
                nn.PReLU(c)
            ))
        unet.append(nn.Sequential(
            GateConvTranspose2d(c*base_num, embed_dim, k1, stride),
            nn.ConstantPad2d((1, 0, 0, 0), value=0.),
            ))
        self.unet_list = nn.ModuleList(unet)


    def forward(self, x: Tensor, en_list: list) -> Tensor:
        if self.inter_connect == "cat":
            for i in range(len(self.unet_list)):
                tmp = torch.cat((x, en_list[-(i+1)]), dim=1)
                x = self.unet_list[i](tmp)
        elif self.inter_connect == "add":
            for i in range(len(self.unet_list)):
                tmp = x + en_list[-(i+1)]
                x = self.unet_list[i](tmp)
        return x


class FreqUNetEncoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 c: int,
                 norm_type: str,
                 ):
        super(FreqUNetEncoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        kernel_begin = (k1[0], 5)
        stride = (1, 2)
        c_final = 64
        global frequency_encoder_list
        unet = []
        unet.append(nn.Sequential(
            GateConv2d(cin, c, kernel_begin, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c, frequency_encoder_list[0], affine=True),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c, frequency_encoder_list[1], affine=True),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c, frequency_encoder_list[2], affine=True),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c, frequency_encoder_list[3], affine=True),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c_final, k1, (1, 2), padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c_final, frequency_encoder_list[-1], affine=True),
            nn.PReLU(c_final)))
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor) -> tuple:
        en_list = []
        for i in range(len(self.unet_list)):
            x = self.unet_list[i](x)
            en_list.append(x)
        return x, en_list


class FreqUNetDecoder(nn.Module):
    def __init__(self,
                 c: int,
                 k1: tuple,
                 embed_dim: int,
                 inter_connect: str,
                 norm_type: str,
                 ):
        super(FreqUNetDecoder, self).__init__()
        self.k1 = k1
        self.c = c
        self.embed_dim = embed_dim
        self.inter_connect = inter_connect
        self.norm_type = norm_type
        c_begin = 64
        kernel_end = (k1[0], 5)
        stride = (1, 2)
        global frequency_decoder_list
        unet = []
        base_num = 2 if inter_connect == "add" else 1
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c_begin*base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frequency_decoder_list[0]),
                nn.PReLU(c)))
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c*base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frequency_decoder_list[1]),
                nn.PReLU(c)
            ))
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c*base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frequency_decoder_list[2]),
                nn.PReLU(c)
            ))
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c*base_num, c, k1, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frequency_decoder_list[3]),
                nn.PReLU(c)
            ))
        unet.append(
            nn.Sequential(
                GateConvTranspose2d(c*base_num, embed_dim, kernel_end, stride),
                NormSwitch(norm_type, "2D", channel_num=c, frequency_num=frequency_decoder_list[-1]),
                nn.PReLU(embed_dim),
                nn.Conv2d(embed_dim, embed_dim, (1, 1), (1, 1))
            ))
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        if self.inter_connect == "cat":
            for i in range(len(self.unet_list)):
                tmp = torch.cat((x, en_list[-(i+1)]), dim=1)
                x = self.unet_list[i](tmp)
        elif self.inter_connect == "add":
            for i in range(len(self.unet_list)):
                tmp = x + en_list[-(i+1)]
                x = self.unet_list[i](tmp)
        return x


class FreqU2NetEncoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 k2: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(FreqU2NetEncoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        c_last = 64
        kernel_begin = (k1[0], 5)
        stride = (1, 2)
        global frequency_encoder_list
        meta_unet = []
        meta_unet.append(
            EnUnetModule(cin, c, kernel_begin, k2, stride, intra_connect, norm_type, scale=4, padding=(0,0,k1[0]-1,0),
                         de_flag=False))
        meta_unet.append(
            EnUnetModule(c, c, k1, k2, stride, intra_connect, norm_type, scale=3, padding=(0,0,k1[0]-1,0),
                         de_flag=False))
        meta_unet.append(
            EnUnetModule(c, c, k1, k2, stride, intra_connect, norm_type, scale=2, padding=(0,0,k1[0]-1,0),
                         de_flag=False))
        meta_unet.append(
            EnUnetModule(c, c, k1, k2, stride, intra_connect, norm_type, scale=1, padding=(0,0,k1[0]-1,0),
                         de_flag=False))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConv2d(c, c_last, k1, stride, (0,0,k1[0]-1,0)),
            NormSwitch(norm_type, "2D", c_last, frequency_num=frequency_encoder_list[-1]),
            nn.PReLU(c_last)
        )

    def forward(self, x: Tensor) -> tuple:
        en_list = []
        for i in range(len(self.meta_unet_list)):
            x = self.meta_unet_list[i](x)
            en_list.append(x)
        x = self.last_conv(x)
        en_list.append(x)
        return x, en_list


class FreqU2NetDecoder(nn.Module):
    def __init__(self,
                 c: int,
                 k1: tuple,
                 k2: tuple,
                 embed_dim: int,
                 intra_connect: str,
                 inter_connect: str,
                 norm_type: str,
                 ):
        super(FreqU2NetDecoder, self).__init__()
        self.c = c
        self.k1 = k1
        self.k2 = k2
        self.embed_dim = embed_dim
        self.intra_connect = intra_connect
        self.inter_connect = inter_connect
        self.norm_type = norm_type
        c_begin = 64
        kernel_end = (k1[0], 5)
        stride = (1, 2)
        global frequency_decoder_list
        meta_unet = []
        base_num = 2 if inter_connect == "cat" else 1
        meta_unet.append(
            EnUnetModule(c_begin*base_num, c, k1, k2, stride, intra_connect, norm_type, scale=1, de_flag=True)
        )
        meta_unet.append(
            EnUnetModule(c*base_num, c, k1, k2, stride, intra_connect, norm_type, scale=2, de_flag=True)
        )
        meta_unet.append(
            EnUnetModule(c*base_num, c, k1, k2, stride, intra_connect, norm_type, scale=3, de_flag=True)
        )
        meta_unet.append(
            EnUnetModule(c*base_num, c, k1, k2, stride, intra_connect, norm_type, scale=4, de_flag=True)
        )
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConvTranspose2d(c*base_num, embed_dim, kernel_end, stride),
            NormSwitch(norm_type, "2D", embed_dim, frequency_decoder_list[-1], affine=True),
            nn.PReLU(embed_dim),
            nn.Conv2d(embed_dim, embed_dim, (1, 1), (1, 1)))

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        if self.inter_connect == "add":
            for i in range(len(self.meta_unet_list)):
                tmp = x + en_list[-(i+1)]
                x = self.meta_unet_list[i](tmp)
            x = x + en_list[0]
            x = self.last_conv(x)
        elif self.inter_connect == "cat":
            for i in range(len(self.meta_unet_list)):
                tmp = torch.cat((x, en_list[-(i+1)]), dim=1)
                x = self.meta_unet_list[i](tmp)
            x = torch.cat((x, en_list[0]), dim=1)
            x = self.last_conv(x)
        return x


class EnUnetModule(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k1: tuple,
                 k2: tuple,
                 stride: tuple,
                 intra_connect: str,
                 norm_type: str,
                 scale: int,
                 padding: tuple = (0, 0, 0, 0),
                 chomp: tuple = (0, 0),  # only in the freq-axis
                 de_flag: bool = False):
        super(EnUnetModule, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k1 = k1
        self.k2 = k2
        self.stride = stride
        self.padding = padding
        self.chomp = chomp
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.scale = scale
        self.de_flag = de_flag

        global frequency_encoder_list, frequency_decoder_list

        in_conv_list = []
        if not de_flag:
            in_conv_list.append(GateConv2d(cin, cout, k1, stride, padding))
            in_conv_list.append(NormSwitch(norm_type, "2D", channel_num=cout,
                                           frequency_num=frequency_encoder_list[len(frequency_encoder_list)-scale-1]))
        else:
            in_conv_list.append(GateConvTranspose2d(cin, cout, k1, stride, chomp))
            in_conv_list.append(NormSwitch(norm_type, "2D", channel_num=cout,
                                           frequency_num=frequency_decoder_list[scale-1]))
        in_conv_list.append(nn.PReLU(cout))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for i in range(scale):
            enco_list.append(Conv2dunit(k2, cout, frequency_encoder_list[len(frequency_encoder_list)-scale+i], norm_type))
        for i in range(scale):
            if i == 0:
                deco_list.append(Deconv2dunit(k2, cout, frequency_decoder_list[i], "add", norm_type))
            else:
                deco_list.append(Deconv2dunit(k2, cout, frequency_decoder_list[i], intra_connect, norm_type))

        self.enco = nn.ModuleList(enco_list)
        self.deco = nn.ModuleList(deco_list)
        self.skip_connect = SkipConnect(intra_connect)

    def forward(self, inputs: Tensor) -> Tensor:
        x_resi = self.in_conv(inputs)
        x = x_resi
        x_list = []
        for i in range(len(self.enco)):
            x = self.enco[i](x)
            x_list.append(x)

        for i in range(len(self.deco)):
            if i == 0:
                x = self.deco[i](x)
            else:
                x_con = self.skip_connect(x, x_list[-(i+1)])
                x = self.deco[i](x_con)
        x_resi = x_resi + x
        del x_list
        return x_resi


class Conv2dunit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 freq: int,
                 norm_type: str,
                 pad: tuple = (0, 0, 0, 0)
                 ):
        super(Conv2dunit, self).__init__()
        self.k = k
        self.c = c
        self.freq = freq
        self.norm_type = norm_type
        self.pad = pad

        k_t = k[0]
        stride = (1, 2)
        self.conv = nn.Sequential(
            nn.ConstantPad2d((pad[0], pad[0], k_t - 1, 0), value=0.),
            nn.Conv2d(c, c, k, stride),
            NormSwitch(norm_type, "2D", channel_num=c, frequency_num=freq, affine=True),
            nn.PReLU(c)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class Deconv2dunit(nn.Module):
    def __init__(self,
                 k: tuple,
                 c: int,
                 freq: int,
                 intra_connect: str,
                 norm_type: str,
                 chomp_f: tuple = (0, 0)
                 ):
        super(Deconv2dunit, self).__init__()
        self.k = k
        self.c = c
        self.freq = freq
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.chomp_f = chomp_f

        k_t = k[0]
        stride = (1, 2)
        deconv_list = []
        if self.intra_connect == "add":
            real_c = c
        else:
            real_c = c*2
        deconv_list.append(nn.ConvTranspose2d(real_c, c, k, stride))
        if k_t > 1:
            deconv_list.append(ChompT(k_t - 1))
        deconv_list.append(ChompF(chomp_f[0], chomp_f[1]))
        deconv_list.append(NormSwitch(norm_type, "2D", channel_num=c, frequency_num=freq, affine=True))
        deconv_list.append(nn.PReLU(c))
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        return self.deconv(inputs)


class GateConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 padding: tuple,
                 ):
        super(GateConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Sequential(
            nn.ConstantPad2d(padding, value=0.),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size, stride=stride))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class GateConvTranspose2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 chomp_f: tuple = (0, 0),
                 ):
        super(GateConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.chomp_f = chomp_f

        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                   stride=stride),
                ChompT(k_t-1),
                ChompF(chomp_f[0], chomp_f[-1]))
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                   stride=stride),
                ChompF(chomp_f[0], chomp_f[-1]))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class SkipConnect(nn.Module):
    def __init__(self, connect):
        super(SkipConnect, self).__init__()
        self.connect = connect

    def forward(self, x_main, x_aux):
        if self.connect == "add":
            x = x_main + x_aux
        elif self.connect == "cat":
            x = torch.cat((x_main, x_aux), dim=1)
        return x


class TCMList(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 norm_type: str,
                 dilations: tuple = (1, 2, 5, 9),
                 is_causal: bool = True,
                 ):
        super(TCMList, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.norm_type = norm_type
        self.dilations = dilations
        self.is_causal = is_causal
        tcm_list = []
        for i in range(len(dilations)):
            tcm_list.append(SqueezedTCM(kd1, cd1, dilation=dilations[i], d_feat=d_feat, norm_type=norm_type,
                                        is_causal=is_causal))
        self.tcm_list = nn.ModuleList(tcm_list)

    def forward(self, inputs: Tensor) -> Tensor:
        x = inputs
        for i in range(len(self.dilations)):
            x = self.tcm_list[i](x)
        return x


class SqueezedTCM(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 dilation: int,
                 d_feat: int,
                 norm_type: str,
                 is_causal: bool = True,
                 ):
        super(SqueezedTCM, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.dilation = dilation
        self.d_feat = d_feat
        self.norm_type = norm_type
        self.is_causal = is_causal

        self.in_conv = nn.Conv1d(d_feat, cd1, kernel_size=1, bias=False)
        if is_causal:
            pad = ((kd1-1)*dilation, 0)
        else:
            pad = ((kd1-1)*dilation//2, (kd1-1)*dilation//2)
        self.left_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1, affine=True),
            nn.ConstantPad1d(pad, value=0.),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False)
        )
        self.right_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1, affine=True),
            nn.ConstantPad1d(pad, value=0.),
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1, affine=True),
            nn.Conv1d(cd1, d_feat, kernel_size=1, bias=False)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        resi = inputs
        x = self.in_conv(inputs)
        x = self.left_conv(x) * self.right_conv(x)
        x = self.out_conv(x)
        x = x + resi
        return x


class ChompT(nn.Module):
    def __init__(self,
                 t: int,
                 ):
        super(ChompT, self).__init__()
        self.t = t

    def forward(self, x: Tensor) -> Tensor:
        return x[..., :-self.t, :]


class ChompF(nn.Module):
    def __init__(self,
                 front_f: int = 0,
                 end_f: int = 0,
                 ):
        super(ChompF, self).__init__()
        self.front_f = front_f
        self.end_f = end_f

    def forward(self, x):
        if self.end_f != 0:
            return x[..., self.front_f:-self.end_f]
        else:
            return x[..., self.front_f:]