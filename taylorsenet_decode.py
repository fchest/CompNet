from pyexpat import model
import torch
import argparse
import os
import numpy as np
# from nets.TaylorSENet import TaylorSENet
# from nets.fscrn import TCNN
from nets.compnet import CompNet
from torch.nn import DataParallel
# from nets.TaylorSENet_v1 import TaylorSENetV1
# from nets.TaylorSENet_v2 import TaylorSENetV2
# from nets.TaylorSENet_v3 import TaylorSENetV3
import soundfile as sf

def enhance(args):
    # model = TaylorSENet(cin=2, k1=(1,3), k2=(2,3), c=64, kd1=5, cd1=64, d_feat=256, dilations=[1,2,5,9], p=2, fft_num=320,
    #                     order_num=3, intra_connect='cat',inter_connect='cat',is_causal=True,is_conformer=False,is_u2=True,
    #                     is_param_share=False,is_encoder_share=True).cuda()
    # model = TaylorSENetV1(cin=2, k1=(1,3), k2=(2,3), c=64, kd1=5, cd1=64, d_feat=256, dilations=[1,2,5,9], p=2,
    #                       fft_num=320, order_num=3, intra_connect='cat', inter_connect='add', is_causal=True,
    #                       is_conformer=False, is_u2=True, is_param_share=False, is_encoder_share=False)
    #model = TaylorSENetV2(kd1=5)
    # model = TaylorSENetV3(kd1=5)
    # set gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    # logger_print(f"gpus: {os.environ['CUDA_VISIBLE_DEVICES']}")
    model=CompNet().cuda()
    if len(os.environ["CUDA_VISIBLE_DEVICES"]) > 1:
            # net = DataParallel(net, device_ids=os.environ["CUDA_VISIBLE_DEVICES"])
            model = DataParallel(model).cuda()
    checkpoint = torch.load(args.Model_path)
    model.load_state_dict(checkpoint)
    print(model)
    model.eval()
    model.cuda()

    with torch.no_grad():
        cnt = 0
        mix_file_path = args.mix_file_path
        seen_flag = args.seen_flag
        noise_type = args.noise_type
        esti_file_path = args.esti_file_path
        snr = args.snr
        for _, snr_key in enumerate(snr):
            if seen_flag == 1:
                mix_file_path1 = os.path.join(mix_file_path, noise_type, str(snr_key))
            else:
                mix_file_path1 = os.path.join(mix_file_path, noise_type, str(snr_key))
            file_list = os.listdir(mix_file_path1)
            for file_id in file_list:
                file_name = file_id.split('.')[0]
                feat_wav, _ = sf.read(os.path.join(mix_file_path1, file_id))
                c = np.sqrt(len(feat_wav) / np.sum((feat_wav ** 2.0)))
                feat_wav = feat_wav * c
                wav_len = len(feat_wav)
                # feat_wav=feat_wav.cuda()
                # feat_wav=torch.from_numpy(feat_wav)
                print("feat_wav.shape",feat_wav.shape)
                # feat_wav=feat_wav.unsqueeze(0)
                feat_wav = torch.FloatTensor(feat_wav)
                feat_wav=feat_wav.unsqueeze(0)
                feat_wav=feat_wav.cuda()


                [wav_esti, esti_mag], post_esti = model(feat_wav)
                post_esti = post_esti.permute(0, 3, 2, 1)  # (B,F,T,2)
                post_mag, post_phase = torch.norm(post_esti, dim=-1)**(1.0/0.5),\
                                                        torch.atan2(post_esti[...,-1], post_esti[...,0])
                post_esti = torch.stack((post_mag*torch.cos(post_phase), post_mag*torch.sin(post_phase)), dim=-1)
                # batch_esti_wav = torch.istft(post_esti,
                #                      n_fft=320,
                #                      hop_length=160,
                #                      win_length=320,
                #                      window=torch.sqrt(torch.hann_window(320).to(post_esti.device)),
                #                      return_complex=False)


                esti_utt = torch.istft(post_esti, 320, 160, 320, window=torch.hann_window(320).cuda(), length=wav_len,return_complex=False).transpose(-2,-1).squeeze()
                esti_utt = esti_utt.cpu().numpy()
                esti_utt = esti_utt[:wav_len]
                esti_utt = esti_utt / c
                if seen_flag == 1:
                    os.makedirs(os.path.join(esti_file_path, noise_type, 'unseen', str(snr_key)), exist_ok=True)
                    sf.write(os.path.join(esti_file_path, noise_type, 'unseen', str(snr_key), file_id), esti_utt, args.fs)
                else:
                    os.makedirs(os.path.join(esti_file_path, noise_type, str(snr_key)), exist_ok=True)
                    sf.write(os.path.join(esti_file_path, noise_type, str(snr_key), file_id), esti_utt, args.fs)
                print(' The %d utterance has been decoded!' % (cnt + 1))
                cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Recovering audio')
    parser.add_argument('--mix_file_path', type=str, default='test_data/unseen_babble')
    parser.add_argument('--esti_file_path', type=str, default='test')
    parser.add_argument('--noise_type', type=str, default='babble')  # babble   cafe  factory1
    parser.add_argument('--seen_flag', type=int, default=1)    # 1   0
    parser.add_argument('--snr', type=int, default=[-6,-3, 0, 3,6,9])     #  -5  -2  0  2  5
    parser.add_argument('--fs', type=int, default=16000,
                        help='The sampling rate of speech')
    parser.add_argument('--Model_path', type=str, default='WSJ0-SI84_100h_compnet_causal_model.pth',
                            # 'inter_cat_causal_True_conformer_False_u2_True_param_share_False_encoder_share_False_lr_'
                            #                               '0.0005_batch_8_epoch_60_model.pth',
                        help='The place to save best model')
    args = parser.parse_args()
    print(args)
    enhance(args=args)
