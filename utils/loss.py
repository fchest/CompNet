import torch


class MagEuclideanLoss(object):
    def __init__(self, l_type):
        self.l_type = l_type

    def __call__(self, esti, label, frame_list):
        """
            esti: (B,T,F)
            label: (B,T,F)
            frame_list: list
        """
        b_size, seq_len, freq_num = esti.shape
        mask_for_loss = []
        with torch.no_grad():
            for i in range(b_size):
                tmp_mask = torch.ones((frame_list[i], freq_num), dtype=esti.dtype)
                mask_for_loss.append(tmp_mask)
            mask_for_loss = torch.nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(esti.device)

        if self.l_type == "L1" or self.l_type == "l1":
            loss_mag = (torch.abs(esti - label) * mask_for_loss).sum() / mask_for_loss.sum()
        elif self.l_type == "L2" or self.l_type == "l2":
            loss_mag = (torch.square(esti - label) * mask_for_loss).sum() / mask_for_loss.sum()
        else:
            raise RuntimeError("only L1 and L2 are supported")
        return loss_mag


class ComMagEuclideanLoss(object):
    def __init__(self,
                 alpha,
                 l_type):
        self.alpha = alpha
        self.l_type = l_type

    def __call__(self, est, label, frame_list):
        """
            est: (B,2,T,F)
            label: (B,2,T,F)
            frame_list: list
            alpha: scalar
            l_type: str, L1 or L2
            """
        b_size, _, seq_len, freq_num = est.shape
        mask_for_loss = []
        with torch.no_grad():
            for i in range(b_size):
                tmp_mask = torch.ones((frame_list[i], freq_num, 2), dtype=est.dtype)
                mask_for_loss.append(tmp_mask)
            mask_for_loss = torch.nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(est.device)
            mask_for_loss = mask_for_loss.permute(0,3,1,2)  # (B,2,T,F)
            mag_mask_for_loss = mask_for_loss[:,0,...]
        est_mag, label_mag = torch.norm(est, dim=1), torch.norm(label, dim=1)

        if self.l_type == "L1" or self.l_type == "l1":
            loss_com = (torch.abs(est - label) * mask_for_loss).sum() / mask_for_loss.sum()
            loss_mag = (torch.abs(est_mag - label_mag) * mag_mask_for_loss).sum() / mag_mask_for_loss.sum()
        elif self.l_type == "L2" or self.l_type == "l2":
            loss_com = (torch.square(est - label) * mask_for_loss).sum() / mask_for_loss.sum()
            loss_mag = (torch.square(est_mag - label_mag) * mag_mask_for_loss).sum() / mag_mask_for_loss.sum()
        else:
            raise RuntimeError("only L1 and L2 are supported!")
        return self.alpha*loss_com + (1 - self.alpha)*loss_mag


class StagewiseComMagEuclideanLoss(object):
    def __init__(self,
                 prev_weight,
                 alpha,
                 l_type,
                 ):
        self.prev_weight = prev_weight
        self.alpha = alpha
        self.l_type = l_type

    def __call__(self, est_list, label, frame_list):
        alpha_list = [self.prev_weight for _ in range(len(est_list))]
        mask_for_loss = []
        utt_num = label.size()[0]
        with torch.no_grad():
            for i in range(utt_num):
                tmp_mask = torch.ones((frame_list[i], label.size()[-2]), dtype=label.dtype)
                mask_for_loss.append(tmp_mask)
            mask_for_loss = torch.nn.utils.rnn.pad_sequence(mask_for_loss, batch_first=True).to(label.device)
            mask_for_loss = mask_for_loss.transpose(-2, -1).contiguous()
            com_mask_for_loss = torch.stack((mask_for_loss, mask_for_loss), dim=1)
        loss1, loss2 = 0., 0.
        mag_label = torch.norm(label, dim=1)
        for i in range(len(est_list)):
            curr_esti = est_list[i]
            mag_esti = torch.norm(curr_esti, dim=1)
            if self.l_type == "L1" or self.l_type == "l1":
                loss1 = loss1 + alpha_list[i] * (
                        (torch.abs(curr_esti - label) * com_mask_for_loss).sum() / com_mask_for_loss.sum())
                loss2 = loss2 + alpha_list[i] * (
                        (torch.abs(mag_esti - mag_label) * mask_for_loss).sum() / mask_for_loss.sum())
            elif self.l_type == "L2" or self.l_type == "l2":
                loss1 = loss1 + alpha_list[i] * (
                        (torch.square(curr_esti - label) * com_mask_for_loss).sum() / com_mask_for_loss.sum())
                loss2 = loss2 + alpha_list[i] * (
                        (torch.square(mag_esti - mag_label) * mask_for_loss).sum() / mask_for_loss.sum())
        return self.alpha*loss1 + (1-self.alpha)*loss2


class SISNRLoss(object):
    def __call__(self, est, label, wav_len_list):
        """
        est: (B, L)
        label: (B, L)
        wav_len_list: list
        return: mean negative sisnr loss
        """
        wav_mask_list = []
        with torch.no_grad():
            for i in range(len(wav_len_list)):
                wav_mask_list.append(torch.ones([wav_len_list[i]], dtype=est.dtype))
            wav_mask = torch.nn.utils.rnn.pad_sequence(wav_mask_list, batch_first=True).to(est.device)  # (B, L)
        est, label = est*wav_mask, label*wav_mask
        label_power = torch.pow(label, 2).sum(1, keepdim=True) + 1e-8  # (B,1)
        scale = torch.sum(label*est, 1, keepdim=True) / label_power  # (B,1)
        est_true = scale * label  # (B, L)
        est_res = est - est_true  # (B, L)
        true_power = torch.pow(est_true, 2).sum(1) + 1e-8
        res_power = torch.pow(est_res, 2).sum(1) + 1e-8
        sdr = 10*torch.log10(true_power) - 10*torch.log10(res_power)
        return -torch.mean(sdr)