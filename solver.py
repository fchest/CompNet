import numpy as np
import os
import torch
import torch.nn as nn
import time
import importlib
from utils.util import logger_print
import hdf5storage
from utils.util import cal_pesq
from utils.util import cal_stoi
from utils.util import cal_sisnr
train_epoch, val_epoch, val_metric_epoch = [], [], []  # for loss, loss and metric score
# from torch.utils.tensorboard import SummaryWriter


class Solver(object):
    def __init__(self,
                 data,
                 net,
                 optimizer,
                 save_name_dict,
                 args,
                 ):
        self.train_dataloader = data["train_loader"]
        self.val_dataloader = data["val_loader"]
        self.net = net
        # optimizer part
        self.optimizer = optimizer
        self.lr = args["optimizer"]["lr"]
        self.gradient_norm = args["optimizer"]["gradient_norm"]
        self.epochs = args["optimizer"]["epochs"]
        self.halve_lr = args["optimizer"]["halve_lr"]
        self.early_stop = args["optimizer"]["early_stop"]
        self.halve_freq = args["optimizer"]["halve_freq"]
        self.early_stop_freq = args["optimizer"]["early_stop_freq"]
        self.print_freq = args["optimizer"]["print_freq"]
        self.metric_options = args["optimizer"]["metric_options"]
        # loss part
        self.loss_path = args["loss_function"]["path"]
        self.l_type = args["loss_function"]["l_type"]
        self.mag_loss = args["loss_function"]["mag"]["classname"]
        self.gamma1 = args["loss_function"]["mag"]["gamma1"]
        self.sisnr_loss = args["loss_function"]["sisnr"]["classname"]
        self.gamma2 = args["loss_function"]["sisnr"]["gamma2"]
        self.com_mag_loss = args["loss_function"]["com_mag"]["classname"]
        self.alpha3 = args["loss_function"]["com_mag"]["alpha3"]
        self.gamma3 = args["loss_function"]["com_mag"]["gamma3"]

        # signal part
        self.sr = args["signal"]["sr"]
        self.win_size = args["signal"]["win_size"]
        self.win_shift = args["signal"]["win_shift"]
        self.fft_num = args["signal"]["fft_num"]
        self.is_mu_compress = args["signal"]["is_mu_compress"]
        # path part
        self.is_checkpoint = args["path"]["is_checkpoint"]
        self.is_resume_reload = args["path"]["is_resume_reload"]
        self.checkpoint_load_path = args["path"]["checkpoint_load_path"]
        self.checkpoint_load_filename = args["path"]["checkpoint_load_filename"]
        self.loss_save_path = args["path"]["loss_save_path"]
        self.model_best_path = args["path"]["model_best_path"]
        # sava name
        self.loss_save_filename = save_name_dict["loss_filename"]
        self.best_model_save_filename = save_name_dict["best_model_filename"]
        self.checkpoint_save_filename = save_name_dict["checkpoint_filename"]

        self.train_loss = torch.Tensor(self.epochs)
        self.val_loss = torch.Tensor(self.epochs)
        # set loss funcs
        loss_module = importlib.import_module(self.loss_path)
        self.mag_loss = getattr(loss_module, self.mag_loss)(self.l_type)
        self.sisnr_loss = getattr(loss_module, self.sisnr_loss)()
        self.com_mag_loss = getattr(loss_module, self.com_mag_loss)(self.alpha3, self.l_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self._reset()

        # summarywriter
        # self.tensorboard_path = "./" + args["path"]["logging_path"] + "/" + args["save"]["tensorboard_filename"]
        # if not os.path.exists(self.tensorboard_path):
        #     os.makedirs(self.tensorboard_path)
        # self.writer = SummaryWriter(self.tensorboard_path, max_queue=5, flush_secs=30)

    def _reset(self):
        # Reset
        if self.is_resume_reload:
            checkpoint = torch.load(os.path.join(self.checkpoint_load_path, self.checkpoint_load_filename))
            self.net.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["start_epoch"]
            self.prev_val_loss = checkpoint["val_loss"]  # val loss
            self.prev_val_metric = checkpoint["val_metric"]
            self.best_val_metric = checkpoint["best_val_metric"]
            self.val_no_impv = checkpoint["val_no_impv"]
            self.halving = checkpoint["halving"]
        else:
            self.start_epoch = 0
            self.prev_val_loss = float("inf")
            self.prev_val_metric = -float("inf")
            self.best_val_metric = -float("inf")
            self.val_no_impv = 0
            self.halving = False

    def train(self):
        logger_print("Begin to train....")
        self.net.to(self.device)
        for epoch in range(self.start_epoch, self.epochs):
            begin_time = time.time()
            # training phase
            logger_print("-" * 90)
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger_print(f"Epoch id:{int(epoch + 1)}, Training phase, Start time:{start_time}")
            self.net.train()
            train_avg_loss = self._run_one_epoch(epoch, val_opt=False)
            # self.writer.add_scalar(f"Loss/Training_Loss", train_avg_loss, epoch)
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger_print(f"Epoch if:{int(epoch + 1)}, Training phase, End time:{end_time}, "
                         f"Training loss:{train_avg_loss}")

            # Cross val
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger_print(f"Epoch id:{int(epoch + 1)}, Validation phase, Start time:{start_time}")
            self.net.eval()    # norm and dropout is off
            val_avg_loss, val_avg_metric = self._run_one_epoch(epoch, val_opt=True)
            # self.writer.add_scalar(f"Loss/Validation_Loss", val_avg_loss, epoch)
            # self.writer.add_scalar(f"Loss/Validation_Metric", val_avg_metric, epoch)
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            logger_print(f"Epoch if:{int(epoch + 1)}, Validation phase, End time:{end_time}, "
                         f"Validation loss:{val_avg_loss}, Validation metric score:{val_avg_metric}")
            end_time = time.time()
            print(f"{end_time-begin_time}s in {epoch+1}th epoch")
            logger_print("-" * 90)

            # whether to save checkpoint at current epoch
            if self.is_checkpoint:
                cpk_dic = {}
                cpk_dic["model_state_dict"] = self.net.state_dict()
                cpk_dic["optimizer_state_dict"] = self.optimizer.state_dict()
                cpk_dic["train_loss"] = train_avg_loss
                cpk_dic["val_loss"] = val_avg_loss
                cpk_dic["val_metric"] = val_avg_metric
                cpk_dic["best_val_metric"] = self.best_val_metric
                cpk_dic["start_epoch"] = epoch+1
                cpk_dic["val_no_impv"] = self.val_no_impv
                cpk_dic["halving"] = self.halving
                torch.save(cpk_dic, os.path.join(self.checkpoint_load_path, "Epoch_{}_{}_{}".format(epoch+1,
                                                 self.net.__class__.__name__, self.checkpoint_save_filename)))
            # record loss
            # self.train_loss[epoch] = train_avg_loss
            # self.val_loss[epoch] = val_avg_loss

            train_epoch.append(train_avg_loss)
            val_epoch.append(val_avg_loss)
            val_metric_epoch.append(val_avg_metric)

            # save loss
            loss = {}
            loss["train_loss"] = train_epoch
            loss["val_loss"] = val_epoch
            loss["val_metric"] = val_metric_epoch

            if not self.is_resume_reload:
                hdf5storage.savemat(os.path.join(self.loss_save_path, self.loss_save_filename), loss)
            else:
                hdf5storage.savemat(os.path.join(self.loss_save_path, "resume_cpk_{}".format(self.loss_save_filename)),
                                    loss)
            # lr halve and Early stop
            if self.halve_lr:
                if val_avg_metric <= self.prev_val_metric:
                    self.val_no_impv += 1
                    if self.val_no_impv == self.halve_freq:
                        self.halving = True
                    if (self.val_no_impv >= self.early_stop_freq) and self.early_stop:
                        logger_print("No improvements and apply early-stopping")
                        break
                else:
                    self.val_no_impv = 0

            if self.halving:
                optim_state = self.optimizer.state_dict()
                optim_state["param_groups"][0]["lr"] = optim_state["param_groups"][0]["lr"] / 2.0
                self.optimizer.load_state_dict(optim_state)
                logger_print("Learning rate is adjusted to %5f" % (optim_state["param_groups"][0]["lr"]))
                self.halving = False
            self.prev_val_metric = val_avg_metric

            if val_avg_metric > self.best_val_metric:
                self.best_val_metric = val_avg_metric
                torch.save(self.net.state_dict(), os.path.join(self.model_best_path, self.best_model_save_filename))
                logger_print(f"Find better model, saving to {self.best_model_save_filename}")
            else:
                logger_print("Did not find better model")

    @torch.no_grad()
    def _val_batch(self, batch_info):
        batch_mix_wav = batch_info.feats.to(self.device)  # (B,L)
        batch_target_wav = batch_info.labels.to(self.device)  # (B,L)
        batch_wav_len_list = batch_info.frame_mask_list

        # feature-target extraction
        win_size, win_shift = int(self.sr * self.win_size), int(self.sr * self.win_shift)
        # extract frame list
        update_batch_frame_list, update_batch_wav_len_list = [], []
        for i in range(len(batch_wav_len_list)):
            curr_frame_num = (batch_wav_len_list[i] - win_size) // win_shift + 1
            curr_wav_len = (curr_frame_num - 1) * win_shift + win_size
            update_batch_wav_len_list.append(curr_wav_len)
            update_frame_num = (curr_wav_len - win_size + win_size) // win_shift + 1
            update_batch_frame_list.append(update_frame_num)

        batch_target_stft = torch.stft(batch_target_wav,
                                       n_fft=self.fft_num,
                                       hop_length=win_shift,
                                       win_length=win_size,
                                       window=torch.sqrt(torch.hann_window(win_size).to(batch_target_wav.device)),
                                       return_complex=False)
        p = 0.5 if self.is_mu_compress else 1.0
        target_mag, target_phase = torch.norm(batch_target_stft, dim=-1) ** p, \
                                   torch.atan2(batch_target_stft[..., -1], batch_target_stft[..., 0])
        batch_target_stft = torch.stack((target_mag*torch.cos(target_phase), target_mag*torch.sin(target_phase)),
                                        dim=-1)
        target_mag, batch_target_stft = target_mag.permute(0, 2, 1), batch_target_stft.permute(0, 3, 2, 1)

        [wav_esti, esti_mag], post_esti = self.net(batch_mix_wav)
        # com-mag loss
        min_frame = min(esti_mag.shape[-2], target_mag.shape[-2])
        min_len = min(wav_esti.shape[-1], batch_target_wav.shape[-1])
        # com-mag loss
        batch_com_mag_loss = self.com_mag_loss(post_esti[..., :min_frame, :], batch_target_stft[..., :min_frame, :],
                                               update_batch_frame_list)
        # cal metric loss, (B,F,T,2)
        post_esti = post_esti.permute(0, 3, 2, 1)  # (B,F,T,2)
        post_mag, post_phase = torch.norm(post_esti, dim=-1)**(1.0/p),\
                                               torch.atan2(post_esti[...,-1], post_esti[...,0])
        post_esti = torch.stack((post_mag*torch.cos(post_phase), post_mag*torch.sin(post_phase)), dim=-1)
        batch_esti_wav = torch.istft(post_esti,
                                     n_fft=self.fft_num,
                                     hop_length=win_shift,
                                     win_length=win_size,
                                     window=torch.sqrt(torch.hann_window(win_size).to(post_esti.device)),
                                     return_complex=False)

        loss_dict = {}
        loss_dict["com_mag_loss"] = batch_com_mag_loss.item()
        # create mask
        mask_list = []
        for id in range(len(update_batch_wav_len_list)):
            mask_list.append(torch.ones((update_batch_wav_len_list[id])))
        wav_mask = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True).to(batch_mix_wav.device)  # (B,L)
        batch_mix_wav, batch_target_wav, batch_est_wav = (batch_mix_wav[:, :min_len]*wav_mask).cpu().numpy(), \
                                                         (batch_target_wav[:, :min_len]*wav_mask).cpu().numpy(), \
                                                         (batch_esti_wav*wav_mask).cpu().numpy()
        if "SISNR" in self.metric_options:
            unpro_score_list, pro_score_list = [], []
            for id in range(batch_mix_wav.shape[0]):
                unpro_score_list.append(cal_sisnr(id, batch_mix_wav, batch_target_wav, self.sr))
                pro_score_list.append(cal_sisnr(id, batch_est_wav, batch_target_wav, self.sr))
            unpro_score_list, pro_score_list = np.asarray(unpro_score_list), np.asarray(pro_score_list)
            unpro_sisnr_mean_score, pro_sisnr_mean_score = np.mean(unpro_score_list), np.mean(pro_score_list)
            loss_dict["unpro_metric"] = unpro_sisnr_mean_score
            loss_dict["pro_metric"] = pro_sisnr_mean_score
        if "NB-PESQ" in self.metric_options:
            unpro_score_list, pro_score_list = [], []
            for id in range(batch_mix_wav.shape[0]):
                unpro_score_list.append(cal_pesq(id, batch_mix_wav, batch_target_wav, self.sr))
                pro_score_list.append(cal_pesq(id, batch_est_wav, batch_target_wav, self.sr))
            unpro_score_list, pro_score_list = np.asarray(unpro_score_list), \
                                               np.asarray(pro_score_list)
            unpro_pesq_mean_score, pro_pesq_mean_score = np.mean(unpro_score_list), np.mean(pro_score_list)
            loss_dict["unpro_metric"] = unpro_pesq_mean_score
            loss_dict["pro_metric"] = pro_pesq_mean_score
        if "ESTOI" in self.metric_options:
            unpro_score_list, pro_score_list = [], []
            for id in range(batch_mix_wav.shape[0]):
                unpro_score_list.append(cal_stoi(id, batch_mix_wav, batch_target_wav, self.sr))
                pro_score_list.append(cal_stoi(id, batch_mix_wav, batch_target_wav, self.sr))
            unpro_score_list, pro_score_list = np.asarray(unpro_score_list), \
                                               np.asarray(pro_score_list)
            unpro_estoi_mean_score, pro_estoi_mean_score = np.mean(unpro_score_list), np.mean(pro_score_list)
            loss_dict["unpro_metric"] = unpro_estoi_mean_score
            loss_dict["pro_metric"] = pro_estoi_mean_score
        return loss_dict


    def _train_batch(self, batch_info):
        batch_mix_wav = batch_info.feats.to(self.device)  # (B,L)
        batch_target_wav = batch_info.labels.to(self.device)  # (B,L)
        batch_wav_len_list = batch_info.frame_mask_list

        # feature-target extraction
        win_size, win_shift = int(self.sr * self.win_size), int(self.sr * self.win_shift)
        # extract frame list
        update_batch_frame_list, update_batch_wav_len_list = [], []
        for i in range(len(batch_wav_len_list)):
            curr_frame_num = (batch_wav_len_list[i] - win_size) // win_shift + 1
            curr_wav_len = (curr_frame_num - 1) * win_shift + win_size     
            update_batch_wav_len_list.append(curr_wav_len)
            update_frame_num = (curr_wav_len - win_size + win_size) // win_shift + 1
            update_batch_frame_list.append(update_frame_num)

        batch_target_stft = torch.stft(batch_target_wav,
                                       n_fft=self.fft_num,
                                       hop_length=win_shift,
                                       win_length=win_size,
                                       window=torch.sqrt(torch.hann_window(win_size).to(batch_target_wav.device)),
                                       return_complex=False)
        p = 0.5 if self.is_mu_compress else 1.0
        target_mag, target_phase = torch.norm(batch_target_stft, dim=-1) ** p, \
                                   torch.atan2(batch_target_stft[..., -1], batch_target_stft[..., 0])
        batch_target_stft = torch.stack((target_mag*torch.cos(target_phase), target_mag*torch.sin(target_phase)), dim=-1)
        target_mag, batch_target_stft = target_mag.permute(0,2,1), batch_target_stft.permute(0,3,2,1)

        with torch.enable_grad():
            [wav_esti, esti_mag], post_esti = self.net(batch_mix_wav)

        # cal loss
        min_frame = min(esti_mag.shape[-2], target_mag.shape[-2])
        min_len = min(wav_esti.shape[-1], batch_target_wav.shape[-1])
        # mag-euclidean loss
        batch_mag_loss = self.mag_loss(esti_mag[:,:min_frame,:], target_mag[:,:min_frame,:], update_batch_frame_list)
        # sisnr-loss
        batch_sisnr_loss = self.sisnr_loss(wav_esti[:, :min_len], batch_target_wav[:, :min_len], update_batch_wav_len_list)
        # com-mag loss
        batch_com_mag_loss = self.com_mag_loss(post_esti[...,:min_frame,:], batch_target_stft[...,:min_frame,:], update_batch_frame_list)
        batch_loss = self.gamma1*batch_mag_loss + self.gamma2*batch_sisnr_loss + self.gamma3*batch_com_mag_loss
        # params update
        self.update_params(batch_loss)
        loss_dict = {}
        loss_dict["mag_loss"] = batch_mag_loss.item()
        loss_dict["sisnr_loss"] = batch_sisnr_loss.item()
        loss_dict["com_mag_loss"] = batch_com_mag_loss.item()
        return loss_dict

    def _run_one_epoch(self, epoch, val_opt=False):
        # training phase
        if not val_opt:
            data_loader = self.train_dataloader
            total_mag_loss, total_sisnr_loss, total_com_mag_loss = 0., 0., 0.
            start_time = time.time()
            for batch_id, batch_info in enumerate(data_loader.get_data_loader()):
                loss_dict = self._train_batch(batch_info)
                total_mag_loss += loss_dict["mag_loss"]
                total_sisnr_loss += loss_dict["sisnr_loss"]       
                total_com_mag_loss += loss_dict["com_mag_loss"]
                if batch_id % self.print_freq == 0:
                    logger_print(
                        "Epoch:{:d}, Iter:{:d}, Instaneous RI+Mag loss:{:.4f}, "
                        "Instaneous SISNR loss:{:.4f}, Instaneous ComMag loss:{:.4f}, "
                        "Average Mag loss:{:.4f}, Average SISNR loss:{:.4f}, Average ComMag loss:{:.4f}, Time: {:d}ms/batch".
                        format(epoch+1, int(batch_id), loss_dict["mag_loss"], loss_dict["sisnr_loss"],
                               loss_dict["com_mag_loss"],
                               total_mag_loss/(batch_id+1),
                               total_sisnr_loss/(batch_id+1),
                               total_com_mag_loss/(batch_id+1),
                        int(1000*(time.time()-start_time)/(batch_id+1))))
            return total_com_mag_loss / (batch_id+1)
        else:  # validation phase
            data_loder = self.val_dataloader
            total_sp_loss, total_pro_metric_loss, total_unpro_metric_loss = 0., 0., 0.
            start_time = time.time()
            for batch_id, batch_info in enumerate(data_loder.get_data_loader()):
                loss_dict = self._val_batch(batch_info)
                total_sp_loss += loss_dict["com_mag_loss"]
                total_unpro_metric_loss += loss_dict["unpro_metric"]
                total_pro_metric_loss += loss_dict["pro_metric"]
                if batch_id % self.print_freq == 0:
                    logger_print(
                        "Epoch:{:d}, Iter:{:d}, Average spectral loss:{:.4f}, Average unpro metric score:{:.4f}, "
                        "Average pro metric score:{:.4f}, Time: {:d}ms/batch".
                            format(epoch+1, int(batch_id), total_sp_loss/(batch_id+1), total_unpro_metric_loss/(batch_id+1),
                                   total_pro_metric_loss/(batch_id+1), int(1000*(time.time()-start_time)/(batch_id+1))))
            return total_sp_loss / (batch_id+1), total_pro_metric_loss / (batch_id+1)

    def update_params(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_norm >= 0.0:
            nn.utils.clip_grad_norm_(self.net.parameters(), self.gradient_norm)
        has_nan_inf = 0
        for params in self.net.parameters():
            if params.requires_grad:
                has_nan_inf += torch.sum(torch.isnan(params.grad))
                has_nan_inf += torch.sum(torch.isinf(params.grad))
        if has_nan_inf == 0:
            self.optimizer.step()