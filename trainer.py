import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from dataset import BridsDataset, prepare_data
from models.tagan_model import TAGAN_Model


def get_transform():
    image_transform = transforms.Compose([
                transforms.Scale(int(128 * 76 / 64))])
    return image_transform


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._build_dataloader()
        self._build_model()
        self._set_logger()
    
    def _build_dataloader(self):
        if self.cfg.data_name == 'birds':
            print("=" * 50)
            print('BIRDS')
            print("=" * 50)
            
            self.train_dataset = BridsDataset(self.cfg.data_dir, transform=get_transform())

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_threads)

        self.num_batches = len(self.train_dataloader)
        self.total_iter = self.cfg.num_epochs * self.num_batches
    
    def _build_model(self):
        print("=" * 50)
        print("building model... {}".format(self.cfg.model_name))
        print('GPU: {}'.format(torch.cuda.is_available()))
        self.model = TAGAN_Model(self.cfg, self.train_dataset.n_words)
        self.g_lr_scheduler, self.d_lr_scheduler = self.model.build_optimizer()
        print("=" * 50)
    
    def _set_logger(self):
        print("=" * 50)
        print("setting logger and tensorboard ...")
        self._logger = logging.getLogger(__name__)
        start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        date = "%s/%s/%s" % (self.cfg.save_path, self.cfg.model_name, start_time)

        save_checkpoint = "%s/%s" % (date, "checkpoints")
        tensorboard = "%s/%s" % (date, "tensorboard")

        if not os.path.exists(date):
            os.makedirs(date)

        if not os.path.exists(tensorboard):
            os.makedirs(tensorboard)
        if not os.path.exists(save_checkpoint):
            os.makedirs(save_checkpoint)

        self.cfg.save_dirpath = save_checkpoint
        self.summary_writer = SummaryWriter(tensorboard)
        print("=" * 50)
        
    def train(self):
        start_time = datetime.now().strftime('%Y%m%d_%H:%M:%S')
        self._logger.info("start train model at %s" % start_time)

        self.global_iteration_step = 0
        for epoch in range(self.cfg.num_epochs):
            self.d_lr_scheduler.step()
            self.g_lr_scheduler.step()

            loss = {
                'avg_D_real_loss' : 0,
                'avg_D_real_c_loss' : 0,
                'avg_D_fake_loss' : 0,
                'avg_D_real_damsm_loss' : 0,
                'avg_G_fake_loss' : 0,
                'avg_G_fake_c_loss' : 0,
                'avg_G_recon_loss' : 0,
                'avg_G_fake_damsm_loss' : 0,
                'avg_kld' : 0,
            }
            
            tqdm_batch_iterator =tqdm(self.train_dataloader)

            for step, data in enumerate(tqdm_batch_iterator):
                img, txt, len_txt = prepare_data(data)
                # img = img.mul(2).sub(1)
                # BTC to TBC
                txt = txt.transpose(1, 0)
                # negative text
                txt_m = torch.cat((txt[:, -1].unsqueeze(1), txt[:, :-1]), 1)
                len_txt_m = torch.cat((len_txt[-1].unsqueeze(0), len_txt[:-1]))
                img_m = torch.cat((img[-1, ...].unsqueeze(0), img[:-1, ...]), 0)

                # =========================== UPDATE DISCRIMINATOR ===========================
                D_loss, d_loss_items, fake_img = self.model.train_d_step(img, txt, len_txt, txt_m, len_txt_m)
                for key in d_loss_items.keys():
                    logging_key = 'avg_' + key
                    loss[logging_key] += d_loss_items[key]

                # =========================== UPDATE GENERATOR ===========================
                G_loss, g_loss_items, recon_img = self.model.train_g_step(img, txt, len_txt, txt_m, len_txt_m)
                
                for key in g_loss_items.keys():
                    logging_key = 'avg_' + key
                    loss[logging_key] += g_loss_items[key]

                description = 'Epoch [%03d/%03d], Iter [%03d/%03d], D_real: %.4f, D_real_c: %.4f, D_fake: %.4f, D_damsm: %.4f, G_fake: %.4f, G_fake_c: %.4f, G_fake_damsm: %.4f, G_recon: %.4f, KLD: %.4f' \
                        % (epoch + 1, self.cfg.num_epochs, step + 1, self.num_batches, loss['avg_D_real_loss'] / (step + 1),
                            loss['avg_D_real_c_loss'] / (step + 1), loss['avg_D_fake_loss'] / (step + 1), loss['avg_D_real_damsm_loss'] / (step+1),
                            loss['avg_G_fake_loss'] / (step + 1), loss['avg_G_fake_c_loss'] / (step + 1), loss['avg_G_fake_damsm_loss'] / (step+1),
                            loss['avg_G_recon_loss'] / (step + 1), loss['avg_kld'] / (step + 1))
                
                tqdm_batch_iterator.set_description(description)

                # TBC to BTC
                txt = txt.transpose(1, 0)
                txt_m = txt_m.transpose(1, 0)

                if (self.global_iteration_step + 1) % self.cfg.log_interval == 0:
                    self._logger.info(description)
                    self._train_summaries(loss, step)
                    fake_fig = self.visualize_output(img, img_m, fake_img.detach(), txt, txt_m)
                    recon_fig = self.visualize_output(img, img, recon_img.detach(), txt, txt)
                    self.summary_writer.add_figure('fake_gen_image', fake_fig, self.global_iteration_step + 1)
                    self.summary_writer.add_figure('recon_image', recon_fig, self.global_iteration_step + 1)
                
                self.global_iteration_step += 1

    def _train_summaries(self, loss, steps):
        self.summary_writer.add_scalar("dis_adv/adv_real", loss["avg_D_real_loss"] / (steps + 1), self.global_iteration_step)
        self.summary_writer.add_scalar("dis_adv/adv_fake", loss["avg_D_fake_loss"] / (steps + 1) , self.global_iteration_step)
        self.summary_writer.add_scalar("dis_pair/real_image_word_pair", loss["avg_D_real_c_loss"] / (steps + 1) , self.global_iteration_step)
        self.summary_writer.add_scalar("dis_pair/real_damsm", loss["avg_D_real_damsm_loss"] / (steps + 1) , self.global_iteration_step)

        self.summary_writer.add_scalar("gen_adv/adv", loss["avg_G_fake_loss"] / (steps + 1) , self.global_iteration_step)
        self.summary_writer.add_scalar("gen_pair/fake_image_word_pair", loss["avg_G_fake_c_loss"] / (steps + 1) , self.global_iteration_step)
        self.summary_writer.add_scalar("gen_pair/fake_damsm", loss["avg_G_fake_damsm_loss"] / (steps + 1) , self.global_iteration_step)
        self.summary_writer.add_scalar("gen_recon/recon_l1", loss["avg_G_recon_loss"] / (steps + 1) , self.global_iteration_step)
        self.summary_writer.add_scalar("gen_recon/kld", loss["avg_kld"] / (steps + 1) , self.global_iteration_step)
              

    def visualize_output(self, src_img, trg_img, gen_img, src_cap, trg_cap):
        def idx2snt(cap):
            output = []
            for i in range(cap.shape[0]):
                tmp = ""
                for j in range(cap.shape[1]):
                    if cap[i, j] == 0:
                        break
                    tmp += " " + self.train_dataset.ixtoword[cap[i, j]]
                output.append(tmp)
            return output
        
        batch_size = src_img.shape[0]
        src_img = (src_img.cpu().numpy() + 1) / 2
        trg_img = (trg_img.cpu().numpy() + 1) / 2
        gen_img = (gen_img.cpu().numpy() + 1) / 2
        src_cap = src_cap.cpu().numpy()
        src_cap_snt = idx2snt(src_cap)
        trg_cap = trg_cap.cpu().numpy()
        trg_cap_snt = idx2snt(trg_cap)
        
        fig = plt.figure(figsize=(6 * 3, 4 * batch_size))
        # print('='*30)
        
        for idx in range(batch_size):
            ax = fig.add_subplot(batch_size, 3, idx * 3 + 1)
            ax.imshow(np.transpose(src_img[idx], (1, 2, 0)))
            ax.set_title('Source Image')
            ax.text(0.5, -0.1, src_cap_snt[idx], size=12, ha="center",
                    transform=ax.transAxes, color='blue')
            ax.axis('off')
            ax = fig.add_subplot(batch_size, 3, idx * 3 + 2)
            ax.imshow(np.transpose(trg_img[idx], (1, 2, 0)))
            ax.set_title('Target Image')
            ax.text(0.5, -0.1, trg_cap_snt[idx], size=12, ha="center",
                    transform=ax.transAxes, color='red')
            ax.axis('off')
            ax = fig.add_subplot(batch_size, 3, idx * 3 + 3)
            ax.imshow(np.transpose(gen_img[idx], (1, 2, 0)))
            ax.set_title('Generated Image')
            ax.axis('off')
        
        return fig