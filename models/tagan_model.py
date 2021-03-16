import torch
import torch.nn as nn

from models.generator import Generator
from models.discriminator import Discriminator, DAMSM
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F


def label_like(label, x):
    assert label == 0 or label == 1
    v = torch.zeros_like(x) if label == 0 else torch.ones_like(x)
    v = v.to(x.device)
    return v

def zeros_like(x):
    return label_like(0, x)

def ones_like(x):
    return label_like(1, x)


class TAGAN_Model(nn.Module):
    def __init__(self, cfg, vocab_size):
        super(TAGAN_Model, self).__init__()
        self.cfg = cfg
        G = Generator(vocab_size)
        D = Discriminator(vocab_size)
        damsm = DAMSM(vocab_size, cfg.w_dim, cfg.embedding_dim, cfg.ch)
        self.G, self.D, self.damsm = G.to(cfg.device), D.to(cfg.device), damsm.to(cfg.device)
    
    def build_optimizer(self):
        self.g_optimizer = torch.optim.Adam(self.G.parameters(),
                                   lr=self.cfg.learning_rate, betas=(self.cfg.momentum, 0.999))
        self.d_optimizer = torch.optim.Adam(list(self.D.parameters()) + list(self.damsm.parameters()),
                                    lr=self.cfg.learning_rate, betas=(self.cfg.momentum, 0.999))
        g_lr_scheduler = lr_scheduler.StepLR(self.g_optimizer, 100, self.cfg.lr_decay)
        d_lr_scheduler = lr_scheduler.StepLR(self.d_optimizer, 100, self.cfg.lr_decay)

        return g_lr_scheduler, d_lr_scheduler
    
    def train(self):
        self.G.train()
        self.D.train()

    def train_g_step(self, img, txt, len_txt, txt_m, len_txt_m):
        self.G.zero_grad()

        fake, (z_mean, z_log_stddev) = self.G(img, txt_m, len_txt_m)

        kld = torch.mean(-z_log_stddev + 0.5 * (torch.exp(2 * z_log_stddev) + torch.pow(z_mean, 2) - 1)) * 0.5

        fake_logit, fake_c_prob, fake_region_sim = self.D(fake, txt_m, len_txt_m)

        # unconditional gan loss
        fake_loss = F.binary_cross_entropy_with_logits(fake_logit, ones_like(fake_logit))

        # fake image word pair loss (original TAGAN)
        fake_c_loss = F.binary_cross_entropy(fake_c_prob, ones_like(fake_c_prob))
        
        # fake image region word pair loss
        fake_word_region_loss = F.binary_cross_entropy_with_logits(fake_region_sim, zeros_like(fake_region_sim))
        
        # reconstruction loss
        recon, (z_mean, z_log_stddev) = self.G(img, txt, len_txt)

        kld += torch.mean(-z_log_stddev + 0.5 * (torch.exp(2 * z_log_stddev) + torch.pow(z_mean, 2) - 1)) * 0.5

        recon_loss = F.l1_loss(recon, img)

        G_loss = fake_loss + self.cfg.lambda_cond_loss * fake_c_loss + 0.5 * kld + self.cfg.lambda_damsm_loss * fake_word_region_loss + self.cfg.lambda_recon_loss * recon_loss + 0.5 * kld

        g_loss_items = {
            'G_fake_loss' : fake_loss.item(),
            'G_fake_c_loss' : fake_c_loss.item(),
            'G_fake_damsm_loss' : fake_word_region_loss.item(),
            'G_recon_loss' : recon_loss.item(),
            'kld' : kld.item(),
        }

        G_loss.backward()
        self.g_optimizer.step()

        return G_loss, g_loss_items, recon
    
    def train_d_step(self, img, txt, len_txt, txt_m, len_txt_m):
        self.D.zero_grad()

        real_logit, real_c_prob, real_region_sim, real_c_prob_n = self.D(img, txt, len_txt, negative=True)

        # unconditional real loss
        real_loss = F.binary_cross_entropy_with_logits(real_logit, ones_like(real_logit))

        # real image word pair loss (original TAGAN)
        real_c_loss = (F.binary_cross_entropy(real_c_prob, ones_like(real_c_prob)) + \
            F.binary_cross_entropy(real_c_prob_n, zeros_like(real_c_prob_n))) / 2

        # real image region word pair loss
        word_region_loss = F.binary_cross_entropy_with_logits(real_region_sim, ones_like(real_region_sim))

        # synthesized images
        fake, _ = self.G(img, txt_m, len_txt_m)
        fake_logit, _, _ = self.D(fake.detach(), txt_m, len_txt_m)

        # unconditional fake loss
        fake_loss = F.binary_cross_entropy_with_logits(fake_logit, zeros_like(fake_logit))

        D_loss = real_loss + self.cfg.lambda_cond_loss * real_c_loss + fake_loss + self.cfg.lambda_damsm_loss * word_region_loss

        d_loss_items = {
            'D_real_loss' : real_loss.item(),
            'D_real_c_loss' : real_c_loss.item(),
            'D_fake_loss' : fake_loss.item(),
            'D_real_damsm_loss' : word_region_loss.item()
        } 

        D_loss.backward()
        self.d_optimizer.step()

        return D_loss, d_loss_items, fake