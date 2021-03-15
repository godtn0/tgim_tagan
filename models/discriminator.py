import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, n_words, init_weights=None):
        super(Discriminator, self).__init__()
        self.n_words = n_words

        self.eps = 1e-7

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.encoder_3 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.GAP_1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.GAP_2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.GAP_3 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # text feature
        self.embed = nn.Embedding(self.n_words, 300)

        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)

        self.gen_filter = nn.ModuleList([
            nn.Linear(512, 256 + 1),
            nn.Linear(512, 512 + 1),
            nn.Linear(512, 512 + 1)
        ])
        self.gen_weight = nn.Sequential(
            nn.Linear(512, 3),
            nn.Softmax(-1)
        )

        self.classifier = nn.Conv2d(512, 1, 4)

        if init_weights is not None:
            self.apply(init_weights)

    def forward(self, img, txt, len_txt, negative=False):
        img_feat_1 = self.encoder_1(img)
        img_feat_2 = self.encoder_2(img_feat_1)
        img_feat_3 = self.encoder_3(img_feat_2)
        img_feats = [self.GAP_1(img_feat_1), self.GAP_2(img_feat_2), self.GAP_3(img_feat_3)]
        D = self.classifier(img_feat_3).squeeze()

        # text attention : calcuate important rate of each word in sentence
        u, m, mask = self._encode_txt(txt, len_txt)
        att_txt = (u * m.unsqueeze(0)).sum(-1)
        att_txt_exp = att_txt.exp() * mask.squeeze(-1)
        att_txt = (att_txt_exp / att_txt_exp.sum(0, keepdim=True))

        # 
        weight = self.gen_weight(u).permute(2, 1, 0)

        sim = 0
        sim_n = 0
        idx = np.arange(0, img.size(0))
        idx_n = torch.tensor(np.roll(idx, 1), dtype=torch.long, device=txt.device)

        for i in range(3):
            img_feat = img_feats[i]
            W_cond = self.gen_filter[i](u).permute(1, 0, 2)
            W_cond, b_cond = W_cond[:, :, :-1], W_cond[:, :, -1].unsqueeze(-1)
            img_feat = img_feat.mean(-1).mean(-1).unsqueeze(-1)

            if negative:
                W_cond_n, b_cond_n, weight_n = W_cond[idx_n], b_cond[idx_n], weight[i][idx_n]
                sim_n += torch.sigmoid(torch.bmm(W_cond_n, img_feat) + b_cond_n).squeeze(-1) * weight_n
            sim += torch.sigmoid(torch.bmm(W_cond, img_feat) + b_cond).squeeze(-1) * weight[i]

        if negative:
            att_txt_n = att_txt[:, idx_n]
            sim_n = torch.clamp(sim_n + self.eps, max=1).t().pow(att_txt_n).prod(0)
        sim = torch.clamp(sim + self.eps, max=1).t().pow(att_txt).prod(0)

        if negative:
            return D, sim, sim_n
        return D, sim

    def _encode_txt(self, txt, len_txt):
        txt = self.embed(txt.transpose(0, 1)).transpose(1, 0)

        # T, B, D
        hi_f = torch.zeros(txt.size(1), 512, device=txt.device) # (B, 512)
        hi_b = torch.zeros(txt.size(1), 512, device=txt.device) # (B, 512)
        h_f = []
        h_b = []
        mask = []
        for i in range(txt.size(0)):
            mask_i = (txt.size(0) - 1 - i < len_txt).float().unsqueeze(1) # 
            mask.append(mask_i)
            hi_f = self.txt_encoder_f(txt[i], hi_f)
            h_f.append(hi_f)
            hi_b = mask_i * self.txt_encoder_b(txt[-i - 1], hi_b) + (1 - mask_i) * hi_b
            h_b.append(hi_b)
        mask = torch.stack(mask[::-1])
        h_f = torch.stack(h_f) * mask
        h_b = torch.stack(h_b[::-1])
        u = (h_f + h_b) / 2
        m = u.sum(0) / mask.sum(0)
        return u, m, mask


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, activation='relu', down_sample=True):
        super(ResBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.down_sample = down_sample
        self.short_conv = nn.Conv2d(self.ch_in, self.ch_out, 1)
        self.conv_0 = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, 1)
        self.conv_1 = nn.Conv2d(self.ch_out, self.ch_out, 3, 1, 1)
        if self.down_sample:
            self.down = nn.AvgPool2d(2, 2)
        if activation == 'relu':
            self.act = nn.ReLU()

    def _short_cut(self, x):
        x = self.short_conv(x)
        if self.down_sample:
            x = self.down(x)
        return x
    
    def _residual(self, x):
        x = self.act(x)
        x = self.conv_0(x)
        x = self.act(x)
        x = self.conv_1(x)
        if self.down_sample:
            x = self.down(x)
        return x
    
    def forward(self, x):
        return self._short_cut(x) + self._residual(x)


class DAMSM(nn.Module):
    def __init__(self, n_words, w_dim, embedding_dim=300, ch=64, img_size=128):
        super(DAMSM, self).__init__()
        self.ch = ch
        self.w_dim = w_dim
        self.embedding_dim = embedding_dim
        self.img_size = img_size
        self.n_words = n_words
        self.num_layers = int(np.log2(self.img_size)) - 4
        self._build_model()
        self._build_model()

    def _build_model(self):
        # imgae_encoder for DAMSM
        self.from_rgb = nn.Conv2d(3, self.ch, 1)
        
        ch_in = self.ch

        encoder = []
        for i in range(self.num_layers):
            ch_out = ch_in * 2
            encoder.append(ResBlock(ch_in, ch_out, down_sample=True))
            ch_in = ch_out
        self.img_encoder = nn.Sequential(*encoder)
        self.proj_to_text = nn.Conv2d(ch_in, self.w_dim, 1)

        # text_encoder for DAMSM
        self.embed = nn.Embedding(self.n_words, self.embedding_dim)
        self.txt_encoder_f = nn.GRUCell(self.embedding_dim, self.w_dim)
        self.txt_encoder_b = nn.GRUCell(self.embedding_dim, self.w_dim)


    def _encode_txt(self, txt, len_txt):
        txt = self.embed(txt.transpose(0, 1)).transpose(1, 0)

        hi_f = torch.zeros(txt.size(1), self.w_dim, device=txt.device)
        hi_b = torch.zeros(txt.size(1), self.w_dim, device=txt.device)
        h_f = []
        h_b = []
        mask = []
        for i in range(txt.size(0)):
            mask_i = (txt.size(0) - 1 - i < len_txt).float().unsqueeze(1)
            mask.append(mask_i)
            hi_f = self.txt_encoder_f(txt[i], hi_f)
            h_f.append(hi_f)
            hi_b = mask_i * self.txt_encoder_b(txt[-i - 1], hi_b) + (1 - mask_i) * hi_b
            h_b.append(hi_b)
        mask = torch.stack(mask[::-1])
        h_f = torch.stack(h_f) * mask
        h_b = torch.stack(h_b[::-1])
        u = (h_f + h_b) / 2
        m = u.sum(0) / mask.sum(0)
        u = u.transpose(0, 1)

        return u, m, mask
    
    def forward(self, img, txt, len_txt, rho_1=4.0, rho_2=5.0):
        img = self.from_rgb(img)
        img_features = self.img_encoder(img)

        # B, C, H, H
        proj_w = self.proj_to_text(img_features)
        # B, C, H * H
        proj_w = proj_w.view(proj_w.size(0), proj_w.size(1), proj_w.size(2) * proj_w.size(2))
        # B, T, C
        word_embs, _, _ = self._encode_txt(txt, len_txt)
        
        sims = []
        for i in range(img.size(0)):
            region_feature = proj_w[i].unsqueeze(0)
            region_feature = region_feature.repeat(img.size(0), 1, 1)
            # B, T, H * H
            attn_map = torch.matmul(word_embs, region_feature)
            word_embs_norm = torch.norm(word_embs, p=2, dim=-1, keepdim=True)
            region_feature_norm = torch.norm(region_feature, p=2, dim=-2, keepdim=True)
            norm = torch.matmul(word_embs_norm, region_feature_norm)
            attn_map = attn_map / (norm + 1e-6)
            attn_map = nn.Softmax()(rho_1 * attn_map)


            # B, T, C
            c = torch.matmul(attn_map, region_feature.transpose(1, 2))
            scores_dot = word_embs * c
            scores_dot = torch.sum(scores_dot, dim=-1)
            word_embs_norm = torch.norm(word_embs, p=2)
            c_norm = torch.norm(c, p=2)
            scores = rho_2 * scores_dot / (word_embs_norm * c_norm + 1e-6)
            scores = scores.exp_()
            scores = torch.sum(scores, dim=1)
            sim = torch.log(scores / rho_2)
            
            sim = nn.Softmax()(sim)[i]
            sims.append(sim)
        
        return torch.stack(sims, dim=0)
            