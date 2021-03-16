import torch
import torch.nn as nn


class LSTM_Text_Encoder(nn.Module):
    def __init__(self):
        super(LSTM_Text_Encoder, self).__init__()
        self.embed = nn.Embedding(self.n_words, 300)
        self.txt_encoder_f = nn.GRUCell(300, 512)
        self.txt_encoder_b = nn.GRUCell(300, 512)


    def forward(self, txt, txt_len):
        txt = self.embed(txt.transpose(0, 1)).transpose(1, 0)

        hi_f = torch.zeros(txt.size(1), 512, device=txt.device)
        hi_b = torch.zeros(txt.size(1), 512, device=txt.device)
        h_f = []
        h_b = []
        mask = []
        for i in range(txt.size(0)):
            mask_i = (txt.size(0) - 1 - i < txt_len).float().unsqueeze(1)
            mask.append(mask_i)
            hi_f = self.txt_encoder_f(txt[i], hi_f)
            h_f.append(hi_f)
            hi_b = mask_i * self.txt_encoder_b(txt[-i - 1], hi_b) + (1 - mask_i) * hi_b
            h_b.append(hi_b)
        mask = torch.stack(mask[::-1])
        h_f = torch.stack(h_f) * mask
        h_b = torch.stack(h_b[::-1])
        h = (h_f + h_b) / 2
        return h
