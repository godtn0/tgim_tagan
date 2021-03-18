import torch
import torch.nn as nn
from transformers import BertModel


class LSTM_Text_Encoder(nn.Module):
    def __init__(self, cfg):
        super(LSTM_Text_Encoder, self).__init__()
        self.cfg = cfg
        self.n_words = cfg.n_words
        self.w_dim = cfg.w_dim
        self.embedding_dim = cfg.embedding_dim
        self.embed = nn.Embedding(self.n_words, self.embedding_dim)
        self.txt_encoder_f = nn.GRUCell(self.embedding_dim, self.w_dim)
        self.txt_encoder_b = nn.GRUCell(self.embedding_dim, self.w_dim)

    def forward(self, txt, txt_len):
        txt = self.embed(txt.transpose(0, 1)).transpose(1, 0)

        hi_f = torch.zeros(txt.size(1), self.w_dim, device=txt.device)
        hi_b = torch.zeros(txt.size(1), self.w_dim, device=txt.device)
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
        m = h.sum(0) / mask.sum(0)
        return h, m, mask


class Transformer(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.cfg = cfg
    self.bert = BertModel.from_pretrained(self.cfg.bert_pretrain)
    self.embedding_dim = self.bert.config.to_dict()['hidden_size']
    
  def forward(self, text, mask=None, type_id=None):
    # text = [batch size, sent len]
    embedded = self.bert(input_ids=text,attention_mask=mask,token_type_ids=type_id)[0]
    sent = embedded[:, 0, :]
    
    words = embedded[:,1:-1, :]
    return words, sent
