# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms

# from dataset import TextDataset, prepare_data
# # from configs import tagan_configs


# image_transform = transforms.Compose([
#         transforms.Scale(int(128 * 76 / 64))])

# dataset = TextDataset('../../data/manigan/birds', transform=image_transform)
# dataloader = torch.utils.data.DataLoader(
#         dataset, batch_size=4,
#         drop_last=True, shuffle=True)

# txt_encoder_f = nn.GRUCell(300, 512)
# txt_encoder_b = nn.GRUCell(300, 512)

# txt = torch.rand(size=(10, 4, 300))
# len_txt = torch.randint(low=7, high=10, size=(4,))

# data_iter = iter(dataloader)
# data = data_iter.next()

# imgs, new_imgs, txt, len_txt, class_ids, keys, wrong_caps, \
#                             wrong_caps_len, wrong_cls_id = prepare_data(data)

# print(txt.shape)

# hi_f = torch.zeros(txt.size(1), 512, device=txt.device) # (T, 512)
# hi_b = torch.zeros(txt.size(1), 512, device=txt.device) # (T, 512)
# h_f = []
# h_b = []
# mask = []
# # for batch_size ...?
# for i in range(txt.size(0)):
#     mask_i = (txt.size(0) - 1 - i < len_txt).float().unsqueeze(1) # 
#     mask.append(mask_i)

#     hi_f = txt_encoder_f(txt[i], hi_f)
#     h_f.append(hi_f)
#     hi_b = mask_i * txt_encoder_b(txt[ -i - 1], hi_b) + (1 - mask_i) * hi_b
#     h_b.append(hi_b)
# mask = torch.stack(mask[::-1])
# h_f = torch.stack(h_f) * mask
# h_b = torch.stack(h_b[::-1])
# u = (h_f + h_b) / 2
# m = u.sum(0) / mask.sum(0)
# u, m, mask

# att_txt = (u * m.unsqueeze(0)).sum(-1)
# att_txt_exp = att_txt.exp() * mask.squeeze(-1)
# att_txt = (att_txt_exp / att_txt_exp.sum(0, keepdim=True))


# print(u.shape)
# print(m.shape)
# print(mask.shape)


# print(att_txt.shape)
# print(att_txt_exp.shape)
# print(att_txt.shape)


import torch
import torch.nn as nn
input1 = torch.randn(4, 100, 128)
input2 = torch.randn(4, 100, 128)
cos = nn.CosineSimilarity(dim=2, eps=1e-6)
output = cos(input1, input2)

print(output[0][1])