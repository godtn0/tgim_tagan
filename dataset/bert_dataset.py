import os
import pickle

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy.random as random

class BirdBertDataset(data.Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.data_dir = self.cfg.data_root
        self.split = split
        self.imsize = self.cfg.img_size
        self.caps_per_img = self.cfg.captions_per_img
        self.overfit = self.cfg.overfit
        self.transform = self._get_transform()
        
        if self.data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
            
        self.filenames, self.captions = self._load_text_data()
        self.class_dict, self.class_id = self.load_class_id(os.path.join(self.data_dir, self.split),self.filenames)
        self.number_example = len(self.filenames)
        
        if self.overfit:
            self.filenames = self.filenames[:100]
        
    def load_bbox(self):
        bbox_path = os.path.join(self.data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,delim_whitespace=True,header=None).astype(int)
        filepath = os.path.join(self.data_dir, 'images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox
        
    def _load_text_data(self):
        train_names = self.load_filenames(self.data_dir, 'train')
        test_names = self.load_filenames(self.data_dir, 'test')
        
        train_captions = self.load_captions(self.data_dir, train_names)
        test_captions = self.load_captions(self.data_dir, test_names)
        
        if self.split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
            
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions
        
    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames
        
    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in tqdm(range(len(filenames))):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    all_captions.append(cap)
                    cnt += 1
                    if cnt == self.caps_per_img:
                        break
                if cnt < self.caps_per_img:
                    print('ERROR: the captions for %s less than %d' % (filenames[i], cnt))
        return all_captions
        
    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        class_set = list(set(class_id))
        class_dict = {}
        min_idx = 0
        for target in class_set:
            max_idx = max(idx for idx, val in enumerate(class_id) if val == target)
            class_dict[target]= [min_idx, max_idx]
            min_idx = max_idx+1
            
        return class_dict, class_id
        
    def _get_transform(self):
        transform = transforms.Compose([
            transforms.Resize(int(self.imsize * 76 / 64)),
            transforms.CenterCrop(self.imsize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        return transform
        
    def get_imgs(self, img_path, imsize, bbox=None, transform=None):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - r)
            y2 = np.minimum(height, center_y + r)
            x1 = np.maximum(0, center_x - r)
            x2 = np.minimum(width, center_x + r)
            img = img.crop([x1, y1, x2, y2])
        
        if transform is not None:
            img = transform(img)
        
        # ret = []
        # ret.append(img)
        return img
        
    def __getitem__(self, index):
        src_img, intra_img, inter_img, wrong_img, src_cap, intra_cap, inter_cap, wrong_cap = self._get_data_pair(index)
        return src_img, intra_img, inter_img, wrong_img, src_cap, intra_cap, inter_cap, wrong_cap
        
    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = self.captions[sent_ix]
        return sent_caption
        
    def _get_data_pair(self, index):
        src_idx = index
        src_key = self.filenames[src_idx]
        src_cls_id = self.class_id[src_idx]
        sent_ix = random.randint(0, self.caps_per_img)
        src_sent_ix = src_idx * self.caps_per_img + sent_ix
        
        if self.cfg.same_image: # intra == same image, inter == same class
            # Intra
            intra_idx = src_idx
            intra_cls_id = src_cls_id
            intra_key = self.filenames[intra_idx]
            intra_cls_id = self.class_id[intra_idx]
            
            intra_ix = sent_ix
            while intra_ix == sent_ix:
                intra_ix = random.randint(0, self.caps_per_img)
            intra_sent_ix = intra_idx * self.caps_per_img + intra_ix
            
            # Inter
            inter_idx = src_idx
            inter_cls_id = src_cls_id
            while inter_idx == src_idx:
                inter_idx = random.randint(self.class_dict[inter_cls_id][0], self.class_dict[inter_cls_id][1]+1)
            inter_key = self.filenames[inter_idx]
            inter_cls_id = self.class_id[inter_idx]
            
            inter_ix = random.randint(0, self.caps_per_img)
            inter_sent_ix = inter_idx * self.caps_per_img + inter_ix
            
            # Wrong
            wrong_idx = src_idx
            wrong_cls_id = src_cls_id
            while self.class_id[wrong_idx] == src_cls_id:
                wrong_idx = random.randint(0, len(self.filenames))
            wrong_key = self.filenames[wrong_idx]
            wrong_cls_id = self.class_id[wrong_idx]
            
            wrong_ix = random.randint(0, self.caps_per_img)
            wrong_sent_ix = wrong_idx * self.caps_per_img + wrong_ix
            
        else:
            # Intra
            intra_idx = src_idx
            intra_cls_id = src_cls_id
            while intra_idx == src_idx:
                intra_idx = random.randint(self.class_dict[intra_cls_id][0], self.class_dict[intra_cls_id][1]+1)
                
            intra_key = self.filenames[intra_idx]
            intra_cls_id = self.class_id[intra_idx]
            
            intra_ix = random.randint(0, self.caps_per_img)
            intra_sent_ix = intra_idx * self.caps_per_img + intra_ix
            
            # Inter
            inter_idx = src_idx
            inter_cls_id = src_cls_id
            while self.class_id[inter_idx] == src_cls_id:
                inter_idx = random.randint(0, len(self.filenames))
            inter_key = self.filenames[inter_idx]
            inter_cls_id = self.class_id[inter_idx]
            inter_ix = random.randint(0, self.caps_per_img)
            inter_sent_ix = inter_idx * self.caps_per_img + inter_ix
            
            # Wrong
            wrong_idx = src_idx
            wrong_cls_id = src_cls_id
            while self.class_id[wrong_idx] == src_cls_id or self.class_id[wrong_idx] == inter_cls_id:
                wrong_idx = random.randint(0, len(self.filenames))
            wrong_key = self.filenames[wrong_idx]
            wrong_cls_id = self.class_id[wrong_idx]
            
            wrong_ix = random.randint(0, self.caps_per_img)
            wrong_sent_ix = wrong_idx * self.caps_per_img + wrong_ix
            
        
        
        if self.bbox is not None:
            src_bbox = self.bbox[src_key]
            intra_bbox = self.bbox[intra_key]
            inter_bbox = self.bbox[inter_key]
            wrong_bbox = self.bbox[wrong_key]
        
        src_img_name = '%s/images/%s.jpg' % (self.data_dir, src_key)
        src_img = self.get_imgs(src_img_name, self.imsize, src_bbox, self.transform)
        
        if self.cfg.same_image:
            intra_img = src_img
        else:
            intra_img_name = '%s/images/%s.jpg' % (self.data_dir, intra_key)
            intra_img = self.get_imgs(intra_img_name, self.imsize, intra_bbox, self.transform)
        
        inter_img_name = '%s/images/%s.jpg' % (self.data_dir, inter_key)
        inter_img = self.get_imgs(inter_img_name, self.imsize, inter_bbox, self.transform)
        
        wrong_img_name = '%s/images/%s.jpg' % (self.data_dir, wrong_key)
        wrong_img = self.get_imgs(wrong_img_name, self.imsize, wrong_bbox, self.transform)
        
        src_cap = self.get_caption(src_sent_ix)
        intra_cap = self.get_caption(intra_sent_ix)
        inter_cap = self.get_caption(inter_sent_ix)
        wrong_cap = self.get_caption(wrong_sent_ix)
        
        return src_img, intra_img, inter_img, wrong_img, src_cap, intra_cap, inter_cap, wrong_cap
        
    def __len__(self):
        return len(self.filenames)
        