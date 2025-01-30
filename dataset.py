import numpy as np
import cv2
import os

import torch
from torch.utils.data import Dataset

from torchvision import tv_tensors
from torchvision.transforms import v2

class MAEHeartDataset(Dataset):
    def __init__(self, vid_path, n_frames=500, balance=False, augment=False, seed=None):
        
        self.vid_path = vid_path
        
        self.n_frames = n_frames
        self.balance = balance
        self.n_bins = 25 # hardcoded
        self.augmentations = augment
        
        self.frames_data = {}

        cap = cv2.VideoCapture(vid_path)
        assert cap.isOpened(), f'Cannot open {vid_path}'
        
        if self.balance:
            
            if seed is not None:
                np.random.seed(seed) # for selecting random frames in each group in __getitem__()
        
            # get reference frame (last frame of vid)
            vid_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, vid_len-1)
            _, ref = cap.read()
            ref = ref[:, :, 0]

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.mses = np.zeros(n_frames)
            for i in range(n_frames):
                _, img = cap.read()
                img = img[:, :, 0]

                self.mses[i] = np.mean((img - ref)**2)

            self.mses = (self.mses - self.mses.min()) / (self.mses.max() - self.mses.min()) # normalize to [0, 1]
            
            for i in range(self.n_frames):
                fd_key = int(self.mses[i] * self.n_bins) # frames_data key
                
                if fd_key not in self.frames_data:
                    self.frames_data[fd_key] = []
                self.frames_data[fd_key].append(i)
        
        else:
            
            for i in range(n_frames):
                self.frames_data[i] = [i]
                
        self.frames_data_keys = list(self.frames_data.keys())
        self.frame_triples = self.find_frame_triples()
        
        # augmentations
        self.transforms = v2.Compose([v2.ColorJitter(brightness = 0.3, contrast = 0.3),
                                      v2.RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-0.1, 0.1, -0.1, 0.1)),
                                      v2.RandomPerspective(distortion_scale=0.3, p=0.5),
                                      v2.RandomHorizontalFlip(p=0.5),
                                      v2.RandomVerticalFlip(p=0.5)
                                     ])
        
        # cv2.VideoCapture should be instantitated by each worker
        # otherwise self.cap will be shared across all workers
        self.cap = None
        
        self.img_shape = (3, 256, 512) #(C, H, W)
        self.patch_shape = (32, 32)
        self.p_mask = 0.25
        
    def find_frame_triples(self):
        
        frame_group = [] # the key in self.frames_data of each frame
        
        if self.balance:
            for i in range(self.n_frames):
                fd_key = int(self.mses[i] * self.n_bins) # frames_data key
                frame_group.append(fd_key)
            
        else:
            frame_group = [i for i in range(self.n_frames)]
        
        
        # holds idxs of three frames that make up model input
        # previously used frames (t-4, t, t+4), but sometimes these frames are nearly the same (no helpful dynamics)
        frame_triples = [] 
        
        d = 8
        for f in range(self.n_frames):
            
            nf = f % self.n_frames
            while True:
                if abs(frame_group[nf] - frame_group[f]) >= d:
                    break
                nf = (nf + 1) % self.n_frames

            pf = f % self.n_frames
            while True:
                if abs(frame_group[pf] - frame_group[f]) >= d:
                    break
                pf = (pf - 1) % self.n_frames

            frame_triples.append((pf, f, nf))
        
        return frame_triples
    
    def __len__(self):
        return len(self.frames_data)
        
    def pad_arr(self, arr):
        _, H, W = self.img_shape
        c, h, w = arr.shape
        
        padded_arr = torch.zeros(c, H, W, dtype=arr.dtype)
        
        min_h = min(H, h)
        min_w = min(W, w)
        dh = (H-h)//2
        dw = (W-w)//2
        
        padded_arr[:, max(dh, 0):max(dh, 0) + min_h, max(dw, 0):max(dw, 0) + min_w] = arr[:, max(-dh, 0):max(-dh, 0) + min_h, max(-dw, 0):max(-dw, 0) + min_w]
        
        return padded_arr
        
    def get_frame(self, fn):
        
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.vid_path)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        _, frame = self.cap.read()        
                
        return frame
        
    def __getitem__(self, i):
        
        fd_key = self.frames_data_keys[i]
        
        n_fns = len(self.frames_data[fd_key])
        rand_fn = self.frames_data[fd_key][np.random.randint(n_fns)] # random frame number
        
        # Create IMG
        prev_d_fn, _, next_d_fn = self.frame_triples[rand_fn]

        img = self.get_frame(prev_d_fn)
        img[:, :, 1] = self.get_frame(rand_fn)[:, :, 0]
        img[:, :, 2] = self.get_frame(next_d_fn)[:, :, 0]
                        
        img = torch.from_numpy(img) # augmentations need img to be torch tensor
        img = torch.permute(img, (2, 0, 1)) # (H, W, C) -> (C, H, W)
        img = self.pad_arr(img)
        img = img / 255
        
        # Create MASK        
        _, h, w = self.img_shape
        c = 1 # mask is only applied to one channel
        ph, pw = self.patch_shape
        nh = h // ph
        nw = w // pw
               
        mask = torch.zeros(1, h, w, dtype=torch.bool)
        mask = mask.reshape(c, nh, ph, nw, pw) # (c, h, w) -> (c, nh, ph, nw, pw)
        mask = torch.permute(mask, (0, 1, 3, 2, 4)) # -> (c, nh, nw, ph, pw)
        mask = mask.reshape(c*nh*nw, ph, pw) # -> (c*nh*nw, ph, pw)
        
        n_masked_patches = int(self.p_mask * c*nh*nw)
        rand_mask_idxs = torch.randperm(c*nh*nw)[:n_masked_patches]
        mask[rand_mask_idxs, :, :] = True
        
        mask = mask.reshape(c, nh, nw, ph, pw) # (c*nh*nw, ph, pw) -> (c, nh, nw, ph, pw)
        mask = torch.permute(mask, (0, 1, 3, 2, 4)) # -> (c, nh, ph, nw, pw)
        mask = mask.reshape(c, h, w) # -> (c, h, w)
        
        mask = tv_tensors.Mask(mask)
        
        if self.augmentations:
            img, mask = self.transforms(img, mask)
        
        return img, mask

    def get_by_frame(self, fn): # retrieve model input for a specific frame number
        
        # Create IMG
        prev_fn, _, next_fn = self.frame_triples[fn]
        
        img = self.get_frame(prev_fn)
        img[:, :, 1] = self.get_frame(fn)[:, :, 0]
        img[:, :, 2] = self.get_frame(next_fn)[:, :, 0]
                        
        img = torch.from_numpy(img) # augmentations need img to be torch tensor
        img = torch.permute(img, (2, 0, 1)) # (H, W, C) -> (C, H, W)
        img = self.pad_arr(img)
        img = img / 255
        
        # Create MASK        
        _, h, w = self.img_shape
        c = 1 # mask is only applied to one channel
        ph, pw = self.patch_shape
        nh = h // ph
        nw = w // pw
               
        mask = torch.zeros(1, h, w, dtype=torch.bool)
        mask = mask.reshape(c, nh, ph, nw, pw) # (c, h, w) -> (c, nh, ph, nw, pw)
        mask = torch.permute(mask, (0, 1, 3, 2, 4)) # -> (c, nh, nw, ph, pw)
        mask = mask.reshape(c*nh*nw, ph, pw) # -> (c*nh*nw, ph, pw)
        
        n_masked_patches = int(self.p_mask * c*nh*nw)
        rand_mask_idxs = torch.randperm(c*nh*nw)[:n_masked_patches]
        mask[rand_mask_idxs, :, :] = True
        
        mask = mask.reshape(c, nh, nw, ph, pw) # (c*nh*nw, ph, pw) -> (c, nh, nw, ph, pw)
        mask = torch.permute(mask, (0, 1, 3, 2, 4)) # -> (c, nh, ph, nw, pw)
        mask = mask.reshape(c, h, w) # -> (c, h, w)
        
        mask = tv_tensors.Mask(mask)
        
        if self.augmentations:
            img, mask = self.transforms(img, mask)
        
        return img, mask
    

    
    

class SEGHeartDataset(Dataset):
    def __init__(self, vid_path, n_frames=500, balance=False, augment=False, seed=None):
        
        self.vid_path = vid_path
        
        self.n_frames = n_frames
        self.balance = balance
        self.n_bins = 25 # hardcoded
        self.augmentations = augment
        
        self.imgs_dir = os.path.join(vid_path, 'default')
        self.masks_dir = os.path.join(vid_path, 'defaultannot')
        
        self.frames_data = {}
        
        self.frames = []
        for f in sorted(os.listdir(self.imgs_dir)):
            if f[:6] == 'frame_':
                self.frames.append(os.path.splitext(f)[0])
            
            if len(self.frames) == self.n_frames:
                self.img_ext = os.path.splitext(f)[1] # get file extension for imgs
                break

        # get file extension for masks
        for f in os.listdir(self.masks_dir):
            if self.frames[0] in f:
                self.mask_ext = os.path.splitext(f)[1]
                break
        
        if self.balance:
            
            if seed is not None:
                np.random.seed(seed) # for selecting random frames in each group in __getitem__()
                
            diams = np.zeros(self.n_frames) 
            for i, f in enumerate(self.frames): 
                mask_file = os.path.join(self.masks_dir, f + self.mask_ext)
                mask = cv2.imread(mask_file)[:, :, 0]
                diams[i] = self.calc_diam(mask)

            self.diams = (diams - diams.min()) / (diams.max() - diams.min()) # normalize to [0, 1]
            
            for i in range(self.n_frames):
                fd_key = int(self.diams[i] * self.n_bins) # frames_data key
                
                if fd_key not in self.frames_data:
                    self.frames_data[fd_key] = []
                self.frames_data[fd_key].append(i)
                
        else:
            for i in range(n_frames):
                self.frames_data[i] = [i]
                
        self.frames_data_keys = list(self.frames_data.keys())
        self.frame_triples = self.find_frame_triples()
        
        # augmentations
        self.transforms = v2.Compose([v2.ColorJitter(brightness = 0.3, contrast = 0.3),
                                      v2.RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-0.1, 0.1, -0.1, 0.1)),
                                      v2.RandomPerspective(distortion_scale=0.3, p=0.5),
                                      v2.RandomHorizontalFlip(p=0.5),
                                      v2.RandomVerticalFlip(p=0.5)
                                     ])
        
        self.img_shape = (3, 256, 512) #(C, H, W)
        
    def calc_diam(self, mask):
                
        mask_col_sums = mask.sum(axis=0) # number of mask pixels per col
        mask_cols = np.where(mask_col_sums > 0)[0]
        
        l = mask_cols[0] # first col where mask exists
        r = mask_cols[-1] # last col where mask exists
        
        countBool = mask_col_sums.sum()
        diameter = countBool / (r-l+1)
        return diameter.item()
        
    def find_frame_triples(self):
        
        frame_group = [] # the key in self.frames_data of each frame
        
        if self.balance:
            for i in range(self.n_frames):
                fd_key = int(self.diams[i] * self.n_bins) # frames_data key
                frame_group.append(fd_key)
            
        else:
            frame_group = [i for i in range(self.n_frames)]
        
        
        # holds idxs of three frames that make up model input
        # previously used frames (t-4, t, t+4), but sometimes these frames are nearly the same (no helpful dynamics)
        frame_triples = [] 
        
        d = 8
        for f in range(self.n_frames):
            
            nf = f % self.n_frames
            while True:
                if abs(frame_group[nf] - frame_group[f]) >= d:
                    break
                nf = (nf + 1) % self.n_frames

            pf = f % self.n_frames
            while True:
                if abs(frame_group[pf] - frame_group[f]) >= d:
                    break
                pf = (pf - 1) % self.n_frames

            frame_triples.append((pf, f, nf))
        
        return frame_triples
    
    def __len__(self):
        return len(self.frames_data)
        
    def pad_arr(self, arr):
        _, H, W = self.img_shape
        c, h, w = arr.shape
        
        padded_arr = torch.zeros(c, H, W, dtype=arr.dtype)
        
        min_h = min(H, h)
        min_w = min(W, w)
        dh = (H-h)//2
        dw = (W-w)//2
        
        padded_arr[:, max(dh, 0):max(dh, 0) + min_h, max(dw, 0):max(dw, 0) + min_w] = arr[:, max(-dh, 0):max(-dh, 0) + min_h, max(-dw, 0):max(-dw, 0) + min_w]
        
        return padded_arr
        
    def get_frame(self, fn):
        
        path = os.path.join(self.imgs_dir, self.frames[fn] + self.img_ext)
        
        return cv2.imread(path)
        
    def __getitem__(self, i):
        
        fd_key = self.frames_data_keys[i]
        
        n_fns = len(self.frames_data[fd_key])
        rand_fn = self.frames_data[fd_key][np.random.randint(n_fns)] # random frame number
        
        # Create IMG
        d = 4
        prev_d_fn = max(0, rand_fn - d)
        next_d_fn = min(self.n_frames-1, rand_fn + d)
        # prev_d_fn, _, next_d_fn = self.frame_triples[rand_fn]
        
        img = self.get_frame(prev_d_fn)
        img[:, :, 1] = self.get_frame(rand_fn)[:, :, 0]
        img[:, :, 2] = self.get_frame(next_d_fn)[:, :, 0]
                        
        img = torch.from_numpy(img) # augmentations need img to be torch tensor
        img = torch.permute(img, (2, 0, 1)) # (H, W, C) -> (C, H, W)
        img = self.pad_arr(img)
        img = img / 255
        
        
        # Create MASK        
        path = os.path.join(self.masks_dir, self.frames[rand_fn] + self.mask_ext)
        mask = cv2.imread(path)
        mask = mask[:, :, 0:1]
        mask = (mask != 0).astype(np.float32)
        
        mask = torch.from_numpy(mask)
        mask = torch.permute(mask, (2, 0, 1)) # (H, W, C) -> (C, H, W)
        mask = self.pad_arr(mask)
        mask = tv_tensors.Mask(mask)
        
        if self.augmentations:
            img, mask = self.transforms(img, mask)
        
        return img, mask

    def get_by_frame(self, fn): # retrieve model input for a specific frame number
                
        # Create IMG
        d = 4
        prev_d_fn = max(0, fn - d)
        next_d_fn = min(self.n_frames-1, fn + d)
        # prev_fn, _, next_fn = self.frame_triples[fn]
        
        img = self.get_frame(prev_d_fn)
        img[:, :, 1] = self.get_frame(fn)[:, :, 0]
        img[:, :, 2] = self.get_frame(next_d_fn)[:, :, 0]
                        
        img = torch.from_numpy(img) # augmentations need img to be torch tensor
        img = torch.permute(img, (2, 0, 1)) # (H, W, C) -> (C, H, W)
        img = self.pad_arr(img)
        img = img / 255
        
        
        # Create MASK        
        path = os.path.join(self.masks_dir, self.frames[fn] + self.mask_ext)
        mask = cv2.imread(path)
        mask = mask[:, :, 0:1]
        mask = (mask != 0).astype(np.float32)
        
        mask = torch.from_numpy(mask)
        mask = torch.permute(mask, (2, 0, 1)) # (H, W, C) -> (C, H, W)
        mask = self.pad_arr(mask)
        mask = tv_tensors.Mask(mask)
        
        if self.augmentations:
            img, mask = self.transforms(img, mask)
        
        return img, mask