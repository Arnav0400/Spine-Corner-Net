import albumentations as aug
from albumentations.pytorch import ToTensor
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

def makeGaussian(H, W, radius = 3, center=None):

    x = np.arange(0, W, 1, float)
    y = np.arange(0, H, 1, float)[:,np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0].cpu().numpy()
        y0 = center[1].cpu().numpy()

    heat = np.exp((-1 * ((x-x0)**2 + (y-y0)**2)) / (2*((radius/3)**2)))
    heat*=cv2.circle(np.zeros_like(heat), (x0,y0), radius, (255,255,255), -1)
    return heat

def create_heat(H, W, points, radius = 20):
    mask = np.zeros((H, W))
    for point in points:
        x, y = point[0], point[1]
        mask+=makeGaussian(H, W, radius,[x,y])
    return mask
    
def get_transforms(phase, size):
    list_transforms = []
    if phase == "train":
        list_transforms.extend([
            aug.HorizontalFlip(),
            aug.OneOf([
                aug.RandomContrast(),
                aug.RandomGamma(),
                aug.RandomBrightness(),
                ], p=0.5
                ),
        ])
    list_transforms.extend(
        [
#             Normalize(mean=mean, std=std, p=1),
#             aug.Resize(size[0], size[1]),
            ToTensor(),
        ]
    )
    list_trfms = aug.Compose(list_transforms,keypoint_params=aug.KeypointParams(format='xy'))
    return list_trfms

class SpineDataset(Dataset):
    def __init__(self, phase = 'train', input_size = (768, 256), output_size = (192, 64), radius = 10): 
        if phase=='train':
            self.path = '../Data/boostnet_labeldata/data/training/'
            self.label_path = '../Data/boostnet_labeldata/labels/training/'
        else:
            self.path = '../Data/boostnet_labeldata/data/test/'
            self.label_path = '../Data/boostnet_labeldata/labels/test/'
            
        self.filenames = pd.read_csv(self.label_path+'filenames.csv', header = None).iloc[:, 0].values
        self.labels = pd.read_csv(self.label_path+'landmarks.csv', header = None)
        self.transform = get_transforms(phase, input_size)
        self.input_size = input_size
        self.output_size = output_size
        self.radius = 10
        self.phase = phase
        
    def __len__(self):
        return len(self.filenames)
            
    def __getitem__(self, idx):
        img = cv2.imread(self.path+self.filenames[idx])
        img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        landmark = self.labels.loc[idx].values
        landmarks = [[int(self.output_size[1]*landmark[m]),int(self.output_size[0]*landmark[m+68])] for m in range (0,68)]
        landmarks_ = [[self.output_size[1]*landmark[m],self.output_size[0]*landmark[m+68]] for m in range (0,68)]
        N=4     
        box = [landmarks[n:n+N] for n in range(0, len(landmarks), N)]
        box = np.array(box)

        box_ = [landmarks_[n:n+N] for n in range(0, len(landmarks_), N)]
        box_regr = np.array(box_) - box

        if self.transform:
            box = box.reshape(-1, 2)
            augmented = self.transform(image=img, keypoints = box)
            img = augmented['image']
            box = augmented['keypoints']
            box = torch.tensor(box)
            box = box.view((-1,4,2))
            
        tl_heatmaps = create_heat(self.output_size[0], self.output_size[1], box[:,0,:], self.radius)
        tr_heatmaps = create_heat(self.output_size[0], self.output_size[1], box[:,1,:], self.radius)
        bl_heatmaps = create_heat(self.output_size[0], self.output_size[1], box[:,2,:], self.radius)
        br_heatmaps = create_heat(self.output_size[0], self.output_size[1], box[:,3,:], self.radius)

        max_tag_len = 17
        tl_regr    = np.zeros((max_tag_len, 2), dtype=np.float32)
        br_regr    = np.zeros((max_tag_len, 2), dtype=np.float32)
        tr_regr    = np.zeros((max_tag_len, 2), dtype=np.float32)
        bl_regr    = np.zeros((max_tag_len, 2), dtype=np.float32)
        tl_tag     = np.zeros((max_tag_len), dtype=np.int64)
        br_tag     = np.zeros((max_tag_len), dtype=np.int64)
        tr_tag     = np.zeros((max_tag_len), dtype=np.int64)
        bl_tag     = np.zeros((max_tag_len), dtype=np.int64)
        tag_mask   = np.zeros((max_tag_len), dtype=np.uint8)
        tag_len    = 0
        
        for ind, (detection, detection_regr) in enumerate(zip(box, box_regr)):
            xtl, ytl = int(detection[0,0]), int(detection[0,1])
            xtr, ytr = int(detection[1,0]), int(detection[1,1])
            xbl, ybl = int(detection[2,0]), int(detection[2,1])
            xbr, ybr = int(detection[3,0]), int(detection[3,1])

            xtl_reg, ytl_reg = detection_regr[0,0], detection_regr[0,1]
            xtr_reg, ytr_reg = detection_regr[1,0], detection_regr[1,1]
            xbl_reg, ybl_reg = detection_regr[2,0], detection_regr[2,1]
            xbr_reg, ybr_reg = detection_regr[3,0], detection_regr[3,1]

            tl_regr[ind] = [xtl_reg, ytl_reg]
            tl_regr[ind] = [xtl_reg, ytl_reg]
            tl_regr[ind] = [xtl_reg, ytl_reg]
            tl_regr[ind] = [xtl_reg, ytl_reg]
            
            tl_tag[ind] = ytl * self.output_size[1] + xtl
            br_tag[ind] = ybr * self.output_size[1] + xbr
            tr_tag[ind] = ytr * self.output_size[1] + xtr
            bl_tag[ind] = ybl * self.output_size[1] + xbl
            
            tag_len+=1

        tag_mask[:tag_len] = 1
        
        
        tl_heatmaps = torch.from_numpy(tl_heatmaps[np.newaxis,...])
        br_heatmaps = torch.from_numpy(br_heatmaps[np.newaxis,...])
        tr_heatmaps = torch.from_numpy(tr_heatmaps[np.newaxis,...])
        bl_heatmaps = torch.from_numpy(bl_heatmaps[np.newaxis,...])
        
        tl_regr    = torch.from_numpy(tl_regr)
        br_regr    = torch.from_numpy(br_regr)
        tr_regr    = torch.from_numpy(tr_regr)
        bl_regr    = torch.from_numpy(bl_regr)
        
        tl_tag     = torch.from_numpy(tl_tag)
        br_tag     = torch.from_numpy(br_tag)
        tr_tag     = torch.from_numpy(tr_tag)
        bl_tag     = torch.from_numpy(bl_tag)
        
        tag_mask   = torch.from_numpy(tag_mask)
        if(self.phase != 'train'):
            return {
            "xs": [img],
            "ys": [tl_heatmaps, br_heatmaps, tr_heatmaps, bl_heatmaps, tag_mask, tl_regr, br_regr, tr_regr, bl_regr]
            }
        return {
            "xs": [img, tl_tag, br_tag, tr_tag, bl_tag],
            "ys": [tl_heatmaps, br_heatmaps, tr_heatmaps, bl_heatmaps, tag_mask, tl_regr, br_regr, tr_regr, bl_regr]
        }
            


def provider(phase, batch_size=8, num_workers=4):
    '''Returns dataloader for the model training'''
    if phase == 'train':
        image_dataset = SpineDataset(phase)
        pin = True
        shuffle = True
    else:
        image_dataset = SpineDataset(phase)
        pin = False
        shuffle = False
        
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin,
        shuffle=shuffle,
        drop_last = True
    )

    return dataloader
