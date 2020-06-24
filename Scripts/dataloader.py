import albumentations as aug
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise,RandomRotate90)
from albumentations.pytorch import ToTensor

def get_transforms(phase):
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
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms

class SpineDataset(Dataset):
    def __init__(self, phase = 'train'): 
        if phase=='train':
            self.path = '../Data/boostnet_labeldata/data/training/'
            self.label_path = '../Data/boostnet_labeldata/labels/training/'
        else:
            self.path = '../Data/boostnet_labeldata/data/test/'
            self.label_path = '../Data/boostnet_labeldata/labels/test/'
            
        self.filenames = pd.read_csv(self.label_path+'filenames.csv', header = None).iloc[:, 0].values
        self.labels = pd.read_csv(self.label_path+'landmarks.csv', header = None)
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = cv2.imread(self.path+self.filenames[idx])
        landmarks = self.labels.loc[idx].values
        landmarks = [[int(round(img.shape[1]*landmarks[m])),int(round(img.shape[0]*landmarks[m+68]))] for m in range (0,68)]
        H,W,_ = img.shape
        N=4     
        box = [landmarks[n:n+N] for n in range(0, len(landmarks), N)]
        box = np.array(box)
        if self.transform:
            box = box.reshape(-1, 2)
            augmented = self.transform(image=img, keypoints = box)
            img = augmented['image']
            box = augmented['keypoints']
            box = box.view(-1,4,2)

        return img, box