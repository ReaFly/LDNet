import os
import os.path as osp
from utils.transform_multi import *
from torch.utils.data import Dataset
from torchvision import transforms


class PolypDataset(Dataset):
    def __init__(self, root, data_dir, mode='train', transform=None):
        super(PolypDataset, self).__init__()
        data_path = osp.join(root, data_dir)
        self.imglist = []
        self.gtlist = []

        datalist = os.listdir(osp.join(data_path, 'images'))
        for data in datalist:
            name = os.path.splitext(data)[0]
            self.imglist.append(osp.join(data_path+'/images', data))
            self.gtlist.append(osp.join(data_path+'/masks', name+'.png'))
            
        if transform is None:
            if mode == 'train':
               transform = transforms.Compose([
                   Resize((256, 256)),
                   RandomHorizontalFlip(),
                   RandomVerticalFlip(),
                   RandomRotation(90),
                   RandomZoom((0.9, 1.1)),
                   RandomCrop((224, 224)),
                   ToTensor(),
               ])
            elif mode == 'valid' or mode == 'test':
                transform = transforms.Compose([
                   Resize((224, 224)),
                   ToTensor(),
               ])
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imglist[index]
        gt_path = self.gtlist[index]
        name = img_path.split('/')[-1].split('.')[0]
        img = Image.open(img_path).convert('RGB')
        gt = Image.open(gt_path).convert('L')
        data = {'image': img, 'label': gt}
        if self.transform:
            data = self.transform(data)
        data['name'] = name
        return data

    def __len__(self):
        return len(self.imglist)
