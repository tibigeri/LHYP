import torch
import numpy as np
import os 
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class PatientDataset(Dataset):

    augmentation  = transforms.Compose([
    #transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomVerticalFlip(p=0.5),
    #transforms.GaussianBlur(5, sigma=(0.1,1.0)),
    transforms.RandomAffine(degrees=90, translate=(0.25, 0.25), scale=(0.9,1.1), shear=None)])

    def __init__(self, dir, transform = None):
        
        self.patients = []
        self.root_folder = os.listdir(dir)
        for i in range(len(self.root_folder)):
            pickle_in = open(dir + self.root_folder[i],'rb')
            self.patients.append(pickle.load(pickle_in))
            pickle_in.close
        
        self.transform = transform
    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx):
        im = self.patients[idx].dcm_images
        meta = self.patients[idx].pathology
  
        im = np.array(im)
        # pylint: disable=E1101
        im = torch.from_numpy(im)
        # pylint: enable=E1101
        
        ### Set labels for 3 groups ###

        #if (meta == 'Normal') or (meta == 'U18_f') or (meta == 'U18_m') or (meta == 'adult_f_sport') or (meta == 'adult_m_sport'):
        #    label = 0
        #elif (meta == 'HCM'):
        #    label = 1
        #else:
        #    label = 2

        ### Set labels for 2 groups ###

        if (meta == 'Normal') or (meta == 'U18_f') or (meta == 'U18_m') or (meta == 'adult_f_sport') or (meta == 'adult_m_sport'):
            label = 0
        else:
            label = 1 
            
        if self.transform:
            im = self.transform(im)
         
        im=im/255

        return {'images': im, 'labels': label}