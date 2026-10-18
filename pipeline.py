import os
import glob
import pandas as pd
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader

data_cat = ['train', 'valid'] # data categories

def get_study_level_data(study_type):
    """
    Returns a dict, with keys 'train' and 'valid' and respective values as study level dataframes.
    Adapted for custom flattened JPG dataset.
    """
    study_data = {}
    for phase in data_cat:
        BASE_DIR = f'MURA-v1.1/{phase}/{study_type}/'
        # Buscar todas las imágenes JPG en subcarpetas (normal/abnormal)
        all_files = glob.glob(BASE_DIR + '*/*.jpg')
        
        studies = {}
        for f in all_files:
            basename = os.path.basename(f) # ej: patient00006_study1_positive_image1.jpg
            parts = basename.replace('.jpg', '').split('_')
            
            if len(parts) >= 3:
                study_id = f"{parts[0]}_{parts[1]}" # ej: patient00006_study1
                label_str = parts[2]
                label = 1 if label_str == 'positive' else 0
                
                if study_id not in studies:
                    studies[study_id] = {'files': [], 'label': label, 'label_str': label_str}
                studies[study_id]['files'].append(f)
        
        # Crear DataFrame incluyendo la columna 'Path' para que utils.py pueda contar las clases
        df = pd.DataFrame(columns=['Path', 'Files', 'Count', 'Label'])
        for i, (s_id, data) in enumerate(studies.items()):
            fake_path = f"{s_id}_{data['label_str']}" 
            df.loc[i] = [fake_path, data['files'], len(data['files']), data['label']]
        
        study_data[phase] = df
    return study_data

class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_list = self.df.iloc[idx]['Files']
        label = self.df.iloc[idx]['Label']
        
        images = []
        for f in file_list:
            image = pil_loader(f)
            if self.transform:
                image = self.transform(image)
            images.append(image)
            
        images = torch.stack(images)
        sample = {'images': images, 'label': label}
        return sample

def get_dataloaders(data, batch_size=8, study_level=False):
    '''
    Returns dataloader pipeline with data augmentation
    '''
    data_transforms = {
        'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: ImageDataset(data[x], transform=data_transforms[x]) for x in data_cat}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in data_cat}
    return dataloaders

if __name__=='__main__':
    pass