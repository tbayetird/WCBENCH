import sys,os
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms


class Dataset :

    def __init__(self,conf):
        self._load_conf(conf)

    """
    Loading the configuration. TODO : describe the configuration file
    """
    def _load_conf(self,conf):
        self.img_size = conf.IMG_SIZE
        # image pre-process
        normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            normalize
        ])
        self._load_dataset(conf.DATASET_PATH)

    """
    Loading the dataset. TODO  : describe the processus, the format.
    """
    def _load_dataset(self,dataset_path):
        self.dataset_path = dataset_path
        self.dataset_df = pd.read_csv(dataset_path)
        self.dataset_gen = self.dataset_df.itertuples()


    """
    Load datas from the dataset TODO : describe the process.
    """
    def load(self,n_data):
        data = []
        for _ in range(n_data) :
            row = (next(self.dataset_gen))
            id = row[2]
            img_path = row[3]
            ann = row[4]
            logit = row[5]
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            img = img.unsqueeze(0)

            data.append((id,img,ann,logit))
        return data
