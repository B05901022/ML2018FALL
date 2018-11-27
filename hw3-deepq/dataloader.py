# 2018 HTC Corporation. All Rights Reserved.
#
# This source code is licensed under the HTC license which can be found in the
# LICENSE file in the root directory of this work.

from torch.utils.data import Dataset, DataLoader
from PIL import Image as img
import csv

class ImageDataset(Dataset):
    def __init__(self, csvfile, transform=None):
        super().__init__()
        self.transform = transform
        self.arr = []
        with open(csvfile) as f:
            r = csv.reader(f)
            for row in r:
                self.arr.append(row)
            del r

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, idx):
        i = img.open('data/image/' + self.arr[idx][0])
        if self.transform:
            i = self.transform(i)
        if len(self.arr[idx]) > 1:
            return {'image': i, 'label': int(self.arr[idx][1])}
        else:
            return {'image': i}

def get_dataloader(name, transform, batchsize):
    assert name in ['train', 'test', 'valid']

    dataset = ImageDataset('data/' + name + '.csv', transform)
    return DataLoader(dataset, batch_size = batchsize, shuffle=(name == 'train'), num_workers=4)

