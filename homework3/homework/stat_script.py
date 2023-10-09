import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from os import path
import csv
from torchvision.transforms import functional as F
import numpy as np
from homework.models import FCN

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']
DENSE_LABEL_NAMES = ['background', 'kart', 'track', 'bomb/projectile', 'pickup/nitro']

def label_to_tensor(lbl):
    """
    Reads a PIL pallet Image img and convert the indices to a pytorch tensor
    """
    return torch.as_tensor(np.array(lbl, np.uint8, copy=False))


def label_to_pil_image(lbl):
    """
    Creates a PIL pallet Image from a pytorch tensor of labels
    """
    if not(isinstance(lbl, torch.Tensor) or isinstance(lbl, np.ndarray)):
        raise TypeError('lbl should be Tensor or ndarray. Got {}.'.format(type(lbl)))
    elif isinstance(lbl, torch.Tensor):
        if lbl.ndimension() != 2:
            raise ValueError('lbl should be 2 dimensional. Got {} dimensions.'.format(lbl.ndimension()))
        lbl = lbl.numpy()
    elif isinstance(lbl, np.ndarray):
        if lbl.ndim != 2:
            raise ValueError('lbl should be 2 dimensional. Got {} dimensions.'.format(lbl.ndim))

    im = Image.fromarray(lbl.astype(np.uint8), mode='P')
    im.putpalette([0xee, 0xee, 0xec, 0xfc, 0xaf, 0x3e, 0x2e, 0x34, 0x36, 0x20, 0x4a, 0x87, 0xa4, 0x0, 0x0] + [0] * 753)
    return im

class ToTensor(object):
    def __call__(self, image, label):
        return F.to_tensor(image), label_to_tensor(label)

class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=ToTensor()):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            self.files.append(im_f.replace('_im.jpg', ''))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        lbl = Image.open(b + '_seg.png')
        if self.transform is not None:
            im, lbl = self.transform(im, lbl)
        return im, lbl


def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)






class StatLoader(Dataset):
    def __init__(self, dataset_path, augment=False):
        self.img_labels = []
        self.short_path = dataset_path
        self.aug_flag = augment

        with open(path.join(dataset_path, 'labels.csv'), newline='') as csvfile:
            labs = csv.reader(csvfile, delimiter=',')
            next(labs)
            for row in labs:
                row[1] = LABEL_NAMES.index(row[1])
                self.img_labels.append(row)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        if self.aug_flag:
            with Image.open(path.join(self.short_path, self.img_labels[idx][0])) as im:
                transform = transforms.Compose([
                                                transforms.RandomResizedCrop(size=(64, 64)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(brightness=15),
                                                transforms.ToTensor()
                                                ]
                                               )
                img = transform(im)
        else:
            with Image.open(path.join(self.short_path, self.img_labels[idx][0])) as im:
                transform = transforms.Compose([transforms.ToTensor()])
                img = transform(im)

        return img, self.img_labels[idx][1]


def load_data_special(dataset_path, num_workers=0, batch_size=1):
    dataset = StatLoader(dataset_path, augment=False)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)






if __name__ == '__main__':

    img_labels = []
    short_path = r'C:\Users\Jason\PycharmProjects\Deep Learning\HW 3\homework3\data\train'

    with open(path.join(r'C:\Users\Jason\PycharmProjects\Deep Learning\HW 3\homework3\data\train', 'labels.csv'), newline='') as csvfile:
        labs = csv.reader(csvfile, delimiter=',')
        next(labs)
        for row in labs:
            row[1] = LABEL_NAMES.index(row[1])
            img_labels.append(row)
    for _ in range(10):
        with Image.open(path.join(short_path, img_labels[2][0])) as im:
            transform = transforms.ColorJitter(brightness=2, contrast=1)
            im = transform(im)
            im.show()


    # stat_data = load_data_special(r'C:\Users\Jason\PycharmProjects\Deep Learning\HW 3\homework3\data\train', batch_size=1)
    # length = stat_data.__len__()
    # dmean = torch.zeros([1, 3])
    # tempsum = torch.zeros([1, 3])
    # for data, label in stat_data:
    #     dmean += data.mean(dim=[2, 3])
    #
    # mn = dmean/length
    #
    # for data, label in stat_data:
    #     tempsum += ((data - mn[:, :, None, None]).square()).mean(dim=[2, 3])
    #
    # stddv = (tempsum/length).sqrt()
    #
    # print(mn[0, 0].item())
    # print(mn[0, 1].item())
    # print(mn[0, 2].item())
    #
    # print('\n')
    #
    # print(stddv[0, 0].item())
    # print(stddv[0, 1].item())
    # print(stddv[0, 2].item())
    #
    # print('\n')
    #
    # print('Mean = ' + str(mn.squeeze().tolist()))
    # print('STDV = ' + str(stddv.squeeze().tolist()))

