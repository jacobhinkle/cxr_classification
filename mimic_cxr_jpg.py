"""PyTorch Datasets for MIMIC-CXR-JPG"""

import os
import pathlib

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

topdir = pathlib.Path('/gpfs/alpine/proj-shared/csc378/data/MIMIC-CXR-JPG')
chexpert_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices']


class MIMICCXRJPGDataset(Dataset):
    """
    This class implements a :class:`torch.utils.data.Dataset` that serves the
    MIMIC-CXR-JPG dataset at native resolution.

    Images are served as uint8 tensors of shape (1, H, W) where H and W are the
    height and width of the image in its native resolution. Notice that these
    numbers vary from image to image, so resampling must be used for
    minibatching.

    There are 14 binary labels associated with each image (actually each group
    of images referred to as a study).
    Labels in this dataset have a high degree of missingness, and come in four
    flavors:
        - Label is positive
        - Label is negative
        - Label was not mentioned (missing)
        - Label was mentioned and specifically indicated to be unknown.
    The first two cases are specified by ones and zeros in a length-14 int8
    vector called labels.  For our purposes the last two cases are treated
    equally as missing data.  Missing data is indicated by an additional
    length-14 int8 vector called labelmask, whose ones indicate positive or
    negative labels and whose zeros indicate missing labels for this example.
    """
    def __init__(
        self,
        dataframe,
        labels=chexpert_labels,
        datadir=None,
        downscale_factor=None,
        transform=None,
        image_subdir='files',
        ):
        self.dataframe = dataframe
        self.labels = labels
        self.downscale_factor = downscale_factor
        self.transform = transform

        if datadir is None:
            datadir = topdir
        self.datadir = pathlib.Path(datadir) / image_subdir

    def __len__(self):
        return len(self.dataframe.index)

    def __getitem__(self, ix):
        row = self.dataframe.iloc[ix]

        im = Image.open(self.datadir / row.path)

        if self.transform is not None:
            im = self.transform(im)

        im = torch.as_tensor(np.array(im)).unsqueeze(0)

        if self.downscale_factor is not None:
            im = F.avg_pool2d(im.type(torch.float32), self.downscale_factor)


        labels = row[self.labels].to_numpy().astype(float)
        labelmask = 1 - torch.as_tensor((
            np.isnan(labels) + (labels == -1.0)
            ).astype(np.int8))
        labels = torch.as_tensor((labels == 1.0).astype(np.int8))

        return im, labels, labelmask


def official_split(**kwargs):
    """
    The MIMIC-CXR-JPG dataset comes with an official train-val-test split, which
    this function implements.

    Three datasets are returned in this order: train, validate, test.
    """
    allrecords = pd.merge(
        pd.read_csv(topdir / 'splitpaths.csv.gz'),
        pd.read_csv(topdir / 'mimic-cxr-2.0.0-chexpert.csv.gz'),
        on=['subject_id', 'study_id'],
    )

    train = MIMICCXRJPGDataset(allrecords.query('split == "train"'), **kwargs)
    val = MIMICCXRJPGDataset(allrecords.query('split == "validate"'), **kwargs)
    test = MIMICCXRJPGDataset(allrecords.query('split == "test"'), **kwargs)

    return train, val, test


def cv(num_folds, fold, val_size=0.1, random_state=0, stratify=False, **kwargs):
    """
    Cross-validation with splitting at subject level.
    """
    allrecords = pd.merge(
        pd.read_csv(topdir / 'splitpaths.csv.gz'),
        pd.read_csv(topdir / 'mimic-cxr-2.0.0-chexpert.csv.gz'),
        on=['subject_id', 'study_id'],
    )

    if stratify:
        # convert to binary labels
        allrecords_binary = allrecords.copy()
        allrecords_binary[chexpert_labels] = (allrecords_binary[chexpert_labels] == 1).astype(int)
        # combine by collecting findings from all studies for each subject
        subject_findings = (allrecords_binary[['subject_id'] + chexpert_labels].groupby('subject_id')).max()
    else:
        from sklearn.model_selection import KFold, train_test_split
        kf = KFold(num_folds)#, random_state=random_state, shuffle=True)
        for k, (trainval_subj, test_subj) in enumerate(kf.split(allrecords['subject_id'].unique())):
            if k != fold: continue
            train_subj, val_subj = train_test_split(
                trainval_subj,
                test_size=val_size,
                random_state=random_state,
                shuffle=False,
            )

    subjrecs = lambda s: pd.DataFrame({'subject_id': s}).merge(allrecords, how='left', on='subject_id')
    trainrecords = subjrecs(train_subj)
    valrecords = subjrecs(val_subj)
    testrecords = subjrecs(test_subj)

    train = MIMICCXRJPGDataset(trainrecords, **kwargs)
    val = MIMICCXRJPGDataset(valrecords, **kwargs)
    test = MIMICCXRJPGDataset(testrecords, **kwargs)

    return train, val, test

if __name__ == '__main__':
    #train, val, test = official_split()
    train, val, test = cv(10, 0, random_state=0, stratify=False)

    print(len(train), len(val), len(test))

    print(train[0])
    print(val[0])
    print(test[0])
