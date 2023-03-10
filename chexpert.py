"""PyTorch Datasets for CheXpert"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

topdir = Path('/gpfs/alpine/proj-shared/csc378/data/CheXpert-v1.0')
chexpert_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices']


class CheXpertDataset(Dataset):
    """
    This class implements a :class:`torch.utils.data.Dataset` that serves the
    CheXpert dataset in the same format as the :class:`MIMICCXRJPGDataset` class.

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

    You can request various behaviors for how these uncertain labels are
    treated, using the 'label_method' argument:
        - 'ignore_uncertain' (default):
            - corresponds to U-Ignore in CheXpert paper
            - 1 = positive labels
            - 0 = negative labels
            - mask: zero whenever label is missing or unknown
        - 'zeros_uncertain':
            - corresponds to U-Zeros in CheXpert paper
            - 1 = positive labels
            - 0 = negative labels or unknown
            - mask: zero whenever label is missing. one otherwise
        - 'ones_uncertain':
            - corresponds to U-Ones in CheXpert paper
            - 1 = positive labels or unknown
            - 0 = negative labels
            - mask: zero whenever label is missing. one otherwise
        - 'zeros_uncertain_nomask':
            - corresponds to U-Zeros in CheXpert paper
            - 1 = positive labels
            - 0 = negative labels, missing, or unknown
            - mask: all ones
        - 'ones_uncertain_nomask':
            - corresponds to U-Ones in CheXpert paper
            - 1 = positive labels or unknown
            - 0 = negative labels or missing
            - mask: all ones
        - 'three_class':
            - corresponds to U-MultiClass in CheXpert paper
            - 1 = positive labels
            - 0 = negative labels
            - 2 = unknown
            - mask: zero whenever label is missing. one otherwise
        - 'four_class':
            - not implemented in CheXpert paper
            - 1 = positive labels
            - 0 = negative labels
            - 2 = unknown
            - 3 = missing
            - mask: all ones
        - 'missing_neg':
            - not implemented in CheXpert paper
            - 1 = positive labels
            - 0 = negative labels and missing
            - mask: zero when unknown. one otherwise

    If 'label_method' is a string, that method is applied to all labels. If it
    is a 'dict', then the keys must be the names of the conditions (exhaustive),
    and the values must be strings indicating which method to use.

    Note that 'No Finding' is "assigned a positive label (1) if
    there is no pathology classified as positive or uncertain" [1]. Therefore it
    only obtains values of 1 or missing, so 'missing_neg' is applied to 'No
    Finding' unless explicitly overridden by providing 'label_method' as a dict.

    [1] CheXpert paper: https://arxiv.org/pdf/1901.07031.pdf
    """
    def __init__(
        self,
        dataframe,
        labels=chexpert_labels,
        datadir=None,
        downscale_factor=None,
        transform=None,
        image_subdir='..',
        label_method='ignore_uncertain',
        ):
        self.dataframe = dataframe
        self.labels = labels
        self.downscale_factor = downscale_factor
        self.transform = transform

        if isinstance(label_method, str):
            label_method = {k: label_method for k in chexpert_labels}
            label_method['No Finding'] = 'missing_neg'
        elif not isinstance(label_method, dict):
            raise TypeError('label_method must be string or dict.')
        assert len(label_method) == len(chexpert_labels)
        self.label_method = label_method

        if datadir is None:
            datadir = topdir
        self.datadir = Path(datadir) / image_subdir

    def __len__(self):
        return len(self.dataframe.index)

    def map_labels(self, row):
        labels = []
        mask = []
        for i, c in enumerate(chexpert_labels):
            m = self.label_method[c]
            l = float(row[i])
            if m == 'ignore_uncertain':
                mask.append(1 - np.isnan(l) + (l == -1.0))
                labels.append(l == 1.0)
            elif m == 'zeros_uncertain':
                mask.append(1 - np.isnan(l))
                if l == -1.0:
                    labels.append(0)
                else:
                    labels.append(l)
            elif m == 'ones_uncertain':
                mask.append(1 - np.isnan(l))
                if l == -1.0:
                    labels.append(1)
                else:
                    labels.append(l)
            elif m == 'zeros_uncertain_nomask':
                mask.append(1)
                if l == -1.0:
                    labels.append(0)
                elif np.isnan(l):
                    labels.append(0)
                else:
                    labels.append(l)
            elif m == 'ones_uncertain_nomask':
                mask.append(1)
                if l == -1.0:
                    labels.append(1)
                elif np.isnan(l):
                    labels.append(0)
                else:
                    labels.append(l)
            elif m == 'three_class':
                mask.append(1 - np.isnan(l))
                if l == -1.0:
                    labels.append(2)
                else:
                    labels.append(l)
            elif m == 'four_class':
                mask.append(1)
                if l == -1.0:
                    labels.append(2)
                elif np.isnan(l):
                    labels.append(3)
                else:
                    labels.append(l)
            elif m == 'missing_neg':
                mask.append(1 - (l == -1.0))
                labels.append(l == 1.0)
            else:
                raise ValueError(f'Unknown label method: {m}')

        labels = torch.as_tensor(labels).type(torch.int8)
        mask = torch.as_tensor(mask).type(torch.int8)

        return labels, mask

    def __getitem__(self, ix):
        row = self.dataframe.iloc[ix]

        im = Image.open(self.datadir / row.Path)

        if self.transform is not None:
            im = self.transform(im)

        im = torch.as_tensor(np.array(im)).unsqueeze(0)

        if self.downscale_factor is not None:
            im = F.avg_pool2d(im.type(torch.float32), self.downscale_factor)

        labels, labelmask = self.map_labels(row[self.labels])

        return im, labels, labelmask


def official_split(datadir=topdir, **kwargs):
    """
    The MIMIC-CXR-JPG dataset comes with an official train-val-test split, which
    this function implements.

    Three datasets are returned in this order: train, validate, test.
    """
    datadir = Path(datadir)

    trainrecords = pd.read_csv(datadir / 'train.csv')
    valrecords = pd.read_csv(datadir / 'valid.csv')

    train = CheXpertDataset(trainrecords,
            datadir=datadir, **kwargs)
    val = CheXpertDataset(valrecords,
            datadir=datadir, **kwargs)
    test = None

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
    train, val, test = official_split()
    #train, val, test = cv(10, 0, random_state=0, stratify=False)

    print(len(train), len(val))

    print(train[0])
    print(val[0])
    #print(test[0])
