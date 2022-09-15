"""PyTorch Datasets for MIMIC-CXR-JPG"""

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import os
from pathlib import Path
import zipfile

topdir = Path("/mnt/DGX01/Personal/4jh/cxr/MIMIC-CXR-JPG")
chexpert_labels = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

normalize = transforms.Normalize(
    mean=[0.449], std=[0.226]  # [0.485, 0.456, 0.406],
)  # [0.229, 0.224, 0.225]),


def load_all_metadata(
        data_dir=topdir,  # directory containing all the .csv.gz files
    ):
    """
    Load all metadata files, joining appropriately.

    This loads the following files and joins on dicom_id and study_id (where
    appropriate):

    - mimic-cxr-2.0.0-metadata.csv.gz
    - mimic-cxr-2.0.0-chexpert.csv.gz
    - mimic-cxr-2.0.0-split.csv.gz
    - mimic-cxr-reports.zip
    """
    data_dir = Path(data_dir)
    print(data_dir)

    metadata = pd.read_csv(data_dir / "mimic-cxr-2.0.0-metadata.csv.gz")
    chexpert = pd.read_csv(data_dir / "mimic-cxr-2.0.0-chexpert.csv.gz")
    splitpaths = pd.read_csv(data_dir / "splitpaths.csv.gz")

    meta = pd.merge(
        metadata,
        splitpaths,
        on=["dicom_id", "subject_id", "study_id"],
    )
    meta = pd.merge(
        meta,
        chexpert,
        on=["subject_id", "study_id"],
    )

    return meta


def label_prevalences(dataloader):
    """Return average of the labels vector, over all examples in a dataset"""
    if isinstance(dataloader, Dataset):
        # convert to DataLoader with some default args
        dataloader = DataLoader(dataloader, batch_size=8, num_workers=1)
    else:
        assert isinstance(dataloader, DataLoader)

    sum_label = 0
    num_instances =0 
    with torch.no_grad():
        for _, labels, *_ in tqdm(dataloader):
            num_instances += labels.shape[0]
            labels = labels.sum(dim=0)
            sum_label = sum_label + labels.type(torch.float64)
    return sum_label / num_instances


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
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=[-20, 20]),
            ]
        ),
        image_subdir="files",
        label_method="zeros_uncertain_nomask",  # old default: "ignore_uncertain"
        load_activations=False,  # If True, load .pth files and do not apply transforms
    ):
        super().__init__()

        self.dataframe = dataframe
        self.labels = labels
        self.downscale_factor = downscale_factor
        self.transform = transform
        self.load_activations = load_activations

        if isinstance(label_method, str):
            label_method = {k: label_method for k in chexpert_labels}
            label_method["No Finding"] = "missing_neg"
        elif not isinstance(label_method, dict):
            raise TypeError("label_method must be string or dict.")
        assert len(label_method) == len(chexpert_labels)
        self.label_method = label_method

        if datadir is None:
            print(f"Using default data directory: {datadir}")
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
            if m == "ignore_uncertain":
                mask.append(1 - np.isnan(l) + (l == -1.0))
                labels.append(l == 1.0)
            elif m == "zeros_uncertain":
                mask.append(1 - np.isnan(l))
                if l == -1.0:
                    labels.append(0)
                else:
                    labels.append(l)
            elif m == "ones_uncertain":
                mask.append(1 - np.isnan(l))
                if l == -1.0:
                    labels.append(1)
                else:
                    labels.append(l)
            elif m == "zeros_uncertain_nomask":
                mask.append(1)
                if l == -1.0:
                    labels.append(0)
                elif np.isnan(l):
                    labels.append(0)
                else:
                    labels.append(l)
            elif m == "ones_uncertain_nomask":
                mask.append(1)
                if l == -1.0:
                    labels.append(1)
                elif np.isnan(l):
                    labels.append(0)
                else:
                    labels.append(l)
            elif m == "three_class":
                mask.append(1 - np.isnan(l))
                if l == -1.0:
                    labels.append(2)
                else:
                    labels.append(l)
            elif m == "four_class":
                mask.append(1)
                if l == -1.0:
                    labels.append(2)
                elif np.isnan(l):
                    labels.append(3)
                else:
                    labels.append(l)
            elif m == "missing_neg":
                mask.append(1 - (l == -1.0))
                labels.append(l == 1.0)
            else:
                raise ValueError(f"Unknown label method: {m}")

        labels = torch.as_tensor(labels).type(torch.int8)
        mask = torch.as_tensor(mask).type(torch.int8)

        return labels, mask

    def get_from_row(self, row):
        if self.load_activations:
            # Replace extension with .pth
            b, _ = os.path.splitext(row.path)
            pthpath = b + ".pth"
            im = torch.load(self.datadir / pthpath)
        else:
            im = Image.open(self.datadir / row.path)

            if self.transform is not None:
                im = self.transform(im)

            if self.downscale_factor is not None:
                im = F.avg_pool2d(im.type(torch.float32), self.downscale_factor)

        labels, labelmask = self.map_labels(row[self.labels])

        return im, labels, labelmask


    def __getitem__(self, ix):
        row = self.dataframe.iloc[ix]

        return self.get_from_row(row)


def collate_studies(studies):
    """Collate studies in a reasonable way"""
    ims = []
    labs = []
    masks = []
    lengths = []
    metas = []
    off = 0
    for im, lab, mask, meta in studies:
        ims.extend(im)
        labs.extend(lab)
        masks.extend(mask)
        metas.append(meta)
        lengths.append(len(im))

    return torch.stack(ims), torch.stack(labs), torch.stack(masks), lengths, pd.concat(metas)



class MIMICCXRJPGStudyDataset(Dataset):
    """
    This class implements a :class:`torch.utils.data.Dataset` that serves the
    MIMIC-CXR-JPG dataset one study (multiple images) at a time.

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
        image_meta,
        report_zip=None,
        **ds_kwargs,
    ):
        super().__init__()

        self.image_meta = image_meta

        self.report_z = zipfile.ZipFile(report_zip, 'r') if report_zip is not None else None

        # instantiate a Dataset for individual images
        self.im_ds = MIMICCXRJPGDataset(image_meta, **ds_kwargs)

        # group the studies
        study_keys = ['subject_id', 'study_id']  # used to identify a study
        self.study_group = self.image_meta.groupby(study_keys)
        self.study_names = sorted(self.study_group.groups.keys())

    def __len__(self):
        return len(self.study_group)

    def get_study(self, study_name):
        study = self.study_group.get_group(study_name)

        # return lists of ragged-shaped tensors, along with ViewPositions as strings
        imgs, labels, labelmasks = [], [], []
        for i in range(len(study)):
            row = study.iloc[i]  # row corresponding to each image
            img, l, m = self.im_ds.get_from_row(row)

            imgs.append(img)
            if i == 0:
                labels.append(l)
                labelmasks.append(m)

        if self.report_z is not None:
            # infer report path from row path
            reppath = 'files/' + os.path.dirname(row.path) + '.txt'
            # load report from zip file
            rep = self.report_z.open(reppath).read()
            # add a new column ReportText (repeated across study)
            study.insert(len(study.columns), 'ReportText', rep)

        return imgs, labels, labelmasks, study

    def __getitem__(self, i):
        return self.get_study(self.study_names[i])


def records_dataset(
    records,
    datadir,
    train_transform=transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ]
    ),
    test_transform=transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    ),
    return_studies=False,
    load_reports=False,
    dataloaders=False,  # whether to create dataloaders or just return datasets
    dl_kwargs={},
    **kwargs,
):
    ds_cls = MIMICCXRJPGStudyDataset if return_studies else MIMICCXRJPGDataset

    if load_reports and return_studies:
        # pass report_zip only if needed
        kwargs['report_zip'] = os.path.join(datadir, 'mimic-cxr-reports.zip')

    ret = ds_cls(
        records,
        datadir=datadir,
        transform=train_transform,
        **kwargs,
    )

    if dataloaders:
        collate_fn = collate_studies if return_studies else None
        ret = DataLoader(ret, collate_fn=collate_fn, **dl_kwargs)

    return ret


def official_split(
    datadir=topdir,
    dicom_id_file=None,
    **kwargs,
):
    """
    The MIMIC-CXR-JPG dataset comes with an official train-val-test split, which
    this function implements.

    Three datasets are returned in this order: train, validate, test.
    """
    datadir = Path(datadir)

    allrecords = load_all_metadata(datadir)

    if dicom_id_file is not None:
        # restrict to only the given dicoms, if given
        dcms = pd.read_csv(dicom_id_file)
        allrecords = pd.merge(allrecords, dcms[['dicom_id']], on='dicom_id')

    trainrecs = allrecords.query('split == "train"')
    valrecs = allrecords.query('split == "validate"')
    testrecs = allrecords.query('split == "test"')

    return (
        records_dataset(trainrecs, datadir, **kwargs),
        records_dataset(valrecs, datadir, **kwargs),
        records_dataset(testrecs, datadir, **kwargs),
    )


def cv(
    num_folds,
    fold,
    datadir=topdir,
    dicom_id_file=None,
    val_size=0.1,
    random_state=0,
    stratify=False,
    **kwargs,
):
    """
    Cross-validation with splitting at subject level.
    """
    datadir = Path(datadir)

    allrecords = load_all_metadata(datadir)

    if dicom_id_file is not None:
        # restrict to only the given dicoms, if given
        dcms = pd.read_csv(dicom_id_file)
        allrecords = pd.merge(allrecords, dcms[['dicom_id']], on='dicom_id')

    if stratify:
        # convert to binary labels
        allrecords_binary = allrecords.copy()
        allrecords_binary[chexpert_labels] = (
            allrecords_binary[chexpert_labels] == 1
        ).astype(int)
        # combine by collecting findings from all studies for each subject
        subject_findings = (
            allrecords_binary[["subject_id"] + chexpert_labels].groupby("subject_id")
        ).max()
    else:
        from sklearn.model_selection import KFold, train_test_split

        kf = KFold(num_folds)  # , random_state=random_state, shuffle=True)
        uniq_subj = allrecords["subject_id"].unique()
        for k, (trainval_ix, test_ix) in enumerate(kf.split(uniq_subj)):
            if k != fold:
                continue
            trainval_subj = uniq_subj[trainval_ix]
            test_subj = uniq_subj[test_ix]
            train_subj, val_subj = train_test_split(
                trainval_subj,
                test_size=val_size,
                random_state=random_state,
                shuffle=False,
            )

    subjrecs = lambda s: pd.DataFrame({"subject_id": s}).merge(
        allrecords, how="left", on="subject_id"
    )
    trainrecords = subjrecs(train_subj)
    valrecords = subjrecs(val_subj)
    testrecords = subjrecs(test_subj)

    return (
        records_dataset(trainrecords, datadir, **kwargs),
        records_dataset(valrecords, datadir, **kwargs),
        records_dataset(testrecords, datadir, **kwargs),
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Explore the MIMIC-CXR-JPG dataset')
    parser.add_argument(
        '--data_dir',
        default="/scratch/4jh/cxr/MIMIC-CXR-JPG",
        help="Location of top-level MIMIC-CXR-JPG directory",
    )
    parser.add_argument(
        '--image_subdir',
        default="files",
        help="Location of JPG file subdirectory under top-level MIMIC-CXR-JPG directory",
    )
    parser.add_argument(
        '--split_type',
        choices=("official", "cv"),
        default="official",
        help="Whether to use the official split or create our own cross-validation splits",
    )
    parser.add_argument(
        '--return_studies',
        action='store_true',
        help="Whether to consider an entire study an example. If false, each "
        "image is an independent example, and the corresponding study's labels "
        "are re-used for each image in the study",
    )
    parser.add_argument(
        '--load_reports',
        action='store_true',
        help="If given, test loading reports into the metadata of each study. "
        "Only has effect if --return_studies also given.",
    )
    parser.add_argument(
        '--dicom_id_file',
        help="If given, restrict to only the dicom_ids in the 'dicom_id' column of a given CSV file",
    )
    args = parser.parse_args()

    print("Split type:", args.split_type)
    common_args = dict(
        datadir=args.data_dir,
        image_subdir=args.image_subdir,
        dicom_id_file=args.dicom_id_file,
        return_studies=args.return_studies,
        load_reports=args.load_reports,
    )
    # NOTE: as of 2022-09-13 CPython's zipfile breaks on
    # multithreaded reads
    num_workers = 1 if args.load_reports else 12
    if args.split_type == "official":
        train, val, test = official_split(**common_args)
        traindl, valdl, testdl = official_split(
            **common_args,
            dataloaders=True,
            dl_kwargs={
                'batch_size': 8,
                'num_workers': num_workers,
                },
        )
    else:
        cv_args = dict(
            **common_args,
            num_folds=10,
            fold=0,
            random_state=0,
            stratify=False,
        )
        train, val, test = cv(**cv_args)
        traindl, valdl, testdl = cv(
            **cv_args,
            dataloaders=True,
            dl_kwargs={
                'batch_size': 8,
                'num_workers': num_workers,
                },
        )

    print("Train/val/test number of examples:", len(train), len(val), len(test))
    print("Train/val/test number of batches:", len(traindl), len(valdl), len(testdl))

    batch = next(iter(traindl))
    print("\nSample batch (train):")
    print("  Types:", *(type(b).__name__ for b in batch))
    if args.return_studies:
        print("  Shapes:", batch[0].shape, batch[1].shape, batch[2].shape,
                len(batch[3]), batch[4].shape)
        print("  Interpretation: images, labels, labelmasks, studylengths, metadata")
        print("  Study lengths (list):", batch[3])
        print("  Metadata (dataframe):")
        print("    Columns:", batch[4].columns)
        print(batch[4])
        if args.load_reports:
            print("  First report:\n", batch[4].iloc[0].ReportText)
    else:
        print("  Shapes:", *(b.shape for b in batch))
        print("  Interpretation: images, labels, labelmasks")
    print("  Labels:", batch[1])
    print("  Label masks:", batch[2])

    prev_train = pd.DataFrame(label_prevalences(traindl), index=chexpert_labels)
    prev_val = pd.DataFrame(label_prevalences(valdl), index=chexpert_labels)
    prev_test = pd.DataFrame(label_prevalences(testdl), index=chexpert_labels)
    print("\nLabel prevalences (train):")
    print(prev_train)
    print("\nLabel prevalences (val):")
    print(prev_val)
    print("\nLabel prevalences (test):")
    print(prev_test)
