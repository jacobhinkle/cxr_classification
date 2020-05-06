# cxr_classification

Studies in classifying chest X-rays

## Installation

You will need a conda environment including pytorch, pandas, numpy,
scikit-learn, torchvision, and pillow.

### Summit

On summit, you will need to clone the `ibm-wml-ce` conda env with the following
commands

```
# this will remove any currently loaded modules and load the defaults
module restore
module load ibm-wml-ce
conda create -n powerai170 --clone ibm-wml-ce-1.7.0-2
```
This will take a while and result in a large conda env in your home directory at
`~/.conda/envs/powerai170`. If you are out of room in your home dir, you can
create this environment anywhere you like by replacing `-n` with `--prefix` but
be warned that GPFS is purged every so often and **you will lose stuff**.

After creating the env, load it:
```
conda activate powerai170
```

### Install `torch_nlp_models`

We currently use my `torch_nlp_models` repo, only because I like my `CSVMeters`
class. I really should split this into its own package. Anyway, until then, you
will need to do the following before running code in `cxr_classification`:
```
git clone https://code-int.ornl.gov/4jh/torch_nlp_models.git
cd torch_nlp_models
pip install .
```
If you get an access denied error please let me (Jacob Hinkle, 4jh@ornl.gov) know.

# Usage

I have included a `densenet.lsf` script that shows an example invokation of
`train_densenet.py`. That script loads the data from `mimic_cxr_jpg.py`, which
simply provides a torch `Dataset` class for MIMIC-CXR-JPG, and functions for
splitting the data by patient, or grabbing the official split.

## Downscaled data

The dataset classes I provide for chexpert and mimic-cxr-jpg each take an
`image_subdir` argument in their constructor that allows you to point to a
different location (relative to the main data directory) to find preprocessed
image files. I have also included a `resize_images.py` script to create one of
those subdirs, keeping the directory structure and filenames intact.

For example, the following will downscale MIMIC-CXR-JPG to 512x512 and place the
result in a `files512x512` subdirectory
```
DATADIR=/gpfs/alpine/proj-shared/csc378/data/MIMIC-CXR-JPG
W=512
python resize_images.py -s ${W},${W} -i $DATADIR/files -o $DATADIR/files${W}x${W}
```
Then, in the constructor for MIMICCXRJPGDataset, you can point to this like so:
```
ds = MIMICCXRJPGDataset(..., image_subdir='files512x512')
```
In fact, you can see this is passed via `**kwargs` in any of the helper
functions. You can see how this is used in `train_densenet.py` which takes an
`--image-subdir` argument.
