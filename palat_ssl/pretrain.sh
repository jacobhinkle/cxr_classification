#!/bin/bash -l

set -ex

#conda activate cxr 

# Specify dataset. In this case, using dicom_id_file pointing to PA+LAT studies only (see get_palat_csv.py)
DATA_DESC=palat
DATA_ARGS="-d /scratch/4jh/cxr/MIMIC-CXR-JPG/ --dicom_id_file palat_dicomids.csv.gz"

RES=256

NUM_FOLDS=5
FOLD=0
CV_ARGS="--num-folds ${NUM_FOLDS} --fold ${FOLD}"

# select initialization here
PRETRAIN_DESC="imagenet"; PRETRAIN_ARGS=""  # defaults
PRETRAIN_DESC="scratch"; PRETRAIN_ARGS="--from-scratch"

RUN_NAME=${DATA_DESC}_pretrain_${PRETRAIN_DESC}_CV${NUM_FOLDS}.${FOLD}

OUTDIR=outputs/$(date +%Y%m%d)/$RUN_NAME

mkdir -p $OUTDIR

# scratch
python ../train_densenet_ssl.py \
	--image-subdir files${RES}x${RES} \
	$DATA_ARGS \
	$CV_ARGS \
	-o $OUTDIR \
	$PRETRAIN_ARGS \
	| tee $OUTDIR/stdout.txt

