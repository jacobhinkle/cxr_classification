# PA+Lateral SemiSupervised Learning Experiment

In this experiment, we restrict to all studies in MIMIC-CXR-JPG containing
exactly two images: one PA and one Lateral (or LL) view.  We then compare classification using  ImageNet pretrained weights, initialized weights from scratch, or pretrained weights using Semi-Supervised Learning (SSL) performs best.

## Preprocessing

- First, run `python ../get_palat_csv.py -o palat_dicomids.csv [-d /path/to/MIMIC-CXR-JPG]` to get the file `palat_dicomids.csv`.
- Then, pretrain with SSL using the `./pretrain.sh` script. Tweak to adjust; for example, to set the pretraining mode for SSL training to imagenet or scratch.
- Then, inspect the output and select a model checkpoint. Then edit the `./compare.sh` script to finetune the model to run on the same training data.
