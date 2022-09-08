#!/bin/sh
# Install gsutil following Google's instructions then run gcloud init first
gsutil -m cp -r \
  "gs://mimic-cxr-jpg-2.0.0.physionet.org/LICENSE.txt" \
  "gs://mimic-cxr-jpg-2.0.0.physionet.org/README" \
  "gs://mimic-cxr-jpg-2.0.0.physionet.org/SHA256SUMS.txt" \
  "gs://mimic-cxr-jpg-2.0.0.physionet.org/files/" \
  "gs://mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv.gz" \
  "gs://mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-metadata.csv.gz" \
  "gs://mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-negbio.csv.gz" \
  "gs://mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-split.csv.gz" \
  "gs://mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-reports.csv.gz" \
  .
