import pandas as pd

import os

import mimic_cxr_jpg

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        "Extract PA/Lateral studies only and write out a dicom_id csv file.",
    )
    parser.add_argument(
        '--data_dir',
        default="/scratch/4jh/cxr/MIMIC-CXR-JPG",
        help="Location of top-level MIMIC-CXR-JPG directory",
    )
    parser.add_argument(
        '--dicom_id_file',
        required=True,
        help="Location to write output to.",
    )
    args = parser.parse_args()

    m = pd.read_csv(os.path.join(args.data_dir, "mimic-cxr-2.0.0-metadata.csv.gz"))

    print("Total unfiltered images:", len(m))
    print("Total unfiltered studies:", len(m.groupby('study_id')))

    # get all studies of size two, consisting of exactly one PA and one Lateral/LL

    paims = m.query('ViewPosition == "PA"')
    latims = m.query('ViewPosition == "LATERAL" | ViewPosition == "LL"')
    # intersect pastuds and latstuds, then find studies of size 2 only
    palatstuds = paims[paims.study_id.isin(latims.study_id)].study_id
    palatims = m[m.study_id.isin(palatstuds)]

    print("Images (PA, Lat, Study-based Intersection):", len(paims), len(latims), len(palatims))

    g = palatims.groupby('study_id')

    print("PA+Lat+... studies:", len(g))
    print("PA+Lat ONLY studies:", g.size().value_counts().loc[2])
    studsizes = g.size()
    pairstuds = studsizes[studsizes == 2].index

    plonlyim = palatims[palatims.study_id.isin(pairstuds)]

    print("PA+Lat images:", len(plonlyim))

    print("Writing dicom_id file to", args.dicom_id_file)
    plonlyim[['dicom_id']].to_csv(args.dicom_id_file, index=False)

