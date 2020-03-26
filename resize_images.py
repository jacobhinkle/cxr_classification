from PIL import Image
from tqdm import tqdm

import argparse
import os
import tempfile

def resize_image(inname, outname, size):
    im = Image.open(inname)
    imre = im.resize(size, 2)
    imre.save(outname)


def resize_dir(indir, outdir, size):
    """
    Load each file in the MIMIC-CXR-JPG dataset and write an identical directory
    structure containing resized JPG images.
    
    Note that this fails if outdir exists.
    """
    tmpname = 'mimic_jpg_filenames.txt'
    if os.path.isfile(tmpname):
        nfiles = 0
        with open(tmpname, 'r') as filetmp:
            for relfile in filetmp:
                # get directory name
                reldir = os.path.dirname(relfile)
                os.makedirs(os.path.join(outdir, reldir), exist_ok=True)
                nfiles += 1
    else:
        with open(tmpname, 'w') as filetmp:
            nfiles = 0
            print("Creating directories and finding JPG files...")
            os.mkdir(outdir)
            for root, dirs, files in os.walk(indir, topdown=True):
                relroot = os.path.relpath(root, indir)
                outroot = os.path.join(outdir, relroot)
                for d in dirs:
                    if not d.startswith('.'):
                        os.mkdir(os.path.join(outroot, d))
                for f in files:
                    if f.endswith('.jpg'):
                        print(os.path.join(relroot, f), file=filetmp)
                        nfiles += 1
                        if nfiles % 10000 == 0:
                            print("nfiles =", nfiles)
                filetmp.flush()

    print(f"Found {nfiles} files, listed in {filetmp.name}.")

    print("Resizing JPG files...")
    with open(tmpname,'r') as filetmp:
        for _ in tqdm(range(nfiles)):
            f = filetmp.readline().strip()
            resize_image(os.path.join(indir, f), os.path.join(outdir, f), size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i',
        default='/gpfs/alpine/proj-shared/csc378/data/MIMIC-CXR-JPG/files',
        help='Location of input JPG images',
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Location of output resized JPG images',
    )
    parser.add_argument(
        '--size', '-s',
        default='2048,2048',
        help='Comma-separated pair of pixel sizes for output images',
    )
    args = parser.parse_args()

    size = [int(s) for s in args.size.split(',')]
    if len(size) != 2:
        raise ValueError("Size argument must be pair of integers")

    resize_dir(args.input, args.output, size)
