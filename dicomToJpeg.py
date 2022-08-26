import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2
import os

def dicomtojpg(source_folder, output_folder):
    list_of_files = os.listdir(source_folder)
    for file in list_of_files:
        try:
            ds = pydicom.dcmread(os.path.join(source_folder,file))

            # Convert to float to avoid overflow or underflow losses.
            image_2d = ds.pixel_array.astype(float)

            # Rescaling grey scale between 0-255
            image_2d_scaled = ((image_2d - image_2d.min()) / image_2d.max()) * 255.0

            # Convert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)

            # check Photometric Interpretation
            pi = ds[0x0028, 0x0004] #Photometric Interpretation

            if pi.value == 'MONOCHROME1':
                #invert
                image = (255-image_2d_scaled)
            else:
                image = image_2d_scaled

            # Use histogram equalization
            hist_image = cv2.equalizeHist(image)

            # convert to jpg
            cv2.imwrite(os.path.join(output_folder,os.path.splitext(file)[0])+'.jpg',hist_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        except:
            print('Could not convert: ', file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_folder", "-s", required="True", help="dicom source filepath")
    parser.add_argument("--output_folder","-o",required="True",help="jpg output filepath")
    args = parser.parse_args()

    source_folder = args.source_folder
    output_folder = args.output_folder

    _ = dicomtojpg(source_folder, output_folder)
