# script to upscale image manually
# python resave_jpeg.py --input_dir=../../dataset/DIV2K/DIV2K_train_LR_bicubic/X2 --output_dir=../../dataset/DIV2K/DIV2K_train_LR_bicubicLQ/X3 --quality=10
import os
import cv2 
import argparse

parser = argparse.ArgumentParser(description='Resave to JPEG script')

parser.add_argument('--input_dir', required=True, type=str,
                    help='input dir of HR images')
parser.add_argument('--output_dir', required=True, type=str,
                    help='output dir of LR iamges')
parser.add_argument('--quality', required=True, type=int, help='quality for image resaved')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir

filepath_list = os.listdir(input_dir)
for filename in filepath_list:
    filepath = os.path.join(input_dir, filename)
    im = cv2.imread(filepath)
    h, w, c = im.shape
    new_path = os.path.join(output_dir, filename[:-4] + '.jpg')
    print(new_path)
    cv2.imwrite(new_path, im, [int(cv2.IMWRITE_JPEG_QUALITY), args.quality])