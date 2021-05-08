# script to upscale image manually
import os
import cv2 
import argparse
 
parser = argparse.ArgumentParser(description='Upscale Image script')
 
parser.add_argument('--input_dir', required=True, type=str,
                    help='input dir')
parser.add_argument('--output_dir', required=True, type=str,
                    help='output dir')
parser.add_argument('--target_dir', required=True, type=str,
                    help='target dir')
parser.add_argument('--scale', required=True, type=int, help='scale factor')
 
args = parser.parse_args()
 
input_dir = args.input_dir
output_dir = args.output_dir
 
filepath_list = os.listdir(input_dir)
for filename in filepath_list:
    filepath = os.path.join(input_dir, filename)
    splitted_filename = filename.split('.')
    target_filepath = os.path.join(args.target_dir, splitted_filename[0][:-2] + '.' + 'png')
    im = cv2.imread(filepath)
    target_im = cv2.imread(target_filepath)
    h, w, c = im.shape
    tar_h, tar_w, tar_c = target_im.shape
    resized_im = cv2.resize(im, (tar_w, tar_h), cv2.INTER_LINEAR)
    print(os.path.join(output_dir, filename))
    cv2.imwrite(os.path.join(output_dir, filename), resized_im)