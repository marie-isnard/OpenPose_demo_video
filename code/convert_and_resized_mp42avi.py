import imageio
import cv2 as cv 
#import ntpath
import argparse
import os 

parser = argparse.ArgumentParser()
parser.add_argument('--input_file')
parser.add_argument('--output_file')

args = parser.parse_args()

src_dir = args.input_file
video_name = os.path.splitext(os.path.basename(args.input_file))[0] + ".avi"
dst_dir = os.path.join(args.output_file, video_name) 

reader = imageio.get_reader(src_dir)
fps = reader.get_meta_data()['fps']
size = reader.get_meta_data()['size']
writer = imageio.get_writer(dst_dir, fps=fps)

for im in reader:
    im = cv.resize(im, (round(size[0]/2), round(size[1]/2)))
    writer.append_data(im[:, :, :])
writer.close()
