import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from SubtractDominantMotion import SubtractDominantMotion 
import cv2
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')

print(seq.shape)

#mask = SubtractDominantMotion(seq[:, :, 0], seq[:, :, 1], threshold, num_iters, tolerance)


for i in range(1, seq.shape[2]):
    temp = seq[:, :, i-1]
    image = seq[:, :, i]
    
    mask = SubtractDominantMotion(temp, image, threshold, num_iters, tolerance)
    
    img = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    img[:, :, 2][mask == 1] = 1
    cv2.imshow('image', img)
    cv2.waitKey(1)
    if i in [30, 60, 90, 120]:
        cv2.imwrite(f"../data/results/AerialSeq_{i}.jpg", img * 255)
        
cv2.destroyAllWindows()