import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from LucasKanade import *
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=1, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("../data/girlseq.npy")
rect = [280, 152, 330, 318]

template = seq[:, :, 0]
girlseqrects_with_correction = []
P_ini = np.zeros(2)
# get the rect without template correction
girlseqrects = np.load("../data/results/girlseqrects.npy")


for i in range(1, seq.shape[2]):
    it1 = seq[:, :, i]
    p = LucasKanade(template, it1, rect, threshold=args.threshold, num_iters=args.num_iters)

    rect_new = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
    girlseqrects_with_correction.append(rect)
    
    
        
    img = np.repeat(it1[:, :, np.newaxis], 3, axis=2)
    cv2.rectangle(
			img, 
			(int(rect_new[0]), int(rect_new[1])), 
			(int(rect_new[2]), int(rect_new[3])),
			color=(255, 0, 0),
            thickness = 3
			)
    cv2.rectangle(
			img, 
			(int(round(girlseqrects[i][0])), int(round(girlseqrects[i][1]))), 
			(int(round(girlseqrects[i][2])), int(round(girlseqrects[i][3]))),
			color=(0, 0, 255),
            thickness = 3
			)
    #red is the drift 
    #blue is the corrected one
    
    # decide whether to update the template
    if (i == 1) or (np.linalg.norm(p-P_ini) < template_threshold):
        template = it1
        rect = rect_new
        P_ini = np.copy(p)
    

    cv2.imshow('image', img)
    cv2.waitKey(1)

    if i in [1, 20, 40, 60, 80]:
        success = cv2.imwrite(f'../data/results/girl_with_corr_{i}.jpg', img)
        if success:
            print(f'Image {i} successfully saved.')
        else:
            print(f'Error saving image {i}.')
        #cv2.imwrite(f"../data/results/girl_with_corr_{i}.jpg", img)

    
    

girlseqrects_with_correction = np.array(girlseqrects_with_correction)
np.save("../data/results/girlseqrects_with_correction.npy", girlseqrects_with_correction)
cv2.destroyAllWindows()
