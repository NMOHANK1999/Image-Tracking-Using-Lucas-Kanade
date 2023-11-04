
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from LucasKanade import LucasKanade
import cv2


from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=10000, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=0.5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]

result_dir = '../result/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

template = seq[:, :, 0]
carseqrects_with_correction = []
P_ini = np.zeros(2)
# get the rect without template correction
carseqrects = np.load("../data/results/carseqrects.npy")


for i in range(1, seq.shape[2]):
    it1 = seq[:, :, i]
    p = LucasKanade(template, it1, rect, threshold=args.threshold, num_iters=args.num_iters)

    rect_new = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
    carseqrects_with_correction.append(rect)
    
    # decide whether to update the template
    if (i == 1) or (np.linalg.norm(p-P_ini) < template_threshold):
        template = it1
        rect = rect_new
        P_ini = np.copy(p)
        
    img = np.repeat(it1[:, :, np.newaxis], 3, axis=2)
    img = (img * 255).astype(np.uint8)
    cv2.rectangle(
			img, 
			(int(rect_new[0]), int(rect_new[1])), 
			(int(rect_new[2]), int(rect_new[3])),
			color=(255, 0, 0),
            thickness = 3
			)
    cv2.rectangle(
			img, 
			(int(round(carseqrects[i][0])), int(round(carseqrects[i][1]))), 
			(int(round(carseqrects[i][2])), int(round(carseqrects[i][3]))),
			color=(0, 0, 255),
            thickness = 3
			)
    #red is the drift 
    #blue is the corrected one
    
    
    

    cv2.imshow('image', img)
    cv2.waitKey(1)

    if i in [1, 100, 200, 300, 400]:
        #img = (img * 255).astype(np.uint8)
        success = cv2.imwrite(f'../data/results/car_with_correction_{i}.jpg', img)
        if success:
            print(f'Image {i} successfully saved.')
        else:
            print(f'Error saving image {i}.')
        #cv2.imwrite(f"../data/results/car_with_correction_{i}.jpg", img)

    
    

carseqrects_with_correction = np.array(carseqrects_with_correction)
np.save("../data/results/carseqrects_with_correction.npy", carseqrects_with_correction)
cv2.destroyAllWindows()


#why does this method not work?
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import cv2


from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]

#this one always stays const
rect2 = [59, 116, 145, 151] 



#will need only one of the foll
P_2 = np.zeros(2)

P_last = None



carseqrects_with_correction = []
carseqrects_with_correction.append(rect)

carseqrects = np.load("../data/results/carseqrects.npy")
#this method is really effed, find another way

template = seq[:, :, 0]

for i in range(1, seq.shape[2]):

    It1 = seq[:, :, i]
    
   
    p = LucasKanade(template, It1, rect, threshold, num_iters)
    P_2 += p
    
    
    p_star = LucasKanade(seq[:, :, 0], It1, rect2, threshold, num_iters, p0 = P_2)
    
    if (np.sum(np.abs(p_star-p))) <= epsil:
        rect += np.array((p[0],p[1],p[0],p[1]))
        template = seq[:, :, i]
        
    else:
        pFromOrig = p_star + P_2
        rect = rect2 + np.array((pFromOrig[0],pFromOrig[1],pFromOrig[0],pFromOrig[1]))
        
 

    img = np.repeat(It1[:, :, np.newaxis], 3, axis=2)
    cv2.rectangle(
			img, 
			(int(rect[0]), int(rect[1])), 
			(int(rect[2]), int(rect[3])),
			color=(255, 0, 0),
            thickness = 3
			)
    cv2.rectangle(
			img, 
			(int(carseqrects[i][0]), int(carseqrects[i][1])), 
			(int(carseqrects[i][2]), int(carseqrects[i][3])),
			color=(0, 0, 255),
            thickness = 3
			)
    #red is the drift 
    #blue is the corrected one
    cv2.imshow("car", img)
    cv2.waitKey(2)

    if i in [1, 100, 200, 300, 400]:
        cv2.imwrite(f"../data/results/car_with_template_correction_{i}.jpg", img * 255)
    
cv2.destroyAllWindows()
carseqrects_with_correction = np.array(carseqrects_with_correction)  
np.save("../data/results/carseqrects_with_correction.npy", carseqrects_with_correction)
"""