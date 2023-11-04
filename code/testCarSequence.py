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
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
carseqrects = []
carseqrects.append(rect)


for i in range(1, seq.shape[2]):
    template = seq[:, :, i-1]
    It1 = seq[:, :, i]
    p = LucasKanade(template, It1, rect, threshold, num_iters)
    rect = rect + np.array([p[0], p[1], p[0], p[1]])
    carseqrects.append(rect)
    img = np.repeat(It1[:, :, np.newaxis], 3, axis=2)
    #img = cv2.cvtColor(It1, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(
			img, 
			(int(rect[0]), int(rect[1])), 
			(int(rect[2]), int(rect[3])),
			color=(0, 0, 255),
            thickness = 2
			)
    cv2.imshow("car", img)
    cv2.waitKey(2)

    if i in [1, 100, 200, 300, 400]:
        
        cv2.imwrite(f"../data/results/car_{i}.jpg", img * 255)
    
cv2.destroyAllWindows()
carseqrects = np.array(carseqrects)  
np.save("../data/results/carseqrects.npy", carseqrects)
    
    


