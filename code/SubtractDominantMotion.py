import numpy as np
from scipy.interpolate import RectBivariateSpline
from LucasKanadeAffine import LucasKanadeAffine
import scipy.ndimage.morphology as morphology

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image1.shape, dtype=bool)
    
    p = LucasKanadeAffine(image1, image2, threshold, num_iters)
    
    x1, y1, x2, y2 = 0, 0, image1.shape[1] - 1, image1.shape[0] - 1 
    
    interpolatorIT1 = RectBivariateSpline(np.arange(image2.shape[0]), np.arange(image2.shape[1]), image2)
    interpolatorIT = RectBivariateSpline(np.arange(image1.shape[0]), np.arange(image1.shape[1]), image1)
    
    x_ = np.arange(x1, x2 + 0.1)
    y_ = np.arange(y1, y2 + 0.1)
    X, Y = np.meshgrid(x_, y_)
    P_ = p.flatten() 
    x_warped = P_[0] * X + P_[1] * Y + P_[2]
    y_warped = P_[3] * X + P_[4] * Y + P_[5]
    
    valid_points = (x_warped >= x1) & (x_warped < x2) & (y_warped >= y1) & (y_warped < y2)
    #X, Y = X[valid_points], Y[valid_points]
    #x_warped, y_warped = x_warped[valid_points], y_warped[valid_points]
    
    
    It1_warped = interpolatorIT1.ev(y_warped, x_warped)
    
    It_warped =  interpolatorIT.ev(Y, X)
    
    #error = It_warped[valid_points] - It1_warped
    error = It_warped - It1_warped
    
    mask = error > tolerance
    
    kernel = np.array(([0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]))
    mask = morphology.binary_dilation(mask, structure=kernel).astype(mask.dtype)
    return mask
 
    