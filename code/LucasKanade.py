import numpy as np
from scipy.interpolate import RectBivariateSpline

#change this code as much as possible

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
   
    # :param It: template image
    # :param It1: Current image
    # :param rect: Current position of the car (top left, bot right coordinates)
    # :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    # :param num_iters: number of iterations of the optimization
    # :param p0: Initial movement vector [dp_x0, dp_y0]
    # :return: p: movement vector [dp_x, dp_y]
 
	
    # Put your implementation here
    p = p0
    iterations = 0 
    delta_p = np.inf

    x1, y1, x2, y2 = rect
    
    
    interpolatorIT1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    interpolatorIT = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)

    while(iterations < int(num_iters) and np.linalg.norm(delta_p) > threshold):
        x_ = np.arange(x1, x2 + 0.1)
        y_ = np.arange(y1, y2 + 0.1)
        X, Y = np.meshgrid(x_, y_)
        x_warped = X + p[0]
        y_warped = Y + p[1]
        
        It1_warped = interpolatorIT1.ev(y_warped, x_warped)

        It_warped = interpolatorIT.ev(Y, X)

        Ix = interpolatorIT1.ev(y_warped, x_warped, dx=0, dy=1).flatten()
        Iy = interpolatorIT1.ev(y_warped, x_warped, dx=1, dy=0).flatten()
        
        dx = Ix.flatten()
        dy = Iy.flatten()

        A = np.vstack([dx, dy]).T
        error = It_warped - It1_warped
        b = error.flatten()

        delta_p = np.linalg.lstsq(A, b, rcond=None)[0]

        p += delta_p

    return p


