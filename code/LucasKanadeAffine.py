import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()
    x1, y1, x2, y2 = 0, 0, It.shape[1] - 1, It.shape[0] - 1 # check if i should reduce this by one so 
    
    iteration = 0
    delta_p_norm = np.inf
    
    # lets define interpolation functions here
    interpolatorIT1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    interpolatorIT = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    
    while iteration < num_iters and delta_p_norm > threshold:
        x_ = np.arange(x1, x2 + 0.1)
        y_ = np.arange(y1, y2 + 0.1)
        X, Y = np.meshgrid(x_, y_)
        x_warped = p[0] * X + p[1] * Y + p[2]
        y_warped = p[3] * X + p[4] * Y + p[5]
        
        # this part was causing dimensionality error due to image flattening
        #valid_points = (x_warped >= x1) & (x_warped <= x2) & (y_warped >= y1) & (y_warped <= y2)
        #X, Y = X[valid_points], Y[valid_points]
        #x_warped, y_warped = x_warped[valid_points], y_warped[valid_points]
        
        It1_warped = interpolatorIT1.ev(y_warped, x_warped)
        
        #cannot use this function as it calc gradient at whole number data points, less accurate but faster
        #Ix, Iy = np.gradient(It1_warped) 
        
        
        #finding the gradient values
        Ix = interpolatorIT1.ev(y_warped, x_warped, dx=0, dy=1)
        Iy = interpolatorIT1.ev(y_warped, x_warped, dx=1, dy=0)
        
        It_warped =  interpolatorIT.ev(Y, X)
        
        #error = It_warped[valid_points] - It1_warped
        
        error = It_warped - It1_warped
        
        dx = Ix.flatten()
        dy = Iy.flatten()
        
        #jacobian terms
        j1 = dx * X.flatten()
        j2 = dx * Y.flatten()
        j3 = dx
        j4 = dy * X.flatten()
        j5 = dy * Y.flatten()
        j6 = dy
        
        B = error.flatten()
        A = np.vstack([j1, j2, j3, j4, j5, j6]).T
        
        delta_p, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        
        p += delta_p
        
        delta_p_norm = np.linalg.norm(delta_p)
        
        print(delta_p)
        
        iteration += 1
        
        
    M = p[:].reshape(2, 3)
    M = np.row_stack([M, np.array([0, 0, 1])]) 
    
    
    return M
