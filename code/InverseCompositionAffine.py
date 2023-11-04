import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = M.flatten()
    x1, y1, x2, y2 = 0, 0, It.shape[1] - 1, It.shape[0] - 1 
    
    iteration = 0
    delta_p_norm = np.inf
    
    # lets define interpolation functions here
    interpolatorIT1 = RectBivariateSpline(np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    interpolatorIT = RectBivariateSpline(np.arange(It.shape[0]), np.arange(It.shape[1]), It)
    
    x_ = np.arange(x1, x2 + 0.1)
    y_ = np.arange(y1, y2 + 0.1)
    X, Y = np.meshgrid(x_, y_)
   
    It_warped =  interpolatorIT.ev(Y, X)
    Ix = interpolatorIT.ev(Y, X, dx=0, dy=1)
    Iy = interpolatorIT.ev(Y, X, dx=1, dy=0)
    
   
    dx = Ix.flatten()
    dy = Iy.flatten()
    
    #jacobian terms
    j1 = dx * X.flatten()
    j2 = dx * Y.flatten()
    j3 = dx
    j4 = dy * X.flatten()
    j5 = dy * Y.flatten()
    j6 = dy
    
    A = np.vstack([j1, j2, j3, j4, j5, j6]).T
    H = A.T @ A
    while(iteration < num_iters and delta_p_norm > threshold):

        x_warped = p[0] * X + p[1] * Y + p[2]
        y_warped = p[3] * X + p[4] * Y + p[5]
        
        # calculating these valid points may not be required, as they may cause dimensionality mismatch as A is calc outside. 
        #in my experience it works both ways
        valid_points = ((x_warped >= x1) & (x_warped <= x2) & (y_warped >= y1) & (y_warped <= y2))
        
        
        #X, Y = X[valid_points], Y[valid_points]
        #x_warped, y_warped = x_warped[valid_points], y_warped[valid_points]
        
        It1_warped = interpolatorIT1.ev(y_warped, x_warped)
        
        #error = It1_warped[valid_points] -It_warped[valid_points]
        error = It1_warped -It_warped
        b = error.flatten()
        
        valid_flat = valid_points.flatten()
        #A_ = A[np.where(valid_flat == True)]
        #A_ = A

    
        B_ = A.T @ b
        
        delta_p = np.linalg.inv(H) @ B_  # [6,6] x [6,1]
        
        delta_p_norm = np.sum(delta_p**2)
        
        #update warp
        M = np.vstack([p.reshape(2,3), [0, 0, 1]])
        
        M_delta = np.vstack([delta_p.reshape(2,3), [0, 0, 1]])
        
        M_delta[0, 0] += 1 
        M_delta[1, 1] += 1

        M = M @ np.linalg.inv(M_delta)
        
        p = M[:2, :].flatten()
    
    return M
