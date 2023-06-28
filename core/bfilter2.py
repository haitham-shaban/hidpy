import numpy as np

def bfilter2(A, w=5, sigma=[0.3, 5]):
    # A: input image on closed interval [0,1] of size NxM
    # w: int
    # sigma: 1x2 list. The spatial-domain standard deviation is given by sigma[0]
    #  and the intensity-domain standard deviation is given by sigma[1]

    # parameters
    w = int(w)
    sigma_d = sigma[0]
    sigma_r = sigma[1]

    # Pre-compute Gaussian distance weights.
    w_vector = np.arange(-w, w+1)
    X,Y = np.meshgrid(w_vector, w_vector)
    G = np.exp(-(X**2+Y**2)/(2*sigma_d**2))
    
    # Apply bilateral filter.
    dim = A.shape
    B = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
        
            # Extract local region.
            iMin = max([i-w,0])
            iMax = min([i+w,dim[0]-1])
            jMin = max([j-w,0])
            jMax = min([j+w,dim[1]-1])
            I = A[iMin:iMax+1,jMin:jMax+1]
        
            # Compute Gaussian intensity weights.
            H = np.exp(-(I-A[i,j])**2/(2*sigma_r**2))
        
            # Calculate bilateral filter response.
            rows = np.arange(iMin,iMax+1)-i+w
            cols = np.arange(jMin,jMax+1)-j+w
            F = H * G[rows][:,cols]
            B[i,j] = sum(F.flatten() * I.flatten())/sum(F.flatten())
    return B

# if __name__ == '__main__':
#     A = np.array([
#         [4, 6, 2, 4, 5, 5, 8, 3], \
#         [3, 7, 3, 2, 4, 8, 9, 12], \
#         [6, 9, 1, 3, 6, 9, 3, 3], \
#         [8, 7, 5 ,6, 7, 8, 9, 3], \
#         [5, 0, 8, 2, 3, 5, 6, 7]
#         ])
#     normalization = max(A.flatten())
#     A = A/normalization

#     w = 1
#     sigma = [5, 5]
#     B = bfilter2(A,w,sigma)
#     print(B)