import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


def apply_denoising(frame,normalization):

    denoised_frame=[]

    A=np.copy(frame)
    A=A/normalization

    denoised_frame=bfilter2(A)
    return denoised_frame

def apply_denoising_multiprocessing(frames,num_cores=0):   
    denoised_frames=[]
    max_values=[np.max(array) for array in frames]
    normalization = np.max(max_values)

    if num_cores == 0:
        ncores = multiprocessing.cpu_count()
        print('Using # cores:'+str(round(ncores)))
        results = Parallel(n_jobs=round(ncores))(delayed(apply_denoising)(frames[i],normalization) for i in tqdm(range(len(frames))))
    elif num_cores > 1:
        print('Using # cores:' + str(round(num_cores)))
        results = Parallel(n_jobs=round(num_cores))(delayed(apply_denoising)(frames[i],normalization) for i in tqdm(range(len(frames))))
    else:
        ### No parallel
        print('Using # cores: 1')
        for i in tqdm(range(len(frames))):
            denoised_frame = apply_denoising(frames[i],normalization)
            denoised_frames.append(denoised_frame)
                
    if num_cores>=0:
        for denoised_frame in results:
            denoised_frames.append(denoised_frame)

    normalized_frames=[frame/normalization for frame in frames]

    return denoised_frames,normalized_frames


def bfilter2(A, w=5, sigma=[5, 0.3]):
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