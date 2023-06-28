from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

import numpy as np
import numpy.matlib
import scipy.ndimage
import scipy.signal
import copy
import cv2
from skimage import transform
from scipy.sparse import csr_matrix
import re
from tqdm import trange
from matplotlib import pyplot as plt
import cProfile
import pstats

import warnings

# 'classic+nl-fast'
class classic_nl_optical_flow:
    def __init__(self):
        self.images = []
        self.lambda_value = 1
        self.lambda_q = 1
        self.lambda2 = 1e-1
        self.lambda3 = 1
        self.sor_max_iters = 1e4
        self.limit_update = True
        self.display = False
        self.solver = 'backslash'
        self.deriv_filter = np.expand_dims(np.array([1, -8, 0, 8, -1]) / 12, axis=1).T
        self.texture = False
        self.fc = False
        self.median_filter_size = []
        self.interpolation_method = 'bi-cubic'
        self.gnc_iters = 3
        self.alpha = 1
        self.max_iters = 10
        self.max_linear = 1
        self.pyramid_levels = 1 # 4 profile with fewer pyramid levels
        self.pyramid_spacing = 2
        self.gnc_pyramid_levels = 2
        self.gnc_pyramid_spacing = 1.25
        method = 'generalized_charbonnier'
        self.spatial_filters = [np.expand_dims(np.array([1, -1]), axis=1), \
                                np.expand_dims(np.array([1, -1]), axis=1).T]
        self.spatial_filters = [[1, -1], [[1], [-1]]]
        a = 0.45
        sig = 1e-3
        self.rho_spatial_u = []
        self.rho_spatial_v = []
        for i in range(len(self.spatial_filters)):
            self.rho_spatial_u.append( robust_function(method, sig, a) )
            self.rho_spatial_v.append( robust_function(method, sig, a) )

        self.rho_data = robust_function(method, sig, a)
        self.seg = []
        self.mfT = 15
        self.imfsz = [7, 7]
        self.filter_weight = []
        self.alp = 0.95
        self.hybrid = False
        self.area_hsz = 10
        self.affine_hsz = 4
        self.sigma_i = 7
        self.color_images = []
        self.auto_level = True
        self.input_seg = []
        self.input_occ = []
        self.fullVersion = False

        # classic+nl params
        self.texture  = True
        self.median_filter_size   = [5, 5]
        
        self.alp = 0.95
        self.area_hsz = 7
        self.sigma_i  = 7
        self.color_images = np.ones((1,1,3))
        
        self.lambdaa = 3
        self.lambda_q =3

        # classic+nl-fast params
        self.max_iters       = 3
        self.gnc_iters       = 2
        self.display  = True


def robust_function(method, arg2, arg3):
    if method == 'generalized_charbonnier':
        output = {
            'param': [arg2, arg3],
            'type': lambda x, arg2, arg3: generalized_charbonnier(x, arg2, arg3)
        }
    elif method == 'quadratic':
        output = {
            'param': [arg2],
            'type': lambda x, arg2, arg3: quadratic(x, arg2, arg3)
        }
    return output

def generalized_charbonnier(x, sigma, qtype):
    
    sig  = sigma[0]
    a    = sigma[1]

    if qtype == 0:
        y = (sig**2 + x**2)**a
    elif qtype == 1:
        y = 2*a*x*(sig**2 + x**2)**(a-1)
    elif qtype == 2:
        y = 2*a*(sig**2 + x**2)**(a-1)
    return y

def quadratic(x, sigma, qtype):
    if qtype == 0:
        y = x**2 / sigma**2
    elif qtype == 1:
        y = 2 * x / sigma**2
    elif qtype == 2:
        y = numpy.matlib.repmat(2 / sigma**2, x.shape[0], x.shape[1])
    return y

def scale_image(im, vlow, vhigh):
    ilow    = np.min(im)
    ihigh   = np.max(im)
    return (im-ilow)/(ihigh-ilow) * (vhigh-vlow) + vlow


def structure_texture_decomposition_rof(im, theta, nIters, alp):
    # Rescale the input image to [-1 1]
    IM   = scale_image(im, -1, 1)

    # Backup orginal images
    im   = copy.copy(IM )

    # stepsize
    delta = 1.0/(4.0*theta)

    for iIm in range(im.shape[2]):
        # Initialize dual variable p to be 0
        p = np.zeros((im.shape[0], im.shape[1], 2))
        
        # Gradient descend        
        I = IM[:,:,iIm]
        
        for iter in range(nIters):
            
            # Compute divergence        eqn(8)                    
            div_p = scipy.signal.correlate2d(p[:,:,0], np.array([[-1], [1], [0]]).T, boundary='fill', mode='same') + \
                    scipy.signal.correlate2d(p[:,:,1], np.array([[-1], [1], [0]]),   boundary='fill', mode='same')
            
            I_x = scipy.signal.correlate2d(I+theta*div_p, np.array([[-1,1]]), boundary='symm', mode='same')
            I_y = scipy.signal.correlate2d(I+theta*div_p, np.array([[-1,1]]).T, boundary='symm', mode='same')
            
            
            # Update dual variable      eqn(9)
            p[:,:,0] += delta*I_x
            p[:,:,1] += delta*I_y
            
            # Reproject to |p| <= 1     eqn(10)    
            reprojection = np.sqrt(p[:,:,0]**2 + p[:,:,1]**2)
            reprojection[reprojection<1] = 1
            p[:,:,0] /= reprojection
            p[:,:,1] /= reprojection

        # compute divergence    
        div_p = scipy.signal.correlate2d(p[:,:,0], np.array([[-1], [1], [0]]).T, boundary='fill', mode='same') + \
                scipy.signal.correlate2d(p[:,:,1], np.array([[-1], [1], [0]]),   boundary='fill', mode='same')
        
        # compute structure component
        IM[:,:,iIm] = I + theta * div_p

    texture = scale_image(im - alp*IM, 0, 255)
    structure = scale_image(IM, 0, 255)
    return texture, structure

def gaussian_kernel(p2, p3):
    
    p2 = np.array([p2, p2])
    siz   = (p2-1)/2
    std   = p3
    
    x, y = np.meshgrid(*(np.arange(-siz[1],siz[1]+1),np.arange(-siz[0],siz[0]+1)))
    arg   = -(x*x + y*y)/(2*std*std)
    
    h     = np.exp(arg)
    h[h<2.2e-16*np.max(h)] = 0
    
    sumh = np.sum(h)
    if sumh != 0:
        h  /= sumh
    return h

def compute_image_pyramid(img, f, nL, ratio):

    P   = [] * nL
    tmp = copy.copy(img)
    P.append( tmp )

    for m in range(1, nL):

        if len(tmp.shape) > 2:
            tmp[:,:,0] = scipy.signal.correlate2d(tmp[:,:,0], f, boundary='symm', mode='same')
            tmp[:,:,1] = scipy.signal.correlate2d(tmp[:,:,1], f, boundary='symm', mode='same')
        else:
            tmp = scipy.signal.correlate2d(tmp, f, boundary='symm', mode='same')
        sz = np.round([ratio*tmp.shape[0], ratio*tmp.shape[1]]).astype(int)

        tmp = transform.resize(tmp, (sz[0], sz[1]), anti_aliasing=False)
        
        P.append( tmp )
    return P

# Interpolation kernel
def kernel(s, a):
    if (abs(s) >= 0) & (abs(s) <= 1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    return 0
  
  
# Padding
def padding(img, H, W):
    zimg = np.zeros((H+4, W+4))
    zimg[2:H+2, 2:W+2] = img
      
    # Pad the first/last two col and row
    zimg[2:H+2, 0:2] = img[:, 0:1]
    zimg[H+2:H+4, 2:W+2] = img[H-1:H, :]
    zimg[2:H+2, W+2:W+4] = img[:, W-1:W]
    zimg[0:2, 2:W+2] = img[0:1, :]
      
    # Pad the missing eight points
    zimg[0:2, 0:2] = img[0, 0]
    zimg[H+2:H+4, 0:2] = img[H-1, 0]
    zimg[H+2:H+4, W+2:W+4] = img[H-1, W-1]
    zimg[0:2, W+2:W+4] = img[0, W-1]
    return zimg
  
  
# Bicubic operation
def bicubic(img, ratio, a, sz):
    
    # Get image size
    H, W = img.shape
      
    # Here H = Height, W = weight,
    # C = Number of channels if the 
    # image is coloured.
    img = padding(img, H, W)
      
    # Create new image
    dH = int( np.floor(H*ratio) )
    dW = int( np.floor(W*ratio) )
  
    # Converting into matrix
    dst = np.zeros((dH, dW))  
    # np.zeroes generates a matrix 
    # consisting only of zeroes
    # Here we initialize our answer 
    # (dst) as zero
  
    h = 1/ratio
  
    # print('Start bicubic interpolation')
    # print('It will take a little while...')
    inc = 0
      
    for j in range(dH):
        for i in range(dW):
            
            # Getting the coordinates of the
            # nearby values
            x, y = i * h + 2, j * h + 2

            x1 = int( 1 + x - np.floor(x) )
            x2 = int( x - np.floor(x) )
            x3 = int( np.floor(x) + 1 - x )
            x4 = int( np.floor(x) + 2 - x )

            y1 = int( 1 + y - np.floor(y) )
            y2 = int( y - np.floor(y) )
            y3 = int( np.floor(y) + 1 - y )
            y4 = int( np.floor(y) + 2 - y )
                
            # Considering all nearby 16 values
            mat_l = np.matrix([[kernel(x1, a), kernel(x2, a), kernel(x3, a), kernel(x4, a)]])
            mat_m = np.matrix([[img[int(y-y1), int(x-x1)],
                                img[int(y-y2), int(x-x1)],
                                img[int(y+y3), int(x-x1)],
                                img[int(y+y4), int(x-x1)]],
                                [img[int(y-y1), int(x-x2)],
                                img[int(y-y2), int(x-x2)],
                                img[int(y+y3), int(x-x2)],
                                img[int(y+y4), int(x-x2)]],
                                [img[int(y-y1), int(x+x3)],
                                img[int(y-y2), int(x+x3)],
                                img[int(y+y3), int(x+x3)],
                                img[int(y+y4), int(x+x3)]],
                                [img[int(y-y1), int(x+x4)],
                                img[int(y-y2), int(x+x4)],
                                img[int(y+y3), int(x+x4)],
                                img[int(y+y4), int(x+x4)]]])
            mat_r = np.matrix(
                [[kernel(y1, a)], [kernel(y2, a)], [kernel(y3, a)], [kernel(y4, a)]])
                
            # Here the dot function is used to get 
            # the dot product of 2 matrices
            dst[j, i] = np.dot(np.dot(mat_l, mat_m), mat_r)
  
    # # If there is an error message, it
    # # directly goes to stderr
    # sys.stderr.write('\n')
      
    # # Flushing the buffer
    # sys.stderr.flush()
    dst_sized = np.zeros(sz)
    dst_sized[:dH, :dW] = dst
    return dst_sized


def resample_flow(uv, sz):
    ratio0 = sz[0] / uv.shape[0]
    ratio1 = sz[1] / uv.shape[1]
    ratio = np.min([ratio0, ratio1])
    # a = -0.5
    # u = bicubic(uv[:,:,0], ratio, a, sz)
    # v = bicubic(uv[:,:,1], ratio, a, sz)
    u = scipy.ndimage.interpolation.zoom(uv[:,:,0], ratio, mode='reflect')
    v = scipy.ndimage.interpolation.zoom(uv[:,:,1], ratio, mode='reflect')
    u_sized = np.zeros(sz)
    v_sized = np.zeros(sz)
    u_sized[:u.shape[0], :u.shape[1]] = u
    v_sized[:v.shape[0], :v.shape[1]] = v
    # u     = transform.resize(uv[:,:,0], sz, anti_aliasing=False)*ratio
    # v     = transform.resize(uv[:,:,1], sz, anti_aliasing=False)*ratio
    out   = np.concatenate((np.expand_dims(u_sized, axis=2), np.expand_dims(v_sized, axis=2)), axis=2)
    return out

def interp2_bicubic(Z, XI, YI, Dxfilter):

    Dyfilter = Dxfilter.T
    Dxyfilter = scipy.signal.convolve2d(Dxfilter, Dyfilter, mode='full')
    
    input_size = XI.shape
    
    # Reshape input coordinates into a vector
    XI = np.reshape(XI.T, (1, np.prod(input_size)))
    YI = np.reshape(YI.T, (1, np.prod(input_size)))
    
    # Bound coordinates to valid region
    sx = Z.shape[1]
    sy = Z.shape[0]

    fXI = np.floor(XI)
    cXI = fXI + 1
    fYI = np.floor(YI)
    cYI = fYI + 1

    indx = (fXI<0) | (cXI>sx-1) | (fYI<0) | (cYI>sy-1)
  
    fXI[fXI>sx-1] = sx-1
    cXI[cXI>sx-1] = sx-1
    fYI[fYI>sy-1] = sy-1
    cYI[cYI>sy-1] = sy-1
    fXI[fXI<0] = 0
    cXI[cXI<0] = 0
    fYI[fYI<0] = 0
    cYI[cYI<0] = 0

    # Image at 4 neighbors
    Z00 = Z[fYI.astype(int), fXI.astype(int)]
    Z01 = Z[cYI.astype(int), fXI.astype(int)] 
    Z10 = Z[fYI.astype(int), cXI.astype(int)] 
    Z11 = Z[cYI.astype(int), cXI.astype(int)] 

    # x-derivative at 4 neighbors
    DX = scipy.signal.correlate2d(Z, Dxfilter, boundary='symm', mode='same')
    DX00 = DX[fYI.astype(int), fXI.astype(int)]
    DX01 = DX[cYI.astype(int), fXI.astype(int)] 
    DX10 = DX[fYI.astype(int), cXI.astype(int)] 
    DX11 = DX[cYI.astype(int), cXI.astype(int)] 

    # y-derivative at 4 neighbors
    DY = scipy.signal.correlate2d(Z, Dyfilter, boundary='symm', mode='same')
    DY00 = DY[fYI.astype(int), fXI.astype(int)]
    DY01 = DY[cYI.astype(int), fXI.astype(int)] 
    DY10 = DY[fYI.astype(int), cXI.astype(int)] 
    DY11 = DY[cYI.astype(int), cXI.astype(int)] 

    # xy-derivative at 4 neighbors
    DXY = scipy.signal.correlate2d(Z, Dxyfilter, boundary='symm', mode='same')
    DXY00 = DXY[fYI.astype(int), fXI.astype(int)]
    DXY01 = DXY[cYI.astype(int), fXI.astype(int)] 
    DXY10 = DXY[fYI.astype(int), cXI.astype(int)] 
    DXY11 = DXY[cYI.astype(int), cXI.astype(int)] 


    W = np.array([ [1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
       [-3,  0,  0,  3,  0,  0,  0,  0, -2,  0,  0, -1,  0,  0,  0,  0],  
        [2,  0,  0, -2,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0],
        [0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
        [0,  0,  0,  0, -3,  0,  0,  3,  0,  0,  0,  0, -2,  0,  0, -1],
        [0,  0,  0,  0,  2,  0,  0, -2,  0,  0,  0,  0,  1,  0,  0,  1],
       [-3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0],
        [9, -9,  9, -9,  6,  3, -3, -6,  6, -6, -3,  3,  4,  2,  1,  2],
       [-6,  6, -6,  6, -4, -2,  2,  4, -3,  3,  3, -3, -2, -1, -1, -2],
        [2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        [0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0],
       [-6,  6, -6,  6, -3, -3,  3,  3, -4,  4,  2, -2, -2, -2, -1, -1],
        [4, -4,  4, -4,  2,  2, -2, -2,  2, -2, -2,  2,  1,  1,  1,  1] ])
    
    V = np.vstack((Z00, Z10, Z11, Z01, DX00, DX10, DX11, DX01, DY00, DY10, DY11, DY01, DXY00, DXY10, DXY11, DXY01))

    C = np.matmul(W, V)

    alpha_x = np.reshape(XI - fXI, [input_size[1],input_size[0]] ).T
    alpha_y = np.reshape(YI - fYI, [input_size[1],input_size[0]] ).T
  
    # Clip out-of-boundary pixels to boundary
    alpha_x[np.reshape(indx, [input_size[1],input_size[0]] ).T] = 0
    alpha_y[np.reshape(indx, [input_size[1],input_size[0]] ).T] = 0

    
    fXI = np.reshape(fXI, [input_size[1],input_size[0]] ).T
    fYI = np.reshape(fYI, [input_size[1],input_size[0]] ).T

    # Interpolation

    ZI  = np.zeros(input_size)
    ZXI = np.zeros(input_size)
    ZYI = np.zeros(input_size)
    
    idx = 0
    for i in range(4):
        for j in range(4):
            ZI = ZI + np.reshape(C[idx, :], [input_size[1],input_size[0]] ).T * alpha_x**i * alpha_y**j
            if (i > 0):
                ZXI = ZXI + i * np.reshape(C[idx, :], [input_size[1],input_size[0]] ).T * alpha_x**(i-1) * alpha_y**j
            if (j > 0):
                ZYI = ZYI + j * np.reshape(C[idx, :], [input_size[1],input_size[0]] ).T * alpha_x**i * alpha_y**(j-1)
     
            idx += 1
    ZI[np.reshape(indx, [input_size[1],input_size[0]]).T] = np.nan

    return ZI, ZXI, ZYI#, C, alpha_x, alpha_y, fXI, fYI

def partial_deriv(images, uv_prev, interpolation_method, deriv_filter=np.expand_dims(np.array([1, -8, 0, 8, -1])/12, axis=1), b=0.5):
    h = copy.copy(deriv_filter)
    H = images.shape[0]
    W = images.shape[1]

    
    x,y   = np.meshgrid(np.arange(W), np.arange(H))
    x2      = x + uv_prev[:,:,0] 
    y2      = y + uv_prev[:,:,1]

    # Record out of boundary pixels
    B = (x2>=W) | (x2<0) | (y2>=H) | (y2<0)

    img1 = images[:,:,0]
    img2 = images[:,:,1]

    warpIm, Ix, Iy = interp2_bicubic(images[:,:,1], x2, y2, h)
    indx        = np.isnan(warpIm)
    It          = warpIm - images[:,:,0]

    # Disable those out-of-boundary pixels in warping
    It[indx]    = 0
    
    # Temporal average
    I1x = scipy.signal.correlate2d(img1, h,   boundary='symm', mode='same')
    I1y = scipy.signal.correlate2d(img1, h.T, boundary='symm', mode='same')
    
    Ix  = b*Ix+(1-b)*I1x
    Iy  = b*Iy+(1-b)*I1y

    Ix[indx] = 0
    Iy[indx] = 0

    return It, Ix, Iy

def compute_flow_base(this, uv):
    # Construct quadratic formulation
    qua_this          = this
    qua_this.lambdaa   = this.lambda_q

    dummy = 0
    for i in range(len(this.rho_spatial_u)):
          a = this.rho_spatial_u[i]['param']
          qua_this.rho_spatial_u[i]   = robust_function('quadratic', a[0], dummy)
          a = this.rho_spatial_u[i]['param']
          qua_this.rho_spatial_v[i]   = robust_function('quadratic', a[0], dummy)

    a = this.rho_data['param']
    qua_this.rho_data = robust_function('quadratic', a[0], dummy)

    for i in range(this.max_iters):
        # print('max iters: '+str(i))
        duv = np.zeros(uv.shape)
            
        # Compute spatial and temporal partial derivatives
        [It, Ix, Iy] = partial_deriv(this.images, uv, this.interpolation_method, this.deriv_filter)
        ## looks ok
        for j in range(this.max_linear):
            # print('max linear: '+str(j))
            [A, b, parm, iterative] = flow_operator(qua_this, uv, duv, It, Ix, Iy)
            ## looks ok
            # x = np.linalg.lstsq(A, b)[0]
            x = scipy.sparse.linalg.spsolve(A,b)
            # x = np.reshape(np.array(x).T, [uv.shape[0], uv.shape[1], uv.shape[2]])
            out = np.zeros(uv.shape)
            halfWay = int(len(x)/2)
            out[:,:,0] = np.reshape(np.array(x)[:halfWay], [uv.shape[1], uv.shape[0]]).T
            out[:,:,1] = np.reshape(np.array(x)[halfWay:], [uv.shape[1], uv.shape[0]]).T
            x = out

            # If limiting the incremental flow to [-1, 1] is requested, do so
            if this.limit_update:
                x[x > 1]  = 1
                x[x < -1] = -1

            duv = copy.copy(x)
            uv0 = copy.copy(uv)
            uv  += duv

            
            if len(this.median_filter_size) > 0:
                
                # Compute weighted median solved by Li & Osher formula
                occ = detect_occlusion(uv, this.images)
                uv = denoise_color_weighted_medfilt2(uv, this.color_images, occ, this.area_hsz, this.median_filter_size, this.sigma_i, this.fullVersion)
            
            
            duv = uv - uv0
            uv  = copy.copy(uv0)
            
        # Update flow fileds
        uv += duv
    
    return uv

def sobel_edge(arr):
    # arr = np.sum(cv2.imread("pic.png").astype(float), axis=2)
    filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ex = scipy.signal.convolve2d(arr, filter, mode='same')
    ey = scipy.signal.convolve2d(arr, np.flip(filter.T, axis=0), mode='same')
    magnitude = ex**2 + ey**2

    scale = 4 # for calculating the automatic threshold
    cutoff = scale * np.sum(magnitude) / np.prod(magnitude.shape)
    # thresh = np.sqrt(cutoff)

    edge = np.zeros(magnitude.shape, dtype=bool)
    edge[magnitude>cutoff] = True
    return edge

def denoise_color_weighted_medfilt2(uv, im, occ, bfhsz, mfsz, sigma_i, fullVersion):
    sigma_x = 7;   #  spatial distance (7)

    uvo = copy.copy(uv)
    uvo[:,:,0] = scipy.signal.medfilt2d(uv[:,:,0], kernel_size=mfsz)
    uvo[:,:,1] = scipy.signal.medfilt2d(uv[:,:,1], kernel_size=mfsz)

    dilate_sz = 5*np.array([1, 1])  # dilation window size for flow edge region [5 5]

    sz = im.shape
    sz = sz[:2]
    
    e1 = sobel_edge(uv[:,:,0])
    e2 = sobel_edge(uv[:,:,1])
    e = e1 | e2
    
    mask = cv2.dilate(e.astype(float), np.ones(dilate_sz) )
    ### EDIT
    # mask = np.ones(mask.shape)
    # mask[1:6, 11:13] = 1
    # mask[9:19, 9:19] = 1

    indx = np.nonzero(mask.T)#np.argwhere(mask ==1)
    indx_row, indx_col = indx[1], indx[0]#np.unravel_index(indx, mask.shape)
    indx_row += 1 # to equalize to matlab
    indx_col += 1 # to equalize to matlab

    pad_u  = np.pad(uv[:,:,0], bfhsz*np.array([1, 1]), mode='symmetric')
    pad_v  = np.pad(uv[:,:,1], bfhsz*np.array([1, 1]), mode='symmetric')
    if len(im.shape)>2: 
        pad_im = np.concatenate(\
            ( np.pad(im[:,:,0], bfhsz*np.array([1, 1]), mode='symmetric'), \
            np.pad(im[:,:,0], bfhsz*np.array([1, 1]), mode='symmetric') ), axis=2) 
    else:
        pad_im = np.pad(im, bfhsz*np.array([1, 1]), mode='symmetric')
    pad_occ= np.pad(occ, bfhsz*np.array([1, 1]), mode='symmetric')

    H, W = pad_u.shape

    # Divide into several groups for memory reasons ~70,000 causes out of memory
    Indx_Row = indx_row
    Indx_Col = indx_col
    N        = len(Indx_Row) # number of elements to process
    n        = 4e4          # number of elements per batch
    nB       = int( np.ceil(N/n) )

    for ib in range(1,nB+1):
        istart = int( (ib-1)*n + 1 ) - 1
        iend   = int( np.min([ib*n, N]) )
        indx_row = Indx_Row[istart:iend]
        indx_col = Indx_Col[istart:iend]

        C, R = np.meshgrid(np.arange(-bfhsz,bfhsz+1), np.arange(-bfhsz,bfhsz+1))
        nindx = R + C*H
        cindx = indx_row + bfhsz  + (indx_col+bfhsz-1)* H

        # cindx = cindx.flatten() # reorder cindx
        # cindx = np.hstack((cindx[::2], cindx[1::2]))
        pad_indx = numpy.matlib.repmat(np.expand_dims(nindx.T.flatten(), axis=1), 1, len(indx_row)) + \
               numpy.matlib.repmat(np.expand_dims(cindx.flatten(), axis=1).T, (bfhsz*2+1)**2, 1 )

        # spatial weight
        tmp = np.exp(- (C**2 + R**2) /2/sigma_x**2 )
        weights = numpy.matlib.repmat(np.expand_dims(tmp.T.flatten(), axis=1), 1, len(indx_row))

        tmp_w = np.zeros(weights.shape)
    
        if len(pad_im.shape) > 2:
            iterations = pad_im.shape[2]
        else:
            iterations = 1
        for i in range(iterations):
            if len(pad_im.shape) > 2:
                tmp = pad_im[:,:,i].T.flatten()
            else:
                tmp = pad_im.T.flatten()
            tmp_w += (tmp[pad_indx-1] - numpy.matlib.repmat(np.expand_dims(tmp[cindx-1], axis=1).T, (bfhsz*2+1)**2, 1)) ** 2

        if len(pad_im.shape) > 2:
            divider = pad_im.shape[2]
        else:
            divider = 1
        tmp_w /= divider
        
        weights = weights * np.exp(-tmp_w/2/sigma_i**2)

        # Occlusion weight    
        weights = weights * pad_occ.T.flatten()[pad_indx-1]

        # Normalize
        weights = weights/np.matlib.repmat(np.sum(weights, axis=0), (bfhsz*2+1)**2, 1)
        
        neighbors_u = pad_u.T.flatten()[pad_indx-1]
        neighbors_v = pad_v.T.flatten()[pad_indx-1]
    
        # solve weighted median filtering
        # indx = np.unravel_index((indx_row, indx_col), sz)#sub2ind(sz, indx_row, indx_col)
        uo   = uvo[:,:,0]
        u    = weighted_median(weights, neighbors_u)
        uo[indx_row-1, indx_col-1] = u
        vo   = uvo[:,:,1]
        v    = weighted_median(weights, neighbors_v)
        vo[indx_row-1, indx_col-1] = v
        uvo[:,:,0] = uo
        uvo[:,:,1] = vo

    return uvo

def weighted_median(w, u):
    H, W = u.shape
    ir = np.argsort(u, axis=0)
    sort_u = np.sort(u, axis=0)
    # sort_u = np.concatenate([np.expand_dims(u[i,:][ir[i,:]], axis=1).T for i in range(H)], axis=1)
    ic       = numpy.matlib.repmat(np.arange(W), H, 1)
    # ind      = np.unravel_index((ir, ic), [H, W])#sub2ind([H W], ir, ic);

    w        = w[ir, ic]

    # number of negative infinites for each column
    k = np.ones(W)
    pp = -sum(w)
    for i in range(H-1,-1,-1):
        
        pc = pp + 2*w[i, :]
        indx = (pc > 0) & (pp < 0)
        k[indx] = H-i+1
        pp = pc
    
    k   = H-k+1
    # ind = sub2ind([H W], k, 1:W);

    uo  = sort_u[k.astype(int), np.arange(W)]
    return uo

def detect_occlusion(uv, images, sigma_d=0.3, sigma_i=20):
    
    a, px = np.gradient(uv[:,:,0])
    qy, b = np.gradient(uv[:,:,1])
    div = px + qy

    div[div>0] = 0

    It, _, _ = partial_deriv(images, uv, '')
    occ = np.exp(-div**2/2/sigma_d**2)*np.exp(-It**2/2/ sigma_i**2)
    return occ

def flow_operator(this, uv, duv, It, Ix, Iy):
    sz = [Ix.shape[0], Ix.shape[1]]
    npixels = np.prod(sz)

    # spatial term
    S = this.spatial_filters
    FU = csr_matrix((npixels, npixels))
    FV = csr_matrix((npixels, npixels))

    for i in range(len(S)):
        if i == 0:
            FMi = make_convn_mat(np.expand_dims(S[i], axis=0), sz, 'valid', 'sameswap')
        else:
            FMi = make_convn_mat(np.array(S[i]), sz, 'valid', 'sameswap')
        Fi = FMi.T

        # Use flow increment to update the nonlinearity
        u_        = FMi * np.reshape(uv[:,:,0]+duv[:,:,0], [npixels, 1]) 
        v_        = FMi * np.reshape(uv[:,:,1]+duv[:,:,1], [npixels, 1])

        pp_su     = deriv_over_x(this.rho_spatial_u[i], u_)
        pp_sv     = deriv_over_x(this.rho_spatial_v[i], v_)    

        # FU        = FU + np.matmul( np.matmul( Fi.todense(), np.diagflat(pp_su.flatten())), FMi.todense())
        FU        = FU + Fi @ scipy.sparse.diags(pp_su.flatten()) @ FMi
        # (Fi.multiply(scipy.sparse.spdiags(pp_su.flatten(), 0, m=npixels, n=npixels))).multiply(FMi)
        FV        = FV + Fi @ scipy.sparse.diags(pp_sv.flatten()) @ FMi
        # FV + np.matmul( np.matmul( Fi.todense(), np.diagflat(pp_sv.flatten())), FMi.todense())
        # (Fi.multiply(scipy.sparse.spdiags(pp_sv.flatten(), 0, m=npixels, n=npixels))).multiply(FMi)

    M1 = scipy.sparse.vstack((-FU, csr_matrix((npixels, npixels))))
    M2 = scipy.sparse.vstack((csr_matrix((npixels, npixels)), -FV))
    M  = scipy.sparse.hstack((M1, M2))

    # Data term   
    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix*Iy
    Itx = It*Ix
    Ity = It*Iy

    # Perform linearization - note the change in It
    if len(It.shape)>2:
        It += Ix * np.repeat(duv[:,:,0], It.shape[2], axis=2) + Iy * np.repeat(duv[:,:,1], It.shape[2], axis=2)
    else:
        It += Ix * duv[:,:,0] + Iy * duv[:,:,1]
    pp_d  = deriv_over_x(this.rho_data, np.expand_dims(It.T.flatten(), axis=1)) 

    
    tmp = pp_d * np.expand_dims(Ix2.T.flatten(), axis=1)
    duu = scipy.sparse.spdiags(tmp.flatten(), 0, m=npixels, n=npixels)
    tmp = pp_d * np.expand_dims(Iy2.T.flatten(), axis=1)
    dvv = scipy.sparse.spdiags(tmp.flatten(), 0, m=npixels, n=npixels)
    tmp = pp_d * np.expand_dims(Ixy.T.flatten(), axis=1)
    dduv = scipy.sparse.spdiags(tmp.flatten(), 0, m=npixels, n=npixels)

    A = scipy.sparse.hstack((scipy.sparse.vstack((duu, dduv)), scipy.sparse.vstack((dduv, dvv))))  - this.lambdaa*M

    # right hand side
    minusPart = np.vstack( ( pp_d * np.expand_dims(Itx.T.flatten(), axis=1), pp_d * np.expand_dims(Ity.T.flatten(), axis=1) ) )
    # b =  this.lambdaa * np.matmul( M.todense(), np.expand_dims(uv.T.flatten(), axis=1) ) - minusPart
    b =  this.lambdaa * M @ uv.T.flatten() - minusPart.flatten()

    # No auxiliary parameters
    params    = []
    
    # If the non-linear weights are non-uniform, do more linearization
    if (np.max(pp_su) - np.min(pp_su) < 1E-6 and  np.max(pp_sv) - np.min(pp_sv) < 1E-6 and  np.max(pp_d) - np.min(pp_d) < 1E-6):
        iterative = False
    else:
        iterative = True

    return A, b, params, iterative

def deriv_over_x(this, x):
    # from inspect import signature
    # sig = signature(this['type'])
    numParams = len(this['param'])
    if numParams == 1:
        return this['type'](x, this['param'][0], 2)
    elif numParams == 2:
        return this['type'](x, this['param'][0], this['param'][1], 2)
    else:
        alksjdhasjd    


def apply_argstring(valid, sub, fill_value):
    arg_string = ''.join([str(x) for x in sub if len(x)>0])
    if arg_string == ':':
        valid[:] = fill_value
    else:
        start = re.sub("[^0-9]", "", arg_string[:arg_string.find(':')])
        end   = re.sub("[^0-9]", "", arg_string[arg_string.find(':')+1:])
        if len(start)>0 and len(end)>0:
            valid[int(start):int(end)] = fill_value
        elif len(start)>0 and len(end)==0:
            valid[int(start):] = fill_value
        elif len(start)==0 and len(end)>0:
            valid[:int(end)] = fill_value

    return valid

def make_convn_mat(F, sz, shape, pad):
    ndims = len(sz)

    # Border sizes for 'same' and 'sameswap'
    Fsize_lo_2 = np.ceil((np.array(F.shape) - 1) / 2)
    Fsize_hi_2 = np.floor((np.array(F.shape) - 1) / 2)

    # Border sizes for 'valid'
    Fsize = np.array(F.shape)-1

    # case 'valid'
    valid = np.ones(sz+np.array(F.shape)-1, dtype=bool)
    
    sub = [0] * ndims
    for d in range(ndims):
        for e in range(ndims):
            sub[e] = ':'

       
        sub[d] = np.arange(Fsize[d])  
        if np.all([len(x)>0 for x in sub]):
            if sub[0] == ':' and sub[1] != ':':
                valid[:, int(sub[1])] = False
            elif sub[1] == ':' and sub[0] != ':':
                valid[int(sub[0]), :] = False
            elif sub[1] == ':' and sub[0] == ':':
                valid[:, :] = False

        start = valid.shape[d] - Fsize[d]+1
        end = valid.shape[d]
        if start == end:
            sub[d] = int(end-1)
        else:
            sub[d] = np.arange(start, end).astype(int)
        if np.all([len(x)>0 if not isinstance(x,int) else True for x in sub]):
            if sub[0] == ':' and sub[1] != ':':
                valid[:, int(sub[1])] = False
            elif sub[1] == ':' and sub[0] != ':':
                valid[int(sub[0]), :] = False
            elif sub[1] == ':' and sub[0] == ':':
                valid[:, :] = False
        # if np.all([len(x)>0 for x in sub]):
        #     valid = apply_argstring(valid, sub, False)
      
    # Mark valid and invalid pixels (i.e. the ones within and outside
    # of the part to be padded), but round the other way
    pad_valid = np.ones(sz+np.array(F.shape)-1, dtype=bool)
    
    for d in range(ndims):
        for e in range(ndims):
            sub[e] = ':'

       
        sub[d] = np.arange(Fsize_hi_2[d])  
        # pad_valid = apply_argstring(pad_valid, sub, False)
        if np.all([len(x)>0 for x in sub]):
            if sub[0] == ':' and sub[1] != ':':
                pad_valid[:, int(sub[1])] = False
            elif sub[1] == ':' and sub[0] != ':':
                pad_valid[int(sub[0]), :] = False
            elif sub[1] == ':' and sub[0] == ':':
                pad_valid[:, :] = False
        # sub[d] = np.arange(valid.shape[d] - Fsize_lo_2[d]+1, valid.shape[d])
        start = valid.shape[d] - Fsize_lo_2[d]+1
        end = valid.shape[d]
        if start == end:
            sub[d] = int(end-1)
        else:
            sub[d] = np.arange(start, end).astype(int)
        if np.all([len(x)>0 if not isinstance(x,int) else True for x in sub]):
        # if np.all([len(x)>0 for x in sub]):
            if sub[0] == ':' and sub[1] != ':':
                pad_valid[:, int(sub[1])] = False
            elif sub[1] == ':' and sub[0] != ':':
                pad_valid[int(sub[0]), :] = False
            elif sub[1] == ':' and sub[0] == ':':
                pad_valid[:, :] = False


    
    # Set coefficients on the border to zero
    M = convmtxn(F, sz, valid)
    
    # Suppress rows of M outside of the padded area
    M = M[pad_valid.T.flatten(), :]
     
    
    return M

def convmtxn(F, sz, valid):
    ndims  = len(sz)
    blksz  = np.prod(F.shape)
    nblks  = np.prod(sz)
    nelems = blksz * nblks

    
    # Build index array for all possible image positions
    tmp = np.zeros(np.array(F.shape) + sz - 1)
    sub = [0] * ndims
    for d in range(ndims):
        sub[d] = np.arange(sz[d])
    # tmp = apply_argstring(tmp, sub, 1)
    if len(sub[0])>1 and len(sub[1])>1:
        tmp[sub[0][0]:sub[0][-1]+1, sub[1][0]:sub[1][-1]+1] = 1
    elif len(sub[0])>1 and len(sub[1])==1:
        tmp[sub[0][0]:sub[0][-1]+1, sub[1]] = 1
    elif len(sub[0])==1 and len(sub[1])>1:
        tmp[sub[0], sub[1][0]:sub[1][-1]+1] = 1
    elif len(sub[0])==1 and len(sub[1])==1:
        tmp[sub[0], sub[1]] = 1
    imgpos = np.where(tmp.T.flatten())[0]

    # Build index array for all possible filter positions
    tmp = np.zeros(np.array(F.shape) + sz - 1)
    for d in range(ndims):
        sub[d] = np.arange(F.shape[d])    
    # tmp = apply_argstring(tmp, sub, 1)
    if len(sub[0])>1 and len(sub[1])>1:
        tmp[sub[0][0]:sub[0][-1]+1, sub[1][0]:sub[1][-1]+1] = 1
    elif len(sub[0])>1 and len(sub[1])==1:
        tmp[sub[0][0]:sub[0][-1]+1, sub[1]] = 1
    elif len(sub[0])==1 and len(sub[1])>1:
        tmp[sub[0], sub[1][0]:sub[1][-1]+1] = 1
    elif len(sub[0])==1 and len(sub[1])==1:
        tmp[sub[0], sub[1]] = 1
    fltpos = np.where(tmp.T.flatten())[0]

    rows = np.reshape(numpy.matlib.repmat(np.expand_dims(imgpos,axis=1).T, blksz, 1).T, (nelems, 1)) + numpy.matlib.repmat(np.expand_dims(fltpos,axis=1) , nblks, 1)
    cols = np.reshape(numpy.matlib.repmat(np.arange(nblks), blksz, 1).T, (nelems, 1))
    vals = numpy.matlib.repmat(np.expand_dims(F.flatten(),axis=1), nblks, 1)


    # Pick out valid rows
    valid_idx = valid.T.flatten()[rows]

    rows = rows[valid_idx]
    cols = cols[valid_idx]
    vals = vals[valid_idx]
    
    # Build sparse output array
    M = csr_matrix((vals, (rows, cols)), shape=(np.prod(np.array(F.shape) + sz - 1), nblks))
    return M


def compute_flow(this):
    
    sz = [this.images.shape[0], this.images.shape[1]]

    # initialize flow fields with zeros
    uv = np.zeros((sz[0], sz[1], 2))

    # Perform ROF structure texture decomposition
    images, _  = structure_texture_decomposition_rof( this.images, 1/8, 100, this.alp)

    # compute number of pyramid levels
    this.pyramid_levels  =  int( 1 + np.floor( np.log(min(images.shape[0], images.shape[1])/16) / np.log(this.pyramid_spacing) ) )

    # Construct image pyramid, using filter setting in Bruhn et al in "Lucas/Kanade.." (IJCV2005') page 218

    # For gnc stage 1
    factor            = np.sqrt(2)
    smooth_sigma      = np.sqrt(this.pyramid_spacing)/factor 
    f                 = gaussian_kernel(int(2*np.round(1.5*smooth_sigma) +1), smooth_sigma)
    pyramid_images    = compute_image_pyramid(images, f, this.pyramid_levels, 1/this.pyramid_spacing)

    
    # For segmentation purpose
    org_pyramid_images       = compute_image_pyramid(this.images, f, this.pyramid_levels, 1/this.pyramid_spacing)
    org_color_pyramid_images = compute_image_pyramid(this.color_images, f, this.pyramid_levels, 1/this.pyramid_spacing)

    
    # For gnc stage 2 to gnc_iters
    smooth_sigma       = np.sqrt(this.gnc_pyramid_spacing)/factor
    f                  = gaussian_kernel(int(2*np.round(1.5*smooth_sigma) +1), smooth_sigma)
    gnc_pyramid_images = compute_image_pyramid(images, f, this.gnc_pyramid_levels, 1/this.gnc_pyramid_spacing)

    
    # For segmentation purpose
    org_gnc_pyramid_images = compute_image_pyramid(this.images, f, this.gnc_pyramid_levels, 1/this.gnc_pyramid_spacing)
    org_color_gnc_pyramid_images = compute_image_pyramid(this.color_images, f, this.gnc_pyramid_levels, 1/this.gnc_pyramid_spacing)

    for ignc in range(this.gnc_iters):

        if ignc == 0:
            pyramid_levels  = this.pyramid_levels
            pyramid_spacing = this.pyramid_spacing
        else:
            pyramid_levels  = this.gnc_pyramid_levels
            pyramid_spacing = this.gnc_pyramid_spacing
        
        
        # Iterate through all pyramid levels starting at the top
        count = 0
        for l in range(pyramid_levels-1, -1, -1):
        # for l in range(pyramid_levels-1, pyramid_levels-3, -1):
            # print('pyramid: '+str(l))
            count += 1
            # Generate copy of algorithm with single pyramid level and the
            # appropriate subsampling
            small = copy.copy( this )

            if ignc == 0:
                nsz                = [pyramid_images[l].shape[0], pyramid_images[l].shape[1]]
                small.images       = pyramid_images[l]
                small.max_linear   = 1 # number of linearization performed per warping, 1 OK for quadratic formulation
                im1                = org_pyramid_images[l][:,:,0]
                small.color_images = org_color_pyramid_images[l]
                
            else:
                small.images         = gnc_pyramid_images[l]
                nsz   = [gnc_pyramid_images[l].shape[0], gnc_pyramid_images[l].shape[1]]
                im1   = org_gnc_pyramid_images[l][:,:,0]
                
                small.color_images      = org_color_gnc_pyramid_images[l]

        
            # Rescale the flow field
            uv        = resample_flow(uv, nsz)
            
            small.seg = im1
            
            # Adaptively determine half window size for the area term
            small.affine_hsz      = np.min([4, np.max([2, np.ceil(min(nsz)/75)]) ])
            
            # Run flow method on subsampled images
            uv = compute_flow_base(small, uv)
            a=1
    
    
        # Update GNC parameters (linearly)
        if this.gnc_iters > 0:
            new_alpha  = 1 - ignc / (this.gnc_iters-1)
            this.alpha = np.min([this.alpha, new_alpha])
            this.alpha = np.max([0, this.alpha])
    
    return uv

def estimate_flow_interface(im1, im2):
    warnings.filterwarnings("ignore",category=FutureWarning)
    ope = classic_nl_optical_flow()

    im1 = im1.astype(float)
    im2 = im2.astype(float)
    
    ope.images  = np.zeros((im1.shape[0], im1.shape[1], 2))
    ope.images[:,:,0] = copy.copy(im1)
    ope.images[:,:,1] = copy.copy(im2)
    ope.color_images  = copy.copy(im1)
    uv  = compute_flow(ope)
    return uv

def compute_optical_flow_classic_nl_fast(im,num_cores=0):
    u = []
    v = []
    numFrames = len(im)

    if num_cores == 0:
        ncores = multiprocessing.cpu_count()
        print('Using # cores:'+str(round(ncores)))
        # results = Parallel(n_jobs=round(ncores))(delayed(estimate_flow_interface)(im[i],im[i+1]) for i in trange(numFrames-1))
        results = Parallel(n_jobs=round(ncores))(delayed(estimate_flow_interface)(im[i],im[i+1]) for i in tqdm(range(numFrames-1)))
    elif num_cores > 1:
        print('Using # cores:' + str(round(num_cores)))
        results = Parallel(n_jobs=round(num_cores))(delayed(estimate_flow_interface)(im[i],im[i+1]) for i in tqdm(range(numFrames-1)))
    else:
        ### No parallel
        print('Using # cores: 1')
        for frame in trange(numFrames-1):
            uv = estimate_flow_interface(im[frame], im[frame+1])
            u.append( uv[:,:,0] )
            v.append( uv[:,:,1] )
    
    if num_cores>=0:
        for uv in results:
            u.append(uv[:,:,0])
            v.append(uv[:,:,1])

    return u, v

# import cProfile
# import pstats

if __name__ == '__main__':
#     with cProfile.Profile() as pr:
    import tifffile
    impath = r"C:\Users\romanbarth\Workspace\MATLAB\PyHi-D\hidpy\hidpy\test_singleCell\2-1.tif"
    im = tifffile.imread(impath)


    # cProfile.run('estimate_flow_interface(im[0], im[1])')
    uv = estimate_flow_interface(im[0], im[1])
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats(.05)
    a=1
    # load matlab
    u = np.loadtxt(r"C:\Users\romanbarth\Downloads\OF\u1.txt", delimiter=',')
    v = np.loadtxt(r"C:\Users\romanbarth\Downloads\OF\v1.txt", delimiter=',')


    ratio = ((uv[:,:,0]-u)/uv[:,:,0]).flatten()
    EE = np.sqrt((u-uv[:,:,0])**2 + (v-uv[:,:,1])**2)
    AE = np.arccos((uv[:,:,0]*u+uv[:,:,1]*v)/np.sqrt(uv[:,:,0]**2+uv[:,:,1]**2)/np.sqrt(u**2+v**2)) * 180/np.pi

    f, ax = plt.subplots(); ax.hist(EE.flatten(),100); ax.set_xlabel('EE (px)')
    f, ax = plt.subplots(); ax.hist(AE.flatten(),100); ax.set_xlabel('AE (deg)')
    # ratio = ratio[np.abs(ratio)<100]
    ratio = ratio[np.isfinite(ratio)]
    f, ax = plt.subplots(); plt.hist(ratio.flatten(), bins=500); ax.set_xlabel('Rel. error py-MAT')
    ax.set_yscale('log')

    plt.figure(), plt.imshow(uv[:,:,0])
    plt.figure(), plt.imshow(u)
    # use this to process all frames
    # |
    # v
    # compute_optical_flow_classic_nl_fast(im)
    a=1

