import cv2 


####################################################################################################
# @compute_optical_flow
####################################################################################################
def compute_optical_flow(frame1, 
                         frame2):

    # Calculates dense optical flow by Farneback method
#    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)    # default parameters
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.8, 20, 15, 20, 3, 1.9, 0)  # optimized parameters after simulation 

    U = flow[:, :, 0]
    V = flow[:, :, 1] 

    return U, V