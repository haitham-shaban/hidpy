# System imports  
import argparse
import os
import numpy as np
from skimage import io
import glob

# Internal imports 
from ..core import optical_flow
from ..core import video_processing
from ..core import plotting

####################################################################################################
# @parse_command_line_arguments
####################################################################################################
def parse_command_line_arguments(arguments=None):
    """Parses the input arguments.
    :param arguments:
        Command line arguments.
    :return:
        Argument list.
    """

    # add all the options
    description = 'hidpy is a pythonic implementation to the technique presented by Shaban et al, 2020.'
    parser = argparse.ArgumentParser(description=description)

    arg_help = 'The input stack or image sequence that will be processed to generate the output data'
    parser.add_argument('--input-sequence', '--i', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artifacts will be stored'
    parser.add_argument('--output-directory', '--o', action='store', help=arg_help)

    arg_help = 'Zero-pad the input sequence to make the dimensions along the X and Y the same'
    parser.add_argument('--zero-pad', action='store', help=arg_help)

    arg_help = 'The regulization parameter, the default value is 0.001'
    parser.add_argument('--alpha', '--a', help=arg_help, type=float, default=0.001)

    arg_help = 'Number of iterations, default 8'
    parser.add_argument('--iterations', '--n', help=arg_help, type=int, default=8)

    # Parse the arguments
    return parser.parse_args()

def hidpy(input, mode, **kwargs):

    args = parse_command_line_arguments()
    args.input_sequence = input

    # load video
    frames = io.imread(input)
    frames = [np.float32(x) for x in frames.tolist()]

    # Compute the optical flow
    print('Computing optical flow') 
    u, v = optical_flow.compute_optical_flow_farneback(frames=frames, **kwargs)

    # save flow fields
    saveDir =  input.replace('.tif','')
    for i in range(len(frames)-1):
        datfile = os.path.join(saveDir, 'u_py_'+mode+str(i+1)+'.dat')
        np.savetxt(datfile, u[i])
        datfile = os.path.join(saveDir, 'v_py_'+mode+str(i+1)+'.dat')
        np.savetxt(datfile, v[i])




filedirs = [
    r".\simulations"
]
param_excel = r"trials.xlsx"
import pandas as pd
out = pd.read_excel(param_excel)
numParams = len(out)
for ind_OFparams in range(74, numParams):
    print('Processing parameter set '+str(ind_OFparams+1)+' of '+str(numParams))
    param_counter = 1 + ind_OFparams
    mode = 'farneback_trial_'+str(int(param_counter))+'_'
    for filedir in filedirs:
        files = glob.glob(filedir+'\*.tif')
        for file in files:
            print(file)
            # if os.path.isfile(os.path.join(file.replace('.tif',''), 'u_py_'+mode+'99.dat')): continue
            hidpy(file, mode, \
                  pyr_scale=out['pyr_scale'][ind_OFparams], \
                  levels=out['levels'][ind_OFparams], \
                  winsize=out['winsize'][ind_OFparams], \
                  iterations=out['iterations'][ind_OFparams], \
                  poly_n=out['poly_n'][ind_OFparams], \
                  poly_sigma=out['poly_sigma'][ind_OFparams], \
                    )
