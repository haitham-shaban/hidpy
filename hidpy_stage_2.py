# System imports  
import argparse

# System packages 
import os
import numpy
import pathlib 
import sys 
import warnings
import pickle
from copyreg import pickle
import glob
import os
import sys 
from tqdm import tqdm
from pathlib import Path
import numpy as np

warnings.filterwarnings('ignore') # Ignore all the warnings 

# Path hidpy
sys.path.append('%s/../' % os.getcwd())

# Internal packages 
import core 
from core import file_utils
from core import optical_flow
from core import video_processing
from core import plotting
from core import msd
from core import inference

import warnings
warnings.filterwarnings('ignore') # Ignore all the warnings 

# Path hidpy
sys.path.append('%s/../' % os.getcwd())

import core 
import core.plotting as cplot

from deconvolution import applyGMM_functions
from deconvolution import applyGMMconstrained_fitout_functions
from deconvolution import importDeconvolutionGMM


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

    description = 'hidpy is a pythonic implementation to the technique presented by Shaban et al, 2020.'
    parser = argparse.ArgumentParser(description=description)

    arg_help = 'The input configuration file that contains all the data. If this file is provided the other parameters are not considered'
    parser.add_argument('--config-file', action='store', help=arg_help, default='EMPTY')

    arg_help = 'The input stack or image sequence that will be processed to generate the output data'
    parser.add_argument('--input-sequence', action='store', help=arg_help)

    arg_help = 'Output directory, where the final results/artifacts will be stored'
    parser.add_argument('--output-directory', action='store', help=arg_help)

    arg_help = 'Histogram plotting number bins'
    parser.add_argument('--n-bins', action='store', type=int, default=30, help=arg_help)

    arg_help = 'Maximum number of distributions'
    parser.add_argument('--n-distributions', action='store', type=int, default=3, help=arg_help)

    arg_help = 'Deconvolve the D parameter'
    parser.add_argument('--deconvolve-d', action='store_true')

    arg_help = 'Deconvolve the A parameter'
    parser.add_argument('--deconvolve-a', action='store_true')

    arg_help = 'Deconvolve the V parameter'
    parser.add_argument('--deconvolve-v', action='store_true')

    # Parse the arguments
    return parser.parse_args()
    

from matplotlib.ticker import ScalarFormatter
def sample_range(start,
                 end,
                 steps):

    # Delta
    delta = 1. * (end - start) / (steps - 1)

    # Data
    data = list()
    for i in range(steps):
        value = start + i * delta
        data.append(value)

    return data

def generate_gmm_plots_for_all_parameters(output_directory, bayes, parameters, number_bins=30, font_size=14):

    import numpy as np
    import pickle
    from tqdm import tqdm
    import os
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy.stats import iqr
    from matplotlib import colors

    import seaborn
    from matplotlib import pyplot
    import matplotlib.pyplot as pyplot
    import matplotlib.font_manager as font_manager
    from matplotlib.ticker import FuncFormatter

    cplot.verify_plotting_packages()
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'False'
    pyplot.rcParams['grid.linewidth'] = 1.0
    pyplot.rcParams['grid.color'] = 'black'
    pyplot.rcParams['grid.alpha'] = 1.0
    pyplot.rcParams['font.family'] = 'Helvetica LT Std'
    pyplot.rcParams['font.family'] = 'NimbusSanL'
    pyplot.rcParams['font.monospace'] = 'Regular'
    pyplot.rcParams['font.style'] = 'normal'
    pyplot.rcParams['axes.labelweight'] = 'light'
    pyplot.rcParams['axes.linewidth'] = 1.0
    pyplot.rcParams['axes.labelsize'] = font_size
    pyplot.rcParams['xtick.labelsize'] = font_size
    pyplot.rcParams['ytick.labelsize'] = font_size
    pyplot.rcParams['legend.fontsize'] = font_size
    pyplot.rcParams['figure.titlesize'] = font_size
    pyplot.rcParams['axes.titlesize'] = font_size
    pyplot.rcParams['axes.edgecolor'] = '1'
    

    # For every analysis result (or file)
    for i in tqdm(range(len(bayes))):

        # Get the file name 
        file_name = bayes[i]['filename']
        
        # File name without extension 
        filename_without_ext = os.path.splitext(file_name)[0]

        colors = ['r', 'r', 'r']

        figure_width = 8
        figure_height = 3
        
        
        # Create the new plot 
        fig, axs = plt.subplots(1, len(parameters), figsize=(figure_width, figure_height))
        fig.clf
        fig.suptitle('Result') #'Filename: '+os.path.basename(filename_without_ext), fontsize=10)
 
        # For every parameter that should be deconvolved 
        for j in range(len(parameters)):

            # Get the parameter that needs analysis 
            parameter = parameters[j]
            
            # Lists 
            xdata = list()
            x = list()

            # MAGIC
            xdata = bayes[i][parameter].reshape(-1, 1)
            xdata[np.where(np.isnan(xdata))] = 0
            xdata = xdata[np.where(xdata > 1e-10)]

            # Remove the outliers  
            if 'D' in parameter:
                data_filtered = list()
                for k in xdata:
                    if k < 0.0021:
                        data_filtered.append(k)
                xdata = data_filtered
            
            # Construct the histogram 
            n, bins, patches = axs[j].hist(xdata, edgecolor=colors[j], color=colors[j], density=True, bins=number_bins, alpha=0.2)
            x = np.arange(min(bins), max(bins), bins[1] - bins[0])


            weights = bayes[i]['Deconvolution'][parameter]['weights']
            mu = bayes[i]['Deconvolution'][parameter]['mu']
            sigma = bayes[i]['Deconvolution'][parameter]['sigma']
            DistributionType = bayes[i]['Deconvolution'][parameter]['DistributionType']
            number_populations = bayes[i]['Deconvolution'][parameter]['number_populations']
            model0 = bayes[i]['Deconvolution'][parameter]['model']
            
            axs[j].set_title('') #'DistType: '+DistributionType + ', # Populations: '+str(number_populations),fontsize=6)
            if 'D' in parameter:
                axs[j].set_xlabel('Diffusion Constant ($\mu$m$^2$/s)', fontsize=font_size)
            elif 'A' in parameter:
                axs[j].set_xlabel('Anomalous Exponent', fontsize=font_size)
            elif 'V' in parameter:
                axs[j].set_xlabel(r'Drift Velocity ($\mu$m/s)', fontsize=font_size)


            
            axs[j].xaxis.set_tick_params(labelsize=font_size)
            axs[j].yaxis.set_tick_params(labelsize=font_size)

            axs[j].xaxis.set_visible(True)
            axs[j].yaxis.set_visible(True)

            axs[j].spines['top'].set_color('none')
            axs[j].spines['right'].set_color('none')

            # Adjust the spines
            axs[j].spines["bottom"].set_color('black')
            axs[j].spines['bottom'].set_linewidth(1)
            axs[j].spines["left"].set_color('black')
            axs[j].spines['left'].set_linewidth(1)

            # Plot the ticks
            axs[j].tick_params(axis='x', width=1, which='both', bottom=True, direction="out")
            axs[j].tick_params(axis='y', width=1, which='both', left=True, direction="out")

            if 'D' in parameter:
                xticks = sample_range(min(bins), max(bins), 3)
                axs[j].set_xlim(0, 0.002)
                #axs[j].set_xticks(xticks)

            elif 'A' in parameter:
                xticks = sample_range(min(bins), max(bins), 3)
                axs[j].set_xlim(0, 1.0)
                #axs[j].set_xticks(xticks)
            
            else:
                xticks = sample_range(min(bins), max(bins), 3)
                axs[j].set_xlim(xticks[0], xticks[-1])
                #axs[j].set_xticks(xticks)

            
            axs[j].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            axs[j].get_xaxis().get_major_formatter().set_useOffset(True)



            
            difference = axs[j].get_ylim()[1] - axs[j].get_ylim()[0]
            yticks = sample_range(axs[j].get_ylim()[0], axs[j].get_ylim()[1] + (difference * 0.1), 4)
            axs[j].set_ylim(yticks[0], yticks[-1])
            axs[j].set_yticks(yticks)

            
            
            tempval=np.zeros(x.shape)

            if DistributionType == 'normal':
                for d in range(int(number_populations)):
                    axs[j].plot(x, weights[d]*applyGMM_functions.normal(x, mu[d], sigma[d]))
                    tempval= tempval+weights[d]*applyGMM_functions.normal(x, mu[d], sigma[d])
                axs[j].plot(x, tempval,'k')

            else:
                if number_populations==1:
                    param = model0.parameters
                    axs[j].plot(x, weights[0]*applyGMM_functions.lognormal(x,param[0],param[1]))
                else:
                    for d in range(int(number_populations)):           
                        param = model0.distributions[d].parameters
                        axs[j].plot(x, weights[d]*applyGMM_functions.lognormal((x), param[0],param[1]))
                        tempval= tempval+weights[d]*applyGMM_functions.lognormal((x), param[0],param[1])
                    axs[j].plot(x, tempval,'k')

            fig.tight_layout(pad=0.5)
        fig.savefig('%s/d_a_v_multiple.png' % output_directory, dpi=300, bbox_inches='tight', pad_inches=0)

from numpy import number


def generateplots_GMMconstrained_fitout(pathBayesCells_Plots,BayesMat,parameters2decon,nbins,Sel_DistributionType,Sel_numDist,showplots):

    import numpy as np
    import pickle
    from tqdm import tqdm
    import os
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy.stats import iqr
    from matplotlib import colors

    import seaborn
    from matplotlib import pyplot
    import matplotlib.pyplot as pyplot
    import matplotlib.font_manager as font_manager
    from matplotlib.ticker import FuncFormatter

    font_size=8

    cplot.verify_plotting_packages()
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'False'
    pyplot.rcParams['grid.linewidth'] = 1.0
    pyplot.rcParams['grid.color'] = 'black'
    pyplot.rcParams['grid.alpha'] = 1.0
    pyplot.rcParams['font.family'] = 'Helvetica LT Std'
    pyplot.rcParams['font.family'] = 'NimbusSanL'
    pyplot.rcParams['font.monospace'] = 'Regular'
    pyplot.rcParams['font.style'] = 'normal'
    pyplot.rcParams['axes.labelweight'] = 'light'
    pyplot.rcParams['axes.linewidth'] = 1.0
    pyplot.rcParams['axes.labelsize'] = font_size
    pyplot.rcParams['xtick.labelsize'] = font_size
    pyplot.rcParams['ytick.labelsize'] = font_size
    pyplot.rcParams['legend.fontsize'] = font_size
    pyplot.rcParams['figure.titlesize'] = font_size
    pyplot.rcParams['axes.titlesize'] = font_size
    pyplot.rcParams['axes.edgecolor'] = '1'

    
    for i in tqdm(range(len(BayesMat))):

        filename=BayesMat[i]['filename']
        filename_without_ext = os.path.splitext(filename)[0]
        
        colors = ['g', 'g', 'g']
        figure_width = 8
        figure_height = 3

        fig, axs = plt.subplots(1, len(parameters2decon), figsize=(figure_width, figure_height))
        fig.clf
        fig.suptitle('') #  'Results_constrained. Filename: '+os.path.basename(filename_without_ext), fontsize=10)

        
        for count3 in range(len(parameters2decon)):
            parameter2analyse=parameters2decon[count3]
            xdata=[]
            x=[]

            xdata=BayesMat[i][parameter2analyse].reshape(-1, 1)
            xdata[np.where(np.isnan(xdata))]=0
            xdata=xdata[np.where(xdata>1e-10)]

            # Remove the outliers  
            if 'D' in parameter2analyse:
                data_filtered = list()
                for k in xdata:
                    if k < 0.0021:
                        data_filtered.append(k)
                xdata = data_filtered

            n,bins,patches=axs[count3].hist(xdata, edgecolor=colors[i], color=colors[i], density=True, bins=nbins, alpha=0.3);
            x=np.arange(min(bins),max(bins),bins[1]-bins[0])

            weights= BayesMat[i]['Deconvolution'][parameter2analyse]['weights']
            mu=BayesMat[i]['Deconvolution'][parameter2analyse]['mu']
            sigma=BayesMat[i]['Deconvolution'][parameter2analyse]['sigma']
            DistributionType=Sel_DistributionType[count3]
            number_populations=Sel_numDist[count3]
            model0=BayesMat[i]['Deconvolution'][parameter2analyse]['model']
            


            title = ''
            if 'lognormal' in DistributionType:
                title += 'Distribution: Log-normal \n'
            elif 'normal' in DistributionType:
                title += 'Distribution: Normal \n'
            
            title += '# Populations: ' + str(number_populations)
            #axs[count3].set_title('Dist Type: '+DistributionType + ', # Populations: '+str(number_populations),fontsize=12)
            axs[count3].set_title(title,fontsize=8)
            
           
            ##
            axs[count3].xaxis.set_tick_params(labelsize=font_size)
            axs[count3].yaxis.set_tick_params(labelsize=font_size)

            axs[count3].xaxis.set_visible(True)
            axs[count3].yaxis.set_visible(True)

            axs[count3].spines['top'].set_color('none')
            axs[count3].spines['right'].set_color('none')

            # Adjust the spines
            axs[count3].spines["bottom"].set_color('black')
            axs[count3].spines['bottom'].set_linewidth(1)
            axs[count3].spines["left"].set_color('black')
            axs[count3].spines['left'].set_linewidth(1)

            # Plot the ticks
            axs[count3].tick_params(axis='x', width=1, which='both', bottom=True, direction="out")
            axs[count3].tick_params(axis='y', width=1, which='both', left=True, direction="out")

            if 'D' in parameter2analyse:
                xticks = sample_range(min(bins), max(bins), 3)
                axs[count3].set_xlim(0, 0.002)
                #axs[j].set_xticks(xticks)

            elif 'A' in parameter2analyse:
                xticks = sample_range(min(bins), max(bins), 3)
                axs[count3].set_xlim(0, 1.0)
                #axs[j].set_xticks(xticks)
            
            else:
                xticks = sample_range(min(bins), max(bins), 3)
                axs[count3].set_xlim(xticks[0], xticks[-1])
                #axs[j].set_xticks(xticks)

            
            axs[count3].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            axs[count3].get_xaxis().get_major_formatter().set_useOffset(True)



            
            difference = axs[count3].get_ylim()[1] - axs[count3].get_ylim()[0]
            yticks = sample_range(axs[count3].get_ylim()[0], axs[count3].get_ylim()[1] + (difference * 0.1), 4)
            axs[count3].set_ylim(yticks[0], yticks[-1])
            axs[count3].set_yticks(yticks)

            ##

            #axs[count3].set_title('') #'DistType: '+DistributionType + ', # Populations: '+str(number_populations),fontsize=6)
            #axs[count3].set_xlabel(parameter2analyse, fontsize=10)

            if 'D' in parameter2analyse:
                axs[count3].set_xlabel('Diffusion Constant ($\mu$m$^2$/s)', fontsize=font_size)
            elif 'A' in parameter2analyse:
               axs[count3].set_xlabel('Anomalous Exponent', fontsize=font_size)
            elif 'V' in parameter2analyse:
               axs[count3].set_xlabel(r'Drift Velocity ($\mu$m/s)', fontsize=font_size)
            
            
            tempval=np.zeros(x.shape)

            if DistributionType == 'normal':
                for d in range(int(number_populations)):
                    axs[count3].plot(x, weights[d]*applyGMM_functions.normal(x, mu[d], sigma[d]))
                    tempval= tempval+weights[d]*applyGMM_functions.normal(x, mu[d], sigma[d])
                axs[count3].plot(x, tempval,'k')

            else:
                if number_populations==1:
                    param = model0.parameters
                    axs[count3].plot(x, weights[0]*applyGMM_functions.lognormal(x,param[0],param[1]))
                else:
                    for d in range(int(number_populations)):           
                        param = model0.distributions[d].parameters
                        axs[count3].plot(x, weights[d]*applyGMM_functions.lognormal((x), param[0],param[1]))
                        tempval= tempval+weights[d]*applyGMM_functions.lognormal((x), param[0],param[1])
                    axs[count3].plot(x, tempval,'k')

            fig.tight_layout(pad=0.5)
        fig.savefig('%s/d_a_v_constrained.png' % pathBayesCells_Plots, dpi=300, bbox_inches='tight', pad_inches=0)

# Generate spatial mapping of population deconvolution
showplots=True

####################################################################################################
# @sample_range
####################################################################################################
def sample_range(start,
                 end,
                 steps):

    # Delta
    delta = 1. * (end - start) / (steps - 1)

    # Data
    data = list()
    for i in range(steps):
        value = start + i * delta
        data.append(value)

    return data

def generate_plots_stats_decon(BayesMatSel,param,output_directory, showplots, tick_count=5):

    
    from matplotlib import colors, pyplot
    import seaborn
    font_size = 14
    import numpy
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'False'
    pyplot.rcParams['grid.linewidth'] = 0.5
    pyplot.rcParams['grid.color'] = 'black'
    pyplot.rcParams['grid.alpha'] = 0.25
    pyplot.rcParams['font.family'] = 'NimbusSanL'
    pyplot.rcParams['font.monospace'] = 'Regular'
    pyplot.rcParams['font.style'] = 'normal'
    pyplot.rcParams['axes.labelweight'] = 'light'
    pyplot.rcParams['axes.linewidth'] = 0.5
    pyplot.rcParams['axes.labelsize'] = font_size
    pyplot.rcParams['xtick.labelsize'] = font_size
    pyplot.rcParams['ytick.labelsize'] = font_size
    pyplot.rcParams['legend.fontsize'] = font_size
    pyplot.rcParams['figure.titlesize'] = font_size
    pyplot.rcParams['axes.titlesize'] = font_size
    pyplot.rcParams['xtick.major.pad'] = '1'
    pyplot.rcParams['ytick.major.pad'] = '1'
    pyplot.rcParams['axes.edgecolor'] = '0'
    pyplot.rcParams['axes.autolimit_mode'] = 'round_numbers'
    from matplotlib import colors, pyplot
    import seaborn

    import numpy
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]

    labels = BayesMatSel['Deconvolution'][param]['labels']
    unique_labels = np.unique(labels)
    thresh = []

    for label in unique_labels:
        data_in_label = BayesMatSel['Deconvolution'][param]['GMM_input'][labels==label]
        thresh.append( np.max(data_in_label))

    thresh = np.sort(thresh)[:-1] # the last entry is not actually a threshold

    # map distributions back to nucleus
    labels_map = np.zeros(BayesMatSel[param].shape, dtype=int)
    numPop = len(thresh)+1
    for t in range(numPop):
        assigned_label = t+1    
        if t == 0:
            labels_map[BayesMatSel[param]<=thresh[t]] = assigned_label
        elif t > 0 and t<numPop-1:
            labels_map[ np.logical_and(BayesMatSel[param]>thresh[t-1], BayesMatSel[param]<=thresh[t]) ] = assigned_label
        else:
            labels_map[BayesMatSel[param]>thresh[t-1]] = assigned_label
    labels_map[BayesMatSel[param]==0] = 0

    
    #fig,ax=plt.subplots(1,2,figsize=[10,5])
    # Plot 
    fig, ax = pyplot.subplots()
        
    listcolors=['w','g','b','purple','r','greenyellow']
    cmap = colors.ListedColormap(listcolors[0:numPop+1])

    # Create the ticks of the images 
    xticks = sample_range(0, labels_map.shape[0], tick_count)
    yticks = sample_range(0, labels_map.shape[1], tick_count)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Color-basr axis 
    #cax = ax.inset_axes([0.00, -0.20, 1.0, 0.05])

    img1=ax.imshow(labels_map, interpolation='nearest', cmap=cmap,origin='lower')
    cbar=fig.colorbar(img1,   ax=ax,spacing='proportional',orientation='vertical',boundaries=[-0.5] + bounds[0:numPop+1] + [numPop+0.5])
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(1)
    labels_cbar = np.arange(0, numPop+1, 1)
    loc = labels_cbar
    cbar.set_ticks(loc)

    if 'D' in param:
        title = r'Diffusion Constant ($\mu$m$^2$/s)'
    elif 'A' in param:
        title = 'Anomalous Exponent'
    elif 'V' in param:
        title = r'Drift Velocity ($\mu$m/s)'
    else:
        print('ERROR')

    #
    ax.set_title(title)

    #output_directory=pickle_files_directory
    #frame_prefix=param

    # Save the figure 
    pyplot.savefig('%s/%s_deconvolution.png' % (output_directory, param), dpi=300, bbox_inches='tight', pad_inches=0)


####################################################################################################
# @__main__
####################################################################################################
if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_command_line_arguments()

    if args.config_file == 'EMPTY':
        
        video_sequence = args.input_sequence
        output_directory = args.output_directory
        nbins = args.n_bins
        numDist = args.n_distributions
        parameters2decon = list()
        if args.deconvolve_d: 
            parameters2decon.append('D')
        if args.deconvolve_a: 
            parameters2decon.append('A')
        if args.deconvolve_v: 
            parameters2decon.append('V')
    
    else:

        import configparser
        config_file = configparser.ConfigParser()

        # READ CONFIG FILE
        config_file.read(args.config_file)

        video_sequence = str(config_file['HID_PARAMETERS']['video_sequence'])
        output_directory = str(config_file['HID_PARAMETERS']['output_directory'])
        numDist = int(config_file['HID_PARAMETERS']['n_distributions'])
        nbins = int(config_file['HID_PARAMETERS']['n_bins'])

        deconvolve_d = config_file['HID_PARAMETERS']['deconvolve_d']
        deconvolve_a = config_file['HID_PARAMETERS']['deconvolve_a']
        deconvolve_v = config_file['HID_PARAMETERS']['deconvolve_v']
        
        parameters2decon = list()
        if deconvolve_d == 'Yes':
            parameters2decon.append('D')
        if deconvolve_a: 
            parameters2decon.append('A')
        if deconvolve_v: 
            parameters2decon.append('V')

    print(parameters2decon)

    # Create an output-directory that is specific to the input sequence 
    path = "%s/%s" % (output_directory, pathlib.Path(video_sequence).stem) 
    pickle_files_directory = '%s/pickle/' % path

    # Read cell pickle files
    pickle_files = core.file_utils.list_files_in_directory(directory=pickle_files_directory, extension='.pickle')

    # Dictionary with all the Bayes struct per cell
    BayesMat=importDeconvolutionGMM.apply_gmm_on_multiple_files(pickle_files, parameters2decon, numDist) # Apply GMM

    gmm_multiple_directory = '%s/gmm_multiple/' % pickle_files_directory
    core.file_utils.create_directory(gmm_multiple_directory)

    importDeconvolutionGMM.generatetable_TestGMM(gmm_multiple_directory, BayesMat, parameters2decon) # Make Table


    # Generating plots for inspection
    showplots = True

    generate_gmm_plots_for_all_parameters(
        output_directory=gmm_multiple_directory, bayes=BayesMat, 
        parameters=parameters2decon, number_bins=nbins)
    # importDeconvolutionGMM.generateplots_TestGMM(gmm_multiple_directory, BayesMat, parameters2decon, nbins, showplots)

    Sel_DistributionType=['normal','normal','lognormal']
    Sel_numDist=[3,3,2]

    BayesMatSel=importDeconvolutionGMM.applyGMMconstrained_dir(pickle_files,parameters2decon,Sel_DistributionType,Sel_numDist)

        
    # Generate plots GMM constrained

    gmm_constrained_directory = '%s/gmm_constrained/' % pickle_files_directory
    core.file_utils.create_directory(gmm_constrained_directory)

    showplots=True

    #importDeconvolutionGMM.generateplots_GMMconstrained_fitout(
    #    gmm_constrained_directory, BayesMatSel, parameters2decon, nbins, Sel_DistributionType, Sel_numDist, showplots)

    generateplots_GMMconstrained_fitout(
        gmm_constrained_directory, BayesMatSel, parameters2decon, nbins, Sel_DistributionType, Sel_numDist, showplots)


    for i in tqdm(range(len(BayesMatSel))):
        for j in range(len(parameters2decon)):
            
            importDeconvolutionGMM.generate_plots_stats_decon(BayesMatSel[i],parameters2decon[j],gmm_constrained_directory,showplots)
            # generate_plots_stats_decon(BayesMatSel[i],parameters2decon[j],pathBayesCells_Populations_Plots,showplots)

            #except:
            #    filename_without_ext = os.path.splitext(BayesMatSel[i]['filename'])[0]
            #    print('WARNING: Error generating population label plot: File: '+filename_without_ext+' Parameter: '+parameters2decon[j])

    for i in tqdm(range(len(BayesMatSel))):
        for j in range(len(parameters2decon)):
            
            #importDeconvolutionGMM.generate_plots_stats_decon(BayesMatSel[i],parameters2decon[j],pathBayesCells_Populations_Plots,showplots)
            generate_plots_stats_decon(BayesMatSel[i],parameters2decon[j],gmm_constrained_directory,showplots, tick_count=3)

            #except:
            #    filename_without_ext = os.path.splitext(BayesMatSel[i]['filename'])[0]
            #    print('WARNING: Error generating population label plot: File: '+filename_without_ext+' Parameter: '+parameters2decon[j])