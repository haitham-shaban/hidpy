from deconvolution import applyGMM_functions
from deconvolution  import applyGMMconstrained_fitout_functions
import numpy as np
import pickle
from tqdm import tqdm
from pylab import *
import os
import pandas as pd
from scipy.stats import mode as ss_mode
from matplotlib import pyplot as plt
from scipy.stats import iqr
from matplotlib import colors

import seaborn
from matplotlib import pyplot
import matplotlib.pyplot as pyplot
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import math



import core.plotting as cplot


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

####################################################################################################
# @verify_plotting_packages
####################################################################################################
def verify_plotting_packages():

    # Import the fonts
    font_dirs = ['/projects/hidpy/fonts']
    # font_dirs .extend([os.path.dirname(os.path.realpath(__file__)) + '/../fonts/'])
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)



def applyGMM_Multiple(listdir,parameters2decon,numDist):
    BayesMat={}
    count=0
    for filename in tqdm(listdir):
        GMM_input=[]
        outputmat={}

        BayesMat[count]=pickle.load(open(filename,'rb'))
        BayesMat[count]['filename']=filename
        Bayes1=BayesMat[count]
    

        for parameter2analyse in parameters2decon:
            HiD_parameter=Bayes1[parameter2analyse]            
            HiD_parameter[np.where(np.isnan(HiD_parameter))]=0
            index=np.where(HiD_parameter>1e-10)
            A = np.squeeze(np.asarray(HiD_parameter[index]))
            A=np.random.choice(A,2000)                                 # use to reduce the data to fit
            GMM_input = A.reshape(-1, 1)          
            outvar, outmat=applyGMM_functions.applyGMMfun(GMM_input,numDist)
            outputmat[parameter2analyse]=outmat
        BayesMat[count]['Deconvolution']=outputmat
            
        count=count+1 
    
    return BayesMat
    

####################################################################################################
# @apply_gmm_on_multiple_files
####################################################################################################
def apply_gmm_on_multiple_files(directory, parameters_to_deconvolve, number_distributions):
    
    # Build this dictionary 
    bayes_matrix = {}

    # Simple counter, starting at zero
    count = 0

    # For every file in the directory 
    for file in tqdm(directory):
        
        # Construct the GMM input 
        gmm_input = list()

        # Construct the output matrix dictionary 
        output_matrix_dic = {}

        # Load the pickle file 
        bayes_matrix[count] = pickle.load(open(file, 'rb'))
        
        # Set the file name 
        bayes_matrix[count]['filename'] = file
        
        # 
        Bayes1 = bayes_matrix[count]

        # For each parameter that will be deconvolved 
        for parameter in parameters_to_deconvolve:
            
            # Get the HiD parameter from the dictionary 
            hid_parameter = Bayes1[parameter]            
            
            # Set all the nan values to Zero 
            hid_parameter[np.where(np.isnan(hid_parameter))] = 0
            
            # Find the index where the hid_parameter is significant 
            index = np.where(hid_parameter > 1e-10)
            
            # use to reduce the data to fit
            A = np.squeeze(np.asarray(hid_parameter[index]))

            #output_variable, output_matrix = applyGMM_functions.applyGMMfun(gmm_input, number_distributions)
            #output_matrix_dic[parameter] = output_matrix

            GMMoutcomes = []
            GMMoutcome_matrices = []
            for _ in range(50): # This value can be adapted in the range of 50-100 
                A_GMM = np.random.choice(A, 2000)                                 
                gmm_input = A_GMM.reshape(-1, 1)
                _, output_matrix = applyGMM_functions.applyGMMfun(gmm_input, number_distributions)
                GMMoutcomes.append( [output_matrix['number_populations'], int(output_matrix['DistributionType']=='normal')] )
                GMMoutcome_matrices.append( output_matrix )

            # get the most frequently occuring outcome
            u, _, c = np.unique(GMMoutcomes, axis=0, return_index=True, return_counts=True)
            mode = u[np.argmax(c)]
            # get all matrices with that outcome
            mode_ind = np.where([np.all(mode==GMMoutcome) for GMMoutcome in GMMoutcomes])[0]
            # get the mu's, sigma's and weights from all of these GMM runs which returned 
            # the mode outcome
            GMMoutcome_matrices = [GMMoutcome_matrices[i] for i in mode_ind]
            # fix data formats
            for ind in range(len(GMMoutcome_matrices)):
                if not isinstance(GMMoutcome_matrices[ind]['weights'], list):
                    GMMoutcome_matrices[ind]['weights'] = GMMoutcome_matrices[ind]['weights'].tolist()
            GMMparams = [[y[0] for y in x['mu'].tolist()] + [y[0] for y in x['sigma'].tolist()] + x['weights'] for x in GMMoutcome_matrices]
            # sort by ascending mu
            for i in range(len(GMMparams)):
                sort_ind = np.argsort(GMMparams[i][:3])
                sort_ind_extended = sort_ind.tolist() + (sort_ind+3).tolist() + (sort_ind+6).tolist()
                GMMparams[i] = [GMMparams[i][sort_ind] for sort_ind in sort_ind_extended]
            u, _, c = np.unique(GMMparams, axis=0, return_index=True, return_counts=True)
            mode = u[np.argmax(c)]
            mode_ind = np.where([np.all(mode==GMMparam) for GMMparam in GMMparams])[0]
            output_matrix_dic[parameter] = GMMoutcome_matrices[mode_ind[0]]
        
        # Update the deconvolution matrix 
        bayes_matrix[count]['Deconvolution'] = output_matrix_dic
        
        # Next file 
        count = count + 1 

    # Return tha analysis matrix 
    return bayes_matrix

def applyGMMconstrained_dir(listdir,parameters2decon,DistributionType,numDist):
    

    BayesMat={}
    count=0
    for filename in tqdm(listdir):
        GMM_input=[]
        outputmat={}

        BayesMat[count]=pickle.load(open(filename,'rb'))
        BayesMat[count]['filename']=filename
        Bayes1=BayesMat[count]

        count2=0
    
        for parameter2analyse in parameters2decon:
            HiD_parameter=Bayes1[parameter2analyse]            
            HiD_parameter[np.where(np.isnan(HiD_parameter))]=0
            index=np.where(HiD_parameter>1e-10)
            A = np.squeeze(np.asarray(HiD_parameter[index]))
            GMM_input = A.reshape(-1, 1)          
            outmat=applyGMMconstrained_fitout_functions.applyGMMfun(GMM_input,DistributionType[count2],numDist[count2])
            outputmat[parameter2analyse]=outmat
            outputmat[parameter2analyse]['GMM_input']=GMM_input
            count2=count2+1

        BayesMat[count]['Deconvolution']=outputmat
            
        count=count+1 
    
    return BayesMat

def generateplots_TestGMM(pathBayesCells_Plots,BayesMat, parameters,nbins,showplots):

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

    verify_plotting_packages()
    
    font_size = 14
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'True'
    pyplot.rcParams['grid.linewidth'] = 1.0
    pyplot.rcParams['grid.color'] = 'black'
    pyplot.rcParams['grid.alpha'] = 0.25
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
    pyplot.rcParams['xtick.major.pad'] = '10'
    pyplot.rcParams['ytick.major.pad'] = '1'
    pyplot.rcParams['axes.edgecolor'] = '1'

    for i in tqdm(range(len(BayesMat))):

        # Get the file name 
        file_name = BayesMat[i]['filename']
        
        # File name without extension 
        filename_without_ext = os.path.splitext(file_name)[0]

        colors = ['r', 'g', 'b']
        figure_width = 8
        figure_height = 3
        # Create the new plot 
        fig, axs = plt.subplots(1, len(parameters), figsize=(figure_width, figure_height))
        fig.clf
        fig.suptitle('Result') #'Filename: '+os.path.basename(filename_without_ext), fontsize=10)
 
        # For every parameter that should be deconvolved 
        for i_paramter in range(len(parameters)):

            # Get the parameter that needs analysis 
            parameter2analyse = parameters[i_paramter]
            
            # Lists 
            xdata = list()
            x = list()

            # MAGIC
            xdata = BayesMat[i][parameter2analyse].reshape(-1, 1)
            xdata[np.where(np.isnan(xdata))] = 0
            xdata = xdata[np.where(xdata > 1e-5)]

            # Construct the histogram 
            n, bins, patches = axs[i_paramter].hist(xdata, edgecolor=colors[i_paramter], color=colors[i_paramter], density=True, bins=nbins, alpha=0.3)
            x = np.arange(min(bins), max(bins), bins[1] - bins[0])

            
            weights = BayesMat[i]['Deconvolution'][parameter2analyse]['weights']
            mu = BayesMat[i]['Deconvolution'][parameter2analyse]['mu']
            sigma = BayesMat[i]['Deconvolution'][parameter2analyse]['sigma']
            DistributionType = BayesMat[i]['Deconvolution'][parameter2analyse]['DistributionType']
            number_populations = BayesMat[i]['Deconvolution'][parameter2analyse]['number_populations']
            model0 = BayesMat[i]['Deconvolution'][parameter2analyse]['model']
            
            axs[i_paramter].set_title('') #'DistType: '+DistributionType + ', # Populations: '+str(number_populations),fontsize=6)
            axs[i_paramter].set_xlabel(parameter2analyse, fontsize=10)
            
            axs[i_paramter].xaxis.set_tick_params(labelsize=font_size)
            axs[i_paramter].yaxis.set_tick_params(labelsize=font_size)

            axs[i_paramter].xaxis.set_visible(True)
            axs[i_paramter].yaxis.set_visible(True)

            # Adjust the spines
            axs[i_paramter].spines["bottom"].set_color('black')
            axs[i_paramter].spines['bottom'].set_linewidth(1)
            axs[i_paramter].spines["left"].set_color('black')
            axs[i_paramter].spines['left'].set_linewidth(1)

            # Plot the ticks
            axs[i_paramter].tick_params(axis='x', width=1, which='both', bottom=True)
            axs[i_paramter].tick_params(axis='y', width=1, which='both', left=True)

            xticks = sample_range(min(bins), max(bins), 3)
            axs[i_paramter].set_xlim(xticks[0], xticks[-1])
            axs[i_paramter].set_xticks(xticks)

            import math
            difference = axs[i_paramter].get_ylim()[1] - axs[i_paramter].get_ylim()[0]
            yticks = sample_range(axs[i_paramter].get_ylim()[0], axs[i_paramter].get_ylim()[1] + (difference * 0.1), 4)
            axs[i_paramter].set_ylim(yticks[0], yticks[-1])
            axs[i_paramter].set_yticks(yticks)

            #axs[i_paramter].set_ylim(yticks[0], yticks[-1])

            
            tempval=np.zeros(x.shape)

            if DistributionType == 'normal':
                for d in range(int(number_populations)):
                    axs[i_paramter].plot(x, weights[d]*applyGMM_functions.normal(x, mu[d], sigma[d]))
                    tempval= tempval+weights[d]*applyGMM_functions.normal(x, mu[d], sigma[d])
                axs[i_paramter].plot(x, tempval,'k')

            else:
                if number_populations==1:
                    param = model0.parameters
                    axs[i_paramter].plot(x, weights[0]*applyGMM_functions.lognormal(x,param[0],param[1]))
                else:
                    for d in range(int(number_populations)):           
                        param = model0.distributions[d].parameters
                        axs[i_paramter].plot(x, weights[d]*applyGMM_functions.lognormal((x), param[0],param[1]))
                        tempval= tempval+weights[d]*applyGMM_functions.lognormal((x), param[0],param[1])
                    axs[i_paramter].plot(x, tempval,'k')

            fig.tight_layout(pad=0.5)
        
        fig.savefig(pathBayesCells_Plots+os.path.basename(filename_without_ext)+'.png', dpi=300, bbox_inches='tight', pad_inches=0)
        if not(showplots):
            close(fig)
    

def generatetable_TestGMM(pathBayesCells, BayesMat, parameters2decon):
    
    row_labels=['normal','lognormal']
    df2={}
    tot=len(BayesMat)
    
    PopMat=np.zeros(len(parameters2decon)*tot)

    counter=0


    for j in range(len(parameters2decon)):
        for i in range(tot):
            PopMat[counter]=BayesMat[i]['Deconvolution'][parameters2decon[j]]['number_populations']    
            counter=counter+1
    
    maxPop=int(max(PopMat))
    
    column_labels = [str(x) for x in range(1,maxPop+1,1)]

    fig, ax = plt.subplots(len(parameters2decon))
    fig.clf
    fig.suptitle('Results_GMM_Multiple, Number of cells: '+str(tot), fontsize=10)

    Sel_DistributionTypeAuto=list()
    Sel_numDistAuto=np.zeros(len(parameters2decon),dtype=int8)


    for j in range(len(parameters2decon)):
        table=np.zeros((2,maxPop))

        dfTitle=pd.DataFrame([parameters2decon[j]])
        if j==0:
            dfTitle.to_csv(pathBayesCells+'Results_GMM_Multiple.csv', index=False, header=False)
        else:
            dfTitle.to_csv(pathBayesCells+'Results_GMM_Multiple.csv', mode='a',index=False, header=False)

        for i in range(tot):
            DistributionType=BayesMat[i]['Deconvolution'][parameters2decon[j]]['DistributionType']
            DistributionType=BayesMat[i]['Deconvolution'][parameters2decon[j]]['DistributionType']
            
            if DistributionType == 'normal':
                row=0
            else: 
                row=1
            
            col=BayesMat[i]['Deconvolution'][parameters2decon[j]]['number_populations']-1

            table[row,col]=table[row,col]+1

        table=table/tot 
        table=np.around(table,decimals=3)


        df= pd.DataFrame(table,index=row_labels,columns=column_labels)
        df2[j]=pd.concat([pd.concat([df],keys=['number_populations'], axis=1)], keys=['Dist_Type'])
        df2[j].to_csv(pathBayesCells+'Results_GMM_Multiple.csv', mode='a')        

        ax[j].table(cellText = df2[j].values,rowLabels = df2[j].index,colLabels = df2[j].columns,loc = "center")
        ax[j].set_title(parameters2decon[j])
        ax[j].axis("off")


        # Find index of maximum value from 2D numpy array
        result = np.where(table == np.amax(table))
        # print('List of coordinates of maximum value in Numpy array : ')
        # # zip the 2 arrays to get the exact coordinates
        # listOfCordinates = list(zip(result[0], result[1]))
        # # travese over the list of cordinates
    
        if result[0][0]==0:
            Sel_DistributionTypeAuto.append(row_labels[0])
        else:
            Sel_DistributionTypeAuto.append(row_labels[1])
        
        Sel_numDistAuto[j]=result[1][0]+1

    
    return Sel_DistributionTypeAuto,Sel_numDistAuto


def generateplots_GMMconstrained_fitout(pathBayesCells_Plots,BayesMat,parameters2decon,nbins,Sel_DistributionType,Sel_numDist,showplots):

    verify_plotting_packages()

    font_size = 10
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'False'
    pyplot.rcParams['grid.linestyle'] = '--'
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
    pyplot.rcParams['xtick.major.pad'] = '10'
    pyplot.rcParams['ytick.major.pad'] = '0'
    pyplot.rcParams['axes.edgecolor'] = '1'

   
    
    for i in tqdm(range(len(BayesMat))):

        filename=BayesMat[i]['filename']
        filename_without_ext = os.path.splitext(filename)[0]
        

        try:
            colors = ['g', 'g', 'g']
            figure_width = 8
            figure_height = 3

            fig, axs = plt.subplots(1, len(parameters2decon), figsize=(figure_width, figure_height))
            fig.clf
            fig.suptitle('') #'Results_constrained. Filename: '+os.path.basename(filename_without_ext), fontsize=10)

        
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
                        if k < 0.0021:     #adapt default:0.0021
                            data_filtered.append(k)
                    xdata = data_filtered   


                n,bins,patches=axs[count3].hist(xdata, edgecolor=colors[count3], color=colors[count3], density=True, bins=nbins, alpha=0.3)
                x=arange(min(bins),max(bins),bins[1]-bins[0])

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
            PrefixName=os.path.basename(filename_without_ext)
            fig.savefig('%s/%s_d_a_v_constrained.png' % (pathBayesCells_Plots,PrefixName), dpi=300, bbox_inches='tight', pad_inches=0)
            if not(showplots):
                close(fig)

        except Exception as e:
            print(e.args)
            print('Error in file: '+filename_without_ext)

        
    return


def generate_plots_stats_decon(BayesMatSel,param,output_directory,showplots, tick_count=5):
    verify_plotting_packages()
    font_size = 14
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
    
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]

    labels = BayesMatSel['Deconvolution'][param]['labels']
    unique_labels = np.unique(labels)
    thresh = []

    for label in unique_labels:
        data_in_label = BayesMatSel['Deconvolution'][param]['GMM_input'][labels==label]
        thresh.append(np.max(data_in_label))

    thresh = np.sort(thresh)[:-1] # the last entry is not actually a threshold

    # map distributions back to nucleus
    labels_map = np.zeros(BayesMatSel[param].shape, dtype=int)
    numPop = len(thresh)+1

    if numPop==1:
        assigned_label = 1  
        labels_map[BayesMatSel[param]>0]= assigned_label

    else:
        for t in range(numPop):
            assigned_label = t+1    
            if t == 0:
                labels_map[BayesMatSel[param]<=thresh[t]] = assigned_label
            elif t > 0 and t<numPop-1:
                labels_map[ np.logical_and(BayesMatSel[param]>thresh[t-1], BayesMatSel[param]<=thresh[t]) ] = assigned_label
            else:
                labels_map[BayesMatSel[param]>thresh[t-1]] = assigned_label

    labels_map[BayesMatSel[param]==0] = 0
    
   

    # Plot 
    fig, ax = pyplot.subplots(1,2)
        
    listcolors=['w','g','b','purple','r','greenyellow']
    cmap = colors.ListedColormap(listcolors[0:numPop+1])

    img1=ax[0].imshow(labels_map, interpolation='nearest',cmap=cmap,origin='lower')
    
    cbar=fig.colorbar(img1,ax=ax[0],spacing='proportional',orientation='vertical',boundaries=[-0.5] + bounds[0:numPop+1] + [numPop+0.5])
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
    ax[0].set_title(title)

    # Compute stats
    stats = {
        'means': [],
        'medians': [],
        'stds': [],
        'iqrs': [],
        }

    table0=np.zeros((4,numPop))

    for t in range(numPop):
        assigned_label = t+1
        data_in_label = BayesMatSel[param][labels_map==assigned_label]
        stats['means'].append( np.nanmean(data_in_label) )
        stats['medians'].append( np.nanmedian(data_in_label) )
        stats['stds'].append( np.nanstd(data_in_label) )
        stats['iqrs'].append( iqr(data_in_label) )
        table0[0,t]=np.nanmean(data_in_label)
        table0[1,t]=np.nanmedian(data_in_label)
        table0[2,t]=np.nanstd(data_in_label)
        table0[3,t]=iqr(data_in_label)

    row_labels=['mean','median','std','iqr']
    column_labels = [str(x) for x in range(1,numPop+1,1)]
    df= pd.DataFrame(table0,index=row_labels,columns=column_labels)

    rounded_df = df.round(decimals=8)

    ax[1].table(cellText = rounded_df.values,rowLabels = rounded_df.index,colLabels = rounded_df.columns,loc='center')
    #ax[1].set_title(param)
    ax[1].axis('off')

    filename=BayesMatSel['filename']
    filename_without_ext = os.path.splitext(filename)[0]

    fig.suptitle('Statistics Populations after Deconvolution. Filename: '+os.path.basename(filename_without_ext), fontsize=10)

    fig.tight_layout(pad=0.5)
        
    fig.savefig(output_directory+os.path.basename(filename_without_ext)+'_Populations_'+param+'.png')
    if not(showplots):
        close(fig)

    filename_csv=output_directory+os.path.basename(filename_without_ext)+'_Populations_'+param+'.csv'

    dfTitle=pd.DataFrame([param])
    dfTitle.to_csv(filename_csv, index=False, header=False)
    df.to_csv(filename_csv, mode='a')

    return table0 


def generate_gmm_plots_for_all_parameters(output_directory, bayes, parameters, showplots,number_bins=30, font_size=14):

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

        # Save image after parameters plots are generated 
        fig.tight_layout(pad=0.5)
        PrefixName=os.path.basename(filename_without_ext)
        fig.savefig('%s/%s_d_a_v_multiple.png' % (output_directory,PrefixName), dpi=300, bbox_inches='tight', pad_inches=0)
        
        if not(showplots):
            close(fig)
    