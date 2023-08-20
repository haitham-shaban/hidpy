import numpy 
import random 
import cv2
import os
from PIL import Image
import matplotlib
from matplotlib import pyplot 
from matplotlib import colors
import matplotlib.pyplot as pyplot
import matplotlib.font_manager as font_manager
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import seaborn
import time
import pandas

from matplotlib.colors import Normalize
import matplotlib.cm as cm

####################################################################################################
# @verify_plotting_packages
####################################################################################################
def verify_plotting_packages():

    # Import the fonts
    font_dirs = list()
    font_dirs.extend([os.path.dirname(os.path.realpath(__file__)) + '/../fonts/'])
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)


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


####################################################################################################
# @plot_trajectories_on_frame
####################################################################################################
def plot_trajectories_on_frame(frame, trajectories, output_path):
    # trajectories are defined as [y,x]

    # Compute the time 
    start = time.time()
    
    # Create an RGB image from the input frame 
    rgb_image = Image.fromarray(frame).convert("RGB")
    
    # Create a numpy array from the image 
    np_image = numpy.array(rgb_image)

    # Draw each trajectory 
    for i, trajectory in enumerate(trajectories):
        
        # Create random colors 
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        # Starting pixel 
        cv2.circle(np_image, (int(trajectory[0][1]), int(trajectory[0][0])), 1, (r,g,b), 1)

        # The rest of the trajectory 
        for kk in range(len(trajectory) - 1):
            
            # First point 
            y0 = int(trajectory[kk][0])
            x0 = int(trajectory[kk][1])

            # Last point 
            y1 = int(trajectory[kk + 1][0])
            x1 = int(trajectory[kk + 1][1])

            # Create the line 
            cv2.line(np_image, (x0,y0), (x1,y1), (r,g,b), 1)
    
    # Save the trajectory image 
    cv2.imwrite('%s.png' % output_path, np_image)


####################################################################################################
# @plot_trajectories
####################################################################################################
def plot_trajectories(size, trajectories, output_path):
# Trajectories corrresponds to [y,x]

    # Create an RGB image from the input frame 
    rgb_image = Image.new(mode="RGB", size=size)
    
    # Create a numpy array from the image 
    np_image = numpy.array(rgb_image)

    # Draw each trajectory 

    for i, trajectory in enumerate(trajectories):
        
        # Create random colors 
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        # Starting pixel 
        cv2.circle(np_image, (int(trajectory[0][1]), int(trajectory[0][0])), 1, (r,g,b), 1)

        # The rest of the trajectory 
        for kk in range(len(trajectory) - 1):
            
            # First point 
            y0 = int(trajectory[kk][0])
            x0 = int(trajectory[kk][1])

            # Last point 
            y1 = int(trajectory[kk + 1][0])
            x1 = int(trajectory[kk + 1][1])

            # Create the line 
            cv2.line(np_image, (x0,y0), (x1,y1), (r,g,b), 1)
    
    # Save the trajectory image 
    cv2.imwrite('%s.png' % output_path, np_image)


####################################################################################################
# @plot_frame
####################################################################################################
def plot_frame(frame, output_directory, frame_prefix, font_size=10, tick_count=5):

    verify_plotting_packages()
    
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


    # Plot 
    fig, ax = pyplot.subplots()
    
    # Create the ticks of the images 
    xticks = sample_range(0, frame.shape[1]-1, tick_count)
    yticks = sample_range(0, frame.shape[0]-1, tick_count)

    # Show the image 
    im = pyplot.imshow(frame)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    
    
    # Color-basr axis 
    cax = ax.inset_axes([0.00, -0.15, 1.0, 0.05])

    # Create the ticks based on the range 
    cbticks = sample_range((frame.min()), (frame.max()), 4)
    cbticks = list(map(int, cbticks))

    # Convert the ticks to a numpy array 
    cbticks = numpy.array(cbticks)
    
    # Color-bar 
    #cb = pyplot.colorbar(im, ax=ax, cax=cax, orientation="horizontal", ticks=cbticks)
    cb = pyplot.colorbar(im, ax=ax, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=font_size, width=0.5) 
    #cb.ax.set_xlim((cbticks[0], cbticks[-1]))
    cb.update_ticks()

    # Save the figure 
    pyplot.savefig('%s/%s.png' % (output_directory, frame_prefix), dpi=300, bbox_inches='tight', pad_inches=0)


####################################################################################################
# @plot_frame
####################################################################################################
def plot_labels_map(labels_map, output_directory, frame_prefix, font_size=10, npop=1):


    from matplotlib import colors, pyplot
    import seaborn

    import numpy
    seaborn.set_style("whitegrid")
    pyplot.rcParams['axes.grid'] = 'True'
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

    pyplot.clf

    # Plot 
    fig, ax = pyplot.subplots()
    
    # Create the ticks of the images 
    xticks = sample_range(0, labels_map.shape[1], 5)
    yticks = sample_range(0, labels_map.shape[0], 5)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    

    listcolors=['w','g','b','purple','r','greenyellow']
    cmap = colors.ListedColormap(listcolors[0:npop+1])

    img1=ax.imshow(labels_map, interpolation='nearest',cmap=cmap,origin='lower')
    # Show the image 
    
     # Color-basr axis 
    cax = ax.inset_axes([0.00, -0.15, 1.0, 0.05])

    cbar=fig.colorbar(img1, ax=ax,spacing='proportional',orientation='horizontal',boundaries=[-0.5] + bounds[0:npop+1] + [npop+0.5], cax=cax)
    cbar.outline.set_color('black')
    cbar.outline.set_linewidth(1)
    labels_cbar = numpy.arange(0, npop+1, 1)
    loc = labels_cbar
    cbar.set_ticks(loc)


    # Save the figure 
    pyplot.savefig('%s/%s.png' % (output_directory, frame_prefix), dpi=300, bbox_inches='tight', pad_inches=0)


####################################################################################################
# @plot_model_selection_image
####################################################################################################
def plot_model_selection_image(model_selection_matrix, 
                               mask_matrix, 
                               output_directory, 
                               frame_prefix, 
                               font_size=14, 
                               title='Model Selection', 
                               tick_count=5):

    verify_plotting_packages()

    # Styles 
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

    # A new figure 
    pyplot.clf
    fig, ax = pyplot.subplots()

    # Create the color-map 
    palette = seaborn.color_palette("hls", 5)
    palette.insert(0, 'w')
    cmap = colors.ListedColormap(palette)
    
    # Render the image 
    image = ax.imshow(model_selection_matrix, interpolation='nearest', cmap=cmap, origin='lower')
    ax.contour(mask_matrix, colors='k', origin='lower')

    # Create the ticks of the images 
    xticks = sample_range(0, model_selection_matrix.shape[1], tick_count)
    yticks = sample_range(0, model_selection_matrix.shape[0], tick_count)

    # Update the axex 
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Update the title 
    ax.set_title(title)

    # Color-bar bounds  
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5]

    cbar = fig.colorbar(image,ax=ax,spacing='proportional',orientation='vertical',boundaries=[-0.5] + bounds + [5.5])
    cbar.set_ticks(numpy.arange(0, 6, 1))
    cbar.set_ticklabels([' ','D','DA','V','DV','DAV'])

    # Save the figure 
    pyplot.savefig('%s/%s.png' % (output_directory, frame_prefix), dpi=300, bbox_inches='tight', pad_inches=0)


####################################################################################################
# @plot_matrix_map
####################################################################################################
def plot_matrix_map(matrix, mask_matrix, output_directory, frame_prefix, font_size=14, title='Matrix', tick_count=5):

    verify_plotting_packages()
    
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

    # New figure  
    pyplot.clf
    fig, ax = pyplot.subplots()
    
    # Create the ticks of the images 
    xticks = sample_range(0, matrix.shape[1], tick_count)
    yticks = sample_range(0, matrix.shape[0], tick_count)

    # Show the image 
    image = pyplot.imshow(matrix, interpolation='nearest',cmap='viridis',origin='lower')
    ax.contour(mask_matrix, colors='k', origin='lower')

    # Axes 
    xticks = list(map(int, xticks))
    yticks = list(map(int, yticks))
    ax.set_xlim(xticks[0], xticks[-1])
    ax.set_ylim(yticks[0], yticks[-1])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Title 
    ax.set_title(title)

    # Color-bar 
    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar = fig.colorbar(image, ax=ax,  spacing='proportional',orientation='vertical', format=fmt)
    cbar.formatter.set_powerlimits((0, 0)) 
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Re-adjust the color-bar 
    cb_range = cbar.ax.get_ylim()
    cbticks = sample_range(cb_range[0], cb_range[-1], 4)
    cbticks = numpy.array(cbticks)
    cbar.update_ticks()

    # Save the figure 
    pyplot.savefig('%s/%s.png' % (output_directory, frame_prefix), dpi=300, bbox_inches='tight', pad_inches=0)
    
    
####################################################################################################
# @plot_trajectories_on_frame_quiver
####################################################################################################
def plot_trajectories_on_frame_quiver(frame, trajectories, output_path,oversampling_factor=10,dpi=100):


    if numpy.max(frame)>255:
        # Convert the 16-bit frame to 8-bit for visualization
        frame = (frame/numpy.max(frame)).astype(numpy.uint8)
    else:
        frame = (frame).astype(numpy.uint8)
    

    # Create an RGB image from the converted frame
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Create an oversampled image
    oversampled_image = cv2.resize(rgb_image, None, fx=oversampling_factor, fy=oversampling_factor, interpolation=cv2.INTER_LINEAR)

    # Create a copy of the oversampled image for drawing vectors
    np_image = numpy.copy(oversampled_image)
    
    #Reduce vectors ploted by 2
    trajectoriesSel =random.sample(trajectories,int(numpy.floor(len(trajectories)/2))) 

    # Draw each trajectory
    for i, trajectory in enumerate(trajectoriesSel):
        # Create random colors
        color = tuple(numpy.random.randint(0, 255, 3).tolist())

        # Starting pixel
        start_point = (int(trajectory[0][1] * oversampling_factor), int(trajectory[0][0] * oversampling_factor))

        # Last pixel
        last_point = (int(trajectory[-1][1] * oversampling_factor), int(trajectory[-1][0] * oversampling_factor))

        
        # Draw a line from start to last point as a single vector
        cv2.arrowedLine(np_image, start_point, last_point, color,2, tipLength=0.2)

    # Create a figure and axis    
    fig, ax = pyplot.subplots(figsize=(np_image.shape[1] / dpi, np_image.shape[0] / dpi), dpi=dpi) 
    
    ax.imshow(np_image)
    ax.axis('off')

    # Save the trajectory image
    pyplot.savefig('%s.png' % output_path,dpi=dpi,bbox_inches='tight')


def plot_flowfields_firstframe_quiver(frame, mask_nucleus, u, v, output_path,spacing,margin=0,**kwargs):
    
    u_single=u.copy()
    v_single=v.copy()

    u_single[mask_nucleus == 0] = numpy.nan
    v_single[mask_nucleus == 0] = numpy.nan

    h,w,*_=u.shape

    nx=int((w-2*margin)/spacing)
    ny=int((h-2*margin)/spacing)
    
    x = numpy.linspace(margin, w - margin - 1, nx, dtype=numpy.int64)
    y = numpy.linspace(margin, h - margin - 1, ny, dtype=numpy.int64)

    u_single = u_single[numpy.ix_(y, x)]
    v_single = v_single[numpy.ix_(y, x)]


    kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}


    colors = numpy.arctan2(u, v)
    norm = Normalize()
    norm.autoscale(colors)
    colormap = cm.viridis

    # Create a figure and axis    
    fig, ax = pyplot.subplots(1,2,figsize=(6, 4))
          
    h=ax[0].quiver(x, y, u_single, v_single, numpy.arctan2(u_single, v_single),pivot='mid',cmap='jet',**kwargs)

    cbar=pyplot.colorbar(h)
    cbar.mappable.set_clim(-numpy.pi,numpy.pi)
    cbar.remove()

    ax[0].imshow(frame,cmap='gray',alpha=0.9,vmax=2*numpy.max(frame))

    ax[0].axis('off')
    
    ax[0].invert_yaxis()
  
    ph = numpy.linspace(-numpy.pi,numpy.pi, 10)
    x2 = numpy.cos(ph)
    y2 = numpy.sin(ph)
    u2 = numpy.cos(ph)
    v2 = numpy.sin(ph)
    
    h2=ax[1].quiver(x2, y2, u2, v2, numpy.arctan2(u2, v2),  angles='xy', scale_units='xy', scale=1, pivot='mid',cmap='jet')

    
    ax[1].set_aspect('equal')
    cbar2=pyplot.colorbar(h2)
    cbar2.mappable.set_clim(-numpy.pi,numpy.pi)

    ax[1].set_facecolor(color='k')
    ax[1].set_xlim(-5,5)
    ax[1].set_ylim(-5,5)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    cbar2.remove()

    # Save the flowfields image
    pyplot.savefig('%s.png' % output_path,dpi=300,bbox_inches='tight')

def plot_trajectories_on_frame_colorcodetime(frame, deltaT, trajectories, output_path,sampling=20, dpi=400):
    np_image=frame.copy()
    
    #Reduce vectors ploted by 
    trajectoriesSel =random.sample(trajectories,int(numpy.floor(len(trajectories)/sampling))) 


    #Create a figure and axis    
    fig, ax = pyplot.subplots(figsize=(6,6), dpi=dpi) 
    
    ax.imshow(frame,cmap='gray',alpha=0.9,vmax=2*numpy.max(frame))
    ax.axis('off')

    timeVector=deltaT*numpy.arange(0,len(trajectories[0]))

    # Draw each trajectory
    for i, trajectory in enumerate(trajectoriesSel):

        xcoord=[point[1] for point in trajectory]
        ycoord=[point[0] for point in trajectory]

        pyplot.scatter(xcoord,ycoord,s=0.05,c=timeVector, cmap='summer',marker='o',edgecolors='none')

    cb = pyplot.colorbar(ax=ax,orientation="horizontal",pad=0.05, shrink=0.8)
    cb.ax.tick_params(labelsize=15, width=0.5) 

    cbticks = numpy.linspace(numpy.min(timeVector), numpy.round(numpy.max(timeVector)), num=4)
    cb.set_ticks(cbticks)

    ax.set_xlim(0,np_image.shape[1])
    ax.set_ylim(0,np_image.shape[0])
    cb.set_label('Time (s)')
    
    # Save the trajectory image
    pyplot.savefig('%s.png' % output_path,dpi=dpi,bbox_inches='tight')

def plot_swarmplotComparison(file_keylists,Paramdecon_keylist,data_dicts,ConditionsSTR):
    NumberConditions=len(file_keylists)
    df_All=[]

    for i in range(len(Paramdecon_keylist)):
        df = list()
        matValues = list()

        for ii in range(NumberConditions):
            meanMat = list()
            meanMatarray = list()

            for j in range(len(file_keylists[ii])):
                mat_temp = numpy.array(data_dicts[ii][file_keylists[ii][j]][Paramdecon_keylist[i]])
                
                # Check if file is empty
                if mat_temp.size == 0:
                    continue

                meanMat.append(mat_temp[0,:])

            meanMatarray = numpy.array(meanMat)

            for iii in range(meanMatarray.shape[0]):
                for jjj in range(meanMatarray.shape[1]):
                    matValues.append([meanMatarray[iii, jjj], jjj, ii])
        
        
        df = pandas.DataFrame(matValues, columns=['Values', 'Populations', 'Conditions'])

        # Define a mapping dictionary for replacement
        mapping = {iiii: f'Pop{iiii+1}' for iiii in range(len(df))}

        # Replace values in the 'Populations' column using the mapping dictionary
        df['Populations'] = df['Populations'].replace(mapping)

        df_All.append(df)
        
        #Plot the data using seaborn violinplot
        fig, axes = pyplot.subplots(figsize=(5,5))
        ax=seaborn.swarmplot(x='Conditions', y='Values', hue='Populations', data=df,ax = axes, dodge=True)
        

        # plot the mean line
        seaborn.boxplot(showmeans=True,
                    meanline=True,
                    meanprops={'color': 'k', 'ls': '--', 'lw': 1},
                    medianprops={'visible': False},
                    whiskerprops={'visible': False},
                    zorder=10,
                    x='Conditions',
                    y='Values',
                    hue='Populations',
                    data=df,
                    showfliers=False,
                    showbox=False,
                    showcaps=False,
                    ax=ax)
        
        handles,labels=pyplot.gca().get_legend_handles_labels()

        pyplot.legend(handles[:3],labels[:3],title='Populations ')
        seaborn.move_legend(ax, "upper left")

        ax.legend_.remove()


        # Set axis labels
        pyplot.xlabel('')
        pyplot.xticks(numpy.arange(NumberConditions),ConditionsSTR)
        #plt.ylabel(Paramdecon_keylist[i])

        if Paramdecon_keylist[i]=='D':
            pyplot.ylim(0,8e-3)
            pyplot.ylabel(r'Diffusion Constant ($\mu$m$^2$/s)')
        elif Paramdecon_keylist[i]=='D_norm':
            pyplot.ylim(0,8e-3)
            pyplot.ylabel(r'$log_{10}$(Diffusion Constant) (unitless)')
        elif Paramdecon_keylist[i]=='A':
            pyplot.ylim(0,1.5)
            pyplot.ylabel('Anomalous Exponent (a. u.)')
        elif Paramdecon_keylist[i]=='V':
            pyplot.ylim(0,0.08)
            pyplot.ylabel(r'Drift Velocity ($\mu$m/s)')


        # Show the plot
        pyplot.show()

    return df_All
