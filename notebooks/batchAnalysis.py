import papermill as pm
import os
import warnings
import numpy
import pathlib 

# User input:
directory_path='%s/data/U2OS_SiR_DNA_Serum/' % os.getcwd() # Directory including *.tif / *.tiff files
threshold_value=0.05
pixel_size = 0.065
dt = 0.200  

# The models. Users must either select all or some of them  
models_selected = ['D','DA','V','DV','DAV'] 

output_folder=directory_path + 'hidpy.output/'
notebook1parameterize_path='%s/notebooks/01-hidpy-stage-1.ipynb' % os.getcwd()


def get_tiff_files(directory):
    tiff_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".tiff") or file.lower().endswith(".tif"):
                tiff_files.append(os.path.join(root, file))
    return tiff_files

tiff_files_list = get_tiff_files(directory_path)

threshMat=threshold_value*numpy.ones(len(tiff_files_list))



counter=0
for file_path in tiff_files_list:  
    print(file_path)  

    # Get the prefix, typically with the name of the video sequence  
    prefix = '%s_pixel-%2.2f_dt-%2.2f_threshold_%s' % (pathlib.Path(file_path).stem, pixel_size, dt, threshMat[counter])

    try:
        print('Analizing: ' +file_path)
        pm.execute_notebook(   
            notebook1parameterize_path,
            output_path=None,
            parameters={'video_sequence': file_path,'root_output_directory':output_folder, 'pixel_threshold': threshMat[counter],'pixel_size': pixel_size, 'dt': dt, 'prefix': prefix, 'models_selected':models_selected}
        )
    
    except:
        print('Error: ' +file_path)

    counter=counter+1
