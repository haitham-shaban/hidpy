import papermill as pm
import os
import warnings
import numpy


# User input:
directory_path='%s/data/U2OS_SiR_DNA_NoSerum/' % os.getcwd()

output_folder=directory_path + 'hidpy.output/'
threshold_value=0.05
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
    pm.execute_notebook(   
        notebook1parameterize_path,
        output_path=None,
        parameters={'video_sequence': file_path,'root_output_directory':output_folder, 'pixel_threshold': threshMat[counter]}
    )
    counter=counter+1
