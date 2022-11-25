
output_directory = '/projects/hidpy/output-protocol/rna/'    

frame_0_image = 'rna_pixel-0.09_dt-0.20_threshold_90.png'
trajectory_image = 'rna_pixel-0.09_dt-0.20_threshold_90_trajectories_threshold_90.png'
model_selection_image = 'rna_pixel-0.09_dt-0.20_threshold_90_model_selection.png'
d_map_image = 'rna_pixel-0.09_dt-0.20_threshold_90_model_selection.png'
a_map_image = 'rna_pixel-0.09_dt-0.20_threshold_90_model_selection.png'
v_map_image = 'rna_pixel-0.09_dt-0.20_threshold_90_model_selection.png'

report_1_template = 'report-templates/report_1.html'

report_1_output = '/projects/hidpy/output-protocol/rna/report_1.html'

output_report_text = ''

f = open(report_1_template, 'r')
for line in f:
    if 'VIDEO_SEQUENCE_FRAME_0_IMAGE' in line:
        img = '%s/%s' % (output_directory, frame_0_image)
        line = line.replace('VIDEO_SEQUENCE_FRAME_0_IMAGE', img)
    elif 'TRAJECTORY_IMAGE' in line:
        line = line.replace('TRAJECTORY_IMAGE', img)
    elif 'MODEL_SELECTION_IMAGE' in line:
        line = line.replace('MODEL_SELECTION_IMAGE', img)
    elif 'D_MAP_IMAGE' in line:
        line = line.replace('D_MAP_IMAGE', img)
    elif 'A_MAP_IMAGE' in line:
        line = line.replace('A_MAP_IMAGE', img)
    elif 'V_MAP_IMAGE' in line:
        line = line.replace('V_MAP_IMAGE', img)
    else: 
        pass 

    output_report_text += line
f.close()

f = open(report_1_output, 'w')
f.write(output_report_text)
f.close()