import pydicom as dicom 
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
from scipy import ndimage as nd 
from dicom_reader import DCMreaderVM
from con_reader import CONreaderVM
from image_converter import im_conv
from patient_class import Patient

# 2 parameters: folder with patient folders, one patient's folder

def get_patient(folder_in, patient):
    
    temp_patient = Patient()
    path = folder_in + patient 
    
    # get pathology from meta.txt
    meta = open(path + '/meta.txt','rt')
    line = meta.readline()
    temp_patient.pathology = line.split("Pathology: ")[1].split(' \n')[0]
    meta.close()
    
    # check if images folder is empty, print Error if yes
    if len(os.listdir(path+'/sa/images')) != 0:
        # read dicom
        dr=DCMreaderVM(path + '/sa/images')
        # check if dicom is broken, print error if yes
        if dr.broken is not True:
            # get contours from con file
            cr=CONreaderVM(path + '/sa/contours.con')
            contours = cr.get_hierarchical_contours()
            # create temorary variables for contours and dcm images
            tmp_dcm = []
            tmp_contours = []
            # get indices of slices with contour
            slice_keys=list(contours.keys())    
            # get middle index in countours 
            if (len(contours) % 2) != 0: 
                mid_idx = int((len(contours)-1)/2)
            else:
                mid_idx = int(len(contours)/2)
            # store the indices of the 7 slice we want to use
            slice_indices=[
                slice_keys[2],
                slice_keys[3],
                slice_keys[mid_idx-1],
                slice_keys[mid_idx],
                slice_keys[mid_idx+1],
                slice_keys[-4],
                slice_keys[-3]
            ]          
            # loop through contours using slice_indices
            # store images in tmp_dcm, call im_conv to convert images (filter, resclale, uint8)
            # check how many frames on one slice (frame_keys)
            # save contours and image for the given frame
            for slc in slice_indices:
                frame_keys = list(contours[slc].keys())
                if len(frame_keys) == 2:
                    tmp_dcm.append(im_conv(dr.dcm_images[slc][frame_keys[0]]))
                    tmp_contours.append(contours[slc][frame_keys[0]])
                    tmp_dcm.append(im_conv(dr.dcm_images[slc][frame_keys[1]]))
                    tmp_contours.append(contours[slc][frame_keys[1]])

                elif len(frame_keys) > 2:
                    tmp_dcm.append(im_conv(dr.dcm_images[slc][frame_keys[0]]))
                    tmp_contours.append(contours[slc][frame_keys[0]])
                    tmp_dcm.append(im_conv(dr.dcm_images[slc][frame_keys[-1]]))
                    tmp_contours.append(contours[slc][frame_keys[-1]])

                elif len(frame_keys) < 2:
                    if len(frame_keys) == 1:
                        tmp_dcm.append(im_conv(dr.dcm_images[slc][frame_keys[0]]))
                        tmp_contours.append(contours[slc][frame_keys[0]])
                        if frame_keys[0] > 10:
                            tmp_dcm.append(im_conv(dr.dcm_images[slc][frame_keys[0]-10]))
                            tmp_contours.append(None)
                        else:
                            tmp_dcm.append(im_conv(dr.dcm_images[slc][frame_keys[0]+10]))
                            tmp_contours.append(None)
                    else:
                        tmp_dcm.append(im_conv(dr.dcm_images[slc][8]))
                        tmp_contours.append(None)
                        tmp_dcm.append(im_conv(dr.dcm_images[slc][20]))
                        tmp_contours.append(None)
    
            #add temporary dcm image list and contours to temp_patient
            temp_patient.dcm_images = tmp_dcm 
            temp_patient.contours = tmp_contours
            
            # get data from con file (CONreaderVM)
            if cr.volume_data['Study_id='] != None:
                temp_patient.study_id = cr.volume_data['Study_id='].split('\n')[0]
            if cr.volume_data['Patient_gender='] != None:
                temp_patient.gender = cr.volume_data['Patient_gender='].split('\n')[0]
            if cr.volume_data['Field_of_view='] != None:
                temp_patient.fov=[]
                temp_patient.fov.append(cr.volume_data['Field_of_view='].split(' ')[0].split('x')[0])
                temp_patient.fov.append(cr.volume_data['Field_of_view='].split(' ')[0].split('x')[1])
            if cr.volume_data['Slicethickness='] != None:
                temp_patient.slicethickness = cr.volume_data['Slicethickness='].split(' ')[0]
            if cr.volume_data['Patient_height='] != None:
                temp_patient.height = cr.volume_data['Patient_height='].split(' ')[0]
            if cr.volume_data['Patient_weight='] != None:
                temp_patient.weight = cr.volume_data['Patient_weight='].split(' ')[0]

            # get pixel_spacing from dicom
            temp_patient.pixel_spacing = dr.pixel_spacing 
            temp_patient.num_slices = dr.num_slices
            return temp_patient
                          
        else:
            print('Error: DCMreaderVM at {}/sa/images is broken'.format(path))
            return None
    else:
        print('Error: ../images folder at {}/sa/images is empty'. format(path))
        return None
        
 
