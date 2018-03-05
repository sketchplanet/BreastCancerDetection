import cv2
import numpy as np

import os
from os import listdir

"""
We already have tissue region mask and tumor region mask of Tumor Slide.
This program makes Normal mask of Tumor Slide.
Method :
    tissue region mask - tumor region mask (subtract)获取有标注图片中的正常区域
"""



### File Path -Camelyon17
file_path_tumor_slide_17_tissue_mask = \
    "/data2/wangruiqiao/camelyon17/Slide_tissue_mask_lv_4/Train_17_Tumor/"
file_path_tumor_slide_17_tumor_mask = \
    "/data2/wangruiqiao/camelyon17/Slide_Ground_Truth_lv_4/tumor_mask_17/"
save_location_path_17_normal_mask = \
    "/data2/wangruiqiao/camelyon17\Slide_tissue_mask_lv_4/Train_17_Normal_of_Tumor/"


def make_normal_mask(path_tis_msk, path_tumor_msk, path_save_location):
    print('==> making normal mask...')

    tis_msk = cv2.imread(path_tis_msk)
    tumor_msk = cv2.imread(path_tumor_msk)

    tumor_msk_bool = (tumor_msk == 255)
    tis_msk_after = tis_msk.copy()
    tis_msk_after[tumor_msk_bool] = 0

    print('==> saving normal mask at' + path_save_location + ' ...')
    cv2.imwrite(path_save_location, tis_msk_after)


if __name__ == '__main__':

    list_file_name_tissue_mask = [name for name in \
                                  listdir(file_path_tumor_slide_17_tissue_mask)]
    list_file_name_tissue_mask.sort()
    list_file_name_tumor_mask = [name for name in \
                                 listdir(file_path_tumor_slide_17_tumor_mask)]
    list_file_name_tumor_mask.sort()

    len_ = len(list_file_name_tissue_mask)
    for i in range(len_):
        file_name_tissue_mask = list_file_name_tissue_mask[i]
        file_name_tumor_mask = list_file_name_tumor_mask[i]
        cur_path_tissue = file_path_tumor_slide_17_tissue_mask + \
                          file_name_tissue_mask
        cur_path_tumor = file_path_tumor_slide_17_tumor_mask + \
                         file_name_tumor_mask


        ## camelyon 17
        file_name_save_word = file_name_tissue_mask.split('_')
        file_name_save = file_name_save_word[0] + ('_') + \
                         file_name_save_word[1] + ('_') + \
                         file_name_save_word[2] + ('_') + \
                         file_name_save_word[3]
        file_name_save = file_name_save + '_normal_mask_lv_4.jpg'
        cur_path_save = save_location_path_17_normal_mask + file_name_save
        make_normal_mask(cur_path_tissue, cur_path_tumor, cur_path_save)
       
