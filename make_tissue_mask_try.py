import cv2
import numpy as np
from os import listdir
from openslide import OpenSlide

#training目录下分别是5个centre目录，centre目录下直接存（20个病人的）100张图
file_path_slide_17 = \
    "/data2/wangruiqiao/camelyon17/training_try/"

file_path_ground_truth_xml_17 = \
    "/data2/wangruiqiao/camelyon17/lesion_annotations_try/"

save_location_path_origin_tumor_17 = \
    "/data2/wangruiqiao/camelyon17/Slide_origin_lv_4/Train_17_Tumor/"
save_location_path_origin_normal_17 = \
    "/data2/wangruiqiao/camelyon17/Slide_origin_lv_4/Train_17_Normal/"
save_location_path_tumor_tissue_mask_17 = \
    "/data2/wangruiqiao/camelyon17/Slide_tissue_mask_lv_4/Train_17_Tumor/"
save_location_path_normal_tissue_mask_17 = \
    "/data2/wangruiqiao/camelyon17/Slide_tissue_mask_lv_4/Train_17_Normal/"


def make_mask(mask_shape, contours):
    wsi_empty = np.zeros(mask_shape[:2])
    wsi_empty = wsi_empty.astype(np.uint8)
    cv2.drawContours(wsi_empty, contours, -1, 255, -1)

    return wsi_empty


def is_tumor_slide(cur_file_name, list_file_name_xml):
    cur_file_name = cur_file_name.split('.')[0]

    for i, file_name_xml in enumerate(list_file_name_xml):
        file_name_xml = file_name_xml.split('.')[0]
        if cur_file_name == file_name_xml:
            return True

    return False


def save_slide_as_jpg_with_level(file_path, save_location, level):
    slide_tif = OpenSlide(file_path)
    print(('==> saving slide_lv_%s at ' + save_location) % level)

    wsi_pil_lv_ = slide_tif.read_region((0, 0), level, \
                                        slide_tif.level_dimensions[level])
    wsi_ary_lv_ = np.array(wsi_pil_lv_)
    wsi_bgr_lv_ = cv2.cvtColor(wsi_ary_lv_, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(save_location, wsi_bgr_lv_)


def save_origin_slide(file_path, save_location, list_file_name_xml, save_tumor):
    list_file_name = [f for f in listdir(file_path)]
    list_file_name.sort()
    level = 4
    for i, file_name in enumerate(list_file_name):

        if save_tumor:
            if (is_tumor_slide(file_name, list_file_name_xml) == False):
                continue
        else:
            if (is_tumor_slide(file_name, list_file_name_xml)):
                continue

        cur_file_path = file_path + file_name
        file_name = file_name.lower()
        file_name = file_name.replace('.tif', '')
        file_name = file_name + '_origin_lv_' + str(level) + '.jpg'
        # check if correct path
        cur_save_loca = save_location + file_name
        save_slide_as_jpg_with_level(cur_file_path, cur_save_loca, level)


def run(file_path, location_path, level):
    slide = OpenSlide(file_path)

    print('==> making contours of tissue region..')

    wsi_pil_lv_ = slide.read_region((0, 0), level, \
                                    slide.level_dimensions[level])
    wsi_ary_lv_ = np.array(wsi_pil_lv_)
    wsi_bgr_lv_ = cv2.cvtColor(wsi_ary_lv_, cv2.COLOR_RGBA2BGR)

    wsi_bgr_lv_black = (wsi_bgr_lv_ == 0)
    wsi_bgr_lv_[wsi_bgr_lv_black] = 255

    wsi_gray_lv_ = cv2.cvtColor(wsi_bgr_lv_, cv2.COLOR_BGR2GRAY)

    ret, wsi_bin_0255_lv_ = cv2.threshold( \
        wsi_gray_lv_, 0, 255, \
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ### Morphology

    kernel_o = np.ones((2, 2), dtype=np.uint8)
    kernel_c = np.ones((4, 4), dtype=np.uint8)
    wsi_bin_0255_lv_ = cv2.morphologyEx( \
        wsi_bin_0255_lv_, \
        cv2.MORPH_CLOSE, \
        kernel_c)
    wsi_bin_0255_lv_ = cv2.morphologyEx( \
        wsi_bin_0255_lv_, \
        cv2.MORPH_OPEN, \
        kernel_o)

    _, contours_tissue_lv_, hierarchy = \
        cv2.findContours( \
            wsi_bin_0255_lv_, \
            cv2.RETR_TREE, \
            cv2.CHAIN_APPROX_SIMPLE)

    print('==> making tissue mask..')

    mask_shape_lv_ = wsi_gray_lv_.shape
    tissue_mask_lv_ = make_mask(mask_shape_lv_, contours_tissue_lv_)

    print('==> saving slide_lv_' + str(level) + ' at ' + location_path)
    cv2.imwrite(location_path, tissue_mask_lv_)


def save_tissue_mask(file_path, save_location, list_file_name_xml, save_tumor):
    list_file_name = [f for f in listdir(file_path)]
    list_file_name.sort()

    level = 4
    for i, file_name in enumerate(list_file_name):

        if save_tumor:
            if (is_tumor_slide(file_name, list_file_name_xml) == False):
                continue
        else:
            if (is_tumor_slide(file_name, list_file_name_xml) == True):
                continue

        cur_file_path = file_path + file_name
        file_name = file_name.lower()
        file_name = file_name.replace('.tif', '')
        file_name = file_name + '_tissue_mask_lv_' + str(level) + '.jpg'
        # check if correct path
        cur_save_loca = save_location + file_name
        # padding = False
        camel_17 = True
        run(cur_file_path, cur_save_loca, level)


if __name__ == '__main__':

    list_file_name_xml = \
        [f for f in listdir(file_path_ground_truth_xml_17)]

    ### Save normal origin slide bgr lv 4 -Camelyon17

    for i in range(5):
        file_path = file_path_slide_17 + 'centre_' + str(i) +'/'
        save_origin_slide(file_path, save_location_path_origin_normal_17, \
                          list_file_name_xml, False)
    #exit()

    ### Save tumor -

    for i in range(5):
        file_path = file_path_slide_17 + 'centre_' + str(i) +'/'
        save_origin_slide(file_path, save_location_path_origin_tumor_17, \
                            list_file_name_xml, True)
    #exit()

    ### Save normal tissue mask -Camelyon17
    for i in range(5):
        file_path = file_path_slide_17 + 'centre_' + str(i) + '/'
        save_tissue_mask(file_path, save_location_path_normal_tissue_mask_17, \
                         list_file_name_xml, False)
    #exit()

    ### Save turmor -
    for i in range(5):
        file_path = file_path_slide_17 + 'centre_' + str(i) + '/'
        save_tissue_mask(file_path, save_location_path_tumor_tissue_mask_17, \
                         list_file_name_xml, True)
    #exit()

