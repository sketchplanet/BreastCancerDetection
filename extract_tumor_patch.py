import os
import cv2
import numpy as np
import csv
from os import listdir
from skimage.transform.integral import integral_image, integrate
from openslide import OpenSlide
from xml.etree.ElementTree import parse

# File path -Camelyon17
file_path_tif_17 = \
    "/data2/wangruiqiao/camelyon17/training/"
file_path_xml_17 = \
    "/data2/wangruiqiao/camelyon17/lesion_annotations/"
file_path_tis_msk_of_tumor_slide_17 = \
    "/data2/wangruiqiao/camelyon17/Slide_Ground_Truth_lv_4/tumor_mask_17/"
# 起演示作用
file_path_jpg_of_tumor_slide_17 = \
    "/data2/wangruiqiao/camelyon17/Slide_origin_lv_4/Train_17_Tumor/"

# Tumor Patch save location
save_location_path_tumor_patch_17 = \
    "/data2/wangruiqiao/camelyon17/tumor_description/"
save_cut_path_negative_patch_17 = \
    "/data2/wangruiqiao/camelyon17/patches_tumor_0.3/"
ID = 0
ratio_tumor = 0.3
def find_contours_of_xml_label(file_path_xml, downsample):
    list_blob = []
    tree = parse(file_path_xml)
    for parent in tree.getiterator():
        for index1, child1 in enumerate(parent):
            for index2, child2 in enumerate(child1):
                for index3, child3 in enumerate(child2):
                    list_point = []
                    for index4, child4 in enumerate(child3):
                        p_x = float(child4.attrib['X'])
                        p_y = float(child4.attrib['Y'])
                        p_x = p_x / downsample
                        p_y = p_y / downsample
                        list_point.append([p_x, p_y])
                    if len(list_point):
                        list_blob.append(list_point)

    contours = []
    for list_point in list_blob:
        list_point_int = [[[int(round(point[0])), int(round(point[1]))]] \
                          for point in list_point]
        contour = np.array(list_point_int, dtype=np.int32)
        contours.append(contour)

    return contours


def get_list_file_name(path_directory):
    file_name_list = [name for name in listdir(path_directory)]
    file_name_list.sort()

    return file_name_list


def extract_patch_on_slide(
        file_path_tif, \
        file_path_xml, \
        file_path_tis_mask, \
        file_path_jpg, \
        save_location_path_patch_position_visualize, \
        save_location_path_patch_position_csv, \
        size_patch):

    slide = OpenSlide(file_path_tif)
    slide_w_lv_4, slide_h_lv_4 = slide.level_dimensions[4]
    downsample = slide.level_downsamples[4]
    size_patch_lv_4 = int(size_patch / downsample)

    # Make integral image of slide
    tissue_mask = cv2.imread(file_path_tis_mask, 0)
    print(file_path_tis_mask)
    integral_image_tissue = integral_image(tissue_mask.T / 255)
    # print('integral_image_tissue',integral_image_tissue)

    # Load original bgr_jpg_lv_4 for visualizing patch position
    wsi_bgr_jpg = cv2.imread(file_path_jpg)
    wsi_jpg_visualizing_patch_position = wsi_bgr_jpg.copy()

    print('==> making contours of tumor region from xml ..')

    # If Tumor_Slide, tumor regions exist.

    # Find and Draw contours_tumor - (color : yellow)
    contours_tumor = find_contours_of_xml_label(file_path_xml, downsample)
    cv2.drawContours(wsi_jpg_visualizing_patch_position, \
                     contours_tumor, -1, (0, 255, 255), 2)

    # Make csv_writer
    csv_file = open(save_location_path_patch_position_csv, 'w')
    fieldnames = ['X', 'Y']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

    print('==> Extracting patches randomly on tumor region...')
    patch_cnt = 0

    ### Extract random patches on tissue region
    for contour in contours_tumor:

        # Check if contour area is samller than patch area
        area = cv2.contourArea(contour)  # 轮廓面积
        area_patch_lv_4 = size_patch_lv_4 ** 2  # patch面积
        if area < area_patch_lv_4:
            continue

        # Determine number of patches to extract
        number_patches = int(round(area / area_patch_lv_4 * 1.5))
        print('contour area : ', area, ' num_patch : ', number_patches)

        # Get coordinates of contour (level : 4)
        coordinates = (np.squeeze(contour)).T
        coords_x = coordinates[0]
        coords_y = coordinates[1]

        # Bounding box vertex #包围顶点
        p_x_left = np.min(coords_x)
        p_x_right = np.max(coords_x)
        p_y_top = np.min(coords_y)
        p_y_bottom = np.max(coords_y)

        # Make candidates of patch coordinate (level : 4)
        candidate_x = \
            np.arange(round(p_x_left), round(p_x_right)).astype(int)
        candidate_y = \
            np.arange(round(p_y_top), round(p_y_bottom)).astype(int)

        # Pick coordinates randomly
        len_x = candidate_x.shape[0]
        len_y = candidate_y.shape[0]

        number_patches = max(number_patches, len_x)
        number_patches = max(number_patches, len_y)

        random_index_x = np.random.choice(len_x, number_patches, replace=True)
        random_index_y = np.random.choice(len_y, number_patches, replace=True)

        for i in range(number_patches):

            patch_x = candidate_x[random_index_x[i]]
            patch_y = candidate_y[random_index_y[i]]

            # Check if out of range
            if (patch_x + size_patch_lv_4 > slide_w_lv_4) or \
                    (patch_y + size_patch_lv_4 > slide_h_lv_4):
                continue

            # Check ratio of tumor region
            tissue_integral = integrate(integral_image_tissue, \
                                        (patch_x, patch_y), \
                                        (patch_x + size_patch_lv_4 - 1,
                                         patch_y + size_patch_lv_4 - 1))
            # print('tissue_integral',tissue_integral)
            tissue_ratio = tissue_integral / (size_patch_lv_4 ** 2)

            if tissue_ratio < ratio_tumor:
                continue

            # Save patches position to csv file.
            patch_x_lv_0 = int(round(patch_x * downsample))
            patch_y_lv_0 = int(round(patch_y * downsample))
            csv_writer.writerow({'X': patch_x_lv_0, 'Y': patch_y_lv_0})
            patch_cnt += 1

            #save cut patches
            im = slide.read_region((patch_x_lv_0, patch_y_lv_0), 0, (size_patch, size_patch))
            im_rgba = np.array(im)
            im_rgb = cv2.cvtColor(im_rgba, cv2.COLOR_RGBA2RGB)
            cur_patient_name = file_path_tif.split('/')[4]
            cur_patient_name = cur_patient_name.split('.')[0]
            #print('cur_patient_name',cur_patient_name)
            global ID
            cur_cut_name = save_cut_path_negative_patch_17 +cur_patient_name + '_' +str(ID) + '.jpg'
            print('cur_cut_name',cur_cut_name)
            cv2.imwrite(cur_cut_name,im_rgb)
            ID+=1
            # Draw patch position (color : Green)
            cv2.rectangle(wsi_jpg_visualizing_patch_position, \
                          (patch_x, patch_y), \
                          (patch_x + size_patch_lv_4, patch_y + size_patch_lv_4), \
                          (0, 255, 0), \
                          thickness=1)

    print('slide    :\t', file_path_tif)
    print('patch_cnt:\t', patch_cnt)

    # Save visualizing image.
    cv2.imwrite(save_location_path_patch_position_visualize, \
                wsi_jpg_visualizing_patch_position)

    csv_file.close()


def extract_patch( \
        file_path_tif, \
        file_path_xml, \
        file_path_tumor_msk, \
        file_path_jpg, \
        save_location_tumor_path_patch):
    size_patch = 299

    file_name_list_tif = get_list_file_name(file_path_tif)
    file_name_list_xml = get_list_file_name(file_path_xml)

    for index in range(len(file_name_list_tif)):
        cur_slide_name = file_name_list_tif[index].split('.')[0]
        # print('cur_slide_name', cur_slide_name)

        for i in range(len(file_name_list_xml)):
            cur_xml_name = file_name_list_xml[i].split('.')[0]
            if cur_xml_name == cur_slide_name:

                cur_path_jpg = cur_slide_name + '_origin_lv_4.jpg'
                cur_path_jpg = os.path.join(file_path_jpg, cur_path_jpg)

                cur_path_tuomr_msk = cur_slide_name + '_mask_lv_4.jpg'
                cur_path_tuomr_msk = os.path.join(file_path_tumor_msk, cur_path_tuomr_msk)

                cur_path_tif = os.path.join(file_path_tif, file_name_list_tif[index])

                cur_xml_name = cur_xml_name + '.xml'
                cur_path_xml = os.path.join(file_path_xml, cur_xml_name)

                cur_save_dir = os.path.join( \
                    save_location_tumor_path_patch, cur_slide_name)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                cur_jpg_file_name = cur_slide_name + '.jpg'
                cur_save_location_path_patch_pos_visualize = \
                    os.path.join(cur_save_dir, \
                                 cur_jpg_file_name)

                cur_csv_file_name = cur_slide_name + '.csv'
                cur_save_location_path_patch_pos_csv = \
                    os.path.join(cur_save_dir, \
                                 cur_csv_file_name)
                extract_patch_on_slide(
                    cur_path_tif, \
                    cur_path_xml, \
                    cur_path_tuomr_msk, \
                    cur_path_jpg, \
                    cur_save_location_path_patch_pos_visualize, \
                    cur_save_location_path_patch_pos_csv, \
                    size_patch)
            else:
                continue


def main():
    for i in range(5):
        dir_name = 'centre_' + str(i)
        cur_file_path_tif = os.path.join(file_path_tif_17, dir_name)

        extract_patch( \
            cur_file_path_tif, \
            file_path_xml_17, \
            file_path_tis_msk_of_tumor_slide_17, \
            file_path_jpg_of_tumor_slide_17, \
            save_location_path_tumor_patch_17)


if __name__ == "__main__":
    main()
