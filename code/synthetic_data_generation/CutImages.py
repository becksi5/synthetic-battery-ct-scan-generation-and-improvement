import cv2
import numpy as np
import pandas as pd

dir = ""
dir_save = dir + "Cut/"
folder = "test"
for j in range(2487,4109):
    print(j)
    # Load image and labels
    img = cv2.imread(dir + folder + "/images/Sideview{}.jpg".format(j))
    labels = pd.read_csv(dir + folder + "/labels/Sideview{}.txt".format(j), header=None, sep=" ")

    # Get original image dimensions
    img_w_orig, img_h_orig = img.shape[1], img.shape[0]
    
    # Get absolut pixel values for labels
    labels.loc[:, 1] = labels.loc[:, 1] * img_w_orig
    labels.loc[:, 2] = labels.loc[:, 2] * img_h_orig
    labels.loc[:, 3] = labels.loc[:, 3] * img_w_orig
    labels.loc[:, 4] = labels.loc[:, 4] * img_h_orig

    # Cut off image edges
    img = img[int(0.125*img_h_orig):int(0.95*img_h_orig), int(0.35*img_w_orig):int(0.65*img_w_orig)]

    # Adjust labels to the cut off edges
    labels.loc[:, 1] = labels.loc[:, 1] - 0.35*img_w_orig
    labels.loc[:, 2] = labels.loc[:, 2] - 0.125*img_h_orig

    # Get new image dimensions
    img_w, img_h = img.shape[1], img.shape[0]

    # Split image into 4 quadrants
    y_cut = 0.15
    x_cut = 0.5
    y_cut = x_cut*img_w/img_h

    img_1 = img[:int(y_cut*img_h), :int(x_cut*img_w)]
    img_2 = img[:int(y_cut*img_h), int(x_cut*img_w):]
    img_3 = img[int((1-y_cut)*img_h):, :int(x_cut*img_w)]
    img_4 = img[int((1-y_cut)*img_h):, int(x_cut*img_w):]

    # Get new image dimensions for slices
    img_w_slice, img_h_slice = img_1.shape[1], img_1.shape[0]
    
    # Split labels into 4 quadrants
    labels_top = labels[labels[2] < 0.5*img_h]

    labels_bot = labels[labels[2] > 0.5*img_h]
    labels_bot.loc[:, 2] = labels_bot.loc[:, 2] - (1-y_cut)*img_h # Adjust y values for bottom half

    labels_1 = labels_top[labels_top[1] < 0.5*img_w]
    labels_1 = labels_1.reset_index(drop=True)

    labels_2 = labels_top[labels_top[1] > 0.5*img_w]
    labels_2 = labels_2.reset_index(drop=True)
    labels_2.loc[:, 1] = labels_2.loc[:, 1] - x_cut*img_w # Adjust x values for right half

    labels_3 = labels_bot[labels_bot[1] < 0.5*img_w]
    labels_3 = labels_3.reset_index(drop=True)

    labels_4 = labels_bot[labels_bot[1] > 0.5*img_w]
    labels_4 = labels_4.reset_index(drop=True)
    labels_4.loc[:, 1] = labels_4.loc[:, 1] - x_cut*img_w # Adjust x values for right half

    # Transform label to relative coordinates
    labels_1.loc[:, 1] = labels_1.loc[:, 1] / img_w_slice
    labels_1.loc[:, 2] = labels_1.loc[:, 2] / img_h_slice
    labels_1.loc[:, 3] = labels_1.loc[:, 3] / img_w_slice
    labels_1.loc[:, 4] = labels_1.loc[:, 4] / img_h_slice

    labels_2.loc[:, 1] = labels_2.loc[:, 1] / img_w_slice
    labels_2.loc[:, 2] = labels_2.loc[:, 2] / img_h_slice
    labels_2.loc[:, 3] = labels_2.loc[:, 3] / img_w_slice
    labels_2.loc[:, 4] = labels_2.loc[:, 4] / img_h_slice

    labels_3.loc[:, 1] = labels_3.loc[:, 1] / img_w_slice
    labels_3.loc[:, 2] = labels_3.loc[:, 2] / img_h_slice
    labels_3.loc[:, 3] = labels_3.loc[:, 3] / img_w_slice
    labels_3.loc[:, 4] = labels_3.loc[:, 4] / img_h_slice

    labels_4.loc[:, 1] = labels_4.loc[:, 1] / img_w_slice
    labels_4.loc[:, 2] = labels_4.loc[:, 2] / img_h_slice
    labels_4.loc[:, 3] = labels_4.loc[:, 3] / img_w_slice
    labels_4.loc[:, 4] = labels_4.loc[:, 4] / img_h_slice
    
    # Save images and labels
    cv2.imwrite(dir_save + folder + "/images/Sideview{}_1.jpg".format(j), img_1)
    cv2.imwrite(dir_save + folder + "/images/Sideview{}_2.jpg".format(j), img_2)
    cv2.imwrite(dir_save + folder + "/images/Sideview{}_3.jpg".format(j), img_3)
    cv2.imwrite(dir_save + folder + "/images/Sideview{}_4.jpg".format(j), img_4)

    labels_1.to_csv(dir_save + folder + "/labels/Sideview{}_1.txt".format(j), header=None, index=None, sep=" ")
    labels_2.to_csv(dir_save + folder + "/labels/Sideview{}_2.txt".format(j), header=None, index=None, sep=" ")
    labels_3.to_csv(dir_save + folder + "/labels/Sideview{}_3.txt".format(j), header=None, index=None, sep=" ")
    labels_4.to_csv(dir_save + folder + "/labels/Sideview{}_4.txt".format(j), header=None, index=None, sep=" ")

