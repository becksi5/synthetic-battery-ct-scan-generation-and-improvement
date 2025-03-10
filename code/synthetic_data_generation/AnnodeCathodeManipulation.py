import cv2
import numpy as np
import pandas as pd
import random

dir = ""
# Read the images

# Read the labels
labels = pd.read_csv(dir + "labels_orig.txt", header=None, sep=" ") 
sorted_labels = labels.sort_values(by=[1])
labels_top = sorted_labels[sorted_labels.iloc[:,2] < 0.5]
labels_top_c = labels_top[sorted_labels.iloc[:,0] == 0]
labels_top_a = labels_top[sorted_labels.iloc[:,0] == 1]

labels_bot = sorted_labels[sorted_labels.iloc[:,2] > 0.5]
labels_bot_c = labels_bot[sorted_labels.iloc[:,0] == 0]
labels_bot_a = labels_bot[sorted_labels.iloc[:,0] == 1]

sorted_labels = pd.concat([labels_top_c, labels_bot_c, labels_top_a, labels_bot_a])

width_multiplier = 1.75
gaussian_kernel = 5
shifty_covertip = 3 #+3 to make sure that all is covered at the tip
takethetip = 10 #Pixels to take the tip from the cathode with

for k in range(1022,100000):
    print(k)
    img = cv2.imread(dir + "Sideview_A1230025.jpg")
    img_height, img_width, img_channels = img.shape
    labels_new = pd.DataFrame({'class':[0], 'center_x':[0], 'center_y':[0], 'width':[0], 'height':[0]})
    c_shortage = {}
    for i in range(len(labels)):
        # Get the bounding box coordinates
        center_x, center_y = int(labels.iloc[i][1]*img_width), int(labels.iloc[i][2]*img_height)
        w, h = int(labels.iloc[i][3]*img_width), int(labels.iloc[i][4]*img_height)  
        
        if labels.iloc[i][0] == 0:
            # How far the cathode is pushed "down", up to 10 times its original height
            h_shortage = int(h * random.randint(1, 10000)/1000)
            c_shortage[i] = (h_shortage-shifty_covertip)/img_height
            w = int(w*1.075)

            start_x = int(center_x - w/2)

            if center_y < 0.5*img_height:
                start_y = int(center_y - h/2)
                # Overhang on the top
                center_y += h_shortage-shifty_covertip
                img_fill = img[start_y-h_shortage:start_y+takethetip, start_x:start_x+w]

                # Define the thickness of the edges to be cut off in percentage
                edge_thickness_percentage = 15

                # Calculate the thickness of the edges to be cut off in pixels
                edge_thickness_x = int(w * edge_thickness_percentage / 100)

                # Cut off the edges of the image
                img_fill = img_fill[:, edge_thickness_x:w-edge_thickness_x]

                # Fill image size to fit the overhang
                img_fill = cv2.resize(img_fill, (w, h_shortage+takethetip))

                img[start_y-shifty_covertip:start_y+h_shortage+takethetip-shifty_covertip, start_x:start_x+w] = img_fill
            else:
                start_y = int(center_y + h/2)
                # Overhang on the bottom
                center_y -= h_shortage-shifty_covertip
                img_fill = img[start_y-takethetip:start_y+h_shortage, start_x:start_x+w]
                
                # Define the thickness of the edges to be cut off in percentage
                edge_thickness_percentage = 15

                # Calculate the thickness of the edges to be cut off in pixels
                edge_thickness_x = int(w * edge_thickness_percentage / 100)

                # Cut off the edges of the image
                img_fill = img_fill[:, edge_thickness_x:w-edge_thickness_x]

                # Fill image size to fit the overhang
                img_fill = cv2.resize(img_fill, (w, h_shortage+takethetip))

                img[start_y-h_shortage-takethetip+shifty_covertip:start_y+shifty_covertip, start_x:start_x+w] = img_fill
            
            
            new_item = {'class':int(labels.iloc[i][0]), 'center_x':center_x/img_width, 'center_y':center_y/img_height, 'width':w/1.075/img_width, 'height':h/img_height}
            labels_new = pd.concat([labels_new, pd.DataFrame(new_item,index=[0])],ignore_index=True)
        
        else: 
            # Random height shortage, randint is how much is taken away from the height
            w = int(w*width_multiplier)
            h_shortage = 1-random.randint(100, 990)/1000
            if center_y < 0.5*img_height:
                # Overhang on the top
                center_y -= int(abs(h_shortage-1)*h/2)+shifty_covertip 
            else:
                # Overhang on the bottom
                center_y += int(abs(h_shortage-1)*h/2)+shifty_covertip
            h = int(h*h_shortage)+1

            # Start coordinates on the top left corner
            start_x = int(center_x - w/2)
            start_y = int(center_y - h/2)

            img_fill = img[start_y:start_y+h, start_x+w:start_x+2*w]

            if i in [24,26,79,104]:
                # Used blanks to the left
                img_fill = img[start_y:start_y+h, start_x-w:start_x]
            elif i == 28:
                img_fill = img[start_y:start_y+h, start_x+w:start_x+int(1.5*w)]
            elif i == 85:
                img_fill = img[start_y:start_y+h, start_x-int(0.45*w):start_x]
            elif i == 83:
                img_fill = img[start_y:start_y+h, start_x-int(0.5*w):start_x]

            if i != 83: 
                # Define the thickness of the edges to be cut off in percentage
                edge_thickness_percentage = 15

                # Calculate the thickness of the edges to be cut off in pixels
                edge_thickness_x = int(w * edge_thickness_percentage / 100)

                # Cut off the edges of the image
                img_fill = img_fill[:, edge_thickness_x:w-edge_thickness_x]

            # Fill image size to fit the overhang
            img_fill = cv2.resize(img_fill, (w, h))

            # Overlay img_fill onto img 
            img[start_y:start_y+h, start_x:start_x+w] = img_fill

            # Define the region to be smoothed 
            smooth_start_x, smooth_start_y, smooth_w, smooth_h = start_x-int(w/4), start_y, int(w/2), h
            # Apply a Gaussian blur to the left
            img[smooth_start_y:smooth_start_y+smooth_h, smooth_start_x:smooth_start_x+smooth_w] = cv2.GaussianBlur(img[smooth_start_y:smooth_start_y+smooth_h, smooth_start_x:smooth_start_x+smooth_w], (gaussian_kernel,gaussian_kernel), 0)

            # Apply a Gaussian blur to the right
            smooth_start_x, smooth_start_y, smooth_w, smooth_h = start_x+int(3*w/4), start_y, int(w/2), h
            img[smooth_start_y:smooth_start_y+smooth_h, smooth_start_x:smooth_start_x+smooth_w] = cv2.GaussianBlur(img[smooth_start_y:smooth_start_y+smooth_h, smooth_start_x:smooth_start_x+smooth_w], (gaussian_kernel,gaussian_kernel), 0)

            # Apply a Gaussian blur to the top
            smooth_start_x, smooth_start_y, smooth_w, smooth_h = start_x, start_y-int(w/4), w, int(w/2)
            img[smooth_start_y:smooth_start_y+smooth_h, smooth_start_x:smooth_start_x+smooth_w] = cv2.GaussianBlur(img[smooth_start_y:smooth_start_y+smooth_h, smooth_start_x:smooth_start_x+smooth_w], (gaussian_kernel,gaussian_kernel), 0)

            # Apply a Gaussian blur to the bottom
            smooth_start_x, smooth_start_y, smooth_w, smooth_h = start_x, start_y+h-int(w/4), w, int(w/2)
            img[smooth_start_y:smooth_start_y+smooth_h, smooth_start_x:smooth_start_x+smooth_w] = cv2.GaussianBlur(img[smooth_start_y:smooth_start_y+smooth_h, smooth_start_x:smooth_start_x+smooth_w], (gaussian_kernel,gaussian_kernel), 0)

            if center_y < 0.5*img_height:
                # Overhang on the top
                center_y += h/2
                h = int(labels.iloc[i][4]*img_height)-h
                center_y += h/2
            else:
                # Overhang on the bottom
                center_y -= h/2
                h = int(labels.iloc[i][4]*img_height)-h
                center_y -= h/2

            new_item = {'class':int(labels.iloc[i][0]), 'center_x':center_x/img_width, 'center_y':center_y/img_height, 'width':w/width_multiplier/img_width, 'height':h/img_height}
            labels_new = pd.concat([labels_new, pd.DataFrame(new_item,index=[0])],ignore_index=True)

    # Display the result

    labels_new = labels_new.drop(labels_new.index[0])

    for i in range(len(labels_new[labels_new['class'] == 1])):
        if i in [0,26,27]:
            # Only cathode to the right (top)
            right_cat = labels_new[labels_new['center_x'] > labels_new.iloc[i]['center_x']]
            right_cat = right_cat[right_cat['center_y'] < 0.5]
            right_cat = right_cat[right_cat['class']==0]
            right_cat = right_cat.sort_values(by=['center_x']).iloc[0]
            labels_new.loc[i+1,'height'] += c_shortage[int(right_cat.name)-1]
            labels_new.loc[i+1,'center_y'] += c_shortage[int(right_cat.name)-1]/2
        elif i in [78,79,80]:
            # Only cathode to the right (bottom)
            right_cat = labels_new[labels_new['center_x'] > labels_new.iloc[i]['center_x']]
            right_cat = right_cat[right_cat['center_y'] > 0.5]
            right_cat = right_cat[right_cat['class']==0]
            right_cat = right_cat.sort_values(by=['center_x']).iloc[0]
            labels_new.loc[i+1,'height'] += c_shortage[int(right_cat.name)-1]
            labels_new.loc[i+1,'center_y'] -= c_shortage[int(right_cat.name)-1]/2
        elif i in [24,25,43]:
            # Only cathode to the left (top)
            left_cat = labels_new[labels_new['center_x'] < labels_new.iloc[i]['center_x']]
            left_cat = left_cat[left_cat['center_y'] < 0.5]
            left_cat = left_cat[left_cat['class']==0]
            left_cat = left_cat.sort_values(by=['center_x']).iloc[-1]
            labels_new.loc[i+1,'height'] += c_shortage[int(left_cat.name)-1]
            labels_new.loc[i+1,'center_y'] += c_shortage[int(left_cat.name)-1]/2
        elif i in [53,104,105]:
            # Only cathode to the left (bottom)
            left_cat = labels_new[labels_new['center_x'] < labels_new.iloc[i]['center_x']]
            left_cat = left_cat[left_cat['center_y'] > 0.5]
            left_cat = left_cat[left_cat['class']==0]
            left_cat = left_cat.sort_values(by=['center_x']).iloc[-1]
            labels_new.loc[i+1,'height'] += c_shortage[int(left_cat.name)-1]
            labels_new.loc[i+1,'center_y'] -= c_shortage[int(left_cat.name)-1]/2
        else:
            # Cathode to the left and right
            right_cat = labels_new[labels_new['center_x'] > labels_new.iloc[i]['center_x']]
            right_cat = right_cat[right_cat['class']==0]

            left_cat = labels_new[labels_new['center_x'] < labels_new.iloc[i]['center_x']]
            left_cat = left_cat[left_cat['class']==0]

            if labels_new.iloc[i]['center_y'] < 0.5:
                right_cat = right_cat[right_cat['center_y'] < 0.5]
                left_cat = left_cat[left_cat['center_y'] < 0.5]
                right_cat = right_cat.sort_values(by=['center_x']).iloc[0]
                left_cat = left_cat.sort_values(by=['center_x']).iloc[-1]
                if left_cat['center_y'] < right_cat['center_y']:
                    labels_new.loc[i+1,'height'] += c_shortage[int(right_cat.name)-1]
                    labels_new.loc[i+1,'center_y'] += c_shortage[int(right_cat.name)-1]/2
                else:
                    labels_new.loc[i+1,'height'] += c_shortage[int(left_cat.name)-1]
                    labels_new.loc[i+1,'center_y'] += c_shortage[int(left_cat.name)-1]/2
            else:
                right_cat = right_cat[right_cat['center_y'] > 0.5]
                left_cat = left_cat[left_cat['center_y'] > 0.5]
                right_cat = right_cat.sort_values(by=['center_x']).iloc[0]
                left_cat = left_cat.sort_values(by=['center_x']).iloc[-1]
                if left_cat['center_y'] > right_cat['center_y']:
                    labels_new.loc[i+1,'height'] += c_shortage[int(right_cat.name)-1]
                    labels_new.loc[i+1,'center_y'] -= c_shortage[int(right_cat.name)-1]/2
                else:
                    labels_new.loc[i+1,'height'] += c_shortage[int(left_cat.name)-1]
                    labels_new.loc[i+1,'center_y'] -= c_shortage[int(left_cat.name)-1]/2
    '''
    for i in range(len(labels_new)):
        center_x, center_y = int(labels_new.iloc[i]['center_x']*img_width), int(labels_new.iloc[i]['center_y']*img_height)
        w, h = int(labels_new.iloc[i]['width']*img_width), int(labels_new.iloc[i]['height']*img_height)
        tl = (int(center_x - w/2), int(center_y - h/2))
        br = (int(center_x + w/2), int(center_y + h/2))
        cv2.rectangle(img, tl, br, (0, 255, 0), 1)
    

    for i in range(len(labels)):
        center_x, center_y = int(labels.iloc[i][1]*img_width), int(labels.iloc[i][2]*img_height)
        w, h = int(labels.iloc[i][3]*img_width), int(labels.iloc[i][4]*img_height)
        tl = (int(center_x - w/2), int(center_y - h/2))
        br = (int(center_x + w/2), int(center_y + h/2))
        color = (255, 0, 0)
        if labels.iloc[i][0] == 0:
            color = (0, 0, 255)
        cv2.rectangle(img, tl, br, color, 1)
    '''

    cv2.imwrite(dir + "test/images/Sideview{}.jpg".format(k), img)
    labels_new.to_csv(dir + "test/labels/Sideview{}.txt".format(k), header=False, index=False, sep=" ")