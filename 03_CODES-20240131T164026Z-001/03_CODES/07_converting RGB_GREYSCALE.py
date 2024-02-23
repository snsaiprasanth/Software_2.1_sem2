import cv2
import os

# defining location for input RGB and output GREYSCALE IMAGES

input_img = r'D:\\masters\\Iaac barcelona\\AT barcelona\\02 software -1_2\Software_2.1_sem2\\iaac_building_oriol'
output_img = r'D:\\masters\\Iaac barcelona\\AT barcelona\\02 software -1_2\Software_2.1_sem2\\grey_oriol'


# convertion of the image to greyscale
for filename in os.listdir(input_img):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        #read image
        img_path = os.path.join(input_img,filename)
        img = cv2.imread(img_path)

        #conversion
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #construct the output file path 

        output_path = os.path.join(output_img,filename)

        #save the grey scale img 

        cv2.imwrite(output_path,gray_img)

        print(f'conversion {filename} to grayscale')
    print('conversion complete.')