import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
from PIL import Image
#to round the sharpness of the plate
def draw_rounded(img, radius):
    color = (0,255,0)
    h, w, _ = img.shape
    
#     cv2.line(img, (radius, 0), (w - radius, 0), color, 1)
#     cv2.line(img, (radius, h), (w - radius, h), color, 1)
#     cv2.line(img, (0, radius), (0, h - radius), color, 1)
#     cv2.line(img, (w, radius), (w, h - radius), color, 1)
#     cv2.ellipse(img, (radius, radius), (radius, radius), 180, 0, 90, color, 1) # top left
#     cv2.ellipse(img, (w - radius, radius), (radius, radius), 270, 0, 90, color, 1) # top right
#     cv2.ellipse(img, (radius, h - radius), (radius, radius), 90, 0, 90, color, 1) # bottom left
#     cv2.ellipse(img, (w - radius, h - radius), (radius, radius), 0, 0, 90, color, 1) # bottom right
    
    cv2.fillPoly(img, np.int32([[(radius, 0), (w - radius, 0), (w, radius), (w, h - radius), (w - radius, h), (radius, h), (0, h - radius), (0, radius)]]), (255, 255, 255))
    cv2.ellipse(img, (radius, radius), (radius, radius), 180, 0, 90, (255, 255, 255), -1) # top left
    cv2.ellipse(img, (w - radius, radius), (radius, radius), 270, 0, 90, (255, 255, 255), -1) # top right
    cv2.ellipse(img, (radius, h - radius), (radius, radius), 90, 0, 90, (255, 255, 255), -1) # bottom left
    cv2.ellipse(img, (w - radius, h - radius), (radius, radius), 0, 0, 90, (255, 255, 255), -1) # bottom right
# to attach the plate onto cars
def attach_plate(plate, car_img_path):
    # 02-10_16-292&475_466&571-460&539_292&571_298&507_466&475-0_0_14_6_30_25_29-55-22
    # 460&539 292&571 298&507 466&475 - bottom right, bottom left, top left, top right
    coordinates = car_img_path.split('/')[-1].split('-')[3].split('_')
    coordinates = [coord.split('&') for coord in coordinates]
    car = cv2.imread(car_img_path)
    h_car, w_car, _ = car.shape
    h_plate, w_plate, _ = plate.shape

    src_points = np.float32([[0, 0], [w_plate - 1, 0], [0, h_plate - 1], [w_plate - 1, h_plate - 1]])
    #print('src_points',src_points)
    dst_points = np.float32([coordinates[2], coordinates[3], coordinates[1], coordinates[0]])
    #print('dst_points', dst_points)
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    projected_plate = cv2.warpPerspective(plate, projective_matrix, (w_car, h_car))
    #print(projected_plate)

    sel = projected_plate != 0
    car[sel] = projected_plate[sel]
    return car

#to add noise of different extent 
def addsalt_pepper(img, SNR):
    img_ = img.copy()
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, c, axis=0) # Copy by channel to have the same shape as img
    img_[mask == 1] = 255 # salt noise
    img_[mask == 2] = 0 # 
    return img_
line_list = ['single_line', 'double_line']
color_list = ['black','red', 'white', 'yellow']
line = ''
color = ''
img_path = ''
idx = 0
for i in range(len(line_list)):
	line = line_list[i]
	print('start')
	print(line, 'line')
for j in range(len(color_list)):
	color = color_list[j]
	print(color, 'color')
	img_path = './License Plate Generator/' + line + '/' + color
	#print(img_path)
	for img in glob.glob(img_path + '/*.png'):
		img_name = os.path.basename(img)
		print(img_name)
		#print(img)
		plate = cv2.imread(img)
		img = np.zeros(plate.shape, dtype=np.uint8)
		draw_rounded(img, radius=50) # the radius can be edited for different extent of roundness
		res = cv2.bitwise_and(plate, img, mask=None)
		img = res
		w,h,c = img.shape
		#print(w,h,c)
		W = int(w*0.7)
		H = int(h*0.7)
		#print(W,H)
        #resize the image for compression
		img = cv2.resize(img, (H,W))
        #applying the gaussian blur
		x = random.randrange(15,35,2)
        #x can be adjusted but only set as odd number; the high range it has been set, the more is the blur level
		img = cv2.GaussianBlur(img,(x,x),0)
		img = cv2.GaussianBlur(img,(x,x),0)
		kernel_size = random.randint(15,35) #can be adjusted but only set as odd number
        	# Create the vertical kernel.
		kernel_v = np.zeros((kernel_size, kernel_size))
        	# Create a copy of the same for creating the horizontal kernel.
		kernel_h = np.copy(kernel_v)
        # Fill the middle row with ones.
		kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
		kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        			# Normalize.
		kernel_v /= kernel_size
		kernel_h /= kernel_size
        	# Apply the vertical kernel.
		SNR_list = random.uniform(0.98,1)
		vertical_img = cv2.filter2D(img, -1, kernel_v)
        #print(vertical_img)
		img_v = vertical_img
		img_s = addsalt_pepper(img_v.transpose(2, 1, 0),SNR_list)     # c,
		img_s = img_s.transpose(2, 1, 0)
		img_s = cv2.resize(img_s, (h,w))
		img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
		img_s = Image.fromarray(img_s)
		#j = random.randint(9,10) #j is the reduction level of quality; can choose from 1 to 10
		#prnt(img_s, 'img_s')
		save_path_s = './Modified_plates/vertical/' + line + '/' + color + '/' + line + '_' + img_name
		img_s.save(save_path_s,"JPEG", quality=20)
		horizontal_img = cv2.filter2D(img, -1, kernel_h)
		img_h = horizontal_img
		img_S = addsalt_pepper(img_h.transpose(2, 1, 0),SNR_list)     # c,
		img_S = img_S.transpose(2, 1, 0)
		img_S = cv2.resize(img_S, (h,w))
		img_S = cv2.cvtColor(img_S, cv2.COLOR_BGR2RGB)
		img_S = Image.fromarray(img_S)
		save_path_S = './Modified_plates/horizontal/' + line + '/' + color + '/' + line + '_' + img_name
		img_S.save(save_path_S,"JPEG", quality=20)
		print('image saved')


        #read the image after modification to avoid the wrong format
		img_s = cv2.imread(save_path_s)
		img_S = cv2.imread(save_path_S)
        #random choose the image modified of horizontal or vertical in motion blur
		img_to_attach = random.choice([img_s,img_S])
		#print(img_to_attach,'img_to_attach')
		img_to_attach = np.array(img_to_attach)
        #output_dir is the path to put the final output
		output_dir = './0727/'
		if not os.path.exists(output_dir):
		    os.makedirs(output_dir)
		dir = './ccpd_bg'
        #dir is the path where the original car backgrounds need to be attached are saved
		filename = random.choice(os.listdir(dir))
		#print(filename,'filename')
		car_path = dir + '/' + filename
        #for car_path in glob.glob('/Users/sunchang/Desktop/ccpd_bg/*.jpg'):
		attached = attach_plate(img_to_attach,car_path)
		#print(attached)
		if line == 'single_line':
			    idx = 0
		elif line == 'double_line':
			    idx = 1
        #print(output_dir + car_path.split('/')[-1].split('\\')[-1])
		cv2.imwrite(output_dir + img_name + '_' + str(idx) + '_' + color + '_' + car_path.split('/')[-1].split('\\')[-1], attached)
            
                    
        
        
        