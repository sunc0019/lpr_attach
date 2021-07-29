from PIL import Image
import random, os
import numpy as np
import cv2 as cv
import glob
import pickle


# read file from train_pic and identify its size
path1 = r"C:\Users\win10\Desktop\all" 
path_train_pic = r"C:\Users\win10\Desktop\PreliminaryData\train_pic"

for img in glob.glob(r"C:\Users\win10\Desktop\PreliminaryData\train_pic\1000*"):
    img_name = os.path.basename(img)
    print(img_name)
    img = Image.open(path_train_pic + "\\" + img_name)
    img = np.array(img)
    h, w, c = img.shape
    #print('width:  ', w)
    #print('height: ', h)
    #print('channel:', c)
    img_bbg = np.zeros((h,w,c), np.uint8) 
    path1 = r"C:\Users\win10\Desktop\all" 
    count = 0 
    while count < 3:
        random_filename1 = random.choice([x for x in os.listdir(path1)
        if os.path.isfile(os.path.join(path1, x))])
        print(random_filename1)
        img_op = Image.open(path1 + r"/" + random_filename1)
        img_op_dim = np.array(img_op)
        #print(img_op_dim)
        #resizing the image
            #img_dim1 = np.array(cv_img1)
            #print(cv_img1)
            #print(pil_img.size)
        if img_op.size == (1190,255):
                #print(img_op_dim)
                #resizing the image
            resized_img = img_op.resize((190,40))
            w1 = 190
            h1 = 40
            x = random.randint(0,500)
            y = random.randint(0,500) 
            #resized_img.save(r"C:\Users\win10\Desktop\resized_img" + r"\\" + random_filename1, 'JPEG') 
            # Pasting img_op image on top of img_bbg
            #color_coverted = cv.cvtColor(img_bbg, cv.COLOR_BGR2RGB)
            # Displaying the converted image
            #img_bbg = Image.fromarray(color_coverted)
            #color_coverted1 = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)
            # Displaying the converted image
            #cv.imshow('resized_img', np.array(resized_img))
            #cv.waitKey(0)
            color_coverted = cv.cvtColor(np.uint8(img_bbg), cv.COLOR_RGB2BGR)
            color_coverted1 = cv.cvtColor(np.uint8(resized_img), cv.COLOR_RGB2BGR) 
            img_bbg = Image.fromarray(color_coverted)        
            resized_img = Image.fromarray(color_coverted1)
            resized_img = resized_img.convert('RGBA')
            img_bbg = img_bbg.convert('RGBA')
            img_bbg.paste(resized_img, (int(x),int(y)), mask = resized_img)
            #print(img_bbg)
            #img_bbg.save(r"C:\Users\win10\Desktop\output" + "\\" + random_filename + '_' + random_filename1, 'PNG') 
            mydict = {}
            mydict[img_name + random_filename1] = x,y
            output = open('myfile.pkl', 'wb')
            pickle.dump(mydict, output)
            output.close()
            
            # read python dict back from the file
            pkl_file = open('myfile.pkl', 'rb')
            mydict2 = pickle.load(pkl_file)
            pkl_file.close()
            
            print(mydict)
            print(mydict2)
                        
            
            img_bbg = np.uint8(img_bbg)
            mask = np.zeros(img_bbg.shape[:2],dtype=np.uint8)
            mask[y:y+h1,x:x+w1] = 255
            #cv.imwrite('mask.jpg', mask)
            #res = cv.bitwise_and(img,img,mask = mask)
            color_coverted2 = cv.cvtColor(mask, cv.COLOR_RGB2BGR)
            # Displaying the converted image
            mask = Image.fromarray(color_coverted2)
            mask = mask.convert("RGBA")
            #mask.save(r"C:\Users\win10\Desktop\mask" + "\\" + random_filename1, 'JPEG')
            random.seed() #make a,b,c,d,e the same for both transformations
            a = random.uniform(0,0.3)
            b = random.uniform(0.8,1)
            c = random.uniform(0,0.7)
            d = random.uniform(0,0.1)
            e = random.uniform(0.8,1)
            #print(a,b,c,d,e)
            #affine transormation
            src = np.array(mask)       
            srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]] ).astype(np.float32)
            dstTri = np.array( [[0, src.shape[1]*a], [src.shape[1]*b, src.shape[0]*c], [src.shape[1]*d, src.shape[0]*e]] ).astype(np.float32)
            warp_mat = cv.getAffineTransform(srcTri, dstTri)
            warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
            #Rotating the image after Warp
            center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)
            angle = 1
            scale = 1
            rot_mat = cv.getRotationMatrix2D( center, angle, scale )
            warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
            #cv.imshow('Source image', src)
            #cv.imshow('Warp', warp_dst)
            #cv.imshow('Warp + Rotate', warp_rotate_dst)
            color_coverted3 = cv.cvtColor(warp_rotate_dst, cv.COLOR_RGB2BGR)
            # Displaying the converted image
            pil_image3 = Image.fromarray(color_coverted3)
            pil_image3 = pil_image3.convert("RGBA")   
    
            #pil_image3.save(r"C:\Users\win10\Desktop\affine_mask" + "\\"  + random_filename1, 'JPEG')
            #affine transform   
            src = np.array(img_bbg)
            #if src is None:
                #print('Could not open or find the image:', args.input)
                #exit(0)
            srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]] ).astype(np.float32)
            dstTri = np.array( [[0, src.shape[1]*a], [src.shape[1]*b, src.shape[0]*c], [src.shape[1]*d, src.shape[0]*e]] ).astype(np.float32)
            warp_mat = cv.getAffineTransform(srcTri, dstTri)
            warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
            # Rotating the image after Warp
            center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)
            angle = 1
            scale = 1
            rot_mat = cv.getRotationMatrix2D( center, angle, scale )
            warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
    
            #cv.imshow('Source image', src)
            #cv.imshow('Warp', warp_dst)
            #cv.imshow('Warp + Rotate', warp_rotate_dst)
            #cv.waitKey(0)
            #color_coverted4 = cv.cvtColor(warp_rotate_dst, cv.COLOR_RGB2BGR)
            # Displaying the converted image
            pil_image4 = Image.fromarray(warp_rotate_dst)
            pil_image4 = pil_image4.convert("RGBA")  
            #cv.imshow('pil_image4', np.array(pil_image4))
            #cv.waitKey(0)
            
            #pil_image4.save(r"C:\Users\win10\Desktop\affine_output" + "\\"  + random_filename1, 'JPEG')        
            #resize the normal plates and add the resized plate on the black bg
            color_coverted5 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            # Displaying the converted image
            pil_image5 = Image.fromarray(color_coverted5)
            pil_image5 = pil_image5.convert("RGBA")  
            #cv.imshow('pil_image5', np.array(pil_image5))
            #cv.waitKey(0)
            imgT = pil_image5.convert('RGB')
            imgA = pil_image4.convert('RGB')
            imgB = pil_image3.convert('RGB')


            imgT = np.array(imgT)
            imgA = np.array(imgA)
            imgB = np.array(imgB)

            #print(img.shape)
            #print(imgA.shape)  
            #print(imgB.shape)
        
            imgT[imgB == (255,255,255)] = imgA[imgB == (255,255,255)]
            color_coverted = cv.cvtColor(imgT, cv.COLOR_RGB2BGR)
            pil_image = Image.fromarray(color_coverted)
            #cv.imshow('pil_image',np.array(pil_image))
            #cv.waitKey(0)
            pil_image.save(r"C:\Users\win10\Desktop\0527_output" + "\\"  + img_name + "_" + random_filename1, 'JPEG')
        elif img_op.size == (785,407):    
            resized_img = img_op.resize((157,81))
            w1 = 157
            h1 = 81
            #print(resized_img2.size)
            x = random.randint(0,500)
            y = random.randint(0,500) 
            #resized_img.save(r"C:\Users\win10\Desktop\resized_img" + r"\\" + random_filename1, 'JPEG') 
            # Pasting img_op image on top of img_bbg
            #color_coverted = cv.cvtColor(img_bbg, cv.COLOR_BGR2RGB)
            # Displaying the converted image
            #img_bbg = Image.fromarray(color_coverted)
            #color_coverted1 = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)
            # Displaying the converted image
            #cv.imshow('resized_img', np.array(resized_img))
            #cv.waitKey(0)
            color_coverted = cv.cvtColor(np.uint8(img_bbg), cv.COLOR_RGB2BGR)
            color_coverted1 = cv.cvtColor(np.uint8(resized_img), cv.COLOR_RGB2BGR) 
            img_bbg = Image.fromarray(color_coverted)        
            resized_img = Image.fromarray(color_coverted1)
            resized_img = resized_img.convert('RGBA')
            img_bbg = img_bbg.convert('RGBA')
            img_bbg.paste(resized_img, (int(x),int(y)), mask = resized_img)
            #print(img_bbg)
            #img_bbg.save(r"C:\Users\win10\Desktop\output" + "\\" + random_filename + '_' + random_filename1, 'PNG') 
            mydict = {}
            mydict[img_name + random_filename1] = x,y
            output = open('myfile.pkl', 'wb')
            pickle.dump(mydict, output)
            output.close()
            
            # read python dict back from the file
            pkl_file = open('myfile.pkl', 'rb')
            mydict2 = pickle.load(pkl_file)
            pkl_file.close()
            
            print(mydict)
            print(mydict2)
                        
            
            img_bbg = np.uint8(img_bbg)
            mask = np.zeros(img_bbg.shape[:2],dtype=np.uint8)
            mask[y:y+h1,x:x+w1] = 255
            #cv.imwrite('mask.jpg', mask)
            #res = cv.bitwise_and(img,img,mask = mask)
            color_coverted2 = cv.cvtColor(mask, cv.COLOR_RGB2BGR)
            # Displaying the converted image
            mask = Image.fromarray(color_coverted2)
            mask = mask.convert("RGBA")
            #mask.save(r"C:\Users\win10\Desktop\mask" + "\\" + random_filename1, 'JPEG')
            random.seed() #make a,b,c,d,e the same for both transformations
            a = random.uniform(0,0.3)
            b = random.uniform(0.8,1)
            c = random.uniform(0,0.7)
            d = random.uniform(0,0.1)
            e = random.uniform(0.8,1)
            #print(a,b,c,d,e)
            #affine transormation
            src = np.array(mask)       
            srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]] ).astype(np.float32)
            dstTri = np.array( [[0, src.shape[1]*a], [src.shape[1]*b, src.shape[0]*c], [src.shape[1]*d, src.shape[0]*e]] ).astype(np.float32)
            warp_mat = cv.getAffineTransform(srcTri, dstTri)
            warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
            #Rotating the image after Warp
            center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)
            angle = 1
            scale = 1
            rot_mat = cv.getRotationMatrix2D( center, angle, scale )
            warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
            #cv.imshow('Source image', src)
            #cv.imshow('Warp', warp_dst)
            #cv.imshow('Warp + Rotate', warp_rotate_dst)
            color_coverted3 = cv.cvtColor(warp_rotate_dst, cv.COLOR_RGB2BGR)
            # Displaying the converted image
            pil_image3 = Image.fromarray(color_coverted3)
            pil_image3 = pil_image3.convert("RGBA")   
    
            #pil_image3.save(r"C:\Users\win10\Desktop\affine_mask" + "\\"  + random_filename1, 'JPEG')
            #affine transform   
            src = np.array(img_bbg)
            #if src is None:
                #print('Could not open or find the image:', args.input)
                #exit(0)
            srcTri = np.array( [[0, 0], [src.shape[1] - 1, 0], [0, src.shape[0] - 1]] ).astype(np.float32)
            dstTri = np.array( [[0, src.shape[1]*a], [src.shape[1]*b, src.shape[0]*c], [src.shape[1]*d, src.shape[0]*e]] ).astype(np.float32)
            warp_mat = cv.getAffineTransform(srcTri, dstTri)
            warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))
            # Rotating the image after Warp
            center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)
            angle = 1
            scale = 1
            rot_mat = cv.getRotationMatrix2D( center, angle, scale )
            warp_rotate_dst = cv.warpAffine(warp_dst, rot_mat, (warp_dst.shape[1], warp_dst.shape[0]))
    
            #cv.imshow('Source image', src)
            #cv.imshow('Warp', warp_dst)
            #cv.imshow('Warp + Rotate', warp_rotate_dst)
            #cv.waitKey(0)
            #color_coverted4 = cv.cvtColor(warp_rotate_dst, cv.COLOR_RGB2BGR)
            # Displaying the converted image
            pil_image4 = Image.fromarray(warp_rotate_dst)
            pil_image4 = pil_image4.convert("RGBA")  
            #cv.imshow('pil_image4', np.array(pil_image4))
            #cv.waitKey(0)
            
            #pil_image4.save(r"C:\Users\win10\Desktop\affine_output" + "\\"  + random_filename1, 'JPEG')        
            #resize the normal plates and add the resized plate on the black bg
            color_coverted5 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            # Displaying the converted image
            pil_image5 = Image.fromarray(color_coverted5)
            pil_image5 = pil_image5.convert("RGBA")  
            #cv.imshow('pil_image5', np.array(pil_image5))
            #cv.waitKey(0)
            imgT = pil_image5.convert('RGB')
            imgA = pil_image4.convert('RGB')
            imgB = pil_image3.convert('RGB')


            imgT = np.array(imgT)
            imgA = np.array(imgA)
            imgB = np.array(imgB)

            #print(img.shape)
            #print(imgA.shape)  
            #print(imgB.shape)
        
            imgT[imgB == (255,255,255)] = imgA[imgB == (255,255,255)]
            color_coverted = cv.cvtColor(imgT, cv.COLOR_RGB2BGR)
            pil_image = Image.fromarray(color_coverted)
            #cv.imshow('pil_image',np.array(pil_image))
            #cv.waitKey(0)
        
            
            pil_image.save(r"C:\Users\win10\Desktop\0527_output" + "\\"  + img_name + "_" + random_filename1, 'JPEG')
        count = count + 1
        continue
                                    
        
                
            
        