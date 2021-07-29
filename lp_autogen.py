import numpy as np
import os
from os import path
from PIL import Image
import cv2
import random
import glob
import re


root = './lpr_letters/'
if not os.path.exists(root):
	os.makedirs(root)

lp_list = ['black', 'red', 'white', 'yellow']
print(lp_list)
lp_input = 'Please choose License Plate Color:'
lp = input(lp_input)
assert lp in lp_list, 'No such License Plate Color'
print('LP Color:', format(lp))

line_list = ['SL', 'DL']
print(line_list)
line_input = "Please choose License Plate, Single Line(SL) or Double Line(DL):"
line = input(line_input)
assert line in line_list, 'No such License Plate'
print('License Plate Style:', format(line))

lpr = line + "_" + lp
print(lpr)


lp_code_list = ['S--', 'GB-', 'QE-', 'QB-', 'QC-', 'TR-', 'G-',  
				'PA', 'PC', 'PD', 'PH', 'PZ', 'X-', 'Y-']
print(lp_code_list)
lp_code_input = "Please choose vehicle class from above list for License Plate:"
lp_code = input(lp_code_input)
assert lp_code in lp_code_list, 'No such vehicle class'
print('vehicle class:', format(lp_code))


if (lp == 'black' or lp == 'red'):
	letter_color = 'white'
else:
	letter_color = 'black'

genInput = 'Please enter amount:'
gen = input(genInput)
gen = int(gen)


filename = "/home/student/Desktop/lpr_letters/%s" % lp
filename = filename + "/unmounted/char/*_%s.png" % letter_color
charfiles = glob.glob(filename)
#print(filename)
#print(lp)

filename = "/home/student/Desktop/lpr_letters/%s" % lp
filename = filename + "/unmounted/num/*_%s.png" % letter_color
Numfiles = glob.glob(filename)
#print(filename)


def generator():
	global gen, lp, line, lpr, lp_code, letter_color, charfiles, Numfiles

	while gen > 0 :

		
		if lp_code == 'S--':
			# generate S_ _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "s_%s.png" % letter_color   
			char2PATH = random.choice(charfiles)
			char3PATH = random.choice(charfiles)

		elif lp_code == 'GB-':
			# generate GB _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "g_%s.png" % letter_color   
			char2PATH = filename + "b_%s.png" % letter_color
			char3PATH = random.choice(charfiles)

		elif lp_code == 'QE-':
			# generate GB _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "q_%s.png" % letter_color   
			char2PATH = filename + "e_%s.png" % letter_color
			char3PATH = random.choice(charfiles)

		elif lp_code == 'QB-':
			# generate GB _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "q_%s.png" % letter_color   
			char2PATH = filename + "b_%s.png" % letter_color
			char3PATH = random.choice(charfiles)

		elif lp_code == 'QC-':
			# generate GC _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "q_%s.png" % letter_color   
			char2PATH = filename + "c_%s.png" % letter_color
			char3PATH = random.choice(charfiles)

		elif lp_code == 'TR-':
			# generate TR _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "t_%s.png" % letter_color   
			char2PATH = filename + "r_%s.png" % letter_color
			char3PATH = random.choice(charfiles)

		elif lp_code == 'G-':
			# generate TR _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "g_%s.png" % letter_color   
			char2PATH = random.choice(charfiles)
			char3PATH = random.choice(charfiles)

		elif lp_code == 'PA':
			# generate TR _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "p_%s.png" % letter_color   
			char2PATH = filename + "a_%s.png" % letter_color  
			char3PATH = random.choice(charfiles)
		
		elif lp_code == 'PC':
			# generate TR _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "p_%s.png" % letter_color   
			char2PATH = filename + "c_%s.png" % letter_color  
			char3PATH = random.choice(charfiles)

		elif lp_code == 'PD':
			# generate TR _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "p_%s.png" % letter_color   
			char2PATH = filename + "d_%s.png" % letter_color  
			char3PATH = random.choice(charfiles)

		elif lp_code == 'PH':
			# generate TR _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "p_%s.png" % letter_color   
			char2PATH = filename + "h_%s.png" % letter_color  
			char3PATH = random.choice(charfiles)

		elif lp_code == 'PZ':
			# generate TR _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "p_%s.png" % letter_color   
			char2PATH = filename + "z_%s.png" % letter_color  
			char3PATH = random.choice(charfiles)
		
		elif lp_code == 'X-':
			# generate TR _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "x_%s.png" % letter_color   
			char2PATH = random.choice(charfiles)
			char3PATH = random.choice(charfiles)

		elif lp_code == 'Y-':
			# generate TR _ 
			filename = "/home/student/Desktop/lpr_letters/%s" % lp
			filename = filename + "/unmounted/char/"
			char1PATH = filename + "y_%s.png" % letter_color   
			char2PATH = random.choice(charfiles)
			char3PATH = random.choice(charfiles)

		num1PATH = random.choice(Numfiles)
		while '0' in num1PATH:   # ensure first number is not 0 
			num1PATH = random.choice(Numfiles)
		num2PATH = random.choice(Numfiles)
		num3PATH = random.choice(Numfiles)
		num4PATH = random.choice(Numfiles)

		#checksum
		regex = re.compile(r'\d+')

		#Extract Char
		if(letter_color == 'black'):
			c1 = os.path.basename(char1PATH.split('_black.png',1)[0])
			c2 = os.path.basename(char2PATH.split('_black.png',1)[0])
			c3 = os.path.basename(char3PATH.split('_black.png',1)[0])
		else:
			c1 = os.path.basename(char1PATH.split('_white.png',1)[0])
			c2 = os.path.basename(char2PATH.split('_white.png',1)[0])
			c3 = os.path.basename(char3PATH.split('_white.png',1)[0])
		
		#convert to num (ASCII values)
		c1num = ord(c1)-96
		c2num = ord(c2)-96
		c3num = ord(c3)-96

		#Extract Num
		i = regex.search(num1PATH).group(0)
		i2 = regex.search(num2PATH).group(0)
		i3 = regex.search(num3PATH).group(0)
		i4 = regex.search(num4PATH).group(0)

		char8 = ['S--', 'GB-', 'QE-', 'QB-', 'QC-', 'TR-']
		char7 = ['G-', 'PA', 'PC', 'PD', 'PH', 'PZ', 'X-', 'Y-']

		if lp_code in char8: 
			#Checksum Formula
			plate_letter_list = ['a', 'z', 'y', 'x', 'u', 't', 's', 'r', 'p', 'm', 'l', 'k', 'j', 'h', 'g', 'e', 'd', 'c', 'b']
			checksum = ((c2num*9)+(c3num*4)+(int(i)*5)+(int(i2)*4)+(int(i3)*3)+(int(i4)*2)) % 19
			#print(checksum)
			postChar = plate_letter_list[checksum]
		elif lp_code in char7: 
			#Checksum Formula
			plate_letter_list = ['a', 'z', 'y', 'x', 'u', 't', 's', 'r', 'p', 'm', 'l', 'k', 'j', 'h', 'g', 'e', 'd', 'c', 'b']
			checksum = ((c1num*9)+(c2num*4)+(int(i)*5)+(int(i2)*4)+(int(i3)*3)+(int(i4)*2)) % 19
			#print(checksum)
			postChar = plate_letter_list[checksum]

		postPATH = "%s" % postChar
		postPATH = postPATH + "_%s.png" % letter_color
		filename = "/home/student/Desktop/lpr_letters/%s" % lp
		filename = filename + "/unmounted/char/"
		suffixPATH = filename + '%s' % postPATH

		filename = "/home/student/Desktop/lpr_letters/%s" % lp
		filename = filename + "/%s_BG/" % lp
		bgPATH = filename + "%s.jpg" % lpr


		
		#Image Sizing / Resizing
		bg = Image.open(bgPATH)
		if(line == 'DL'):
			width = 785
			height = 407
			bg = bg.resize((width,height),Image.LANCZOS)
		else:
			width = 1192
			height = 256
			bg = bg.resize((width,height),Image.LANCZOS)


		char1 = Image.open(char1PATH)
		w_c1, h_c1 = char1.size

		char2 = Image.open(char2PATH)
		w_c2, h_c2 = char2.size

		char3 = Image.open(char3PATH)
		w_c3, h_c3 = char3.size

		num1 = Image.open(num1PATH)
		w_n1, h_n1 = num1.size

		num2 = Image.open(num2PATH)
		w_n2, h_n2 = num2.size

		num3 = Image.open(num3PATH)
		w_n3, h_n3 = num3.size

		num4 = Image.open(num4PATH)
		w_n4, h_n4 = num4.size

		suffix = Image.open(suffixPATH)
		w_s, h_s = suffix.size
		


		# Putting the Characters and Numbers

		# Default X value for the character Image.
		same_spacing = 23
		diff_spacing = 46

		if (line == 'SL') and (lp_code in char8):
			#SingleLine
			x = (width - (w_c1 + w_c2 + w_c3 + w_n1 + w_n2 + w_n3 + w_n4 + w_s + 5*same_spacing + 2*diff_spacing)) // 2
			y = (height - 162) // 2
			bg.paste(char1,(x,y), char1.convert('RGBA'))
			bg.paste(char2,(x+w_c1+same_spacing,y), char2.convert('RGBA'))
			bg.paste(char3,(x+(w_c1+w_c2+same_spacing*2),y), char3.convert('RGBA'))

			bg.paste(num1,(x+(w_c1+w_c2+w_c3+same_spacing*2+diff_spacing),y), num1.convert('RGBA'))
			bg.paste(num2,(x+(w_c1+w_c2+w_c3+w_n1+same_spacing*3+diff_spacing),y), num2.convert('RGBA'))
			bg.paste(num3,(x+(w_c1+w_c2+w_c3+w_n1+w_n2+same_spacing*4+diff_spacing),y), num3.convert('RGBA'))
			bg.paste(num4,(x+(w_c1+w_c2+w_c3+w_n1+w_n2+w_n3+same_spacing*5+diff_spacing),y), num4.convert('RGBA'))

			bg.paste(suffix,(x+(w_c1+w_c2+w_c3+w_n1+w_n2+w_n3+w_n4+same_spacing*5+diff_spacing*2),y), suffix.convert('RGBA'))
			cv2img = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
		elif (line == 'SL') and (lp_code in char7):
			#SingleLine
			x = (width - (w_c1 + w_c2 + w_n1 + w_n2 + w_n3 + w_n4 + w_s + 4*same_spacing + 2*diff_spacing)) // 2
			y = (height - 162) // 2
			bg.paste(char1,(x,y), char1.convert('RGBA'))
			bg.paste(char2,(x+w_c1+same_spacing,y), char2.convert('RGBA'))

			bg.paste(num1,(x+(w_c1+w_c2+same_spacing*1+diff_spacing),y), num1.convert('RGBA'))
			bg.paste(num2,(x+(w_c1+w_c2+w_n1+same_spacing*2+diff_spacing),y), num2.convert('RGBA'))
			bg.paste(num3,(x+(w_c1+w_c2+w_n1+w_n2+same_spacing*3+diff_spacing),y), num3.convert('RGBA'))
			bg.paste(num4,(x+(w_c1+w_c2+w_n1+w_n2+w_n3+same_spacing*4+diff_spacing),y), num4.convert('RGBA'))

			bg.paste(suffix,(x+(w_c1+w_c2+w_n1+w_n2+w_n3+w_n4+same_spacing*4+diff_spacing*2),y), suffix.convert('RGBA'))
			cv2img = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
		elif (line == 'DL') and (lp_code in char8):
			#DoubleLine
			x1 = (width - (w_c1 + w_c2 + w_c3 + same_spacing*2)) // 2
			y1 = (height - int(162*2+15*2.3)) // 2
			bg.paste(char1,(x1,y1), char1.convert('RGBA'))
			bg.paste(char2,(x1+w_c1+same_spacing,y1), char2.convert('RGBA'))
			bg.paste(char3,(x1+w_c1+w_c2+same_spacing*2,y1), char3.convert('RGBA'))

			x2 = (width - (w_n1 + w_n2 + w_n3 + w_n4 + w_s + same_spacing*3 + diff_spacing)) // 2
			y2 = y1 + 162 + int(15*2.3)
			bg.paste(num1,(x2,y2), num1.convert('RGBA'))
			bg.paste(num2,(x2+w_n1+same_spacing,y2), num2.convert('RGBA'))
			bg.paste(num3,(x2+w_n1+w_n2+same_spacing*2,y2), num3.convert('RGBA'))
			bg.paste(num4,(x2+w_n1+w_n2+w_n3+same_spacing*3,y2), num4.convert('RGBA'))

			bg.paste(suffix,(x2+w_n1+w_n2+w_n3+w_n4+same_spacing*3+diff_spacing,y2), suffix.convert('RGBA'))
			cv2img = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
		elif (line == 'DL') and (lp_code in char7):
			#DoubleLine
			x1 = (width - (w_c1 + w_c2 + same_spacing*2)) // 2
			y1 = (height - int(162*2+15*2.3)) // 2
			bg.paste(char1,(x1,y1), char1.convert('RGBA'))
			bg.paste(char2,(x1+w_c1+same_spacing,y1), char2.convert('RGBA'))

			x2 = (width - (w_n1 + w_n2 + w_n3 + w_n4 + w_s + same_spacing*3 + diff_spacing)) // 2
			y2 = y1 + 162 + int(15*2.3)
			bg.paste(num1,(x2,y2), num1.convert('RGBA'))
			bg.paste(num2,(x2+w_n1+same_spacing,y2), num2.convert('RGBA'))
			bg.paste(num3,(x2+w_n1+w_n2+same_spacing*2,y2), num3.convert('RGBA'))
			bg.paste(num4,(x2+w_n1+w_n2+w_n3+same_spacing*3,y2), num4.convert('RGBA'))

			bg.paste(suffix,(x2+w_n1+w_n2+w_n3+w_n4+same_spacing*3+diff_spacing,y2), suffix.convert('RGBA'))
			cv2img = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)

		#print(bg.size,char1.size)

		#debug commands
		#print('Bg resize to:',bg)
		#cv2.imshow('img',cv2img)

		#save file
		skipped_suffix_list2 = ['sa','se','si','so','su']
		skipped_suffix_list3 = ['she','sky','sbs']
		first2 = c1 + c2
		first3 = c1 + c2 + c3
		if lp_code in char8: 
			print(first3.upper() + i + i2 + i3 + i4 + postChar.capitalize())

			if any(first3 in sub_list for sub_list in skipped_suffix_list3):
				print('Skipped suffix Generated! Skipping...\n')
				return generator()

			if any(first2 in sub_list for sub_list in skipped_suffix_list2):
				print('Skipped suffix Generated! Skipping...\n')
				return generator()
			
			imgfilename = lp + "_" + first3.upper() + i + i2 + i3 + i4 + postChar.capitalize()
			
		elif lp_code in char7: 
			print(first2.upper() + i + i2 + i3 + i4 + postChar.capitalize())

			if any(first2 in sub_list for sub_list in skipped_suffix_list2):
				print('Skipped suffix Generated! Skipping...\n')
				return generator()
			
			imgfilename = lp + "_" + first2.upper() + i + i2 + i3 + i4 + postChar.capitalize()

		

		if(line == 'SL'):
			if(lp == 'red'):
				root = './License Plate Generator/single_line/red/'
				if not os.path.exists(root):
					os.makedirs(root)
			if(lp == 'black'):
				root = './License Plate Generator/single_line/black/'
				if not os.path.exists(root):
					os.makedirs(root)
			if(lp == 'yellow'):
				root = './License Plate Generator/single_line/yellow/'
				if not os.path.exists(root):
					os.makedirs(root)
			if(lp == 'white'):
				root = './License Plate Generator/single_line/white/'
				if not os.path.exists(root):
					os.makedirs(root)

		else:
			if(lp == 'red'):
				root = './License Plate Generator/double_line/red/'
				if not os.path.exists(root):
					os.makedirs(root)
			if(lp == 'black'):
				root = './License Plate Generator/double_line/black/'
				if not os.path.exists(root):
					os.makedirs(root)
			if(lp == 'yellow'):
				root = './License Plate Generator/double_line/yellow/'
				if not os.path.exists(root):
					os.makedirs(root)
			if(lp == 'white'):
				root = './License Plate Generator/double_line/white/'
				if not os.path.exists(root):
					os.makedirs(root)
		

		filepath = root + imgfilename
		#print(filepath)
		if path.exists(filepath) == True:
			print('License Plate exisits in folder, making another one...')
			return generator()

		cv2.imwrite('%s.png' % filepath ,cv2img)
		print('Image saved!\n')
		

		gen = gen - 1

generator()