import numpy as np
import os
from PIL import Image, ImageFilter
import sys
import cv2
import glob

def load_paths(directory):
    paths = []
    for files in os.listdir(directory):
        if (files != ".DS_Store"):
            paths.append(directory+'/'+files)
    return paths

def blur(image):
    
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.fastNlMeansDenoisingColored(image,None,10,48,7,5)
    dst = cv2.filter2D(dst,-1,kernel)
    #blur = cv2.bilateralFilter(image,9,75,75)
    #blur = cv2.blur(image,(5,5))
    #blur = cv2.GaussianBlur(image,(5,5),0)
    return dst

def resize_and_save(path,size,directory):
    try:
        img = Image.open(path)
        img = img.resize(size)
        img.save(path)
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        blurred = blur(im)
        img = Image.fromarray(blurred)
        #img = img.filter(ImageFilter.SHARPEN)
        img.save(directory)
    except IOError:
        print "cannot create thumbnail for '%s'" % path

for i in range(1,11):
    directory = 'Val_Set_Blurred_'+str(i)
    if (not os.path.exists('Val_Set_Blurred_'+str(i))):
        for sign in load_paths("Traffic_Signs_Templates/1_Input_Images"):
            head,tail = sign.split('.')
            name = []
            name = head.split('/')
            os.makedirs('Val_Set_Blurred_'+str(i)+'/'+name[-1])




    paths = load_paths("Val_Set_"+str(i))
    for path in paths:
        temp = load_paths(path)
        for p in temp:
            elements = p.split('/')
            d = 'Val_Set_Blurred_'+str(i)+'/'+elements[-2]+"/"+elements[-1]
            resize_and_save(p,(48,48), d)

"""

directory = 'SGTSD/Images_Blurred/'
if (not os.path.exists('SGTSD/Images_Blurred/')):
    for sign in load_paths("Traffic_Signs_Templates/1_Input_Images"):
        head,tail = sign.split('.')
        name = []
        name = head.split('/')
        os.makedirs('SGTSD/Images_Blurred/'+name[-1])


paths = load_paths("SGTSD/1_Input_Images/")
classi = 0
for path in paths:
    print ("Processing class: "+str(float(classi)/len(paths)))
    classi = classi + 1
    temp = load_paths(path)
    for p in temp:
        elements = p.split('/')
        d = 'SGTSD/Images_Blurred/'+elements[-2]+"/"+elements[-1]
        resize_and_save(p,(48,48), d)

"""
