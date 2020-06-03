import glob

import PIL.Image
import numpy as np
from numpy import asarray
from resize import resize_image
from sklearn.preprocessing import StandardScaler


def flatten(image):
    result = []
    for mat in image:
        for line in mat:
            for el in line:
                result.append(el)
    return result

def loadSepia(output,inputImageList):
    for filename in glob.glob('D:\\utils\\faculta\\sem4\\AI\\Laborator\\Lab10\\imgs\\sepia\\/*.jpg'):

        imagine = PIL.Image.open(filename)
        imagine = imagine.resize((60,60), PIL.Image.ANTIALIAS)
        data = asarray(imagine)

        inputImageList.append(data)
        output.append(1)
    return inputImageList, output


def loadNormal(output,inputImageList):
    for filename in glob.glob('D:\\utils\\faculta\\sem4\\AI\\Laborator\\Lab10\\imgs\\color\\/*.jpg'):

        imagine = PIL.Image.open(filename)
        imagine = imagine.resize((60,60), PIL.Image.ANTIALIAS)
        data = asarray(imagine)

        inputImageList.append(data)
        output.append(0)
    return inputImageList, output
