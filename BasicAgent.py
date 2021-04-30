from copy import deepcopy

import PIL
from PIL import ImageOps
from PIL.Image import Image
from PIL.Image import fromarray
from AutoAgent import AutoAgent, colorDistance
from collections import Counter
import numpy as np


###########################################################
# Credits to: PRATEEKAGRAWAL
# https://www.analyticsvidhya.com/blog/2021/03/a-beginners-guide-to-image-similarity-using-python/
###########################################################

def compare2Images(Image1, Image2):
    """
    Compare 2 images

    :param Image1:
    :param Image2:
    :return: int, representing euclidean distance between the images
    """
    Image1_arr = np.asarray(Image1)
    Image2_arr = np.asarray(Image2)

    flat1 = Image1_arr.flatten()
    flat2 = Image2_arr.flatten()

    RH1 = Counter(flat1)
    RH2 = Counter(flat2)

    H1 = normalizeHistogramVector(RH1)
    H2 = normalizeHistogramVector(RH2)
    return L2Norm(H1, H2)


def normalizeHistogramVector(RH):
    histNorm = []
    for i in range(256):
        if i in RH.keys():
            histNorm.append(RH[i])
        else:
            histNorm.append(0)

    return histNorm


def L2Norm(H1, H2):
    distance = 0
    for i in range(len(H1)):
        distance += np.square(H1[i] - H2[i])
    return np.sqrt(distance)


###########################################################

def regionGetPixels(photo: Image, i, j):
    """
    From a given set of coords, generate 3x3 region around that coord

    :param photo:
    :param i:
    :param j:
    :return:
    """
    width, length = photo.size
    photo = photo.convert("RGB")

    if i == 0 or j == 0 or j == width - 1 or j == length - 1:
        return

    pixels = [
        [photo.getpixel((i - 1, j - 1)), photo.getpixel((i - 1, j)), photo.getpixel((i - 1, j + 1))],
        [photo.getpixel((i, j - 1)), photo.getpixel((i, j)), photo.getpixel((i, j + 1))],
        [photo.getpixel((i + 1, j - 1)), photo.getpixel((i + 1, j)), photo.getpixel((i + 1, j + 1))],
    ]

    array = np.array(pixels, dtype=np.uint8)
    patch = fromarray(array)
    return patch


def get6patches(patch1, img2):
    width, length = img2.size
    results = []

    for i in range(1, width - 2):
        for j in range(1, length - 2):
            patch2 = regionGetPixels(img2, i, j)
            if len(results)<6:
                results.append(patch2)
            else:
                distArr = []
                imgDist = compare2Images(patch2, patch2)
                for m in range(len(results)):
                    dist = compare2Images(patch1, results[m])
                    distArr.append(dist)
                for n in range(len(results)):
                    if imgDist<distArr[n]:
                        results[n] = patch2
                        break

    return results


class BasicAgent(AutoAgent):
    """
    BasicAgent

    Executes the basic coloring agent as described by Dr Cowan

    Returns right half of image
    """

    def __init__(self, img: Image, numColors=5):
        super().__init__(img, numColors)

    def execute(self):
        rightWidth, rightLen = self.rightHalf.size
        grayLeftHalf = ImageOps.grayscale(self.leftHalf)
        grayRightHalf = ImageOps.grayscale(self.rightHalf)

        resultRightHalf = PIL.Image.new(self.rightHalf.mode, self.rightHalf.size)
        pixelMap = resultRightHalf.load()

        for i in range(1, rightWidth - 2):
            for j in range(1, rightLen - 2):
                patch = regionGetPixels(grayRightHalf, i, j)
                six_patches = get6patches(patch, grayLeftHalf)
                distances = []
                for n in range(len(six_patches)):
                    leftPixel = six_patches[n].getpixel((2, 2))
                    rightPixel = patch.getpixel((2, 2))
                    distances.append(colorDistance(leftPixel, rightPixel))
                minDist = rightWidth * rightLen
                minDistIndex = len(distances)+1
                for m in range(len(distances)):
                    if distances[m] < minDist:
                        minDistIndex = m
                        minDist = distances[minDistIndex]
                pixelMap[i, j] = six_patches[minDistIndex].getpixel((2, 2))

        return resultRightHalf
