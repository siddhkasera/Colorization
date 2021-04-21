import math

import numpy as np
import matplotlib as plt
from sklearn.cluster import KMeans
from PIL import Image, ImageOps
import cv2


###########################################################
# Credits to: Juan De Dios Santos
###########################################################
def compute_histogram(model, numColors=5):
    labels_list = np.arange(0, numColors + 1)
    # this histogram says how many pixels fall into one of the bins
    (hist, _) = np.histogram(model.labels_, bins=labels_list)
    hist = hist.astype('float')
    hist /= hist.sum()

    return hist


def draw_leading_color_plot(hist, centroids):
    # the first two values of np.zeros(...) represent the size of the rectangle
    # the 3 is because of RGB
    plot_width = 700
    plot_length = 150
    plot = np.zeros((plot_length, plot_width, 3), dtype='uint8')
    start = 0

    for (percent, color) in sorted(zip(hist, centroids), key=lambda x: x[0], reverse=True):
        end = start + (percent * plot_width)
        # append the leading colors to the rectangle
        cv2.rectangle(plot, (int(start), 0), (int(end), plot_length),
                      color.astype('uint8').tolist(), -1)
        # print(color)
        start = end

    # return the rectangle chart
    return plot


###########################################################
def colorDistance(c1, c2):
    """
    Get distance between two colors

    :param c1: RGB list representing color 1 (like [255, 255, 0] )
    :param c2: RGB list representing color 2
    :return: Float representing distance between the 2 colors
    """
    # Formula: sqrt(distance between red + distance between green + distance between blue)
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2)


def get5LeadingColors(hist, centroids):
    colorList = []
    for (percent, color) in sorted(zip(hist, centroids), key=lambda x: x[0], reverse=True):
        colorList.append([color[0], color[1], color[2]])

    return colorList


class AutoAgent:
    """
    AutoAgent

    This class is the "interface" of the agents
    """

    def __init__(self, img: Image, numColors=5):
        self.img = img
        width, height = img.size
        self.leftHalf = img.crop((0, 0, width / 2, height))
        self.rightHalf = img.crop((width / 2, 0, width, height))
        self.colorList = self.KMeansFunction(numColors)

        # self.leftHalf.show()
        # self.rightHalf.show()

        self.grayLeftHalf = ImageOps.grayscale(self.leftHalf)
        self.grayRightHalf = ImageOps.grayscale(self.rightHalf)

        self.recolorLeftHalf = self.recolorLeftHalf()
        self.recolorRightHalf = self.execute()

    def KMeansFunction(self, numColors=5):
        """
        Function that takes in the first numColors most frequent colors of the left half of the image using K Means

        :return:
        """
        pilImg = self.leftHalf
        cv2Img = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_BGR2RGB)
        cv2Img = cv2Img.reshape((cv2Img.shape[0] * cv2Img.shape[1], 3))
        model = KMeans(n_clusters=numColors)
        model.fit(cv2Img)

        hist = compute_histogram(model, numColors)
        # Functionality to display the 5 most frequent colors
        # rect = draw_leading_color_plot(hist, model.cluster_centers_)

        # plt.axis('off')
        # plt.imshow(rect)
        # plt.show()

        colorList = get5LeadingColors(hist, model.cluster_centers_)

        # Display the RGB values of the 5 most frequent colors
        print("Here are the RGB values of the colors:")
        for i in colorList:
            print(i)

        return colorList

    def recolorLeftHalf(self):
        """
        Recolors left half with the nearest rep color

        :return: Image
        """
        leftWidth, leftLen = self.leftHalf.size
        rgb_leftHalf = self.leftHalf.convert("RGB")
        recolorLeftHalf = self.leftHalf
        for x in range(leftWidth):
            for y in range(leftLen):
                r, g, b = rgb_leftHalf.getpixel((x, y))
                closestColors = sorted(self.colorList, key=lambda color: colorDistance(color, [r, g, b]))
                closest = closestColors[0]
                recolorLeftHalf.putpixel((x, y), (int(closest[0]), int(closest[1]), int(closest[2])))
                # Note: pixel RGB values in putpixel are rounded to the nearest int
        # recolorLeftHalf.show()
        return recolorLeftHalf

    def execute(self):
        """
        Function to do the recolor of the right half. Please override this in the appropriate agents

        :return: Image
        """
        pass
