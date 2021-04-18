import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import cv2


###########################################################
# Credits to: Juan De Dios Santos
###########################################################
def compute_histogram(model):
    labels_list = np.arange(0, 5 + 1)
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

    def __init__(self, img: Image):
        self.img = img
        self.width, self.height = img.size
        self.leftHalf = img.crop((0, 0, self.width / 2, self.height))
        self.rightHalf = img.crop((self.width / 2, 0, self.width, self.height))
        self.colorList = []
        self.KMeansFunction()

        # self.leftHalf.show()
        # self.rightHalf.show()

    def KMeansFunction(self):
        """
        Function that takes in the first 5 most frequent colors of the left half of the image using K Means

        :return:
        """
        pilImg = self.leftHalf
        cv2Img = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_BGR2RGB)
        cv2Img = cv2Img.reshape((cv2Img.shape[0] * cv2Img.shape[1], 3))
        model = KMeans(n_clusters=5)
        model.fit(cv2Img)

        hist = compute_histogram(model)
        # Functionality to display the 5 most frequent colors
        # rect = draw_leading_color_plot(hist, model.cluster_centers_)

        # plt.axis('off')
        # plt.imshow(rect)
        # plt.show()

        self.colorList = get5LeadingColors(hist, model.cluster_centers_)

        # Display the RGB values of the 5 most frequent colors
        # for i in self.colorList:
            # print(i)
