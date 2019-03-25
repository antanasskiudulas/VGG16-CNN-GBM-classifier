import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb_histogram(img):
    color = ('blue', 'green', 'red')
    for i, col in enumerate(color):
        histogram = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histogram, color=col)
        plt.xlim([0, 256])

def rgb_hist_equalisation(img):
    blue, green, red = cv2.split(img)
    new_blue = cv2.equalizeHist(blue)
    new_green = cv2.equalizeHist(green)
    new_red = cv2.equalizeHist(red)
    return cv2.merge((new_blue, new_green, new_red))

def rgb_hist_stretch(img):
    for colormap in range(img.shape[2]):
        max = np.max(img[:, :, colormap].flatten())
        min = np.min(img[:, :, colormap].flatten())
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                img[row, col, colormap] = np.astype('uint8',(img[row, col, colormap]-max)/(max-min)*255)



img = mpimg.imread('Pathology Data Analysis/7.JPG').copy()

plt.figure(), plt.title("BGR channel histogram before application of equalisation"),
rgb_histogram(img)

plt.figure(), plt.title("Original image before application of equalisation")
plt.imshow(img)
plt.show()

plt.figure(), plt.title("BGR channel histogram after application of equalisation"),
rgb_histogram(rgb_hist_equalisation(img))

plt.figure(), plt.title("Image after equalisation")
plt.imshow(rgb_hist_equalisation(img))



plt.figure(), plt.title("test")
plt.imshow(rgb_hist_stretch(img))
plt.show()






