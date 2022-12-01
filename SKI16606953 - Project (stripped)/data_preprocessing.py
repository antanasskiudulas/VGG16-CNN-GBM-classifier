import numpy as np
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import scipy.misc
import os
import shutil
from pandas.io.json import json_normalize
from PIL import Image
from os import listdir
from os.path import splitext


def jpg_to_png(folder_path):
	for img in listdir(folder_path):
		file, filetype = splitext(img)
		if filetype not in [".py", ".png"]:
			temp = Image.open(folder_path + file + filetype)
			temp.save("test/" + file + ".png")

def move_split(start_index, split, input_directory, output_directory):
	for img in range(start_index,split):
		curr_dir = input_directory + listdir(input_directory)[img]
		dest_dir = output_directory + listdir(input_directory)[img]
		shutil.copy(curr_dir, dest_dir)

def extract_all_roi(folder_path, formatted_dataset, num_of_images):
	for img in range(num_of_images):
		extract_roi(folder_path+listdir(folder_path)[img], formatted_dataset)

def extract_roi(image_path, formatted_dataset):
	temp_set = formatted_dataset.loc[formatted_dataset["filepath"] == image_path]
	uniques = temp_set.groupby("label").size()
	limit = uniques.immunopositive
	negative_counter = 0
	positive_counter = 0
	for index, record in temp_set.iterrows():
		width = record.x2 - record.x1
		height = record.y2 - record.y1
		if record.label == "immunopositive" and positive_counter <= limit:
			image = cv2.imread(record.filepath)
			cell = image[record.y1:record.y1+height,record.x1:record.x1+width]
			scipy.misc.imsave("extracted/positive/positive-"+str(index)+".png",cell)
			positive_counter = positive_counter + 1
		if record.label == "immunonegative" and negative_counter < limit:
			image = cv2.imread(record.filepath)
			cell = image[record.y1:record.y1+height,record.x1:record.x1+width]
			scipy.misc.imsave("extracted/negative/negative-"+str(index)+".png",cell)
			negative_counter = negative_counter + 1

def display_roi(image_path, formatted_dataset):
	frame = plt.figure()
	axes = frame.add_axes([0,0,1,1])
	img = plt.imread(image_path)
	temp_set = formatted_dataset.loc[formatted_dataset["filepath"] == image_path]
	for index, record in temp_set.iterrows():
		width = record.x2 - record.x1
		height = record.y2 - record.y1
		if record.label == "immunopositive":
			color = "red"
			axes.annotate("P", xy=(record.x2-40, record.y1+20))
		if record.label == "immunonegative":
			color = "green"
			axes.annotate("N", xy=(record.x2-40, record.y1+20))	
		bounding_box = patches.Rectangle((record.x1,record.y1), width, height, edgecolor = color, facecolor = 'none')
		axes.add_patch(bounding_box)
	plt.imshow(img)
	plt.show()




def data_conversion(data, folder_path, output_path, extension = None):
	formatted_dataset = pd.DataFrame(columns=["filepath","x1","y1","x2","y2","label"])
	for record in range(len(data)):
		row_list = []
		json_dict = json.loads(data.loc[record, "Label"])
		positive_label_dict = json_dict["immunopositive"]
		negative_label_dict = json_dict["immunonegative"]
		if (extension != None):
			path = output_path + data.loc[record,"External ID"][:-3]+extension
		else:
			path = output_path + data.loc[record,"External ID"]

		for box in range(len(positive_label_dict)):
			box = positive_label_dict[box]["geometry"]
			x_list = []
			y_list = []
			for point in box:
				x_list.append(point["x"])
				y_list.append(point["y"])
			x_min = min(x_list)
			y_min = min(y_list)
			x_max = max(x_list)
			y_max = max(y_list)
			formatted_dataset = formatted_dataset.append({"filepath":path
				, "x1":x_min,"y1":y_min,"x2":x_max,"y2":y_max
				,"label":"immunopositive"}
				, ignore_index=True)

		for box in range(len(negative_label_dict)):
			box = negative_label_dict[box]["geometry"]
			x_list = []
			y_list = []
			for point in box:
				x_list.append(point["x"])
				y_list.append(point["y"])
			x_min = min(x_list)
			y_min = min(y_list)
			x_max = max(x_list)
			y_max = max(y_list)
			formatted_dataset = formatted_dataset.append({"filepath":path
				, "x1":x_min,"y1":y_min,"x2":x_max,"y2":y_max
				,"label":"immunonegative"}
				, ignore_index=True)

	return formatted_dataset

dataset = pd.read_csv("dataset.csv")
dataset.pop("ID")
dataset.pop("DataRow ID")
dataset.pop("Labeled Data")
dataset.pop("Created By")
dataset.pop("Seconds to Label")
dataset.pop("Agreement")
dataset.pop("Reviews")
dataset.pop("View Label")
dataset.pop("Project Name")
dataset.pop("Created At")
dataset.pop("Dataset Name")

#this function should be ran once.
#jpg_to_png("original_images/")
formatted_dataset = data_conversion(dataset, "original_images/", "png_images/" , extension="png")
print(formatted_dataset)
#formatted_dataset.to_csv(r'dataset.txt', header=None, index=None, sep=',', mode='w')

#Getting extracts from 3 images (UNCOMMENT WHEN NEEDED TO EXTRACT AGAIN)
#extract_roi("png_images/10.5.png",formatted_dataset)
#extract_roi("png_images/10.4.png",formatted_dataset)
#extract_roi("png_images/10.3.png",formatted_dataset)

#extract_all_roi("png_images/", formatted_dataset, 21)

#ALL THE NEGATIVE CLASS MIGRATION
#moving 30% test split from extracted folder to test
#move_split(0,296, "extracted/negative/", "positive-negative/test/negative/")
#moving 60% train split from extracted folder to train
#move_split(296,888, "extracted/negative/", "positive-negative/train/negative/")
#moving 10% validation split from extracted folder to validation
#move_split(888,986, "extracted/negative/", "positive-negative/validation/negative/")
#ALL THE POSITIVE CLASS MIGRATION
#moving 30% test split from extracted folder to test
#move_split(0,296, "extracted/positive/", "positive-negative/test/positive/")
#moving 60% train split from extracted folder to train
#move_split(296,888, "extracted/positive/", "positive-negative/train/positive/")
#moving 10% validation split from extracted folder to validation
#move_split(888,986, "extracted/positive/", "positive-negative/validation/positive/")
#visualise an image
#display_roi("png_images/10.3.png",formatted_dataset)