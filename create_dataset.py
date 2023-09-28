import os
import random
import shutil

# Define the path to your dataset folder
dataset_path = os.getcwd()

import argparse
parser = argparse.ArgumentParser(description='Create a dataset')
parser.add_argument('--data', type=str, default='data', help='path to data folder: /data/images/')
args = parser.parse_args()


# Define the path to your data folder
data_path = os.path.join(dataset_path, args.data)

# get the list of all the subfolders in the data folder
subfolders = os.listdir(data_path)
# Define the paths to the two subfolders
for i in range(len(subfolders)):
    subfolders[i] = os.path.join(data_path, subfolders[i])

# Define the paths to the train and test subfolders inside the dataset folder
train_path = os.path.join(dataset_path, "dataset/train")
test_path = os.path.join(dataset_path, "dataset/test")

# Define the paths to the subfolders inside train and test folders
train_path_list = dict()
test_path_list = dict()
for sf in subfolders:
    train_path_list[sf] = os.path.join(train_path, os.path.basename(sf))
    test_path_list[sf] = os.path.join(test_path, os.path.basename(sf))

# Delete the train and test folders if they already exist
if os.path.exists(train_path):
    shutil.rmtree(train_path)
if os.path.exists(test_path):
    shutil.rmtree(test_path)

# Create the train and test directories
for path in train_path_list.values():
    os.makedirs(path)
for path in test_path_list.values():
    os.makedirs(path)


# Define the size of the train set as a percentage of the total data
train_size = 0.8

# loop through the subfolders inside the data folder
for subfolder_name in subfolders:
    file_names = os.listdir(os.path.join(data_path, subfolder_name))
    random.shuffle(file_names)
    train_file_names = file_names[:int(len(file_names)*train_size)]
    test_file_names = file_names[int(len(file_names)*train_size):]
    for file_name in train_file_names:
        src_path = os.path.join(data_path, subfolder_name, file_name)
        dst_path = os.path.join(train_path_list[subfolder_name], file_name)
        shutil.copy(src_path, dst_path)
    for file_name in test_file_names:
        src_path = os.path.join(data_path, subfolder_name, file_name)
        dst_path = os.path.join(test_path_list[subfolder_name], file_name)
        shutil.copy(src_path, dst_path)
