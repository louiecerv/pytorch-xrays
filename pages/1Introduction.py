#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from IPython.display import display
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from tqdm import tqdm

class_name = ['NORMAL','PNEUMONIA']

def get_list_files(dirName):
    """
    input - directory location
    output - list the files in the directory
    """
    files_list = os.listdir(dirName)
    return files_list

# Define the Streamlit app
def app():

    st.subheader('Chest X-ray Classifier using Pytorch')
    text = """Insert App Decription"""
    st.write(text)

    data_path = 'dataset'
    files_list_normal_train = get_list_files(data_path + '/training_set/'+class_name[0])
    files_list_pneu_train = get_list_files(data_path+'/training_set/'+class_name[1])
    files_list_normal_test = get_list_files(data_path+'/test_set/'+class_name[0])
    files_list_pneu_test = get_list_files(data_path+'/test_set/'+class_name[1])
    st.write("Number of train samples in Normal category {}".format(len(files_list_normal_train)))
    st.write("Number of train samples in Pneumonia category {}".format(len(files_list_pneu_train)))
    st.write("Number of test samples in Normal category {}".format(len(files_list_normal_test)))
    st.write("Number of test samples in Pneumonia category {}".format(len(files_list_pneu_test)))


    rand_img_no = np.random.randint(0,len(files_list_normal_train))
    img = data_path + '/training_set/NORMAL/'+ files_list_normal_train[rand_img_no]
    print(plt.imread(img).shape)
    img = mpimg.imread(img)
    # Create a figure and an axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img)
    ax.set_title('Randomly Selected Sample Image from the NORMAL Class')
    st.pyplot(fig)

    img = data_path + '/training_set/PNEUMONIA/'+ files_list_pneu_train[np.random.randint(0,len(files_list_pneu_train))]
    print(plt.imread(img).shape)
    img = mpimg.imread(img)
    # Create a figure and an axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img)
    ax.set_title('Randomly Selected Sample Image from the PNEUMONIA Class')
    st.pyplot(fig)

    train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(os.path.join(data_path, 'train_set'), transform= train_transform)
    test_data = datasets.ImageFolder(os.path.join(data_path, 'test_set'), transform= test_transform)

    train_loader = DataLoader(train_data, batch_size= 16, shuffle= True, pin_memory= True)
    test_loader = DataLoader(test_data, batch_size= 1, shuffle= False, pin_memory= True)
    class_names = train_data.classes
    print(class_names)
    st.write(f'Number of train images: {len(train_data)}')
    st.write(f'Number of test images: {len(test_data)}')

#run the app
if __name__ == "__main__":
    app()
