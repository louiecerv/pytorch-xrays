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
%matplotlib inline
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
    files_list_normal_train = get_list_files(data_path + '/train/'+class_name[0])
    files_list_pneu_train = get_list_files(data_path+'/train/'+class_name[1])
    files_list_normal_test = get_list_files(data_path+'/test/'+class_name[0])
    files_list_pneu_test = get_list_files(data_path+'/test/'+class_name[1])
    st.write("Number of train samples in Normal category {}".format(len(files_list_normal_train)))
    st.write("Number of train samples in Pneumonia category {}".format(len(files_list_pneu_train)))
    st.write("Number of test samples in Normal category {}".format(len(files_list_normal_test)))
    st.write("N`umber of test samples in Pneumonia category {}".format(len(files_list_pneu_test)))

#run the app
if __name__ == "__main__":
    app()
