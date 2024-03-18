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
#from torchsummary import summary
import torchsummary
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
    st.write(plt.imread(img).shape)
    img = mpimg.imread(img)
    # Create a figure and an axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img)
    ax.set_title('Randomly Selected Sample Image from the NORMAL Class')
    st.pyplot(fig)

    img = data_path + '/training_set/PNEUMONIA/'+ files_list_pneu_train[np.random.randint(0,len(files_list_pneu_train))]
    st.write(plt.imread(img).shape)
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

    train_data = datasets.ImageFolder(os.path.join(data_path, 'training_set'), transform= train_transform)
    test_data = datasets.ImageFolder(os.path.join(data_path, 'test_set'), transform= test_transform)

    train_loader = DataLoader(train_data, batch_size= 16, shuffle= True, pin_memory= True)
    test_loader = DataLoader(test_data, batch_size= 1, shuffle= False, pin_memory= True)
    class_names = train_data.classes
    st.write(class_names)
    st.write(f'Number of train images: {len(train_data)}')
    st.write(f'Number of test images: {len(test_data)}')

    if st.button("Begin Training"):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        st.write("Available processor {}".format(device))
        model = Net().to(device)
        #show_model_summary(model, input_size=(3, 224, 224))
        st.write(torchsummary.summary(model, input_size=(3, 224, 224)))


def show_model_summary(model, input_size):
    """Displays the model summary in a Streamlit app.
    Args:
        model: PyTorch model object.
        input_size: Input size for the model (tuple).
    """
    import contextlib  # For capturing standard output
    st.subheader("Model Summary")

    # Capture standard output using a context manager
    with contextlib.redirect_stdout(None):  # Redirect stdout to nowhere
        summary_str = summary(model, input_size=(3, 224, 224))

    # Display the summary string
    st.code(summary_str, language="python")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Input Block
        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=8,
            kernel_size=(3, 3),
            padding=0, bias=False), nn.ReLU(),
            #nn.BatchNorm2d(4)
        )

        self.pool11 = nn.MaxPool2d(2, 2)
        # CONVOLUTION BLOCK
        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=16,
            kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            #nn.BatchNorm2d(16)
        )

        # TRANSITION BLOCK
        self.pool22 = nn.MaxPool2d(2, 2)
        self.convblock3 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        #nn.BatchNorm2d(10),
        nn.ReLU()
        )

        self.pool33 = nn.MaxPool2d(2, 2)

        # CONVOLUTION BLOCK
        self.convblock4 = nn.Sequential(
        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
        nn.ReLU(),
        #nn.BatchNorm2d(10)
        )

        self.convblock5 = nn.Sequential(
        nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        #nn.BatchNorm2d(32),
        nn.ReLU(),
        )

        self.convblock6 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        nn.ReLU(),
        #nn.BatchNorm2d(10),
        )

        self.convblock7 = nn.Sequential(
        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
        nn.ReLU(),
        #nn.BatchNorm2d(10)
        )

        self.convblock8 = nn.Sequential(
        nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        #nn.BatchNorm2d(32),
        nn.ReLU()
        )

        self.convblock9 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        nn.ReLU(),
        #nn.BatchNorm2d(10),
        )

        self.convblock10 = nn.Sequential(
        nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
        nn.ReLU(),
        #nn.BatchNorm2d(14),
        )

        self.convblock11 = nn.Sequential(
        nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
        nn.ReLU(),
        #nn.BatchNorm2d(16),
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=4)
        )

        self.convblockout = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=2, kernel_size=(4, 4), padding=0, bias=False),
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool11(x)
        x = self.convblock2(x)
        x = self.pool22(x)
        x = self.convblock3(x)
        x = self.pool33(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.convblock11(x)
        x = self.gap(x)
        x = self.convblockout(x)
        x = x.view(-1, 2)
        return F.log_softmax(x, dim=-1)

#run the app
if __name__ == "__main__":
    app()
