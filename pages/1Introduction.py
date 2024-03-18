#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
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

        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []

        model = Net().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
        momentum=0.9)
        scheduler = StepLR(optimizer, step_size=6, gamma=0.5)

        EPOCHS = 3
        for epoch in range(EPOCHS):
            st.write("EPOCH:", epoch)
            train(model, device, train_loader, optimizer, epoch)
            scheduler.step()
            st.write('current Learning Rate: ', optimizer.state_dict()["param_groups"][0]["lr"])
            test(model, device, test_loader)

        train_losses1 = [float(i.cpu().detach().numpy()) for i in
        train_losses]
        train_acc1 = [i for i in train_acc]
        test_losses1 = [i for i in test_losses]
        test_acc1 = [i for i in test_acc]
        fig, axs = plt.subplots(2,2,figsize=(16,10))
        axs[0, 0].plot(train_losses1,color='green')
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(train_acc1,color='green')
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses1)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(test_acc1)
        axs[1, 1].set_title("Test Accuracy")
        st.pyplot(fig)
    
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get data
        data, target = data.to(device), target.to(device)
        # Initialization of gradient
        optimizer.zero_grad()
        # In PyTorch, gradient is accumulated over backprop and even though thats used in RNN generally not used in CNN
        # or specific requirements
        ## prediction on data
        y_pred = model(data)
        # Calculating loss given the prediction
        loss = F.nll_loss(y_pred, target)
        train_losses.append(loss)
        # Backprop
        loss.backward()
        optimizer.step()
        # get the index of the log-probability corresponding to the max value
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target,
            reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{}({:.2f}%)\n'.format(test_loss,
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    test_acc.append(100. * correct / len(test_loader.dataset))




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
