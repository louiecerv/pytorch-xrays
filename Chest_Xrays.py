#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
import time

# Define the Streamlit app
def app():

    if "reset_app" not in st.session_state:
        st.session_state.reset_app = False

    text = """Insert Your App Title Here"""
    st.subheader(text)

    # Use session state to track the current form

    if "clf" not in st.session_state: 
        st.session_state["clf"] = []

    if "X_train" not in st.session_state: 
        st.session_state["X_train"] = []

    if "X_test" not in st.session_state: 
        st.session_state["X_test"] = []
    
    if "y_train" not in st.session_state: 
        st.session_state["X_train"] = []
    
    if "y_test" not in st.session_state: 
        st.session_state["y_yest"] = []

    if "selected_model" not in st.session_state: 
        st.session_state["selected_model"] = 0
    
    if "mnist" not in st.session_state: 
        st.session_state["mnist"] = []

    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('MNIST.png', caption="Replace he image and replace this text with the description""")

    if "dataset_ready" not in st.session_state:
        # Create a progress bar object
        progress_bar = st.progress(0, text="Loading the dataset please wait...")

        # replace with your dataset
        # Load MNIST dataset
        #st.session_state.mnist = fetch_openml('mnist_784', version=1, data_home=".", return_X_y=True)

        for i in range(100):
            # Update progress bar value
            progress_bar.progress(i + 1)
            # Simulate some time-consuming task (e.g., sleep)
            time.sleep(0.01)

        # Progress bar reaches 100% after the loop completes
        st.success("Dataset loading completed!")
        st.session_state.dataset_ready = True

 
    """
    # Extract only the specified number of images and labels
    size = 10000
    X, y = st.session_state.mnist
    X = X[:size]
    y = y[:size]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #save the values to the session state    
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test    
    """

    st.sidebar.subheader('Select the classifier')

    # Create the selection of classifier
    clf = DecisionTreeClassifier()
    options = ['Decision Tree', 'Random Forest Classifier', 'Extreme Random Forest Classifier', 'K-Nearest Neighbor']
    selected_option = st.sidebar.selectbox('Select the classifier', options)
    if selected_option =='Random Forest Classifier':
        clf = RandomForestClassifier(n_jobs=2, random_state=0)
        st.session_state['selected_model'] = 1
    elif selected_option=='Extreme Random Forest Classifier':        
        clf = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0)
        st.session_state['selected_model'] = 2
    elif selected_option == 'K-Nearest Neighbor':
        clf = KNeighborsClassifier(n_neighbors=5)
        st.session_state['selected_model'] = 3
    else:
        clf = DecisionTreeClassifier()
        st.session_state['selected_model'] = 0

    # save the clf to the session variable
    st.session_state['clf'] = clf
    
#run the app
if __name__ == "__main__":
    app()
