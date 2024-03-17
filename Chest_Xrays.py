#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Define the Streamlit app
def app():


    text = """Louie F. Cervantes, M. Eng. (Information Engineering) \n\n
    CCS 229 - Intelligent Systems
    Computer Science Department
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('chest-xray.jpg', caption='Chest X-ray Dataset')

    text = """Insert Description of the dataset and classifier."""
    st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
