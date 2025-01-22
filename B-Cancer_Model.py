#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 07:38:35 2025

@author: chrisbecil
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd
import requests

url ='https://github.com/Abudho/Trained_Model_Cancer/blob/main/trained_model_cancer.sav'
loaded_model = requests.get(url)

with open ('trained_model_cancer.sav','wb') as f:
    pickle.dump(loaded_model,f)
    
with open ('trained_model_cancer.sav','rb') as f:
    loaded_model = pickle.load(f)
    
def breast_cancer_prediction(input_data):
    input_data_as_numpy_array = np.array(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape (1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0]==0:
        return "The person does not have Breast Cancer"
    else:
        return "The person has Breast Cancer"
    
def main():
    st.title("Breast Cancer Prediction Machine Learning Model")
    radius =st.text_input('Enter the radius of the cancer cell (1,2,3..)') 
    texture =st.text_input('Enter the texture of the cancer cell (1,2,3..)')
    perimeter =st.text_input('Enter the perimeter of the cancer cell (1,2,3..)')
    area =st.text_input('Enter the area of the cancer cell (1,2,3..)')
    smoothness =st.text_input('Enter the smoothnes of the cancer cell (1,2,3..)')
    
    
    radius =pd.to_numeric(radius,errors ='coerce')
    texture = pd.to_numeric(texture,errors ='coerce')
    perimeter =pd.to_numeric(perimeter,errors ='coerce')
    area = pd.to_numeric(area,errors ='coerce')
    smoothness = pd.to_numeric(smoothness,errors ='coerce')
    
    
    diagnosis = ''
    if st.button ("PREDICT"):
        diagnosis = breast_cancer_prediction([radius,texture,perimeter,area,smoothness])
    st.success(diagnosis)