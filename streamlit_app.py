#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:35:57 2024

@author: shyamnimje
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st

# Load your trained model
with open('/Users/shyamnimje/Downloads/sav parkinson/parkinsons_model.sav', 'rb') as file:
    model = pickle.load(file)

def park_prediction(input_data):
    # Convert input values to float
    input_data_as_float = [float(value) for value in input_data]
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data_as_float)
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not suffering from Parkinson\'s disease'
    else:
        return 'The person may have Parkinson\'s disease. Consult your doctor.'

def main():
    st.title('Parkinson Prediction')

    # Use valid variable names
    MDVP_Fo_Hz = st.text_input('insert MDVP:Fo(Hz)')	
    MDVP_Fhi_Hz = st.text_input('insert MDVP:Fhi(Hz)')
    MDVP_Flo_Hz = st.text_input('insert MDVP:Flo(Hz)')
    MDVP_Jitter = st.text_input('insert MDVP:Jitter(%)')
    MDVP_Jitter_Abs = st.text_input('Insert MDVP:Jitter(Abs)')
    MDVP_RAP = st.text_input('Insert MDVP:RAP')
    MDVP_PPQ = st.text_input('Insert MDVP:PPQ')
    Jitter_DDP = st.text_input('Insert Jitter:DDP')
    MDVP_Shimmer = st.text_input('Insert MDVP:Shimmer')
    MDVP_Shimmer_dB = st.text_input('Insert MDVP:Shimmer(dB)')
    Shimmer_APQ3 = st.text_input('Insert Shimmer:APQ3')
    Shimmer_APQ = st.text_input('Insert Shimmer:APQ')
    MDVP_APQv = st.text_input('Insert MDVP:APQv')
    Shimmer_DDA = st.text_input('Insert Shimmer:DDA')
    NHR = st.text_input('Insert NHR')
    HNR = st.text_input('Insert HNR')
    RPDE = st.text_input('Insert RPDE')
    DFA = st.text_input('Insert DFA')
    spread1 = st.text_input('Insert spread1')
    spread2 = st.text_input('Insert spread2')
    D2 = st.text_input('Insert D2')
    PPE = st.text_input('Insert PPE')

    # Code for Prediction
    Prediction = ' '

    # Creating button for prediction
    if st.button('Parkinson\'s Disease Prediction:'):
        Prediction = park_prediction([MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs,
                                      MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB,
                                      Shimmer_APQ3, Shimmer_APQ, MDVP_APQv, Shimmer_DDA, NHR, HNR,
                                      RPDE, DFA, spread1, spread2, D2, PPE])

    st.success(Prediction)

if __name__ == '__main__':
    main()
