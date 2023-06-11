#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import numpy as np
import pandas as pd

from pycaret.regression import load_model, predict_model,setup

file="https://raw.githubusercontent.com/cchuang2009/2022-1/main/Python_IM/2023-2/data/FuelConsumptionCo2.csv"

df = pd.read_csv(file)

num_cols=['ENGINESIZE', 'CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY',          'FUELCONSUMPTION_COMB','FUELCONSUMPTION_COMB_MPG']
cat_cols=['MAKE','VEHICLECLASS','TRANSMISSION','FUELTYPE']
target=['CO2EMISSIONS']
model_cat = load_model('model_pkl')
st.title('CO2 Emission of Vehicles')

st.markdown("## 汽車二氧化碳排放量")
st.subheader('CO2 Emission of Vehicle')

features = num_cols + cat_cols

# for num_cols
col_values = []
for col in num_cols:
    col_value = st.slider(col, min_value=float(df[col].min()), max_value=float(df[col].max()), value=float(df[col].median()))
    col_values.append(col_value)
num_values = [col_value for col_value in col_values if isinstance(col_value, (int, float))]    

# for cat_cols
cat_values = []    
for col in cat_cols:
    ops = list(df[col].unique())   
    cat_value = st.selectbox(col, options=ops, index=0)
    cat_values.append(cat_value)
cat_values = [cat_value for cat_value in cat_values if isinstance(cat_value, str)]

final_features = np.array(num_values + cat_values).reshape(1, -1)
if st.button('Estimate'):
    new_data=pd.DataFrame(data=final_features,columns=num_cols + cat_cols)
    prediction=predict_model(estimator=model_cat, data=new_data)
    st.balloons()
    result=int(prediction['prediction_label'][0])
    st.success(
        f' Estimated CO2 Emission is {result}')


# In[ ]:





# In[ ]:




