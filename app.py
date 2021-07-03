import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import os
import joblib
import hashlib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from model import*
matplotlib.use('Agg')

def load_model(model):
	loaded_model = joblib.load(open(os.path.join(model),"rb"))
	return loaded_model

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

def main():
    st.title("WAREHOUSEIFY")
    menu = ["Home","Signup","Login"]
    submenu = ["Dataset","Prediction"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice == "Home":
         st.subheader("Home")
         st.markdown(f'<div class="markdown-text-container stText" style="width: 698px;"><div style="font-size: medium;">Most products lose their market value (outdate) over time. Some products lose valuefaster than others; these are known as perishable products. Traditionally, perishables outdate due to their chemical structure. Examples of such perishable products are grocery, fresh produce, frozen products, dairy products, delicassens etc. So  This application is used to predict the demands required in a warehouse for a short period of time. So, we aim to analyse the data regarding sales and production and to extract the daily data from warehouse which is used to detect the fluctuations in the sales. The web application uses machine learning algorithms to predict the requirements for production which helps in managing the warehouse. </div>',unsafe_allow_html=True)
         st.text(" ")
         st.image('https://3.imimg.com/data3/UK/AH/MY-8266344/fruits-and-vegetables-storage-500x500.jpg')
    elif choice == "Signup":
         st.text("Username")
         new_username = st.text_input("")
         st.text("Password")
         new_password = st.text_input(" ",type='password')
         st.text("Confirm Password")
         confirm_password = st.text_input("  ", type='password')
         if new_password == confirm_password:
            st.success("Password Confirmed")
         else:
            st.warning("Passwords not the same")
         if st.button("Submit"):
            st.success("Your account was created successfully")
            st.info("Login to get started")
            pass
    elif choice == "Login":
         username = st.sidebar.text_input("Username")
         password = st.sidebar.text_input("Password",type='password')
         if st.sidebar.checkbox("Login"):
            if password == "12345":
               st.subheader("Welcome {}".format(username))
               st.text("Activity")
               activity = st.selectbox(" ",submenu)
               if activity == "Dataset":
                  st.subheader("Our Dataset")
                  df=pd.read_csv(r'C:\Users\Shivaani\Desktop\Dataset.csv')
                  st.write(df)
                  Analytics = "Show Data Analysis"
                  if st.checkbox(Analytics):
                    fig, ax = plt.subplots(figsize=(10,10))
                    sns.heatmap(df.corr(), annot=True, ax=ax)
                    st.pyplot()
                    st.text('Effect of the different classes')
                    sns.pairplot(df, vars=['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'], hue='Channel')
                    st.pyplot()
                    st.line_chart(df)
               elif activity == "Prediction":
                   st.header("""*PREDICTION*""")
                   if st.checkbox('Demand for next 10 weeks'):
                      df=pd.read_csv(r'C:\Users\Shivaani\Desktop\Dataset.csv')
                      table = df['Fresh'].values
                      mean=0
                      for i in range (100):
                        mean = mean+table[i]
                        i=i+1
                      total1 = mean/10
                      table = df['Milk'].values
                      mean=0
                      for i in range (100):
                        mean = mean+table[i]
                        i=i+1
                      total2 = mean/10
                      table = df['Grocery'].values
                      mean=0
                      for i in range (100):
                        mean = mean+table[i]
                        i=i+1
                      total3 = mean/10
                      table = df['Frozen'].values
                      mean=0
                      for i in range (100):
                        mean = mean+table[i]
                        i=i+1
                      total4 = mean/10
                      table = df['Detergents_Paper'].values
                      mean=0
                      for i in range (100):
                        mean = mean+table[i]
                        i=i+1
                      total5 = mean/10
                      table = df['Delicassen'].values
                      mean=0
                      for i in range (100):
                        mean = mean+table[i]
                        i=i+1
                      total6 = mean/10
                      data = pd.DataFrame({'products': ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergenst_Paper', 'Delicassen'],'demand_requirements': [total1, total2, total3, total4, total5, total6],})
                      st.write(data)
                      fig = alt.Chart(chartsize =(10, 7)) 
                      st.write(alt.Chart(data).mark_bar().encode(x=alt.X('products', sort=None),y='demand_requirements',))

                   if st.checkbox('Demand for next 20 weeks'):
                      table = df['Fresh'].values
                      mean=0
                      for i in range (200):
                        mean = mean+table[i]
                        i=i+1
                      total7 = mean/20
                      ans1 = total7+total1
                      table = df['Milk'].values
                      mean=0
                      for i in range (200):
                        mean = mean+table[i]
                        i=i+1
                      total8 = mean/20
                      ans2 = total8+total2
                      table = df['Grocery'].values
                      mean=0
                      for i in range (200):
                        mean = mean+table[i]
                        i=i+1
                      total9 = mean/20
                      ans3 = total9+total3
                      table = df['Frozen'].values
                      mean=0
                      for i in range (200):
                        mean = mean+table[i]
                        i=i+1
                      total10 = mean/20
                      ans4 = total10+total4
                      table = df['Detergents_Paper'].values
                      mean=0
                      for i in range (200):
                        mean = mean+table[i]
                        i=i+1
                      total11 = mean/20
                      ans5 = total11+total5
                      table = df['Delicassen'].values
                      mean=0
                      for i in range (200):
                        mean = mean+table[i]
                        i=i+1
                      total12 = mean/20
                      ans6 = total12+total6
                      data = pd.DataFrame({'products': ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergenst_Paper', 'Delicassen'],'demand_requirements': [ans1, ans2, ans3, ans4, ans5, ans6],})
                      st.write(data)
                      fig = alt.Chart(chartsize =(10, 7)) 
                      st.write(alt.Chart(data).mark_bar().encode(x=alt.X('products', sort=None),y='demand_requirements',))
        

                     
                  


if __name__ == '__main__':
    main()