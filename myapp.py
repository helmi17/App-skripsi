#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import cv2
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, GlobalAveragePooling2D,BatchNormalization
import keras
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,precision_score,recall_score,f1_score
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import StratifiedKFold
import tk
from tkinter.filedialog import askopenfilename
import seaborn as sns
from PIL import Image, ImageOps
from streamlit_option_menu import option_menu
from matplotlib.pyplot import cm

# In[ ]:

def mymodel():
    model = tf.keras.Sequential()
    model.add(Conv2D(128,(3,3), activation="relu", padding="same", input_shape=(128, 128, 3)))
    model.add(Conv2D(128,(3,3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256,(3,3), activation="relu", padding="same"))
    model.add(Conv2D(256,(3,3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(384,(3,3), activation="relu", padding="same"))
    model.add(Conv2D(256,(3,3), activation="relu", padding="same"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(tf.keras.optimizers.SGD(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
model = mymodel()
def getData():
    idx = np.load('indexnew.npy', allow_pickle=True)
    akurasi=[]
    c =1
    for train_indek, val_indek in idx:
#         print("training fold ke - {}".format(c))
        x_train = x[train_indek]
        y_train = y[train_indek]
        x_val = x[val_indek]
        y_val = y[val_indek]
    #     training (x_train,y_train,x_val,y_val,c,filename)
        c +=1
    return x_val,y_val
def load():
    citra = []
    label = []
    kelas =[]
    directory = "dataaug"
    c=0
    for data in os.listdir(directory):
        kelas.append(data)
#         print(kelas)
        for dataset in os.listdir(os.path.join(directory,data)):
            file = os.path.join(directory,data,dataset)
            image = cv2.imread(file)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            citra.append(image)
            label.append(c)
        c+=1
    # print(kelas)
    data_citra = np.array(citra)
    label_citra = np.array(label)
    # plt.imshow(data_citra[0])
#     print(label_citra.shape)
    return data_citra,label_citra
x,y =load()
def nilai2persen(i):
    nilai = (str(i*100)[:5]+"%")
    return nilai

image = Image.open("Logo utm terbaru.png")
st.sidebar.image(image,width=210)
sidebar = st.sidebar.header("Tugas akhir")
st.sidebar.subheader("M. Alauddin Helmi (170411100115)")
with st.sidebar:
    selected = option_menu(
        menu_title="Main menu",
        options=["Home","Models","Klasifikasi"],
    )
if selected == "Home":
    st.header("Klasifikasi penyakit tanaman jagung menggunakan CNN")
    image = Image.open ("093416100_1563978081-iStock-599971330.jpg")
    st.image(image,width=665)
if selected == "Models":
    st.header("Accuracy trained models")
    tampilkan = st.button("tampilkan accuracy")
    if tampilkan:
        x_val,y_val=getData()
        model.load_weights('trainmodel.hdf5')
        prediksi = model.predict(x_val)
        y_prediksi=np.argmax(prediksi,axis=1)
    #     print(confusion_matrix(y_true=y_val, y_pred=y_prediksi))

        presisi = precision_score(y_val, y_prediksi,average='macro')
        recall = recall_score(y_val, y_prediksi,average='macro')
        f1 = f1_score(y_val, y_prediksi,average='macro')
        acc = accuracy_score(y_val, y_prediksi)

        st.write("=========================================================================================")
        st.write("precission :",nilai2persen(presisi))
        st.write("reacall :",nilai2persen(recall))
        st.write("F1-score :",nilai2persen(f1))
        st.write("Classification accuracy :",nilai2persen(acc))
        st.write("=========================================================================================")
        n_test = y_val.shape[0]
        a_test = x.shape[1:4]
        st.write("Data validasi yang digunakan sebanyak : {}".format(n_test))
        st.write("Each image is of size: {}".format(a_test))
if selected == "Klasifikasi":

    selected = option_menu(
    menu_title="Preprocessing",
    options=["Citra asli","thresholding","Ratakan mask","Hasil"],
        menu_icon = "cast",
        default_index = 0,
    orientation="horizontal",
    )
    uploaded_files = st.sidebar.file_uploader("Choose a jpg file", accept_multiple_files=True)
    for upload_file in uploaded_files:
        bytes_data = upload_file.read()
        with open(os.path.join("Temp",upload_file.name),"wb")as f:
            gg = f.write(upload_file.getbuffer())
            simpanmor = "Temp"+"/"+(upload_file.name)[:-5]+"Z.jpg"
            kampret = upload_file
            save = (simpanmor)[:-5]+"M.jpg"
            simpanmor = "Temp"+"/"+(upload_file.name)[:-5]+"M.jpg"
            simp = "Temp"+"/"+(upload_file.name)[:-5]+"G.jpg"
            if selected == "Citra asli":
                st.write("=========================================================================================")
                asli = Image.open(kampret)
                st.image(asli, caption='Citra asli',width=250)
                st.write("=========================================================================================")

        if selected == "thresholding":
            st.write("=========================================================================================")
            directir = ("Temp"+"/"+upload_file.name)
            citraasli = cv2.imread(directir)
            rgb = cv2.cvtColor(citraasli,cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            lower = np.array([12, 70, 70])
            upper = np.array([106, 255, 230])
            mask = cv2.inRange(hsv, lower, upper)
            simpanmor = "Temp"+"/"+(upload_file.name)[:-5]+"Z.jpg"
            cve = cv2.imwrite(simpanmor,mask)
            trs = Image.open(simpanmor)
            st.image(trs, caption='Citra mask',width=250)
            st.write("=========================================================================================")
        if selected == "Ratakan mask":
            st.write("=========================================================================================")
            asw = ("Temp"+"/"+upload_file.name)
            citraasli = cv2.imread(asw)
            rgb = cv2.cvtColor(citraasli,cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            lower = np.array([12, 70, 70])
            upper = np.array([106, 255, 230])
            mask = cv2.inRange(hsv, lower, upper)
            kernel = np.ones((70,20), np.uint8)
            morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            # newimg2=np.where(morphed[...,None]!=0,rgb,[0,0,0])
            save = (simpanmor)[:-5]+"M.jpg"
            cve = cv2.imwrite(save,morphed)
            tr = Image.open(save)
            st.image(tr, caption='Hasil perataan mask dengan operasi closing',width=250)
            st.write("=========================================================================================")
        if selected == "Hasil":
            st.write("=========================================================================================")
            directir = ("Temp"+"/"+upload_file.name)
            citraasli = cv2.imread(directir)
            rgb = cv2.cvtColor(citraasli,cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            lower = np.array([12, 70, 70])
            upper = np.array([106, 255, 230])
            mask = cv2.inRange(hsv, lower, upper)

            kernel = np.ones((70,20), np.uint8)
            morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            newimg=np.where(morphed[...,None]!=0,rgb,[0,0,0])
            newimg2 = newimg.astype("uint8")
            simp = "Temp"+"/"+(upload_file.name)[:-5]+"G.jpg"
            cve = plt.imsave(simp,newimg2)
            trs = Image.open(simp)
            st.image(trs, caption='Hasil penggabungan citra asli dan citra mask',width=250)
            st.write("=========================================================================================")
            cls = st.button("klasifikasi")
            if cls :
                read = cv2.imread(simp)
                img = cv2.cvtColor(read,cv2.COLOR_BGR2RGB)
                dims=np.expand_dims(img,axis=0)
                model.load_weights('trainmodel.hdf5')
                prediksi = model.predict(dims)
                acc=np.argmax(prediksi,axis=1)
                for i in acc :
                    if i == 0 :
                        st.warning('Batang mengalami busuk Anthracnose !!!')
                    elif i == 1 :
                        st.warning("Batang mengalami busuk Gibberella !!!")
                    else :
                        st.success('Batang sehat :)')
