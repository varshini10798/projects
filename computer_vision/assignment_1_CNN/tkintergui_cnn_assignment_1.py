#%gui tk
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *

import subprocess
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import tensorflow as tf
import os
import keras
from keras.layers import Dense,Conv2D,Dropout,BatchNormalization,Activation,Flatten,MaxPool2D,Input,LeakyReLU
from keras.models import Sequential
from keras.activations import relu
from keras.optimizers import Adam,RMSprop,SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import load_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from sklearn import svm
from keras.preprocessing import image
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import cv2
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as sk
#%matplotlib inline

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
def quit():
    root.quit() 
    root.destroy()
def import_data():
    global img
    global image
    global x_pred
    install('pillow')
    from PIL import ImageTk, Image
    import_file_path = filedialog.askopenfilename()
    image = ImageTk.PhotoImage(Image.open(import_file_path))
    label1 = tk.Label(root,image=image)
    #label.pack()
    label1 = tk.Label(root,image=image)
    label1.image = image

# Position image
    label1.place(x=400, y=400)
    #canvas.create_image(100,100, image=image)
    #canvas1 = tk.Canvas(root, height=200, width=200, bg='grey')
    #canvas1.pack()
    #label = tk.Label(canvas1, image=image)
    #canvas.create_window(125,10, window=label, anchor=tk.NW)
    img = cv.imread(import_file_path)
    img1=cv.resize(img,(224,224))
    img1 = img1.reshape(1,224,224,3)
    x_pred=np.array(img1)
    x_pred=x_pred/255
    print(x_pred.shape)
    label = tk.Label(root, text= "Done")
    label.pack()
    custName.set(import_file_path)
    fileName.delete(0, "end")
    fileName.insert(0, import_file_path)
    #fileName.insert(import_file_path)
    print ('Done')
def load_downloaded_model():
    global model_trained
    import_file_path = filedialog.askopenfilename()
    model_trained=load_model(import_file_path)
    #custName.set(import_file_path)
    #fileName.delete(0, "end")
    #fileName.insert(0, import_file_path)
    #fileName.insert(import_file_path)
    custname1.set(import_file_path)
    entry1.delete(0, "end")
    entry1.insert(0, import_file_path)
    print ('Loaded')
def download_model():
    global model_trained
    #global gdown
    url = 'https://drive.google.com/uc?export=downloadh&id=1FFB0W3NGCqUTNUKohQ3QloAzuqlNAKBL'
    output = filedialog.asksaveasfilename( defaultextension=".hdf5",filetypes=[("default", "*.hdf5"),
                           ("all", "*.*")])
    
    install('gdown')
    import gdown
    gdown.download(url, output, quiet=False) 
    label = tk.Label(root, text= "Loaded Model")
    label.pack()
    model_trained=load_model(output)
      
    #custName.set(import_file_path)
    #fileName.delete(0, "end")
    #fileName.insert(0, import_file_path)
    #fileName.insert(import_file_path)
    custname1.set(import_file_path)
    entry1.delete(0, "end")
    entry1.insert(0, output)
    print ('Done')
def predict():
    global model_trained
    global x_pred
    flower_class=model_trained.predict(x_pred)
    label = tk.Label(root, text= 'Predicted Class: '+ np.argmax(flower_class).astype(str))
    label.pack()
    #fileName.insert(import_file_path)
    print ('Done')

root = tk.Tk()
root.title('Neural Network')
#root.iconbitmap('class.ico')
root.resizable(True, True)
tit = tk.Label(root, text="Neural Network", padx=25, pady=6, font=("", 12)).pack()
canvas = tk.Canvas(root, height=500, width=500, bg='grey')
canvas.pack()
frame = tk.Frame(root, bg='white')
frame.place(relwidth=2, relheight=0.8, relx=-0.1, rely=0.1)
custName = tk.StringVar()
custname1=tk.StringVar()
fileName = Entry(root, textvariable=custName)
entry1 = tk.Entry (root,textvariable=custname1)
#fileName1 = Entry(root, textvariable=custname1)
#fileName.grid(row=0, column=1)
#entry1.grid(row=1, column=1)
canvas.create_window(360, 100, window=fileName,width=300)
canvas.create_window(360, 160, window=entry1,width=300)
choose_csv= tk.Button(root, text='Import Data',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=import_data)
#choose_csv.pack(side=tk.LEFT)
canvas.create_window(100, 100, window=choose_csv,width=200)
choose_target= tk.Button(root, text='Download Trained Model and Predict',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=download_model)
#choose_target.pack(side=tk.LEFT)
canvas.create_window(100, 160, window=choose_target,width=200)
choose_target= tk.Button(root, text='Load Downloaded Model and Predict',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=load_downloaded_model)
#choose_target.pack(side=tk.LEFT)
canvas.create_window(100, 220, window=choose_target,width=200)
button = tk.Button(root, text = 'root quit', command=quit)
button.pack(side=tk.RIGHT)

train_csv_reg = tk.Button(root, text='Predict',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=predict)
train_csv_reg.pack(side=tk.LEFT)

root.mainloop()
