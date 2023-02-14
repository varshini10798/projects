

import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder
from keras.utils.np_utils import to_categorical
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.layers import BatchNormalization, Dropout,Activation
import matplotlib.pyplot as plt

def close_plot():
    global chart_type
    global chart_type1
    canvas.delete(canvas.create_window(100, 300, window=chart_type.get_tk_widget()))
    canvas.delete(canvas.create_window(300, 300, window=chart_type1.get_tk_widget()))

def quit():
    root.quit() 
    root.destroy()

def import_data():
    global df
    import_file_path = filedialog.askopenfilename()
    df = pd.read_csv (import_file_path)
    label = tk.Label(root, text= "Done")
    label.pack()
    custName.set(import_file_path)
    fileName.delete(0, "end")
    fileName.insert(0, import_file_path)
    #fileName.insert(import_file_path)
    print ('Done')

def import_target():
    global df
    global target_column
    col=entry1.get()
    if col in df.columns:
        label1 = tk.Label(root, text= "Found")
        label1.pack()
        print('Found')
        target_column=df[col]

def train_regression():
    global df
    global target_column
    global reg_model
    x_adasyn,y_adasyn=data_preprocess(df,target_column)
    reg_model=train_regmodel(x_adasyn,y_adasyn)

def data_preprocess(df1,target_column1):
    global df
    global target_column
    x=df.drop(columns='Signal_Strength')
    y=target_column
    for col in x.describe().columns:
        print(col)
        Q1 = x[col].quantile(0.25)
        Q3 = x[col].quantile(0.75)
        IQR = Q3 - Q1
        val1=(Q1 - 1.5 * IQR)
        val2=(Q3 + 1.5 * IQR)
        print(Q1,Q3)
        x.loc[x[col]<(Q1 - 1.5 * IQR),col]=val1
        x.loc[x[col]>(Q3 + 1.5 * IQR),col]=val2
    x_norm=x.apply(zscore)
    x_norm1=x_norm.drop(columns=['Parameter 4','Parameter 6','Parameter 1','Parameter 3'])
    from imblearn.over_sampling import ADASYN
    # create the object to resample theminority majority class.
    adasyn = ADASYN(sampling_strategy='minority',n_neighbors=3)
    x_adasyn = x_norm1
    y_adasyn = y
    classes=y.unique().tolist()
    for cl in classes[2:]:
        print(cl)
    # fit the object to the training data.
        x_adasyn, y_adasyn = adasyn.fit_resample(x_adasyn, y_adasyn)
    return x_adasyn,y_adasyn

def train_regmodel(x_adasyn,y_adasyn):
    global chart_type
    X_train, X_test, y_train, y_test = train_test_split(x_adasyn, y_adasyn,random_state=42)
    model = Sequential()
    model.add(Dense(20, input_dim=7, kernel_initializer='normal'))
    #model.add(BatchNormalization())  
    model.add(Activation('relu'))  
    #model.add(Dropout(0.2))
    model.add(Dense(12,kernel_initializer='normal'))
    #model.add(BatchNormalization())  
    model.add(Activation('relu'))  
    #model.add(Dropout(0.2))
    #model.add(Dense(8,kernel_initializer='he_normal'))
    #model.add(BatchNormalization())  
    #model.add(Activation('relu'))  
    model.add(Dense(1, activation='linear'))
    model.summary()
    model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=5,  verbose=1, validation_split=0.2)
    results = model.evaluate(X_test, y_test)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    label_reg = tk.Label(root, text= "Trained and Evaluated Regressor")
    label_reg.pack()
    figure = plt.Figure(figsize=(3,2), dpi=100)
    ax = figure.add_subplot(111)
    chart_type = FigureCanvasTkAgg(figure, root)
    #chart_type.draw()
    #chart_type.get_tk_widget().pack()
    canvas.create_window(100, 300, window=chart_type.get_tk_widget())
    ax.plot(train_loss, label='Training Loss')
    ax.plot(val_loss, label='Validation Loss')
    ax.legend()
    ax.set_title('Epochs vs. Training and Validation Loss-Regression',fontsize=8) 
    #ax.set_title('The Title for your chart')
    return model

def pickle_regression():
    global reg_model
    filename = filedialog.asksaveasfilename( defaultextension=".hdf5",filetypes=[("default", "*.hdf5"),
                           ("all", "*.*")])
    print(filename)
    reg_model.save(filename)
    label_picklereg = tk.Label(root, text= "Saved Regressor Model to Disk")
    label_picklereg.pack()

def train_classifier_model(x_adasyn,y_adasyn):
    global chart_type1
    pca1 = PCA(n_components=7)
    pca1.fit(x_adasyn)
    x_pca2 = pca1.transform(x_adasyn)
    x_pca3=pd.DataFrame(x_adasyn)
    oe=OrdinalEncoder()
    y2=oe.fit_transform(np.array(y_adasyn).reshape(-1,1))
    X_train, X_test, y_train, y_test = train_test_split(x_pca3, y2,random_state=42)
    y_train1 = to_categorical(y_train)
    y_test1= to_categorical(y_test)
    model2 = Sequential()
    model2.add(Dense(100, input_dim=7,kernel_initializer='glorot_normal'))
    #model1.add(BatchNormalization())  
    model2.add(Activation('relu'))  
    model2.add(Dropout(0.1))
    model2.add(Dense(100,kernel_initializer='glorot_normal'))
    #model1.add(BatchNormalization())  
    model2.add(Activation('relu'))  
    model2.add(Dropout(0.1))
    model2.add(Dense(100,kernel_initializer='glorot_normal'))
    #model1.add(BatchNormalization())  
    model2.add(Activation('relu'))
    #model1.add(Dropout(0.1))
    model2.add(Dense(100,kernel_initializer='glorot_normal'))
    #model1.add(BatchNormalization())  
    model2.add(Activation('relu'))
    model2.add(Dropout(0.1))
    model2.add(Dense(100,kernel_initializer='glorot_normal'))
    #model1.add(BatchNormalization())  
    model2.add(Activation('relu'))
    model2.add(Dropout(0.1))
    model2.add(Dense(6, activation='softmax'))
    model2.summary()
    model2.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    history = model2.fit(X_train, y_train1, epochs=100, batch_size=10,  verbose=1)
    results = model2.evaluate(X_test, y_test1)
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    label_class = tk.Label(root, text= "Trained and Evaluated  Classifier")
    label_class.pack()
    figure = plt.Figure(figsize=(3,2), dpi=100)
    ax = figure.add_subplot(111)
    chart_type1 = FigureCanvasTkAgg(figure, root)
    #chart_type.draw()
    #chart_type.get_tk_widget().pack()
    canvas.create_window(500, 300, window=chart_type1.get_tk_widget())
    ax.plot(train_accuracy, label='Training Accuracy')
    ax.plot(val_accuracy, label='Validation Accuracy')
    ax.legend()
    ax.set_title('Epochs vs. Training and Validation Accuracy-Classifier',fontsize=6) 
    #ax.set_title('The Title for your chart')
    return model2

def train_classifier():
    global df
    global target_column
    global classifier_model
    x_adasyn,y_adasyn=data_preprocess(df,target_column)
    classifier_model=train_classifier_model(x_adasyn,y_adasyn)

def pickle_classifier():
    global classifier_model
    filename = filedialog.asksaveasfilename( defaultextension=".hdf5",filetypes=[("default", "*.hdf5"),
                           ("all", "*.*")])
    print(filename)
    classifier_model.save(filename)
    label_pickleclass = tk.Label(root, text= "Saved Classifier Model to Disk")
    label_pickleclass.pack()

root = tk.Tk()
root.title('Neural Network')
#root.iconbitmap('class.ico')
root.resizable(False, False)
tit = tk.Label(root, text="Neural Network", padx=25, pady=6, font=("", 12)).pack()
canvas = tk.Canvas(root, height=500, width=500, bg='grey')
canvas.pack()
frame = tk.Frame(root, bg='white')
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
custName = tk.StringVar()
fileName = Entry(root, textvariable=custName)
entry1 = tk.Entry (root) 
#fileName.grid(row=0, column=1)
#entry1.grid(row=1, column=1)
canvas.create_window(300, 100, window=fileName,width=400)
canvas.create_window(300, 160, window=entry1)
choose_csv= tk.Button(root, text='Import Data',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=import_data)
#choose_csv.pack(side=tk.LEFT)
canvas.create_window(0, 100, window=choose_csv,width=100)
choose_target= tk.Button(root, text='Import Target',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=import_target)
#choose_target.pack(side=tk.LEFT)
canvas.create_window(0, 160, window=choose_target,width=100)
button = tk.Button(root, text = 'root quit', command=quit)
button.pack(side=tk.RIGHT)

train_csv_reg = tk.Button(root, text='Preprocess Data and Train Regressor',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=train_regression)
train_csv_reg.pack(side=tk.LEFT)
pickle_reg = tk.Button(root, text='Pickle Regression Model',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=pickle_regression)
pickle_reg.pack(side=tk.LEFT)
train_csv_class = tk.Button(root, text='Preprocess Data and Train Classifier',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=train_classifier)
train_csv_class.pack(side=tk.LEFT)
pickle_reg = tk.Button(root, text='Pickle Classifier Model',
                        padx=35, pady=10,
                        fg="white", bg="grey", command=pickle_classifier)
pickle_reg.pack(side=tk.LEFT)
root.mainloop()
