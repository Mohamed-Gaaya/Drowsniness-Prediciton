import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import cv2
classes=['open eyes','closed eyes']
data_training=[]
for category in classes:
    path='C:/Users/LENOVO/Desktop/'+category
    classes_num=classes.index(category)
    for img in os.listdir(path):
        try:
            img_array=cv2.imread(path+'/'+img,cv2.IMREAD_GRAYSCALE)
            backtorgb=cv2.cvtColor(img_array,cv2.COLOR_GRAY2RGB)
            new_array=cv2.resize(backtorgb,(224,224))
            data_training.append([new_array,classes_num])
        except:
            pass
X=[]
Y=[]
for features,labels in data_training:
    X.append(features)
    Y.append(labels)
X=np.array(X).reshape(-1,224,224,3)
Y=np.array(Y)
#pickle
pickle_out=open('X.pickle',"wb")
pickle.dump(X,pickle_out)
pickle_out.close()
pickle_out=open('Y.pickle',"wb")
pickle.dump(Y,pickle_out)
pickle_out.close()
pickle_in=open('X.pickle',"rb")
X=pickle.load(pickle_in)
pickle_in=open('Y.pickle',"rb")
Y=pickle.load(pickle_in)
model = tf.keras.applications.mobilenet.MobileNet()

# As our first layer is our input layer.
input_layer = model.layers[0].input 

# The final layer corrosponds to output layer.
output_layer = model.layers[-4].output


flat_layer = layers.Flatten()(input_layer)
final_layer_output = layers.Dense(1)(flat_layer)

final_layer_output = layers.Activation('sigmoid')(final_layer_output)

# Defining our new model.
model_new = keras.Model(inputs = input_layer, outputs = final_layer_output)

# Setting loss to binary_crossentropy loss and adam optimizer
# Binary crossentropy loss is used because our model will predict a probability of whether the eyes are open.
# Adam optimizer is used because it is of the most efficient optimization technique.
model_new.compile(loss="binary_crossentropy", optimizer= "adam", metrics=["accuracy"])

# Training model with 10 epochs
model_new.fit(X, Y, epochs=10, validation_split=0.1)

# Saving the model for further use
model_new.save('wa7ed_model.h5')


