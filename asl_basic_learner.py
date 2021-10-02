from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,SpatialDropout2D
from tensorflow.keras.models import Sequential
import cv2
import numpy as np
from random import sample
import os
import time
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.mixed_precision.set_global_policy('float32')#'mixed_float16''float32'

data_path = "D:/ASL/data/" #put your image folder here directory... subfolders should be named validation and training
model_path = "D:/ASL/"
classes = 28
color_mode="rgb"#can change to 'grayscale' for edge detection
BATCHES = 80
IMAGESIZE = 200
if color_mode=='rgb':
  depth = 3
else:
  depth = 1

"""Put all of your data into a folder named 'unsorted' under data_path. The data_path and model_path variables may be altered according to setup. The structure is as follows:
{model_path} ---> /{data_path}, /lite ---> /{data_path}/unsorted/

Combined Datasets:
https://www.kaggle.com/ammarnassanalhajali/american-sign-language-letters
https://www.kaggle.com/grassknoted/asl-alphabet
https://www.kaggle.com/belalelwikel/asl-and-some-words
https://www.kaggle.com/allexmendes/asl-alphabet-synthetic
https://github.com/ruslan-kl/asl_recognition


"""
  
def abstract_dataset(threshold=10.1,gamma=2.8,ratio = 1.3,kernel_size = 3,image_size=(IMAGESIZE, IMAGESIZE),sigmaColor=4,sigma=20,sigmaSpace=3,data = data_path):
      
      def canny_lines(image_path='',threshold=threshold,gamma=gamma,ratio = ratio, kernel_size = kernel_size,image_size=image_size):
        def adjust_gamma(image_path, gamma=gamma):
              invGamma = 1.0 / gamma
              table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
              return cv2.LUT(image_path, table)
        detected_edges = cv2.imread(image_path, 1)
        detected_edges = cv2.resize(detected_edges, image_size, interpolation=cv2.INTER_CUBIC)
        detected_edges = adjust_gamma(detected_edges, gamma=gamma)
        detected_edges = cv2.cvtColor(detected_edges, cv2.COLOR_BGR2GRAY)
        detected_edges = cv2.bilateralFilter(detected_edges,sigma,sigmaColor=sigmaColor,sigmaSpace=sigmaSpace)
        detected_edges = cv2.Canny(detected_edges, threshold, threshold*ratio, kernel_size,L2gradient=True)
        #detected_edges = 255 - detected_edges
        return detected_edges

      #create the proper folder structure to store our canny edge detection dataset
      names = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","nothing","space"]
      if not os.path.exists(f"{data}canny"):
        os.mkdir(f"{data}canny")
      if not os.path.exists(f"{data}canny/training"):
            os.mkdir(f"{data}canny/training/")
      if not os.path.exists(f"{data}canny/validation"):
            os.mkdir(f"{data}canny/validation/")
      for i in names:
        if not os.path.exists(f"{data}canny/validation/{i}"):
            os.mkdir(f"{data}canny/validation/{i}")
        if not os.path.exists(f"{data}canny/training/{i}"):
              os.mkdir(f"{data}canny/training/{i}")
      for i in names:
        if len([file for file in os.listdir(f"{data}canny/training/{i}") if os.path.exists(os.path.join(f"{data}canny/training/{i}", file))]) !=0:
              names.remove(i)
              print(f"removing {i}")

      train_or_val = "training"
      for i in names:     
        image_path = data+f"{train_or_val}/"          
        filenames = [file for file in os.listdir(image_path+i+"/") if os.path.exists(os.path.join(image_path+i+"/", file))]#https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
        c = 0
        for file in filenames:
            c+=1
            #uncomment to use canny detection when creating dataset
            #image = canny_lines(image_path+i+"/"+file)
            image_name = f"{data}canny/{train_or_val}/{i}/c{file}".split('.')[0] + ".jpg" #messy function to put file in correct folder
            cv2.imwrite(image_name, image,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
            print(f"Progress edge detecting on {i}: {round(c/len(filenames)*100,0)}%",end='\r                     ')
      
      train_or_val = "validation"
      for i in names:     
        image_path = data+f"{train_or_val}/"          
        filenames = [file for file in os.listdir(image_path+i+"/") if os.path.isfile(os.path.join(image_path+i+"/", file))]
        c = 0
        for file in filenames:
            c+=1
            image = canny_lines(image_path+i+"/"+file)
            image_name = f"{data}canny/{train_or_val}/{i}/c{file}".split('.')[0] + ".jpg"
            cv2.imwrite(image_name, image,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
            print(f"Progress edge detecting on {i}: {round(c/len(filenames)*100,0)}%",end='\r                     ')

def mse(A, B):
      #https://towardsdatascience.com/image-classification-using-ssim-34e549ec6e12. 
      #Detects how similar two images are
        A = A.resize((200,200), resample=Image.LANCZOS) 
        B = B.resize((200,200), resample=Image.LANCZOS) 
        A = np.array(A)
        A = A.astype(np.float32)
        B = np.array(B)
        B = B.astype(np.float32)
        mse_err = np.sum((A - B)**2)
        mse_err /= float(A.shape[0] * A.shape[1])# return the MSE, the lower the error, the more "similar"
        return mse_err 

def sort_and_avoid_duplicates(folder=f"{data_path}/unsorted/",percent_val=.08,remove_duplicates_threshold=0,display_dups=False):
    names = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","nothing","space"]
    if not os.path.exists(f"{data_path}training"):
          os.mkdir(f"{data_path}training/")
    if not os.path.exists(f"{data_path}validation"):
          os.mkdir(f"{data_path}validation/")
    for i in names:
      if not os.path.exists(f"{data_path}validation/{i}"):
          os.mkdir(f"{data_path}validation/{i}")
      if not os.path.exists(f"{data_path}training/{i}"):
            os.mkdir(f"{data_path}training/{i}")
    for folders in names:
          asl_images = [file for file in os.listdir(folder+folders+"/") if os.path.exists(os.path.join(folder+folders+"/"))]
          validation = sample(asl_images,round(float(len(asl_images))*percent_val))
          training = [i for i in asl_images if i not in validation]
          asl_images = None
          for v in validation:
                nospam = True
                if remove_duplicates_threshold != 0:
                #The time cost is very exponential for this function, so it is not practical for our timeframe to scan for and remove duplicate images
                      for t in training:
                            p_similarity = mse(Image.open(folder+folders+"/"+v),Image.open(folder+folders+"/"+t))
                            if p_similarity<remove_duplicates_threshold:
                                  if nospam:
                                    if display_dups:
                                      os.system(folder+folders+'/'+v)
                                      os.system(folder+folders+'/'+t)
                                    nospam=False
                                    break
                                  print(f"{p_similarity}   {folder+folders+'/'+v}    {folder+folders+'/'+t}")
                            print(v,"---> ",t,end='\r                  ')
                if nospam:
                      try:
                        image = cv2.imread(folder+folders+"/"+v, 1)
                        cv2.imwrite(data_path+"validation/"+folders+"/"+v.split(".")[0]+".jpg", image,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        print("collected ",data_path+"validation/"+folders+"/"+v.split(".")[0]+".jpg")
                      except Exception as error:
                        print(error)
                else:
                      try:
                        image = cv2.imread(folder+folders+"/"+v, 1)
                        cv2.imwrite(data_path+"training/"+folders+"/"+v.split(".")[0]+".jpg", image,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                      except Exception as error:
                        print(error)
          for t in training:
                if t not in validation:
                      try:
                        image = cv2.imread(folder+folders+"/"+t, 1)
                        cv2.imwrite(data_path+"training/"+folders+"/"+t.split(".")[0]+".jpg", image,[int(cv2.IMWRITE_JPEG_QUALITY), 90])
                      except Exception as error:
                        print(error)

if not os.path.exists(f"{data_path}training") or not os.path.exists(f"{data_path}validation"):sort_and_avoid_duplicates(percent_val=.08,remove_duplicates_threshold=0) #sort images into 

if not os.path.exists(data_path+"canny/validation") or not os.path.exists(data_path+"canny/training"): abstract_dataset()
    

train_data = tf.keras.utils.image_dataset_from_directory(
    data_path + "canny/" + "training", labels='inferred', label_mode='categorical',
    class_names=None, color_mode=color_mode, batch_size=BATCHES, image_size=(IMAGESIZE,
    IMAGESIZE), shuffle=True, seed=34667,
    interpolation='lanczos5', follow_links=False,
    crop_to_aspect_ratio=True
)

test_data = tf.keras.utils.image_dataset_from_directory(
    data_path + "canny/" + "validation", labels='inferred', label_mode='categorical',
    class_names=None, color_mode=color_mode, batch_size=BATCHES, image_size=(IMAGESIZE,
    IMAGESIZE), shuffle=True, seed=34667,
    interpolation='lanczos5', follow_links=False,
    crop_to_aspect_ratio=True)


def define_model(pooling = (2,2),kernel_size = (3,3),filters=32,activation='swish',noise=.9,dropout=.2,rotation=.1,zoom=(-.01,.05)):
      
  #https://www.tensorflow.org/tutorials/images/data_augmentation
  train_mod = Sequential([
    layers.RandomFlip(mode="horizontal",input_shape=(IMAGESIZE, IMAGESIZE,depth)),
    layers.RandomRotation(rotation,fill_mode='nearest'),
    layers.GaussianNoise(noise),
    layers.RandomZoom(height_factor=zoom,fill_mode='nearest'),
    layers.experimental.preprocessing.Resizing(height=IMAGESIZE, width=IMAGESIZE,crop_to_aspect_ratio=True),
    layers.experimental.preprocessing.Rescaling(scale=1./255),
    
    Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, input_shape=(IMAGESIZE, IMAGESIZE,depth)),
    Conv2D(filters=filters, kernel_size=kernel_size, activation=activation),
    MaxPooling2D(pooling),
    SpatialDropout2D(dropout),
    Conv2D(filters=filters*2, kernel_size=kernel_size, activation=activation),
    Conv2D(filters=filters*2, kernel_size=kernel_size, activation=activation),
    MaxPooling2D(pooling),
    
    Flatten(),
    Dense(250, activation=activation),
    Dense(128, activation=activation),
    Dense(64, activation=activation),
    Dense(classes, activation='softmax')
    ])
  return train_mod

model = define_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, amsgrad=False), loss='categorical_crossentropy', metrics='accuracy')
model.summary()

#model training
model.fit(train_data,shuffle=True,use_multiprocessing=True,epochs=200,validation_data = test_data,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,mode='auto',restore_best_weights=True)])


converter = tf.lite.TFLiteConverter.from_keras_model(model)#https://medium.com/analytics-vidhya/optimization-techniques-tflite-5f6d9ae676d5
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()
with open(model_path+"/lite/"+f'model{time.time()}.tflite', 'wb') as f:
  f.write(tflite_model)