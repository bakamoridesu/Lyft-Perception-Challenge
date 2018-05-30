import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

from keras.models import model_from_json
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2

file = sys.argv[-1]

if file == 'demo.py':
    print ("Error loading video")
    quit

# Define encoder function
def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

video = skvideo.io.vread(file)

answer_key = {}

json_file = open('model_with_augmentation.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_with_augmentation.h5")
print("Loaded model from disc")

# Frame numbering starts at 1
frame = 1

for rgb_frame in video:
    img = cv2.resize(img, (666,500))
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    pred = pred[0]
    pred = pred.reshape(500,666,2)
    car_pred = pred[:,:,0]
    road_pred = pred[:,:,1]
    
    car_pred_ = np.zeros(car_pred.shape)
    car_pred_[car_pred>0.3] = 1
    
    road_pred_ = np.zeros(road_pred.shape)
    road_pred_[road_pred>0.3] = 1
    
    car_pred_ = cv2.resize(car_pred_, (800,600), interpolation = cv2.INTER_NEAREST)
    road_pred_ = cv2.resize(road_pred_, (800,600), interpolation = cv2.INTER_NEAREST)

    answer_key[frame] = [encode(car_pred_), encode(road_pred_)]
    
    # Increment frame
    frame+=1

# Print output in proper json format
print (json.dumps(answer_key))