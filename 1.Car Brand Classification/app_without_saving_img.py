

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
import tensorflow_hub as hub
import io
import base64
import cv2


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='car_brand_resnet152v2.h5'

# Load your trained model
model = load_model(MODEL_PATH,custom_objects={'KerasLayer':hub.KerasLayer})




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

   

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The Car is Audi"
    elif preds==1:
        preds="The Car is Lamborghini"
    else:
        preds="The Car is Mercedes"
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        responses = f.read()
        xx = cv2.imdecode(np.frombuffer(responses, np.uint8), cv2.IMREAD_COLOR)
        ##print(xx)
        ##print(type(xx))
        print(xx.shape)
        
        pil_image = image.array_to_img(xx, data_format=None, scale=True, dtype=None)
        print(pil_image.size)
        pil_image_resize = pil_image.resize((224,224))
        print(pil_image_resize.size)
        x = image.img_to_array(pil_image_resize)
        ##print(x)
        print(x.shape)
        
        '''
        image_string = base64.b64encode(f.read())
        jpg_original = base64.b64decode(image_string)
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
        print(jpg_as_np)
        print(type(jpg_as_np))
        '''
        

        '''
        # Save the file to ./uploadsq
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)'''

        # Make prediction
        # Preprocessing the image
        #x = image.img_to_array(img)
        # x = np.true_divide(x, 255)
        ## Scaling
        x=x/255
        x = np.expand_dims(x, axis=0)
        print(x.shape)
    
        preds = model.predict(x)
        preds=np.argmax(preds, axis=1)
        if preds==0:
            preds="The Car is Audi"
        elif preds==1:
            preds="The Car is Lamborghini"
        else:
            preds="The Car is Mercedes"
        
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
