from flask import (Flask, jsonify, request)
from matplotlib.pyplot import imshow
import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import PIL
from PIL import ImageFilter
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from flask_restful import Resource, Api
from flask_cors import CORS
from flask_ngrok import run_with_ngrok

import struct
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D
# from keras.layers.merge import add, concatenate
from tensorflow.keras.models import Model

# We change here the origin
# model = load_model('C:/Users/User/Desktop/Fakultet/POOS/Projekat model i ostalo/POOS project/top_model.h5')

app = Flask(__name__)
run_with_ngrok(app)
CORS(app)
api = Api(app)


@app.route("/process-image", methods=["POST"])
def process_image():
    bytesOfImage = request.get_data()
    with open('image.jpeg', 'wb') as out:
        out.write(bytesOfImage)

    width, height = load_img('image.jpeg')

    # load the image with the required size
    image = load_img('image.jpeg', target_size=(416, 416))
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0

    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # define our new photo
    photo_filename = 'image.jpeg'
    # load and prepare image
    image, image_w, image_h = load_image_pixels(
        photo_filename, (input_w, input_h))

    # make prediction
    # yhat = model.predict(image)

    return {
        "result": "sdsdsd"
    }

    # Hardcoded value due to not enough time
    # img_path = request.get_json().get('image_path')
    # img_path = "Lato.png"
    # pil_im = PIL.Image.open(img_path).convert('L')
    # pil_im = blur_image(pil_im)
    # org_img = img_to_array(pil_im)
    # data = []
    # data.append(org_img)
    # data = np.asarray(data, dtype="float") / 255.0
    # y = model.predict_classes(data)
    # label = rev_conv_label(int(y[0]))
    # return {
    #     'label': label
    # }

# load and prepare an image


def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = np.expand_dims(image, 0)
    return image, width, height


@app.route('/')
def index_page():
    return 'This is the font recognition API!'


def blur_image(pil_im):
    blur_img = pil_im.filter(ImageFilter.GaussianBlur(radius=3))
    blur_img = blur_img.resize((105, 105))
    return blur_img


# labels for different fonts
def rev_conv_label(label):
    if label == 0:
        return 'Lato'
    elif label == 1:
        return 'Raleway'
    elif label == 2:
        return 'Roboto'
    elif label == 3:
        return 'Sansation'
    elif label == 4:
        return 'Walkway'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
