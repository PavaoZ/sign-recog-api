from flask import (Flask, jsonify, request)
from matplotlib.pyplot import imshow
import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import PIL
from PIL import ImageFilter
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask_restful import Resource, Api
from flask_cors import CORS
from flask_ngrok import run_with_ngrok

# We change here the origin
# model = load_model('C:/Users/User/Desktop/Fakultet/POOS/Projekat model i ostalo/POOS project/top_model.h5')

app = Flask(__name__)
run_with_ngrok(app)
CORS(app)
api = Api(app)


@app.route("/process-image", methods=["POST"])
def process_image():
    data = request.form.to_dict()

    return {
        "result": "example"
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
    app.run(debug=True)
