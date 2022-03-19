import os
import uuid
import urllib
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, send_file
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model('./seedmodel.h5')

ALLOWED_EXT = set(['jpg'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


classes = ["Bad Groundnut","Bad Moongdal","Bad Pigeonpea","Good Groundnut","Good Moongdal","Good Pigeonpea"]


def predict(filename, model):
    data = []
    path = f"D:/SeedsClassification/testdata/test13.jpg"
    image = load_img(path, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    data.append(image)
    data = np.array(data, dtype="float32")
    lt = model.predict(data)
    pred = np.argmax(lt)
    lt=classes[pred]

    return lt


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if (request.form):
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result= predict(img_path, model)

                predictions = {
                    "class1": class_result,
                }

            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)


        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename

                class_result = predict(img_path, model)

                predictions = {
                    "class1": class_result,
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)