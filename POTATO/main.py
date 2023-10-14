from flask import Flask, request, render_template, send_from_directory, jsonify
import tensorflow as tf
import numpy as np
import os
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_img(img):
    model = tf.keras.models.load_model('my_new__model.h5')

    test_data = tf.keras.preprocessing.image.load_img(img)
    test_data = tf.keras.preprocessing.image.img_to_array(test_data)
    test_data = test_data.reshape((1, 256, 256, 3))

    predictions = model.predict(test_data)

    if np.where(predictions[0] == max(predictions[0]))[0] == 0:
        return ["Early Blight: ", float(predictions[0][0])]
    elif np.where(predictions[0] == max(predictions[0]))[0] == 1:
        return ["Healthy: ", float(predictions[0][1])]
    elif np.where(predictions[0] == max(predictions[0]))[0] == 2:
        return ["Late Blight: ", float(predictions[0][2])]

@app.route("/")
def index():
    return render_template("server.html")

@app.route("/predict", methods=["GET", "POST"])
def upload_photo():
    if request.method == "POST":
        file = request.files["file"]
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        encoded_image = base64.b64encode(file.read()).decode('utf-8')
        prediction = predict_img(filename)

        return jsonify({'prediction': prediction})

    return render_template("server.html", image=None)


if __name__ == "__main__":
  app.run(debug=True)