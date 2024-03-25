from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request
import face_recognition
import numpy as np

app = Flask(__name__)
model = load_model('model_bmi.h5')
# Define function to get face encoding
def get_face_encoding(image_path):
    picture_of_me = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(picture_of_me)
    my_face_encoding = face_recognition.face_encodings(picture_of_me, face_locations)
    if not my_face_encoding:
        print("no face found !!!")
        return np.zeros(128).tolist()
    return my_face_encoding[0].tolist()

# Define route for face recognition
@app.route('/bmi', methods=['POST'])
def face_recognition_endpoint():
    image_path = request.files['image']

    if not image_path:
        return jsonify({'error': 'Image path is required'}), 400

    try:
        face_encoding = get_face_encoding(image_path)
        pred = model.predict(np.expand_dims(np.array(face_encoding),axis=0))
        pred = round(float(pred[0][0]), 2)  # Konversi nilai float32 ke float
        return jsonify({'bmi': pred}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
