from flask import Flask, render_template, request, jsonify, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == '__main__':
    app.run(debug=True)
