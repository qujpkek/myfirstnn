from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt
import numpy as np
import random

from utils import *
images, labels = load_dataset()

learning_rate = 0.01

model = load_model("soakers")

app = Flask(__name__)

@app.route('/receive-grid', methods=['POST'])
def receive_grid():
    data = request.json
    grid = data.get("grid", [])
    np_grid = np.array(grid, dtype="float32").reshape(28, 28)

    print("Received grid shape:", np_grid.shape)

    if np_grid.shape != (28, 28):
        return jsonify({"status": "error", "message": f"Wrong shape: {np_grid.shape}, expected (28, 28)"}), 400

    prediction = predict(model, np_grid.flatten())
    print("Prediction:", prediction)

    return jsonify({"status": "ok", "prediction": int(prediction)})


@app.route('/')
def index():
    return render_template("index.html")

app.run(debug=True)
