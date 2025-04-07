from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

@app.route('/receive-grid', methods=['POST'])
def receive_grid():
    data = request.json
    grid = data.get("grid", [])
    print(np.array(grid, dtype="float32"))

    return jsonify({"status": "ok", "received": len(grid)})


@app.route('/')
def index():
    return render_template("index.html")

app.run(debug=True)
