from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other pre-trained models

@app.route('/embed', methods=['POST'])
def embed():
    try:
        data = request.get_json()
        sentence = data.get('sentence', '')

        if sentence == '':
            return jsonify({'error': 'No sentence provided'}), 400

        embedding = model.encode(sentence)
        return jsonify({'embedding': embedding.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
