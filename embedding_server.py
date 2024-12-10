from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)
CORS(app, origins=["https://project-production-6c3a.up.railway.app"], methods=["POST", "OPTIONS"], supports_credentials=True)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other pre-trained models

@app.route('/embed', methods=['POST', 'OPTIONS'])
def embed():
    if request.method == 'OPTIONS':
        # Respond to the preflight request
        response = jsonify({"message": "Preflight OK"})
        response.headers.add('Access-Control-Allow-Origin', 'https://project-production-6c3a.up.railway.app')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response, 200

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
