from flask import Flask, request, jsonify
from flask_cors import CORS
from image_classifier import ImageClassifier

app = Flask(__name__)
CORS(app)

# Initialize classifier globally
try:
    print("Loading classifier...")
    classifier = ImageClassifier()
    print("Classifier loaded successfully")
except Exception as e:
    print(f"Failed to load classifier: {e}")
    classifier = None

@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None:
        return jsonify({'error': 'Classifier not loaded'}), 500
        
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
            
        predicted_class, confidence = classifier.predict(data['image'])
        is_showerhead = predicted_class.lower() == "showerhead"
        
        # Define a confidence threshold
        CONFIDENCE_THRESHOLD = 0.7
        is_confident = confidence >= CONFIDENCE_THRESHOLD
        
        # Return the prediction results
        return jsonify({
            'is_showerhead': is_showerhead and is_confident,
            'confidence': confidence,
            'predicted_class': predicted_class,
            'is_confident': is_confident
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': classifier is not None,
        'device': str(classifier.device) if classifier else None
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'online',
        'endpoints': {
            'predict': '/predict (POST) - Send base64 image for prediction',
            'health': '/health (GET) - Check server status'
        }
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Route not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

@app.before_request
def log_request_info():
    print('Headers:', dict(request.headers))
    if request.method != 'GET':
        print('Request method:', request.method)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response

if __name__ == '__main__':
    try:
        print("Starting server...")
        print("Available routes:")
        print("  - GET  /")
        print("  - GET  /health")
        print("  - POST /predict")
        app.run(host='localhost', port=8000, debug=True)
    except Exception as e:
        print(f"Failed to start server: {e}")