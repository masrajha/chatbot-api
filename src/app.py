from flask import Flask, request, jsonify
from src.model_loader import load_models
from src.ner_processor import compare_model
from src.utils import format_response

app = Flask(__name__)

# Load models saat aplikasi start
tokenizer, model1, model2 = load_models()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "Text input required"}), 400
    
    results = compare_model(text, tokenizer, model1, model2)
    
    formatted = {
        "model1": format_response(results['model1']),
        "model2": format_response(results['model2']),
        "hybrid": format_response(results['hybrid'])
    }
    
    return jsonify(formatted)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)