#app.py
from flask import Flask, request, jsonify
from src.model_loader import load_models, load_model_classify, classify_intent
from src.ner_processor import compare_model
from src.google_sheets import get_sheet_data
from src.searcher import SheetSearcher
from src.utils import group_entities_by_type, convert_relative_dates, extract_relative_dates
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
from flask_cors import CORS

app = Flask(__name__)

# Inisialisasi CORS dengan konfigurasi
# CORS(app, resources={
#     r"/search": {
#         "origins": "http://localhost:3000",
#         "methods": ["POST", "OPTIONS"],
#         "allow_headers": ["Content-Type"]
#     }
# })

CORS(app, resources={r"/search": {"origins": "*"}})
CORS(app, resources={r"/predict": {"origins": "*"}})

# Muat model Keras
model_path = os.path.join('models/fer_models', 'fer_model_20250530-073815_final.keras')
model = load_model(model_path)

# Label emosi sesuai FER-2013
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_image(image):
    """
    Preprocessing gambar untuk model:
    - Konversi ke grayscale
    - Resize ke 48x48 pixel
    - Normalisasi pixel [0, 1]
    - Ekspansi dimensi untuk input model
    """
    image = image.convert('L')  # Konversi ke grayscale
    image = image.resize((48, 48))  # Resize sesuai input model
    image = img_to_array(image)  # Konversi ke array
    image = image.astype('float32') / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambahkan dimensi batch
    image = np.expand_dims(image, axis=-1)  # Tambahkan channel (1 channel untuk grayscale)
    return image

KULIAH_KEYWORDS = {
    'kuliah', 'perkuliahan', 'belajar', 'mengajar', 
    'matkul', 'mata kuliah', 'kelas', 'jadwal kuliah', 
    'materi', 'pertemuan', 'perkuliahan'
}
SEMINAR_KEYWORDS = {
    'seminar', 'usul', 'hasil', 'kompre', 'ujian', 
    'skripsi', 'proposal', 'kolokium',
    'tugas akhir', 'ta'
}

def classify_query(query: str) -> list:
    """Tentukan jenis dataset yang perlu dicari berdasarkan kata kunci"""
    query_lower = query.lower()
    
    has_kuliah = any(keyword in query_lower for keyword in KULIAH_KEYWORDS)
    has_seminar = any(keyword in query_lower for keyword in SEMINAR_KEYWORDS)
    
    if has_kuliah and has_seminar:
        return ['kuliah', 'seminar']  # Cari kedua dataset
    elif has_kuliah:
        return ['kuliah']
    elif has_seminar:
        return ['seminar']
    else:
        return ['kuliah', 'seminar']  # Default cari semua
    
try:
    tokenizer1, model1, tokenizer2, model2 = load_models()
    print("Model dan tokenizer berhasil dimuat")
    # CHECKPOINT_PATH = "models/classification-v3"
    # model_klasifikasi, tokenizer, id2label, device = load_model_classify(CHECKPOINT_PATH)
    # model_klasifikasi.eval()
    # print("Model klasifikasi berhasil dimuat")
except Exception as e:
    print(f"Gagal memuat model: {str(e)}")
    sys.exit(1)

@app.route('/predict', methods=['POST'])
def predict_emotion():
    # Cek apakah request memiliki file gambar
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['file']
    
    # Validasi ekstensi file
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not (file.filename.lower().endswith(('.png', '.jpg', '.jpeg'))):
        return jsonify({'error': 'Invalid file format. Use PNG, JPG, or JPEG'}), 400

    try:
        # Baca dan preprocess gambar
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        
        # Prediksi emosi
        predictions = model.predict(processed_image)
        emotion_index = np.argmax(predictions[0])
        emotion = emotion_labels[emotion_index]
        confidence = float(predictions[0][emotion_index])
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'all_predictions': dict(zip(emotion_labels, predictions[0].astype(float)))
        })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/search', methods=['POST'])
def search():
    # Ambil data JSON dari request
    request_data = request.get_json()
    text = request_data.get('text', '')
    
    if not text:
        return jsonify({
            "message": "Teks tidak boleh kosong",
            "data": [],
            "entities": {}
        }), 400

    try:
        # Klasifikasi jenis pencarian berdasarkan teks
        # search_types = classify_intent(text, model_klasifikasi, tokenizer, id2label, device)
        search_types = classify_query(text)
        print(f"Jenis pencarian yang terdeteksi: {search_types}")
        
        # Ekstrak tanggal relatif dan proses teks
        relative_dates = extract_relative_dates(text)
        print(f"Tanggal relatif ditemukan: {relative_dates}")
        processed_text = convert_relative_dates(text)
        print(f"Teks setelah diproses: {processed_text}")
        
        # Ekstrak entitas dari teks
        # results = compare_model(processed_text, tokenizer1, model1, tokenizer2, model2)
        results = compare_model(text, tokenizer1, model1, tokenizer2, model2)
        entities_raw = results.get('hybrid', [])
        entities = group_entities_by_type(entities_raw)
        
        # Tambahkan tanggal relatif ke entitas DAT
        if relative_dates:
            if 'DAT' in entities:
                entities['DAT'].extend(relative_dates)
            else:
                entities['DAT'] = relative_dates
            print(f"Entitas DAT setelah ditambah: {entities.get('DAT', [])}")
        
        if 'MK' in entities:
            search_types = ['kuliah']
                
        print(f"Entitas yang ditemukan: {entities}")
        
        combined_data = []
        
        # Cari data hanya untuk jenis yang diperlukan
        if 'kuliah' in search_types:
            print("Mengambil dan mencari data kuliah...")
            kuliah_data = get_sheet_data('Kuliah')
            kuliah_searcher = SheetSearcher(kuliah_data, 'Kuliah')
            hasil_kuliah = kuliah_searcher.search(entities)
            for item in hasil_kuliah:
                item['jenis_data'] = 'kuliah'
            combined_data.extend(hasil_kuliah)
            print(f"Found {len(hasil_kuliah)} kuliah records")
        
        else: 
            if 'seminar' in search_types:
                print("Mengambil dan mencari data seminar...")
                seminar_data = get_sheet_data('Seminar')
                seminar_searcher = SheetSearcher(seminar_data, 'Seminar')
                hasil_seminar = seminar_searcher.search(entities)
                for item in hasil_seminar:
                    item['jenis_data'] = 'seminar'
                combined_data.extend(hasil_seminar)
                print(f"Found {len(hasil_seminar)} seminar records")
        
        # Format respons berdasarkan hasil pencarian
        if not combined_data:
            # Potong teks jika terlalu panjang
            display_text = text[:50] + "..." if len(text) > 50 else text
            return jsonify({
                "message": f"Data '{display_text}' tidak ditemukan",
                "data": [],
                "entities": entities,
                "search_types": search_types  # Tambahkan info jenis pencarian
            })
        else:
            return jsonify({
                "message": "Berhasil",
                "data": combined_data,
                "entities": entities,
                "search_types": search_types  # Tambahkan info jenis pencarian
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "message": f"Terjadi kesalahan: {str(e)}",
            "data": [],
            "entities": {}
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)