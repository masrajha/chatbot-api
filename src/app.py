#app.py
from flask import Flask, request, jsonify
from src.model_loader import load_models, load_model_classify, classify_intent
from src.ner_processor import compare_model
from src.google_sheets import get_sheet_data
from src.searcher import SheetSearcher
from src.utils import group_entities_by_type, convert_relative_dates, extract_relative_dates
import sys
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

try:
    tokenizer1, model1, tokenizer2, model2 = load_models()
    print("Model dan tokenizer berhasil dimuat")
    CHECKPOINT_PATH = "models/classification"
    model_klasifikasi, tokenizer, id2label, device = load_model_classify(CHECKPOINT_PATH)
    model_klasifikasi.eval()
    print("Model klasifikasi berhasil dimuat")
except Exception as e:
    print(f"Gagal memuat model: {str(e)}")
    sys.exit(1)

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
        search_types = classify_intent(text, model_klasifikasi, tokenizer, id2label, device)
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