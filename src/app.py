from flask import Flask, request, jsonify
from src.model_loader import load_models
from src.ner_processor import compare_model
from src.google_sheets import get_sheet_data
from src.searcher import SheetSearcher
from src.utils import group_entities_by_type, convert_relative_dates, extract_relative_dates  # Perubahan disini
import sys

app = Flask(__name__)

# Load model saat aplikasi dimulai
try:
    tokenizer1, model1, tokenizer2, model2 = load_models()
    print("Model dan tokenizer berhasil dimuat")
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
        relative_dates = extract_relative_dates(text)
        print(f"Tanggal relatif ditemukan: {relative_dates}")
        # Ambil data terbaru dari Google Sheets
        print("Mengambil data kuliah...")
        kuliah_data = get_sheet_data('Kuliah')
        print("Mengambil data seminar...")
        seminar_data = get_sheet_data('Seminar')
        
        # Ekstrak entitas dari teks
        processed_text = convert_relative_dates(text)
        print(f"Teks setelah diproses: {processed_text}")
        
        kuliah_data = get_sheet_data('Kuliah')
        seminar_data = get_sheet_data('Seminar')
        
        # Gunakan teks yang sudah diproses untuk NER
        results = compare_model(text, tokenizer1, model1, tokenizer2, model2)
        entities_raw = results.get('hybrid', [])
        entities = group_entities_by_type(entities_raw)
        
        if relative_dates:
            if 'DAT' in entities:
                entities['DAT'].extend(relative_dates)
            else:
                entities['DAT'] = relative_dates
            print(f"Entitas DAT setelah ditambah: {entities['DAT']}")
            
        print (entities)
        
        # Cari data kuliah
        kuliah_searcher = SheetSearcher(kuliah_data, 'Kuliah')
        hasil_kuliah = kuliah_searcher.search(entities)
        
        # Cari data seminar
        seminar_searcher = SheetSearcher(seminar_data, 'Seminar')
        hasil_seminar = seminar_searcher.search(entities)
        
        # Gabungkan hasil dan tambahkan jenis data
        combined_data = []
        for item in hasil_kuliah:
            item['jenis_data'] = 'kuliah'
            combined_data.append(item)
        
        for item in hasil_seminar:
            item['jenis_data'] = 'seminar'
            combined_data.append(item)
        
        # Format respons berdasarkan hasil pencarian
        if not combined_data:
            # Potong teks jika terlalu panjang
            display_text = text[:50] + "..." if len(text) > 50 else text
            return jsonify({
                "message": f"Data '{display_text}' tidak ditemukan",
                "data": [],
                "entities": entities
            })
        else:
            return jsonify({
                "message": "Berhasil",
                "data": combined_data,
                "entities": entities
            })
            
    except Exception as e:
        return jsonify({
            "message": f"Terjadi kesalahan: {str(e)}",
            "data": [],
            "entities": {}
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)