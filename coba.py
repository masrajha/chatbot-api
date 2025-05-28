import pandas as pd
from src.model_loader import load_models
from src.ner_processor import compare_model
from src.google_sheets import get_sheet_data
from src.searcher import SheetSearcher
from src.utils import group_entities_by_type
def main():
    # Load model dan tokenizer   
    try:
        print("Mengambil data kuliah...")
        kuliah_data = get_sheet_data('Kuliah')
        print("Mengambil data seminar...")
        seminar_data = get_sheet_data('Seminar')
        tokenizer1, model1, tokenizer2, model2 = load_models()
        text = """
        Bagaimana mekanisme evaluasi untuk mata kuliah Temu Kembali Informasi 
        S2 Komputer di Dekanat L3 1 —apakah berbasis proyek atau ujian teoritis
        Apakah ada kolaborasi dengan perpustakaan digital universitas dalam 
        praktikum Temu Kembali Informasi S2 Komputer di Dekanat L3 2 besok
        Apakah perkuliahan Pembelajaran Mesin S2 Komputer di Dekanat L3 3 
        akan mencakup studi kasus terkait computer vision atau NLP dan bagaimana pembagian kelompok proyek    
        """
    
        # Ekstrak entitas
        results = compare_model(text, tokenizer1, model1, tokenizer2, model2)
        entities_raw = results.get('hybrid', [])
        entities = group_entities_by_type(entities_raw)

        print("Entitas yang terdeteksi:")
        for ent_type, values in entities.items():
            print(f"{ent_type}: {values}")
        
        # entities = {
        #     'PS': ['S2Komputer'],
        #     'LOC': ['DekanatL33', 'DekanatL32', 'DekanatL31'],
        #     'PRD': ['N', 'computervision']  # PRD akan diabaikan
        # }

        print("\n🔍 Mencari data kuliah:")
        kuliah_searcher = SheetSearcher(kuliah_data, 'Kuliah')
        hasil_kuliah = kuliah_searcher.search(entities)

        if hasil_kuliah:
            print(f"Ditemukan {len(hasil_kuliah)} hasil:")
            for idx, item in enumerate(hasil_kuliah[:5], 1):
                print(f"{idx}. {item['NAMA MK']}")
                print(f"   PS: {item['PS']} | Ruang: {item['Ruang']}")
                print(f"   Waktu: {item['Waktu']} | Dosen: {item['Dosen PJ']}\n")
        else:
            print("Tidak ditemukan data kuliah yang sesuai")

        print("\n🔍 Mencari data seminar:")
        seminar_searcher = SheetSearcher(seminar_data, 'Seminar')
        hasil_seminar = seminar_searcher.search(entities)

        if hasil_seminar:
            print(f"Ditemukan {len(hasil_seminar)} hasil:")
            for idx, item in enumerate(hasil_seminar[:5], 1):
                print(f"{idx}. {item.get('Judul', 'Tidak ada judul')[:50]}...")
                print(f"   Tanggal: {item.get('Tanggal', '')} | Jam: {item.get('Jam', '')}")
                print(f"   Dosen Penguji: {item.get('Dosen 1', '')}\n")
        else:
            print("Tidak ditemukan data seminar yang sesuai")

    except Exception as e:
        print(f"Error: {str(e)}")
        
    
    # # Cari data kuliah dan seminar berdasarkan entitas
    # kuliah_results = searcher.search_kuliah(entities)
    # seminar_results = searcher.search_seminar(entities)
    
    # # Tampilkan hasil pencarian
    # print("\nHasil pencarian data Kuliah:")
    # for item in kuliah_results:
    #     print(item)
    
    # print("\nHasil pencarian data Seminar:")
    # for item in seminar_results:
    #     print(item)

if __name__ == "__main__":
    main()
