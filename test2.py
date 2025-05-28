from src.google_sheets import get_sheet_data
from src.searcher import SheetSearcher

def main():
    try:
        print("Mengambil data kuliah...")
        kuliah_data = get_sheet_data('Kuliah')
        print("Mengambil data seminar...")
        seminar_data = get_sheet_data('Seminar')

        entities = {
            'PS': ['S2Komputer'],
            'LOC': ['DekanatL33', 'DekanatL32', 'DekanatL31'],
            'PRD': ['N', 'computervision']  # PRD akan diabaikan
        }

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

if __name__ == "__main__":
    main()
