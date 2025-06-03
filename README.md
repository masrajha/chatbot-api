Berikut adalah contoh **README.md** untuk proyek pengembangan **API Chatbot Akademik** berdasarkan file-file Python yang Anda unggah dan fungsionalitas yang saya pahami:

````markdown
# API Chatbot Akademik

API ini merupakan layanan backend chatbot akademik yang menyediakan fitur utama seperti ekstraksi entitas dari teks, pencarian data akademik (kuliah dan seminar) berbasis Google Sheets, klasifikasi intent, dan prediksi emosi dari gambar wajah.

## Fitur Utama

- **Named Entity Recognition (NER) Hybrid**  
  Menggunakan dua model BERT untuk ekstraksi entitas (seperti MK, DOSEN, TANGGAL) dari teks input pengguna.

- **Klasifikasi Intent**  
  Menentukan jenis pencarian data akademik yang sesuai (kuliah, seminar, atau keduanya) berdasarkan kata kunci.

- **Pencarian Data Akademik**  
  Mengambil dan mencari data jadwal kuliah dan seminar dari Google Sheets, menggunakan fuzzy matching dan alias mapping untuk pencarian lebih akurat.

- **Prediksi Emosi Wajah**  
  Menggunakan model Keras untuk memprediksi emosi dari gambar wajah yang diunggah melalui endpoint `/predict`.

- **Pengolahan Tanggal Relatif**  
  Mengkonversi kata-kata relatif seperti "besok", "kemarin" menjadi format tanggal standar.

## Struktur Proyek

- `app.py`  
  Entrypoint aplikasi Flask yang menyediakan endpoint `/search` dan `/predict`.

- `model_loader.py`  
  Modul untuk memuat model BERT NER dan klasifikasi intent.

- `ner_processor.py`  
  Modul untuk menjalankan inferensi NER hybrid menggunakan dua model dan menggabungkan hasilnya.

- `searcher.py`  
  Modul pencarian data di Google Sheets dengan teknik fuzzy matching dan alias mapping.

- `google_sheets.py`  
  Modul untuk autentikasi dan pengambilan data dari Google Sheets.

- `utils.py`  
  Fungsi utilitas seperti proses entitas, penggabungan entitas, dan konversi tanggal relatif.

## Instalasi dan Persiapan

1. **Install dependensi**  
   Pastikan Python 3.7+ sudah terpasang, kemudian install paket yang dibutuhkan:

   ```bash
   pip install -r requirements.txt
````

2. **Siapkan kredensial Google API**
   Letakkan file `credentials.json` untuk autentikasi Google Sheets API di root proyek.

3. **Siapkan model**

   * Model NER dan klasifikasi harus disimpan di folder `models/` sesuai dengan path di kode `model_loader.py`.
   * Model Keras untuk prediksi emosi wajah ada di `models/fer_models/`.

4. **Jalankan server Flask**

   ```bash
   python app.py
   ```

   Server berjalan di `http://0.0.0.0:5000`

## Dokumentasi API

### Endpoint `/search`

Menerima request POST JSON dengan format:

```json
{
  "text": "contoh pertanyaan tentang jadwal kuliah atau seminar"
}
```

**Response sukses:**

```json
{
  "message": "Berhasil",
  "data": [ /* list hasil pencarian kuliah atau seminar */ ],
  "entities": { /* entitas yang ditemukan dari NER */ },
  "search_types": ["kuliah", "seminar"]  // jenis data yang dicari
}
```

**Response error:**

* Jika teks kosong, status 400 dengan pesan error.
* Jika terjadi kesalahan server, status 500 dengan pesan error.

---


## Cara Kerja Sistem (High-Level)

1. Pengguna mengirimkan teks ke `/search`.
2. Teks diklasifikasikan untuk menentukan apakah pencarian berfokus pada data kuliah, seminar, atau keduanya.
3. Entitas diidentifikasi menggunakan model NER hybrid.
4. Data akademik diambil dari Google Sheets, dicari menggunakan entitas tersebut dengan teknik fuzzy matching dan alias mapping.
5. Hasil pencarian dikembalikan sebagai JSON.


---

## Dependensi Utama

* Flask, Flask-CORS
* Transformers (Hugging Face)
* Torch / PyTorch
* gspread, oauth2client (Google Sheets API)
* fuzzywuzzy
* pandas
* PIL / Pillow

---

## Lisensi

Proyek ini menggunakan lisensi MIT.

---

## Kontak dan Kontribusi

Untuk pertanyaan, bug report, atau kontribusi silakan hubungi pengembang utama.
