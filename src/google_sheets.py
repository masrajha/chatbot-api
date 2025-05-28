import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Fungsi untuk otentikasi dengan Google Sheets API
def authenticate_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    return client

# Fungsi untuk mendapatkan data dari sheet tertentu
def get_sheet_data(sheet_name):
    # Otentikasi ke Google Sheets
    client = authenticate_google_sheets()
    
    # Buka spreadsheet dengan ID
    spreadsheet = client.open_by_key('1MWieoIUPwgJGbwfLIawj1Rtka02VTI3xRVt-XHkrj3k')
    
    # Pilih sheet berdasarkan nama
    sheet = spreadsheet.worksheet(sheet_name)
    
    # Ambil data mentah termasuk header
    raw_data = sheet.get_all_values()
    
    # Bersihkan header dari spasi dan sel kosong
    headers = [cell.strip() for cell in raw_data[0] if cell.strip()]
    
    # Hapus duplikat header kosong
    unique_headers = []
    seen = set()
    for header in headers:
        if header not in seen:
            seen.add(header)
            unique_headers.append(header)
    
    # Validasi header sesuai tipe sheet
    expected_headers = {
        'Kuliah': ['KODE MK', 'NAMA MK', 'NAMA KELAS', 'PS', 'SEM', 
                   'Dosen PJ', 'Dosen Anggota', 'Hari', 'Waktu', 'Ruang'],
        'Seminar': ['Nama Mahasiswa', 'NPM','Seminar','Judul','Tanggal', 'Jam', 
                   'Dosen 1', 'Dosen 2', 'Dosen 3']
    }
    # print (unique_headers)
    if unique_headers != expected_headers[sheet_name]:
        raise ValueError(f"Format header {sheet_name} tidak valid. Pastikan kolom sesuai!")
    
    # Bangun data secara manual
    data = []
    for row in raw_data[1:]:
        record = {}
        for i, header in enumerate(unique_headers):
            record[header] = row[i] if i < len(row) else ''
        data.append(record)
    
    return data

# Fungsi untuk mencari data pada sheet tertentu
def search_in_sheet(sheet_name, search_term):
    data = get_sheet_data(sheet_name)
    
    # Filter data berdasarkan search_term
    filtered_data = [row for row in data if search_term.lower() in str(row).lower()]
    
    return filtered_data
