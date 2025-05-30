#searcher.py
from fuzzywuzzy import fuzz
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

class SheetSearcher:
    ENTITY_MAPPING = {
        'Kuliah': {
            'MK': ['NAMA MK'],
            'PS': ['PS'],
            'PER': ['Dosen PJ', 'Dosen Anggota'],
            'LOC': ['Ruang'],
            'TIM': ['Waktu'],
            'HARI': ['Hari']  # Tambahan mapping untuk hari
        },
        'Seminar': {
            'PER': ['Nama Mahasiswa', 'Dosen 1', 'Dosen 2', 'Dosen 3'],
            'TIM': ['Jam'],
            'DAT': ['Tanggal']
        }
    }
    
    ALIAS_MAPPING = {
        'PS': {
            's2komputer': 's2ilkom',
            's1ilkom': 's1ilkom',
            's1ilkomp': 's1ilkom',
            's1komputer': 's1ilkom',
            's1ilmukomputer': 's1ilkom',
            'si': 'sif',
            's1si': 'sif',
            'sisfo': 'sif',
            's1sisfo': 'sif',
            'sif': 'sif',
            'd3': 'd3',
            'd3manajemeninformatika': 'd3mi',
            'mi': 'd3mi',
            # Tambahkan mapping baru untuk 'D3 MI'
            'd3 mi': 'd3mi',
            'd3mi': 'd3mi'
        },
        'LOC': {
            'dekanatl33': 'sidangdknl33',
            'dekanatl32': 'sidangdknl32',
            'dekanatl31': 'sidangdknl31',
            'GIKA': 'GIKL1A',
            'GIKB': 'GIKL1B',
            'GIKC': 'GIKL1C',
            'GIKR2': 'GIKLT2',
            'MIPATA': 'MIPATL1A',
            'MIPATB': 'MIPATL1B',
            'rseminar': 'ruangseminar',
        }
    }

    def __init__(self, sheet_data: List[Dict], sheet_type: str):
        self.df = pd.DataFrame(sheet_data)
        self.sheet_type = sheet_type
        print(f"Initializing SheetSearcher with sheet_type = {self.sheet_type}")
        print(f"Columns available: {list(self.df.columns)}")
        self._preprocess_data()

    def _normalize(self, value: Any, entity_key: Optional[str] = None) -> str:
        """Normalize string and apply alias mapping if needed"""
        if pd.isna(value):
            return ""
            
        if isinstance(value, (int, float)):
            norm = str(value).lower().strip()
        else:
            norm = ''.join(c for c in str(value).lower().strip() if c.isalnum())
        
        if entity_key and entity_key in self.ALIAS_MAPPING:
            return self.ALIAS_MAPPING[entity_key].get(norm, norm)
        return norm

    def _preprocess_data(self):
        """Preprocess dataframe columns: normalize and alias"""
        # Untuk data seminar: konversi kolom Tanggal
        if self.sheet_type == 'Seminar' and 'Tanggal' in self.df.columns:
            try:
                self.df['Tanggal'] = pd.to_datetime(self.df['Tanggal'], errors='coerce')
                self.df['Tanggal'] = self.df['Tanggal'].dt.strftime('%Y-%m-%d')
            except Exception as e:
                print(f"Error converting Tanggal: {e}")
        
        # Normalisasi kolom lainnya
        for entity_key, cols in self.ENTITY_MAPPING.get(self.sheet_type, {}).items():
            for col in cols:
                if col in self.df.columns:
                    self.df[f'norm_{col}'] = self.df[col].apply(
                        lambda v: self._normalize(v, entity_key)
                    )

    def _convert_date_to_day(self, date_str: str) -> str:
        """Convert date string to Indonesian day name"""
        days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu"]
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            day_index = date_obj.weekday()  # Monday=0, Sunday=6
            if day_index < 6:  # 0-5 = Senin-Sabtu
                return days[day_index]
            return None
        except ValueError:
            return None

    def _fuzzy_match(self, query: str, target: str) -> bool:
        """Return True if fuzzy match score token_set_ratio > 85"""
        if not query or not target:
            return False
        return fuzz.token_set_ratio(query, target) > 85

    def _exact_match(self, query: str, target: str) -> bool:
        """Check for exact match after normalization"""
        if not query or not target:
            return False
        return self._normalize(query) == self._normalize(target)

    def search(self, entities: Dict[str, List[str]]) -> List[Dict]:
        # Siapkan queries untuk entitas non-DAT
        queries = {}
        date_values = []
        day_values = []  # Untuk menyimpan hasil konversi tanggal->hari
        
        # Tangani entitas DAT
        if 'DAT' in entities:
            if self.sheet_type == 'Seminar':
                date_values = [date for date in entities['DAT'] if self._is_valid_date(date)]
                print(f"Found DAT entities: {date_values}")
            elif self.sheet_type == 'Kuliah':
                # Konversi tanggal ke nama hari
                for date in entities['DAT']:
                    day = self._convert_date_to_day(date)
                    if day:
                        day_values.append(day)
                day_values = list(set(day_values))  # Hapus duplikat
                print(f"Converted DAT to days: {day_values}")
                
                # Tambahkan ke entitas HARI
                if day_values:
                    if 'HARI' in entities:
                        entities['HARI'].extend(day_values)
                    else:
                        entities['HARI'] = day_values
        
        # Siapkan queries HANYA untuk entitas yang valid dan ada di mapping
        valid_entities = self.ENTITY_MAPPING.get(self.sheet_type, {})
        queries = {}
        for entity_type, values in entities.items():
            if entity_type in valid_entities:
                cols = valid_entities[entity_type]
                norm_values = [self._normalize(v, entity_type) for v in values]
                queries[entity_type] = {'columns': cols, 'values': norm_values}
                print(f"Prepared query for {entity_type}: values={norm_values}, columns={cols}")

        # Build mask dengan logika AND antar entitas yang valid
        mask = pd.Series([True] * len(self.df))
        found_any_entity = False
        
        # Terapkan filter untuk setiap entitas yang valid
        for entity_type, query in queries.items():
            entity_mask = pd.Series([False] * len(self.df))
            found_col = False
            
            for col in query['columns']:
                norm_col = f'norm_{col}'
                if norm_col not in self.df.columns:
                    print(f"Warning: Normalized column {norm_col} not found in DataFrame")
                    continue
                    
                found_col = True
                found_any_entity = True  # Setidaknya ada 1 entitas valid
                
                # Gunakan exact match untuk HARI, fuzzy untuk lainnya
                if entity_type == 'HARI':
                    col_mask = self.df[norm_col].apply(
                        lambda cell_val: any(
                            self._exact_match(q_val, cell_val) 
                            for q_val in query['values']
                        )
                    )
                else:
                    # PERBAIKAN UTAMA: Gunakan partial_ratio untuk MK dengan threshold lebih rendah
                    if entity_type == 'MK':
                        col_mask = self.df[norm_col].apply(
                            lambda cell_val: any(
                                fuzz.partial_ratio(q_val, cell_val) > 75 or 
                                fuzz.token_set_ratio(q_val, cell_val) > 75
                                for q_val in query['values']
                            )
                        )
                    elif entity_type == 'TIM':  # Penanganan khusus untuk waktu
                        col_mask = self.df[norm_col].apply(
                            lambda cell_val: any(
                                q_val in cell_val  # Substring matching setelah normalisasi
                                for q_val in query['values']
                            )
                        )
                    else:
                        col_mask = self.df[norm_col].apply(
                            lambda cell_val: any(
                                self._fuzzy_match(q_val, cell_val) 
                                for q_val in query['values']
                            )
                        )
                entity_mask |= col_mask
            
            # Jika ditemukan kolom yang sesuai, terapkan mask
            if found_col:
                mask &= entity_mask
                print(f"Applied filter for {entity_type}, matches found: {entity_mask.sum()}")
            else:
                print(f"No valid columns found for {entity_type}, skipping")

        # Handle kasus tidak ada entitas valid sama sekali
        if not found_any_entity:
            print("No valid entities found in query, returning empty results")
            return []

        # Tambahkan filter tanggal untuk seminar
        if date_values and self.sheet_type == 'Seminar' and 'Tanggal' in self.df.columns:
            date_mask = self.df['Tanggal'].apply(
                lambda x: any(date_val == x for date_val in date_values)
            )
            mask &= date_mask
            print(f"Applied date filter for seminar, matches: {date_mask.sum()}")

        # Jika tidak ada hasil, kembalikan list kosong
        if mask.sum() == 0:
            print("No matching records found after all filters")
            return []

        results = self.df[mask].copy()
        print(f"Found {len(results)} matching records before sorting")

        # Compute score for sorting
        def compute_score(row):
            total = 0
            for entity_type, query in queries.items():
                for col in query['columns']:
                    norm_col = f'norm_{col}'
                    if norm_col not in row:
                        continue
                    cell_val = row[norm_col]
                    for q_val in query['values']:
                        # Beri bonus besar untuk match exact hari
                        if entity_type == 'HARI':
                            if self._exact_match(q_val, cell_val):
                                total += 100
                        else:
                            # Beri bonus lebih tinggi untuk match MK
                            score = max(
                                fuzz.partial_ratio(q_val, cell_val),
                                fuzz.token_set_ratio(q_val, cell_val)
                            )
                            if entity_type == 'MK' and score > 70:
                                score *= 1.5  # Beri bobot lebih untuk MK
                            total += score
            return total

        results['score'] = results.apply(compute_score, axis=1)
        results = results.sort_values('score', ascending=False)

        print(f"Returning {len(results)} sorted results")
        return results.drop(columns='score').to_dict('records')

    def _is_valid_date(self, date_str: str) -> bool:
        """Check if string is in valid YYYY-MM-DD format"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False