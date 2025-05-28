from fuzzywuzzy import fuzz
import pandas as pd
from typing import Dict, List, Any, Optional

class SheetSearcher:
    ENTITY_MAPPING = {
        'Kuliah': {
            'MK': ['NAMA MK'],
            'PS': ['PS'],
            'PER': ['Dosen PJ', 'Dosen Anggota'],
            'LOC': ['Ruang'],
            'TIM': ['Waktu']
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
            'SI': 's1sisteminformasi',
            'S1SI': 's1sisteminformasi',
            'SISFO': 's1sisteminformasi',
            'S1sisfo': 's1sisteminformasi',
            'SIF': 's1sisteminformasi',
            'MI': 'D3ManajemenInformatika',
            'D3MI': 'D3ManajemenInformatika',
            'D3': 'D3ManajemenInformatika',
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
            'rseminar': 'Ruang Seminar',
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
        for entity_key, cols in self.ENTITY_MAPPING.get(self.sheet_type, {}).items():
            for col in cols:
                if col in self.df.columns:
                    self.df[f'norm_{col}'] = self.df[col].apply(
                        lambda v: self._normalize(v, entity_key)
                    )

    def _fuzzy_match(self, query: str, target: str) -> bool:
        """Return True if fuzzy match score token_set_ratio > 85"""
        if not query or not target:
            return False
        return fuzz.token_set_ratio(query, target) > 85

    def search(self, entities: Dict[str, List[str]]) -> List[Dict]:
        # Prepare normalized queries for existing entities in mapping
        queries = {}
        for entity_type, values in entities.items():
            if entity_type in self.ENTITY_MAPPING.get(self.sheet_type, {}):
                cols = self.ENTITY_MAPPING[self.sheet_type][entity_type]
                norm_values = [self._normalize(v, entity_type) for v in values]
                queries[entity_type] = {'columns': cols, 'values': norm_values}

        if not queries:
            return []

        # Build mask with AND logic between entity types
        mask = pd.Series([True] * len(self.df))
        for entity_type, query in queries.items():
            entity_mask = pd.Series([False] * len(self.df))
            for col in query['columns']:
                norm_col = f'norm_{col}'
                if norm_col not in self.df.columns:
                    continue
                
                col_mask = self.df[norm_col].apply(
                    lambda cell_val: any(
                        self._fuzzy_match(q_val, cell_val) 
                        for q_val in query['values']
                    )
                )
                entity_mask |= col_mask
            mask &= entity_mask

        results = self.df[mask].copy()

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
                        total += fuzz.ratio(q_val, cell_val)
            return total

        results['score'] = results.apply(compute_score, axis=1)
        results = results.sort_values('score', ascending=False)

        return results.drop(columns='score').to_dict('records')
