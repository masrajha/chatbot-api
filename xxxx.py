from fuzzywuzzy import fuzz
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

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
    
    # ... (ALIAS_MAPPING tetap sama)

    