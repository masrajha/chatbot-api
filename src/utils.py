import torch
from collections import defaultdict
import re
from datetime import datetime, timedelta

def format_response(entities):
    unique_entities = defaultdict(set)
    for entity in entities:
        unique_entities[entity['type']].add(entity['text'])
    
    return {k: list(v) for k, v in unique_entities.items()}

def merge_entities(entities):
    merged = []
    for entity in entities:
        if not merged:
            merged.append(entity)
            continue
            
        last = merged[-1]
        if (entity["type"] == last["type"] and 
            entity["start"] == last["end"]):
            last["end"] = entity["end"]
            last["text"] += " " + entity["text"]
        else:
            merged.append(entity)
    return merged

def process_entities(model_output, id2label, offset_mapping, text):  # Tambahkan parameter text
    preds = torch.argmax(model_output.logits, dim=2).squeeze().tolist()
    labels = [id2label[p] for p in preds]
    
    entities = []
    current_entity = None
    
    for offset, label in zip(offset_mapping, labels):
        start, end = offset
        if start == 0 and end == 0:
            continue
            
        if label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        
        label_type = label.split("-")[-1]
        prefix = label.split("-")[0]
        
        if prefix == "B":
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "type": label_type,
                "start": start,
                "end": end,
                "text": text[start:end]  # Gunakan parameter text
            }
        elif prefix == "I" and current_entity and current_entity["type"] == label_type:
            current_entity["end"] = end
            current_entity["text"] += text[start:end]  # Gunakan parameter text
    
    if current_entity:
        entities.append(current_entity)
    
    # Proses penggabungan entitas
    return merge_entities(entities)

def group_entities_by_type(entities_list):
    grouped = {}
    for ent in entities_list:
        ent_type = ent.get("type")
        ent_text = ent.get("text")
        if ent_type and ent_text:
            grouped.setdefault(ent_type, set()).add(ent_text)
    return {k: list(v) for k, v in grouped.items()}

# Fungsi baru untuk mengonversi teks relatif menjadi tanggal
def convert_relative_dates(text):
    """
    Mengganti kata-kata relatif dalam teks dengan tanggal aktual dalam format YYYY-MM-DD
    Contoh:
      "hari ini" -> "2023-10-15" (tanggal hari ini)
      "besok"    -> "2023-10-16"
      "lusa"     -> "2023-10-17"
      "kemarin"  -> "2023-10-14"
    """
    today = datetime.now().date()
    
    # Mapping untuk kata relatif dan delta waktu
    date_mapping = {
        r'kemarin\b': today - timedelta(days=1),
        r'hari ini\b': today,
        r'besok\b': today + timedelta(days=1),
        r'lusa\b': today + timedelta(days=2),
    }
    
    # Fungsi untuk melakukan penggantian
    def replace_match(match):
        word = match.group(0).lower()
        for pattern, date_val in date_mapping.items():
            if re.search(pattern, word):
                return date_val.strftime("%Y-%m-%d")
        return word
    
    # Regex untuk menemukan semua kata relatif
    pattern = r'\b(kemarin|hari ini|besok|lusa)\b'
    return re.sub(pattern, replace_match, text, flags=re.IGNORECASE)

def extract_relative_dates(text):
    """
    Mengekstrak kata relatif dari teks dan mengonversinya menjadi tanggal
    Mengembalikan daftar tanggal dalam format YYYY-MM-DD
    """
    today = datetime.now().date()
    date_mapping = {
        'kemarin': today - timedelta(days=1),
        'hari ini': today,
        'besok': today + timedelta(days=1),
        'lusa': today + timedelta(days=2),
    }
    
    # Pola regex untuk menemukan kata relatif
    pattern = r'\b(kemarin|hari ini|besok|lusa)\b'
    found_dates = []
    
    # Cari semua kemunculan kata relatif
    matches = re.finditer(pattern, text, flags=re.IGNORECASE)
    for match in matches:
        keyword = match.group(0).lower()
        if keyword in date_mapping:
            date_val = date_mapping[keyword]
            found_dates.append(date_val.strftime("%Y-%m-%d"))
    
    return found_dates