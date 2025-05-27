import torch
from collections import defaultdict

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
