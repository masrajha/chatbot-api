import torch
from .utils import process_entities

def hybrid_ner(text, tokenizer1, model1, tokenizer2, model2):
    inputs1 = tokenizer1(text, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    offset_mapping = inputs1.pop('offset_mapping').squeeze().tolist()  # hapus offset_mapping dari inputs1

    inputs2 = tokenizer2(text, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    inputs2.pop('offset_mapping')  # HAPUS offset_mapping dari inputs2 JUGA
    
    with torch.no_grad():
        outputs1 = model1(**inputs1)
        outputs2 = model2(**inputs2)

    preds1 = torch.argmax(outputs1.logits, dim=2).squeeze().tolist()
    preds2 = torch.argmax(outputs2.logits, dim=2).squeeze().tolist()
    
    id2label1 = model1.config.id2label
    id2label2 = model2.config.id2label
    
    combined_labels = []
    for p1, p2 in zip(preds1, preds2):
        label1 = id2label1[p1]
        label2 = id2label2[p2]
        
        entity_type = None
        if any(tag in label1 for tag in ['PS', 'LOC', 'MK']):
            entity_type = label1
        else:
            entity_type = label2 if label2 not in ['PS', 'LOC', 'MK'] else 'O'
        
        combined_labels.append(entity_type)

    entities = []
    current_entity = None
    
    for i, (offset, label) in enumerate(zip(offset_mapping, combined_labels)):
        start, end = offset
        if start == 0 and end == 0:
            continue
        
        if label == "O":
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
        
        label_parts = label.split('-')
        prefix = label_parts[0]
        entity_type = label_parts[-1]
        
        if prefix == "B":
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "type": entity_type,
                "start": start,
                "end": end,
                "text": text[start:end]
            }
        elif prefix == "I" and current_entity:
            if current_entity["type"] == entity_type:
                current_entity["end"] = end
                current_entity["text"] += text[start:end]
            else:
                entities.append(current_entity)
                current_entity = None
    
    merged_entities = []
    for entity in entities:
        if not merged_entities:
            merged_entities.append(entity)
            continue
        
        last_entity = merged_entities[-1]
        if (entity["type"] == last_entity["type"] and
            entity["start"] == last_entity["end"] and
            text[last_entity["end"]:entity["start"]].strip() == ''):
            last_entity["end"] = entity["end"]
            last_entity["text"] += text[last_entity["end"]:entity["start"]] + entity["text"]
        else:
            merged_entities.append(entity)
    
    return merged_entities

def compare_model(text, tokenizer1, model1, tokenizer2, model2):
    inputs1 = tokenizer1(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )
    offset_mapping = inputs1.pop('offset_mapping').squeeze().tolist()  # hapus offset_mapping dari inputs1
    
    inputs2 = tokenizer2(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True
    )
    inputs2.pop('offset_mapping')  # HAPUS offset_mapping dari inputs2 juga
    
    with torch.no_grad():
        outputs1 = model1(**inputs1)
        outputs2 = model2(**inputs2)
    
    model1_entities = process_entities(outputs1, model1.config.id2label, offset_mapping, text)
    model2_entities = process_entities(outputs2, model2.config.id2label, offset_mapping, text)
    
    return {
        "model1": model1_entities,
        "model2": model2_entities,
        "hybrid": hybrid_ner(text, tokenizer1, model1, tokenizer2, model2)
    }
