import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_model_and_tokenizer():
    model1_path = "./models/cahya-ner-finetuned-anie-v2"  # sesuaikan path folder model Anda
    tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
    model1 = AutoModelForTokenClassification.from_pretrained(model1_path)
    return tokenizer1, model1

def process_entities(model_output, id2label, offset_mapping, text):
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
                "text": text[start:end]
            }
        elif prefix == "I" and current_entity and current_entity["type"] == label_type:
            current_entity["end"] = end
            current_entity["text"] += text[start:end]
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

def main():
    tokenizer, model = load_model_and_tokenizer()
    model.eval()

    text = """
        Apakah perkuliahan Manajemen Proyek IT S2 Ilkom di Sidang DKN L3 1 akan 
        menggunakan metodologi Agile dan Scrum dalam studi kasus nyata proyek TI
        Bagaimana mekanisme penilaian untuk mata kuliah Manajemen Proyek IT S2 
        Ilkom yang dilaksanakan di Sidang DKN L3 1 apakah berbasis simulasi manajemen 
        proyek atau penyusunan dokumen proyek lengkap
        Apakah perkuliahan Aplikasi Pengolah Angka D3 MI di MIPA T L1 A 
        akan fokus pada pengembangan macro dan automasi menggunakan Excel VBA
    """
    
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)
    offset_mapping = inputs.pop("offset_mapping").squeeze().tolist()

    with torch.no_grad():
        outputs = model(**inputs)
    
    id2label = model.config.id2label
    
    entities = process_entities(outputs, id2label, offset_mapping, text)

    print("Detected Entities:")
    for e in entities:
        print(f"Type: {e['type']}, Text: {e['text']}, Span: ({e['start']}, {e['end']})")

if __name__ == "__main__":
    main()
