from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import torch,json,re

def load_models():
    model1_path = "models/cahya-ner-finetuned-anie-v3"
    model2_name = "cahya/bert-base-indonesian-NER"
    
    # Load tokenizer masing-masing model
    tokenizer1 = AutoTokenizer.from_pretrained(model1_path)
    tokenizer2 = AutoTokenizer.from_pretrained(model2_name)
    
    # Load model
    model1 = AutoModelForTokenClassification.from_pretrained(model1_path)
    model2 = AutoModelForTokenClassification.from_pretrained(model2_name)
    
    return tokenizer1, model1, tokenizer2, model2

def load_model_classify(checkpoint_path):
    # Load tokenizer dari model base
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    # Load model dari checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        use_safetensors=True
    )

    # Load label mapping
    with open(f"{checkpoint_path}/config.json") as f:
        config = json.load(f)
        id2label = config.get("id2label", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, tokenizer, id2label, device

def remove_question_mark(text):
    # Hapus semua tanda tanya
    return re.sub(r'\?', '', text).strip()

def classify_intent(text: str, model, tokenizer, id2label, device):
    # Tokenisasi input tunggal
    
    text_remove = remove_question_mark(text)
    # text_remove = text
    
    inputs = tokenizer(
        [text_remove],  # tetap harus dalam list karena tokenizer batch
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
        return_attention_mask=True
    ).to(device)

    # Prediksi
    with torch.no_grad():
        outputs = model(**inputs)

    # Hitung probabilitas
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    confidence, pred_index = torch.max(probs, dim=1)

    predicted_label = id2label.get(str(pred_index.item()), "unknown")

    details = {id2label.get(str(j), str(j)): prob.item()
               for j, prob in enumerate(probs[0])}

    result = {
        "text": text,
        "intent": predicted_label,
        "confidence": confidence.item(),
        "details": details
    }

    return [result["intent"]]

