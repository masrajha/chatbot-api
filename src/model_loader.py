from transformers import AutoTokenizer, AutoModelForTokenClassification

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
