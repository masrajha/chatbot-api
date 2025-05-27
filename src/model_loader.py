from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_models():
    model1_path = "./models/cahya-ner-finetuned-anie-v2"
    model2_name = "cahya/bert-base-indonesian-NER"
    
    # Gunakan tokenizer dari model2 karena model1 adalah fine-tune dari model2
    tokenizer = AutoTokenizer.from_pretrained(model2_name)
    
    # Muat model
    model1 = AutoModelForTokenClassification.from_pretrained(model1_path)
    model2 = AutoModelForTokenClassification.from_pretrained(model2_name)
    
    return tokenizer, model1, model2