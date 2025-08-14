from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
import chromadb

def detect_language(text):
    """Detect the language of the input text."""
    try:
        return detect(text)
    except:
        return 'en'

def translate_text(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text
        
    try:
        model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(**inputs)
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translated_text
    except:
        return text
