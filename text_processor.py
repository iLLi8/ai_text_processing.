import re
import random
import nltk
from nltk.corpus import wordnet
from transformers import BartForConditionalGeneration, BartTokenizer
from bs4 import BeautifulSoup
import torch
import multiprocessing

# Force CPU usage instead of MPS (Mac Metal)
torch.device('cpu')

# Load BART Paraphrase Model
model_name = "eugenesiow/bart-paraphrase"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to("cpu")  # Force CPU

# Minimal Text Cleaning
def minimal_clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove excessive white spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove specific unwanted characters
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)

    return text

# Synonym Replacement
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    return synonyms

def replace_with_synonyms(text):
    words = text.split()
    for i in range(len(words)):
        if random.random() < 0.1:  # 10% chance of replacing a word with a synonym
            synonyms = get_synonyms(words[i])
            if synonyms:
                words[i] = random.choice(list(synonyms))
    return ' '.join(words)

# Paraphrasing with BART
def paraphrase_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest")
    outputs = model.generate(
        **inputs,
        num_beams=5,  # Adjust beam size
        num_return_sequences=1,
        max_length=60,  # Adjust based on input length
        temperature=1.2,  # Slightly lower for creativity
        do_sample=True  # Add this line
    )
    paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased_text

# Avoid 3-Part Listings
def avoid_3_part_listings(text):
    sentences = text.split('. ')
    modified_sentences = []
    for sentence in sentences:
        if sentence.count(', ') == 2:
            parts = sentence.split(', ')
            if random.choice([True, False]):
                modified_sentence = parts[0] + ' and ' + parts[1] + ', as well as ' + parts[2]
            else:
                modified_sentence = parts[0] + ', ' + parts[1] + ' and ' + parts[2]
        else:
            modified_sentence = sentence
        modified_sentences.append(modified_sentence)
    return '. '.join(modified_sentences)

# Add Natural Transitions
transitions = [
    "Furthermore,", "Moreover,", "In addition,", "On the other hand,", "However,", "Meanwhile,",
    "At the same time,", "Similarly,", "In contrast,", "Consequently,", "Therefore,", "Accordingly,",
    "Thus,", "Hence,", "So,", "For instance,", "and", "again", "and then", "besides",
    "equally important", "finally", "further", "nor", "too", "next", "lastly", "what's more",
    "whereas", "but", "yet", "on the other hand", "nevertheless", "on the contrary",
    "by comparison", "where", "compared to", "up against", "balanced against", "vis a vis",
    "although", "conversely", "meanwhile", "after all", "in contrast", "although this may be true"
]

def add_natural_transitions(text):
    sentences = text.split('. ')
    for i in range(1, len(sentences)):
        if random.random() < 0.2:  # 20% chance of adding a transition
            transition = random.choice(transitions)
            sentences[i] = transition + ' ' + sentences[i]
    return '. '.join(sentences)

# Introduce Imperfections
def introduce_imperfections(text):
    words = text.split()
    for i in range(len(words)):
        if random.random() < 0.05:  # 5% chance of introducing an error
            if random.choice([True, False]):
                # Spelling mistake
                words[i] = words[i].replace(random.choice(words[i]),
                                         random.choice('abcdefghijklmnopqrstuvwxyz'), 1)
            else:
                # Punctuation error
                words[i] = words[i] + random.choice([',', '.', '!', '?'])
    return ' '.join(words)

# Final Post-Processing Function (Renamed to process_text)
def process_text(text, style='casual'):
    text = minimal_clean_text(text)
    text = paraphrase_text(text)  # Use BART for paraphrasing
    text = replace_with_synonyms(text)
    text = avoid_3_part_listings(text)
    text = add_natural_transitions(text)
    text = introduce_imperfections(text)

    return text

if __name__ == '__main__':
    # Add multiprocessing support
    multiprocessing.freeze_support()

    original_text = """
    Traveling is one of the most enriching experiences a person can have. It opens your eyes to new cultures, traditions, and ways of life that you might never have encountered otherwise. Whether it’s exploring the bustling streets of a vibrant city, hiking through serene natural landscapes, or simply relaxing on a quiet beach, every journey offers something unique. Traveling also teaches valuable life skills, such as adaptability, problem-solving, and independence. It pushes you out of your comfort zone and encourages you to embrace the unknown. Moreover, it provides an opportunity to disconnect from the daily grind and reconnect with yourself and the world around you. The memories and stories you gather from your travels stay with you forever, shaping your perspective and enriching your life in countless ways. So, pack your bags, step out of your routine, and embark on an adventure that will leave you with a lifetime of unforgettable experiences.
    """

    processed_text = process_text(original_text, style='formal')
    print("\nOriginal Text:")
    print(original_text)
    print("\nProcessed Text:")
    print(processed_text)




