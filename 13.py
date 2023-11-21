import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
reviews_data = pd.DataFrame({
    'Review': ["The product is great and works well.",
               "I am not satisfied with the quality.",
               "Excellent service and fast delivery."]
})
def calculate_word_frequency(review_text):
    words = word_tokenize(review_text.lower())
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_frequencies = Counter(filtered_words)
    return word_frequencies
all_word_frequencies = Counter()
for review in reviews_data['Review']:
    word_frequencies = calculate_word_frequency(review)
    all_word_frequencies += word_frequencies
print("Word Frequency Distribution:")
for word, frequency in all_word_frequencies.items():
    print(f"{word}: {frequency}")
