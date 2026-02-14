import re
import string
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    return text

if __name__ == "__main__":
    sample = "Congratulations! Visit http://example.com NOW!!! Win ₹1000"
    print(preprocess_text(sample))
