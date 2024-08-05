import nltk
import pandas as pd
import re
import json
from bs4 import BeautifulSoup
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(cleaned_tokens)

    def scrape_website(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from paragraphs (you may need to adjust this based on the website structure)
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text

    def process_data(self, input_data, input_type):
        if input_type == 'text':
            cleaned_text = self.clean_text(input_data)
            return {'processed_text': cleaned_text}
        elif input_type == 'url':
            scraped_text = self.scrape_website(input_data)
            cleaned_text = self.clean_text(scraped_text)
            return {'url': input_data, 'processed_text': cleaned_text}
        elif input_type == 'csv':
            df = pd.read_csv(input_data)
            # Assume the CSV has a 'text' column
            df['processed_text'] = df['text'].apply(self.clean_text)
            return df.to_dict(orient='records')
        else:
            raise ValueError("Unsupported input type")

    def analyze_description(self, text):
        # Basic sentiment analysis using NLTK
        tokens = word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        return {'pos_tags': pos_tags}

    def create_unified_format(self, processed_data, input_type):
        if input_type in ['text', 'url']:
            return processed_data
        elif input_type == 'csv':
            return {'records': processed_data}

    def run_pipeline(self, input_data, input_type):
        processed_data = self.process_data(input_data, input_type)
        unified_data = self.create_unified_format(processed_data, input_type)
        analysis = self.analyze_description(json.dumps(unified_data))
        unified_data['analysis'] = analysis
        return json.dumps(unified_data, indent=2)

# Example usage
preprocessor = DataPreprocessor()

# Process text input
text_input = "This is a sample text for preprocessing. It contains some numbers (123) and special characters!?"
result_text = preprocessor.run_pipeline(text_input, 'text')
print("Text Input Result:")
print(result_text)

# Process URL input
url_input = "https://example.com"
result_url = preprocessor.run_pipeline(url_input, 'url')
print("\nURL Input Result:")
print(result_url)

# Process CSV input (assuming you have a CSV file named 'sample_data.csv' with a 'text' column)
csv_input = "sample_data.csv"
result_csv = preprocessor.run_pipeline(csv_input, 'csv')
print("\nCSV Input Result:")
print(result_csv)
