import json
import plotly
import pandas as pd
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

# IBM Cloud API Setup
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions



# Ensure these environment variables are set in your system
api_key = os.getenv('ApiKey-bc1f211e-32d9-4212-930f-c9c9c8919eaa')
url = os.getenv('IBM_URL')

# Example usage
print(f"API Key: {api_key}")
print(f"Service URL: {url}")

# Authenticate and create a NaturalLanguageUnderstanding instance
authenticator = IAMAuthenticator(api_key)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url(url)

# Flask Setup
app = Flask(__name__)

# Function to process text (tokenization and lemmatization)
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

# Load data from SQLite database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Disasters', engine)

# Load model
model = joblib.load("../models/classifier.pkl")

# Function to analyze sentiment using IBM Watson
def analyze_sentiment(text):
    try:
        response = nlu.analyze(
            text=text,
            features=Features(sentiment=SentimentOptions())
        ).get_result()
        sentiment = response['sentiment']['document']['label']
        return sentiment
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return None

# Index route to display visuals and receive input text
@app.route('/')
@app.route('/index')
def index():
    # Extract data for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    cats = df[df.columns[5:]]
    cats_counts = cats.mean() * cats.shape[0]
    cats_names = list(cats_counts.index)
    nlarge_counts = cats_counts.nlargest(5)
    nlarge_names = list(nlarge_counts.index)

    # Create visuals
    graphs = [
        {
            'data': [Bar(x=genre_names, y=genre_counts)],
            'layout': {'title': 'Distribution of Message Genres', 'yaxis': {'title': "Count"}, 'xaxis': {'title': "Genre"}}
        },
        {
            'data': [Bar(x=nlarge_names, y=nlarge_counts)],
            'layout': {'title': 'Top Message Categories', 'yaxis': {'title': "Count"}, 'xaxis': {'title': "Category"}}
        },
        {
            'data': [Bar(x=cats_names, y=cats_counts)],
            'layout': {'title': 'Distribution of Message Categories', 'yaxis': {'title': "Count"}, 'xaxis': {'title': "Category"}}
        }
    ]
    
    # Encode plotly graphs in JSON
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('master.html', graphJSON=graphJSON)

# Go route to process user input and return classification and sentiment
@app.route('/go')
def go():
    query = request.args.get('query', '')

    # Use IBM API to get sentiment
    sentiment = analyze_sentiment(query)

    # Use the machine learning model to classify the message
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        sentiment=sentiment  # Display sentiment result
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
