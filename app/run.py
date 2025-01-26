import json
import joblib
import plotly
import pandas as pd
import nltk
nltk.download('punkt_tab')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    return clean_tokens

# Load data
try:
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql('SELECT * FROM Disasters', engine)
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame()

# Load model
try:
    model = joblib.load("../models/classifier.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
@app.route('/index')
def index():
    try:
        # Extract data for visuals
        genre_counts = df.groupby('genre').count()['message']
        genre_names = list(genre_counts.index)

        cats = df[df.columns[5:]]  # Assuming columns 5 onward contain category labels
        cats_counts = cats.mean() * cats.shape[0]
        cats_names = list(cats_counts.index)

        nlarge_counts = cats_counts.nlargest(5)
        nlarge_names = list(nlarge_counts.index)

        # Create visuals
        graphs = [
            {
                'data': [
                    Bar(x=genre_names, y=genre_counts)
                ],
                'layout': {
                    'title': 'Distribution of Message Genres',
                    'yaxis': {'title': "Count"},
                    'xaxis': {'title': "Genre"}
                }
            },
            {
                'data': [
                    Bar(x=nlarge_names, y=nlarge_counts)
                ],
                'layout': {
                    'title': 'Top Message Categories',
                    'yaxis': {'title': "Count"},
                    'xaxis': {'title': "Category"}
                }
            },
            {
                'data': [
                    Bar(x=cats_names, y=cats_counts)
                ],
                'layout': {
                    'title': 'Distribution of Message Categories',
                    'yaxis': {'title': "Count"},
                    'xaxis': {'title': "Category"}
                }
            },
            {
                'data': [
                    Pie(labels=cats_names, values=cats_counts)
                ],
                'layout': {
                    'title': 'Proportion of Message Categories'
                }
            }
        ]

        ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
        graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('master.html', ids=ids, graphJSON=graphJSON)

    except Exception as e:
        return f"An error occurred while generating the visuals: {e}"

@app.route('/go')
def go():
    query = request.args.get('query', '')
    mode_message = ""

    if model:
        classification_labels = model.predict([query])[0]
        classification_results = dict(zip(df.columns[4:], classification_labels))

        # Check specific categories to trigger mode messages
        # (Adjust keys below to match your actual column names)
        if 'earthquake' in classification_results and classification_results['earthquake'] == 1:
            mode_message += "// earthquake mode activated // "
        if 'aid_related' in classification_results and classification_results['aid_related'] == 1:
            mode_message += "// first aid mode activated // "
        if 'alert' in classification_results and classification_results['alert'] == 1:
            mode_message += "// this mode is being activated // "

    else:
        classification_results = {}

    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        mode_message=mode_message
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
