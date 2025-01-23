

# Emergency Dispatch System

This project involves the implementation of an ETL pipeline and a Machine Learning model that classifies disaster-related messages in real-time. The system uses a Flask Web App to assist emergency response teams by providing classified disaster messages that can be used to dispatch aid or resources efficiently.

## Instructions

This project is divided into three sections:

1. **Data Processing**: Building an ETL pipeline to extract data from source, clean the data, and save it in a SQLite database.
2. **Machine Learning Pipeline**: Build a machine learning pipeline that classifies text messages (such as tweets) into 36 different categories related to disasters.
3. **Web Application**: Running a Flask web application that displays the model's predictions in real-time based on user input.

## Dependencies

- Python 3.5+
- Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn
- Natural Language Processing Libraries: NLTK
- SQLite Database Libraries: SQLAlchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

## Installation

To clone the repository:

```bash
git clone https://github.com/abg3/Emergency-Dispatch-System.git
```

### Running the application

You can run the following commands in the project directory to set up the database, train the model, and save the model.

#### Run ETL pipeline to clean data and store the processed data in the database:

```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

#### Run the ML pipeline to load data from the database, train the classifier, and save the classifier as a pickle file:

```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

#### Start the Flask Web App:

```bash
python run.py
```

Open a browser and navigate to `http://0.0.0.0:3001/`. You can input any disaster-related message and view the classification results.

## Notebooks

In the **notebooks** folder, you can find two Jupyter notebooks that explain how the data was processed and how the model was trained:

- **ETL Preparation Notebook**: Describes the ETL pipeline implementation.
- **ML Pipeline Preparation Notebook**: Explains the machine learning pipeline, including NLTK and Scikit-Learn implementations. This notebook can also be modified to re-train or tune the model using Grid Search.

## Important Files

- `app/templates/*`: HTML templates used for the web application.
- `data/process_data.py`: The ETL pipeline used for data cleaning, feature extraction, and storing processed data in a SQLite database.
- `models/train_classifier.py`: A machine learning pipeline that loads the data, trains the model, and saves the trained model as a `.pkl` file for later use.
- `app/run.py`: Launches the Flask web application to classify disaster tweet messages.
- `disaster_response.PNG`: Screenshot showing how the website looks.

