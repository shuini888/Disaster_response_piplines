# Disaster Response Pipeline Project

### Project introduction
Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

### Project components
1. ETL Pipeline
process_data.py---- a data cleaning pipeline

2.ML Pipeline
train_classifier.py---- a machine learning pipeline

3. Flask Web App

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

The README file includes a summary of the project, how to run the Python scripts and web app,
and an explanation of the files in the repository.
