# Disaster Response Pipeline Project

The Disaster Response Pipeline Project aims to build a machine learning model and a web application that can classify disaster messages and help direct them to the appropriate relief agencies. The project includes three main components: an ETL (Extract, Transform, Load) pipeline, an ML (Machine Learning) pipeline, and a Flask web app.

## Project Components

### 1. ETL Pipeline
In the `process_data.py` script, the ETL pipeline is implemented. This pipeline performs the following tasks:
- Loads the messages and categories datasets.
- Merges the two datasets based on a common key.
- Cleans the data by transforming the categories column into separate binary columns.
- Stores the cleaned data into a SQLite database.

### 2. ML Pipeline
The ML pipeline is implemented in the `train_classifier.py` script. This pipeline includes the following steps:
- Loads data from the SQLite database created by the ETL pipeline.
- Splits the dataset into training and test sets.
- Builds a text processing and machine learning pipeline that preprocesses the text data and applies a classification algorithm.
- Trains and tunes the model using GridSearchCV to find the best parameters.
- Outputs the evaluation results on the test set, including accuracy, precision, recall, and F1-score.
- Exports the final trained model as a pickle file.

### 3. Flask Web App
The Flask web app provides a user interface for entering new messages and obtaining classification results. It also displays visualizations of the data. The web app interacts with the trained model and the database. Users can enter a message and receive the predicted categories for that message. The web app also includes visualizations of the training dataset, such as genre distribution and message categories.

## Running the Project

To run the project, follow these steps:

1. Run the ETL pipeline to process and store the data in a SQLite database:
```
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

2. Run the ML pipeline to train the model and save it as a pickle file:
```
python train_classifier.py DisasterResponse.db classifier.pkl
```

3. Start the Flask web app:
```
python run.py
```

4. Open a web browser and go to `http://localhost:3001` to access the web app.

## Project Structure

The project consists of the following files and directories:

- `app/`
  - `run.py`: The main file to start the Flask web app.
  - `templates/`: Directory containing the HTML templates for the web app.
  - `static/`: Directory containing the CSS and JavaScript files for the web app.

- `data/`
  - `disaster_messages.csv`: CSV file containing the raw disaster messages.
  - `disaster_categories.csv`: CSV file containing the raw categories of the disaster messages.
  - `process_data.py`: Python script to perform the ETL pipeline tasks and store the cleaned data in a SQLite database.

- `models/`
  - `train_classifier.py`: Python script to implement the ML pipeline, train the model, and save it as a pickle file.

- `README.md`: This README file providing an overview of the project and instructions on running it.

## Dependencies

The project requires the following dependencies:
- Python (3.6 or higher)
- Libraries: pandas, numpy, sklearn, sqlalchemy, nltk, pickle, Flask, plotly

All the necessary packages can be installed using the `requirements.txt` file provided with the project. Run the following command to install the dependencies:
```
pip install -r requirements.txt
```

## Additional Notes

- The paths to the input data files (`disaster_messages.csv` and `disaster_categories.csv`) should
