import sys
import pandas as pd
import os
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sqlalchemy import create_engine

def load_data(database_filepath):
    """
    Load data from the SQLite database and extract features and target variables.
    
    Args:
        database_filepath (str): Filepath of the SQLite database.
    
    Returns:
        X (Series): Features (messages) as a Series.
        y (DataFrame): Target variables (categories) as a DataFrame.
        category_names (Index): Names of the target categories.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('InsertTableName', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names

def tokenize(text): 
     """
    Tokenize and preprocess the text data.
    
    Args:
        text (str): Input text to be tokenized and preprocessed.
    
    Returns:
        clean_tokens (list): List of cleaned and lemmatized tokens.
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls= re.findall(url_regex,text)
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")
    tokens= word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline and perform grid search for parameter tuning.
    
    Returns:
        cv (GridSearchCV): GridSearchCV object containing the pipeline and parameter grid.
    """
    pipeline = Pipeline([
                            ('vect', CountVectorizer(tokenizer=tokenize)),
                            ('tfidf', TfidfTransformer()),
                            ('clf', moc)
                        ])
    
    parameters = {
        'clf__estimator__max_depth':[10, 50],
        'clf__estimator__min_samples_leaf':[2,5,10]}
        
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters) 
    
    return cv
    
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of the model on the test set.
    
    Args:
        model: Trained model object.
        X_test (Series): Test set features (messages) as a Series.
        Y_test (DataFrame): Test set target variables (categories) as a DataFrame.
        category_names (Index): Names of the target categories.
    """
    Y_pred_test = model.predict(X_test)
    print(classification_report(Y_test.values, Y_pred_test, target_names=category_names))
    

def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.
    
    Args:
        model: Trained model object.
        model_filepath (str): Filepath to save the model.
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
        
if __name__ == '__main__':
    main()
