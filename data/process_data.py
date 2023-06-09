import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge the messages and categories datasets.
    
    Args:
        messages_filepath (str): Filepath of the messages dataset.
        categories_filepath (str): Filepath of the categories dataset.
    
    Returns:
        df (DataFrame): Merged DataFrame containing messages and categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
     """
    Perform cleaning and preprocessing of the merged DataFrame.
    
    Args:
        df (DataFrame): Merged DataFrame containing messages and categories.
    
    Returns:
        df (DataFrame): Cleaned DataFrame.
    """
    categories = df.categories.str.split(";", expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda i: i[:-2])

    categories.columns = category_colnames
    for column in categories:
        # Convert values other than 0 and 1 to NaN
        categories[column] = pd.to_numeric(categories[column], errors='coerce')
        # Fill NaN values with 0
        categories[column] = categories[column].fillna(0)
        # Convert values greater than 1 to 1
        categories[column] = categories[column].apply(lambda x: 1 if x > 1 else x)

    df = df.drop(columns='categories') 
    df = df.join(categories)
    
    df = df.drop_duplicates()
        
    return df


def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to a SQLite database.
    
    Args:
        df (DataFrame): Cleaned DataFrame.
        database_filename (str): Filename for the SQLite database.
    """

    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('InsertTableName', engine, index = False,if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
