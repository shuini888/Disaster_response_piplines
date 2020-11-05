import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    # split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    categories.index = df.id
    row = categories[0:1]
    category_colnames = row.apply(lambda x: list(x.str.split('-'))[0][0])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace `categories` column in `df` with new category columns
    df.drop(columns=['categories'], index=1, inplace=True)
    df.index = df.id
    df = pd.merge(df, categories, left_index=True, right_index=True)
    df = df.reset_index(drop=True)

    # check number of duplicates
    if df.duplicated().sum():
        # drop duplicates
        df = df.drop_duplicates(subset=['id', 'message', 'original', 'genre'], keep='first')

    # check number of duplicates
    if not df.duplicated().sum():
        return df


def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('DisasterResponse', engine, index=False)


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()