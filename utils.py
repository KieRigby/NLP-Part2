import re
import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import array
from numpy import asarray
from numpy import zeros


nltk.download('wordnet')

def clean_lyrics(lyrics):
    replace_with_space_regex = re.compile('[/(){}\[\]\|@,;]')
    bad_symbol_regex = re.compile('[^a-z ]')
    
    lyrics = lyrics.lower() 

    lyrics = re.sub(replace_with_space_regex," ",lyrics)

    lyrics = re.sub(bad_symbol_regex,"", lyrics)
    lyrics = re.sub(r'\s+'," ", lyrics) # replace multiple spaces with single space
    
    return lyrics

def lemmatise(lyrics):
    words = lyrics.split()
    lemmatizer = WordNetLemmatizer()
    stem_tokens = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(stem_tokens)

# Generates a tokenizer for a training and test data split and returns the tokenized data sets with the tokenizer
def tokenize(x_train, x_test, maxlen=250, num_words=5000):
  tokenizer = Tokenizer(num_words)
  tokenizer.fit_on_texts(x_train)

  x_train = tokenizer.texts_to_sequences(x_train)
  x_test = tokenizer.texts_to_sequences(x_test)

  x_train = pad_sequences(x_train, padding='post',maxlen=maxlen)
  x_test = pad_sequences(x_test, padding='post',maxlen=maxlen)

  return x_train, x_test, tokenizer

# Loads the pretrained glove model from txt file
def load_glove():
    embeddings_dictionary = dict()

    glove_file = open('drive/MyDrive/UoS/Year 3/NLP/CW/Submission/glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()
    return embeddings_dictionary

#returns the word embeddings given a tokenizer object and glove model.
def get_embeddings_matrix(glove_model, tokenizer):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    return embedding_matrix

def preprocess_lyrics(data):
    lyrics = data['lyrics'].tolist()
    cleaned_lyrics = [clean_lyrics(song) for song in lyrics]
    no_stop_words = [remove_stopwords(song) for song in cleaned_lyrics]
    lemmatised = [lemmatise(song) for song in no_stop_words]
    data['preprocessed_lyrics'] = lemmatised
    return data


# loads the data file given it's path
def load_data(data_file):
    data = pd.read_csv(data_file)
    del data['Unnamed: 0']

    data = fix_genre_formatting(data)
    data = summarize_genres(data)
    data = balance_genres(data)
    
    genre_list = data["genres"].tolist()
    wrapped_genres = [[genre] for genre in genre_list]
    genres_for_plot = [genre for genre in genre_list]
    data["genres"] = wrapped_genres
    data["genres_for_plot"] = genres_for_plot
    
    data = data.rename({'genres': 'tags'}, axis=1)

    return data