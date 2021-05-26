import os, sys
import argparse
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from pathlib import Path
from utils.utils import preprocess_lyrics, tokenize, load_glove, get_embeddings_matrix
from train_utils import generate_tags, get_text_and_labels, load_data

from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import GlobalMaxPooling1D, MaxPooling1D
from keras.layers import Conv1D
from keras.layers import Dropout, Flatten

parser = argparse.ArgumentParser(description='Train the model.')
parser.add_argument('-n', '--model-no', required=True, help='a unique integer for the model version number')
args = parser.parse_args()

file_path = Path("../data/cleaned_dataset.csv")

data = load_data(file_path)

data_selective_tags = generate_tags(data, {"valence":0.5})
preprocessed_selective_data = preprocess_lyrics(data_selective_tags)
x_train, x_test, y_train, y_test, labels = get_text_and_labels(preprocessed_selective_data)

x_train, x_test, tokenizer = tokenize(x_train, x_test, maxlen=250)
glove_model = load_glove()
embeddings = get_embeddings_matrix(glove_model, tokenizer)

def create_model():
  model = Sequential()
  model.add(Embedding(len(tokenizer.word_index) + 1, 100, weights=[embeddings], input_length=250))
  model.add(Conv1D(128, 5, activation='relu'))
  model.add(MaxPooling1D())
  model.add(Dropout(0.2))
  model.add(Conv1D(128, 4, activation='relu'))
  model.add(MaxPooling1D())
  model.add(Dropout(0.2))
  model.add(Conv1D(128, 3, activation='relu'))
  model.add(MaxPooling1D())
  model.add(Dropout(0.2))
  model.add(GlobalMaxPooling1D())
  model.add(Flatten())
  model.add(Dense(len(labels), activation='sigmoid'))
  model.compile(loss = "binary_crossentropy",
                optimizer = "sgd",
                metrics=['acc'])
  return model

model = create_model()
model.summary()

history = model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1, validation_split=0.2)

model_path = Path('../models') / str(args.model_no)
print(model_path)
model.save(model_path)