import pandas as pd
import copy

from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# Converts a string representation of a list into an actual list
def fix_genre_string(string):
    genres = string[1:-1].split(",")
    return [genre.replace("'", "").strip() for genre in genres]

# Converts the genre list which is read as a string from the csv file into a python list
def fix_genre_formatting(data):
    genres = data['genres'].tolist()
    genres = [fix_genre_string(genrelist) for genrelist in genres]
    data['genres'] = genres
    return data

def summarize_genres(data):
    genres_to_collect = ['rock', 'pop', 'alternative rock', 'rap', 'folk rock']
    
    # returns all songs with at least one of the genres in genres to collect
    data = data[pd.DataFrame(data.genres.tolist()).isin(genres_to_collect).any(1).values]
    
    for index, row in data.iterrows():
        if "folk rock" in row["genres"]:
            data.loc[index, "genres"] = ["folk"]
        elif "rap" in row["genres"]:
            data.loc[index, "genres"] = ["rap"]
        elif "alternative rock" in row["genres"]:
            data.loc[index, "genres"] = ["alternative"]
        elif "pop" in row["genres"]:
            data.loc[index, "genres"] = ["pop"]
        elif "rock" in row["genres"]:
            data.loc[index, "genres"] = ["rock"]  

    return data

def balance_genres(data):
    data = data.sample(frac=1, random_state=0)
    g = data.groupby('genres')
    data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
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

# Splits a preprocessed lyrics data set into one hot encoded training and test data split
def get_text_and_labels(data):
    tags = data['tags'].to_list()
    lyrics = data['preprocessed_lyrics'].to_list()
    mlb = MultiLabelBinarizer()
    one_hot_labels = mlb.fit_transform(tags)
    
    X_train, X_test, y_train, y_test = train_test_split(lyrics, one_hot_labels)
    return X_train, X_test, y_train, y_test, mlb.classes_

# Generates all the tags based upon the information in the dataframe collected from spotify
def generate_tags(dataframe, thresholds={
"long_duration":300000,
"short_duration":120000,
"explicit":1,
"popularity":50,
"danceability":0.5,
"energy":0.5,
"loudness":-14.0,
"speechiness":0.5,
"acousticness":0.5,
"instrumentalness":0.5,
"liveness":0.8,
"valence":0.5
}):  
    
    # Create a deep copy to not affect the original dataframe 
    df = pd.DataFrame(columns = dataframe.columns, data = copy.deepcopy(dataframe.values))
    
    # Add tags to the tags list based upon threshold values passed to this function
    for index, row in df.iterrows():
        if "long_duration" in thresholds and row["duration_ms"] > thresholds["long_duration"]:
            df.loc[index, 'tags'].append("long_duration")
        elif "short_duration" in thresholds and row["duration_ms"] < thresholds["short_duration"]:
            df.loc[index, 'tags'].append("short_duration")
        
        if "explicit" in thresholds and row["explicit"]:
            df.loc[index, 'tags'].append("explicit")

        if "popularity" in thresholds and row["popularity"] > thresholds["popularity"]:
            df.loc[index, 'tags'].append("popular")

        if "danceability" in thresholds and row["danceability"] > thresholds["danceability"]:
            df.loc[index, 'tags'].append("danceable")

        if "energy" in thresholds and row["energy"] > thresholds["energy"]:
            df.loc[index, 'tags'].append("energetic")

        if "loudness" in thresholds and row["loudness"] > thresholds["loudness"]:
            df.loc[index, 'tags'].append("loud")

        if "speechiness" in thresholds and row["speechiness"] > thresholds["speechiness"]:
            df.loc[index, 'tags'].append("spoken_word")

        if "acousticness" in thresholds and row["acousticness"] > thresholds["acousticness"]:
            df.loc[index, 'tags'].append("acoustic")

        if "instrumentalness" in thresholds and row["instrumentalness"] > thresholds["instrumentalness"]:
            df.loc[index, 'tags'].append("instrumental")

        if "liveness" in thresholds and row["liveness"] > thresholds["liveness"]:
            df.loc[index, 'tags'].append("live")

        if "valence" in thresholds and row["valence"] > thresholds["valence"]:
            df.loc[index, 'tags'].append("postive")
        elif "valence" in thresholds:
            df.loc[index, 'tags'].append("negative")

    #Remove any unecessary columns in the data frame
    del df['track_id']
    del df['artist_id']
    del df['duration_ms']
    del df['explicit']
    del df['popularity']
    del df['danceability']
    del df['energy']
    del df['loudness']
    del df['speechiness']
    del df['acousticness']
    del df['instrumentalness']
    del df['liveness']
    del df['valence']
    del df['tempo']
    
    return df
        