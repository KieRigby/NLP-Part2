from ../utils.py import preprocess_lyrics

data_selective_tags = generate_tags(data, {"valence":0.5})
preprocessed_selective_data = preprocess_lyrics(data_selective_tags)