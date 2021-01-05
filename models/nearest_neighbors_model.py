import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

# changed from relative to to full path
model = pickle.load(open("models/nearest_neighbors_model.sav", "rb"))
transformer = pickle.load(open("models/transformer.sav", "rb"))
strains = pd.read_csv("src/data/nn_model_strains.csv")


def predict(request_text):
    drop_cols = ['Unnamed: 0', 'name', 'ailment', 'all_text', 'lemmas']
    transformed = transformer.transform([request_text])
    dense = transformed.todense()
    recommendations = model.kneighbors(dense)[1][0]
    output_array = []
    for recommendation in recommendations:
        strain_row = strains.iloc[recommendation]
        info = strain_row.drop(drop_cols)
        strain_dict = info.to_dict()
        output_array.append(strain_dict)
    return output_array
