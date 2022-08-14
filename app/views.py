from django.shortcuts import render
import numpy as np
import pandas as pd
import ast
import ast
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Create your views here.

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies.head()
credits.head()

movies = movies.merge(credits, on='title')
movies.shape
movies.head()

movies.head(1)

movies = movies[['movie_id', 'title', 'overview','genres', 'keywords', 'cast', 'crew']]
movies.head()

movies.isnull().sum()

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies.head()

movies['keywords'] = movies['keywords'].apply(convert)
movies.head()

ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
    return L

movies['cast'] = movies['cast'].apply(convert)
movies.head()

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

movies.head()

movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies.head()

movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(" ", "")for i in x])
movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(" ", "")for i in x])
movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(" ", "")for i in x])
movies['crew'] = movies['crew'].apply(
    lambda x: [i.replace(" ", "")for i in x])

movies.head()
movies['tags'] = movies['overview']+movies['genres'] + movies['keywords']+movies['cast']+movies['crew']

movies.head()
new_df = movies[['movie_id', 'title', 'tags']]

new_df

new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

new_df.head()

ps = PorterStemmer()

def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
new_df['tags'][0]

new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

new_df.head()

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

vectors[0]
len(cv.get_feature_names())

ps.stem('loved')

similarity = cosine_similarity(vectors)
sorted(list(enumerate(similarity[0])),reverse=True, key=lambda x: x[1])[1:6]



def home(request):
    if(request.method=="POST"):
        name = request.POST['movie']
        new = []
        
        def recommend(movie):
            try:
                movie_index = new_df[new_df['title'] == movie].index[0]
                distances = similarity[movie_index]
                movies_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x: x[1])[1:10]
                for i in movies_list:
                    new.append(new_df.iloc[i[0]].title)
            except IndexError:
                pass
        recommend(name)

        if len(new)==0:
            msg = "empty"
            return render(request,'recommend1.html',{'content':new,'moviename':name,'message':msg})
        else:
            msg = "nonempty"
            return render(request,'recommend1.html',{'content':new,'moviename':name,'message':msg})
    return render(request,'index.html')