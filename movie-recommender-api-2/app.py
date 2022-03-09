import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from flask import Flask, request, render_template
import re
import joblib

#Neccessary files has been imported. final_data having the cleaned and combined text with movie title.
#model has the parameter of the model trained.
#count_matrix has the matrix of the vector for the combined features.

data = pd.read_csv('final_data.csv')
model = joblib.load('model.pkl')
count_matrix = joblib.load('count_matrix.pkl')

#This method will be called ,once we have the movie choice filled by user on main_page.html
def recommend(choice):
    
    #If the movie name exactly matches with the name of the movie in the dataset 'title' column.
    
    if choice in data['title'].values:
        choice_index = data[data['title'] == choice].index.values[0]
        distances,indices = model.kneighbors(count_matrix[choice_index],n_neighbors=16)
        movie_list = []
        for i in indices.flatten():
            movie_list.append(data[data.index==i]['original_title'].values[0].title())
        return movie_list

    #If excat movie name is not present in our dataset 'title' column: 
    #Then below will try to find the similar movie name in 'title' column using contains or in.
    
    elif (data['title'].str.contains(choice).any() == True):
        
        #getting list of similar movie names as choice.
        similar_names = list(str(s) for s in data['title'] if choice in str(s) )
        #sorting the list to get the most matched movie name.
        similar_names.sort()
        #taking the first movie from the sorted similar movie name.
        new_choice = similar_names[0]
        print(new_choice)
        #getting index of the choice from the dataset
        choice_index = data[data['title'] == new_choice].index.values[0]
        #getting distances and indices of 16 mostly related movies with the choice.
        distances,indices = model.kneighbors(count_matrix[choice_index],n_neighbors=16)
        #creating movie list
        movie_list = []
        for i in indices.flatten():
            movie_list.append(data[data.index==i]['original_title'].values[0].title())
        return movie_list
    
    #If similar movie name is also not matches/present.
    else:
        return "opps! movie not found in our database"
            
#Flask API

app = Flask(__name__)

#for /search, it will open the main_page.html to take the user input.
@app.route("/search")
def home():
    return render_template('main_page.html')

#for /recommeded, it will provide the output/resultto user in movie_list.html web page.
@app.route("/Recommend")
def search_movies():
    #getting user input
    choice = request.args.get('movie')
    #removing all the characters except alphabets and numbers.
    choice = re.sub("[^a-zA-Z1-9]","",choice).lower()
    #passing the choice to the recommend() function
    movies = recommend(choice)

    #if rocommendation is a string(when movie name does not matches exactly or similar) and not list, 
    #then it will return movies and 'opps' as string: else it will return movies(list of movie name) 
    if type(movies) == type('string'):
        return render_template('movie_list.html',movie= movies,s='opps')
    else:
        return render_template('movie_list.html',movie=movies)

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=8080)
    app.run(debug=False)
    
    
