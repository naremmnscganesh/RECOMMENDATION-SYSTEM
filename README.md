# RECOMMENDATION-SYSTEM

🎬 Movie Recommendation System

This repository contains a **Movie Recommendation System** developed using Python and machine learning techniques. It aims to suggest movies to users based on their preferences by leveraging collaborative filtering and content-based filtering techniques.

## 📌 Project Overview

Recommendation systems are a crucial component of modern streaming services like Netflix and Amazon Prime. They enhance user experience by personalizing content suggestions. This project builds a hybrid recommendation engine that can suggest movies using both user ratings and movie content information such as genres, tags, and keywords.

The project is implemented using:

* **Python**
* **Pandas & NumPy** for data manipulation
* **Scikit-learn** for modeling and similarity computation
* **NLTK / TfidfVectorizer** for natural language processing
* **Cosine Similarity** for calculating content-based similarity

## 📂 Dataset

The dataset consists of:

* `movies.csv`: Contains movie titles, genres, and IDs.
* `ratings.csv`: Contains user ratings for different movies.
* `tags.csv`: Includes user-assigned tags for movies.
* `links.csv` (optional): Contains IMDB/ TMDB links for movies.

This data is combined and cleaned to create two types of recommendation engines:

* **Collaborative Filtering** (based on user ratings)
* **Content-Based Filtering** (based on movie descriptions and tags)

## 🧹 Data Preprocessing

Key preprocessing steps:

* Merge datasets on `movieId`
* Handle missing values and duplicates
* Convert genres and tags into a format suitable for NLP
* Create a pivot table for collaborative filtering
* Use TF-IDF vectors for textual similarity

## 🧠 Recommendation Techniques

### 1. Content-Based Filtering

* Use `TfidfVectorizer` on movie overviews and tags
* Compute **cosine similarity** between movies
* Recommend movies that are textually similar to the selected title

### 2. Collaborative Filtering

* Create a user-item matrix
* Recommend movies by analyzing user behavior (i.e., users who liked similar movies)

(Optional) Advanced methods such as SVD (Singular Value Decomposition) can be used for better collaborative filtering performance.

## 📈 Results and Evaluation

* Users can input a movie title and receive top N recommendations
* Most popular movies (based on ratings count) can be displayed
* Recommendations can be evaluated using metrics like RMSE (for collaborative models) or user satisfaction feedback

## 🚀 How to Use

1. Clone the repository
2. Install the required dependencies: pip install -r requirements.txt
3. Run the Jupyter Notebook: jupyter notebook Movie_Recommender.ipynb
4. Enter a movie title to get similar movie recommendations.

## 📦 Requirements
* Python 3.x
* pandas
* numpy
* scikit-learn
* nltk
* jupyter
* matplotlib (for optional visualization)

Install all dependencies with: pip install pandas numpy scikit-learn nltk matplotlib

**output:-**
![Image](https://github.com/user-attachments/assets/144204ad-6b22-483b-9a5c-8579844882be)
