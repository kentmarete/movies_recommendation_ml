"""Content-based movie recommender: give it a movie you like, get five others.

    python main.py "Avatar"
    python main.py            # prompts for titles
"""
import difflib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).parent / "data"


# 1. Load both files and merge them on title
def load_data():
    movies = pd.read_csv(DATA / "tmdb_5000_movies.csv")
    credits = pd.read_csv(DATA / "tmdb_5000_credits.csv")
    return movies.merge(credits, on="title")


# 2. Explore
def explore(movies):
    print("Shape:", movies.shape)
    print(movies.head())
    print("\nRaw genres (JSON):", movies["genres"].iloc[0])


# 3. Clean: the JSON columns become plain lists of names
def names(text, limit=None):
    return [item["name"] for item in json.loads(text)][:limit]


def directors(text):
    return [p["name"] for p in json.loads(text) if p["job"] == "Director"]


def clean(movies):
    # Different films can share a title (e.g. Batman 1966 and 1989). Merging on
    # title gives them each other's credits, so keep only rows whose ids agree.
    movies = movies[movies["id"] == movies["movie_id"]]
    movies = movies.drop_duplicates(subset="movie_id").reset_index(drop=True)
    return pd.DataFrame({
        "movie_id": movies["movie_id"],
        "title": movies["title"],
        "year": movies["release_date"].str[:4].fillna("unknown"),
        "overview": movies["overview"].fillna(""),
        "genres": movies["genres"].map(names),
        "keywords": movies["keywords"].map(names),
        "cast": movies["cast"].map(lambda c: names(c, 3)),
        "crew": movies["crew"].map(directors),
    })


# 4. Prepare: combine overview, genres, keywords, cast and crew into one "tags" column
def remove_spaces(values):
    # "Sam Worthington" -> "samworthington", so the full name is one word and
    # doesn't match every other Sam.
    return [value.replace(" ", "").lower() for value in values]


def words(text):
    # Plot summaries are ordinary text: lowercase words, no punctuation.
    return re.findall(r"\w+", re.sub(r"['’]s\b", "", text.lower()))


def prepare(movies):
    movies = movies.copy()
    parts = [movies["overview"].map(words)]
    parts += [movies[col].map(remove_spaces)
              for col in ("genres", "keywords", "cast", "crew")]
    movies["tags"] = sum(parts[1:], parts[0]).map(" ".join)
    return movies[["movie_id", "title", "year", "tags"]]


# 5. Model: each movie's tags become a vector of word counts; similar movies have
#    a high cosine similarity
@lru_cache  # build once, then reuse for every recommend() call
def build_model():
    movies = prepare(clean(load_data()))
    # Take the 5000 most common words and count them in each movie's tags.
    # Split on spaces only, so names like "jean-claudevandamme" stay whole.
    # Stop words ("the", "and", ...) say nothing about a movie, so drop them.
    vectorizer = CountVectorizer(max_features=5000, stop_words="english",
                                 tokenizer=str.split, token_pattern=None)
    vectors = vectorizer.fit_transform(movies["tags"])
    labels = movies["title"] + " (" + movies["year"] + ")"
    return movies, vectors, labels


def find_row(movie):
    """Row number for 'Title' or 'Title (year)', ignoring case."""
    movies, _, labels = build_model()
    query = movie.strip().lower()
    matches = movies.index[(movies["title"].str.lower() == query)
                           | (labels.str.lower() == query)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Several movies match. Did you mean: {', '.join(labels[matches])}?")
    lowered = {t.lower(): t for t in movies["title"]}
    close = [lowered[t] for t in difflib.get_close_matches(query, lowered)]
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    raise ValueError(f"'{movie}' is not in the dataset.{hint}")


# 6. Recommend
def recommend(movie):
    """Return the five most similar movies as (title, score) pairs."""
    _, vectors, labels = build_model()

    # Find the row of the movie (Avatar is row 0)
    row = find_row(movie)

    # Score every movie against it, then sort from most to least similar
    scores = cosine_similarity(vectors[row], vectors).ravel()
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

    # Skip the movie itself and keep the next five. Skip by row rather than by
    # position: movies with identical tags also score 1 and could come first.
    top = [(i, score) for i, score in ranked if i != row and score > 0][:5]
    return [(labels[i], float(score)) for i, score in top]


def show(movie):
    try:
        results = recommend(movie)
    except ValueError as error:
        print(error)
        return
    if not results:
        print("No similar movies found (this movie has too little data).")
    for rank, (label, score) in enumerate(results, 1):
        print(f"{rank}. {label}  similarity {score:.2f}")


def main():
    movie = " ".join(sys.argv[1:])
    if movie:
        show(movie)
        return
    movies = load_data()
    explore(movies)
    cleaned = clean(movies)
    print("\nCleaned:", cleaned.shape)
    print(cleaned.head())
    print("\nTags for Avatar:", prepare(cleaned)["tags"].iloc[0])
    while movie := input("\nMovie you like (Enter to quit): ").strip():
        show(movie)


if __name__ == "__main__":
    main()
