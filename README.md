# Movie Recommendation Model

Give the model a movie you like and it recommends five similar ones.

It is a **content-based** recommender: it compares what movies are about (plot,
genres, keywords, cast and director). It does not use viewer ratings.

## Quick start

```bash
# Set up (only needed once)
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# Recommend for one movie
.venv/bin/python main.py "Avatar"

# Show each data step, then ask for movies
.venv/bin/python main.py
```

From Python:

```python
from main import recommend

recommend("Avatar")   # list of (title, similarity) pairs
```

Titles are not case-sensitive. If several movies share a title, add the year:
`"Batman (1989)"`. Misspelled titles get suggestions.

## Files

| File | Purpose |
|---|---|
| `main.py` | All the code |
| `requirements.txt` | pandas and scikit-learn versions |
| `data/tmdb_5000_movies.csv` | Movie details: 4,803 rows, 20 columns |
| `data/tmdb_5000_credits.csv` | Cast and crew: 4,803 rows, 4 columns |

The data is the public TMDB 5000 dataset, downloaded from a
[GitHub mirror](https://github.com/beingpratheek/Movie-recommender-system). A
second mirror had byte-for-byte identical files.

## How it works

### 1. Load and merge: `load_data()`

Read both files with pandas and merge them on `title`.

### 2. Explore: `explore()`

Print the shape, `movies.head()` and a sample of the raw genres. Genres,
keywords, cast and crew are stored as JSON with extra details such as ids:

```
[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, ...]
```

### 3. Clean: `clean()`

- Drop duplicates.
- Drop rows whose movie ids disagree (see below).
- Keep the id, title, year and overview. A missing overview becomes empty text.
- Keep only the names of genres and keywords.
- Keep the top 3 actors from the cast.
- Keep only the directors from the crew.

Example:

| Column | Avatar |
|---|---|
| genres | Action, Adventure, Fantasy, Science Fiction |
| keywords | culture clash, future, space war, ... |
| cast | Sam Worthington, Zoe Saldana, Sigourney Weaver |
| crew | James Cameron |

**Why check the ids?** *Batman*, *The Host* and *Out of the Blue* are each the
title of two different films. Merging on title pairs each film with the other's
credits too, adding 6 wrong rows. Removing them takes 4,809 rows back to 4,803.

### 4. Prepare the tags: `prepare()`

Combine overview, genres, keywords, cast and crew into one column,
`movies["tags"]`.

- **Genres, keywords and names** are lowercased with spaces removed. "Sam
  Worthington" becomes `samworthington`, so the model doesn't mix this Sam up
  with another Sam.
- **Overview** stays as ordinary lowercase words without punctuation.

Avatar's tags, shortened:

```
in the 22nd century a paraplegic marine ... action adventure fantasy
sciencefiction cultureclash ... samworthington zoesaldana sigourneyweaver
jamescameron
```

### 5. Turn words into numbers: `build_model()`

`CountVectorizer(max_features=5000)` takes the 5,000 most common words and counts
how often each appears in each movie's tags. Every movie becomes a list of 5,000
numbers, giving a 4,803 x 5,000 table. Common English words such as "the" and
"and" are ignored.

The model is built once, then reused for every `recommend()` call.

### 6. Compare movies with cosine similarity

```
similarity = (A · B) / (length of A x length of B)
```

Here scores run from 0 to 1. **The closer to 1, the more similar the movies.** 0
means no counted words in common. Dividing by the lengths stops a movie scoring
high just because it has many words.

A score is not a percentage chance that you will like the movie.

### 7. Recommend: `recommend(movie)`

1. Find the movie's row (Avatar is row 0).
2. Score every movie against it.
3. Sort from most to least similar.
4. Skip the movie itself.
5. Return the next five.

The movie is skipped by its row number, not by being first in the list. Another
movie with the same proportions of tags also scores 1 and could come first.

## Example results

| Movie | Recommendations |
|---|---|
| The Dark Knight Rises | The Dark Knight, Batman Begins, Batman Returns, Batman (1989), Batman Forever |
| Spectre | Quantum of Solace, Never Say Never Again, Skyfall, From Russia with Love, Thunderball |
| Avatar | Titan A.E., Small Soldiers, Aliens vs Predator: Requiem, Ender's Game, Independence Day |
| Titanic | Raise the Titanic, Captain Phillips, The Notebook, In the Heart of the Sea, Ghost Ship |

All 4,803 movies get five recommendations, and none recommends itself.

## What we tried

Quality was judged by eye on a handful of movies.

| Version | Result |
|---|---|
| TF-IDF, no overview | Good for franchises |
| CountVectorizer, no overview | Weaker; favored movies with few tags |
| CountVectorizer, 5,000 features, no overview | Weaker again; 22 movies got fewer than five picks |
| **CountVectorizer, 5,000 features, with overview (current)** | Better for most movies checked; all get five picks |

## Limitations

- **Not measured:** without user ratings there is no accuracy score.
- **Character names can mislead:** *Toy Story* gets *The 40 Year Old Virgin*, most
  likely because both have a main character called Andy.
- **Some picks are poor:** *Inception*'s top match is *Duplex*, and *The
  Godfather*'s top five misses Part III.
- **Cast and director count for less** now that plot words outnumber them.
- **Limited to the dataset:** movies that aren't in it can't be used or recommended.

## Ideas to try next

None of these has been tested with the overview included.

- Use `TfidfVectorizer`, which gives common words less weight.
- Repeat the director and cast in the tags so they count for more.
- Apply stemming so that words like "love" and "loved" match.
