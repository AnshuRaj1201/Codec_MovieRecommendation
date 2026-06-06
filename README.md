# Codec_MovieRecommendation


A lightweight content-based movie recommendation engine built with Python. It uses **TF-IDF vectorization** and **cosine similarity** to suggest movies based on genre similarity — no ratings or user history needed.

---

## 📌 How It Works

1. **Dataset** — A small movie database with titles and genres is loaded into a Pandas DataFrame.
2. **Vectorization** — Movie genres (text) are converted into numerical vectors using `TfidfVectorizer`.
3. **Similarity Scoring** — Cosine similarity is computed between all movie vectors to measure how alike two movies are.
4. **Recommendation** — Given a movie title, the system finds the top 2 most similar movies and returns them as recommendations.

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data handling |
| `scikit-learn` | TF-IDF vectorization & cosine similarity |

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
```

**2. Install dependencies**
```bash
pip install pandas scikit-learn
```

---

## 🚀 Usage

Run the script directly:
```bash
python MovieRecommender.py
```

Or call `get_recommendations()` inside a Python session:
```python
get_recommendations('Inception', movies_df, similarity_scores)
```

---

## 📊 Sample Output

```
Our Movie Database:
          Title                   Genres
0    The Matrix             Action Sci-Fi
1     Inception    Action Sci-Fi Thriller
2  The Notebook            Romance Drama
3      Avengers  Action Adventure Sci-Fi
4       Titanic            Romance Drama

Because you watched 'The Matrix', we recommend:
- Inception (Match Score: 0.89)
- Avengers (Match Score: 0.58)
```

---

## 🗂️ Project Structure

```
movie-recommender/
│
├── MovieRecommender.py   # Main script
└── README.md             # Project documentation
```

---

## 🔮 Possible Improvements

- Add a larger, real-world dataset (e.g., from [TMDB](https://www.themoviedb.org/) or [MovieLens](https://grouplens.org/datasets/movielens/))
- Expand features beyond genres (cast, director, plot keywords)
- Build a simple CLI or web interface using Streamlit
- Incorporate collaborative filtering for user-based recommendations
- Add fuzzy matching for movie title lookup

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
