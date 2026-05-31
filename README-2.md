# Movie Genre Classification (Multi-Label NLP)

A multi-label movie genre prediction system built on **42,000+ films** from the MovieLens/TMDB dataset, using transformer-based sentence embeddings and classical ML classifiers.

Built as part of the **Data Science Society (DSS) Decal** at UC Berkeley.

---

## 🎯 Project Goal

Predict a movie's genres from its **title, tagline, and overview text** using NLP embeddings and multi-label classification. The core challenge: movies belong to multiple genres simultaneously, and the genre distribution is highly imbalanced.

---

## 📂 Dataset

- **Source:** MovieLens + TMDB metadata (~42,324 films after cleaning)
- **Target:** Multi-label genre tags (20 genres → filtered to 14 after cutoff)
- **Key features:** `overview` (primary NLP feature), `title`, `tagline`, `budget`, `revenue`, `release_date`

---

## 🛠️ Pipeline

### Preprocessing
- Parsed semi-structured IMDb metadata fields using regex
- Applied log transforms to `budget` and `revenue` (heavily right-skewed)
- One-hot encoded multi-label genre targets using `MultiLabelBinarizer`
- Dropped rows with missing overview text

### Embeddings
Converted movie text into dense semantic vectors using transformer models:

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `all-MiniLM-L6-v2` | 384 | Fast, less semantic depth |
| `all-mpnet-base-v2` | 768 | Balanced speed/quality |
| `intfloat/e5-large-v2` | 1024 | Best performance — captures storyline, tone, and theme |

Best embedding: **e5-large-v2** with `title + tagline + overview` concatenation.

### Class Imbalance Treatment
- Applied a **genre frequency cutoff of 3,000** — reduced from 20 to 14 genres
- Compared cutoff vs. weight rebalancing vs. both — cutoff alone performed best
- Threshold tuning: optimal decision threshold of **0.28** to balance precision/recall

---

## 🤖 Models

All models tested on MiniLM-L6-v2 embeddings, cutoff 2,000, threshold 0.25:

| Model | Macro F1 | Micro F1 | Training Time |
|-------|----------|----------|---------------|
| Logistic Regression | 0.5454 | 0.6014 | ~1 min |
| XGBoost | 0.5278 | 0.5965 | ~12 min |
| LightGBM | 0.5252 | 0.5929 | ~6 min |

**Why Logistic Regression won:** e5-large-v2 produces 1,024-dim vectors where genre clusters are approximately linearly separable. Tree models split one feature at a time — they need exponentially more splits to match what logistic regression captures in one hyperplane, with no accuracy gain.

---

## 📊 Final Results

**Best configuration:** `e5-large-v2` + `title + tagline + overview` + cutoff 3,000 + threshold 0.28 + Logistic Regression

| Metric | Score |
|--------|-------|
| Micro F1 | **0.6825** |
| Macro F1 | **0.6608** |

**Per-genre highlights:**
- Best: Documentary (0.795), Horror (0.744), Drama (0.738)
- Hardest: Adventure (0.502) — heavily overlaps with Action, Drama, and Sci-Fi

---

## 🔮 Future Work

- Fine-tune e5-large-v2 on movie overviews with contrastive genre labels
- Test newer embeddings: GTE-large, BGE-M3, OpenAI text-embedding-3-large
- Explore BERT + classification head for joint multi-label learning
- Per-genre threshold tuning to recover recall on underrepresented labels

---

## 🛠️ Tech Stack

**Python** · **PyTorch** · **SentenceTransformers** · **scikit-learn** · **XGBoost** · **LightGBM** · **pandas** · **NumPy**

---

## 👥 Authors

**Dhana Karthikeya Ventrapragada**, Dominic, Ian, Patrick, Heesung  
UC Berkeley — Data Science Society (DSS) Decal, Spring 2026  
TA: Sameer · Tutor: Chris
