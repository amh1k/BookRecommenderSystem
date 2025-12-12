# Recommender System Project Report

## 1. Project Overview & Flow

This project implements a **Hybrid Recommendation System** for books, combining the strengths of Deep Learning (Neural Collaborative Filtering) and Information Retrieval (Content-Based Filtering).

The end-to-end flow of the system is as follows:

1.  **Data Ingestion**: Loading raw metadata (`books.csv`), user ratings (`ratings.csv`), and descriptive tags (`tags.csv`).
2.  **Data Cleaning & Preprocessing**:
    *   Dropping irrelevant columns (image URLs, language codes).
    *   Merging tag data to create richness for content analysis.
3.  **Feature Engineering ("Tag Soup")**:
    *   Aggregating Title, Authors, and Tags into a single text string per book.
    *   This "soup" represents the semantic content of the item.
4.  **Model A: Neural Collaborative Filtering (NCF)**:
    *   **Goal**: Learn user preferences from historical interaction data.
    *   **Input**: User IDs + Book IDs.
    *   **Process**: Maps IDs to embeddings, learns non-linear interactions via a neural network, and outputs a probability of "like".
5.  **Model B: Content-Based Filtering (CBF)**:
    *   **Goal**: Find books similar to what a user already likes based on metadata.
    *   **Input**: "Tag Soup" text data.
    *   **Process**: Converts text to vectors (TF-IDF) and calculates geometric similarity (Cosine).
6.  **Hybrid Fusion**:
    *   **Goal**: Combine scores to balance *personalization* (NCF) with *relevance* (CBF).
    *   **Formula**: $Score_{final} = 0.7 \times NCF + 0.3 \times CBF$.

---

## 2. Main Concepts

### A. Implicit Feedback
Unlike explicit ratings (e.g., "I give this 5 stars"), real-world data often lacks negative feedback. We infer interest:
*   **Positive (1)**: User rated the book highly (>= 4 stars).
*   **Negative (0)**: User has *not* interacted with the book (Assumed negative).

### B. Negative Sampling
To train the model to distinguish good from bad, we artificially generate negative examples by pairing users with random books they haven't seen. This prevents the model from just predicting "1" for everything.

### C. The Cold Start Problem
*   **New Users/Items**: Collaborative filtering fails when there is no history.
*   **Solution**: We use **Content-Based Filtering** to bridge this gap. A new book has no ratings, but it *does* have a title and tags, allowing us to recommend it immediately if it matches a user's known interests.

---

## 3. Recommendation Techniques Implemented

### Neural Collaborative Filtering (NeuMF)
We use the **NeuMF** architecture, which is state-of-the-art for implicit feedback. It fuses two subnetworks:
1.  **GMF (Generalized Matrix Factorization)**: A neural interpretation of standard matrix factorization (dot product). It captures linear relationships.
2.  **MLP (Multi-Layer Perceptron)**: A standard feed-forward deep network. It captures non-linear, complex relationships.

### TF-IDF (Term Frequency - Inverse Document Frequency)
A statistical measure used to evaluate how important a word is to a document in a collection.
*   **TF**: Frequency of a word in a specific book (e.g., "Magic" appears 5 times).
*   **IDF**: Inverse frequency across *all* books (e.g., "Magic" is rare, so it has high weight; "Book" is common, so it has low weight).

### Cosine Similarity
We measure similarity between two books by calculating the cosine of the angle between their TF-IDF vectors.
*   **1.0**: Perfect match (Same content).
*   **0.0**: No overlap.
