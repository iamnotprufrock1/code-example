The code is a semi-supervised algorithm that generated embeddings using a supervised deep learning model and then grouped clusters of risky sign-up sessions based on these embeddings.

We first train an auto encoder with contrastive loss, extract weights in hidden layers, then train a clustering model on risky sessions data only. We apply the embedding process and clustering model to predict on real distribution sign-up data. Selected risky sessions are auto enforced.
