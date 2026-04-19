"""
Семинар 3. Контентная фильтрация
Цель: Разработать методы контентной фильтрации по пользователям и по фильмам.
В качестве контента используем описание жанров для каждого фильма из movies.csv.
Для векторизации жанров используем CountVectorizer с разделителем "|".
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import build_user_item_matrix, id_to_movie, load_data, print_user_rated_items


class ContentRecommender:
    """
    Класс для построения рекомендаций на основе контента - описания жанров.
    Матрица эмбеддингов размером (max_movie_id+1, n_genres), где строки
    соответствуют movieId, а столбцы — one-hot кодированию жанров.
    Матрица строится при инициализации экземпляра класса.
    """

    def __init__(self):
        self.embeddings = None
        self.ui_matrix = build_user_item_matrix()
        self._build_embeddings()

    def _build_embeddings(self):
        _, movies_df = load_data()
        self.movies_df = movies_df.copy()
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        vectorizer = CountVectorizer(tokenizer=lambda s: s.split("|"), lowercase=False)
        ###########################################################################
        # TODO: Строим матрицу эмбеддингов для фильмов и сохраняем в self.embeddings                       
        embeddings_sparse = vectorizer.fit_transform(self.movies_df["genres"])
        max_id = self.movies_df["movieId"].max()
        n_genres = len(vectorizer.get_feature_names_out())
        self.embeddings = np.zeros((max_id + 1, n_genres))
        for idx, row in self.movies_df.iterrows():
            movie_id = row["movieId"]
            vec = embeddings_sparse[idx].toarray()[0]
            self.embeddings[movie_id] = vec
        ###########################################################################
        

    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Предсказывает рейтинг user_id для item_id на основе контентной фильтрации.

        Алгоритм:
        1) Берём вектор целевого фильма: target_vec.
        2) Находим все фильмы, оцененные пользователем.
        3) Считаем косинусное сходство target_vec с векторами оцененных фильмов.
        4) Отбираем топ-k похожих оцененных фильмов (k-параметр).
        5) Предсказываем рейтинг как взвешенное среднее оценок по сходствам.
        6) Если не удаётся предсказать (нет оценок или нулевые векторы), возвращаем 0.0.
        7) Клипируем результат в [0.0, 5.0].

        Args:
            user_id: индекс пользователя
            item_id: индекс фильма
            k: сколько наиболее похожих оцененных фильмов использовать

        Returns:
            float: предсказанный рейтинг
        """
        target_vec = self.embeddings[item_id]
        if np.sum(target_vec) == 0:
            return 0.0
        rated_items = np.where(self.ui_matrix[user_id] > 0)[0]
        if len(rated_items) == 0:
            return 0.0
        similarities = []
        ratings = []
        for item in rated_items:
            item_vec = self.embeddings[item]
            if np.sum(item_vec) == 0:
                continue
            sim = cosine_similarity([target_vec], [item_vec])[0][0]
            if sim > 0:
                similarities.append(sim)
                ratings.append(self.ui_matrix[user_id][item])
        if not similarities:
            return 0.0
        # топ-k
        if len(similarities) > k:
            sorted_indices = np.argsort(similarities)[::-1][:k]
            similarities = [similarities[i] for i in sorted_indices]
            ratings = [ratings[i] for i in sorted_indices]
        pred = np.average(ratings, weights=similarities)
        return np.clip(pred, 0.0, 5.0)

    def predict_items_for_user(
        self, user_id: int, k: int = 5, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы пользователю user_id на основе контента фильма.

        Алгоритм:
        1) Берем все фильмы, которые оценил пользователь.
        3) Строим профиль пользователя как взвешенное среднее жанров оцененных фильмов.
        4) Для всех фильмов, которые пользователь не оценил, считаем сходство с профилем.
        5) Сортируем по убыванию сходства и возвращаем top-n.
        """
        rated_items = np.where(self.ui_matrix[user_id] > 0)[0]
        if len(rated_items) == 0:
            return []
        ratings = self.ui_matrix[user_id][rated_items]
        vectors = self.embeddings[rated_items]
        user_profile = np.average(vectors, axis=0, weights=ratings)
        if np.sum(user_profile) == 0:
            return []
        movie_ids = self.movies_df["movieId"].tolist()
        unrated_items = [i for i in movie_ids if self.ui_matrix[user_id][i] == 0]
        similarities = []
        for item in unrated_items:
            item_vec = self.embeddings[item]
            sim = cosine_similarity([user_profile], [item_vec])[0][0]
            similarities.append((item, sim))
        # sort by sim descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        recommendations = [item for item, sim in similarities[:n_recommendations]]
        return recommendations


# Пример использования для дебага:
if __name__ == "__main__":
    user_id = 10
    item_id = 2
    k = 5
    content_recommender = ContentRecommender()
    print_user_rated_items(user_id, content_recommender.ui_matrix)

    pred_rating = content_recommender.predict_rating(user_id, item_id, k)
    print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating:.2f}")

    recommendations = content_recommender.predict_items_for_user(
        user_id, k=5, n_recommendations=10
    )
    for rec in recommendations:
        print(f"Recommended movie ID: {rec}, Title: {id_to_movie(rec)}")
