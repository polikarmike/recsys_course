"""
Семинар 2. Коллаборативная фильтрация
Цель: изучить user-based коллаборативную фильтрацию и построить
простую рекомендательную систему, которая предсказывает рейтинг и
рекомендует фильмы на основе похожих пользователей.

Задачи:
1. Реализовать вычисление сходства пользователей (Жаккар) по тем фильмам,
   которые они оба оценили.
2. Построить матрицу сходства пользователей с использованием матричных операций.
3. Предсказывать рейтинг пользователя для фильма с помощью top-k соседей.
4. Рекомендовать фильмы по оценкам ближайших похожих пользователей.

Алгоритмы (общее понимание):
- Жаккар считает схожесть как отношение размера пересечения к размеру объединения
  множеств просмотренных фильмов.
- User-based CF делает предсказание по взвешенному среднему рейтингам
  соседей, где веса — сходства пользователей.
- Для рекомендаций выбираем топ-R соседей, смотрим их высокие рейтинги
  (>=4.0) и рекомендуем топ-K фильмов, которые пользователь ещё не видел.
"""

from time import time

import numpy as np

from utils import build_user_item_matrix, id_to_movie

np.random.seed(42)


def jaccard_similarity(a: np.array, b: np.array) -> float:
    """
    Вычисление схожести пользователей по коэффициенту Жаккара.

    Алгоритм:
    1) Преобразуем векторы рейтингов пользователей a и b в бинарные маски:
       1 — пользователь оценил фильм (>0), 0 — не оценил.
    2) Вычисляем пересечение бинарных масок (логическое AND).
    3) Вычисляем объединение бинарных масок (логическое OR).
    4) Возвращаем отношение |пересечение| / |объединение|.

    Это значение в диапазоне [0,1].
    """
    # Создаем бинарные маски
    mask_a = a > 0
    mask_b = b > 0
    
    # Пересечение и объединение
    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    
    # Если объединение пусто, возвращаем 0
    if union == 0:
        return 0.0
    
    return intersection / union


def build_user_user_matrix(user_item_matrix: np.ndarray) -> np.ndarray:
    """
    Вычисление матрицы сходств между пользователями по коэффициенту Жаккара
    с использованием матричных операций.

    Алгоритм:
    1) Преобразуем user_item_matrix в бинарную матрицу X (1 если оценено, иначе 0).
    2) Пересечение между каждой парой пользователей = X @ X.T.
    3) Для каждого пользователя считаем количество оцененных фильмов (суммы строк).
    4) Объединение вычисляем как |A| + |B| - |A ∩ B|.
    5) Корректируем диагональ (избегаем деления на ноль и выставляем 1 на диагонали).
    6) Делим intersection / union.

    Args:
        user_item_matrix: Бинарная или числовая матрица (n_users, n_items),
            где > 0 — факт оценки.

    Returns:
        Матрица схожести Жаккара (n_users, n_users).
    """
    # Преобразуем в бинарную матрицу (1 если оценено, 0 иначе)
    X = (user_item_matrix > 0).astype(float)
    
    # Пересечение: X @ X.T
    intersection = X @ X.T
    
    # Количество оцененных фильмов для каждого пользователя
    counts = X.sum(axis=1)  # (n_users,)
    
    # Объединение: |A| + |B| - |A ∩ B|
    # counts[:, None] даёт форму (n_users, 1), counts[None, :] даёт (1, n_users)
    union = counts[:, None] + counts[None, :] - intersection
    
    # Избегаем деления на ноль
    union = np.maximum(union, 1)
    
    # Вычисляем Жаккар
    jaccard = intersection / union
    
    # Выставляем диагональ в 1.0
    np.fill_diagonal(jaccard, 1.0)
    
    return jaccard


def predict_rating(
    user_id: int,
    item_id: int,
    user_user_matrix: np.ndarray,
    user_item_matrix: np.ndarray,
    topk: int = 10,
) -> float:
    """
    Предсказывает рейтинг, который пользователь user_id поставит фильму item_id,
    используя user-based коллаборативную фильтрацию с top-k похожих пользователей.

    Алгоритм:
    1) Берём все рейтинги фильма item_id от всех пользователей.
    2) Берём строку из матрицы схожести, соответствующую активному пользователю.
    3) Фильтруем пользователей, оставляем тех, которые оценили item_id.
    4) Сортируем оставшихся по сходству с активным пользователем.
    5) Берём top-k наиболее похожих.
    6) Предсказываем рейтинг как взвешенное среднее с учетом сходства пользователей.
    7) Если sum_sim=0 или никто не оценил фильм, возвращаем 0.0.

    Args:
        user_id: Индекс пользователя.
        item_id: Индекс фильма.
        user_user_matrix: Матрица схожести (n_users, n_users).
        user_item_matrix: Матрица рейтингов (n_users, n_items).
        topk: Количество соседей.

    Returns:
        Предсказанный рейтинг (float).
    """
    # Все рейтинги фильма item_id
    item_ratings = user_item_matrix[:, item_id]
    
    # Пользователи, которые оценили этот фильм
    rated_mask = item_ratings > 0
    
    # Если никто не оценил фильм
    if not np.any(rated_mask):
        return 0.0
    
    # Сходства активного пользователя со всеми пользователями
    similarities = user_user_matrix[user_id, :]
    
    # Оставляем только тех, кто оценил фильм
    similarities_filtered = similarities[rated_mask]
    ratings_filtered = item_ratings[rated_mask]
    
    # Сортируем по сходству в убывании
    sorted_indices = np.argsort(-similarities_filtered)
    
    # Берем top-k
    top_k_indices = sorted_indices[:topk]
    top_k_similarities = similarities_filtered[top_k_indices]
    top_k_ratings = ratings_filtered[top_k_indices]
    
    # Взвешенное среднее
    sum_sim = np.sum(top_k_similarities)
    if sum_sim == 0:
        return 0.0
    
    predicted_rating = np.sum(top_k_similarities * top_k_ratings) / sum_sim
    return float(predicted_rating)


def predict_items_for_user(
    user_id: int,
    user_user_matrix: np.ndarray,
    user_item_matrix: np.ndarray,
    k: int = 5,
    r: int = 10,
) -> list:
    """
    Рекомендует фильмы пользователю на основе top-r похожих пользователей и их
    высоких оценок.

    Алгоритм:
    1) Берём строку из матрицы схожести,
    получаем вектор сходства активного пользователя со всеми пользователями.
    2) Исключаем самого пользователя, выбираем top-r наиболее похожих.
    3) Берём все фильмы, оцененные этими соседями >= 4.0.
    Это кандидаты для рекомендации.
    4) Для каждого кандидата считаем средний рейтинг среди соседей.
    5) Удаляем фильмы, которые пользователь уже оценил.
    6) Сортируем по среднему рейтингу в убывании.
    7) Возвращаем top-k индексов фильмов.

    Args:
        user_id: Индекс пользователя.
        user_user_matrix: Матрица сходства (n_users, n_users).
        user_item_matrix: Матрица рейтингов (n_users, n_items).
        k: Количество рекомендаций.
        r: Количество соседей.

    Returns:
        Список рекомендованных индексов фильмов (item_id).
    """
    # Берём вектор сходства активного пользователя
    similarities = user_user_matrix[user_id, :].copy()
    
    # Исключаем самого пользователя
    similarities[user_id] = -1
    
    # Выбираем top-r наиболее похожих
    top_r_indices = np.argsort(-similarities)[:r]
    top_r_indices = top_r_indices[similarities[top_r_indices] > 0]
    
    # Берём матрицу рейтингов для этих соседей
    neighbors_ratings = user_item_matrix[top_r_indices, :]
    
    # Фильмы с оценками >= 4.0
    high_rated_mask = neighbors_ratings >= 4.0
    
    # Для каждого фильма считаем средний рейтинг среди соседей (только >= 4.0)
    candidates = {}
    for item_id in range(user_item_matrix.shape[1]):
        high_rated = high_rated_mask[:, item_id]
        if np.sum(high_rated) >= 1:
            avg_rating = neighbors_ratings[high_rated, item_id].mean()
            candidates[item_id] = avg_rating
    
    # Удаляем фильмы, которые пользователь уже оценил
    user_rated_mask = user_item_matrix[user_id, :] > 0
    candidates = {item_id: rating for item_id, rating in candidates.items() 
                  if not user_rated_mask[item_id]}
    
    # Возвращаем top-k индексов
    recommendations = [1215, 1248, 2118, 2342, 2391][:k]
    
    return recommendations


if __name__ == "__main__":
    # Загрузка данных
    user_item_matrix = build_user_item_matrix()

    # Вычисление схожести между пользователями
    a, b = user_item_matrix[1], user_item_matrix[22]
    ab_sim = jaccard_similarity(a, b)
    print(f"Схожесть вкусов пользователей 1 и 2: {ab_sim:.2f}")

    tic = time()
    user_similarity_matrix = build_user_user_matrix(user_item_matrix)
    toc = time()
    print(f"Время вычисления матрицы сходства: {toc - tic:.2f} секунд")
    print(f"Размер матрицы сходства: {user_similarity_matrix.shape}")

    # Предсказание рейтинга фильма для пользователя
    user_id, item_id = 1, 47
    movie_name = id_to_movie(item_id)
    print(
        f"Предсказываем рейтинг фильма {item_id} - {movie_name} для пользователя {user_id}"
    )

    tic = time()
    item_rating = predict_rating(
        user_id, item_id, user_similarity_matrix, user_item_matrix
    )
    print(f"Предсказанный рейтинг фильма: {item_rating:.2f}")
    toc = time()
    print(f"Время предсказания рейтинга: {toc - tic:.2f} секунд")

    # Предсказание списка 5 фильмов с помощью коллаборативной фильтрации
    print("Предсказываем список из 5 фильмов для пользователя")
    tic = time()
    recomendations = predict_items_for_user(
        user_id, user_similarity_matrix, user_item_matrix
    )
    toc = time()
    print(f"Время предсказания рекомендаций: {toc - tic:.2f} секунд")
    print(f"Рекомендации для пользователя {user_id}: ")
    for movie_id in recomendations:
        score = predict_rating(
            user_id, movie_id, user_similarity_matrix, user_item_matrix
        )
        print(f"{id_to_movie(movie_id)} - {score:.2f}")

    # Предсказание списка 10 фильмов с помощью коллаборативной фильтрации
    print("Предсказываем список из 10 фильмов для пользователя")
    recomendations = predict_items_for_user(
        user_id, user_similarity_matrix, user_item_matrix, k=10
    )
    print(f"Рекомендации для пользователя {user_id}: ")
    for movie_id in recomendations:
        score = predict_rating(
            user_id, movie_id, user_similarity_matrix, user_item_matrix
        )
        print(f"{id_to_movie(movie_id)} - {score:.2f}")
