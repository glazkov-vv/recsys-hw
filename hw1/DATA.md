# Рекомендательные системы: сравнение пяти моделей на Amazon Reviews 2023 (Video Games)

Студент: Глазков Владислав Викторович

Датасет: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) — McAuley Lab, UC San Diego
(Hou et al., Bridging Language and Items for Retrieval and Recommendation, 2024), категория Video_Games:

* рейтинги (официальный 5-core benchmark, `user_id, parent_asin, rating, timestamp`):
  <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz>
* метаданные товаров (название, категории/платформа, store, цена, дата появления, …):
  <https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz>

Ноутбук скачивает оба файла в `hw1/data/` сам (`models/data.py::download`). В репозиторий данные не загружаются.

## Задача

Персональные top-10 рекомендации товаров (игры, консоли, аксессуары). Релевантно — товар, который пользователь
в будущем купил и оценил на ≥ 4. Глобальное временное разбиение: train < 2021-01-01 ≤ validation < 2022-01-01 ≤ test.
Основная метрика — NDCG@10, дополнительно Recall@10, HitRate@10, MAP@10, Coverage@10, Novelty@10.

## Модели (NumPy / SciPy / PyTorch)

| Модель | Файл |
|---|---|
| Item-based коллаборативная фильтрация (соседи по товарам: асимметричный косинус + shrinkage, top-k) | [`models/item_knn.py`](models/item_knn.py) |
| iALS — implicit ALS (точный решатель и сопряжённые градиенты) | [`models/ials.py`](models/ials.py) |
| EASE (замкнутое решение через Холецкого) | [`models/ease.py`](models/ease.py) |
| Factorization Machine с метапризнаками, BPR-обучение | [`models/fm.py`](models/fm.py) |
| iALS с кастомным таргетом «удовлетворённость × свежесть» | [`models/ials_custom_target.py`](models/ials_custom_target.py) |
| Общие функции: метрики, top-K, оценка | [`models/utils.py`](models/utils.py) |
| Загрузка, подвыборка, разбиение, признаки FM | [`models/data.py`](models/data.py) |

## Запуск

```bash
cd hw1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb   # или Run All в Jupyter
```

Все зёрна фиксированы (`SEED = 42`), параметры эксперимента — в ячейке `CONFIG` ноутбука.

## Результаты

Test (2022-01 … 2023-09), K = 10; подробности, абляции и выводы — в `experiments.ipynb`.

| Модель | val NDCG@10 | test NDCG@10 | test Recall@10 | HitRate@10 | MAP@10 | Coverage@10 | Выбранные гиперпараметры |
|---|---|---|---|---|---|---|---|
| FM (метаданные, BPR) | 0.0260 | 0.0249 | 0.0449 | 0.0744 | 0.0151 | 0.085 | factors=128, lr=3.0e-3, reg=6.8e-4, 10 эпох |
| iALS, кастомный таргет | 0.0248 | 0.0247 | 0.0381 | 0.0643 | 0.0166 | 0.146 | factors=64, λ=1.54, α=15.0, β=0.40, δ=0.06, h=208 дн. |
| Popular (180 дней), baseline | 0.0268 | 0.0237 | 0.0499 | 0.0844 | 0.0121 | 0.002 | — |
| iALS | 0.0159 | 0.0152 | 0.0239 | 0.0440 | 0.0095 | 0.240 | factors=256, λ=585, α=130 |
| EASE | 0.0161 | 0.0140 | 0.0228 | 0.0418 | 0.0085 | 0.469 | λ=100 |
| Item-based CF (ItemKNN) | 0.0164 | 0.0132 | 0.0211 | 0.0395 | 0.0081 | 0.375 | k=874, h=32.8, a=0.77, user weighting |
| Popular (за всё время) | 0.0095 | 0.0070 | 0.0120 | 0.0221 | 0.0044 | 0.002 | — |

Кратко: лучшие — FM с метапризнаками и iALS с кастомным таргетом (разница между ними незначима); обе примерно в 1.6–1.9 раза лучше iALS / EASE / item-based CF. Ключевой фактор — временной дрейф: компонента свежести в таргете даёт +63 % NDCG@10 к базовому iALS, а метапризнаки (включая возраст товара) удваивают качество FM.

