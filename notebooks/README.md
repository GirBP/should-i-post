# Ноутбуки — повністю ноутбучний проєкт ShouldIPost?

Проєкт переведено з `.py`-модулів у пов'язані ноутбуки. Зв'язок між ними — через `%run`:
кожен модуль-ноутбук наприкінці пакує свій API у простір імен (`config`, `features`,
`labels`, `data_prep`, `evaluate`, `inference`), а залежні ноутбуки підтягують його.

## Порядок запуску (з кореня репозиторію `shouldipost/`)

| # | Ноутбук | Роль |
|---|---|---|
| — | `eda.ipynb` | Розвідувальний аналіз (необов'язковий, але корисний першим) |
| 0 | `00_config.ipynb` | Константи, схема, leakage-allowlist |
| 1 | `01_features.ipynb` | Ознаки без стану + стіна проти витоку |
| 2 | `02_labels.ipynb` | Мітка within-creator (пороги на train) |
| 3 | `03_data_prep.ipynb` | Завантаження → очищення → поділ за часом → ознаки |
| 4 | `04_evaluate.ipynb` | Метрики, пороги, калібрування, A-vs-C |
| 5 | `05_train.ipynb` | **Навчання + персист** `models/model.joblib` (запусти перед інференсом) |
| 6 | `06_inference.ipynb` | Кандидат → рекомендація |
| 7 | `07_tests.ipynb` | Перевірки інваріантів (заміна pytest) |
| 8 | `08_demo.ipynb` | Інтерактивне демо (заміна Streamlit-застосунку) |
| — | `solution.ipynb` | Наскрізна розповідь усім пайплайном (огляд) |

### Advanced-методи (кожен незалежний, на початку `%run 03_data_prep.ipynb`)

| # | Ноутбук | Метод | Висновок (чесно) |
|---|---|---|---|
| 9 | `09_text_embeddings.ipynb` | MiniLM-ембединги підписів (+SVD); fallback TF-IDF | малий лифт (test AUC 0.52→0.54, OOF 0.64→0.65) |
| 10 | `10_xgboost_optuna.ipynb` | XGBoost + Optuna (байєсівський тюнінг) | тюнінгований XGB **не б'є** просту LogReg під дрейфом |
| 11 | `11_shap_explain.ipynb` | SHAP (beeswarm + waterfall) | узгоджується з permutation; ефекти слабкі |
| 12 | `12_conformal_prediction.ipynb` | Split-conformal (гарантія покриття) | покриття ~0.93 ціною ~62% абстенцій |

> Advanced потребують `sentence-transformers`, `xgboost`, `optuna`, `shap` (див.
> `requirements.txt`). **Запускай кожен у власному кернелі** — torch і xgboost в одному
> процесі можуть конфліктувати.

**Мінімальний шлях до результату:** `05_train.ipynb` (підтягне 0–4 через `%run`, навчить і
збереже модель) → `06_inference.ipynb` / `08_demo.ipynb`.

## Запуск
```bash
pip install jupyterlab ipywidgets     # ipywidgets — для інтерактивної форми у 08_demo
jupyter lab
# відкрий потрібний ноутбук; «Run All». Кожен сам підтягне залежності через %run.
```

> Увага: зміна формату проти ТЗ — вихідне ТЗ вимагало `.py`-модулі + pytest + Streamlit. За
> рішенням користувача проєкт переведено повністю в ноутбуки: pytest → `07_tests.ipynb`,
> Streamlit-застосунок → `08_demo.ipynb`.
