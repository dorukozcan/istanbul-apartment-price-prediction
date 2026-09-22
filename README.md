# Istanbul Apartment Price Prediction

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-R²%200.85-9ACD32)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end data mining project on **24,767 Istanbul apartment listings**: exploratory analysis, EDA-driven feature engineering, and a three-model benchmark. LightGBM reaches **R² 0.85** in 5-fold cross-validation (log price) and **21% MAPE** on a held-out test set.

---

## The two notebooks

Both run in the browser — no local setup needed:

| Notebook | What it covers | |
|---|---|---|
| [`comprehensive-eda.ipynb`](comprehensive-eda.ipynb) | Data quality assessment, price distribution and outlier analysis, district and neighborhood breakdown, size/room analysis, correlation study | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dorukozcan/istanbul-apartment-price-prediction/blob/main/comprehensive-eda.ipynb) |
| [`price-prediction.ipynb`](price-prediction.ipynb) | Cleaning informed by the EDA, feature engineering, model tournament, and interpretation of the winning model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dorukozcan/istanbul-apartment-price-prediction/blob/main/price-prediction.ipynb) |

Every preparation step in the modeling notebook traces back to a specific finding in the EDA — nothing is cleaned or engineered "just in case".

## Dataset

`data/istanbul_apartment_prices_2026.csv` — 24,767 listings across **38 districts** and **583 neighborhoods**, with 30 columns covering price, area (gross/net m²), room layout, floor, building age and type, heating and fuel, bathroom count, furnishing, complex membership, orientation and maintenance fee.

No duplicate listings; core fields (price, location, room count, area) are complete.

## Key findings from the EDA

- Prices are heavily right-skewed — the **median apartment is ₺6.75M**, so log-transformation is applied before modeling.
- **Bakırköy** has the highest median price (₺24M), followed by **Beşiktaş** (₺22M) and **Kadıköy** (₺19.75M).
- **2+1** is by far the most common layout in the Istanbul market.
- Extreme price outliers, missing values in secondary attributes, and skew were the three issues that drove the cleaning strategy.

## Modeling

After cleaning and feature engineering — 19 columns kept, 11 dropped, **9 new features engineered from EDA insights**, giving **28 features across 24,250 samples** — three models of increasing complexity were trained on the same 80/20 split. Metrics are computed on the **held-out 20% in Turkish lira**, after transforming predictions back from log price:

| Model | MAE (₺) | RMSE (₺) | R² | MAPE (%) |
|---|---:|---:|---:|---:|
| Linear Regression | 4,192,394 | 7,856,768 | 0.339 | 40.3 |
| Decision Tree (depth = 6) | 3,906,839 | 7,240,735 | 0.438 | 38.7 |
| **LightGBM** | **2,276,012** | **4,514,059** | **0.782** | **21.0** |

LightGBM was then validated with **5-fold cross-validation**:

```
Fold 1: R² = 0.8520      Fold 4: R² = 0.8535
Fold 2: R² = 0.8454      Fold 5: R² = 0.8451
Fold 3: R² = 0.8436
─────────────────────────────────────────────
Mean:   R² = 0.8479 ± 0.0040
```

The tight ±0.004 spread across folds indicates the model generalizes consistently rather than fitting one lucky split.

> **Two different R² values.** The cross-validated R² (0.85) is measured on the **log-price** target the model is trained on. The holdout R² in the table above (0.78) is measured **in Turkish lira**, where the most expensive listings dominate the squared error. Both are reported on purpose; MAPE (21%) is the easiest number to interpret.

## Caveats and next steps

- **Weak linear baseline.** District and neighbourhood are label-encoded (one integer per category), which suits trees but penalises linear regression. One-hot or target encoding would give a fairer baseline.
- **Outlier filter before the split.** Price outliers (1st–99th percentile) are removed before the train/test split, so the test set never contains extreme listings.
- **Default hyperparameters.** LightGBM runs with default settings. Tuning and SHAP-based interpretation are the natural next steps.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## Note on the notebooks

The EDA notebook mixes Matplotlib/Seaborn static charts with interactive Plotly figures. The embedded Plotly output was stripped from the committed version to keep the repository lightweight (it accounted for ~70 MB of inlined JavaScript). Static charts are preserved; re-run the notebook to regenerate the interactive figures.

## Tech stack

Python · pandas · NumPy · scikit-learn · LightGBM · Matplotlib · Seaborn · Plotly · statsmodels

## Data provenance

The dataset was compiled from public Istanbul real-estate listings for academic coursework and is included here for reproducibility. It is not affiliated with, endorsed by, or licensed from any listing platform, and should not be used commercially.

## License

The code and notebooks in this repository are released under the [MIT License](LICENSE). **The license covers the code only — not `data/istanbul_apartment_prices_2026.csv`**, which is provided for reproducibility under the terms described above.
