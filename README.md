# Istanbul Apartment Price Prediction

An end-to-end data mining project on **24,767 Istanbul apartment listings**: exploratory analysis, EDA-driven feature engineering, and a three-model benchmark that lands at a cross-validated **R² of 0.85**.

---

## The two notebooks

| Notebook | What it covers |
|---|---|
| [`comprehensive-eda.ipynb`](comprehensive-eda.ipynb) | Data quality assessment, price distribution and outlier analysis, district and neighborhood breakdown, size/room analysis, correlation study |
| [`price-prediction.ipynb`](price-prediction.ipynb) | Cleaning informed by the EDA, feature engineering, model tournament, and interpretation of the winning model |

Every preparation step in the modeling notebook traces back to a specific finding in the EDA — nothing is cleaned or engineered "just in case".

## Dataset

`data/istanbul_apartment_prices_2026.csv` — 24,767 listings across **38 districts** and **583 neighborhoods**, with 26 columns covering price, area (gross/net m²), room layout, floor, building age and type, heating and fuel, bathroom count, furnishing, complex membership, orientation and maintenance fee.

No duplicate listings; core fields (price, location, room count, area) are complete.

## Key findings from the EDA

- Prices are heavily right-skewed — the **median apartment is ₺6.75M**, so log-transformation is applied before modeling.
- **Bakırköy** has the highest median price (₺24M), followed by **Beşiktaş** (₺22M) and **Kadıköy** (₺19.75M).
- **2+1** is by far the most common layout in the Istanbul market.
- Extreme price outliers, missing values in secondary attributes, and skew were the three issues that drove the cleaning strategy.

## Modeling

After cleaning and feature engineering — 19 columns kept, 11 dropped, **9 new features engineered from EDA insights**, giving **28 features across 24,250 samples** — three models of increasing complexity were trained on the same split:

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
