# Soil & Content Impact Prediction

Two ML pipelines built as part of applied data science work — one predicting soil organic carbon content from spectroscopic data, the other predicting the impact score of news articles from editorial metadata.

---

## Project 1 — Soil Organic Carbon (SOC) Prediction

**The problem:** Predict the percentage of organic carbon in soil samples taken from three agricultural fields, using near-infrared (NIR) spectroscopy readings as features.

**The data:** 1,575 soil measurements × 478 NIR spectral features, labelled with GPS coordinates and field location (field_A, field_B, field_C).

**The pipeline:**
1. Data cleaning — remove NaN/Inf values, filter GPS zeros, fix malformed entries in the target variable
2. Outlier removal — SOC distribution analysed against normal distribution; one outlier (SOC > 9%) removed
3. Dimensionality reduction — PCA applied to 478 features → 170 principal components (needed due to severe multicollinearity detected via VIF)
4. Regression — Linear Regression trained on PCA components; SVR tested for comparison
5. Evaluation — R², MAE, MSE, explained variance score; model tested separately on each of the three fields to check generalization

**Results:** Explained variance score of ~0.71 on held-out test data.

**Stack:** `pandas` `numpy` `scikit-learn` `scipy` `matplotlib`

```
initial_analysis_and_filtering.py   # EDA, cleaning, outlier removal, field visualization
prediction_model.py                 # PCA, regression, per-field evaluation
func_common.py                      # Shared utilities: scaling, PCA, VIF, metrics
```

---

## Project 2 — Content Impact (Sentiment) Prediction

**The problem:** Predict the impact score of news articles published on BILD, using editorial and structural metadata as features — article category, title type, sentiment, emotion, clickbait score, department, and more.

**The pipeline:**
1. Deduplication — remove duplicate `content_id` / `page_id` pairs
2. Feature engineering — label encoding of 14 categorical columns (department, category, title emotion, sentiment, clickbait flag, etc.)
3. Dimensionality reduction — Pearson correlation heatmap + threshold-based feature removal (drop features with correlation ≥ 0.9); VIF computed for multicollinearity check
4. Classification — three algorithms compared: Random Forest, SVM (polynomial kernel), Logistic Regression (MaxEnt with L2 regularisation)
5. Evaluation — confusion matrix, F1 score, accuracy per classifier

**Stack:** `pandas` `numpy` `scikit-learn` `statsmodels` `seaborn`

```
content_impact_prediction.py    # Main pipeline: preprocessing → classification
classification_functions.py     # MaxEnt, Random Forest, SVM classifiers + confusion matrix
data_prep_functions.py          # Deduplication, column removal, heatmap, Pearson reduction, VIF
```

---

## Why these two projects are in one repo

Both pipelines share the same structural pattern: feature engineering → dimensionality reduction → model comparison → evaluation. The soil project is a regression problem; the content impact project is a classification problem. Together they demonstrate the same workflow applied to two very different domains.

---

## License

MIT
