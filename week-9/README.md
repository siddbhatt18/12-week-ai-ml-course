Week 9 deepens your **capstone**: stronger models, better features, more robust evaluation, and clearer understanding of *why* your model behaves as it does.

Assume ~1–2 hours/day.  
You’ll keep working inside your `capstone/` project from Week 8.

I’ll keep saying “classification”; if your capstone is regression, I’ll note the adaptations (swap accuracy/ROC-AUC for RMSE/R², etc.).

Each day: **Goal → Core study → Core practice → Stretch → Reflection**.

---

## Day 1 – Re‑orient, Set Advanced Goals, and Plan Experiments

**Goal:** Revisit Week 8’s results, define **specific improvement goals**, and plan a small set of **experiments** to run this week.

### 1. Core study (45–60 min)

Notebook: `notebooks/week9_day1_plan_and_status.ipynb`.

1. **Re‑load your Week 8 best model & results**

- Open your Week 8 summary notebook and/or load from disk:

```python
import joblib

best_model = joblib.load("../models/best_model.joblib")  # if you saved it
```

If you didn’t save it, just re‑create the pipeline and fit it once on `X_train, y_train` again.

2. **Re‑state current performance**

- Print or restate:

  - Baseline metric(s) (naive + rule-based).
  - Best model’s test performance.

Classification (example):

```python
from sklearn.metrics import accuracy_score, roc_auc_score

y_test_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_test_pred))

if hasattr(best_model, "predict_proba"):
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    print("ROC AUC:", roc_auc_score(y_test, y_test_proba))
```

Regression:

```python
from sklearn.metrics import mean_squared_error, r2_score

y_test_pred = best_model.predict(X_test)
rmse = mean_squared_error(y_test, y_test_pred, squared=False)
r2 = r2_score(y_test, y_test_pred)
rmse, r2
```

3. **Set concrete improvement goals (markdown)**

Examples:

- “Improve ROC-AUC from 0.78 → 0.82.”
- “Reduce RMSE by at least 10% vs Week 8.”
- “Understand and fix worst‑performing subgroup (e.g., users with short tenure).”

Write 2–3 **specific, measurable** goals.

4. **Plan experiments**

In markdown, define 4–6 experiments you might run this week, grouped by:

- **Feature engineering**:
  - New ratio features.
  - Age / tenure buckets.
  - Flags for special conditions.
- **Models**:
  - Gradient boosting (HistGradientBoosting, XGBoost if you want).
  - Better tuned RandomForest / regularized linear model.
- **Evaluation**:
  - K‑fold CV.
  - Segment-wise performance.
  - Calibration (if classification).

Example table (markdown):

| ID | Category           | Description                                  |
|----|--------------------|----------------------------------------------|
| E1 | Features           | Add age buckets + interaction `X * Y`       |
| E2 | Features           | Handle rare categories with “Other” bucket  |
| M1 | Model              | Try HistGradientBoosting with pipeline      |
| M2 | Model              | Retune RandomForest with smaller max_depth  |
| V1 | Evaluation         | 5‑fold CV for final candidate models        |

### 2. Core practice (30–45 min)

1. **Create an experiment log**

Create `notebooks/experiment_log.csv` or a DataFrame in this notebook with columns:

- `exp_id`
- `description`
- `model_type`
- `features_version`
- `cv_metric`
- `test_metric`
- `notes`

You’ll fill it gradually this week.

2. **Record current baseline as first row**

Example row:

| exp_id | description                 | model_type       | features_version | cv_metric  | test_metric | notes                  |
|--------|-----------------------------|------------------|------------------|-----------:|-----------:|------------------------|
| BASE   | Week8 best_model baseline   | RandomForest     | v0_basic         | 0.80 (cv)  | 0.78 (test)| from Week 8 tuning     |

### 3. Stretch (optional)

- If you don’t have clear `features_version` names yet, define them now:
  - `v0_basic` = raw feature set.
  - `v1_fe1` = + engineered features (Day 2).
  - etc.

### 4. Reflection

- What’s your **main success criterion** for the capstone now?
- Which lever do you *suspect* will help most: features, model type, or evaluation strategy? Why?

---

## Day 2 – Systematic Numeric Feature Engineering

**Goal:** Add **meaningful numeric features** (buckets, ratios, interactions) and test if they improve performance.

### 1. Core study (45–60 min)

Notebook: `notebooks/week9_day2_numeric_features.ipynb`.

1. **Brainstorm numeric feature ideas (markdown)**

Examples (adapt to your problem):

- Age / tenure / amount buckets (bins).
- Ratios: e.g. `amount / income`, `spend / visits`.
- Aggregations or differences: `max - min`, `current - previous`.
- Log transforms for skewed variables (e.g. `log1p(amount)`).

List at least **3–5** candidates, and for each:
- Why might it help the model?
- Any risk of leakage (does it use future info or target info?).

2. **Implement a few new features in pandas**

In code, working on a copy that you use consistently for train/test:

```python
df = pd.read_csv("../data/your_file.csv")
df_model = df.dropna(subset=[target]).copy()

# Example: age buckets
if "age" in df_model.columns:
    df_model["age_bucket"] = pd.cut(
        df_model["age"],
        bins=[0, 25, 40, 60, 120],
        labels=["young", "young_adult", "middle_aged", "older"]
    )

# Example: log transform for a skewed amount
if "amount" in df_model.columns:
    df_model["log_amount"] = np.log1p(df_model["amount"])

# Example: ratio
if {"spend", "visits"}.issubset(df_model.columns):
    df_model["spend_per_visit"] = df_model["spend"] / (df_model["visits"] + 1)
```

3. **Recreate X, y with new features**

```python
X = df_model.drop(columns=[target])
y = df_model[target]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if classification else None
)
```

Update `numeric_features` and `categorical_features` based on `X_train.dtypes`.

4. **Rebuild preprocessing + champion model pipeline**

Copy your best pipeline definition from Week 8 / Day 4, but use updated feature lists.

### 2. Core practice (30–45 min)

1. **Compare old vs new features**

- Model A: pipeline with **old** features (v0_basic).
- Model B: pipeline with **new** numeric features (v1_fe_numeric).

You can simulate Model A by **dropping** newly added columns from `X_train` and `X_test`.

Record test metric for both, and add rows to `experiment_log`.

2. **Interpret results**

In markdown:

- Did numeric feature engineering:
  - Help clearly?
  - Do almost nothing?
  - Make things worse (overfitting, instability)?

### 3. Stretch (optional)

- Try using `FunctionTransformer` to encapsulate your numeric feature creation and plug it into a top-level Pipeline:
  - More advanced, but helps keep all transformations inside scikit‑learn.

### 4. Reflection

- Which new feature had the clearest logical justification and impact?
- If numerical FEs didn’t help much, what might that say about your original features?

---

## Day 3 – Categorical Feature Handling & Rare Categories

**Goal:** Improve handling of **categorical features**, especially rare categories, and see if better encoding helps.

### 1. Core study (45–60 min)

Notebook: `notebooks/week9_day3_categorical_features.ipynb`.

1. **Inspect category cardinality & rarity**

From `X_train`:

```python
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

for col in cat_cols:
    vc = X_train[col].value_counts()
    print(col, "n_unique:", vc.shape[0])
    print(vc.head(10))
    print("---")
```

Identify:

- Columns with **many unique categories** (high cardinality).
- Columns with **many rare categories** (e.g. values with < 1% frequency).

2. **Plan category treatment (markdown)**

For each problematic column, decide:

- Merge rare categories into `"Other"`.
- Possibly drop columns that are mostly unique IDs.
- Keep only top N categories and group rest into `"Other"`.

3. **Implement rare-category bucketing**

Example:

```python
def bucket_rare_categories(series, min_freq=0.01):
    vc = series.value_counts(normalize=True)
    rare = vc[vc < min_freq].index
    return series.where(~series.isin(rare), other="Other")

df_model = df_model.copy()
for col in cat_cols:
    df_model[col] = bucket_rare_categories(df_model[col], min_freq=0.01)
```

Rebuild `X_train, X_test, y_train, y_test` from this updated `df_model`.

4. **Rebuild ColumnTransformer and pipeline**

Same as Day 2, but now on cleaned categories.

### 2. Core practice (30–45 min)

1. **Compare before vs after category cleaning**

- Model C: pipeline with original categoricals (no bucketing).
- Model D: pipeline with rare categories bucketed.

Again, record test metrics and add to `experiment_log`.

2. **Simple frequency encoding (optional within practice)**

For a **single high-cardinality column**, try:

```python
col = "some_cat_col"
freq = X_train[col].value_counts(normalize=True)

X_train[col + "_freq"] = X_train[col].map(freq)
X_test[col + "_freq"] = X_test[col].map(freq).fillna(0)
```

And then:
- Drop original `col`, keep `col_freq`.
- Add to numeric_features, run a quick model.

Be careful: this simple frequency encoding is relatively safe; **true target encoding** (using target means) must be done with CV to avoid leakage, so just read about it for now if curious.

### 3. Stretch (optional)

- Read about **target encoding** (e.g. “mean encoding”), understand conceptually:
  - How it uses target statistics.
  - Why you must avoid fitting it on the full train set without cross-validation (leakage).
- If you feel adventurous, implement **target encoding with CV** for one column (but only if you’re comfortable).

### 4. Reflection

- Did better categorical handling improve performance or stability?
- Are there any categorical columns you now think should be **dropped** entirely?

---

## Day 4 – Gradient Boosting Models (GBMs) with Pipelines

**Goal:** Add a **gradient boosting** model (e.g. HistGradientBoosting or XGBoost) to your model zoo and compare it against your current champion.

### 1. Core study (45–60 min)

Notebook: `notebooks/week9_day4_gradient_boosting.ipynb`.

We’ll use **scikit‑learn’s HistGradientBoosting** (faster and built‑in).  
(If you know XGBoost/LightGBM, you can try them as stretch.)

1. **Set up preprocessing**

Re-use your best `preprocessor` from earlier this week (numeric + categorical pipelines, rare-category handling, etc.).

2. **Define GBM model**

Classification:

```python
from sklearn.ensemble import HistGradientBoostingClassifier

gb_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", HistGradientBoostingClassifier(
        max_depth=None,
        learning_rate=0.1,
        max_iter=200,
        random_state=42
    ))
])
```

Regression:

```python
from sklearn.ensemble import HistGradientBoostingRegressor

gb_reg = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", HistGradientBoostingRegressor(
        max_depth=None,
        learning_rate=0.1,
        max_iter=200,
        random_state=42
    ))
])
```

3. **Fit & evaluate**

Use your usual metric:

```python
gb_clf.fit(X_train, y_train)
y_pred_gb = gb_clf.predict(X_test)

# classification:
from sklearn.metrics import accuracy_score, roc_auc_score

acc_gb = accuracy_score(y_test, y_pred_gb)
print("GB test accuracy:", acc_gb)

if hasattr(gb_clf, "predict_proba"):
    y_proba_gb = gb_clf.predict_proba(X_test)[:, 1]
    print("GB ROC AUC:", roc_auc_score(y_test, y_proba_gb))
```

### 2. Core practice (30–45 min)

1. **Compare RF vs GB**

Create a small comparison table:

- RandomForest (tuned from Week 8/9).
- GradientBoosting (this new model).

Report test metrics for both.

2. **Light tuning of GB**

Try a small manual grid:

- Vary `learning_rate` and `max_iter`.
- Optionally `max_depth`.

Record metrics for each variant (even ~3–4 combos is fine).

Example:

```python
configs = [
    {"learning_rate": 0.1, "max_iter": 200},
    {"learning_rate": 0.05, "max_iter": 400},
    {"learning_rate": 0.2, "max_iter": 100},
]

rows = []
for cfg in configs:
    gb = HistGradientBoostingClassifier(
        learning_rate=cfg["learning_rate"],
        max_iter=cfg["max_iter"],
        random_state=42
    )
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", gb)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rows.append({**cfg, "test_acc": acc})

import pandas as pd
pd.DataFrame(rows)
```

### 3. Stretch (optional)

- Use `cross_val_score` on the best GB pipeline to compare it more fairly to your RF pipeline.
- If comfortable installing external libs:
  - Try XGBoost or LightGBM in a similar pipeline setup.

### 4. Reflection

- Does GBM outperform your RandomForest and logistic/linear baselines?
- Given performance vs complexity, which model would you now name as **current champion**?

---

## Day 5 – Robust Evaluation: Cross-Validation and Model Stability

**Goal:** Move beyond one train/test split: use **k‑fold CV** to estimate performance and check **stability** across folds.

### 1. Core study (45–60 min)

Notebook: `notebooks/week9_day5_robust_evaluation.ipynb`.

1. **Set up candidate models**

Pick 2–3 finalists, e.g.:

- `log_clf` / `linear_reg`
- `rf_clf` / `rf_reg`
- `gb_clf` / `gb_reg`

2. **Use cross_val_score**

Classification:

```python
from sklearn.model_selection import cross_val_score

models = {
    "Logistic/Linear": log_clf,
    "RandomForest": rf_clf,
    "HistGB": gb_clf
}

rows = []
for name, model in models.items():
    scores = cross_val_score(
        model, X, y, cv=5,
        scoring="accuracy"  # or "roc_auc" for primary
    )
    rows.append({
        "model": name,
        "cv_mean": scores.mean(),
        "cv_std": scores.std()
    })

import pandas as pd
cv_results = pd.DataFrame(rows)
cv_results
```

Regression: use `scoring="neg_root_mean_squared_error"` and negate mean.

3. **Interpret CV results**

In markdown:

- Which model has highest **mean** CV score?
- Which has lowest **std** (most stable)?
- Do CV rankings match test‑set rankings from earlier this week?

### 2. Core practice (30–45 min)

1. **Try different CV schemes**

- `cv=3` vs `cv=5` vs `cv=10` for the champion model.
- If time series: consider `TimeSeriesSplit` instead of regular K‑fold.

2. **Add CV results to experiment log**

For each model:

- Add row with `cv_mean`, `cv_std`, and best test metric.

### 3. Stretch (optional)

- For classification, compare **accuracy** vs **ROC-AUC** as scoring:
  - Does the “best” model change?
- For regression, compare RMSE vs MAE (using custom scoring).

### 4. Reflection

- Do you trust your model’s performance estimate more now than at the end of Week 8? Why?
- Which model gives the **best trade‑off** between performance and stability?

---

## Day 6 – Interpretability: Feature Importance, Partial Dependence, (Optional SHAP)

**Goal:** Understand **why** your best model makes its predictions: global importance and how key features affect the prediction.

### 1. Core study (45–60 min)

Notebook: `notebooks/week9_day6_interpretability.ipynb`.

Assume your champion is a tree‑based ensemble (RF or HistGB). If it’s a linear model, adapt by looking at coefficients instead.

1. **Global feature importance (tree-based)**

Fit `best_model` on full training data (if not already).

Get processed feature names:

```python
import numpy as np
import pandas as pd

pre = best_model.named_steps["preprocess"]
model = best_model.named_steps["model"]

numeric_features = [...]          # from earlier
categorical_features = [...]

ohe = pre.named_transformers_["cat"].named_steps["onehot"]
ohe_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numeric_features, ohe_names])

importances = pd.Series(model.feature_importances_, index=all_feature_names)
importances.sort_values(ascending=False).head(20)
```

Plot:

```python
importances.sort_values(ascending=False).head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 15 feature importances")
plt.show()
```

2. **Partial dependence plots (PDPs)**

Pick 1–2 key features from importances and plot partial dependence.

```python
from sklearn.inspection import PartialDependenceDisplay

features_to_plot = ["age", "log_amount"]  # adjust to your top features

PartialDependenceDisplay.from_estimator(
    best_model,
    X_train,
    features=features_to_plot,
    kind="average"
)
plt.show()
```

Interpret:

- As feature X increases, what tends to happen to the prediction?

### 2. Core practice (30–45 min)

1. **Choose 2–3 important features and analyze**

For each:

- Look at their partial dependence.
- Cross‑check with EDA:
  - Did you already suspect this relationship?
  - Does the model pick up something new?

2. **Local interpretation (basic)**

Pick a few individual test examples:

```python
sample = X_test.sample(3, random_state=42)
sample
best_model.predict(sample)
```

For each:

- Look at feature values.
- Using feature importance & PDP intuition, explain (qualitatively) why the model might predict that output.

### 3. Stretch (optional)

- Try **permutation importance** from `sklearn.inspection.permutation_importance`:
  - It can be more reliable than built‑in importance for some models.
- If comfortable, install `shap` and try a basic `shap.TreeExplainer` visualization for a handful of samples.

### 4. Reflection

- Do model explanations match your domain intuition, or are there surprises?
- Which feature relationships (from PDPs) would you show to a stakeholder to help them trust or question the model?

---

## Day 7 – Week 9 Milestone: Consolidated Model + Expanded Report

**Goal:** Choose your **current final model**, update your capstone report with Week 9 improvements, and decide what’s left for Weeks 10–12.

### 1. Core study (45–60 min)

Notebook: `notebooks/week9_milestone_summary.ipynb`.

1. **Rebuild final chosen model**

- Using best:
  - Feature set (including good numeric/categorical engineering).
  - Preprocessor.
  - Model type (RF or GBM or linear).
  - Hyperparameters (from tuning).

Keep the code clean and minimal: load data, split, build pipeline, fit, evaluate.

2. **Print key evaluation metrics**

Classification:

- Accuracy
- ROC-AUC (if applicable)
- Precision, recall, F1 (for overall and/or positive class)

Regression:

- RMSE
- MAE
- R²

Compare in one markdown table:

| Model           | Features    | CV metric | Test metric | Notes          |
|----------------|------------|-----------|------------|----------------|
| Week8 best     | v0_basic   | …         | …          | baseline       |
| Week9 final    | v1_fe_cat+num + GBM | … | … | +X% improvement |

3. **Add 1–2 interpretability plots**

- Feature importance bar plot.
- 1–2 PDPs.

### 2. Core practice (30–45 min)

1. **Update README or main project report**

Add/extend sections:

- **Modeling v1 (Week 8)**:
  - Brief summary: feature set, model, metrics.
- **Modeling v2 (Week 9)**:
  - New feature engineering (numeric + categorical).
  - New models (GBM).
  - Updated metrics.
- **Error analysis & interpretability**:
  - Key takeaways about which segments are hard, which features matter.

2. **List open questions / next steps**

In markdown, list things you *might* tackle in Weeks 10–12:

- More sophisticated feature engineering (time‑aware, domain-specific).
- Trying other models (XGBoost/LightGBM/CatBoost, neural nets if appropriate).
- Model calibration (for probabilistic predictions).
- Deployment (Flask/FastAPI/Streamlit app).
- Monitoring & retraining strategy.

Rank them by:

- Impact (how helpful).
- Effort (how hard).

### 3. Stretch (optional)

- Save your final model pipeline (if not already) and note the **exact version** of libraries used (see `pip freeze` or `conda list`).
- If using Git, make a `week9_milestone` tag or branch.

### 4. Reflection

- Compared to the end of Week 8, how much better is your **best model** (numerically and in understanding)?
- If you had to present this to someone in 5 minutes, what 3 slides/plots/tables would you show?

---

If you tell me whether your capstone is **classification or regression**, and your domain (e.g., health, finance, churn, etc.), I can tailor **Week 10** to focus on the most relevant next step (e.g., calibration & thresholds for classification, or time‑aware validation for regression, or a simple deployment path).
