Week 8 is where your **capstone project really starts**.  
You’ll take one dataset/problem you care about and push it **beyond** the Week‑7 project:

- Deeper EDA & problem framing  
- Stronger baselines  
- Solid pipelines & model comparison  
- More systematic tuning  
- Clear experiment tracking & documentation

Assume ~1–2 hours/day.

I’ll call your dataset **CAPSTONE_DATA** and your target **`target`**.  
Where I say “classification”, adapt for regression by swapping in RMSE/R² instead of accuracy/ROC, etc.

Each day: **Goal → Core study → Core practice → Stretch → Reflection**.

---

## Day 1 – Finalize Capstone Problem, Data, and Evaluation Plan

**Goal:** Lock in your dataset, clarify the problem and success criteria, and set up a clean project structure.

### 1. Core study (45–60 min)

Create folder structure, e.g.:

```text
capstone/
  data/
  notebooks/
  src/         # optional, for later
  models/
  README.md
```

Notebook: `notebooks/day1_problem_and_data.ipynb`.

1. **Load data & skim**

- Put your raw CSV(s) into `capstone/data/`.
- In the notebook:

  ```python
  import pandas as pd

  df = pd.read_csv("../data/your_file.csv")
  df.head()
  df.shape
  df.info()
  df.describe(include="all").T.head(20)
  ```

2. **Problem statement (markdown)**

Write clearly:

- What does **one row** represent?
- What is the **target** you want to predict?
- Is this **classification or regression**?
- Who could use this model and **for what decision**?

3. **Metric choice (markdown)**

Decide and justify:

- Classification:
  - Primary: accuracy **or** ROC-AUC.
  - Secondary: precision/recall/F1, especially if imbalance.
- Regression:
  - Primary: RMSE.
  - Secondary: MAE, R².

Explain:
- Why is this metric appropriate?
- Which kind of error is costlier (false positive vs false negative; big overestimates vs underestimates)?

4. **Initial data sanity checks**

- Check for:
  - Duplicate rows (`df.duplicated().sum()`).
  - Columns with only one value.
  - Obvious ID-only columns.

### 2. Core practice (30–45 min)

1. **Column inventory**

Create a small table (markdown or a pandas DataFrame) with:

- Column name
- Inferred type (numeric / categorical / date / ID / text)
- Short description (from source docs or your guess)
- Initial decision: `keep`, `drop`, or `unsure`

2. **Target inspection**

Classification:

```python
df[target].value_counts(dropna=False)
df[target].value_counts(normalize=True)
```

Regression:

```python
df[target].describe()
df[target].hist(bins=30)
```

Write 3–4 bullet points about:

- Balance/imbalance (classification).
- Range, skewness, outliers (regression).

### 3. Stretch (optional)

- Add a short **“Risk & Ethics”** section in markdown:
  - Any sensitive attributes (e.g., gender, race, location)?
  - How could an incorrect model hurt users?
  - How might you **mitigate** that (e.g., not using certain features, monitoring subgroup performance)?

### 4. Reflection

- Can you explain your capstone problem in **two plain-language sentences**?
- On a 1–10 scale, how well do you understand your target and metric?  
  What would move that score up by 2 points?

---

## Day 2 – Deeper EDA and Robust Train/Test Split

**Goal:** Understand your data more deeply and design an appropriate **splitting strategy** (random vs time-based, stratified, etc.).

### 1. Core study (45–60 min)

Notebook: `notebooks/day2_eda_and_split.ipynb`.

1. **Reload data**

```python
import pandas as pd

df = pd.read_csv("../data/your_file.csv")
```

2. **Missingness profile**

```python
missing = df.isnull().sum().to_frame("n_missing")
missing["pct_missing"] = missing["n_missing"] / len(df) * 100
missing.sort_values("pct_missing", ascending=False).head(20)
```

For top 10 columns:

- Numeric vs categorical?
- Likely to be important for prediction?
- Early idea: impute vs drop (note in markdown; don’t implement full imputation yet).

3. **Univariate distributions**

- For 3–5 key numeric features:
  - `.describe()`
  - Histogram (`.hist()` or `sns.histplot`)
- For 3–5 key categorical features:
  - `value_counts()`
  - Barplots (`sns.countplot`)

Write brief notes:
- Which features are skewed?
- Any bizarre values (e.g., negative ages, impossible amounts)?

4. **Train/test split design**

Decide:

- Is there a **time component**?
  - If yes: consider splitting by time (earlier for train, later for test).
- If classification with imbalance:
  - Use `stratify=y` in `train_test_split`.

Implement:

```python
target = "your_target"

df_model = df.dropna(subset=[target]).copy()
X = df_model.drop(columns=[target])
y = df_model[target]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if your_problem_is_classification else None
)
```

(If time‑based, do the split manually using a date column.)

### 2. Core practice (30–45 min)

1. **Check split integrity**

- Compare target distribution in `y_train` vs `y_test`.
- For 1–2 categorical features, compare proportions by split (e.g., `value_counts(normalize=True)`).

Ensure there’s no obvious drift introduced by the split.

2. **Potential leakage check (thought exercise)**

Scan your columns and mark:

- Any columns that might directly encode the target or post‑outcome info (e.g., “closed_loan_flag” when predicting default).
- Note these as likely **leaky** and plan to drop them.

### 3. Stretch (optional)

- If there’s a timestamp, plot count of rows per month or year to see if data distribution changes over time.

### 4. Reflection

- Are you confident your train/test split mimics **real-world deployment**? Why or why not?
- Which 2–3 features do you **expect** to be most predictive, based on EDA?

---

## Day 3 – Baselines: Naive, Rule-Based, and First Simple Model

**Goal:** Establish **multiple baselines**, including at least one **rule-based** and one **simple ML** model.

### 1. Core study (45–60 min)

Notebook: `notebooks/day3_baselines.ipynb`.

1. **Reload / reuse split data**

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/your_file.csv")
df_model = df.dropna(subset=[target]).copy()
X = df_model.drop(columns=[target])
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y if classification else None
)
```

2. **Naive baseline**

Classification:

```python
import numpy as np
from sklearn.metrics import accuracy_score

majority_class = y_train.value_counts().idxmax()
y_pred_majority = np.full_like(y_test, fill_value=majority_class)
baseline_acc = accuracy_score(y_test, y_pred_majority)
baseline_acc
```

Regression:

```python
from sklearn.metrics import mean_squared_error

import numpy as np
baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
baseline_rmse = mean_squared_error(y_test, baseline_pred, squared=False)
baseline_rmse
```

3. **Rule-based baseline (design first)**

Based on your EDA, define a simple rule in markdown before coding:

Examples:
- Heart disease: “If age > 50 and cholesterol above X → predict disease.”
- Churn: “If tenure < 12 months and contract is month-to-month → predict churn.”

Implement in code (classification example):

```python
def simple_rule(X):
    # Example structure; you fill in specifics
    cond = (
        (X["some_feature"] > threshold) &
        (X["another_feature"] == "some_category")
    )
    return cond.astype(int)  # adjust labels for your case
```

Evaluate:

```python
y_pred_rule = simple_rule(X_test)
# use appropriate metric
```

4. **First simple ML model**

- Use a **very simple model**:

  - Classification: `LogisticRegression` (with minimal preprocessing).
  - Regression: `LinearRegression` or `Ridge`.

For today, you can:

- Drop rows with missing values in key columns.
- One-hot encode categoricals using `pd.get_dummies`.

You’ll replace this with pipelines later; for now, keep it simple.

### 2. Core practice (30–45 min)

1. **Build a quick DataFrame of baselines**

Include:

- Naive baseline
- Rule-based baseline
- Simple ML model

Columns: model name + main metric (accuracy or RMSE).

2. **Interpret results**

In markdown:

- Is the simple ML model clearly better than rule-based?
- Is the rule-based significantly better than naive?

### 3. Stretch (optional)

- For classification: compute **precision, recall, and F1** for your rule-based baseline and simple ML model.
- For regression: compute **MAE** as well as RMSE.

### 4. Reflection

- If you only had time to deploy **one** of these baselines, which would you choose and why?
- What performance gap would you hope to achieve with more advanced models?

---

## Day 4 – Full Preprocessing Pipeline & Two Strong Models

**Goal:** Build a **proper ColumnTransformer + Pipeline** and plug in two serious models (linear + tree ensemble).

### 1. Core study (45–60 min)

Notebook: `notebooks/day4_pipelines_and_models.ipynb`.

1. **Identify numeric & categorical features (on X_train)**

```python
import numpy as np

numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
numeric_features, categorical_features
```

2. **Build preprocessing pipeline**

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
```

3. **Define two models**

Classification:

- Model A: `LogisticRegression`
- Model B: `RandomForestClassifier`

Regression:

- Model A: `Ridge` or `LinearRegression`
- Model B: `RandomForestRegressor`

Example (classification):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

log_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

rf_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])
```

### 2. Core practice (30–45 min)

1. **Fit and evaluate both models**

Use your main metric:

- Accuracy (classification)
- RMSE/R² (regression)

Create a small results table including:

- Naive baseline
- Rule-based baseline
- Model A
- Model B

2. **Choose a provisional “champion” model**

Based on test performance and simplicity:

- If performance difference is small, prefer simpler.
- If one clearly dominates, choose that as current champion.

### 3. Stretch (optional)

- Run **5-fold cross-validation** on each pipeline with `cross_val_score` and compare CV mean + std.

### 4. Reflection

- How much improvement did you get over your best baseline?
- Does the more complex model justify its complexity (training time, interpretability)?

---

## Day 5 – Systematic Hyperparameter Tuning

**Goal:** Use `GridSearchCV` or `RandomizedSearchCV` on your **champion pipeline** to seek better performance.

### 1. Core study (45–60 min)

Notebook: `notebooks/day5_tuning.ipynb`.

1. **Pick champion model**

From Day 4: say it’s `rf_clf` or `log_clf` (or the regression analog).

2. **Design parameter search space**

Before coding, in markdown:

- List 3–5 hyperparameters you think matter (e.g., `n_estimators`, `max_depth`, `min_samples_leaf`).
- Reasonable ranges (not too huge).

Example (RandomForestClassifier):

```python
param_grid = {
    "model__n_estimators": [100, 200, 400],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 5, 10],
}
```

3. **Run GridSearchCV**

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=rf_clf,        # or your chosen pipeline
    param_grid=param_grid,
    cv=3,
    scoring="accuracy" if classification else "neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
grid_search.best_params_, grid_search.best_score_
```

4. **Evaluate on test set**

```python
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)

# classification:
from sklearn.metrics import accuracy_score

acc_test = accuracy_score(y_test, y_test_pred)
acc_test
```

Or regression: RMSE/R².

### 2. Core practice (30–45 min)

1. **Experiment log**

Create a small DataFrame or markdown table summarizing:

- Baseline(s)
- Simple models
- Champion pre-tuning
- Champion post-tuning

Include metrics for each.

2. **Interpret tuning gain**

- Is improvement from tuning:
  - Negligible (< 1–2% relative)?
  - Moderate?
  - Large?

Think about whether expanding the search space is worth it.

### 3. Stretch (optional)

- Try `RandomizedSearchCV` with a slightly larger param space for a limited number of iterations.
- If classification: try optimizing **ROC-AUC** instead of accuracy and see if that changes best_params.

### 4. Reflection

- Did tuning confirm your intuition about which hyperparameters matter?
- At what point would you **stop tuning** and focus on other improvements (features, data quality, etc.)?

---

## Day 6 – Deeper Error Analysis, Segments, and Fairness Checks

**Goal:** Understand **where** your model fails (by segments), and check for potential bias or systematic weaknesses.

### 1. Core study (45–60 min)

Notebook: `notebooks/day6_error_analysis.ipynb`.

1. **Build a test set DataFrame with predictions**

```python
import numpy as np
import pandas as pd

df_test = X_test.copy()
df_test[target] = y_test
df_test["y_pred"] = best_model.predict(X_test)

if classification:
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    print(confusion_matrix(y_test, df_test["y_pred"]))
    print(classification_report(y_test, df_test["y_pred"]))
else:
    from sklearn.metrics import mean_squared_error
    errors = y_test - df_test["y_pred"]
    df_test["error"] = errors
    df_test["abs_error"] = np.abs(errors)
```

2. **Identify mispredictions / large errors**

Classification:

```python
df_test["correct"] = (df_test[target] == df_test["y_pred"])
mis = df_test[~df_test["correct"]]
mis.head()
```

Regression:

```python
df_test.sort_values("abs_error", ascending=False).head(20)
```

3. **Define 1–2 key segments**

Examples:

- Age groups
- Income tiers
- Product category
- Region

Create bins/segments, e.g.:

```python
df_test["age_group"] = pd.cut(
    df_test["age"],
    bins=[0, 30, 50, 100],
    labels=["young", "middle", "older"]
)
```

### 2. Core practice (30–45 min)

1. **Segment performance**

Classification (accuracy per segment):

```python
df_test.groupby("age_group")["correct"].mean()
```

Regression (RMSE per segment):

```python
from sklearn.metrics import mean_squared_error

for g, group in df_test.groupby("age_group"):
    rmse_seg = mean_squared_error(group[target], group["y_pred"], squared=False)
    print(g, rmse_seg)
```

2. **Check at least one sensitive / ethically relevant feature** (if present)

- E.g., gender, region, etc.
- Compare accuracy/RMSE across groups.

In markdown:
- Any big performance gaps?
- Might this matter in real use?

3. **List top 3 failure modes**

From mispredictions / large errors, describe:

- “Model underestimates [target] when [condition].”
- “Model often misclassifies [class] when [feature pattern].”

### 3. Stretch (optional)

- Use **feature importances** (trees) or **coefficients** (linear/logistic) to reason about whether the model is underusing an important feature.
- Consider adding a simple new feature to help with a specific failure mode (even if you won’t fully implement it now).

### 4. Reflection

- Which user or case type is your model **least reliable** for?
- If you had to present your model to a stakeholder, what caveat would you emphasize based on this analysis?

---

## Day 7 – Week 8 Milestone: Clean Summary Notebook & Next Steps Plan

**Goal:** Consolidate Week 8 into a clean, reproducible summary and decide what to tackle in Weeks 9–12.

### 1. Core study (45–60 min)

Create `notebooks/week8_milestone_summary.ipynb`.

1. **Rebuild a minimal end-to-end flow**

In this notebook:

- Import packages.
- Load raw data.
- Minimal cleaning (dropping clear junk columns).
- Train/test split.
- Build `preprocessor` (`ColumnTransformer`).
- Build `best_model` pipeline with tuned hyperparameters (you can copy from earlier but clean it up).
- Fit on train, evaluate on test.
- Print key metrics and (optionally) a confusion matrix or RMSE + R².

2. **Add 1–2 key visualizations**

- Either:
  - Feature importance plot (trees), or
  - Coefficient bar plot (linear model).
- And/or a small error plot (e.g., histogram of errors or predicted vs actual).

### 2. Core practice (30–45 min)

1. **Write a concise Week 8 report (markdown inside summary notebook)**

Include sections:

- **Problem**: 2–3 sentences.
- **Data**: Rows, columns, key features.
- **Methods**:
  - Baselines.
  - Models and pipelines.
  - Tuning approach.
- **Results**:
  - Baseline vs best model metrics.
  - Short note on error analysis / segments.
- **Current limitations**:
  - Data issues, performance level, interpretability.
- **Planned next steps (Weeks 9–12)**:
  - Ideas like:
    - More feature engineering.
    - Trying gradient boosting (XGBoost/LightGBM/CatBoost).
    - Handling imbalance better.
    - Simple deployment (API or batch scoring).
    - More rigorous evaluation (time-based CV, more metrics, calibration).

2. **Skill self-check**

In markdown, rate (1–10) your confidence in:

- Building pipelines for mixed-type tabular data.
- Designing baselines and comparing them fairly.
- Running small hyperparameter searches.
- Doing basic error analysis & segmentation.
- Writing clear, concise project summaries.

For each, write *why* you chose that number and one action to move it up.

### 3. Stretch (optional)

- Create a simple `requirements.txt` (or `environment.yml` if using conda).
- Initialize a Git repo for `capstone/` (if you haven’t yet), make a commit with:
  - Clean notebooks.
  - README.
  - requirements file.

### 4. Reflection

- What is the **single biggest win** from Week 8 for you?
- If someone gave you a **new** supervised dataset tomorrow, how confident are you that you could take it from raw CSV to:
  - Baselines
  - Pipelines
  - Tuned model
  - Short report  
  in a few focused days?

---

If you’d like, next I can generate a **Week 9 plan** that either:
- Pushes this capstone further (advanced models, calibration, deployment), or  
- Adds depth in a specific area you care about (time series, NLP, recommendation, etc.), tailored to the capstone you chose.
