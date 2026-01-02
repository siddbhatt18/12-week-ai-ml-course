Week 7 is about doing your **first end‑to‑end supervised ML project**, using everything so far:

- Problem framing
- EDA
- Baselines
- Pipelines & model comparison
- Error analysis & interpretation
- A short, coherent write‑up

Assume ~1–2 hours/day.  
Each day: **Goal → Core study → Core practice → Stretch → Reflection**.

Pick **one main dataset** for the whole week, ideally **new** to you and interesting:

Suggested options (classification or regression, any is fine):
- Classification:  
  - Kaggle: “Heart Disease UCI”  
  - Kaggle: “Telco Customer Churn”  
  - Kaggle: “Loan Prediction” / “Give Me Some Credit”  
- Regression:  
  - Kaggle: “Bike Sharing Demand”  
  - Kaggle: “NYC Taxi Trip Duration”

I’ll call it `PROJECT_DATA` below; adapt column names as needed.

---

## Day 1 – Choose Dataset, Define Problem & Metric, Set Up Project Structure

**Goal:** Choose a dataset, understand the context, and define a clear prediction problem + evaluation metric.

### 1. Core study (45–60 min)

Create folder: `week7_project/` and inside it:
- `data/` (raw CSVs)
- `notebooks/`
- `src/` (optional, for later)
- `README.md`

Create notebook: `notebooks/day1_problem_definition.ipynb`.

1. **Choose dataset & download**

Pick one dataset and:
- Download CSVs into `week7_project/data`
- Skim the Kaggle/README description:
  - What does each row represent?
  - What’s the main outcome / label (target)?
  - Any warnings about quality, missing data, etc.?

2. **Load and quick inspect**

```python
import pandas as pd

df = pd.read_csv("../data/your_file.csv")  # adjust path/name
df.head()
df.shape
df.info()
df.describe(include="all").T.head(20)
```

3. **Define the ML problem (markdown)**

Write, in your own words:

- **Task type**: classification or regression?
- **Target column**: e.g. `churn`, `default`, `duration`, `demand`.
- **Features**: initial guess of which columns are useful.
- **Business / practical goal**:  
  - What decisions will this help with?  
  - Who would care about this model?

4. **Choose an evaluation metric**

For classification (binary):
- Default: **accuracy** + **ROC-AUC** (and possibly precision/recall).
For regression:
- Default: **RMSE** (and possibly MAE, R²).

Justify your choice in markdown:
- If misclassifying positives is very costly → care more about recall.
- If you care about large errors more → RMSE.

### 2. Core practice (30–45 min)

1. **Identify basic column types**

Create a small “schema” table in pandas or markdown:

For each column, list:
- Name
- Type (`numeric`, `categorical`, `date`, `id`, `text`)
- Description (1 line, from README)
- Initial guess: **keep/drop/unsure**

2. **Target distribution**

If classification:

```python
df[target].value_counts()
df[target].value_counts(normalize=True)
```

If regression:

```python
df[target].describe()
df[target].hist(bins=30)
```

In markdown:
- Is the target balanced or imbalanced?  
- Rough range & typical values?

### 3. Stretch (optional)

- Add a short **“Risks / Ethics”** section:
  - Any sensitive attributes (e.g. gender, race)?  
  - How could misuse of the model harm people?

### 4. Reflection

- Can you explain your project in **2–3 sentences** to a non‑technical friend?
- On a scale of 1–10, how clear is your **metric choice** and why it matters?

---

## Day 2 – Data Cleaning, Splits & Baselines

**Goal:** Clean the data enough to work with, decide splits, and compute a **very simple baseline**.

### 1. Core study (45–60 min)

New notebook: `notebooks/day2_cleaning_baselines.ipynb`.

1. **Reload data and basic checks**

```python
import pandas as pd
import numpy as np

df = pd.read_csv("../data/your_file.csv")
df.head()
df.info()
```

2. **Handle obvious bad columns**

In markdown, mark:
- Clearly **useless** columns: purely IDs, constant columns.
- Drop them:

```python
drop_cols = ["ID_column_name"]  # adjust
df = df.drop(columns=drop_cols)
```

3. **Inspect missing values**

```python
missing = df.isnull().sum().to_frame("n_missing")
missing["pct_missing"] = missing["n_missing"] / len(df) * 100
missing.sort_values("pct_missing", ascending=False).head(20)
```

For each top-missing column, decide (markdown):
- Drop column entirely?  
- Impute later?  
- Drop few rows?

Don’t overthink – just note decisions; you’ll implement proper imputation in a pipeline later.

4. **Train/validation/test split strategy**

Decide now:

- If data is large:
  - Train/validation/test (e.g. 60/20/20).
- If smaller:
  - Train/test (80/20) + use cross-validation on train.

Implement **train/test** split from raw data, using only rows where target is not null:

```python
target = "your_target_column"

df_model = df.dropna(subset=[target]).copy()
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

### 2. Core practice (30–45 min)

1. **Very naive baseline**

Classification:
- Majority class baseline:

```python
y_train.value_counts(normalize=True)
```

Let `p_majority` = largest proportion → baseline accuracy.

Compute test baseline explicitly:

```python
from sklearn.metrics import accuracy_score

majority_class = y_train.value_counts().idxmax()
y_pred_baseline = np.full_like(y_test, fill_value=majority_class)
baseline_acc = accuracy_score(y_test, y_pred_baseline)
baseline_acc
```

Regression:
- Mean baseline:

```python
from sklearn.metrics import mean_squared_error
import numpy as np

baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
baseline_rmse = mean_squared_error(y_test, baseline_pred, squared=False)
baseline_rmse
```

2. **Write baseline interpretation**

In markdown:
- “If I always predict X, I get Y% accuracy / RMSE = Z.”
- What would a model need to beat this *significantly*?

### 3. Stretch (optional)

- If time dimension exists (e.g., time‑series-ish), think about whether you should split by **time** rather than random (no leakage from future).

### 4. Reflection

- Did the baseline perform better or worse than you intuitively expected?
- What do you think would be an **ambitious but realistic** performance goal for a first good model?

---

## Day 3 – Focused EDA on Features vs Target & Simple Rule-Based Baselines

**Goal:** Understand how key features relate to the target and build at least **one simple rule-based baseline**.

### 1. Core study (45–60 min)

New notebook: `notebooks/day3_eda_target_relationships.ipynb`.

1. **Reload split data**

Either re-split or load from Day 2 (you can also save splits to disk if you want; re-splitting is fine for now).

```python
df = pd.read_csv("../data/your_file.csv")
df_model = df.dropna(subset=[target]).copy()
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

2. **Choose 3–5 promising features**

From your Day 1 schema and intuition, pick:
- 2–3 numeric
- 1–2 categorical

3. **Numeric vs target**

Classification:
- Plot distributions of a numeric feature for each class:

```python
import seaborn as sns
import matplotlib.pyplot as plt

feature = "some_numeric_feature"
df_train = X_train.copy()
df_train[target] = y_train

sns.boxplot(x=target, y=feature, data=df_train)
plt.show()
```

Regression:
- Scatter `feature` vs target:

```python
plt.scatter(X_train[feature], y_train, alpha=0.3)
plt.xlabel(feature)
plt.ylabel(target)
plt.show()
```

4. **Categorical vs target**

Classification:
```python
cat_feature = "some_categorical_feature"
pd.crosstab(df_train[cat_feature], df_train[target], normalize="index")
```

Plot with countplot / barplot.

Regression:
- Group by category and take mean target:
```python
Xy_train = X_train.copy()
Xy_train[target] = y_train
Xy_train.groupby(cat_feature)[target].mean().sort_values()
```

### 2. Core practice (30–45 min)

1. **Design one simple rule-based baseline**

Based on EDA, define a rule:

- Heart disease example:
  - If `age > 50` and `cholesterol` high → predict disease; else no.
- Churn:
  - If `contract_type` = month‑to‑month and `tenure` < 12 → predict churn; else no.

Implement:

```python
def simple_rule(X):
    # return a Series of predictions
    # e.g., for binary classification:
    return ((X["tenure"] < 12) & (X["Contract"] == "Month-to-month")).astype(int)
```

Evaluate on `X_test`:

```python
y_pred_rule = simple_rule(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred_rule)
```

If regression, you can define a rule like: predict a different mean for different groups.

2. **Compare rule-based vs naive baseline**

In markdown:
- Is rule-based baseline better than majority/mean?
- If yes, by how much?

### 3. Stretch (optional)

- Compute more metrics for your rule baseline: precision, recall, F1, or RMSE/MAE.
- For classification, examine confusion matrix for the rule.

### 4. Reflection

- What did EDA change about your understanding of which features matter?
- Does your rule-based baseline already seem “good enough” in some sense, or clearly improvable?

---

## Day 4 – First Proper Models: Pipelines, Preprocessing, Model Comparison

**Goal:** Build **end-to-end pipelines** for at least **two supervised models**, and compare them to baselines.

### 1. Core study (45–60 min)

New notebook: `notebooks/day4_first_models.ipynb`.

1. **Identify numeric & categorical features**

```python
import numpy as np

numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_features, categorical_features
```

2. **Build preprocessing**

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

- If classification: LogisticRegression vs RandomForestClassifier.
- If regression: LinearRegression / Ridge vs RandomForestRegressor.

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

```python
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

models = {"Logistic/Linear": log_clf, "RandomForest": rf_clf}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if classification:
        acc = accuracy_score(y_test, y_pred)
        results.append({"model": name, "test_acc": acc})
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results.append({"model": name, "rmse_test": rmse, "r2_test": r2})

import pandas as pd
pd.DataFrame(results)
```

2. **Compare to baselines**

In markdown:
- How do these models compare to:
  - Naive baseline?
  - Rule-based baseline?
- Which model currently looks promising?

### 3. Stretch (optional)

- Run cross-validation on the best‑looking model pipeline:

```python
from sklearn.model_selection import cross_val_score

best_model = rf_clf  # or log_clf
scores = cross_val_score(best_model, X, y, cv=5,
                         scoring="accuracy" if classification else "neg_root_mean_squared_error")
scores.mean(), scores.std()
```

### 4. Reflection

- Did the more complex model clearly outperform the simpler one?
- If not, what might be causing that (data size, noise, feature quality)?

---

## Day 5 – Hyperparameter Tuning & Error Analysis

**Goal:** Tune your best model a bit and do **error analysis**: understand where it fails.

### 1. Core study (45–60 min)

New notebook: `notebooks/day5_tuning_and_errors.ipynb`.

1. **Select the best model so far**

From Day 4’s comparison, choose a “winner” (e.g., RandomForest).

Assume it’s called `base_clf` / `base_reg`.

2. **Set up GridSearchCV**

Classification example:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 5, 10],
}

grid_search = GridSearchCV(
    rf_clf,  # pipeline
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
grid_search.best_params_, grid_search.best_score_
```

Regression: use `scoring="neg_root_mean_squared_error"`.

3. **Evaluate tuned model on test set**

```python
best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)

# classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

acc = accuracy_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)
print("Test accuracy:", acc)
cm
print(classification_report(y_test, y_test_pred))
```

Regression: RMSE, MAE, R².

### 2. Core practice (30–45 min)

1. **Error analysis table**

Classification:
- Add predictions & a “correct/incorrect” column:

```python
df_test = X_test.copy()
df_test[target] = y_test
df_test["y_pred"] = y_test_pred
df_test["correct"] = (df_test[target] == df_test["y_pred"])
df_test["correct"].value_counts(normalize=True)
```

- Inspect some **misclassified** examples:

```python
misclassified = df_test[~df_test["correct"]].head(20)
misclassified
```

Ask:
- Do errors cluster in certain ranges of a feature (age, amount, etc.)?
- Any few categories where model is particularly bad?

Regression:
- Add `error = y_test - y_pred` and inspect top absolute errors:

```python
import numpy as np

errors = y_test - y_test_pred
df_test = X_test.copy()
df_test[target] = y_test
df_test["y_pred"] = y_test_pred
df_test["error"] = errors
df_test["abs_error"] = np.abs(errors)

df_test.sort_values("abs_error", ascending=False).head(20)
```

2. **Segmented performance**

Pick **one feature** (e.g., age group, customer type, region) and compare performance per group.

Classification example:

```python
df_test["segment"] = pd.cut(df_test["age"], bins=[0, 30, 50, 100],
                            labels=["young", "middle", "older"])

segment_perf = df_test.groupby("segment")["correct"].mean()
segment_perf
```

Regression example: compute RMSE per segment.

### 3. Stretch (optional)

- For RandomForest, inspect feature importances:

```python
import pandas as pd
import numpy as np

# Get transformer & names (similar to earlier weeks)
pre = best_model.named_steps["preprocess"]
model = best_model.named_steps["model"]

# Numeric + OHE names (requires a bit of care; you can reuse earlier working code)
```

- Look at top ~10 features and see if errors are larger when those features have extreme values.

### 4. Reflection

- Where does your model **struggle most** (which segment, which feature range)?
- If you had more time/data, what would you do to reduce these errors?

---

## Day 6 – Interpretation, Documentation & Light Packaging

**Goal:** Make your project **understandable and reproducible**: interpret feature impacts (where possible), clean up notebooks, and draft a short README.

### 1. Core study (45–60 min)

New notebook: `notebooks/day6_interpretation_and_docs.ipynb`.

1. **Interpretability (simple)**

For tree-based models:
- Feature importance:

```python
import pandas as pd
import numpy as np

pre = best_model.named_steps["preprocess"]
model = best_model.named_steps["model"]

# Get feature names
numeric_features = ...
categorical_features = ...
ohe = pre.named_transformers_["cat"].named_steps["onehot"]
ohe_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([numeric_features, ohe_names])

importances = pd.Series(model.feature_importances_, index=all_feature_names)
importances.sort_values(ascending=False).head(20)
```

Classification (logistic regression):
- Coefficients:

```python
coef = best_model.named_steps["model"].coef_[0]
coef_series = pd.Series(coef, index=all_feature_names)
coef_series.sort_values(ascending=False).head(20)
coef_series.sort_values().head(20)
```

2. **Notebook clean-up plan**

Decide:
- Which notebooks are “exploration” and which are final?
- For a “final” notebook:
  - Sections: Load Data → EDA → Baselines → Modeling → Tuning → Evaluation → Conclusions.

### 2. Core practice (30–45 min)

1. **Create a concise summary notebook**

Create `notebooks/final_summary.ipynb`:

- Only essential cells:
  - Load data.
  - Minimal cleaning.
  - Train/test split.
  - Build + fit best pipeline model.
  - Evaluate and print key metrics.
  - Plot 1–2 important charts (e.g., feature importance).

2. **Write a README (markdown file)**

In `week7_project/README.md`, include:

- **Title**
- **Problem statement** (2–3 sentences)
- **Data description** (rows, columns, source)
- **Modeling approach**:
  - Main models tried.
  - Best model chosen.
- **Results**:
  - Baseline vs final metrics.
- **How to run**:
  - Python version, `pip install -r requirements.txt` (if you create one), run notebook.
- **Limitations & future work**

### 3. Stretch (optional)

- Save model pipeline:

```python
import joblib
joblib.dump(best_model, "../models/best_model.joblib")
```

- Write a tiny script `src/predict_one.py` that:
  - Loads the model.
  - Creates a DataFrame from a hard-coded example.
  - Prints prediction.

### 4. Reflection

- If a stranger cloned your repo, would they know:
  - What the project is about?
  - How to reproduce your results?
- What’s the weakest part of your documentation right now?

---

## Day 7 – Week 7 Wrap-Up & Planning the Bigger Capstone

**Goal:** Step back, consolidate what you’ve built, and plan how you’ll tackle a **larger capstone project** in coming weeks.

### 1. Core study (45–60 min)

New notebook: `notebooks/day7_review_and_capstone_plan.ipynb`.

1. **Summarize Week 7 project**

In markdown, write a structured summary:

- **1. Problem & Data**
  - Short description of task and dataset.
- **2. Methods**
  - Baseline(s), models, preprocessing, tuning.
- **3. Results**
  - Baseline vs final metrics.
  - Key error analysis findings.
- **4. Interpretation**
  - Which features mattered most?
  - Where is the model less reliable?
- **5. Lessons learned**
  - Pipeline skills, tuning insights, etc.

2. **Self-assessment of skills**

Rate (1–10) your comfort with:

- Loading & cleaning tabular data.
- EDA focused on target.
- Building pipelines with numeric + categorical features.
- Choosing and justifying metrics.
- Training and tuning tree‑based models.
- Doing basic error analysis.

Write 1–2 sentences for each rating explaining *why*.

### 2. Core practice (30–45 min)

1. **Capstone brainstorming (markdown)**

Brainstorm **2–3 ideas** for a larger capstone (Weeks 8–12), e.g.:

- Domain (health / finance / sports / e‑commerce / personal data).
- Potential dataset(s) (Kaggle, public APIs, your own data).
- Why you care about this problem.
- What would “success” look like (metrics, insights, or both).

2. **High-level capstone plan**

For your favorite idea, outline a rough plan:

1. **Week A**: Data collection & cleaning.
2. **Week B**: EDA + baselines.
3. **Week C**: Modeling & tuning.
4. **Week D**: Interpretation, documentation, optional deployment.

Include:
- Likely model types (classification/regression?).
- Any special wrinkles (time series? text? strong class imbalance?).

### 3. Stretch (optional)

- Read 1–2 high-quality Kaggle notebooks/kernels for a similar problem:
  - Note how they structure their analysis.
  - Pick 2–3 ideas to emulate (not code, but approach/organization).

### 4. Reflection

- What part of doing a **full project** (start to finish) felt most different from isolated exercises?
- If you repeated this Week 7 project from scratch, what would you do differently?
- What’s one specific habit you want to keep (e.g., always writing a baseline first, always checking segment-wise performance)?

---

If you’d like, next I can generate a **Week 8 (Day 1–7) plan** focused on either:
- Starting your **true capstone** (using the idea you just brainstormed), or  
- Building depth in a particular direction (e.g., time series, NLP, or deeper tree‑based methods like XGBoost/LightGBM).
