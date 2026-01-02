Week 5 is about turning everything you’ve done so far into **clean, reusable workflows** using:

- `SimpleImputer`, `StandardScaler`, `OneHotEncoder`
- `Pipeline` and `ColumnTransformer`
- Basic **hyperparameter tuning** with `GridSearchCV` / `RandomizedSearchCV`
- Simple **feature engineering**

Assume ~1–2 hours/day.  
You’ll mainly use:

- **House Prices** (regression)
- **Adult Income** (classification)
- Optionally Titanic at the end

Each day: **Goal → Core study → Core practice → Stretch → Reflection**.  
Keep the mindset: *“Can I build this from scratch in a new notebook without copy-paste?”*

---

## Day 1 – Why Pipelines? Imputation & Scaling for Regression (House Prices)

**Goal:** Understand **why** we use pipelines, what **data leakage** is, and build your first **numeric preprocessing + model pipeline**.

### 1. Core study (45–60 min)

Create `week5_day1_houseprices_pipeline.ipynb`.

1. **Concept: what preprocessing does & why leakage is bad**

In a markdown cell, briefly think through (before coding):

- Real data has:
  - Missing values
  - Mixed scales (e.g. “YearBuilt” vs “GrLivArea”)
  - Outliers
- Models (esp. linear, kNN, SVM) usually prefer:
  - No missing values
  - Reasonable scales  
- **Data leakage**: using information from the **test set** during training/preprocessing.
  - Example: computing the mean of a feature on **all data** before split and using it to impute training and test.

You’ll see how pipelines help avoid this.

2. **Reload House Prices & basic split**

Recreate a numeric-only dataset (similar to Week 3–4):

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

df = pd.read_csv("train.csv")
target = "SalePrice"

feature_cols = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
]

df_model = df[feature_cols + [target]].copy()
X = df_model[feature_cols]
y = df_model[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

3. **Deliberate “bad” preprocessing (for understanding)**

Compute **global** means and fill missing values **before** split (leaky way):

```python
X_leaky = X.copy()
X_leaky = X_leaky.fillna(X_leaky.mean())  # this uses info from all data
```

You already split above; imagine if you had done this *before* splitting – that’s leakage.

4. **Now: proper preprocessing with Pipeline**

Learn the basic objects (look up docs):

- `SimpleImputer` – fills missing values.
- `StandardScaler` – scales features to 0 mean, unit variance.
- `Pipeline` – chains steps.

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

numeric_pipeline.fit(X_train, y_train)

y_pred = numeric_pipeline.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
rmse, r2
```

Note: imputer and scaler are **fitted only on X_train** inside the pipeline.

### 2. Core practice (30–45 min)

1. **Compare against a non-pipeline version**

Implement the equivalent “manual” steps without Pipeline:

```python
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
model = LinearRegression()

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

model.fit(X_train_scaled, y_train)
y_pred_manual = model.predict(X_test_scaled)

rmse_manual = mean_squared_error(y_test, y_pred_manual, squared=False)
r2_manual = r2_score(y_test, y_pred_manual)
rmse_manual, r2_manual
```

Check that the **Pipeline and manual** approaches give (almost) identical results.

2. **Inspect fitted steps**

```python
numeric_pipeline.named_steps["imputer"].statistics_
numeric_pipeline.named_steps["scaler"].mean_
numeric_pipeline.named_steps["model"].coef_
```

Write down:

- What does each object store after fitting?

### 3. Stretch (optional)

- Manually introduce some missing values into `X_train` (e.g., set random 5% of values in a column to `np.nan`), then fit pipeline again and check that it still works.
- Try a different imputation strategy (`"mean"` vs `"median"`) and see if metrics change noticeably.

### 4. Reflection

- In your own words, how does a Pipeline help avoid data leakage?
- Which part of the manual approach is most error-prone compared to using a Pipeline?

---

## Day 2 – Handling Categorical Data: ColumnTransformer + Pipeline (Adult Income)

**Goal:** Build your first **mixed-type preprocessing pipeline** (numeric + categorical) using `ColumnTransformer` and `OneHotEncoder`.

### 1. Core study (45–60 min)

Create `week5_day2_adult_columntransformer.ipynb`.

1. **Reload Adult Income data**

```python
import pandas as pd
import numpy as np

df = pd.read_csv("adult.csv")  # adjust file name if needed
df.replace("?", pd.NA, inplace=True)
df.head()
df.info()
```

2. **Define target and features**

Assuming income column:

```python
target = "income"
df["income_binary"] = (df[target].str.contains(">50K")).astype(int)

features = [
    "age", "education-num", "hours-per-week", "capital-gain",
    "capital-loss", "workclass", "marital-status", "occupation",
    "relationship", "sex", "native-country"
]

df_model = df[features + ["income_binary"]].dropna()
X = df_model[features]
y = df_model["income_binary"]
```

3. **Identify numeric & categorical columns**

```python
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_features, categorical_features
```

4. **Build transformers**

Look up docs for `SimpleImputer` and `OneHotEncoder`:

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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

5. **Full pipeline with classifier**

```python
clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
acc
```

### 2. Core practice (30–45 min)

1. **Inspect transformed feature space**

Use `preprocessor` alone to see transformed shape:

```python
preprocessor.fit(X_train)
X_train_transformed = preprocessor.transform(X_train)
X_train_transformed.shape
```

- How many features now? (OneHotEncoder expanded them.)

2. **Experiment with different numeric scaling strategies**

- Create a second numeric transformer that **does not scale** (only impute).
- Compare accuracy between:
  - Pipeline with scaling
  - Pipeline without scaling  
  (Keep everything else same.)

### 3. Stretch (optional)

- Get feature names after OneHotEncoder (slightly advanced):

```python
ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
ohe_feature_names = ohe.get_feature_names_out(categorical_features)

len(ohe_feature_names), ohe_feature_names[:20]
```

- Inspect some of these names to understand how categories were encoded.

### 4. Reflection

- Why is `handle_unknown="ignore"` important for OneHotEncoder in real-world deployments?
- How does using ColumnTransformer improve over manually doing `pd.get_dummies`?

---

## Day 3 – Pipelines with Tree-Based Models & Model Swapping

**Goal:** Reuse the **same preprocessing** but plug in different models (LogReg vs RandomForest) and compare them cleanly.

### 1. Core study (45–60 min)

Create `week5_day3_adult_models_with_pipeline.ipynb`.

Assume you have:

- `X`, `y`, `numeric_features`, `categorical_features` from Day 2.

1. **Rebuild preprocessor**

(Or import from yesterday.)

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

2. **Create two model pipelines**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logreg_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

rf_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    ))
])
```

3. **Fit and evaluate both**

```python
models = {"LogReg": logreg_clf, "RandomForest": rf_clf}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({"model": name, "test_acc": acc})

import pandas as pd
pd.DataFrame(results)
```

### 2. Core practice (30–45 min)

1. **Add cross-validation comparison**

Use `cross_val_score` directly on pipelines:

```python
from sklearn.model_selection import cross_val_score

cv_results = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    cv_results.append({
        "model": name,
        "cv_mean_acc": scores.mean(),
        "cv_std_acc": scores.std()
    })

pd.DataFrame(cv_results)
```

Compare CV performance to single train/test split.

2. **Investigate training time vs performance**

- Roughly time model fitting (even manual `%%time` in Jupyter).
- Note if RandomForest is significantly slower than LogisticRegression and whether the accuracy change justifies it.

### 3. Stretch (optional)

- Retrieve **feature importances** from the forest inside the pipeline:

```python
rf_clf.fit(X_train, y_train)
rf_model = rf_clf.named_steps["model"]

# Get feature names from preprocessor
ohe = rf_clf.named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
ohe_names = ohe.get_feature_names_out(categorical_features)
num_names = numeric_features
all_feature_names = np.concatenate([num_names, ohe_names])

importances = pd.Series(rf_model.feature_importances_, index=all_feature_names)
importances.sort_values(ascending=False).head(15)
```

Plot top 15 if you like.

### 4. Reflection

- How does swapping out the model become easier when everything else is in a pipeline?
- Based on CV, which model would you choose as your default for Adult Income and why?

---

## Day 4 – Hyperparameter Tuning with GridSearchCV on Pipelines

**Goal:** Use `GridSearchCV` (and optionally `RandomizedSearchCV`) to tune **pipeline hyperparameters** cleanly.

### 1. Core study (45–60 min)

Create `week5_day4_adult_gridsearch.ipynb`.

1. **Set up base pipeline (RandomForest)**

Reuse Adult Income `X`, `y`, `preprocessor`:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

rf_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
])
```

2. **Define a parameter grid**

Remember: to access step parameters, use `stepname__paramname`.

```python
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 5, 10],
}
```

3. **Run GridSearchCV**

```python
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    estimator=rf_clf,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)
```

4. **Inspect best results**

```python
grid_search.best_params_
grid_search.best_score_
```

Convert results to DataFrame:

```python
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results[["params", "mean_test_score", "std_test_score"]].sort_values(
    "mean_test_score", ascending=False
).head(10)
```

### 2. Core practice (30–45 min)

1. **Compare tuned model vs default**

- Default RF pipeline CV accuracy (from Day 3) vs `best_score_` from grid search.
- Is the improvement significant?

2. **Apply best model to train/test split**

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
acc_test
```

Compare with non-tuned pipeline.

### 3. Stretch (optional)

Try `RandomizedSearchCV`:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    "model__n_estimators": randint(50, 300),
    "model__max_depth": [None, 10, 20, 30],
    "model__min_samples_leaf": randint(1, 20),
}

rand_search = RandomizedSearchCV(
    rf_clf,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42,
    verbose=1
)
rand_search.fit(X, y)
rand_search.best_params_, rand_search.best_score_
```

### 4. Reflection

- Did hyperparameter tuning give you a **meaningful** boost, or just a tiny one?
- How would you decide when the extra computation/time of tuning is worth it?

---

## Day 5 – Basic Feature Engineering & Custom Transformations

**Goal:** Learn to **create new features** and (optionally) incorporate them into a pipeline in a clean way.

### 1. Core study (45–60 min)

Create `week5_day5_feature_engineering.ipynb`.

Use Adult Income again (though you can pick any dataset).

1. **Brainstorm possible new features**

In markdown, list at least 3 ideas, e.g.:

- Age buckets: young / middle / older.
- Interaction between `education-num` and `hours-per-week`.
- Binary flag: `capital-gain > 0` or `capital-loss > 0`.
- Country grouped into “US” vs “Other”.

2. **Implement feature engineering in pandas (simple way)**

Example features:

```python
df_fe = df_model.copy()  # from Day 2: features + income_binary

# Age bucket
df_fe["age_bucket"] = pd.cut(
    df_fe["age"],
    bins=[0, 25, 45, 65, 100],
    labels=["young", "middle", "older", "senior"]
)

# Binary flags
df_fe["has_capital_gain"] = (df_fe["capital-gain"] > 0).astype(int)
df_fe["has_capital_loss"] = (df_fe["capital-loss"] > 0).astype(int)

# Interaction
df_fe["edu_hours"] = df_fe["education-num"] * df_fe["hours-per-week"]
```

Redefine X, y:

```python
target = "income_binary"
feature_cols = features + ["age_bucket", "has_capital_gain", "has_capital_loss", "edu_hours"]
X = df_fe[feature_cols]
y = df_fe[target]
```

3. **Update numeric & categorical lists**

```python
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
numeric_features, categorical_features
```

4. **Reuse preprocessor & a simple model pipeline**

```python
# rebuild transformers with new feature lists
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

from sklearn.linear_model import LogisticRegression
clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])
```

### 2. Core practice (30–45 min)

1. **Compare with/without engineered features**

Train/test split, then:

- Model A: original features (no new ones).
- Model B: extended features.

Record accuracy (or ROC-AUC, if you want) for both (using same random_state/split).

2. **Interpretation**

Pick 1–2 engineered features and reason:

- Did they improve model performance?
- Are they likely to be useful logically (beyond just numbers)?

### 3. Stretch (optional)

Create a **custom transformer** using `FunctionTransformer`:

```python
from sklearn.preprocessing import FunctionTransformer

def add_interaction_features(df):
    df = df.copy()
    df["edu_hours"] = df["education-num"] * df["hours-per-week"]
    return df

interaction_transformer = FunctionTransformer(add_interaction_features)

pipeline_with_custom = Pipeline(steps=[
    ("add_features", interaction_transformer),
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])
```

You’ll need to adapt the preprocessor to expect the new column. This is more advanced; try only if you’re comfortable.

### 4. Reflection

- What’s the difference between **feature engineering** and just adding more raw columns?
- Which engineered feature do you believe is most meaningful from a real-world standpoint, not just a numerical one?

---

## Day 6 – End-to-End Adult Income Pipeline: From Raw Data to Tuned Model

**Goal:** Do a **small end-to-end project** on Adult Income using everything from Weeks 4–5:

- Clean raw data
- Feature engineering
- Pipeline + ColumnTransformer
- Hyperparameter tuning
- Evaluation

### 1. Core study (45–60 min)

Create `week5_day6_adult_end_to_end.ipynb`.

Try to **write this mostly from scratch**, referencing earlier notebooks as needed.

1. **Outline your steps (markdown)**

Write out your plan:

1. Load raw CSV.
2. Handle `"?"` missing markers.
3. Define target + initial features.
4. Train/test split (from raw features).
5. Build preprocessing + model pipeline.
6. Add 1–2 engineered features.
7. Tune model hyperparameters.
8. Evaluate final model on test set.
9. Summarize.

2. **Implement steps 1–5**

Without copying big blocks:

- Load `df`.
- Replace `"?"` with `pd.NA`.
- Build `df_model`, `X`, `y`.
- Split train/test **before** any fitting.
- Define numeric/categorical features.
- Build `preprocessor` (impute + scale + one-hot).
- Build a base model pipeline (pick either LogReg or RandomForest).

3. **Implement feature engineering (simple)**

You can:

- Add `age_bucket`.
- Add `has_capital_gain`.
- Add `edu_hours`.

Decide whether you:

- Engineer before splitting (then split), *or*
- Engineer only on train and apply same transformation to test (safer, but more code).  
For now, since features are simple and don’t use target, engineering before split is acceptable in this practice.

4. **Hyperparameter tuning**

Use `GridSearchCV` or `RandomizedSearchCV` on your chosen model, like Day 4.

### 2. Core practice (30–45 min)

1. **Evaluate final tuned model on test set**

- Compute:
  - Accuracy
  - Precision
  - Recall
  - F1
  - ROC-AUC (if comfortable)

2. **Write a short “results” section (markdown)**

Include:

- Which model you chose (LogReg vs RF).
- Which hyperparameters ended up best.
- CV score of best model during tuning.
- Final test metrics.

### 3. Stretch (optional)

- Save your final pipeline to disk:

```python
import joblib
joblib.dump(best_model, "adult_income_pipeline.joblib")
```

- Write a tiny function or script that:
  - Takes a small dictionary of features for a single person.
  - Converts to DataFrame.
  - Calls `best_model.predict` and `predict_proba`.

### 4. Reflection

- Which part of the end-to-end build felt most “fragile” or easy to break?
- If you had to hand this to a co-worker, what would you improve (clearer code, comments, config, etc.)?

---

## Day 7 – Apply Pipelines to Another Dataset (Titanic or House Prices) & Weekly Review

**Goal:** Generalize your pipeline skills by building a **full pipeline** on a different dataset (Titanic or House Prices), then reflect on your progress.

### 1. Core study (45–60 min)

Create `week5_day7_second_dataset_pipeline.ipynb`.

Pick **either**:

- Titanic (classification), or
- House Prices (regression, but now with both numeric & some categorical features)

For illustration, assume **Titanic**.

1. **Rebuild Titanic pipeline from scratch**

Steps:

- Load data.
- Simple imputation of `Age`, `Embarked`, etc. (or use `SimpleImputer` inside pipeline).
- Define target = `Survived`.
- Choose a feature set (numeric + categorical).
- Split train/test.
- Build:

  ```python
  numeric_features = [...]
  categorical_features = [...]

  numeric_transformer = Pipeline([...])
  categorical_transformer = Pipeline([...])
  preprocessor = ColumnTransformer([...])
  ```

- Build **two pipelines**:
  - LogisticRegression
  - RandomForestClassifier (with some reasonable default params)

2. **Evaluate and (light) tune**

Use `cross_val_score` or a **small** `GridSearchCV` (e.g., 1–2 params each).

### 2. Core practice (30–45 min)

1. **Write a mini-comparison report (markdown)**

For this second dataset, summarize:

- Baseline (majority) performance.
- Best pipeline model and its CV performance.
- Final test metrics.
- One or two key feature insights (via coefficients or feature importances).

2. **Weekly self-quiz**

Without looking back (at first), answer in your own words:

- What do `Pipeline` and `ColumnTransformer` each do?
- What problem does `handle_unknown="ignore"` solve?
- How do you specify a hyperparameter in a pipeline for `GridSearchCV`?
- What is data leakage, and give one concrete example.

Then check against your code and notes.

### 3. Stretch (optional)

- For House Prices: build a **regression pipeline** with imputation, scaling, one-hot encoding of a couple of categorical features (e.g., `Neighborhood`, `HouseStyle`), and tune a RandomForestRegressor.
- Compare to your original Week 3 linear regression without pipelines.

### 4. Reflection

- On a scale from 1–10, how confident do you feel now about:
  - Using `Pipeline` for numeric-only data?
  - Using `ColumnTransformer` for mixed-type data?
  - Doing basic hyperparameter tuning with `GridSearchCV`?
- What is the **single biggest bottleneck** right now (e.g., debugging pipeline errors, remembering parameter names, understanding metrics)?  
  This will guide what to emphasize in Week 6.

---

If you’d like, next I can generate a **Week 6 (Day 1–7)** plan focused on **unsupervised learning (clustering & PCA)** using the same pipeline mindset, plus another small project.
