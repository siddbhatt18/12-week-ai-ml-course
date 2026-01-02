Week 4 will move you into **tree-based models** and **better model evaluation**.  
Theme: **Decision Trees → Random Forests → Cross-Validation → Small Tree-Based Mini-Project**.

Assume ~1–2 hours/day.

You’ll mainly reuse:

- **Titanic (classification)** – from Weeks 2–3  
- **House Prices (regression)** – from Week 3  
- Plus one new small dataset at the end of the week (for a mini-project).

Each day: **Goal → Core study → Core practice → Stretch → Reflection**.  
Try to keep a small “results table” notebook as you go (models + metrics).

---

## Day 1 – Decision Trees for Classification (Titanic)

**Goal:** Understand and train a **DecisionTreeClassifier** on Titanic and see how depth affects overfitting.

### 1. Core study (45–60 min)

Create `week4_day1_titanic_tree.ipynb`.

1. **Load your Titanic modeling data**

If you still have `df_imputed` from Week 2–3, reuse it. If not, quickly recreate:

```python
import pandas as pd
import numpy as np

# Load data (adjust path if needed)
df = pd.read_csv("train.csv")

# Simple imputation example (you can reuse your own choices)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df_imputed = df.copy()
```

Define features/target (same as Week 3):

```python
target = "Survived"
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

df_model = df_imputed[features + [target]].dropna()
X = pd.get_dummies(df_model[features], drop_first=True)
y = df_model[target]
```

Train/test split:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

2. **Train a basic Decision Tree**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

tree_clf = DecisionTreeClassifier(random_state=42)  # default depth
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)

y_train_pred = tree_clf.predict(X_train)
acc_train = accuracy_score(y_train, y_train_pred)

acc_train, acc_test
```

3. **Inspect tree complexity**

```python
tree_clf.get_depth(), tree_clf.get_n_leaves()
```

Note how deep the tree is.

4. **Try controlled depth**

```python
tree_clf_3 = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf_3.fit(X_train, y_train)

y_pred_3 = tree_clf_3.predict(X_test)
acc_train_3 = accuracy_score(y_train, tree_clf_3.predict(X_train))
acc_test_3 = accuracy_score(y_test, y_pred_3)

acc_train_3, acc_test_3
```

### 2. Core practice (30–45 min)

1. **Depth vs performance table**

Try a few depths: `[2, 3, 4, 5, None]` (None = unlimited):

```python
from sklearn.metrics import accuracy_score

depths = [2, 3, 4, 5, None]
results = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    results.append({"max_depth": d, "train_acc": train_acc, "test_acc": test_acc})

import pandas as pd
res_df = pd.DataFrame(results)
res_df
```

Look for patterns: when do you see clear overfitting?

2. **Gini vs Entropy**

Try `criterion="entropy"` for one depth:

```python
clf_gini = DecisionTreeClassifier(max_depth=3, criterion="gini", random_state=42)
clf_entropy = DecisionTreeClassifier(max_depth=3, criterion="entropy", random_state=42)

for name, clf in [("gini", clf_gini), ("entropy", clf_entropy)]:
    clf.fit(X_train, y_train)
    print(name, "train_acc:", accuracy_score(y_train, clf.predict(X_train)),
          "test_acc:", accuracy_score(y_test, clf.predict(X_test)))
```

### 3. Stretch (optional)

- Visualize the shallow tree (`max_depth=3`):

```python
from sklearn import tree
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 6))
tree.plot_tree(
    clf_gini,
    feature_names=X.columns,
    class_names=["Died", "Survived"],
    filled=True,
    max_depth=3
)
plt.show()
```

Try to interpret at least the **first two splits** in plain language.

### 4. Reflection

- For which `max_depth` do you get the **best test accuracy**? How does train accuracy compare?
- Do deep trees seem to overfit? How can you tell?

---

## Day 2 – Decision Trees for Regression (House Prices)

**Goal:** Use **DecisionTreeRegressor** on House Prices and see how depth affects RMSE and overfitting.

### 1. Core study (45–60 min)

Create `week4_day2_houseprices_tree.ipynb`.

1. **Load House Prices data**

```python
import pandas as pd
import numpy as np

df = pd.read_csv("train.csv")
target = "SalePrice"

feature_cols = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "YearBuilt",
]

df_model = df[feature_cols + [target]].dropna()
X = df_model[feature_cols]
y = df_model[target]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

2. **Fit DecisionTreeRegressor**

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

tree_reg = DecisionTreeRegressor(random_state=42)  # default unlimited depth
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)
rmse_test = mean_squared_error(y_test, y_pred, squared=False)
r2_test = r2_score(y_test, y_pred)

y_train_pred = tree_reg.predict(X_train)
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
r2_train = r2_score(y_train, y_train_pred)

rmse_train, r2_train, rmse_test, r2_test
```

3. **Repeat for controlled depths**

```python
depths = [2, 4, 6, 8, 10, None]
results = []

for d in depths:
    reg = DecisionTreeRegressor(max_depth=d, random_state=42)
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    results.append({
        "max_depth": d,
        "rmse_train": mean_squared_error(y_train, y_train_pred, squared=False),
        "rmse_test": mean_squared_error(y_test, y_test_pred, squared=False),
        "r2_train": r2_score(y_train, y_train_pred),
        "r2_test": r2_score(y_test, y_test_pred),
    })

res_df = pd.DataFrame(results)
res_df
```

### 2. Core practice (30–45 min)

1. **Plot depth vs RMSE**

```python
import matplotlib.pyplot as plt

plt.plot(res_df["max_depth"].astype(str), res_df["rmse_train"], label="Train RMSE")
plt.plot(res_df["max_depth"].astype(str), res_df["rmse_test"], label="Test RMSE")
plt.xlabel("max_depth")
plt.ylabel("RMSE")
plt.legend()
plt.title("Decision Tree depth vs RMSE")
plt.show()
```

2. **Interpret the plot**

In a markdown cell, explain:

- At what depth does **test RMSE** stop improving or start worsening?
- What does the gap between train and test RMSE tell you about **overfitting**?

### 3. Stretch (optional)

- Compare to your **linear regression** from Week 3: which has lower test RMSE?
- Try adding 1–2 more numeric features and see if the **best-depth** changes.

### 4. Reflection

- Do trees perform better or worse than your linear regression on this subset? Any guess why?
- What’s one advantage and one disadvantage of decision trees that you’ve observed?

---

## Day 3 – Random Forests for Classification (Titanic)

**Goal:** Learn **RandomForestClassifier**, compare to a single tree and logistic regression, and examine feature importance.

### 1. Core study (45–60 min)

Create `week4_day3_titanic_random_forest.ipynb`.

1. **Reuse Titanic training split**

From Day 1 (`X_train`, `X_test`, `y_train`, `y_test`).

2. **Fit RandomForestClassifier**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)
acc_rf_test = accuracy_score(y_test, y_pred_rf)
acc_rf_test
```

3. **Compare with single tree and logistic regression**

If you still have those models from Week 3/Day 1, reuse their metrics; otherwise quickly re-train logistic regression:

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
acc_log = accuracy_score(y_test, log_reg.predict(X_test))
acc_log
```

Summarize in a small DataFrame:

```python
results = pd.DataFrame([
    {"model": "LogisticRegression", "test_acc": acc_log},
    {"model": "RandomForest", "test_acc": acc_rf_test},
])
results
```

4. **Feature importance**

```python
import pandas as pd
feat_importances = pd.Series(
    rf_clf.feature_importances_, index=X.columns
).sort_values(ascending=False)
feat_importances.head(10)
```

Plot:

```python
feat_importances.head(10).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 10 feature importances (RandomForest)")
plt.show()
```

### 2. Core practice (30–45 min)

1. **Overfitting check**

Compute train accuracy:

```python
acc_rf_train = accuracy_score(y_train, rf_clf.predict(X_train))
acc_rf_train, acc_rf_test
```

Is the train accuracy much higher than test?

2. **Hyperparameter exploration (shallow)**

Try different `max_depth` and `n_estimators`:

```python
configs = [
    {"n_estimators": 50,  "max_depth": 5},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 10},
]

rows = []
for cfg in configs:
    clf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    rows.append({
        "n_estimators": cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "train_acc": accuracy_score(y_train, clf.predict(X_train)),
        "test_acc": accuracy_score(y_test, clf.predict(X_test)),
    })

pd.DataFrame(rows)
```

### 3. Stretch (optional)

- For the best configuration, recompute more metrics (precision, recall, F1, ROC-AUC) and compare to logistic regression.

### 4. Reflection

- Which model gives you the **best test accuracy** on Titanic so far?
- From the feature importance plot, which 2–3 features seem most important? Are they the ones you expected?

---

## Day 4 – Random Forests for Regression (House Prices)

**Goal:** Use **RandomForestRegressor** for House Prices and compare to Linear Regression and Decision Trees.

### 1. Core study (45–60 min)

Create `week4_day4_houseprices_random_forest.ipynb`.

1. **Reuse House Prices modeling split**

From Day 2 (`X_train`, `X_test`, `y_train`, `y_test`).

2. **Fit RandomForestRegressor**

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

rf_reg = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_train)

y_pred_rf = rf_reg.predict(X_test)
rmse_rf_test = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf_test = r2_score(y_test, y_pred_rf)

y_pred_rf_train = rf_reg.predict(X_train)
rmse_rf_train = mean_squared_error(y_train, y_pred_rf_train, squared=False)
r2_rf_train = r2_score(y_train, y_pred_rf_train)

rmse_rf_train, r2_rf_train, rmse_rf_test, r2_rf_test
```

3. **Compare against previous models**

If you saved them:

- Linear Regression (Week 3, Day 2–3)
- Best Decision Tree (Week 4, Day 2)

Create a results table (fill in with your real values):

```python
results = pd.DataFrame([
    {"model": "Baseline_mean", "rmse_test": ...},
    {"model": "LinearRegression", "rmse_test": ...},
    {"model": "DecisionTree_best", "rmse_test": ...},
    {"model": "RandomForest", "rmse_test": rmse_rf_test},
])
results
```

4. **Random Forest feature importances**

```python
feat_importances = pd.Series(
    rf_reg.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
feat_importances
```

Plot top:

```python
feat_importances.head(10).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 10 feature importances (House Prices RF)")
plt.show()
```

### 2. Core practice (30–45 min)

1. **Small hyperparameter grid**

Try changing `max_depth` / `min_samples_leaf`:

```python
configs = [
    {"max_depth": None, "min_samples_leaf": 1},
    {"max_depth": 10,   "min_samples_leaf": 1},
    {"max_depth": 10,   "min_samples_leaf": 5},
    {"max_depth": 20,   "min_samples_leaf": 5},
]

rows = []

for cfg in configs:
    reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        random_state=42,
        n_jobs=-1
    )
    reg.fit(X_train, y_train)
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    rows.append({
        "max_depth": cfg["max_depth"],
        "min_samples_leaf": cfg["min_samples_leaf"],
        "rmse_train": mean_squared_error(y_train, y_train_pred, squared=False),
        "rmse_test": mean_squared_error(y_test, y_test_pred, squared=False),
    })

pd.DataFrame(rows)
```

2. **Interpret results**

- Which config gives the **lowest test RMSE**?
- Does increasing `min_samples_leaf` reduce overfitting?

### 3. Stretch (optional)

- Scatter plot **predicted vs actual** sale prices:
  ```python
  plt.scatter(y_test, y_pred_rf, alpha=0.5)
  plt.xlabel("Actual SalePrice")
  plt.ylabel("Predicted SalePrice")
  plt.plot([y_test.min(), y_test.max()],
           [y_test.min(), y_test.max()], "r--")
  plt.title("RandomForest: Actual vs Predicted")
  plt.show()
  ```

### 4. Reflection

- How does the Random Forest compare to Linear Regression and a single tree?
- Does it seem to overfit heavily, or is the gap between train/test reasonable?

---

## Day 5 – Cross-Validation & More Robust Model Comparison

**Goal:** Learn **k-fold cross-validation** and use it to compare models more reliably.

### 1. Core study (45–60 min)

Create `week4_day5_cross_validation.ipynb`.

You can use **Titanic (classification)** for this.

1. **Set up data again**

Recreate or reuse `X`, `y` for Titanic (with `get_dummies`).

2. **Basic cross_val_score**

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

log_reg = LogisticRegression(max_iter=1000)
rf_clf = RandomForestClassifier(
    n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
)
```

Compute 5-fold **cross-validated accuracy**:

```python
scores_log = cross_val_score(log_reg, X, y, cv=5, scoring="accuracy")
scores_rf = cross_val_score(rf_clf, X, y, cv=5, scoring="accuracy")

scores_log, scores_log.mean(), scores_log.std()
scores_rf, scores_rf.mean(), scores_rf.std()
```

3. **Conceptual points**

- CV gives a **distribution** of scores, not just one.
- Mean ≈ expected performance; std ≈ stability/variance.

### 2. Core practice (30–45 min)

1. **Add Decision Tree to comparison**

```python
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)

models = {
    "LogReg": log_reg,
    "DecisionTree": tree_clf,
    "RandomForest": rf_clf
}

rows = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    rows.append({
        "model": name,
        "cv_mean_acc": scores.mean(),
        "cv_std_acc": scores.std(),
    })

pd.DataFrame(rows)
```

2. **Try different CV folds**

- Compare `cv=3`, `cv=5`, `cv=10` for one model (e.g., RandomForest).  
  Note how mean and std change.

### 3. Stretch (optional)

- Use another scoring metric (e.g., ROC-AUC):

```python
scores_rf_auc = cross_val_score(
    rf_clf, X, y, cv=5, scoring="roc_auc"
)
scores_rf_auc.mean(), scores_rf_auc.std()
```

- Compare **accuracy vs AUC** for the same model.

### 4. Reflection

- Based on CV means, which classifier do you prefer for Titanic?
- Do CV results roughly match your earlier single train/test split impressions?

---

## Day 6 – New Dataset Mini-Project (Part 1): Adult Income EDA & Baseline

**Goal:** Start a **small tabular project** on a new dataset: Adult Income (predict if income >50K).  
You’ll do EDA + baseline and prepare for tree-based models.

### 1. Core study (45–60 min)

Create `week4_day6_adult_income_intro.ipynb`.

1. **Get dataset**

- Kaggle: search for **“Adult Census Income”** or similar.
- Download the CSV (e.g., `adult.csv`) to your `week4/` folder.

2. **Load and inspect**

```python
import pandas as pd
df = pd.read_csv("adult.csv")  # adjust filename if needed

df.head()
df.shape
df.info()
```

Identify:

- Target column (often `income`, e.g. `<=50K` / `>50K`).
- Feature columns: age, education, occupation, hours-per-week, etc.

3. **Clean missing markers**

Adult often uses `"?"` as missing:

```python
df.replace("?", pd.NA, inplace=True)
df.isnull().sum().sort_values(ascending=False).head(10)
```

4. **Define target & basic encoding**

Assume target column is `income`:

```python
target = "income"

df[target].value_counts()
```

Create binary target:

```python
df["income_binary"] = (df[target].str.contains(">50K")).astype(int)
```

5. **Initial feature selection**

Pick a moderate set, e.g.:

```python
features = [
    "age", "education-num", "hours-per-week", "capital-gain",
    "capital-loss", "workclass", "marital-status", "occupation",
    "relationship", "sex", "native-country"
]

df_model = df[features + ["income_binary"]].dropna()
X = pd.get_dummies(df_model[features], drop_first=True)
y = df_model["income_binary"]
```

6. **Train/test split + baseline**

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Majority baseline
y_train.value_counts(normalize=True)
```

Compute baseline accuracy as the majority class proportion.

### 2. Core practice (30–45 min)

1. **Quick univariate EDA**

- For numeric: `age`, `hours-per-week`, `education-num`
  - `.describe()`, histogram.
- For a few categorical: `workclass`, `marital-status`
  - `value_counts()`, bar plot.

2. **First simple model: Logistic Regression**

```python
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
acc = accuracy_score(y_test, y_pred)
acc
```

Compare with baseline.

### 3. Stretch (optional)

- Compute confusion matrix and classification report for this first model.
- Note whether the model is better at catching **>50K** (positive class) or the other.

### 4. Reflection

- How imbalanced is the target (`<=50K` vs `>50K`)?
- Is the logistic regression clearly better than the majority baseline?

---

## Day 7 – New Dataset Mini-Project (Part 2): Trees, Forests & CV

**Goal:** Build **DecisionTree** and **RandomForest** models on Adult Income, evaluate with cross-validation, and summarize.

### 1. Core study (45–60 min)

Continue in `week4_day7_adult_income_models.ipynb`  
(or extend Day 6 notebook).

Assuming you have `X`, `y`, `X_train`, `X_test`, `y_train`, `y_test`.

1. **Decision Tree on Adult Income**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

tree_clf = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=50,
    random_state=42
)
tree_clf.fit(X_train, y_train)

y_pred_tree = tree_clf.predict(X_test)
acc_tree_test = accuracy_score(y_test, y_pred_tree)
acc_tree_test
```

2. **Random Forest on Adult Income**

```python
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)
acc_rf_test = accuracy_score(y_test, y_pred_rf)
acc_rf_test
```

3. **Cross-validation for model comparison**

```python
from sklearn.model_selection import cross_val_score

models = {
    "LogReg": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=50, random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
}

rows = []
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    rows.append({
        "model": name,
        "cv_mean_acc": scores.mean(),
        "cv_std_acc": scores.std(),
    })

import pandas as pd
pd.DataFrame(rows)
```

### 2. Core practice (30–45 min)

1. **Feature importance for Random Forest**

```python
feat_importances = pd.Series(
    rf_clf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

feat_importances.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 15 feature importances – Adult Income RF")
plt.show()
```

2. **Mini project summary (in markdown)**

Write a short summary (8–12 sentences) with:

- **Problem**: Predict if income is >50K based on census info.
- **Data**: Number of rows/columns used after cleaning; main feature types.
- **Baselines & models**:
  - Majority baseline accuracy.
  - Logistic Regression CV accuracy.
  - Decision Tree CV accuracy.
  - Random Forest CV accuracy.
- **Insights**:
  - Which features matter most?
  - Which model would you choose and why?

### 3. Stretch (optional)

- Try a small hyperparameter search manually for RandomForest:
  - Vary `min_samples_leaf` (`[1, 5, 10, 20]`) and record CV mean accuracy.
- Or focus on ROC-AUC instead of accuracy using `scoring="roc_auc"`.

### 4. Reflection

- Across **Titanic, House Prices, and Adult Income**, in which case do trees/forests shine most vs linear models?
- Which part of tree-based modeling feels most intuitive to you now, and which part is still murky (e.g., how splits are chosen, how feature importance is computed)?

---

If you’d like next, I can generate a **Week 5 (Day 1–7)** plan focused on **data preprocessing, scaling/encoding, and scikit-learn Pipelines + ColumnTransformer**, building on these tree and forest models in a more systematic way.
