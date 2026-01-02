Here is your **Week 3 (Day 1–Day 7)** plan.  
Theme: **“From EDA to First Real Models (Regression + Classification)”**

Assume ~1–2 hours/day.  
You’ll work with **two datasets** this week:

- **Regression project**: Kaggle **House Prices** (`train.csv`)  
- **Classification**: your **Titanic** dataset from Week 2

Each day has:
- **Goal**
- **Core study (learn)**
- **Core practice (do)**
- **Stretch (optional)**  
- **Reflection**

Try to:
- Write down **predictions** before running models/plots.
- Use **docs / search** instead of copy-pasting answers.
- Treat errors as data: ask *“what is this error telling me?”*

---

## Day 1 – Intro to scikit-learn & Simple Regression on Toy Data

**Goal:** Understand the basic **scikit-learn workflow** and fit your very first regression model on a simple synthetic dataset.

### 1. Core study (45–60 min)

Create `week3_day1_sklearn_intro.ipynb`.

1. **Install / import scikit-learn**
   - If not installed:
     ```bash
     pip install scikit-learn
     ```
   - In the notebook:
     ```python
     import numpy as np
     import pandas as pd
     import matplotlib.pyplot as plt

     from sklearn.linear_model import LinearRegression
     from sklearn.model_selection import train_test_split
     from sklearn.metrics import mean_squared_error, r2_score
     ```

2. **Generate a toy linear dataset**
   - Imagine a simple relationship: `y = 3x + 5 + noise`.
   ```python
   np.random.seed(42)
   X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
   y = 3 * X[:, 0] + 5 + np.random.randn(100)  # add noise
   ```
   - Plot:
     ```python
     plt.scatter(X, y)
     plt.xlabel("x")
     plt.ylabel("y")
     plt.title("Toy linear data")
     plt.show()
     ```

3. **Train/test split**
   ```python
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   ```

4. **Fit Linear Regression model**
   ```python
   model = LinearRegression()
   model.fit(X_train, y_train)
   ```

5. **Inspect model parameters**
   ```python
   model.coef_, model.intercept_
   ```
   - Compare to the true underlying relationship `3x + 5`.

6. **Evaluate model**
   ```python
   y_pred = model.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   rmse = mse ** 0.5
   r2 = r2_score(y_test, y_pred)
   rmse, r2
   ```

7. **Plot fitted line**
   ```python
   X_line = np.linspace(0, 2, 100).reshape(-1, 1)
   y_line = model.predict(X_line)

   plt.scatter(X_train, y_train, label="Train data", alpha=0.7)
   plt.scatter(X_test, y_test, label="Test data", alpha=0.7)
   plt.plot(X_line, y_line, color="red", label="Model prediction")
   plt.legend()
   plt.show()
   ```

### 2. Core practice (30–45 min)

1. **Different noise levels**
   - Regenerate data with **higher noise**:
     ```python
     X2 = 2 * np.random.rand(100, 1)
     y2 = 3 * X2[:, 0] + 5 + 5 * np.random.randn(100)  # more noise
     ```
   - Repeat train/test split, fit, and evaluation.
   - Compare RMSE and R² to the original.

2. **Change sample size**
   - Try with only 20 samples:
     ```python
     X_small = 2 * np.random.rand(20, 1)
     y_small = 3 * X_small[:, 0] + 5 + np.random.randn(20)
     ```
   - Fit and evaluate.
   - How stable are the estimated slope and intercept now?

### 3. Stretch (optional)

- Fit a **“wrong” model**:
  - Use only **two training samples** (very small train set).
  - Fit linear regression and evaluate on the rest.
  - Observe how wildly parameters and predictions vary.

### 4. Reflection

- How close did the model’s slope and intercept get to the true values 3 and 5?
- What effect did **more noise** and **less data** have on RMSE and R²?

---

## Day 2 – Real Regression: House Prices – Data Prep & First Model

**Goal:** Build your **first real regression model** to predict **house prices** with Linear Regression.

### 1. Core study (45–60 min)

Create `week3_day2_houseprices_intro.ipynb`.

1. **Get the dataset**
   - From Kaggle: “House Prices: Advanced Regression Techniques”.
   - Download `train.csv` into your `week3/` folder.

2. **Load and inspect**
   ```python
   df = pd.read_csv("train.csv")
   df.shape
   df.head()
   df.info()
   df["SalePrice"].describe()
   ```

3. **Choose a small numeric feature set**
   - Start simple (for example):
     ```python
     target = "SalePrice"
     feature_cols = [
         "OverallQual",  # overall material and finish quality
         "GrLivArea",    # above ground living area (sq ft)
         "GarageCars",   # size of garage in car capacity
         "TotalBsmtSF",  # total basement area
         "YearBuilt",    # original construction date
     ]
     ```
   - Check missingness:
     ```python
     df[feature_cols + [target]].isnull().sum()
     ```

4. **Simple cleaning**
   - For now, **drop rows** with missing values in these columns:
     ```python
     df_model = df[feature_cols + [target]].dropna()
     df_model.shape
     ```

5. **Define X and y, train/test split**
   ```python
   X = df_model[feature_cols]
   y = df_model[target]

   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )
   ```

6. **Fit Linear Regression**
   ```python
   from sklearn.linear_model import LinearRegression
   from sklearn.metrics import mean_squared_error, r2_score

   reg = LinearRegression()
   reg.fit(X_train, y_train)

   y_pred = reg.predict(X_test)
   mse = mean_squared_error(y_test, y_pred)
   rmse = mse ** 0.5
   r2 = r2_score(y_test, y_pred)
   rmse, r2
   ```

7. **Inspect coefficients**
   ```python
   coef_df = pd.DataFrame({
       "feature": feature_cols,
       "coefficient": reg.coef_
   })
   coef_df
   ```

### 2. Core practice (30–45 min)

1. **Compare train vs test performance**
   ```python
   y_train_pred = reg.predict(X_train)
   rmse_train = mean_squared_error(y_train, y_train_pred) ** 0.5
   r2_train = r2_score(y_train, y_train_pred)
   rmse_test = rmse
   r2_test = r2

   rmse_train, r2_train, rmse_test, r2_test
   ```
   - Write down:
     - Is train performance **much better** than test?
     - Any early signs of overfitting?

2. **Interpret coefficients**
   - For each feature, reason:
     - Is a **positive** or **negative** relationship with price expected?
     - Do the signs and relative magnitudes of coefficients match your intuition?

### 3. Stretch (optional)

- Create a scatter plot of `GrLivArea` vs `SalePrice`:
  ```python
  plt.scatter(df_model["GrLivArea"], df_model["SalePrice"], alpha=0.5)
  plt.xlabel("GrLivArea")
  plt.ylabel("SalePrice")
  plt.show()
  ```
- Identify if any **obvious outliers** exist (e.g., very large area but relatively low price).

### 4. Reflection

- Are your first model’s RMSE and R² **better than a naive baseline** (like predicting the mean price)?  
  (You’ll quantify this tomorrow.)
- Which feature seems most influential from the coefficients?

---

## Day 3 – Improving Regression: Baselines, Residuals, and Feature Tweaks

**Goal:** Compare your model to a **naive baseline**, inspect **residuals**, and try small improvements.

### 1. Core study (45–60 min)

Continue in `week3_day3_houseprices_improve.ipynb`.

1. **Compute a naive baseline**
   - Baseline: always predict the **mean SalePrice** from the training set.
   ```python
   baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)

   baseline_mse = mean_squared_error(y_test, baseline_pred)
   baseline_rmse = baseline_mse ** 0.5
   baseline_rmse
   ```
   - Compare baseline RMSE to your model’s test RMSE.

2. **Residual analysis**
   - Residuals = actual – predicted:
     ```python
     residuals = y_test - y_pred
     ```
   - Plot histogram:
     ```python
     plt.hist(residuals, bins=30)
     plt.title("Residuals distribution")
     plt.show()
     ```
   - Scatter residuals vs predicted:
     ```python
     plt.scatter(y_pred, residuals, alpha=0.5)
     plt.axhline(0, color="red")
     plt.xlabel("Predicted SalePrice")
     plt.ylabel("Residual (actual - predicted)")
     plt.show()
     ```

3. **Try adding/removing features**
   - Add 1–2 more numeric features you think might help (e.g., `GarageArea`, `1stFlrSF`).
   - Or remove one feature you suspect is noisy.

### 2. Core practice (30–45 min)

1. **Systematic comparison**
   - Train **two models**:
     - Model A: original 5 features.
     - Model B: with 1–2 added (or one removed).
   - For each:
     - Compute train RMSE/R² and test RMSE/R².
     - Compare in a small DataFrame:
       ```python
       results = pd.DataFrame({
           "model": ["A_original", "B_modified"],
           "rmse_train": [...],
           "rmse_test": [...],
           "r2_train": [...],
           "r2_test": [...],
       })
       results
       ```

2. **Interpret residual patterns**
   - From your residual plots:
     - Do residuals look roughly centered around 0?
     - Any pattern with predicted price (e.g., underpredicting expensive houses)?

### 3. Stretch (optional)

- Log-transform the target:
  ```python
  df_model["SalePrice_log"] = np.log1p(df_model["SalePrice"])
  ```
- Redefine y as `SalePrice_log`, retrain the model, compare:
  - RMSE in log-space
  - Also compare the distribution of residuals for log vs raw target.

### 4. Reflection

- How much **better** is your linear regression than the mean-baseline?
- Does adding more features always help? What did you observe?

---

## Day 4 – Logistic Regression: From Regression to Classification (Titanic)

**Goal:** Move from regression to **classification** using **logistic regression** on your Titanic dataset.

### 1. Core study (45–60 min)

Create `week3_day4_titanic_logreg.ipynb`.

Use the **imputed Titanic DataFrame** from Week 2 (or recreate the imputation quickly if needed).

1. **Recall target and features**
   ```python
   target = "Survived"
   base_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
   df = df_imputed.copy()  # from Week 2 notebook
   df_model = df[base_features + [target]].dropna()
   ```

2. **One-hot encode categorical variables**
   - A quick but OK approach for now:
     ```python
     X = pd.get_dummies(df_model[base_features], drop_first=True)
     y = df_model[target]
     X.head()
     X.shape
     ```

3. **Train/test split**
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )
   ```

4. **Fit Logistic Regression**
   ```python
   from sklearn.linear_model import LogisticRegression

   log_reg = LogisticRegression(max_iter=1000)
   log_reg.fit(X_train, y_train)
   ```

5. **Predict and compute accuracy**
   ```python
   from sklearn.metrics import accuracy_score

   y_pred = log_reg.predict(X_test)
   acc = accuracy_score(y_test, y_pred)
   acc
   ```

6. **Compare with baselines**
   - Majority-class baseline (from Week 2) and the simple rule (`Sex == female → survive`) if you built it.

### 2. Core practice (30–45 min)

1. **Train vs test accuracy**
   ```python
   y_train_pred = log_reg.predict(X_train)
   acc_train = accuracy_score(y_train, y_train_pred)
   acc_test = accuracy_score(y_test, y_pred)
   acc_train, acc_test
   ```
   - Are they similar? Large gap might indicate overfitting.

2. **Coefficient interpretation (basic)**
   ```python
   coef_df = pd.DataFrame({
       "feature": X.columns,
       "coefficient": log_reg.coef_[0]
   }).sort_values("coefficient", ascending=False)
   coef_df.head(10)
   coef_df.tail(10)
   ```
   - Positive coefficient → increases log-odds of survival.
   - Negative → decreases.

### 3. Stretch (optional)

- Compute predicted **probabilities** instead of just class labels:
  ```python
  y_proba = log_reg.predict_proba(X_test)[:, 1]  # probability of Survived = 1
  y_proba[:10]
  ```
- Check a few examples where predicted probability is ~0.5 vs very close to 0 or 1.

### 4. Reflection

- How much better is logistic regression accuracy vs:
  - Majority-class baseline?
  - Simple rule-based baseline (if used)?
- Which features have the largest positive and negative influence according to the coefficients? Do they match your intuition?

---

## Day 5 – Classification Metrics: Confusion Matrix, Precision, Recall, F1, ROC-AUC

**Goal:** Go beyond accuracy and understand **what kinds of mistakes** your classifier makes.

### 1. Core study (45–60 min)

Continue in `week3_day5_titanic_metrics.ipynb`.

1. **Confusion matrix**
   ```python
   from sklearn.metrics import confusion_matrix, classification_report

   cm = confusion_matrix(y_test, y_pred)
   cm
   ```
   - Interpret:
     - `cm[0,0]`: true negatives (died predicted died)
     - `cm[0,1]`: false positives (died predicted survived)
     - `cm[1,0]`: false negatives (survived predicted died)
     - `cm[1,1]`: true positives (survived predicted survived)

2. **Precision, recall, F1**
   ```python
   from sklearn.metrics import precision_score, recall_score, f1_score

   precision = precision_score(y_test, y_pred)
   recall = recall_score(y_test, y_pred)
   f1 = f1_score(y_test, y_pred)
   precision, recall, f1
   ```

3. **Classification report**
   ```python
   print(classification_report(y_test, y_pred))
   ```

4. **ROC curve and AUC**
   ```python
   from sklearn.metrics import roc_curve, roc_auc_score

   y_proba = log_reg.predict_proba(X_test)[:, 1]

   fpr, tpr, thresholds = roc_curve(y_test, y_proba)
   auc = roc_auc_score(y_test, y_proba)
   auc
   ```

   - Plot ROC:
     ```python
     plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
     plt.plot([0, 1], [0, 1], "k--", label="Random")
     plt.xlabel("False Positive Rate")
     plt.ylabel("True Positive Rate")
     plt.legend()
     plt.show()
     ```

### 2. Core practice (30–45 min)

1. **Manual reasoning about confusion matrix**
   - From `cm`, compute:
     - Total predictions.
     - Overall accuracy (check it matches `accuracy_score`).
   - In markdown:
     - How many **false negatives**? Interpret what that means in Titanic terms.

2. **Threshold tuning (basic)**
   - Instead of default 0.5 threshold, try 0.3 and 0.7:
     ```python
     y_pred_03 = (y_proba >= 0.3).astype(int)
     y_pred_07 = (y_proba >= 0.7).astype(int)

     for name, preds in [("0.3", y_pred_03), ("0.5", y_pred), ("0.7", y_pred_07)]:
         print("Threshold", name)
         print("Accuracy:", accuracy_score(y_test, preds))
         print("Precision:", precision_score(y_test, preds))
         print("Recall:", recall_score(y_test, preds))
         print("F1:", f1_score(y_test, preds))
         print()
     ```
   - Observe trade-offs: precision vs recall.

### 3. Stretch (optional)

- Plot **precision-recall curve**:
  ```python
  from sklearn.metrics import precision_recall_curve

  precs, recs, ths = precision_recall_curve(y_test, y_proba)
  plt.plot(recs, precs)
  plt.xlabel("Recall")
  plt.ylabel("Precision")
  plt.title("Precision-Recall Curve")
  plt.show()
  ```

### 4. Reflection

- In the Titanic context, would you rather:
  - Favor **higher recall** (catch more survivors at cost of more false alarms)?
  - Or **higher precision** (fewer “wrong” survivors predicted)?
- Which threshold (0.3, 0.5, 0.7) would you choose and why?

---

## Day 6 – Model Comparison: Logistic Regression vs kNN

**Goal:** Compare **two different classifiers** on the same problem and see how hyperparameters affect performance.

### 1. Core study (45–60 min)

Create `week3_day6_titanic_knn.ipynb`.

1. **Import and fit kNN**
   ```python
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.metrics import accuracy_score

   # Reuse X_train, X_test, y_train, y_test from previous days
   knn = KNeighborsClassifier(n_neighbors=5)
   knn.fit(X_train, y_train)
   y_pred_knn = knn.predict(X_test)
   acc_knn = accuracy_score(y_test, y_pred_knn)
   acc_knn
   ```

2. **Compare with logistic regression**
   - Record:
     - Logistic regression: accuracy, precision, recall, F1.
     - kNN: same metrics.

3. **Effect of k (number of neighbors)**
   - Try different k values:
     ```python
     ks = [1, 3, 5, 7, 9, 11, 15]
     scores = []

     for k in ks:
         knn = KNeighborsClassifier(n_neighbors=k)
         knn.fit(X_train, y_train)
         y_pred_k = knn.predict(X_test)
         scores.append(accuracy_score(y_test, y_pred_k))

     scores
     ```
   - Plot:
     ```python
     plt.plot(ks, scores, marker="o")
     plt.xlabel("k (n_neighbors)")
     plt.ylabel("Test accuracy")
     plt.title("kNN accuracy vs k")
     plt.show()
     ```

### 2. Core practice (30–45 min)

1. **Train vs test performance for kNN**
   - For a few k values (e.g., 1, 5, 15), compute both **train** and **test** accuracy:
     ```python
     for k in [1, 5, 15]:
         knn = KNeighborsClassifier(n_neighbors=k)
         knn.fit(X_train, y_train)
         y_train_pred = knn.predict(X_train)
         y_test_pred = knn.predict(X_test)
         print("k =", k)
         print("train acc:", accuracy_score(y_train, y_train_pred))
         print("test acc:", accuracy_score(y_test, y_test_pred))
         print()
     ```
   - Observe:
     - k=1 often → very high train accuracy, potentially lower test accuracy (overfitting).
     - Larger k → smoother, more general.

2. **Write a short comparison**
   - In markdown, compare LogisticRegression vs kNN:
     - Which performs better on test accuracy?
     - Which seems more stable (less overfit)?

### 3. Stretch (optional)

- Normalize features for kNN:
  ```python
  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  knn_scaled = KNeighborsClassifier(n_neighbors=5)
  knn_scaled.fit(X_train_scaled, y_train)
  accuracy_score(y_test, knn_scaled.predict(X_test_scaled))
  ```
- Compare accuracy before vs after scaling.

### 4. Reflection

- How does changing **k** illustrate the idea of **bias vs variance** (complex vs simple models)?
- Which model would you pick as your **default baseline classifier** for Titanic now, and why?

---

## Day 7 – Week 3 Wrap-Up: Mini Regression & Classification Reports

**Goal:** Consolidate what you’ve built into **two mini-project summaries** and reflect on what you learned.

### 1. Core study (45–60 min)

Create `week3_day7_summary.ipynb`.

1. **Regression summary (House Prices)**
   - Copy key results from Days 2–3:
     - Feature list
     - Baseline RMSE (predicting mean)
     - Best model’s train/test RMSE & R²
     - Short table of results if you created one

2. **Classification summary (Titanic)**
   - Copy key results from Days 4–6:
     - Baseline accuracy (majority class)
     - Logistic Regression metrics
     - kNN metrics for best k
     - One confusion matrix and ROC AUC

### 2. Core practice (30–45 min)

In `week3_day7_summary.ipynb`:

1. **Write a short regression report (8–12 sentences)**
   Sections (in markdown):

   - **Problem**: “Predict house sale price from property attributes.”
   - **Data**: Briefly describe the dataset and features used.
   - **Model**: Linear Regression; which features you included.
   - **Baseline vs model**: Compare baseline RMSE to model RMSE.
   - **Insights**:
     - Which features are most important (coefficients)?
     - Any limitations (e.g., non-linearity, outliers)?

2. **Write a short classification report (8–12 sentences)**
   Sections:

   - **Problem**: “Predict Titanic passenger survival.”
   - **Data & features**: Summarize key columns and preprocessing (one-hot).
   - **Models & metrics**:
     - Baseline accuracy vs Logistic Regression vs kNN.
     - Mention at least accuracy, precision, recall, and AUC for your chosen best model.
   - **Insights**:
     - Which features most strongly influence survival?
     - What types of errors (false positives/negatives) are most common?

3. **Self-quiz (no code)**
   In markdown, answer:

   - What is the difference between **regression** and **classification**?
   - What are **RMSE** and **R²** measuring?
   - What do **precision** and **recall** measure in classification?

### 3. Stretch (optional)

- List **2–3 concrete improvements** you’d try next for each project:
  - Regression: e.g., more features, better outlier handling, log-transform target.
  - Classification: e.g., more features, more flexible models (trees), cross-validation.

### 4. Reflection

- What was the **hardest conceptual step** this week (e.g., interpreting coefficients, understanding ROC)?
- If you had only **one metric** for regression and one for classification, which would you choose and why?
- On a scale of 1–10, how confident do you feel now about:
  - Fitting a basic regression model?
  - Fitting a basic classifier?

---

If you want, next I can outline **Week 4 (Day 1–7)** focused on tree-based models, cross-validation, and starting to work more systematically with pipelines.
