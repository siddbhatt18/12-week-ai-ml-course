Below is a structured **Week 2 plan (Day 1–Day 7)** focusing on **EDA (Exploratory Data Analysis) and ML basics**.  
Assume ~1–2 hours per day.

We’ll work with **one main dataset all week** (recommended: **Titanic** from Kaggle).  
If you can’t use Kaggle, you can use `seaborn`’s built-in `titanic` instead.

Each day has:
- **Goal**
- **Core study (learn)**
- **Core practice (do)**
- **Stretch (optional challenge)**
- **Reflection**

Try to:
- Type everything yourself.
- Look up pandas/Matplotlib/seaborn docs when stuck.
- Before each plot or calculation, **write down what you expect** to see.

---

## Week 2 – Day 1: Choose Dataset & Understand the Problem

**Goal:** Pick your main dataset (Titanic recommended), load it, understand columns, and frame the prediction problem.

### 1. Core study (45–60 min)

Create a new notebook: `week2_day1_titanic_intro.ipynb`.

1. **Get the dataset**
   - Option A: Kaggle Titanic (preferred)
     - Go to https://www.kaggle.com/c/titanic
     - Download `train.csv` to your `week2/` folder.
   - Option B: seaborn Titanic
     ```python
     import seaborn as sns
     df = sns.load_dataset("titanic")
     ```

2. **Import libraries**
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns

   sns.set(style="whitegrid")
   ```

3. **Load and inspect the data**
   ```python
   df = pd.read_csv("train.csv")  # if using Kaggle file

   df.head()
   df.tail()
   df.shape
   df.info()
   df.describe()
   df.nunique()
   ```

4. **Understand the context (read description)**
   - If Kaggle: read the **“Data Description”** section on the Titanic competition page.
   - For each column (e.g., `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Survived`), note:
     - What does it represent?
     - Is it numeric or categorical?
   - Identify **target**:
     - Titanic: `Survived` (0 = no, 1 = yes).

### 2. Core practice (30–45 min)

In the notebook:

1. **Column overview table**
   - Create a small table (e.g., a dictionary or markdown) listing:
     - Column name
     - Data type (`df.dtypes`)
     - Short description (in your own words)
     - Whether you think it’s:
       - **numeric** or **categorical**
       - Potentially useful for prediction

2. **Quick statistics**
   - For main numeric columns (`Age`, `Fare`, `SibSp`, `Parch`):
     ```python
     df[["Age", "Fare", "SibSp", "Parch"]].describe()
     ```
   - For `Survived`:
     ```python
     df["Survived"].value_counts()
     df["Survived"].value_counts(normalize=True)
     ```

3. **Initial questions**
   - In a markdown cell, write at least **5 questions** you want to answer with EDA, e.g.:
     - Do women have a higher survival rate than men?
     - Does passenger class (`Pclass`) affect survival?
     - Does age affect survival?

### 3. Stretch (optional)

- Download and quickly inspect a **second dataset** (e.g., House Prices) and compare:
  - Which seems easier to understand?
  - Which has more missing values?

### 4. Reflection

- In 4–6 sentences, describe:
  - What the dataset is about.
  - What your **prediction goal** is (e.g., “Predict whether a passenger survived the Titanic sinking”).
  - Any immediate concerns (missing data, weird values, etc.).

---

## Week 2 – Day 2: Univariate EDA – Numeric Features

**Goal:** Understand distributions of individual numeric features and spot outliers.

### 1. Core study (45–60 min)

Create `week2_day2_univariate_numeric.ipynb`.

1. **Select numeric columns**
   ```python
   numeric_cols = df.select_dtypes(include=[np.number]).columns
   numeric_cols
   ```

2. **Descriptive stats & percentiles**
   ```python
   df[numeric_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
   ```

3. **Histograms & boxplots**
   - For `Age` and `Fare`:
     ```python
     df["Age"].hist(bins=30)
     plt.title("Age distribution")
     plt.show()

     sns.boxplot(x=df["Age"])
     plt.title("Age boxplot")
     plt.show()
     ```

4. **Interpreting distributions**
   - Note concepts:
     - **Skewness** (long tail right/left)
     - **Outliers** (values far from most data)

### 2. Core practice (30–45 min)

1. **Explore all numeric features**
   - For each numeric column:
     - Plot a histogram (`.hist()` or `sns.histplot`).
     - Look at `describe()` output.
   - In a markdown cell, for **each numeric feature**, answer:
     - Is it roughly symmetric or skewed?
     - Are there obvious outliers?
     - Are the values in a realistic range?

2. **Outlier detection (IQR rule)**
   - For `Fare`:
     ```python
     Q1 = df["Fare"].quantile(0.25)
     Q3 = df["Fare"].quantile(0.75)
     IQR = Q3 - Q1
     lower_bound = Q1 - 1.5 * IQR
     upper_bound = Q3 + 1.5 * IQR

     outliers = df[(df["Fare"] < lower_bound) | (df["Fare"] > upper_bound)]
     len(outliers), outliers["Fare"].head()
     ```
   - Interpret: are those plausible fares or data errors?

### 3. Stretch (optional)

- Choose one **skewed** feature (likely `Fare`):
  - Create a new column with log-transform:
    ```python
    df["Fare_log"] = np.log1p(df["Fare"])
    ```
  - Plot histograms for `Fare` and `Fare_log` side-by-side.
  - Which looks “more normal”? Why might this help modeling later?

### 4. Reflection

- Which numeric feature surprised you the most and why?
- If you had to remove just **one** numeric feature based purely on today’s analysis, which would it be and why?

---

## Week 2 – Day 3: Univariate EDA – Categorical Features

**Goal:** Understand distributions of categorical features and how they might matter.

### 1. Core study (45–60 min)

Create `week2_day3_univariate_categorical.ipynb`.

1. **Select categorical columns**
   ```python
   cat_cols = df.select_dtypes(include=["object", "category"]).columns
   cat_cols
   ```

2. **Value counts**
   - For each categorical column:
     ```python
     df["Sex"].value_counts()
     df["Sex"].value_counts(normalize=True)
     ```
   - Do the same for `Embarked`, `Pclass` (even though it’s numeric, treat as categorical for now).

3. **Bar plots**
   ```python
   sns.countplot(x="Sex", data=df)
   plt.title("Count of passengers by Sex")
   plt.show()
   ```

### 2. Core practice (30–45 min)

1. **Summarize each categorical variable**
   - For each `cat_cols`:
     - Print `value_counts()` and `value_counts(normalize=True)`.
   - In markdown, write:
     - Does the distribution look balanced (similar counts each category)?
     - Any rare categories (very small count)?

2. **Basic interpretation**
   - For at least 2 columns (e.g., `Sex`, `Pclass`):
     - Create bar plots with `sns.countplot`.
     - In words, describe:
       - Which category is most common?
       - Is that what you expected?

3. **Cross-tab preview (no target yet)**
   - Build one cross-tab between two categorical features, e.g. `Pclass` and `Embarked`:
     ```python
     pd.crosstab(df["Pclass"], df["Embarked"])
     ```
   - Then with `normalize="index"`:
     ```python
     pd.crosstab(df["Pclass"], df["Embarked"], normalize="index")
     ```

### 3. Stretch (optional)

- Build a small **“data dictionary”** in markdown:
  - For each categorical column:
    - Name, meaning, categories (with counts), and any concerns (e.g., too many rare values, unclear meaning).

### 4. Reflection

- Which categorical column seems **most promising** for predicting survival, and why?
- Are there any columns you suspect might be **leaky** (only knowable after the outcome)?

---

## Week 2 – Day 4: Relationships Between Features (Bivariate EDA)

**Goal:** Explore how features relate to each other (and start peeking at the target).

### 1. Core study (45–60 min)

Create `week2_day4_bivariate.ipynb`.

1. **Numeric vs numeric (scatter + correlation)**
   - Identify numeric columns again:
     ```python
     numeric_cols = df.select_dtypes(include=[np.number]).columns
     ```
   - Compute correlation matrix:
     ```python
     corr = df[numeric_cols].corr()
     corr
     ```
   - Visualize with heatmap:
     ```python
     plt.figure(figsize=(8, 6))
     sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
     plt.title("Correlation matrix")
     plt.show()
     ```

2. **Scatter plots**
   - Example:
     ```python
     sns.scatterplot(x="Age", y="Fare", data=df)
     plt.show()
     ```
   - Note any patterns or clusters.

3. **Category vs numeric**
   - Boxplot: e.g., `Fare` by `Pclass`:
     ```python
     sns.boxplot(x="Pclass", y="Fare", data=df)
     plt.show()
     ```

### 2. Core practice (30–45 min)

1. **Correlation inspection**
   - From `corr`, identify:
     - The pair of features with **highest positive correlation** (excluding self-correlation).
     - The pair with a notable **negative correlation** (if any).
   - In words: why might they be correlated?

2. **At least 3 plots**
   - Plot:
     - A scatter plot of two numeric features of your choice.
     - A boxplot of a numeric feature by a categorical feature (e.g., `Age` by `Sex`).
     - A boxplot of `Fare` by `Pclass`.
   - For each plot, write 2–3 bullet points describing what you see.

3. **Target-related sneak peek**
   - Treat `Survived` as numeric and look at its correlation with other numeric features:
     ```python
     corr["Survived"].sort_values(ascending=False)
     ```
   - Which numeric feature(s) have higher absolute correlation with `Survived`?

### 3. Stretch (optional)

- Try `sns.pairplot` on a subset of numeric columns:
  ```python
  sns.pairplot(df[["Age", "Fare", "SibSp", "Parch", "Survived"]], hue="Survived")
  ```
- This can be slow; use fewer rows if needed:
  ```python
  sns.pairplot(df.sample(200), vars=["Age", "Fare", "SibSp", "Parch"], hue="Survived")
  ```

### 4. Reflection

- Which two features seem to have the strongest relationship (of any kind)?
- Did anything contradict your intuition from earlier days?

---

## Week 2 – Day 5: Missing Data – Detection and Simple Handling

**Goal:** Understand missingness in your dataset and try simple strategies to handle it.

### 1. Core study (45–60 min)

Create `week2_day5_missing_values.ipynb`.

1. **Detect missing values**
   ```python
   df.isnull().sum()
   (df.isnull().sum() / len(df)).sort_values(ascending=False)
   ```

2. **Visualize missingness (optional but useful)**
   - Install `missingno` (optional):
     ```bash
     pip install missingno
     ```
   - Usage:
     ```python
     import missingno as msno
     msno.matrix(df)
     plt.show()
     ```

3. **Simple strategies**
   - Drop rows with missing values (`dropna`) – usually **not** ideal if many rows are dropped.
   - Fill numeric with mean/median (simple imputation).
   - Fill categorical with most frequent value (mode).

### 2. Core practice (30–45 min)

1. **Missingness profile**
   - Create a small table in a DataFrame:
     ```python
     missing = df.isnull().sum().to_frame(name="n_missing")
     missing["pct_missing"] = missing["n_missing"] / len(df) * 100
     missing.sort_values("pct_missing", ascending=False)
     ```
   - For top 3 columns with most missing values:
     - Is it numeric or categorical?
     - Would you drop the column? Why or why not?

2. **Impute one numeric and one categorical**
   - Example:
     ```python
     df_imputed = df.copy()

     # Numeric (Age)
     median_age = df_imputed["Age"].median()
     df_imputed["Age"] = df_imputed["Age"].fillna(median_age)

     # Categorical (Embarked)
     mode_embarked = df_imputed["Embarked"].mode()[0]
     df_imputed["Embarked"] = df_imputed["Embarked"].fillna(mode_embarked)
     ```
   - Confirm no missing values remain in these columns:
     ```python
     df_imputed[["Age", "Embarked"]].isnull().sum()
     ```

3. **Compare distributions before/after**
   - Plot `Age` distribution before and after imputation on same axes:
     ```python
     df["Age"].hist(alpha=0.5, label="original", bins=30)
     df_imputed["Age"].hist(alpha=0.5, label="imputed", bins=30)
     plt.legend()
     plt.show()
     ```
   - Is there a noticeable difference?

### 3. Stretch (optional)

- Group-wise imputation:
  - For example, compute median `Age` by `Pclass` and use that to fill:
    ```python
    df_imputed2 = df.copy()
    df_imputed2["Age"] = df_imputed2.groupby("Pclass")["Age"].transform(
        lambda x: x.fillna(x.median())
    )
    ```
  - Compare distribution with global median imputation.

### 4. Reflection

- Which column’s missingness concerns you the most? Why?
- Do you think your simple imputation choices might **bias** the model later?

---

## Week 2 – Day 6: Framing the ML Problem – Features & Target, Baselines, Train/Test Split

**Goal:** Connect EDA to ML: define features & target, compute simple baselines, and create a train/test split.

### 1. Core study (45–60 min)

Create `week2_day6_ml_framing.ipynb`.

1. **Supervised vs unsupervised (conceptual)**
   - Supervised: we have inputs **X** and a target **y** (e.g., `Survived`).
   - Unsupervised: we only have X, no labels (e.g., clustering passengers without survival info).

2. **Classification vs regression**
   - Classification: categorical target (e.g., Survived 0/1).
   - Regression: numeric target (e.g., house price).

3. **Define X and y for Titanic**
   - Choose some candidate features:
     - Example: `["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]`
   - Keep things simple: don’t encode categorical yet; just select these columns.
   ```python
   target = "Survived"
   features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

   df_model = df_imputed[features + [target]].dropna()
   X = df_model[features]
   y = df_model[target]
   ```

4. **Baselines**
   - Naive classifier: predict the **most frequent class** for everyone.
     ```python
     y.value_counts(normalize=True)
     ```
   - If majority is 0 (did not survive), baseline accuracy = that fraction.

5. **Train/test split (intro)**
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42, stratify=y
   )

   X_train.shape, X_test.shape, y_train.shape, y_test.shape
   ```

### 2. Core practice (30–45 min)

1. **Compute baseline accuracy**
   - Majority class in **train set**:
     ```python
     y_train.value_counts(normalize=True)
     ```
   - Baseline accuracy = proportion of the majority class.
   - Write a few sentences:
     - “If I always predict class X, I will be right Y% of the time.”

2. **Think about cost of errors**
   - In markdown, describe:
     - What is a **false positive** vs **false negative** in Titanic survival?
     - Which is worse, and why? (There’s no single right answer; justify your thinking.)

3. **Different splits**
   - Try `test_size=0.1` and `0.3`.
   - For each, record:
     - Shapes of train/test.
     - Class distribution in train/test (`value_counts(normalize=True)`).
   - Does stratification (`stratify=y`) help keep class balance similar?

### 3. Stretch (optional)

- Very simple **rule-based baseline** using domain intuition:
  - Example rule: “Predict Survived=1 if Sex='female', else 0.”
  - Implement:
    ```python
    y_pred_rule = (X_test["Sex"] == "female").astype(int)
    (y_pred_rule == y_test).mean()  # rule accuracy
    ```
  - Compare with majority baseline accuracy.

### 4. Reflection

- Are you surprised by how strong/weak the naive baselines are?
- How might a **model** do better than the majority-class or simple-rule baselines?

---

## Week 2 – Day 7: Mini EDA Report & ML Workflow Overview

**Goal:** Consolidate EDA into a mini-report and outline the ML workflow you’ll implement in Week 3.

### 1. Core study (45–60 min)

Create `week2_day7_report_and_workflow.ipynb`.

1. **Review what you’ve done**
   - Skim through notebooks from Days 1–6.
   - Copy **key plots/tables** into this new notebook:
     - 1–2 histograms (e.g., Age, Fare)
     - 1–2 bar plots (e.g., Sex, Pclass)
     - 1–2 bivariate plots that involve Survived (e.g., Fare vs Survived, Sex vs Survived via groupby)

2. **High-level ML workflow (conceptual)**
   In a markdown cell, write a brief outline like:

   1. Define problem and target metric.
   2. Collect and clean data (handle missing values, inconsistent types).
   3. Exploratory Data Analysis (EDA).
   4. Feature engineering and preprocessing (encoding, scaling).
   5. Train baseline models.
   6. Train more complex models and tune hyperparameters.
   7. Evaluate on validation/test sets.
   8. Interpret results and iterate.

### 2. Core practice (30–45 min)

1. **Write a mini EDA report (in markdown)**
   Include the following sections (with plots/tables inserted above/below as needed):

   - **1. Dataset Description**
     - What each row represents.
     - Number of rows and columns.
   - **2. Target Variable**
     - Distribution of `Survived`.
     - Baseline accuracy (majority class).
   - **3. Univariate Analysis**
     - Summary of distributions for key numeric features (Age, Fare).
     - Summary of key categorical features (Sex, Pclass).
   - **4. Bivariate Analysis**
     - How survival appears to relate to:
       - Sex
       - Pclass
       - Maybe one numeric feature
   - **5. Missing Data**
     - Which features have missing values and how you handled them.
   - **6. Initial Feature Set**
     - List features you plan to use for modeling in Week 3.
     - For each: why you think it might help predict survival.

2. **Design your first model plan**
   - In markdown, write a **short plan** for Week 3:
     - Which model will you try first? (e.g., Logistic Regression).
     - What metric will you focus on (accuracy, maybe also precision/recall)?
     - Any specific hypotheses (e.g., “Including Sex and Pclass will significantly improve performance over baseline”).

### 3. Stretch (optional)

- Implement a **tiny evaluation** of your simple rule-based baseline (from Day 6) with a confusion matrix:
  ```python
  from sklearn.metrics import confusion_matrix

  cm = confusion_matrix(y_test, y_pred_rule)
  cm
  ```
- Manually interpret:
  - True positives, true negatives, false positives, false negatives.
  - Write 3–4 sentences on what this says about your rule.

### 4. Reflection

- What are the **three most important things** you learned about the dataset this week?
- Which part of EDA felt most useful for preparing for modeling, and which felt like extra work?
- Are there any features you wish you had but are missing from the dataset?

---

If you’d like, next I can generate a **Week 3 Day 1–7 plan** that walks you through building your **first real models** (regression & classification) using scikit-learn, starting from this Titanic analysis.
