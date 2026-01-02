Week 12 is about **consolidation and transition**:

- Prove to yourself you can do ML end‑to‑end without hand‑holding  
- Solidify core concepts (not just code recipes)  
- Turn your work into a portfolio piece you can talk about confidently  
- Design your “what next?” learning path

Assume ~1–2 hours/day.  
Each day: **Goal → Core study → Core practice → Stretch → Reflection**.

You’ll use:

- Your **capstone project**
- One or two **small new datasets** (e.g., from scikit‑learn or Kaggle) for “from‑scratch” practice.

---

## Day 1 – Global Review & Knowledge Audit

**Goal:** Build a clear picture of **what you know**, **what’s fuzzy**, and **what you can now do without help**.

### 1. Core study (45–60 min)

Notebook: `notebooks/week12_day1_knowledge_audit.ipynb`.

1. **Write down the “big topics” from memory (no notes yet)**

In a markdown cell, list everything you remember you’ve touched in 12 weeks, grouped roughly as:

- Supervised learning
- Unsupervised learning
- Evaluation & validation
- Data preprocessing & pipelines
- Models
- Deployment/packaging

Then, under each, bullet the concepts you **remember** without checking:

Example:

- Supervised:
  - Regression vs classification
  - Train/test split, baselines
  - Linear/logistic regression, trees, forests, GBMs
- Unsupervised:
  - k-means, PCA
  - Silhouette score

2. **Now cross-check with notes / previous weeks**

Open:

- Week 1–9 notebooks (skim)
- Any outlines / logs you wrote

Add missing items you forgot in a **different color or bullet marker**, e.g.:

- Supervised:
  - (Forgot) Cross-validation (k-fold, stratified)
  - (Forgot) Regularization (L1/L2)
  - (Forgot) ROC-AUC

3. **Mark each concept with a confidence rating**

Turn your list into a small table (markdown or pandas):

Columns:

- Concept
- Category
- Confidence (High / Medium / Low)
- Notes (what’s unclear)

Example:

| Concept                 | Category      | Confidence | Notes                          |
|-------------------------|--------------|------------|--------------------------------|
| Train/test split        | Supervised   | High       | Comfortable                    |
| Cross-validation        | Evaluation   | Med        | Unsure when to use which cv    |
| Regularization (L1/L2)  | Models       | Low        | Not sure intuition             |
| Silhouette score        | Unsupervised | Med        | Formula fuzzy, use okay        |

### 2. Core practice (30–45 min)

1. **Pick 3–5 “Medium/Low” concepts and write 2–3 sentence explanations**

For each chosen concept:

- Explain it **in your own words**, as if to a non‑ML friend.
- Don’t look up the definition first; try from memory, then correct with docs.

Examples to consider:

- Cross-validation
- Regularization
- Bias–variance / overfitting vs underfitting
- Precision vs recall vs ROC-AUC
- k-means vs PCA

2. **Verify and refine using docs**

After writing your explanations:

- Check scikit-learn docs or a trusted source.
- Adjust your explanation where it was off.

### 3. Stretch (optional)

- Draw a **concept map** on paper or in a tool (e.g., Excalidraw, draw.io):
  - Nodes: concepts (e.g., “train/test split”, “overfitting”, “RandomForest”).
  - Arrows: “depends on”, “helps with”, “used in”.
- This is for *you*; no need to be pretty, just coherent.

### 4. Reflection

- Which 2–3 concepts moved from “unclear” to “I at least have a story now” during this exercise?
- Is your **weakness** more on:
  - Theory / intuition?
  - Practical implementation?
  - Communicating what you know?

---

## Day 2 – Timed End-to-End Mini-Project (New Dataset)

**Goal:** Prove you can do an **end-to-end supervised ML workflow** on a *new* dataset with minimal guidance.

### 1. Core study (10–15 min: setup & rules)

Notebook: `notebooks/week12_day2_timed_project.ipynb`.

Pick a **small, clean dataset** you haven’t used yet. Options:

- From `sklearn.datasets`:
  - `load_breast_cancer` (binary classification)
  - `load_wine` (multi-class)
  - `load_diabetes` (regression)
- Or a small Kaggle dataset you haven’t used.

Rules for yourself:

- Timebox: **max 90 minutes** for the main workflow.
- Use your **memory and docs**, but **don’t copy old notebooks**.
- Aim for:
  - Problem framing.
  - Basic EDA.
  - Baseline.
  - One or two models with pipelines.
  - Test metric.

### 2. Core practice (60–90 min)

Treat this section like a live “coding exercise”:

1. **Problem definition & loading**

- Load data into DataFrame (if using sklearn, convert to DataFrame).
- Write:
  - What each row represents.
  - Target variable.
  - Is it classification or regression?
  - What metric you’ll use.

2. **Quick EDA**

- `df.info()`, `df.describe()`
- Target distribution.
- 2–3 quick histograms or barplots for key features.

3. **Train/test split & naive baseline**

- Do `train_test_split`.
- Compute:
  - Majority baselines (classification) or mean baseline (regression).

4. **Build one proper pipeline**

- Separate numeric vs categorical (if any).
- Use `ColumnTransformer` + `Pipeline`.
- Try:
  - Simple model (LogisticRegression / LinearRegression/Ridge).
  - Tree-based model (RandomForest or HistGradientBoosting).

5. **Evaluate**

- Use your primary metric on the test set.
- If classification, maybe also precision/recall or ROC-AUC.

6. **Short summary**

In markdown:

- What did you do?
- Best test metric.
- How much above baseline?

### 3. Stretch (optional)

- Add **cross-validation** for your best model.
- Do a tiny bit of feature importance review.

### 4. Reflection

- Which steps felt natural vs forced?
- If you had to do this same exercise again tomorrow on another dataset, where would you expect to be faster?

---

## Day 3 – Conceptual Deepening: Overfitting, Regularization, and Bias–Variance

**Goal:** Cement your intuition about **model complexity**, **overfitting**, and **regularization** via small, focused experiments.

### 1. Core study (45–60 min)

Notebook: `notebooks/week12_day3_bias_variance_regularization.ipynb`.

Use synthetic or simple data (e.g., scikit-learn).

1. **Overfitting with polynomial regression (regression example)**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y_true = np.sin(2 * np.pi * X[:, 0])
y = y_true + 0.2 * np.random.randn(100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Fit models with degrees `[1, 3, 10]`:

```python
degrees = [1, 3, 10]
rows = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    rows.append({
        "degree": d,
        "rmse_train": mean_squared_error(y_train, y_train_pred, squared=False),
        "rmse_test": mean_squared_error(y_test, y_test_pred, squared=False),
    })

rows
```

Plot train vs test RMSE against degree.

2. **Regularization idea (short)**

- L2 (Ridge) shrinks weights, reducing variance but adding bias.
- L1 (Lasso) can zero out coefficients (feature selection).

### 2. Core practice (30–45 min)

1. **Extend experiment with regularization**

Use Ridge for polynomial regression (say degree=10) and vary `alpha`:

```python
from sklearn.linear_model import Ridge

alphas = [0.0, 0.1, 1.0, 10.0]  # 0.0 ~ plain linear regression
rows = []

for alpha in alphas:
    poly = PolynomialFeatures(degree=10)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    rows.append({
        "alpha": alpha,
        "rmse_train": mean_squared_error(y_train, y_train_pred, squared=False),
        "rmse_test": mean_squared_error(y_test, y_test_pred, squared=False),
    })

import pandas as pd
pd.DataFrame(rows)
```

Interpret:

- As `alpha` increases, what happens to train vs test RMSE?

2. **Relate to your capstone**

In markdown:

- Where did you see signs of overfitting in your capstone (if at all)?
- Which hyperparameters (e.g., `max_depth`, `min_samples_leaf`, `C`) served as **regularization**?

### 3. Stretch (optional)

- Do a similar overfitting experiment for classification:
  - e.g., k‑NN with varying `k` on a simple dataset (`make_moons`), plot decision boundaries or metrics.

### 4. Reflection

- How would you explain **bias–variance tradeoff** with 2–3 sentences referencing your experiments?
- If your capstone model started overfitting more, what 2–3 levers would you reach for first?

---

## Day 4 – Explain & Defend Your Capstone (Interview-Style)

**Goal:** Practice explaining your project and core ML ideas **in plain language**, as if to an interviewer or non‑technical stakeholder.

### 1. Core study (45–60 min)

Notebook: `notebooks/week12_day4_explaining_capstone.ipynb`.

1. **“Tell me about your project” answer**

Write a **2–3 minute** spoken answer in text (200–400 words):

- Context & problem.
- Data source & main features.
- What models you tried.
- How well it works (metrics).
- What you learned / trade-offs.

Try to avoid heavy jargon; then, if needed, add 1–2 technical details at the end.

2. **Five key concept explanations tied to your project**

For each of these (adapt if you like), write ~3 sentences, referencing your capstone where possible:

- Train/validation/test split or cross-validation.
- Overfitting vs underfitting.
- Feature engineering.
- One metric you used (e.g., ROC-AUC, RMSE).
- Why you chose your final model (RF vs GBM vs logistic/linear).

### 2. Core practice (30–45 min)

1. **Anticipate 5 “tough” questions**

Examples:

- “If you had 10x more data, what would you do differently?”
- “Why didn’t you use deep learning for this problem?”
- “How would you know when your model is failing in production?”
- “What if your best feature becomes unavailable in future data?”
- “How do you handle imbalanced classes?”

For each, write a short (4–6 sentence) answer.

2. **Optional: speak out loud**

If possible, actually **say** your “project summary” and answers out loud:

- Notice where you stumble, over‑explain, or sound uncertain.
- Edit your text to be smoother / clearer.

### 3. Stretch (optional)

- Ask a friend (technical or not) to read your project summary and ask 2–3 follow-up questions.
- Or imagine their follow‑ups and answer those too.

### 4. Reflection

- Which explanation did you find hardest to write clearly?
- Are there parts of your project you still don’t fully understand, that you are glossing over verbally?

---

## Day 5 – Rebuild an Old Project from Scratch (Without Looking)

**Goal:** Test your **true mastery** by recreating one earlier, smaller project with no copy-paste.

### 1. Core study (10–15 min: choose project & recall from memory)

Pick **one** earlier project:

- Titanic (binary classification)
- House Prices (regression)
- Adult Income (classification)
- Mall Customers (clustering)

Notebook: `notebooks/week12_day5_rebuild_old_project.ipynb`.

Write from memory (markdown):

- The problem.
- The dataset shape (roughly).
- What you previously did at a high level.

### 2. Core practice (60–90 min)

Try to **rebuild a minimal but decent version** without opening your old notebooks:

1. **Load data & define target**

2. **Basic cleaning & EDA**

- `info()`, `describe()`, a couple of simple plots.

3. **Train/test split & baseline**

4. **Preprocessing + pipeline**

- Numeric + categorical features.
- Imputation + scaling + one-hot.

5. **Model(s)**

- At least one linear/logistic.
- At least one tree-based (RF or GBM).

6. **Evaluate & short summary**

- Baseline metric vs best model metric.
- 3–5 sentence summary of what works.

Don’t worry about matching old scores exactly; focus on process.

### 3. Stretch (optional)

- After finishing, open your original Week 2–4 notebooks:
  - Compare your old vs new approach.
  - Note 3 improvements in your current style/understanding.

### 4. Reflection

- Which parts flowed easily without reference?
- Where did you get stuck and need to check docs or old code?
- Does your **new version** look cleaner than your original one?

---

## Day 6 – Portfolio & Storytelling: Multi-Project View

**Goal:** Step back and see your work as a **portfolio**, not just one project; make it easy for someone to understand your overall journey.

### 1. Core study (45–60 min)

You can do this in a markdown file (`PORTFOLIO.md`) or notebook.

1. **List your main ML projects so far**

For each:

- Name (e.g., “Titanic Survival Prediction”)
- Problem type (classification/regression/clustering)
- Dataset source
- 1–2 sentences on goals
- 1–2 sentences on methods/models
- 1–2 sentences on results/insights

2. **Pick 2–3 projects worth showing**

Likely candidates:

- Your capstone (main)
- One supervised mini-project (e.g., Adult Income or Titanic)
- One unsupervised project (e.g., Mall Customers segmentation)

3. **For each chosen project, write a 1‑paragraph “portfolio blurb”**

Example structure:

- One sentence: problem and why it matters.
- One sentence: data size and main features.
- One sentence: core methods (pipelines, models).
- One sentence: key metric and improvement vs baseline.
- One sentence: main insight / lesson learned.

### 2. Core practice (30–45 min)

1. **Integrate into README and/or GitHub profile**

- If you have a GitHub:
  - Add a short “Projects” section linking to your capstone repo and 1–2 other repos.
- If not:
  - Add the project blurbs to your capstone README under a “Related Work / Other Projects” section.

2. **Optional: draft a short “About Me (ML)” paragraph**

3–5 sentences about:

- Your background.
- Why you started learning ML.
- What you’ve focused on (tabular data, end‑to‑end projects).
- What you want to learn next or work on.

### 3. Stretch (optional)

- Choose 3–5 plots/figures from all your projects that best show your skills:
  - Feature importance
  - ROC curves / error plots
  - PCA / cluster visualizations  
  Save them in an `img/` folder and reference them in your reports/README.

### 4. Reflection

- If someone only saw your portfolio blurbs, what impression would they get of your skills?
- Are you leaning more toward:
  - “Applied ML engineer” (pipelines, deployment),
  - “Data scientist” (EDA, insights), or
  - Some other angle?

---

## Day 7 – Final Review, Self-Assessment, and Next Steps

**Goal:** Lock in what you’ve learned, identify your strongest/weakest areas, and design a realistic **next‑3‑months** learning plan.

### 1. Core study (45–60 min)

Notebook: `notebooks/week12_day7_final_review_and_plan.ipynb`.

1. **Self-quiz (closed book first)**

In markdown, create and answer (from memory):

- 5 conceptual questions, e.g.:
  - Explain overfitting and how to detect it.
  - What is cross-validation and why is it used?
  - Difference between precision and recall.
  - Why pipelines are useful.
  - k-means vs PCA: goals and differences.

- 5 practical questions, e.g.:
  - How do you handle missing numeric vs categorical values?
  - How do you avoid data leakage?
  - How would you pick a threshold for a classifier?
  - How to evaluate a regression model.
  - How to check performance across subgroups.

After answering, check against notes/docs, and correct misconceptions in a different color or bullet.

2. **Skill matrix**

Create a table with rows = skills, columns = Confidence (1–10), Evidence, Next Step.

Skills to include:

- Python & pandas for data work
- NumPy basics
- EDA & visualization
- Supervised ML (regression + classification)
- Unsupervised ML (k-means, PCA)
- Pipelines & ColumnTransformer
- Model evaluation & metrics
- Hyperparameter tuning
- Error analysis & fairness
- Packaging / deployment basics
- Communication & reporting

Example row:

| Skill                         | Confidence (1–10) | Evidence                               | Next Step                          |
|-------------------------------|-------------------|----------------------------------------|------------------------------------|
| Pipelines & ColumnTransformer | 7                 | Built for capstone + 3 other projects  | Use them in time-series/NLP later  |

### 2. Core practice (30–45 min)

1. **Design a 3‑month learning plan (high level)**

Pick **1–2 tracks**, e.g.:

- Deepen tabular ML:
  - XGBoost/LightGBM/CatBoost
  - Advanced feature engineering
  - Model calibration, drift, monitoring
- New domain:
  - NLP with Hugging Face
  - Computer vision with CNNs
  - Time series forecasting
- Engineering:
  - MLOps basics, Docker, CI/CD
  - More systematic testing/monitoring

For each month:

- Month 1: What to focus on, 1–2 small sub‑goals.
- Month 2: A project to apply it.
- Month 3: Consolidate + share (blog post, repo, etc.).

2. **Write a short letter to “past you” (optional but powerful)**

Imagine writing to yourself 12 weeks ago, describing:

- Where you’ve arrived.
- What you wish you’d known.
- What surprised you.

### 3. Stretch (optional)

- Share your capstone / repo link with one trusted person (friend, colleague, online mentor) and ask for:
  - 1 thing they like.
  - 1 thing that’s unclear.
  - 1 suggestion.

- Start a small public log (Notion, GitHub repo, blog) for your new 3‑month plan.

### 4. Reflection

- What are the **3 biggest things** you’ve gained from these 12 weeks (skills, confidence, habits)?
- If you had to summarize your ML journey so far in **one paragraph**, what would it say?
- What’s the very **next concrete action** you’ll take after finishing this week (e.g., pick a new dataset, read a specific book, start an online course)?

---

You now have:

- A capstone that’s end‑to‑end, explainable, and (lightly) deployable  
- Multiple smaller projects showing range  
- A clearer sense of what you know and what to learn next  

If you’d like, you can tell me your capstone’s **domain** and your preferred next track (e.g., NLP, CV, MLOps), and I can suggest a **3‑month follow‑up study plan** tailored to that direction.
