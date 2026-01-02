Week 11 is about **polishing and deepening** your capstone so it’s something you could confidently show in a portfolio:

- Validate which features and design choices actually matter  
- Check fairness / subgroup performance more seriously  
- Tighten testing & robustness of your code and interface  
- Turn your work into a clean, shareable project (report + repo)

Assume ~1–2 hours/day.  
You’ll keep using your existing `capstone/` structure.

I’ll say “classification” by default; if you’re doing regression, swap in RMSE/MAE/R² where I mention accuracy/ROC‑AUC, and interpret accordingly.

Each day: **Goal → Core study → Core practice → Stretch → Reflection**.

---

## Day 1 – Final Objectives & “What Matters?” Plan

**Goal:** Decide what *exactly* you want your final capstone to demonstrate, and define a short list of **targeted improvements** to pursue in Week 11.

### 1. Core study (45–60 min)

Notebook: `notebooks/week11_day1_final_objectives.ipynb`.

1. **Revisit your Week 9 & Week 10 summaries**

Open:

- Week 9 milestone notebook
- Week 10 milestone notebook
- Your README

Collect in markdown:

- Current best metrics (CV + test)
- Current best model type + hyperparameters
- Key known issues (segments, features, robustness)

2. **Define “final version” criteria (markdown)**

Write 5–8 bullet points answering:

- What should your capstone **prove** about your skills?
  - e.g., “I can build robust pipelines with preprocessing + tuning”  
  - “I can interpret and critique a model, not just train it”
- What is **non‑negotiable** for final version?
  - e.g., at least X% improvement over baseline  
  - clear documentation + from‑scratch script  
  - basic fairness / subgroup analysis

3. **List potential improvements**

Brainstorm 6–10 possible improvements, grouped:

- Features:
  - Better domain features
  - Removing noisy / leaky ones
- Models:
  - Slightly improved GBM settings or regularized linear model
  - Maybe a simple ensemble of 2 strong models
- Evaluation:
  - Deeper subgroup/fairness check
  - Better calibration / thresholding (if classification)
- Engineering:
  - Unit tests
  - Cleaner predict interface, error handling
  - Repo/README polish

4. **Prioritize 3–4 improvements**

From that list, pick **3–4** to actually pursue this week. Mark them with:

- Expected impact (high/med/low)
- Effort (high/med/low)

Pick ones with **high impact, medium or low effort** as your Week 11 focus.

### 2. Core practice (30–45 min)

1. **Create a “Week 11 plan” table**

In markdown or a small DataFrame:

| ID | Area        | Description                                   | Impact | Effort | Status |
|----|-------------|-----------------------------------------------|--------|--------|--------|
| F1 | Features    | Drop weak features & run ablation study       | High   | Med    | TODO   |
| E1 | Evaluation  | Fairness/subgroup analysis for key attribute  | High   | Low    | TODO   |
| T1 | Testing     | Add basic tests for predict_one & pipeline    | Med    | Med    | TODO   |
| R1 | Reporting   | Final polished report & GitHub‑ready README   | High   | Med    | TODO   |

2. **Mark anything out of scope**

Explicitly write 2–3 ideas you will *not* do now (to avoid scope creep), e.g.:

- “Not integrating deep learning for this tabular problem.”
- “Not building a full production API with auth/logging; a simple script/app is enough.”

### 3. Stretch (optional)

- Create a simple **Kanban board** for yourself (even just markdown lists: “To Do / Doing / Done”) and put your Week 11 tasks there.

### 4. Reflection

- If you had to show your capstone to someone in **one week**, what one thing must change between now and then?
- Are your Week 11 goals ambitious but realistic?

---

## Day 2 – Systematic Feature Ablation & Simplification

**Goal:** Find out **which features truly matter** by systematically removing them (or groups of them) and seeing how performance changes.

### 1. Core study (45–60 min)

Notebook: `notebooks/week11_day2_feature_ablation.ipynb`.

1. **Load data and your best preprocessing + model**

Recreate or import:

- `df` (raw data)
- `X`, `y`
- `X_train, X_test, y_train, y_test`
- `preprocessor` (ColumnTransformer)
- `best_model` (pipeline)

2. **Identify feature groups**

In markdown, group features logically:

- Demographics (age, gender, region)
- Behavior (usage, purchases, tenure)
- Financial (income, charges, balances)
- Engineered (ratios, buckets, log transforms)

Make a small table:

| Group         | Features in group                       |
|--------------|------------------------------------------|
| Demographic  | age, gender, region                     |
| Usage        | num_calls, num_logins, tenure           |
| Engineered   | log_amount, spend_per_visit, age_bucket |

3. **Implement ablation: drop a group at a time**

You can do this by:

- Creating new versions of `X_train` and `X_test` with columns removed.
- Keeping the same `preprocessor` structure, just referencing fewer columns (or building separate preprocessors for each experiment).

Example pattern:

```python
from sklearn.model_selection import cross_val_score

def evaluate_with_dropped_features(drop_cols, model_pipeline):
    X_train_mod = X_train.drop(columns=drop_cols, errors="ignore")
    X_test_mod = X_test.drop(columns=drop_cols, errors="ignore")

    # Rebuild feature lists
    num_feats = X_train_mod.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = X_train_mod.select_dtypes(exclude=[np.number]).columns.tolist()

    # Rebuild preprocessor & pipeline quickly here (copy from earlier)
    # ...

    scores = cross_val_score(pipe, X_train_mod, y_train, cv=5,
                             scoring="accuracy" if classification else "neg_root_mean_squared_error")
    return scores.mean(), scores.std()
```

You don’t have to be ultra‑DRY; a little controlled duplication is fine for clarity.

### 2. Core practice (30–45 min)

1. **Run ablation for 3–4 groups**

Record results in a table:

| Group dropped  | CV mean metric | Δ vs full model | Notes                    |
|----------------|----------------|-----------------|--------------------------|
| None (full)    | 0.82           | –               | baseline                 |
| Demographic    | 0.81           | -0.01           | slight drop              |
| Usage          | 0.78           | -0.04           | big drop → important     |
| Engineered     | 0.82           | 0               | maybe not so impactful?  |

(For regression, use RMSE, and interpret higher RMSE as worse.)

2. **Try single-feature ablations for top 3–5 features**

Based on feature importance or intuition:

- Drop one feature at a time.
- See if performance changes more than random noise (~0.5–1% depending on score scale).

### 3. Stretch (optional)

- Try a **“minimal” feature set**: only the top ~5–10 features identified as most important.
  - Build a model using only them and compare performance vs full model.

### 4. Reflection

- Which feature group seems **critical**, and which could be removed with little cost?
- How does this affect how you’d explain your model to a stakeholder (“the model mostly uses X and Y”)?

---

## Day 3 – Fairness & Subgroup Performance (More Serious Pass)

**Goal:** Take a more deliberate look at **performance across subgroups**, and reason about fairness / risk for your use case.

### 1. Core study (45–60 min)

Notebook: `notebooks/week11_day3_fairness_and_subgroups.ipynb`.

1. **Choose 1–2 sensitive or important attributes**

Examples:

- Classification:
  - Gender, age group, region, customer type
- Regression:
  - Same, to see if errors are worse for specific groups

2. **Ensure attributes are present in your test set DataFrame**

Build a `df_test`:

```python
df_model = df.dropna(subset=[target]).copy()

# recreate X_train, X_test, y_train, y_test or reuse

df_test = X_test.copy()
df_test[target] = y_test
df_test["y_pred"] = best_model.predict(X_test)

if classification and hasattr(best_model, "predict_proba"):
    df_test["y_proba"] = best_model.predict_proba(X_test)[:, 1]
```

3. **Define subgroup variable(s)**

Examples:

```python
# Age groups
if "age" in df_test.columns:
    df_test["age_group"] = pd.cut(
        df_test["age"],
        bins=[0, 25, 40, 60, 120],
        labels=["young", "young_adult", "middle", "older"]
    )

# Use existing categorical columns like "gender", "region"
```

4. **Compute performance per subgroup**

Classification (accuracy + recall, at least):

```python
from sklearn.metrics import accuracy_score, recall_score

def subgroup_metrics(df, group_col):
    rows = []
    for g, group in df.groupby(group_col):
        if len(group) < 20:  # skip very tiny groups
            continue
        acc = accuracy_score(group[target], group["y_pred"])
        rec = recall_score(group[target], group["y_pred"])
        rows.append({"group": g, "n": len(group), "accuracy": acc, "recall": rec})
    return pd.DataFrame(rows)

age_results = subgroup_metrics(df_test, "age_group")
age_results
```

Regression (RMSE per subgroup):

```python
from sklearn.metrics import mean_squared_error

def subgroup_rmse(df, group_col):
    rows = []
    for g, group in df.groupby(group_col):
        if len(group) < 20:
            continue
        rmse = mean_squared_error(group[target], group["y_pred"], squared=False)
        rows.append({"group": g, "n": len(group), "rmse": rmse})
    return pd.DataFrame(rows)

age_rmse = subgroup_rmse(df_test, "age_group")
age_rmse
```

### 2. Core practice (30–45 min)

1. **Interpret subgroup differences**

In markdown:

- Which groups have **worse performance**?
  - Lower recall / accuracy (classification).
  - Higher RMSE (regression).
- Are these differences large enough to matter?

2. **Think about fairness / ethics (contextual)**

Write 6–10 sentences:

- Given your domain, are these differences problematic?
- Could they reflect:
  - Data imbalance?
  - Real differences in population?
  - Model bias / lack of features for certain groups?

3. **Optional simple mitigation thought experiment**

Even if you don’t implement it, outline 1–2 ideas:

- Collect more data for underperforming groups.
- Add group‑specific features or models.
- Adjust thresholds per group (cautious, as this can be controversial).

### 3. Stretch (optional)

- For binary classification: read about and, if you like, attempt basic fairness metrics (like demographic parity, equal opportunity) using e.g. simple `p(y_pred=1 | group)` comparisons.
- If your dataset includes something very sensitive (e.g., race), think carefully if you really want to use it as a feature at all.

### 4. Reflection

- Are there any groups for whom you would **not** feel comfortable deploying this model as‑is?
- How would you communicate these results honestly to a stakeholder or user?

---

## Day 4 – Try One “Alternative Approach” (But Keep It Scoped)

**Goal:** Explore **one** alternative modeling approach *in a controlled way*, to see if it brings a meaningful improvement or insight.

You must keep the scope tight. Examples:

- If you’ve focused on tree models → try a **regularized linear model** (L1/L2) for interpretability.
- If you’ve done mostly linear/logistic → try a **GBM** or more tuned RandomForest.
- If it’s time‑dependent → try a simple time‑aware split or lag feature.

### 1. Core study (45–60 min)

Notebook: `notebooks/week11_day4_alternative_model.ipynb`.

1. **Decide what to try (markdown)**

Pick ONE from:

- `Lasso` / `Ridge` / `ElasticNet` (regression)
- `LogisticRegression` with stronger regularization / penalty
- Another tree method you’ve not tried (e.g., XGBoost / LightGBM)  
  (only if you’re comfortable installing extra libs)

Write:

- Why you picked this.
- What you **hope** it might achieve that current champion doesn’t (e.g., sparsity/interpretability vs raw performance).

2. **Implement using same preprocessor**

Example: L1‑regularized LogisticRegression (classification):

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

lasso_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        penalty="l1",
        solver="saga",
        C=0.5,  # regularization strength
        max_iter=2000
    ))
])
```

Or XGBoost (if you choose it):

```python
# pip install xgboost
from xgboost import XGBClassifier

xgb_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    ))
])
```

3. **Evaluate quickly**

Use:

- `cross_val_score` on full `X, y` (cv=5).
- Test performance on `X_test, y_test`.

### 2. Core practice (30–45 min)

1. **Compare with champion model**

Create a table:

| Model           | CV mean metric | Test metric | Notes (pros/cons)        |
|----------------|----------------|------------|--------------------------|
| Champion (GBM) | …              | …          | Better non-linearity     |
| New (L1 LR)    | …              | …          | More interpretable       |

2. **Decide if it’s worth keeping**

If performance is:

- Clearly worse → treat as learning experiment, not final model.
- Similar but simpler → consider making it your **“interpretable baseline”** to show alongside champion.
- Better → consider promoting it to new champion (but be honest in report).

### 3. Stretch (optional)

- For L1/Lasso: inspect which coefficients are non-zero → can lead to a **sparse** feature set.
- For XGBoost/LightGBM: look at feature importance and compare to RF/HistGB.

### 4. Reflection

- Did trying this alternative teach you something new about your data or features?
- If performance is similar, which model would you rather deploy and maintain, and why?

---

## Day 5 – Testing & Code Quality: Minimal but Real

**Goal:** Add **basic tests** for your key functions (especially `predict_one`) and reduce the risk of silent breakage.

### 1. Core study (45–60 min)

Create:

- `tests/` folder.
- File: `tests/test_predict_one.py`.

You can use either `pytest` (recommended) or simple `assert` statements.

1. **Think about “contracts”**

In markdown, define what `predict_one` must satisfy:

- Input: dict with specific keys; values of certain types/ranges.
- Output: dict containing at least `prediction` and maybe `probability`.
- Must **not crash** on a typical valid input.

2. **Write simple tests**

If using `pytest`:

```python
# tests/test_predict_one.py
from src.predict_one import predict_one

def test_predict_one_basic():
    example = {
        "age": 30,
        "gender": "Male",
        "monthly_charges": 70.0,
        # ... fill required fields ...
    }
    result = predict_one(example)
    assert "prediction" in result
    assert result["prediction"] is not None

    if "probability" in result:
        assert 0.0 <= result["probability"] <= 1.0
```

Add a test for **missing field** if you like:

```python
import pytest

def test_predict_one_missing_feature():
    example = {
        # intentionally incomplete
        "age": 30,
    }
    with pytest.raises(KeyError):
        predict_one(example)
```

(or, if your code handles it more gracefully, assert that behavior.)

### 2. Core practice (30–45 min)

1. **Run tests**

Install pytest if needed:

```bash
pip install pytest
pytest
```

Fix any failing tests or obvious issues.

2. **Basic refactor if needed**

If `predict_one` is messy, refactor slightly to:

- Separate loading of model from prediction logic.
- Maybe define a small `INPUT_SCHEMA` list of required keys and check them.

### 3. Stretch (optional)

- Add **one test** for your training pipeline:
  - Ensure that re-running training produces a model file.
  - Optionally, check that test metrics are within a reasonable range (not exactly equal, but above some threshold).

### 4. Reflection

- Which bug or fragility did you discover while writing tests?
- How many tests do you think you *really* need for this project to feel safe?

---

## Day 6 – Final Written Report & Narrative

**Goal:** Turn your work into a **cohesive narrative**: a final report/notebook/markdown that tells the story from problem to solution.

### 1. Core study (45–60 min)

Create:

- `notebooks/final_capstone_report.ipynb` (or `.md` if you prefer plain text).

1. **Outline your report (markdown)**

Suggested structure:

1. Introduction
   - Problem description
   - Why it matters
2. Data
   - Source, size, key features
   - Any cleaning you did
3. Methodology
   - Baselines (naive + rule-based)
   - Preprocessing (pipelines)
   - Models tried
4. Evaluation
   - Metrics used and why
   - Cross-validation & test results
   - Subgroup/fairness analysis
5. Interpretation
   - Feature importances
   - Key relationships (PDPs or intuitive statements)
6. Robustness & Limitations
   - Stability, calibration, known weaknesses
7. Deployment & Usage
   - How to run `predict_one` or app
   - Example usage
8. Conclusion & Future Work

2. **Populate each section**

- Reuse **existing code cells** sparingly; focus on final results, not every experiment.
- Include a few key plots:
  - Target distribution
  - Performance comparison (bar chart of models)
  - Feature importance
  - Maybe one subgroup performance table

### 2. Core practice (30–45 min)

1. **Write in complete sentences**

- Ensure each section has 1–3 paragraphs of actual explanation, not just bullet points.
- Aim for clarity over formality; imagine a smart non‑ML colleague reading it.

2. **Add references / acknowledgments**

- Cite your data source (Kaggle, UCI, etc.).
- Mention any tutorials or documentation you leaned on heavily.

### 3. Stretch (optional)

- Export report to HTML or PDF (e.g., via Jupyter’s “Download as”).
- If you know Markdown well, consider making the main report a `REPORT.md` in the repo, with images saved in an `img/` folder.

### 4. Reflection

- If someone read only this report (no code), would they understand:
  - What you did?
  - How well it works?
  - When not to trust it?
- Which section feels weakest and may need another editing pass?

---

## Day 7 – Portfolio-Ready Repo & Final Week 11 Review

**Goal:** Make your capstone **portfolio-ready**: clean Git repo, clear README, and a short self‑evaluation.

### 1. Core study (45–60 min)

In your `capstone/` root:

1. **Clean directory structure**

You might end up with something like:

```text
capstone/
  data/          # raw (untracked or .gitignored if large)
  models/        # trained models (optionally gitignored)
  notebooks/
    final_capstone_report.ipynb
    ...
  src/
    predict_one.py
    train_model.py
  tests/
    test_predict_one.py
  requirements.txt
  README.md
  REPORT.md      # optional, or the final notebook serves this purpose
```

2. **Polish README**

Ensure it covers:

- Project title & short description
- Problem & data (1–2 paragraphs)
- Setup:
  - How to create environment (`pip install -r requirements.txt`)
- How to:
  - Run training from scratch
  - Run basic evaluation
  - Use the prediction interface (script/app)
- Link to:
  - Final report notebook / markdown
  - Key screenshots (if any)

3. **(Optional but recommended) Initialize Git and commit**

If you haven’t already:

```bash
git init
git add .
git commit -m "Initial capstone project commit"
```

You can push to GitHub later if/when you’re ready.

### 2. Core practice (30–45 min)

1. **Do a “fresh eyes” walkthrough**

Pretend you just cloned this repo:

- Read README top to bottom.
- Follow instructions to:
  - Install requirements.
  - Run training from scratch (or at least confirm instructions are correct).
  - Run prediction/app.

2. **Week 11 self‑review (markdown)**

Answer:

- What new *skills* did Week 11 add (vs Week 9–10)?
- What’s the **strongest part** of your capstone now (methodology, code, report, interface)?
- What’s the **weakest part**, and is it okay for a “first serious project”?

### 3. Stretch (optional)

- If comfortable, create a **GitHub repo** and push your project:
  - Add a short project description and tags.
  - Consider a simple “About this project” paragraph on the repo page.
- Optionally, write a brief blog‑style post (on a platform of your choice) summarizing your project.

### 4. Reflection

- On a 1–10 scale, how ready are you to show this project to:
  - A friend or peer?
  - A potential employer or mentor?
- What’s the ONE improvement you’d like to make in Weeks 12+ before you call it “v1.0 done”?

---

If you share your capstone **domain** (e.g., churn prediction, medical risk, price prediction), I can shape **Week 12** around either:

- Final polish + a second smaller project in the same domain, or  
- A “capstone of the capstone”: preparing interview‑style explanations and walk‑throughs using your project as your central example.
