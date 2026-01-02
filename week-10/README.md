Week 10 focuses on making your **capstone model “usable” in the real world**:

- Better use of probabilities / predictions (calibration, thresholds, error ranges)
- Robustness checks (stability, drift-like tests)
- Saving and loading the model cleanly
- A simple interface (script, CLI, or tiny web/app)
- Reproducibility and documentation

Assume ~1–2 hours/day.  
You’ll keep working in your `capstone/` project.

I’ll say “classification”; if you’re doing regression, I’ll note how to adapt.

Each day: **Goal → Core study → Core practice → Stretch → Reflection**.

---

## Day 1 – Re‑orient & Define “Production‑ish” Goals

**Goal:** Clarify what “ready to use” means for your capstone, and decide what you’ll build this week (calibration, API, app, etc.).

### 1. Core study (45–60 min)

Notebook: `notebooks/week10_day1_state_and_goals.ipynb`.

1. **Summarize current model (from Week 9)**

In code, quickly:

- Load your best model (or re-create quickly):

```python
import joblib
import pandas as pd

df = pd.read_csv("../data/your_file.csv")

# If you saved the model; otherwise re-create from Week 9 code
best_model = joblib.load("../models/best_model.joblib")
```

- Rebuild `X_train, X_test, y_train, y_test` the same way as Week 9.
- Print main metrics again (accuracy/ROC-AUC or RMSE/R²).

2. **Define “production‑ish” tasks in plain language**

In markdown, list goals such as:

- Reliable estimates of **how confident** the model is.
- Ability to **handle new data** with the same preprocessing.
- A simple **function or API** that others can call.
- Clear documentation of:
  - How to retrain.
  - How to evaluate.

3. **Decide this week’s deliverables**

Pick 2–3 deliverables, e.g.:

- For classification:
  - Calibrated probabilities + threshold selection.
  - A Python script `predict.py` that loads the model and predicts for new data.
  - A tiny web/demo app (Streamlit or Flask) *or* a CLI.

- For regression:
  - Better error analysis + “uncertainty bands” using residuals.
  - Same scripting/UI ideas.

Write them as checkboxes in markdown so you can tick them at the end of the week.

### 2. Core practice (30–45 min)

1. **Create a simple project “status” snapshot**

In markdown (and/or README), briefly capture:

- Data size, target, type (classification/regression).
- Current best model:
  - Model type + key hyperparameters.
  - Latest CV metric and test metric.
- Known weaknesses (e.g., certain segments, imbalance).

2. **Identify critical risks**

Write 3–5 bullet points:

- Data issues (missing, drift, outliers).
- Performance concerns (low recall in class 1, big errors for large amounts).
- Misuse risks (using predictions out of domain).

### 3. Stretch (optional)

- Sketch a very simple **architecture diagram** (even as text/markdown) showing:
  - Inputs → preprocessing → model → outputs (predictions/probabilities).
  - Where a user, API, or batch process fits in.

### 4. Reflection

- What does “good enough to use” actually mean for your problem?
- What do you feel most nervous about: performance, robustness, or ease of use?

---

## Day 2 – Classification: Calibration & Thresholds  
*(Regression: error distribution & simple uncertainty)*

**Goal:** Learn how well your model’s **probabilities** or predictions reflect reality, and pick meaningful **decision thresholds** or error ranges.

### 1. Core study (45–60 min)

Notebook: `notebooks/week10_day2_calibration_and_thresholds.ipynb`.

Use your **current best model** from Week 9.

#### A. Classification: Calibration & Thresholds

1. **Get predicted probabilities**

```python
from sklearn.metrics import brier_score_loss, roc_auc_score

y_test_proba = best_model.predict_proba(X_test)[:, 1]
```

2. **Calibration curve**

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)

plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration curve")
plt.show()
```

- Perfect calibration: points near the diagonal.

3. **Brier score**

```python
brier = brier_score_loss(y_test, y_test_proba)
roc = roc_auc_score(y_test, y_test_proba)
brier, roc
```

4. **Try a calibration model (Platt or isotonic)**

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_clf = CalibratedClassifierCV(best_model, method="isotonic", cv=3)
calibrated_clf.fit(X_train, y_train)

y_test_proba_cal = calibrated_clf.predict_proba(X_test)[:, 1]
brier_cal = brier_score_loss(y_test, y_test_proba_cal)
roc_cal = roc_auc_score(y_test, y_test_proba_cal)
brier, brier_cal, roc, roc_cal
```

Plot new calibration curve.

5. **Threshold analysis**

Try several thresholds:

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

thresholds = np.linspace(0.1, 0.9, 9)
rows = []
for t in thresholds:
    y_pred_t = (y_test_proba_cal >= t).astype(int)  # use calibrated if you like
    rows.append({
        "threshold": t,
        "accuracy": accuracy_score(y_test, y_pred_t),
        "precision": precision_score(y_test, y_pred_t),
        "recall": recall_score(y_test, y_pred_t),
        "f1": f1_score(y_test, y_pred_t)
    })

import pandas as pd
thresh_df = pd.DataFrame(rows)
thresh_df
```

#### B. Regression: Error distribution & uncertainty (if applicable)

If regression, instead:

- Compute residuals and plot:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

y_pred = best_model.predict(X_test)
residuals = y_test - y_pred
plt.hist(residuals, bins=30)
plt.title("Residuals distribution")
plt.show()

rmse = mean_squared_error(y_test, y_pred, squared=False)
rmse
```

- Look at how residuals depend on predicted values or key features.
- For a simple uncertainty band: approximate e.g. 90% of errors within some ± range.

### 2. Core practice (30–45 min)

1. **Pick an operating threshold (classification)**

In markdown:

- Decide which trade‑off you care about (e.g., high recall vs balanced F1).
- Choose a threshold from `thresh_df` that aligns with that.
- Justify: “We choose threshold 0.X because …”

2. **Error band (regression)**

- Compute 10th and 90th percentiles of residuals.
- Say: “For most predictions, we’re within roughly [q10, q90] of true.”

### 3. Stretch (optional)

- For classification, plot a **precision–recall curve** and pick operating point visually.
- For regression, compute **prediction intervals** by grouping residuals by predicted bins.

### 4. Reflection

- Are your model’s predicted probabilities **overconfident** or **underconfident**?
- How different is your ideal threshold from the usual 0.5, and what does that mean for your use case?

---

## Day 3 – Robustness: Stability Across Seeds & Data Perturbations

**Goal:** Check whether your model’s performance is **stable** under small changes: random seeds, shuffles, and minor noise.

### 1. Core study (45–60 min)

Notebook: `notebooks/week10_day3_robustness.ipynb`.

1. **Performance across random seeds**

Define a helper to train + eval your champion model:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

def train_eval(seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y if classification else None
    )
    model = best_model  # or re-create with given seed if necessary
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if classification:
        return accuracy_score(y_test, y_pred)
    else:
        return -mean_squared_error(y_test, y_pred, squared=False)  # negative RMSE for convenience
```

Try multiple seeds:

```python
seeds = [0, 1, 2, 3, 4, 42, 123]
scores = [train_eval(s) for s in seeds]
scores
```

2. **Interpret seed stability**

- Compute mean and std of scores.
- Large std → performance sensitive to split/model initialization.

3. **Noise robustness (numeric features)**

Simple test:

- Add small Gaussian noise to numeric features in test set:

```python
import numpy as np

X_test_noisy = X_test.copy()
for col in numeric_features:
    X_test_noisy[col] = X_test_noisy[col] + np.random.normal(0, 0.01 * X_test_noisy[col].std(), size=len(X_test_noisy))

y_pred_original = best_model.predict(X_test)
y_pred_noisy = best_model.predict(X_test_noisy)
```

Compare:

- Classification: % of predictions that changed.
- Regression: distribution of differences between original and noisy predictions.

### 2. Core practice (30–45 min)

1. **Record stability metrics**

In markdown:

- Seed experiments:
  - Mean score, std, min, max.
  - Is this acceptable for your use case?
- Noise experiments:
  - Classification: what fraction of predictions flip?
  - Regression: typical absolute change in prediction.

2. **Identify fragile regions**

If feasible:

- For cases where predictions change under noise:
  - Inspect some examples.
  - Are they near threshold / boundary?

### 3. Stretch (optional)

- Try training with **less data** (e.g., 50% of training set) and see how performance changes.
- Consider simple **bootstrapping**: repeatedly sample with replacement from train, train model, evaluate.

### 4. Reflection

- Do you consider your model **robust enough** to small data changes and noise?
- Where is it most fragile, and is that acceptable for your context?

---

## Day 4 – Model Packaging: Save, Load, and Predict for a Single User / Row

**Goal:** Wrap your model into a **clean Python interface**: saving, loading, and predicting for a single new example.

### 1. Core study (45–60 min)

Create a script and a small notebook:

- Notebook: `notebooks/week10_day4_model_packaging.ipynb`
- Script: `src/predict_one.py`

1. **Consistent saving of final model**

From your Week 9 summary or updated best model:

```python
import joblib

joblib.dump(best_model, "../models/final_model_week10.joblib")
```

2. **Design a simple input schema**

In markdown:

- List the fields your model expects (raw features), e.g.:

  - `age` (int)
  - `gender` (str)
  - `monthly_charges` (float)
  - etc.

This is what a user (or API) must provide.

3. **Write a small prediction function in a script**

Create `src/predict_one.py` with something like:

```python
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_model_week10.joblib"

model = joblib.load(MODEL_PATH)

def predict_one(input_dict):
    """
    input_dict: Python dict of raw feature values for one example.
    Returns model prediction (and probability for classification if available).
    """
    X = pd.DataFrame([input_dict])  # single-row DataFrame
    y_pred = model.predict(X)[0]
    result = {"prediction": y_pred}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0, 1]
        result["probability"] = float(proba)
    return result

if __name__ == "__main__":
    # simple manual test
    example = {
        # fill with reasonable example feature values
    }
    print(predict_one(example))
```

### 2. Core practice (30–45 min)

1. **Test prediction function in notebook**

In `week10_day4_model_packaging.ipynb`:

```python
from src.predict_one import predict_one

example = {
    # same fields you listed in the schema; fill with meaningful dummy values
}
predict_one(example)
```

Check:

- Output format.
- Reasonableness of prediction/probability.

2. **Edge case test**

Try:

- Missing a field.
- Extra field not used by model.

Notice how your code behaves.  
(For now, catching these with simple `KeyError`/logging is fine; just notice the issues.)

### 3. Stretch (optional)

- Add some very basic **argument parsing** to `predict_one.py` so you can run:

```bash
python src/predict_one.py --age 42 --gender "Male" --monthly_charges 79.5
```

(using `argparse`).

### 4. Reflection

- If you gave this function to a teammate, would they know how to use it?
- What’s the weakest point right now: missing input validation, unclear schema, or something else?

---

## Day 5 – Simple Interface: CLI, Notebook “App”, or Tiny Web App

**Goal:** Build a very simple **user interface** to interact with your model (even if it’s just for you).

Pick **one** of:

- CLI script
- Notebook “app” (input widgets)
- Simple web app (Streamlit or Flask/FastAPI)

### 1. Core study (45–60 min)

Option A: **CLI / notebook UI** (simpler, stick to Python only).

Notebook: `notebooks/week10_day5_simple_interface.ipynb`.

1. **Notebook UI (Jupyter widgets, simple)**

- Use `input()` or `ipywidgets` (if installed) to get user input:

```python
from src.predict_one import predict_one

age = int(input("Age: "))
# ask for other features...
# then:
example = {
    "age": age,
    # ...
}
result = predict_one(example)
print(result)
```

Or with `ipywidgets` (optional):

```python
import ipywidgets as widgets
from IPython.display import display

age_widget = widgets.IntSlider(min=18, max=80, step=1, description="Age")
display(age_widget)

def on_click(_):
    example = {
        "age": age_widget.value,
        # ...
    }
    print(predict_one(example))

button = widgets.Button(description="Predict")
button.on_click(on_click)
display(button)
```

2. **CLI wrapper (if you prefer)**

Extend `predict_one.py` with argument parsing, as in the stretch from Day 4.

Option B: **Tiny web app** (Streamlit, easier for beginners).

Install:

```bash
pip install streamlit
```

Create `app.py` in `capstone/`:

```python
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/final_model_week10.joblib")
model = joblib.load(MODEL_PATH)

st.title("Capstone Model Demo")

# simple input widgets
age = st.number_input("Age", min_value=0, max_value=120, value=30)
# add more inputs...

if st.button("Predict"):
    example = {
        "age": age,
        # ...
    }
    X = pd.DataFrame([example])
    y_pred = model.predict(X)[0]
    st.write("Prediction:", y_pred)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0, 1]
        st.write("Probability:", float(proba))
```

Run:

```bash
streamlit run app.py
```

### 2. Core practice (30–45 min)

1. **Use the interface for 3–5 examples**

- Try “typical” and “extreme” input values.
- Note whether outputs align with your expectations from EDA and interpretability.

2. **Document how to run it**

Update README:

- How to start the app or use the CLI.
- Brief description of inputs and outputs.

### 3. Stretch (optional)

- Add simple **input validation**:
  - Check ranges.
  - Check that categorical values are from allowed sets.
- For Streamlit: add a chart or explanation (e.g., feature importance text) based on prediction.

### 4. Reflection

- How close does this feel to something a non‑technical user could try?
- What’s still missing before you’d dare show it to someone else?

---

## Day 6 – Reproducibility: Environments, Seeds, and Clear “Run From Scratch”

**Goal:** Make your project **reproducible**: environment, random seeds, and clear steps to run from raw data to model.

### 1. Core study (45–60 min)

Notebook: `notebooks/week10_day6_reproducibility.ipynb`.

1. **Environment capture**

- Create `requirements.txt`:

```bash
pip freeze > requirements.txt
```

(or manually list main libs: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, etc.)

2. **Random seed policy**

In code, define a global `RANDOM_STATE = 42` constant and use it consistently:

- For `train_test_split`, model constructors, etc.

3. **“Run from scratch” script or notebook**

Decide:

- Either:
  - `notebooks/run_from_scratch.ipynb`, or
  - `src/train_model.py`.

This should:

1. Load raw data from `data/`.
2. Apply minimal cleaning and feature engineering.
3. Split data.
4. Build preprocessor + model pipeline.
5. Fit model.
6. Evaluate (print metrics).
7. Save model to `models/`.

### 2. Core practice (30–45 min)

1. **Actually run the “from scratch” flow**

- Delete or move any pre‑made model file.
- Run your script/notebook and confirm:
  - Model is trained.
  - Metrics are printed.
  - Model file is saved.
  - `predict_one` and/or `app.py` still work with the new model.

2. **Update README**

Add a **“Reproducibility”** section:

- How to create environment.
- How to run training from scratch.
- How to run tests/evaluation (if you have them).

### 3. Stretch (optional)

- If using Git:
  - Commit your `requirements.txt` and training script.
  - Tag a version (e.g., `v0.1`).
- Add a very basic **test** (e.g., `tests/test_predict_one.py`) that checks `predict_one` runs on a sample input and returns expected fields.

### 4. Reflection

- If you re‑ran everything on another machine, what might still break?
- What’s the clearest missing piece for full reproducibility?

---

## Day 7 – Week 10 Milestone: “Ready Enough to Demo” & Roadmap

**Goal:** Consolidate Week 10 into a more “demo‑ready” capstone and plan the final polishing steps for Weeks 11–12.

### 1. Core study (45–60 min)

Notebook: `notebooks/week10_milestone_summary.ipynb`.

1. **Smoke test full flow**

In this notebook:

- Import / call your **training script/notebook** or re-run key steps.
- Load the saved final model.
- Run `predict_one` on a few examples.
- If you built an app:
  - Just note how to run it; no need to start here, but ensure instructions are correct.

2. **Summarize Week 10 additions (markdown)**

Under headings:

- **Calibration / Thresholds**:
  - What you did.
  - Chosen threshold or uncertainty description.
- **Robustness**:
  - Stability across seeds, noise tests.
- **Packaging & Interfaces**:
  - `predict_one` function.
  - CLI / notebook UI / Streamlit or Flask app.
- **Reproducibility**:
  - requirements.txt
  - training-from-scratch script/notebook.
  - model saving/loading.

### 2. Core practice (30–45 min)

1. **Write a “Demo Script” (for yourself)**

In markdown, outline what you would do in a 5–10 minute live demo:

1. Briefly explain problem and data.
2. Show baseline vs best model performance.
3. Show one interpretability plot (feature importance or PDP).
4. Show the interface (CLI/app) making 1–2 predictions.
5. Mention limitations and next steps.

2. **Rank priorities for the last 2 weeks (11–12)**

Examples:

- Better feature engineering.
- Trying another model family.
- Improving documentation & tests.
- Hardening the app / deployment.
- Fairness analysis / ethical considerations.

Rank by:

- Importance (1–5)
- Feasibility in your available time.

### 3. Stretch (optional)

- Record a short screen capture (even just for you) walking through:
  - Notebook summary.
  - The interface.
  - A quick explanation of results.

This will clarify where explanations or flows are confusing.

### 4. Reflection

- On a 1–10 scale, how “demo‑ready” is your capstone right now?
- What’s the one thing you’re **most proud of** in this project so far, and the **one thing** you most want to fix or improve before the end of Week 12?

---

If you share whether your capstone is **classification or regression** and its domain (e.g., churn, health, prices), I can tailor **Week 11** toward the most relevant final improvements: e.g., advanced calibration & decision rules (classification), or better error modeling & time-awareness (regression), plus more polished deployment.
