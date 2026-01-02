Week 6 will introduce **unsupervised learning**:  
you’ll learn **k-means clustering**, **cluster evaluation**, and **PCA** (dimensionality reduction), and apply them in a small **customer segmentation mini‑project**.

Assume ~1–2 hours/day.

We’ll mainly use:

- A simple clustering dataset: **Mall Customers** from Kaggle (or a similar customer dataset).
- Optionally: synthetic data (from scikit-learn) for intuition.

Each day: **Goal → Core study → Core practice → Stretch → Reflection**.  
Keep asking: *“What structure is the algorithm finding, and does it make sense?”*

---

## Day 1 – Unsupervised vs Supervised & First k-means on Toy Data

**Goal:** Understand what unsupervised learning is and run your **first k-means clustering** on simple synthetic data.

### 1. Core study (45–60 min)

Create `week6_day1_kmeans_toy.ipynb`.

1. **Concepts (markdown, think first)**

Write a short note (in your own words):

- **Supervised learning**: we have features `X` and labels `y`. Models learn to map X → y.
- **Unsupervised learning**: we only have `X`. Models find **patterns/structure** on their own.
- **Clustering**: group similar points together (e.g., similar customers).
- Contrast:
  - Supervised: “Will this customer churn?” (needs labels).
  - Unsupervised: “What types of customers do I have?” (no labels).

2. **Generate synthetic blobs**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y_true = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=0.60,
    random_state=42
)

plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title("Toy data (3 clusters)")
plt.show()
```

3. **Fit k-means**

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_
centers = kmeans.cluster_centers_
```

4. **Visualize clusters & centers**

```python
plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap="viridis")
plt.scatter(centers[:, 0], centers[:, 1], c="red", s=200, marker="X")
plt.title("k-means clustering (k=3)")
plt.show()
```

5. **Concepts to note**

In markdown, summarize briefly:

- k-means idea:
  - Randomly initialize k centroids.
  - Assign each point to nearest centroid.
  - Recompute centroids.
  - Repeat until convergence.
- It minimizes **within-cluster sum of squared distances**.

### 2. Core practice (30–45 min)

1. **Different k values**

Try `k=2`, `k=3`, `k=5`:

```python
for k in [2, 3, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    labels_k = km.labels_
    centers_k = km.cluster_centers_
    
    plt.scatter(X[:, 0], X[:, 1], c=labels_k, s=30, cmap="viridis")
    plt.scatter(centers_k[:, 0], centers_k[:, 1], c="red", s=200, marker="X")
    plt.title(f"k-means with k={k}")
    plt.show()
```

In markdown:

- Which k **looks** best on this toy dataset?
- What happens when k is too small vs too large?

2. **Distance intuition**

Pick one point `X[0]` and compute distances to all centers:

```python
from numpy.linalg import norm

x0 = X[0]
for i, c in enumerate(centers):
    print(i, norm(x0 - c))
```

- Confirm it’s assigned to the **closest** center.

### 3. Stretch (optional)

- Use `init="random"` vs default `init="k-means++"` and see if centers change noticeably.
- Change `cluster_std` in `make_blobs` (e.g. to 1.5) and see how well k-means separates clusters.

### 4. Reflection

- In your own words: what is k-means trying to optimize?
- Why do you think k-means works well on the blob dataset, but might struggle on weird shapes (e.g., two moons)?

---

## Day 2 – k-means on Real Data: Mall Customers (Part 1)

**Goal:** Run k-means on the **Mall Customers** dataset and see how scaling affects clustering.

### 1. Core study (45–60 min)

Create `week6_day2_mall_kmeans_basic.ipynb`.

1. **Get dataset**

- Kaggle: “Mall Customers” dataset (usually `Mall_Customers.csv`).
- Place in your `week6/` folder.

2. **Load & inspect**

```python
import pandas as pd

df = pd.read_csv("Mall_Customers.csv")  # adjust filename if needed
df.head()
df.info()
df.describe()
```

Usually columns: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`.

3. **Select features for clustering**

Start simple: only numeric, and drop ID:

```python
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]
X.head()
```

4. **k-means without scaling**

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_
df["cluster_raw"] = labels
df["cluster_raw"].value_counts()
```

5. **Visualize 2D projections**

Plot `Annual Income` vs `Spending Score` colored by cluster:

```python
import matplotlib.pyplot as plt

plt.scatter(
    df["Annual Income (k$)"], 
    df["Spending Score (1-100)"], 
    c=df["cluster_raw"], cmap="viridis", s=50
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Clusters (no scaling)")
plt.show()
```

### 2. Core practice (30–45 min)

1. **Effect of scaling**

Many algorithms (including k-means) are distance-based and sensitive to feature scale.

Use `StandardScaler`:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Fit k-means with `k=4` again:

```python
kmeans_scaled = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_scaled.fit(X_scaled)
df["cluster_scaled"] = kmeans_scaled.labels_
df["cluster_scaled"].value_counts()
```

Plot again for same 2D projection (color by new clusters):

```python
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["cluster_scaled"], cmap="viridis", s=50
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Clusters (with scaling)")
plt.show()
```

In markdown:

- How did scaling change the clusters (visually & by counts)?

2. **Cluster centers (interpretation)**

For scaled data:

```python
centers_scaled = kmeans_scaled.cluster_centers_
centers_scaled
```

Convert them back to original scale:

```python
centers_original = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers_original, columns=features)
centers_df
```

Interpret each cluster center:

- Age, income, spending score – what kind of customer does each represent?

### 3. Stretch (optional)

- Try `k=3`, `k=5` with scaled data, and visually compare.
- Check whether some clusters are **very small** (potentially unstable) vs others.

### 4. Reflection

- Why is scaling especially important for k-means?
- Which feature (age, income, spending) seems to dominate the segmentation in your plots?

---

## Day 3 – Choosing k: Elbow Method & Silhouette Score

**Goal:** Use simple metrics (distortion / inertia and **silhouette score**) to choose the number of clusters k more systematically.

### 1. Core study (45–60 min)

Create `week6_day3_mall_k_selection.ipynb`.

Use `X_scaled` from Day 2.

1. **Inertia (within-cluster SSE) and elbow method**

```python
from sklearn.cluster import KMeans

ks = range(2, 11)
inertias = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)  # sum of squared distances to cluster centers

inertias
```

Plot:

```python
import matplotlib.pyplot as plt

plt.plot(ks, inertias, marker="o")
plt.xlabel("k (number of clusters)")
plt.ylabel("Inertia (within-cluster SSE)")
plt.title("Elbow plot")
plt.show()
```

2. **Silhouette score**

```python
from sklearn.metrics import silhouette_score

sil_scores = []

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)

sil_scores
```

Plot silhouette scores:

```python
plt.plot(ks, sil_scores, marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette score")
plt.title("Silhouette vs k")
plt.show()
```

### 2. Core practice (30–45 min)

1. **Interpret elbow & silhouette**

In markdown:

- From the elbow plot: where does the curve start to “bend” or flatten significantly?
- From the silhouette plot: which k gives the highest silhouette score?

2. **Choose a k and refit**

Choose a k based on both plots (e.g., 4 or 5). Refit k-means with that k:

```python
best_k = ...  # your chosen value
best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
best_kmeans.fit(X_scaled)
df["cluster_best"] = best_kmeans.labels_
```

Compute silhouette score for this specific model:

```python
best_sil = silhouette_score(X_scaled, df["cluster_best"])
best_sil
```

3. **Group-level stats**

Group by cluster and compute means:

```python
cluster_summary = df.groupby("cluster_best")[features].mean()
cluster_summary
```

In markdown, describe each cluster qualitatively.

### 3. Stretch (optional)

- Plot silhouette **per sample** for your chosen k (a bit more advanced; optional).  
  You can look up `sklearn.metrics.silhouette_samples` and standard silhouette plot examples.
- Try re-running the whole k-selection process **without scaling** and compare metrics.

### 4. Reflection

- Are the “best” clusters (by silhouette) also the most interpretable to you?
- If business wanted exactly 3 customer segments, would you stick to the mathematical best k or adapt? Why?

---

## Day 4 – Intro to PCA: Dimensionality Reduction & Explained Variance

**Goal:** Understand **PCA** conceptually and use it to reduce Mall Customers (or another numeric dataset) to 2D for visualization.

### 1. Core study (45–60 min)

Create `week6_day4_pca_intro.ipynb`.

Use `X_scaled` (Mall Customers numeric features).

1. **Conceptual notes (markdown)**

Write in your own words:

- PCA finds new axes (principal components) that:
  - Are linear combinations of original features.
  - Capture **maximum variance**.
- First component explains the most variance, second explains next most, etc.
- Common uses:
  - Visualization (2D/3D).
  - Noise reduction.
  - Decorrelating features.

2. **Fit PCA**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3)  # ask for 3 components for now
pca.fit(X_scaled)

pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum()
```

3. **Transform data to 2D**

```python
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)
X_pca_2d.shape
```

4. **Visualize PCA space**

```python
plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], s=50)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Mall Customers in PCA space")
plt.show()
```

Optionally color by chosen cluster (`cluster_best`):

```python
plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=df["cluster_best"], cmap="viridis", s=50
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clusters visualized in PCA space")
plt.show()
```

### 2. Core practice (30–45 min)

1. **Explained variance and choosing components**

Plot cumulative explained variance:

```python
pca_full = PCA().fit(X_scaled)

cum_var = pca_full.explained_variance_ratio_.cumsum()
plt.plot(range(1, len(cum_var) + 1), cum_var, marker="o")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Explained variance vs #components")
plt.ylim(0, 1.05)
plt.show()
```

In markdown:

- How many components are needed to explain, say, **90%** of variance?
- Is going from 3D → 2D losing a lot of information here?

2. **Component loadings (basic interpretation)**

Inspect how original features contribute to components:

```python
components = pd.DataFrame(
    pca_2d.components_,
    columns=features,
    index=["PC1", "PC2"]
)
components
```

In markdown:

- Which features contribute most (in absolute value) to PC1 & PC2?

### 3. Stretch (optional)

- Try PCA on a higher-dimensional dataset (e.g., Adult Income numeric features only) and do a 2D scatter plot colored by income_binary, just to see separation.
- Try setting `whiten=True` in PCA and see how that changes component values (more advanced).

### 4. Reflection

- What does it **mean** that PC1 explains, for example, 50–60% of variance?
- How might PCA help before clustering in high-dimensional data?

---

## Day 5 – k-means + PCA: Clustering in Reduced Space

**Goal:** Combine PCA and k-means: cluster in PCA-reduced space and compare to clustering in original space.

### 1. Core study (45–60 min)

Create `week6_day5_pca_kmeans_pipeline.ipynb`.

Use `X_scaled` and `X_pca_2d` from previous days (or recompute).

1. **k-means on PCA-reduced data**

```python
from sklearn.cluster import KMeans

k = df["cluster_best"].nunique()  # use your chosen k from Day 3, e.g. 4
kmeans_pca = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_pca = kmeans_pca.fit_predict(X_pca_2d)

df["cluster_pca"] = labels_pca
```

2. **Visualize clusters in PCA space**

```python
plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=df["cluster_pca"], cmap="viridis", s=50
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("k-means clustering in PCA space")
plt.show()
```

3. **Compare silhouette scores**

```python
from sklearn.metrics import silhouette_score

sil_original = silhouette_score(X_scaled, df["cluster_best"])
sil_pca = silhouette_score(X_pca_2d, df["cluster_pca"])
sil_original, sil_pca
```

### 2. Core practice (30–45 min)

1. **Compare cluster assignments**

How many points have the same cluster in both methods?

```python
(df["cluster_best"] == df["cluster_pca"]).mean()
```

(Not expected to be very high; just an interesting sanity check.)

2. **Cluster summary in PCA-based clustering**

```python
cluster_summary_pca = df.groupby("cluster_pca")[features].mean()
cluster_summary_pca
```

In markdown:

- Do PCA-based clusters have a similar **interpretation** (e.g., low income/high spenders) compared to original-space clusters?
- Which segmentation feels more meaningful?

3. **Build a small pipeline**

Use `Pipeline` to combine scaling, PCA, and k-means:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

k = ...  # chosen number of clusters

pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2)),
    ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10))
])

pipe.fit(df[features])
labels_pipe = pipe.named_steps["kmeans"].labels_
sil_pipe = silhouette_score(pipe.named_steps["pca"].transform(df[features]), labels_pipe)
sil_pipe
```

Compare `sil_pipe` with previous scores.

### 3. Stretch (optional)

- Try different `n_components` in PCA (e.g., 2 vs 3) and see if silhouette improves.
- For each `n_components`, record silhouette in a small table.

### 4. Reflection

- Did clustering after PCA improve, worsen, or roughly match your original clustering quality?
- In what situations would PCA + clustering be especially useful (think: many features, noise, correlations)?

---

## Day 6 – Mini Unsupervised Project (Part 1): Problem Framing & EDA

**Goal:** Start a **small unsupervised project** (e.g., customer segmentation) and do EDA + basic clustering.

You can either:

- Continue with **Mall Customers** but pretend you’re doing this for a business stakeholder, or
- Pick another dataset (e.g., credit card customers, retail data).

### 1. Core study (45–60 min)

Create `week6_day6_unsupervised_project_part1.ipynb`.

1. **Problem definition (markdown)**

Write a clear problem statement, for example:

- “Segment mall customers into behavior-based groups using age, income, and spending patterns.”
- State why this is useful:
  - Targeted marketing.
  - Personalized promotions.
  - Understanding customer types.

2. **Data loading and quick EDA**

- Load your dataset.
- Inspect columns, types, missing values.
- Choose an **initial set of features** for clustering (mostly numeric; possibly encode categorical if needed).

Example (Mall Customers):

```python
import pandas as pd

df = pd.read_csv("Mall_Customers.csv")
df.info()
df.describe()
```

3. **Univariate and bivariate EDA (light)**

- Histograms for main numeric features.
- Simple scatter plots (e.g., income vs spending).
- Check for outliers or obvious data issues.

4. **Preprocessing & initial clustering**

- Handle missing values (drop or simple imputation).
- Scale numeric features using `StandardScaler`.
- Run k-means with a small set of k values (e.g., 3–6), compute silhouette, pick one k for now.

### 2. Core practice (30–45 min)

1. **Choose a “working” k**

- Based on elbow/silhouette and interpretability, choose k.
- Save this as your **current candidate** number of clusters.

2. **Initial cluster profiling**

Group by cluster and compute:

- Means of key numeric features.
- If you have any categorical variables (e.g., Gender): highest proportion per cluster.

Describe each cluster in a sentence:

- “Cluster 0: young, low income, high spend.”
- “Cluster 1: older, high income, medium spend,” etc.

### 3. Stretch (optional)

- Try a simple PCA (2D) visualization of your clusters to see if they separate clearly.
- If using a dataset with more features, try removing one noisy or redundant feature and see if silhouette improves.

### 4. Reflection

- How would you explain your clusters to a **non-technical** stakeholder?
- What additional data (features) do you wish you had to make more meaningful segments?

---

## Day 7 – Mini Unsupervised Project (Part 2): Refinement & Short Report

**Goal:** Refine your clustering, compare a couple of variants (with/without PCA, different k), and write a short **unsupervised project report**.

### 1. Core study (45–60 min)

Continue in `week6_day7_unsupervised_project_part2.ipynb`.

1. **Refinement plan (markdown)**

Decide:

- What two or three variations you’ll compare, e.g.:
  1. k-means on scaled data with k = K1.
  2. k-means on scaled data with k = K2.
  3. k-means on PCA(2)-reduced data with k = K1 or K2.

Write that plan explicitly.

2. **Implement model variations**

For each variant:

- Fit k-means.
- Compute silhouette score.
- Store in a small table.

Example:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

results = []

# variant 1: original scaled features, k = k1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

k1 = ...
k2 = ...

for k in [k1, k2]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    results.append({"variant": f"scaled_k={k}", "silhouette": sil})

# variant 3: PCA + k-means
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
km_pca = KMeans(n_clusters=k1, random_state=42, n_init=10)
labels_pca = km_pca.fit_predict(X_pca)
sil_pca = silhouette_score(X_pca, labels_pca)
results.append({"variant": f"PCA2_scaled_k={k1}", "silhouette": sil_pca})

results_df = pd.DataFrame(results)
results_df
```

3. **Pick a final model**

Based on silhouette and interpretability, choose one setup as your **final** clustering approach.

### 2. Core practice (30–45 min)

1. **Write a short project report (10–15 sentences)**

In markdown, include:

- **1. Problem & Data**
  - What were you trying to do?
  - What data and features did you use?
- **2. Methods**
  - k-means, with which preprocessing (scaling, PCA or not).
  - How you chose k (elbow/silhouette + judgment).
- **3. Results**
  - Silhouette scores for the main variants.
  - Final chosen clustering and a short profile of each cluster.
- **4. Business Interpretation**
  - How each cluster could be targeted differently (e.g., discounts, VIP offers).
- **5. Limitations & Next Steps**
  - Data limitations.
  - Possible improvements (more features, other clustering methods, trying Gaussian Mixture Models, etc.).

2. **Self-quiz**

Without looking back, answer in your own words:

- What is the difference between **k-means** and **PCA** (purpose and outcome)?
- What does a **silhouette score** measure?
- Why is **scaling** especially important before k-means and PCA?

Then check against your notes.

### 3. Stretch (optional)

- Try clustering a completely different dataset (e.g., numeric subset of Adult Income) without using the target label at all, then see if clusters roughly align with income level when you check afterwards.
- Experiment with a different clustering algorithm (e.g., `AgglomerativeClustering`) and compare qualitatively.

### 4. Reflection

- Which part of unsupervised learning feels most intuitive, and which part feels most “magic”?
- Do you feel more comfortable now explaining **what clusters mean** rather than just “the algorithm produced labels 0/1/2/3”?  
- How might you use clustering or PCA in a personal project you care about (music, fitness, finance, etc.)?

---

If you want to continue, next I can outline **Week 7 (Day 1–7)** focusing on integrating everything into more **end-to-end workflows**, plus starting to frame and plan your **capstone supervised project**.
