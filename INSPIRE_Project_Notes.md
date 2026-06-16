# INSPIRE Project — Complete Understanding Guide

> Everything you need to know about the project, explained simply and in full detail.

---

## Table of Contents

1. [What is the INSPIRE Project?](#1-what-is-the-inspire-project)
2. [The Dataset — What the Data Actually Is](#2-the-dataset)
3. [The Three Models Being Compared](#3-the-three-models)
4. [The DNN Pipeline — Step by Step](#4-the-dnn-pipeline-step-by-step)
5. [How the Transformer Works](#5-how-the-transformer-works)
6. [Pre-Training and Transfer Learning](#6-pre-training-and-transfer-learning)
7. [Class Imbalance — The Core Problem](#7-class-imbalance)
8. [How We Measure Performance](#8-how-we-measure-performance)
9. [The Feature Correlation Matrix — What It Tells Us](#9-the-feature-correlation-matrix)
10. [The t-SNE Embeddings Plot — What It Tells Us](#10-the-tsne-embeddings-plot)
11. [Frailty and Saranya's Work](#11-frailty-and-saranayas-work)
12. [The Code — File by File](#12-the-code-file-by-file)
13. [Key Problems and What Needs Fixing](#13-key-problems-and-what-needs-fixing)
14. [What to Explore Next](#14-what-to-explore-next)

---

## 1. What is the INSPIRE Project?

**The goal:** Predict whether a surgical patient will die within 30 days of their operation.

This matters because surgeons and anaesthetists need to assess risk before an operation. If a patient is very high risk, the surgical team can take extra precautions, involve intensive care, or have a different conversation with the patient about the risks.

**The dataset:** INSPIRE is a real Korean hospital dataset containing records from over 100,000 surgical patients collected between 2011 and 2020.

**The research question:** Can machine learning — and specifically a deep neural network that reads time series data — do a better job of predicting surgical mortality than the existing clinical tools doctors use today?

**The comparison:** Three different approaches are tested:
- A fixed clinical scorecard (NELA)
- A gradient boosted machine (GBM) 
- A deep neural network with a transformer reading time series 

---

## 2. The Dataset

### What is in the INSPIRE dataset?

The full INSPIRE dataset has five tables, all linked by a patient ID:

| Table | What it contains | Example |
|---|---|---|
| `operations.csv` | One row per surgery — age, weight, anaesthesia type, department, outcome | Patient 156911 had surgery in General Surgery, survived |
| `labs.csv` | Blood test results over time | Glucose = 5.2 at minute 100, 6.1 at minute 400 |
| `vitals.csv` | Intraoperative measurements (during surgery) | Heart rate = 72 bpm at minute 210 |
| `ward_vitals.csv` | Pre and post-operative floor measurements | Blood pressure = 130/80 the morning before surgery |
| `medications.csv` | 9.9 million drug administrations | Fentanyl 50mcg at minute 215 |

### The `dnn_mortality.csv` file you received

This is a pre-processed, flattened version. Each row is one patient with 19 columns.

**Key facts:**
- **97,260 patients** total
- **268 deaths** (0.28% death rate)
- **Target column:** `inhosp_death_30day` — True if the patient died within 30 days of surgery

The 19 columns are:

| Column | What it means |
|---|---|
| `age` | Patient age in years |
| `sex` | Male = True |
| `emop` | Emergency operation? 1 = yes, 0 = no (very important risk factor) |
| `bmi` | Body mass index |
| `andur` | Anaesthesia duration in minutes |
| `preop_hb` | Haemoglobin — carries oxygen in blood |
| `preop_platelet` | Platelets — blood clotting |
| `preop_wbc` | White blood cells — immune system, infection marker |
| `preop_aptt` | Blood clotting test |
| `preop_ptinr` | Another clotting test |
| `preop_glucose` | Blood sugar |
| `preop_bun` | Blood urea nitrogen — kidney function |
| `preop_albumin` | Protein made by liver — nutritional status |
| `preop_ast` | Liver enzyme |
| `preop_alt` | Liver enzyme (very similar to AST) |
| `preop_creatinine` | Kidney function marker |
| `preop_sodium` | Electrolyte balance |
| `preop_potassium` | Electrolyte balance |
| `inhosp_death_30day` | **The outcome we are predicting** |

### Critical data problems

**Massive class imbalance:** Only 268 of 97,260 patients died. That is 0.28%. A model that always predicts "survived" would be 99.72% accurate but completely useless — it would miss every single death. The model needs special treatment (see Section 7).

**Missing values:** Many blood tests are not taken for every patient before surgery. For example, glucose is missing for 13,388 patients (13.8%) and albumin for 9,076 patients (9.3%). Before training any model, these gaps must be filled (imputed) or handled.

### Why time series matters

The CSV above is a **snapshot** — one value per feature, taken before surgery. It throws away all information about how values changed over time.

The DNN pipeline uses a different approach: it reads the **full time series** of lab values — glucose at hour 1, hour 5, hour 12, hour 24 — and tries to learn patterns from how values evolve. A patient whose glucose is rising steadily over days before surgery may be at different risk than one whose glucose is stable, even if the final value is the same.

This is the fundamental advantage the DNN is trying to demonstrate.

---

## 3. The Three Models

### Model 1 — NELA (the clinical gold standard)

**File:** `nela.py`

NELA stands for National Emergency Laparotomy Audit. It is a fixed mathematical equation developed by UK clinicians from large studies of surgical patients.

**How it works:** You plug in 25 values for a patient (age, albumin level, blood pressure, urgency of surgery, etc.) and get back a probability of death. The equation looks like:

```
logit = -3.04678
      + 0.06660 × age
      + 1.13007 × (ASA class 3)
      + 1.76293 × (ASA class 4)
      - 0.04323 × albumin
      + 0.38002 × log(urea)
      ... (25 variables total)

probability = 1 / (1 + e^(-logit))
```

**The key point:** No machine learning. No training. The coefficients were set by statisticians studying thousands of patients and published in a clinical document. This is how doctors currently assess risk — using tools like this.

**Pros:** Transparent, validated, fast, requires no data to run.
**Cons:** Rigid — cannot learn from new data. Only uses a snapshot. Cannot use time series. Fixed to the variables it was built with.

**What it scores:** Because it was not trained on INSPIRE data, its performance on the INSPIRE dataset gives us the baseline that any machine learning model must beat to be useful.

---

### Model 2 — GBM (Saranya's model)

**File:** `gbm_mortality.py`

GBM stands for Gradient Boosted Machine - comparing versions with and without frailty scores.

**How it works:** Think of it as building hundreds of decision trees, one at a time, where each new tree focuses specifically on fixing the mistakes of all the previous trees combined. Over many iterations, this ensemble of trees becomes very good at finding patterns in tabular data.

A decision tree is like a flowchart:
- Is the patient's albumin below 30? → If yes, go left (higher risk)
- Is the patient older than 70? → If yes, go left again (even higher risk)
- ...and so on until you reach a prediction

**What data it uses:** The 18 pre-operative features from `operations.csv` merged with the most recent lab values before surgery. Same snapshot approach as NELA but the model learns its own rules from the data rather than using fixed coefficients.

**current work:** Adding the Hospital Frailty Risk Score (HFRS) as an extra feature to the GBM and measuring whether it improves predictions — and whether this improvement differs for scheduled vs emergency surgery.

**Pros:** Better than NELA because it learns from data. Handles feature interactions automatically.
**Cons:** Still uses snapshots — ignores the temporal pattern of how values changed. Requires the original dataset to train (cannot be computed from first principles like NELA).

---

### Model 3 — DNN Transformer 

**File:** `dnn_mortality_pipeline.py`

This is a fundamentally different approach. Instead of looking at a single snapshot of lab values before surgery, it reads the full sequence of lab measurements over time and tries to learn patterns from the trajectory.

**What data it uses:** Four lab time series for each patient:
- Glucose (blood sugar)
- Potassium (electrolyte)
- Sodium (electrolyte)
- Creatinine (kidney function)

**Why these four:** They are measured frequently, available for most patients, and are clinically meaningful indicators of how sick a patient is.

**The core idea:** A patient who has a sudden spike in creatinine over the 3 days before surgery may be at much higher risk than a patient with the same absolute creatinine value but who has been stable. The transformer learns these patterns.

---

## 4. The DNN Pipeline Step by Step

The pipeline runs in two completely separate phases. Think of it like this: first you teach the model to understand how labs behave generally, then you teach it what patterns predict death.

### Step 1 — Generate or load data

In development mode (without the full INSPIRE dataset), `dnn_mortality_data.generate_dataset()` creates synthetic patients:
- 5,000 patients generated
- 20% are assigned as "died"
- Each patient gets random glucose, potassium, sodium, creatinine measurements over 1,000 time steps
- For patients who died, glucose and sodium have a slight upward slope added — just enough of a signal for the model to learn something

In production mode, `dnn_mortality_data.extract_frames()` reads the real INSPIRE subjects from their folder, pulling ward vitals and labs from 5 days before the last operation.

### Step 2 — Align the time series

Different patients have measurements at different times. Patient A might have glucose measured at minutes 100, 250, 400. Patient B at minutes 50, 300, 600. They need to be aligned to the same time grid so they can be processed together.

`align_time_series()` does this using **linear interpolation** — it draws a straight line between any two known points and estimates the values in between. Every patient ends up with values at every minute in a common time range.

For every interpolated value, a **mask column** is added:
- Mask = 1.0 → this value was actually measured
- Mask = 0.0 → this value was estimated by interpolation

This matters because the model should trust observed values more than interpolated ones.

So the pipeline goes from 4 feature columns to **8 columns total**: glucose, potassium, sodium, creatinine + glucose_mask, potassium_mask, sodium_mask, creatinine_mask.

### Step 3 — Normalise

Each column is scaled using `StandardScaler`:
- Subtracts the mean
- Divides by the standard deviation
- Result: every feature has mean 0, standard deviation 1

**Critical rule:** The scaler is fitted on training data only. Test data is transformed using the training data's mean and standard deviation. This prevents data leakage — the model must not see any statistics from the test set.

### Step 4 — Split train/test

67% of subjects go to training, 33% to test. The split is stratified — it ensures the proportion of deaths is roughly the same in both sets.

### Phase 1 — Autoencoder pre-training

The training subjects' time series are broken into **sliding windows** of length `seq_length` (25 time steps, derived from the data). A window slides along the whole time series one step at a time, generating many training examples from each patient.

These windows are fed into the transformer with a simple objective: **compress then reconstruct**. The model reads 25 time steps of 8 values (4 features + 4 masks) and tries to output the 4 original feature values. If it can reconstruct them accurately, it must have learned something about how these labs behave together.

No mortality labels are used in this phase. The model learns purely from the structure of the time series themselves.

Loss function: Mean Squared Error (MSE) — how far off is the reconstruction?

### Phase 2 — Classification fine-tuning

The encoder weights from Phase 1 are **frozen** — they cannot change anymore.

A new classification layer (a single `Linear → sigmoid` layer) is added on top. This is the only part that gets trained in Phase 2.

Each subject's full time series is passed through the frozen encoder. The encoder's output is averaged across all time steps (mean pooling) to produce one embedding vector per patient. The classifier then learns: "given this embedding, how likely is this patient to die?"

Loss function: Binary Cross Entropy with Logits, with `pos_weight` set to the ratio of survived/died to compensate for the imbalance.

### Evaluation

After training, the model is evaluated on the held-out test set:
- **AUROC** — overall discrimination ability
- **AUPRC** — precision-recall, more informative when deaths are rare
- **Best F1 threshold** — scans thresholds from 0 to 1 to find the probability cutoff that gives the best F1 score
- **t-SNE embeddings** — visualise whether died and survived patients end up in different regions of the embedding space

---

## 5. How the Transformer Works

### The basic problem

A standard neural network takes a fixed-size input (one vector of numbers) and produces an output. But a time series is a variable-length sequence — Patient A might have 500 time steps, Patient B might have 800.

Transformers were designed specifically to handle sequences of variable length.

### What attention means

At each time step, the transformer needs to decide: which other time steps are most relevant to understanding what is happening right now?

**Attention** is the mechanism that does this. For every time step in the sequence, the model computes a score for every other time step — "how much should I pay attention to that one when I am processing this one?" These scores are turned into weights, and the model takes a weighted combination of all time steps.

In practice, for a patient whose glucose spikes at day 3, the model might learn to pay high attention to the days immediately before and after the spike, and low attention to stable periods.

**8 attention heads** means the model runs this process 8 times in parallel, each head potentially learning to attend to different types of patterns (slow trends, sudden spikes, periodic patterns, etc.).

### Positional encoding

Transformers do not naturally know the order of inputs. Without help, they would treat time step 1 and time step 500 as interchangeable.

**Positional encoding** adds a unique numerical pattern to each time step's values, encoding its position. The formula uses sine and cosine waves at different frequencies:

```
PE(position, i) = sin(position / 10000^(2i / d_model))   for even i
PE(position, i) = cos(position / 10000^(2i / d_model))   for odd i
```

The result is that nearby time steps get similar encodings, and distant time steps get different ones. The model can now tell "this is step 50, this is step 200" without being told explicitly.

### The architecture in this project

```
Input: 8 values per time step (4 features + 4 masks)
  ↓
Input projection: Linear layer (8 → 8)
  ↓
+ Positional encoding (added element-wise, not concatenated)
  ↓
Transformer encoder:
  - 5 layers
  - 8 attention heads
  - Feedforward dimension 128
  ↓
Output: 8 contextual values per time step
        (same shape in, same shape out — but now each position
         knows about the context of all other positions)
```

For autoencoding: add an output projection layer (8 → 4) to reconstruct the 4 features.

For classification: take the mean across all time steps (mean pooling) to get one 8-dimensional vector per patient, then pass through `Linear(8 → 1)` to get the mortality probability.

---

## 6. Pre-Training and Transfer Learning

### Why pre-train at all?

The mortality prediction task has a fundamental problem: you have very few examples of the thing you are trying to predict. In the real INSPIRE data, only 268 patients out of 97,260 died. Training a complex neural network on 268 positive examples is very difficult — the model has almost nothing to learn from.

The autoencoder pre-training gets around this by using all patients. You do not need mortality labels to reconstruct lab values. Every patient — whether they died or survived — gives you a training example for the autoencoder. Once the encoder has learned the general structure of how labs behave, you use that knowledge as a starting point for the harder, label-scarce classification task.

This is **transfer learning**: learn one task (reconstruct labs) to build general knowledge, then transfer that knowledge to another task (predict death).

### The frozen encoder question

After pre-training, the encoder weights are frozen and only the classifier head is trained. This is a conservative approach — it assumes the encoder already learned everything useful and just needs a new output layer.

**The problem:** The autoencoder was trained to reconstruct lab values. A good reconstruction does not necessarily mean the encoder learned features that discriminate between patients who will die and patients who will survive. These are different objectives.

This is the most likely cause of the near-random classifier output (probabilities of ~0.47 for both classes) seen in the debug output.

**Possible fixes:**
1. Fine-tune the full network (unfreeze the encoder) on the classification task
2. Use a different pre-training objective that is more related to mortality — for example, contrastive learning where patients with similar outcomes are pushed together in embedding space
3. Use the encoder embeddings as features for the GBM instead of the linear classifier

---

## 7. Class Imbalance

This is the single most important practical problem in the project.

### The real numbers

| Dataset | Patients | Deaths | Death rate |
|---|---|---|---|
| Real INSPIRE (`dnn_mortality.csv`) | 97,260 | 268 | **0.28%** |
| Synthetic (used in pipeline) | 5,000 | ~1,000 | 20% (artificial) |

The synthetic data uses 20% deaths — 70 times higher than reality. This makes the model easier to train in development but means results on synthetic data do not reflect real-world performance.

### Why this is a problem

If 99.72% of patients survive, a model that always predicts "survived" achieves 99.72% accuracy. It would never flag any patient as high risk. Accuracy is completely useless as a metric here.

### How the pipeline handles it

`pos_weight` in `BCEWithLogitsLoss`. Setting `pos_weight = num_survived / num_died` tells the loss function: "getting a death wrong is 362 times more costly than getting a survival wrong." This forces the model to try harder to predict the rare class.

In the synthetic data with 20% deaths: pos_weight ≈ 4.
In the real data: pos_weight ≈ 362.

The difference between these two numbers is why the model might look fine on synthetic data but fail on real data.

---

## 8. How We Measure Performance

### AUROC — Area Under the ROC Curve

This is the main metric. AUROC answers the question: if you pick one random patient who died and one random patient who survived, what is the probability that your model gives the patient who died a higher risk score?

- 0.5 = random guessing (a coin flip)
- 0.7 = decent, better than random
- 0.8 = good for clinical use
- 0.9+ = excellent
- 1.0 = perfect

AUROC is useful because it does not depend on choosing a specific threshold and works well even with imbalanced classes.

### AUPRC — Area Under the Precision-Recall Curve

For very rare events (like 0.28% mortality), AUPRC is more informative than AUROC.

- **Precision:** Of all the patients the model flagged as high risk, what fraction actually died?
- **Recall:** Of all the patients who actually died, what fraction did the model catch?

These two trade off against each other. AUPRC measures performance across all possible thresholds.

A random model on data with 0.28% deaths has AUPRC ≈ 0.003. A good clinical model might achieve 0.2–0.4.

### F1 Score and Threshold Selection

The model outputs a probability (e.g., 0.73). You need to decide: above what probability do you call it a predicted death?

`evaluate_model()` scans thresholds from 0 to 1 and finds the one that gives the best F1 score. F1 is the harmonic mean of precision and recall — it balances both concerns.

### t-SNE — Visualising Whether the Model Learned Anything

t-SNE (t-distributed Stochastic Neighbour Embedding) takes the high-dimensional embedding vectors (8 dimensions in this model) and squashes them into 2D so they can be plotted.

If the pre-training worked well, patients who died (red) should cluster together and patients who survived (blue) should cluster together — meaning the encoder learned representations that separate the two groups.

---

## 9. The Feature Correlation Matrix

The correlation matrix (`FeatureCorrelationMatrix.png`) shows how strongly each pair of features is related. Values range from -1 (perfectly inversely related) to +1 (perfectly related).

### Key findings from your image

**AST and ALT: 0.90 correlation**
Both are liver enzymes. They move together because they are measuring essentially the same thing (liver damage). Using both in a GBM is redundant — you could drop `preop_alt` and lose almost no information.

**WBC and Albumin: -0.75 correlation**
High white blood cells = infection/inflammation. Low albumin = malnutrition or liver disease. These tend to co-occur in sick patients. Strong negative correlation means they carry related but complementary information.

**Frailty score and Sodium: -0.51 correlation**
Frail patients tend to have lower sodium. This is clinically known — sodium dysregulation is common in frailty. This is one of the strongest correlations in the matrix and suggests frailty is capturing real physiological deterioration.

**Frailty score and Albumin: -0.36 correlation**
Low albumin is a marker of malnutrition and poor physiological reserves — exactly what frailty measures. Expected correlation.

**What this means for the GBM:**
The strong AST–ALT redundancy (0.90) suggests you should drop one. The negative WBC–Albumin correlation means both carry signal but from different directions — keep both. The frailty–sodium correlation confirms frailty is a real physiological signal worth including.

---

## 10. The t-SNE Embeddings Plot

The `embeddings.png` you received shows the t-SNE visualisation of the test subjects' encoder embeddings after pre-training.

### What you are looking at

Each dot is one patient. Blue = survived (label 0). Red/pink = died (label 1). The position is determined by the similarity of the encoder's representation — patients the encoder thinks are similar are placed close together.

### What the plot shows

**Partial separation:** The red dots are noticeably concentrated in the upper half of the plot (high values on the y-axis, roughly y > 8). The lower half is almost entirely blue. This is not random — there is a real tendency for died and survived patients to end up in different regions.

**Incomplete separation:** There are red dots scattered throughout the blue region, and the boundary between the two groups is not sharp. This means the encoder learned some features correlated with mortality, but not enough for the classifier to make clean predictions.

**What this means for the near-random classifier output:** The encoder has partially useful representations. The problem is that the single linear classifier head on top cannot find the right boundary in this messy embedding space. Options to fix this: a deeper classifier head (e.g., two linear layers with ReLU), fine-tuning the encoder, or different pre-training.

**Conclusion:** Pre-training was not completely wasted. There is signal in the embeddings. But the current architecture does not exploit it effectively.

---

## 11. Frailty and Saranya's Work

### What is frailty?

Frailty is a medical concept describing a patient with reduced physical reserves — typically older, weaker, less able to recover from physiological stress. A frail patient undergoing surgery is at significantly higher risk of complications and death.

Frailty is not a single measurement. It is a composite score computed from multiple factors.

### The Hospital Frailty Risk Score (HFRS)

The HFRS (from Gilbert et al. 2018) assigns points to ICD-10 diagnostic codes. If a patient has any of 109 specific diagnoses in their history, they get points. The points add up to a frailty score.

Examples:
- `A41` (Septicaemia): +1.6 points
- `B96` (Bacterial infection): +2.9 points
- `D64` (Anaemia): +0.4 points

A patient with several of these codes gets a high frailty score. The categories:
- Low frailty: 0–5 points
- Intermediate: 5–15 points
- High: 15+ points

**File:** `frailty_hfrs.py` contains all 109 ICD-10 codes and their weights.

### Saranya's comparison

Her paper compares the GBM with the 18 standard pre-operative features against a version that also includes the frailty score.

**The hypothesis:** Frailty captures information about a patient's baseline health that the standard lab values and demographics miss. Including it should improve predictions.

**The complication — Bayesian model and data sparsity:**
Adding more variables to a Bayesian model increases the number of parameters that need to be estimated. But the number of data points (especially deaths) stays the same. This means each parameter is estimated from fewer examples, increasing uncertainty (higher variance in estimates).

With only 268 deaths in 97,260 patients, you have very little data to estimate the effect of frailty on mortality reliably. This is the specific tension Saranya is navigating.

### Conditional independence of frailty

**For scheduled surgery:** When a surgeon decides to operate, they have already assessed the patient's fitness. Frail patients are either not operated on (selection bias) or the surgical plan accounts for the frailty. In this context, knowing the frailty score adds relatively little information beyond what the other pre-operative features already capture. Frailty is **conditionally independent** of mortality given that the surgery was scheduled — meaning: once you know the surgery was scheduled and you have the other clinical features, frailty does not tell you much extra.

**For emergency surgery:** There is no pre-selection. A frail 85-year-old can arrive in the emergency department needing urgent surgery regardless of their fitness. In this case, frailty is not conditionally independent — it is highly informative about risk *beyond* all the other features.

This is why Saranya's analysis stratifies by emergency vs scheduled surgery.

### The `inspire_analysis_department.py` script

This script loads all INSPIRE subjects, computes HFRS frailty scores, and produces four plots broken down by surgical department (Cardiothoracic Surgery, General Surgery, Neurosurgery, Orthopaedics, etc.):
1. Survival status by department
2. Frailty category by department
3. Frailty score distribution (boxplot) by department
4. Frailty score distribution (violin plot) by department

It also plots ASA physical status and anaesthesia type by department. This is exploratory data analysis — understanding how patient characteristics differ across departments before building models.

---

## 12. The Code — File by File

### `dnn_mortality_pipeline.py` — the main file (your focus)

The complete two-phase training pipeline. Major functions:

| Function | What it does |
|---|---|
| `align_time_series()` | Interpolates all time series to a common time grid, adds mask columns |
| `normalize_data()` | StandardScaler normalisation, fit on train only |
| `create_sequences()` | Breaks aligned series into sliding windows of `seq_length` |
| `positional_encoding()` | Generates sinusoidal position encodings |
| `TimeSeriesDataset` | PyTorch Dataset for autoencoder (windows + position encoding) |
| `SubjectDataset` | PyTorch Dataset for classifier (full subject series + labels + masks) |
| `TimeSeriesTransformer` | The model — handles both autoencoding and classification via `mode` parameter |
| `preprocess_for_autoencode()` | Aligns, normalises, sequences all training subjects for Phase 1 |
| `train_autoencoder()` | Trains the transformer to reconstruct lab values (Phase 1) |
| `train_classifier()` | Freezes encoder, trains classifier head on mortality labels (Phase 2) |
| `create_train_test_split()` | Stratified 67/33 split |
| `evaluate_model()` | AUROC, AUPRC, F1 threshold scan on test set |
| `extract_embeddings()` | Gets embedding vectors from the trained encoder |
| `visualise_embeddings()` | t-SNE plot of embeddings coloured by class label |

**Known bug:** `get_device()` detects Apple Silicon (MPS) but falls back to CPU with a comment explaining MPS produces NaN values during classifier training. This means even on a Mac, training runs on CPU.

**Design concern noted in code comments:** The `TimeSeriesTransformer` class is used for both autoencoding and classification. The code comments explicitly flag this as a confusing design that should be refactored into two separate, cleaner classes.

### `dnn_mortality_data.py` — data generation and real data loading

Two completely separate modes:

**Synthetic (development):**
`generate_dataset(num_subjects, feature_columns, proportion_died)` creates fake patients with random lab values. For patients who died, glucose and sodium get a slight upward slope. This is purely for testing the pipeline without the real data.

**Real INSPIRE (production):**
`extract_frames(subject, minutes_before_operation)` reads a real patient's ward vitals and labs from the 5 days before their last operation. Returns a dictionary of time series dictionaries.

`align_time_series()` in this file is more sophisticated than the one in the pipeline — it uses a `smooth_fade_to_mean_interpolator` that gracefully handles extrapolation by fading toward the mean rather than extrapolating wildly.

### `gbm_mortality.py` — Saranya's baseline model

Loads `operations.csv` and `labs.csv`. Defines the 30-day mortality outcome as:

```python
inhosp_death_time < orout_time + 30 * 24 * 60  # died within 30 days of leaving operating room
```

Merges the most recent lab value before each surgery using `merge_asof` (an ordered merge that looks backwards in time). Trains two models:
- Logistic Regression (baseline)
- XGBoost GBM (commented out — needs XGBoost installed)

Produces an AUROC plot comparing ASA class alone vs Logistic Regression.

### `nela.py` — clinical gold standard

A single function `compute_nela_score()` implementing the published NELA risk equation. Takes 25 clinical parameters, returns a mortality probability. No data needed, no training.

### `inspire_analysis_department.py` — exploratory analysis by department

Loads all real INSPIRE subjects, computes frailty scores, and produces comparative plots by surgical department. Used to understand how patient populations differ across departments before modelling.

### `frailty_hfrs.py` — HFRS calculation

Contains all 109 ICD-10 code weights from Gilbert et al. 2018. The `compute_hfrs(subject)` function looks up the patient's diagnosis codes and sums up the relevant weights to produce their frailty score and category.

### `inspire_dataset.py` — loading the raw INSPIRE data

Reads the raw CSV files from the `inspire_subjects` folder structure, constructs `Subject` objects that provide methods like `get_labs()`, `get_ward_vitals()`, `inhosp_death_30day()`, `get_operations()`.

---

## 13. Key Problems and What Needs Fixing

### Problem 1 — Classifier outputs ~0.47 for both classes

The debug output in `evaluate_model()` shows:
```
prob=0.47 label=0.0  (survived)
prob=0.47 label=1.0  (died)
```
The model cannot distinguish between the two groups. Most likely cause: the autoencoder pre-training objective (reconstruct labs) does not directly help learn mortality-discriminative features. Freezing those weights locks in that limitation.

**Fix options:**
- Fine-tune the full model (unfreeze encoder) on mortality labels
- Use contrastive pre-training instead of reconstruction
- Use encoder embeddings as features for GBM instead of a linear classifier

### Problem 2 — pos_weight mismatch between synthetic and real data

The synthetic data uses 20% mortality to train. The real data has 0.28% mortality. The `pos_weight` needed for real data (≈362) is far higher than for synthetic data (≈4). The model trained on synthetic data would need extensive retuning for the real dataset.

### Problem 3 — Only 4 features used from 100+ available

The pipeline uses glucose, potassium, sodium, creatinine. The real INSPIRE dataset has 38 lab features, 74 intraoperative vital signs, 16 ward vital signs, and 1,143 medication types. Most of this information is unused.

**Possible direction:** Group all features by physiological system (cardiovascular, respiratory, renal, metabolic, hepatic) and build one embedding per system. This would both increase predictive power and improve interpretability — you could say "the renal system embedding drove this prediction."

### Problem 4 — MPS (Apple Silicon GPU) bug

The pipeline automatically falls back to CPU, which is slower. The NaN bug during classification training on MPS is unresolved.

---

## 14. What to Explore Next

### Immediate (to unblock everything)

1. **Get `embeddings.png` from James** — you already have it from your WhatsApp images. The t-SNE plot confirms partial separation. Use this to discuss with James.
2. **Run the pipeline end to end** — once you understand `dnn_mortality_data.py`, run `dnn_mortality_pipeline.py` and replicate the results you see in the embeddings plot.
3. **Understand the 0.47 probability problem** — read the `evaluate_model()` output carefully and trace it back to why the classifier is outputting near-constant probabilities.

### This week

4. **Try unfreezing the encoder** — change `param.requires_grad = False` to `True` in `train_classifier()` and train the full network end-to-end on the classification task.
5. **Add more features** — add haemoglobin (`preop_hb`) and albumin (`preop_albumin`) to the 4 time series. Both are clinically important and were in the correlation matrix.
6. **Understand Saranya's frailty comparison** — read `frailty_hfrs.py` and `inspire_analysis_department.py` to understand the HFRS computation before meeting with her.

### Longer term

7. **System-level embeddings** — group all 100+ features by physiological system and build one embedding per system. More information, more interpretable.
8. **FFT experiment** — apply Fast Fourier Transform to the lab time series before the transformer, converting them to frequency domain. This may capture periodic patterns more efficiently.
9. **Real data validation** — once the pipeline works well on synthetic data, run it on the real INSPIRE dataset and compare AUROC against the GBM and NELA baselines.

---

*Document compiled from: `dnn_mortality_pipeline.py`, `dnn_mortality_data.py`, `gbm_mortality.py`, `nela.py`, `frailty_hfrs.py`, `inspire_analysis_department.py`, `dnn_mortality.csv`, `FeatureCorrelationMatrix.png`, `embeddings.png`, and all prior conversation analysis.*


### dnn_mortality_pipeline.py 

imports 
pandas - handles data tables - dataframes 
numpy - fast maths on arrays 
scripy - interpolation filling gaps in time series 
sklearn - scaling, train/test split, AUROC, F1 
torch - the deep learning engine (Pytorch)
torch.nn - neural network building blocks 
matplotli - plotting embeddings and curves 
dnn_motatlity_data - generates or loads patient data 
charts - draws AUROC and AUPRC plots

import pandas as pd 
import numpy as np 
from sklearn.processing import StandardScaler
from scipy.interpolate import interplt
from sklearn.manifold import TSNE


import torch 
import torch.nn 
from torch.util.data import dataloader, dataset
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import random 
import dnn_mortality_data
from sklearn.metrics import f1_score


-- functions
align_time_series(data)
