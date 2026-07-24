# INSPIRE Research Roadmap

# Main Aim

Develop a **Clinically Interpretable Deep Learning Model for 30-day Peri-operative Mortality Prediction**.

Unlike existing work that explains predictions using post-hoc explainability methods (e.g. SHAP or LIME), this project aims to design a model whose architecture itself mirrors clinical reasoning.

The model should learn clinically meaningful latent representations from:

- Medications
- Laboratory tests
- Vital signs
- Diagnoses
- Procedures
- Clinical risk scores

before integrating them into a final mortality prediction.

---

# Repository Structure

```
INSPIRE
│
├── Research Aim
├── Clinical Motivation
├── Dataset
├── Exploratory Analysis
├── Current Results
├── Clinical Risk Scores
├── Proposed Deep Learning Architecture
├── Experiments
├── Research Questions
├── Future Work
└── References
```

---

# Current Explorations

## Exploratory Data Analysis (EDA)

- Mortality distribution
- Age distribution
- ASA distribution
- ICD-10 diagnosis frequencies
- Department-level mortality
- Missingness analysis
- Frailty analysis
- Hospital Frailty Risk Score (HFRS)
- Multi-operation analysis

## Feature Engineering

Current engineered features include:

- Diagnosis
- Medications
- Laboratory tests
- Vital signs
- Frailty
- Demographics

## Existing Models

- Gradient Boosting Machine (GBM)
- Deep Neural Networks (DNN)

Evaluation metrics:

- AUROC
- Precision
- Recall
- Calibration

---

# Research Questions

## Research Question 1: Handling Multiple Surgeries

**How should patients with multiple surgeries be handled?**

### Experiment A
Exclude patients with more than one surgery.

### Experiment B
Use the first surgery.

### Experiment C
Use the last surgery.

### Experiment D
Treat surgeries as sequential events using temporal models (Transformer/LSTM/GRU).

---

## Research Question 2: Which Clinical Phase is Most Predictive?

Compare predictive performance using:

- Pre-operative data only
- Pre-operative + Intra-operative
- Pre-operative + Intra-operative + Post-operative

Determine how each phase contributes to mortality prediction.

---

## Research Question 3: When Does Clinical Deterioration Begin?

Move beyond binary mortality prediction to estimate risk trajectories over time.

Potential approaches:

- Survival Analysis
- DeepHit
- Dynamic Deep Survival Models
- Transformer-based survival models

---

## Research Question 4: Can Embeddings be Learned for Each Clinical Modality?

Instead of concatenating raw features, learn embeddings separately for:

- Medications
- Laboratory Tests
- Vital Signs
- Diagnoses
- Procedures

Fuse these learned embeddings into a unified patient representation.

---

## Research Question 5: Can the Architecture Itself be Clinically Interpretable?

Rather than relying on SHAP or LIME after training, design an architecture where intermediate layers represent clinically meaningful concepts such as:

- Respiratory Risk
- Cardiovascular Risk
- Neurological Risk
- Renal Risk
- Frailty Risk

The final mortality prediction is derived from these clinically interpretable intermediate representations.

---

## Research Question 6: Integration of Clinical Risk Scores

Compare and evaluate:

- ASA
- POSSUM
- P-POSSUM
- NELA
- Charlson Comorbidity Index
- Hospital Frailty Risk Score (HFRS)

Potential experiments:

- Use as model inputs
- Predict them as auxiliary tasks
- Compare performance against deep learning models

---

## Research Question 7: Hierarchical Representation of ICD-10 Codes

Instead of treating ICD-10 codes independently:

```
ICD-10 Code
      ↓
Disease Category
      ↓
Organ System
      ↓
Clinical Embedding
```

Investigate ontology-aware or graph-based representations.

---

# Proposed Deep Learning Architecture

```
Raw Data
──────────────

Medications
Laboratory Tests
Vital Signs
Diagnoses
Procedures
Demographics
Clinical Risk Scores

        ↓

Feature Encoders
────────────────

Medication Encoder
Laboratory Encoder
Vital Signs Encoder
Diagnosis Encoder
Procedure Encoder

        ↓

Clinical State Layer
────────────────────

Respiratory
Cardiovascular
Neurological
Renal
Haematology
Metabolic
Frailty
Inflammation

        ↓

Interaction Layer

        ↓

Temporal Progression Layer

        ↓

30-Day Mortality Risk

        ↓

Clinical Explanation Layer
```

---

# Additional Research Ideas

- Multi-task learning (mortality, ICU admission, complications, length of stay, readmission)
- Temporal modelling using Transformers or recurrent models
- Cross-modal attention between medications, labs, and vitals
- Counterfactual reasoning for modifiable risk factors
- Uncertainty estimation and calibration
- External validation across hospitals and specialties
- Patient phenotype discovery using learned embeddings
- Knowledge-guided modelling with medical ontologies or graph neural networks
- Clinician-facing explanations through intermediate organ-system risk representations

---

# Overall Vision

Transform the project from a conventional mortality prediction model into a **Clinically Interpretable Multimodal Deep Learning Framework**.

The key novelty is to design a model whose architecture itself reflects clinical reasoning, making predictions inherently explainable rather than relying on post-hoc explanation methods such as SHAP or LIME.
