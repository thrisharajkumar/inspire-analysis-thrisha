# Predicting Surgical Risk by Body System — A Plain-Language Summary

**Prepared for clinical review. No coding or machine-learning background assumed.**

---

## 1. What this project actually does, in one paragraph

We built a computer model that predicts a patient's risk of dying within 30 days of
surgery — but instead of producing one unexplained number, it reads the patient's data
**one body system at a time** (kidneys, heart, lungs, and so on), and shows you how much
each system contributed to that patient's risk. The goal is a prediction a clinician can
actually interrogate — "why does it think this patient is high-risk?" — rather than a
black box.

---

## 2. How the current model is built — the flowchart

```mermaid
flowchart TB
    subgraph Systems["Six body systems, each read independently"]
        R["Kidneys"]
        C["Heart & circulation"]
        P["Lungs & breathing"]
        M["Metabolism & liver"]
        H["Blood"]
        N["Brain & nervous system"]
    end

    subgraph Proxy["Two more systems, built differently\n(no lab test exists for these\nin this dataset)"]
        GI["Digestive system\n(from diagnosis codes)"]
        MSK["Bones & joints\n(from diagnosis codes)"]
    end

    subgraph Other["Other information used"]
        AGE["Age, sex, ASA score,\nsurgery type, department"]
        NEWS2B["NEWS2 early-warning score\n(computed from vitals)"]
        HFRS["Frailty score\n(elderly patients only)"]
        INF["Infection/fever signal"]
        HIST["Prior operations,\nprocedure type"]
    end

    C -.->|"heart data also feeds\ninto the kidney reading\n(known clinical link)"| R

    R --> COMBINE
    C --> COMBINE
    P --> COMBINE
    M --> COMBINE
    H --> COMBINE
    N --> COMBINE
    GI --> COMBINE
    MSK --> COMBINE
    AGE --> COMBINE
    NEWS2B --> COMBINE
    HFRS --> COMBINE
    INF --> COMBINE
    HIST --> COMBINE

    COMBINE["Combine all systems --\neach one's contribution\nstays individually visible"]
    COMBINE --> OUT["Risk prediction, with a\nbreakdown: 'kidneys: concerning,\nheart: normal, lungs: normal...'"]

    style R fill:#0a7d6e,color:#fff
    style C fill:#0a7d6e,color:#fff
    style P fill:#0a7d6e,color:#fff
    style M fill:#0a7d6e,color:#fff
    style H fill:#0a7d6e,color:#fff
    style N fill:#0a7d6e,color:#fff
    style GI fill:#dd8452,color:#fff
    style MSK fill:#dd8452,color:#fff
    style COMBINE fill:#2980b9,color:#fff
    style OUT fill:#2c3e50,color:#fff
```

**The key design choice, in plain terms:** each body system gets its own small reader,
and the final answer is built by *adding up* each system's opinion — not blending them
into an unreadable mush. That's what makes "why is this patient high risk" answerable.

---

## 3. Exactly what data goes into each body system

This is the literal, verified mapping from the working code — not an approximation.

| Body system | Lab tests used | Vital signs used |
|---|---|---|
| **Kidneys (renal)** | BUN, calcium, chloride, creatinine, ionised calcium, phosphorus, potassium, sodium | Urine output, dialysis (CRRT) use |
| **Heart & circulation (cardiovascular)** | CK, CK-MB, troponin I | Heart rate, blood pressure (systolic/diastolic/mean), balloon pump (IABP) use |
| **Lungs & breathing (respiratory)** | Blood gas: base excess, bicarbonate, PaCO2, PaO2, pH, SaO2 | Inspired oxygen %, respiratory rate, SpO2, ventilator use, ECMO use |
| **Metabolism & liver (metabolic/hepatic)** | Albumin, ALP, ALT, AST, glucose, HbA1c, lactate, bilirubin, total protein | Body temperature |
| **Blood (haematology)** | aPTT, CRP, fibrinogen, haemoglobin, haematocrit, lymphocytes, platelets, INR, band cells, white cell count | *(none — no dedicated blood vitals in this dataset)* |
| **Brain & nervous system (neurological)** | *(no dedicated neuro lab exists in this dataset)* | Glasgow Coma Scale (eye, motor, verbal components) |
| **Digestive (GI)** | Not lab-based — built from diagnosis codes (ICD-10 chapter XI) and whether General Surgery treated the patient | — |
| **Bones & joints (MSK)** | Not lab-based — built from diagnosis codes (ICD-10 chapter XIII) and whether Orthopaedics treated the patient | — |

**Also feeding into the model, alongside the above:**
- **NEWS2** — the standard UK early-warning score, computed directly from the vitals above
- **Frailty score (HFRS)** — a published scoring system, applied to patients aged 75+
- **Infection signal** — fever, abnormal white cell count, elevated CRP, and specific high-risk diagnosis codes, combined into one flag
- **Operation history** — how many prior operations in the same department, and — specifically for cardiac surgery — a hard rule requiring 6 months' recovery before another operation counts as a new episode
- **Procedure type (ICD-10-PCS)** — *what* operation was actually done, not just which department performed it

---

## 4. Interpretability — what's actually built, and what's next

### What's built now: the "adding up" explanation

Every prediction can be broken down into how much each body system pushed the risk up or
down. This is not a guess bolted on afterward — it's a literal read-out of the model's
own internal calculation. If the model says a patient is high-risk, you can see whether
that's because of their kidneys, their infection markers, or several systems at once.

### What's proposed next: risk that changes with time, and systems that "talk"

The next version of this model — internally called **PACO-Net** — adds two things this
version doesn't have:

```mermaid
flowchart LR
    subgraph Now["Current version"]
        A1["One risk number\n(fixed, pre-op only)"]
        A2["One hand-picked link\n(kidney <- heart)"]
    end
    subgraph Next["Proposed (PACO-Net)"]
        B1["A risk CURVE over time\n(before, during, and\nafter surgery)"]
        B2["The model DISCOVERS which\nsystems are linked, rather\nthan being told just one"]
    end
    A1 -.->|"upgrade"| B1
    A2 -.->|"upgrade"| B2
    style B1 fill:#0a7d6e,color:#fff
    style B2 fill:#8e44ad,color:#fff
```

1. **A risk curve, not one number** — instead of a single pre-op risk score, the model
   would show how risk evolves before, during, and after surgery, so a clinician could see
   *when* things started changing, not just that they did.
2. **Learned connections between body systems** — right now we manually told the model
   "kidneys and heart are linked." The proposed version would let the model discover
   these connections itself from the data — potentially finding real clinical patterns we
   didn't think to specify, like abnormal blood results preceding a ventilation need.

**This is designed and grounded in real research — not built yet.** Worth being precise
about that distinction when discussing this with anyone.

---

## 5. How this compares to existing approaches — is it actually new?

| What we're doing | Closest published comparison | The real difference |
|---|---|---|
| Combining body-system predictions by literally adding them up, so each system's contribution is exact and readable | A 2021 study of ~60,000 surgical patients used a similar "adding up" approach, but per individual lab value, not per body system | We decompose by *whole body system* (each one a small model reading many readings over time), not by single numbers |
| Two body systems (digestive, bones/joints) built from diagnosis codes instead of lab data, in the same framework as the six lab-based systems | A 2025 ICU study linked six body systems together, but all six had real continuous monitoring data | No comparable study mixes "systems with real readings" and "systems built from diagnosis codes" in one unified model |
| Keeping some medical knowledge as a fixed rule (the 6-month cardiac recovery rule) rather than making the model learn everything | Not discussed as a deliberate approach in the studies we reviewed — most either hard-code everything or learn everything | A stated, reasoned choice: some things (like a known clinical protocol) don't need to be "discovered," and shouldn't be left for the model to guess at with limited data |

**An honest caveat, stated plainly:** this comparison is based on a targeted search of
relevant published work, not an exhaustive systematic review. Treat it as good evidence
this is a real, distinctive approach — not as a guaranteed, independently verified claim
of novelty.

---

## 6. What's working — real results, real numbers

- Tested on **10,942 real patients** (a large working subset; not yet the full ~99,886-patient
  cohort)
- **AUPRC 0.658** — the metric that matters when an outcome is rare (deaths are ~4% of
  patients here). In plain terms: when the model flags someone as high-risk, it's right
  roughly 6-7 times out of 10
- **Catches 76% of real deaths** in the group of patients it was tested on
- **Every clinical sanity-check came back correct**: higher surgical-risk score (ASA)
  reliably meant higher death rate; cardiac surgery patients showed meaningfully higher
  risk than general surgery — the underlying data behaves the way real clinical experience
  says it should

---

## 7. Limitations — stated plainly, not buried

- **Not yet run on the full patient cohort** — current results are from a large working
  subset (10,942 of ~99,886 patients), not the complete dataset
- **Single hospital system, no external validation** — this hasn't yet been tested on data
  from a different hospital or population, which any real clinical use would require
- **Not yet compared against a simple model** — a straightforward statistical model on the
  same data hasn't yet been benchmarked against this one, which is a necessary check before
  claiming the added complexity is worth it
- **The consciousness component of NEWS2 is approximated**, not a true bedside assessment —
  it's estimated from Glasgow Coma Scale readings already in the data, not the real ACVPU
  check a nurse would perform
- **PACO-Net (§4) is a designed proposal, not a working system** — the risk-curve and
  learned-connections ideas are researched and grounded in real papers, but not yet built
  or tested
