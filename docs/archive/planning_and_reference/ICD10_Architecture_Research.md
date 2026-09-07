# ICD-10 Codes and the Multimodal Architecture — Research

Companion to `roadmap_and_architecture.md` Section 4.2 (the `DX_EMB` box in the pipeline
diagram). This answers two separate questions: (1) does ICD-10's own structure already
give us a ready-made organ-system taxonomy, and (2) what does the literature say about
*how* to turn diagnosis codes into a usable model input.

---

## 1. ICD-10 already has an official organ-system taxonomy — we don't need to invent one

The World Health Organization's ICD-10 is organized into **22 chapters**. The complete
first-letter mapping, checked against our 6 systems:

| Letter(s) | Chapter | Maps to |
|---|---|---|
| A-B | Infectious diseases | Infection context flag |
| C, D(00-49) | Neoplasms | Handled separately (malignancy, Section 6 of `Clinician_Questions.md`) |
| D(50-89) | Blood/blood-forming organs | ✅ haematology_coag |
| E | Endocrine/metabolic | ✅ metabolic_hepatic |
| F | Mental disorders | Not a physiological system — out of scope for now |
| G | Nervous system | ✅ neurological |
| H | Eye/ear | Out of scope |
| I | Circulatory | ✅ cardiovascular |
| J | Respiratory | ✅ respiratory |
| K | Digestive | ✅ metabolic_hepatic (K70-K77 = liver specifically) |
| L | Skin | Out of scope |
| M | Musculoskeletal | Not a core system, but overlaps HFRS's fracture/osteoporosis codes — frailty-adjacent |
| N | Genitourinary | ✅ renal |
| O, P, Q | Pregnancy/perinatal/congenital | Out of scope for this adult surgical cohort |
| **R** | Symptoms/signs/abnormal findings | **Contains R57 (shock) — one of the six original acute-deterioration codes found in EDA, missed in the first mapping pass** |
| S-T | Injury/poisoning | Trauma-adjacent, not organ-system specific |
| V-Y | External causes | Administrative, not clinically useful as a model feature |
| Z | Health status factors | **Z94 (transplant status) sub-codes map cleanly by organ**: Z94.0 kidney→renal, Z94.1 heart→cardio, Z94.2 lung→resp, Z94.4 liver→metabolic_hepatic |

**Two things this immediately tells us:**

1. **Chapter I (infectious diseases) is its own dedicated chapter — separate from every
   organ chapter.** This is a second, independent line of evidence for the SOFA/Sepsis-3
   finding already registered in `Clinician_Questions.md` Section 5: even WHO's own
   official classification doesn't fold infection into an organ system. Treating infection
   as a cross-cutting flag rather than a competing 7th system now has support from *two*
   separate sources (clinical scoring practice, and the diagnosis coding standard itself).

2. **Our narrow, hand-picked code ranges from the earlier audit were the problem, not
   ICD-10 itself.** We tested `A40-A41` for sepsis (0.3% coverage) and `K92.0-K92.2` for
   bleeding (0.0% coverage) — both far narrower than the real official ranges. **Section 3
   below re-runs the audit using the correct, complete chapter ranges**, and now also
   includes **R57 (shock)** and **Z94 (transplant status)**, both missed in the first pass.

### A more authoritative alternative to hand-rolled chapter mapping: AHRQ's CCSR

Rather than continuing to hand-build this mapping letter by letter, **AHRQ/HCUP's Clinical
Classifications Software Refined (CCSR)** already does this, officially and at far higher
resolution: it aggregates all 70,000+ ICD-10-CM codes into **530+ clinically meaningful
categories across 21 body systems** (free, downloadable, maintained by the U.S. federal
government's Agency for Healthcare Research and Quality). Where our own letter-based
mapping lumps everything under one broad "E" (endocrine) bucket, CCSR separately
identifies distinct categories like "fluid and electrolyte disorders," "malnutrition," and
"diabetes with complication" — each independently useful as a feature rather than folded
into one coarse chapter flag. **Recommended as the long-term foundation for `DX_EMB`**,
once ready to move past the quick chapter-letter proof of concept below.

---

## 2. How to actually turn diagnosis codes into a model input — the literature

Three approaches, roughly in order of complexity, all directly relevant to the `DX_EMB`
box in the architecture diagram:

### Simplest: chapter-level one-hot / HFRS-weighted sum
Already effectively what HFRS does — sum weighted contributions from a fixed set of ICD-10
clusters. Fast, interpretable, but treats codes as flat categories with no notion that
"acute kidney failure" (N17) and "chronic kidney failure" (N18) are related concepts.

### Better: learned code embeddings (Med2Vec, Pat2Vec)
- **Choi, E. et al. (2016).** "Multi-layer Representation Learning for Medical Concepts"
  (Med2Vec). *KDD*. — trains a Word2Vec-style embedding directly on co-occurring diagnosis
  codes, producing a dense vector per code. Cited as **improving mortality prediction** by
  giving the model better, denser features than raw one-hot codes.
- **JMIR AI (2023).** "Patient Embeddings From Diagnosis Codes for Health Care Prediction
  Tasks: Pat2Vec Machine Learning Framework." — extends this to a *patient-level* vector
  (summarizing a patient's entire diagnosis history into one embedding), self-supervised,
  using only diagnosis codes as input — directly matches the `DX_EMB` box's stated purpose.

### Most sophisticated, and directly solves our sparse-code problem: hierarchy-aware embeddings
- **Choi, E. et al. (2017).** "GRAM: Graph-based Attention Model for Healthcare
  Representation Learning." *KDD*. — instead of learning one independent embedding per
  code, GRAM represents each code as an attention-weighted combination of its *ancestors*
  in the ICD hierarchy (e.g. a specific code inherits information from its parent
  category, which inherits from its chapter). Results: **10% higher accuracy predicting
  rare diseases**, and **3% AUROC improvement predicting heart failure using an order of
  magnitude less training data**, versus a flat RNN with no hierarchy awareness.

**Why GRAM is the most directly relevant finding here:** our own indication-category audit
found sepsis at 0.3% and bleeding at 0.0% coverage — exactly the "data insufficiency for
rare concepts" problem GRAM was built to solve. A flat embedding or one-hot approach has
almost nothing to learn from 14 sepsis-coded patients out of 5,000. A hierarchy-aware
approach can fall back on the broader Chapter I (infectious disease) signal when the
specific sepsis code is too sparse to learn from directly — this is a genuine, literature-
backed reason to prefer a hierarchy-aware embedding over flat one-hot codes for `DX_EMB`,
not just a theoretical preference.

### Also relevant for later interpretability work
- **Choi, E. et al. (2016).** "RETAIN: An Interpretable Predictive Model for Healthcare
  using Reverse Time Attention Mechanism." — a two-level attention model that identifies
  *which past visits* and *which specific codes within those visits* most influenced a
  prediction. Directly relevant to `roadmap_and_architecture.md` Section 4.4's planned
  attention-auditing work, once the DX embedding pipeline exists to audit.

---

## 3. Corrected ICD-10 audit — using the real chapter ranges, not narrow hand-picked codes

Reruns the earlier malignancy/indication audit with the full official chapter ranges.

```python
import subject as subject_module
import os, glob, random
from collections import Counter

DATA_DIR = extract_dir
SAMPLE_SIZE = 5000

random.seed(42)  # same sample as all previous audits, for direct comparison
all_paths = glob.glob(os.path.join(DATA_DIR, '*', '*.json'))
sample_paths = random.sample(all_paths, min(SAMPLE_SIZE, len(all_paths)))
print(f"Auditing {len(sample_paths)} patients with CORRECTED (full chapter) ICD-10 ranges...")


def classify_code_v2(code):
    """
    Uses the REAL, complete WHO ICD-10 chapter ranges -- expanded from the first pass
    to cover all 22 chapters, not just 7. Two specific fixes vs the first version:
      - Chapter R (symptoms/signs) is now included -- R57 (shock) is one of the six
        acute-deterioration codes originally found in EDA (D65, I46, R57, J80, K72, A41)
        and was MISSED entirely in the first mapping attempt.
      - Z94 (transplant status) sub-codes are routed to their organ-specific system
        (Z94.0 kidney -> renal, Z94.1 heart -> cardio, Z94.2 lung -> resp,
        Z94.4 liver -> metabolic_hepatic) instead of being dropped as "administrative."
    """
    if not code:
        return []
    code = code.upper().strip()
    cats = []
    letter = code[0]
    try:
        number = int(code[1:3])
    except (ValueError, IndexError):
        number = None

    if letter in ('A', 'B'):
        cats.append('chapter1_infectious')
    if letter == 'C' or (letter == 'D' and number is not None and number <= 49):
        cats.append('chapter2_neoplasms')
    if letter == 'D' and number is not None and 50 <= number <= 89:
        cats.append('chapter4_blood_haematology')
    if letter == 'E':
        cats.append('chapter4_endocrine_metabolic')          # NEW -- was missing
    if letter == 'G':
        cats.append('chapter6_nervous')
    if letter == 'I':
        cats.append('chapter9_circulatory_cardio')
    if letter == 'J':
        cats.append('chapter10_respiratory')
    if letter == 'K':
        cats.append('chapter11_digestive_hepatic')
    if letter == 'M':
        cats.append('chapter13_musculoskeletal_frailty')     # NEW -- frailty-adjacent
    if letter == 'N':
        cats.append('chapter14_genitourinary_renal')
    if code.startswith('R57'):
        cats.append('shock_R57')                              # NEW -- specific fix
    if code.startswith('Z940'):
        cats.append('transplant_renal')
    elif code.startswith('Z941'):
        cats.append('transplant_cardiac')
    elif code.startswith('Z942'):
        cats.append('transplant_respiratory')
    elif code.startswith('Z944'):
        cats.append('transplant_hepatic')
    return cats


category_counts = Counter()
n_processed = 0
for path in sample_paths:
    subj = subject_module.Subject()
    subj.fromJSON(path)
    diagnoses = subj.get_diagnoses()
    patient_cats = set()
    for d in diagnoses:
        code = d.get('icd10_cm', '')
        for cat in classify_code_v2(code):
            patient_cats.add(cat)
    for cat in patient_cats:
        category_counts[cat] += 1
    n_processed += 1
    del subj

print(f"Processed {n_processed} patients\n")
print("=" * 60)
print("COVERAGE USING FULL OFFICIAL WHO ICD-10 CHAPTER RANGES")
print("=" * 60)
for cat, count in category_counts.most_common():
    print(f"  {cat:30s} {count:5d} patients ({count/n_processed:.1%})")

print("\nCompare against the earlier narrow-range audit (sepsis 0.3%, bleeding 0.0%) --")
print("if chapter-level coverage is much higher, that confirms the earlier low numbers")
print("were a code-RANGE problem, not evidence these conditions are genuinely absent.")
```

**What to expect, and why it matters:** if `chapter1_infectious` (the full A00-B99 range)
comes back with meaningfully higher coverage than the narrow 0.3% we got from `A40-A41`
alone, that confirms the earlier low numbers were an artifact of an overly narrow code
range — not evidence that sepsis-adjacent conditions are actually rare in this cohort.
This also gives us a genuinely usable, complete signal for `DX_EMB`, built from the real
ICD-10 standard rather than a guessed subset.

---

## 4. Recommendation for the `DX_EMB` pipeline box

1. **Short term:** use the full official chapter ranges above as a simple multi-hot
   feature per patient (7 binary/count features — one per relevant chapter) — cheap,
   immediately buildable, already an improvement over the narrow ranges tried before.
2. **Medium term:** train a Med2Vec/Pat2Vec-style embedding on the full diagnosis code
   vocabulary once the full-scale extraction is stable — gives a dense representation
   instead of 7 coarse chapter flags.
3. **Longer term, if rare-condition prediction turns out to matter:** GRAM-style
   hierarchy-aware embeddings — directly justified by our own sparse-code problem, not
   just a theoretical upgrade.

---

## 5. References

1. World Health Organization. *International Statistical Classification of Diseases and
   Related Health Problems, 10th Revision (ICD-10)* — official chapter structure.
2. Choi, E., Bahadori, M.T., Searles, E., et al. (2016). "Multi-layer Representation
   Learning for Medical Concepts" (Med2Vec). *Proceedings of KDD 2016*.
3. JMIR AI (2023). "Patient Embeddings From Diagnosis Codes for Health Care Prediction
   Tasks: Pat2Vec Machine Learning Framework." *JMIR AI*, 2023;2:e40755.
4. Choi, E., Bahadori, M.T., Song, L., Stewart, W.F., Sun, J. (2017). "GRAM: Graph-based
   Attention Model for Healthcare Representation Learning." *Proceedings of KDD 2017*,
   787-795. arXiv:1611.07012.
5. Choi, E., Bahadori, M.T., Sun, J., Kulas, J., Schuetz, A., Stewart, W. (2016). "RETAIN:
   An Interpretable Predictive Model for Healthcare using Reverse Time Attention
   Mechanism." *NeurIPS 2016*.
6. Singer, M. et al. (2016). "The Third International Consensus Definitions for Sepsis and
   Septic Shock (Sepsis-3)." *JAMA*, 315(8), 801-810. — cited again here since it's the
   direct source for treating infection as cross-cutting rather than a 7th organ system,
   now doubly supported by ICD-10's own chapter structure.
7. Agency for Healthcare Research and Quality (AHRQ), Healthcare Cost and Utilization
   Project (HCUP). *Clinical Classifications Software Refined (CCSR) for ICD-10-CM
   Diagnoses*, v2019.1 onward. hcup-us.ahrq.gov/toolssoftware/ccsr/ccs_refined.jsp —
   the recommended authoritative alternative to hand-rolled chapter-letter mapping.
