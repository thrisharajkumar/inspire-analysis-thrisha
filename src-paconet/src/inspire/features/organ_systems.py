"""
Organ-system feature mapping — the single source of truth for which raw INSPIRE
parameter belongs to which organ system, which phase(s) it's observed in, and whether
it's a raw physiology signal or an organ-*support* signal (SOFA-style: a vasopressor
dose or ventilator setting is an organ-support signal, not raw physiology).

Every other module (loader, encoders, coupling, diffusion) imports from here rather
than repeating the mapping. If the mapping changes, it changes in exactly one place.

Full reasoning behind each assignment: docs/current/PACO_Net_Latest_Work_and_Results.md §2.
"""

from enum import Enum


class Phase(Enum):
    PRE = "pre"
    PERI = "peri"
    POST = "post"


class SourceTable(Enum):
    LABS = "labs"
    VITALS = "vitals"              # intra-op, 72 params — peri-op phase
    WARD_VITALS = "ward_vitals"    # pre/post-op, 14 params


class Role(Enum):
    PHYSIOLOGY = "physiology"       # raw measurement of organ state
    SUPPORT = "support"             # drug dose / device use — organ *support*, SOFA-style


# feature_name -> (organ_system, source_table, phase(s), role)
FEATURE_MAP = {
    # ---------------- Renal ----------------
    "creatinine":  ("renal", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "bun":         ("renal", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "sodium":      ("renal", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "potassium":   ("renal", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "chloride":    ("renal", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "calcium":     ("renal", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "phosphorus":  ("renal", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "ica":         ("renal", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "crrt":        ("renal", SourceTable.WARD_VITALS, (Phase.PRE, Phase.POST), Role.SUPPORT),
    "uo":          ("renal", SourceTable.WARD_VITALS, (Phase.PRE, Phase.PERI, Phase.POST), Role.PHYSIOLOGY),
    "hes":         ("renal", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),  # dual-tagged, see docs

    # ---------------- Cardiovascular ----------------
    "troponin_i":  ("cardiovascular", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "troponin_t":  ("cardiovascular", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "ck":          ("cardiovascular", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "ckmb":        ("cardiovascular", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "hr":          ("cardiovascular", SourceTable.WARD_VITALS, (Phase.PRE, Phase.PERI, Phase.POST), Role.PHYSIOLOGY),
    "nibp_sbp":    ("cardiovascular", SourceTable.WARD_VITALS, (Phase.PRE, Phase.PERI, Phase.POST), Role.PHYSIOLOGY),
    "nibp_dbp":    ("cardiovascular", SourceTable.WARD_VITALS, (Phase.PRE, Phase.PERI, Phase.POST), Role.PHYSIOLOGY),
    "nibp_mbp":    ("cardiovascular", SourceTable.WARD_VITALS, (Phase.PRE, Phase.PERI, Phase.POST), Role.PHYSIOLOGY),
    "art_sbp":     ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "art_dbp":     ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "art_mbp":     ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "ci":          ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "cvp":         ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "svi":         ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "pap_sbp":     ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "pap_dbp":     ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "pap_mbp":     ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "sti":         ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "stii":        ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "stiii":       ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "stv5":        ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "dobui":       ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "dopai":       ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "mlni":        ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "eph":         ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "epi":         ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "epii":        ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "nepi":        ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "pepi":        ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "phe":         ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "vaso":        ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "ntgi":        ("cardiovascular", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "iabp":        ("cardiovascular", SourceTable.WARD_VITALS, (Phase.PRE, Phase.POST), Role.SUPPORT),

    # ---------------- Respiratory ----------------
    "pao2":        ("respiratory", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "paco2":       ("respiratory", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "ph":          ("respiratory", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "hco3":        ("respiratory", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "be":          ("respiratory", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "sao2":        ("respiratory", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "spo2":        ("respiratory", SourceTable.WARD_VITALS, (Phase.PRE, Phase.PERI, Phase.POST), Role.PHYSIOLOGY),
    "rr":          ("respiratory", SourceTable.WARD_VITALS, (Phase.PRE, Phase.PERI, Phase.POST), Role.PHYSIOLOGY),
    "fio2":        ("respiratory", SourceTable.WARD_VITALS, (Phase.PRE, Phase.PERI, Phase.POST), Role.SUPPORT),
    "air":         ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "o2":          ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "n2o":         ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "etco2":       ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "minvol":      ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "peep":        ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "pip":         ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "pmean":       ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "pplat":       ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "vt":          ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "etdes":       ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "etgas":       ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "etiso":       ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "etsevo":      ("respiratory", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "vent":        ("respiratory", SourceTable.WARD_VITALS, (Phase.PRE, Phase.POST), Role.SUPPORT),
    "ecmo":        ("respiratory", SourceTable.WARD_VITALS, (Phase.PRE, Phase.POST), Role.SUPPORT),

    # ---------------- Metabolic / Hepatic ----------------
    "glucose":         ("metabolic_hepatic", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "hba1c":           ("metabolic_hepatic", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "albumin":         ("metabolic_hepatic", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "alp":             ("metabolic_hepatic", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "alt":             ("metabolic_hepatic", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "ast":             ("metabolic_hepatic", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "total_bilirubin": ("metabolic_hepatic", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "total_protein":   ("metabolic_hepatic", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "lacate":          ("metabolic_hepatic", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),  # 'lacate' spelling matches parameters.csv
    "bt":              ("metabolic_hepatic", SourceTable.WARD_VITALS, (Phase.PRE, Phase.PERI, Phase.POST), Role.PHYSIOLOGY),
    "d5w":             ("metabolic_hepatic", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "d10w":            ("metabolic_hepatic", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "d50w":            ("metabolic_hepatic", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "alb5":            ("metabolic_hepatic", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "alb20":           ("metabolic_hepatic", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),

    # ---------------- Haematology / Coagulation ----------------
    "hb":          ("haematology", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "hct":         ("haematology", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "wbc":         ("haematology", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "platelet":    ("haematology", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "aptt":        ("haematology", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "ptinr":       ("haematology", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "fibrinogen":  ("haematology", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "d_dimer":     ("haematology", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "crp":         ("haematology", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "lymphocyte":  ("haematology", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "seg":         ("haematology", SourceTable.LABS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "ebl":         ("haematology", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "rbc":         ("haematology", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "ffp":         ("haematology", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "pc":          ("haematology", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "cryo":        ("haematology", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "pheresis":    ("haematology", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),

    # ---------------- Neurological ----------------
    "gcs_e":       ("neurological", SourceTable.WARD_VITALS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "gcs_m":       ("neurological", SourceTable.WARD_VITALS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "gcs_v":       ("neurological", SourceTable.WARD_VITALS, (Phase.PRE, Phase.POST), Role.PHYSIOLOGY),
    "bis":         ("neurological", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "cbro2":       ("neurological", SourceTable.VITALS, (Phase.PERI,), Role.PHYSIOLOGY),
    "ppf":         ("neurological", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "ppfi":        ("neurological", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "mdz":         ("neurological", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "ftn":         ("neurological", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "sft":         ("neurological", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "rfti":        ("neurological", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "aft":         ("neurological", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),  # alfentanil, opioid — same bucket as fentanyl/sufentanil

    # ---------------- Fluid / Resuscitation (cross-cutting, NOT an organ — see docs) ----------------
    "ns":          ("fluid_resuscitation", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "hns":         ("fluid_resuscitation", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "hs":          ("fluid_resuscitation", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
    "psa":         ("fluid_resuscitation", SourceTable.VITALS, (Phase.PERI,), Role.SUPPORT),
}

# Diagnosis-derived systems are not raw parameters — they're computed from ICD-10 chapters.
DIAGNOSIS_DERIVED_SYSTEMS = {
    "gi":  {"icd10_chapter": "XI", "phase": (Phase.PRE,), "static": True},
    "msk": {"icd10_chapter": "XIII", "phase": (Phase.PRE,), "static": True},
}

# The organ systems that feed into per-organ transformer encoders (excludes the
# diagnosis-derived and fluid buckets, which are handled separately — see models/fusion.py)
ENCODER_SYSTEMS = [
    "renal", "cardiovascular", "respiratory",
    "metabolic_hepatic", "haematology", "neurological",
]

# Systems fed as static/aggregate features at the fusion level, not through a timeline encoder
STATIC_SYSTEMS = ["gi", "msk", "fluid_resuscitation"]

# Systems too sparse in real data to trust diffusion-generated synthetic values for
# (see config diffusion.skip_systems — kept here too so code and config can cross-check)
LOW_SIGNAL_SYSTEMS = ["neurological"]  # ~1% observed-cell rate at 10,942-patient scale


def features_for_system(system_name):
    """All raw feature names mapped to a given organ system."""
    return [f for f, (sys_, *_rest) in FEATURE_MAP.items() if sys_ == system_name]


def features_for_phase(phase: Phase):
    """All raw feature names observed during a given phase, across all systems."""
    return [f for f, (_sys, _tbl, phases, _role) in FEATURE_MAP.items() if phase in phases]


def support_features_for_system(system_name):
    """Organ-support (drug/device) features for a system — the SOFA-style subset."""
    return [
        f for f, (sys_, _tbl, _phases, role) in FEATURE_MAP.items()
        if sys_ == system_name and role == Role.SUPPORT
    ]


def all_systems():
    return sorted(set(sys_ for sys_, *_ in FEATURE_MAP.values())) + list(DIAGNOSIS_DERIVED_SYSTEMS)


if __name__ == "__main__":
    # Quick sanity check: print feature counts per system, so a future edit to this
    # file (adding/removing a feature) is easy to eyeball.
    for system in ENCODER_SYSTEMS:
        feats = features_for_system(system)
        support = support_features_for_system(system)
        print(f"{system:20s} {len(feats):3d} features ({len(support)} support/drug-device)")
    print(f"{'fluid_resuscitation':20s} {len(features_for_system('fluid_resuscitation')):3d} features (cross-cutting, not an organ)")
    print(f"Total mapped raw parameters: {len(FEATURE_MAP)}")
