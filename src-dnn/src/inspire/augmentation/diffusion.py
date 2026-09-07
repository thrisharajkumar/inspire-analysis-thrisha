"""
TabDDPM-based minority (died) patient generation — replaces SMOTENC as the default
sampling strategy (config: sampling.strategy == "diffusion"), per your supervisor's
guidance.

Key design decisions, all justified in docs/current/PACO_Net_Latest_Work_and_Results.md:
  - Conditioned on the same (department, ASA) strata as grouped SMOTENC.
  - Generated PER ORGAN SYSTEM, not as one flat vector — keeps the model's job smaller
    and lets each system's synthetic output be sanity-checked independently.
  - Short diffusion chain (num_timesteps=100, not the default 1000) and dropout, since
    the real minority pool is only 469 patients full-cohort, well below the ~1,500-2,000
    generally recommended for GAN-based alternatives (TimeGAN/medGAN/TabDDPM's GAN
    cousins) — diffusion tolerates small-N better but isn't immune to it.
  - Pretrain pooled across ALL minority patients first, then fine-tune per stratum —
    stratifying an already-small dataset thins some strata to single digits.
  - Skips 'neurological' by default (config: diffusion.skip_systems) — only ~1%
    observed-cell rate at the 10,942-patient EDA scale, not enough real signal to trust
    a synthetic distribution learned from it; falls back to standard imputation instead.

Requires: pip install synthcity  (wraps TabDDPM with mixed continuous/categorical support)
"""

from inspire.features.organ_systems import ENCODER_SYSTEMS, features_for_system, LOW_SIGNAL_SYSTEMS


def _get_plugin(num_timesteps, n_iter, dropout):
    # Imported lazily so the rest of the package doesn't hard-require synthcity.
    from synthcity.plugins import Plugins
    return Plugins().get(
        "ddpm",
        n_iter=n_iter,
        num_timesteps=num_timesteps,
        dropout=dropout,
    )


def train_pooled_model(minority_df, system_name, categorical_cols, config):
    """Stage A: pretrain one diffusion model per organ system, pooled across ALL
    minority patients (before any per-stratum split thins the data further)."""
    from synthcity.plugins.core.dataloader import GenericDataLoader

    feats = [f for f in features_for_system(system_name) if f in minority_df.columns]
    if not feats:
        return None
    loader = GenericDataLoader(minority_df[feats], categorical_columns=categorical_cols)
    model = _get_plugin(config["diffusion"]["num_timesteps"], config["diffusion"]["n_iter"],
                         config["diffusion"]["dropout"])
    model.fit(loader)
    return model


def finetune_per_stratum(pooled_model, stratum_df, system_name, categorical_cols, config):
    """Stage A continued: lightly fine-tune the pooled model on one (department, ASA)
    stratum. Falls back to the pooled model itself if the stratum is too small."""
    feats = [f for f in features_for_system(system_name) if f in stratum_df.columns]
    if len(stratum_df) < 5:
        print(f"Stratum too small ({len(stratum_df)} patients) to fine-tune {system_name} — "
              f"using pooled model as-is for this stratum.")
        return pooled_model
    from synthcity.plugins.core.dataloader import GenericDataLoader
    loader = GenericDataLoader(stratum_df[feats], categorical_columns=categorical_cols)
    pooled_model.fit(loader)  # continues training from pooled weights
    return pooled_model


def generate_synthetic_patients(model, n_samples):
    """Stage B: sample n_samples synthetic patients from a trained (pooled or
    fine-tuned) diffusion model for one organ system."""
    return model.generate(count=n_samples).dataframe()


def generate_all_systems(minority_df, strata, categorical_cols, config, n_needed_per_stratum):
    """
    Top-level orchestration: for each organ system (except LOW_SIGNAL_SYSTEMS), pretrain
    pooled, fine-tune per stratum, and generate. Returns dict[system_name -> synthetic_df].

    IMPORTANT: run eval.validate_synthetic.py on the output before trusting it in
    training — this function does not validate plausibility itself.
    """
    skip = set(config["diffusion"].get("skip_systems", []) or LOW_SIGNAL_SYSTEMS)
    results = {}

    for system_name in ENCODER_SYSTEMS:
        if system_name in skip:
            print(f"Skipping diffusion for '{system_name}' (low real signal) — "
                  f"falls back to standard imputation instead.")
            continue

        pooled_model = train_pooled_model(minority_df, system_name, categorical_cols, config)
        if pooled_model is None:
            continue

        synthetic_frames = []
        for stratum_key, stratum_df in minority_df.groupby(strata, observed=True):
            model = (finetune_per_stratum(pooled_model, stratum_df, system_name, categorical_cols, config)
                     if config["diffusion"]["pretrain_pooled_then_finetune"] else pooled_model)
            n_needed = n_needed_per_stratum.get(stratum_key, 0)
            if n_needed <= 0:
                continue
            synthetic_frames.append(generate_synthetic_patients(model, n_needed))

        if synthetic_frames:
            import pandas as pd
            results[system_name] = pd.concat(synthetic_frames, ignore_index=True)

    return results
