

from stopwatch import Stopwatch
from subject import Subject
import subject as subject_module

import matplotlib.pyplot as plt


def plot_age_frailty_analysis(subjects: dict) -> None:
    """
    Plots:
    1. Histogram of ages
    2. Histogram of frailty scores
    3. Scatter plot showing correlation between age and frailty score

    Parameters:
        subjects: dict where key is subject_id (any hashable type) and value is a Subject object
                  Each Subject must have .age (numeric) and .frailty_score (numeric) attributes
    """
    # Extract ages and frailty scores
    ages = []
    frailty_scores = []

    for subject_id, subject in subjects.items():
        # NB: We get the age at their last operation
        ages.append(subject.get_age_last_operation() )
        frailty_score, frailty_category = subject_module.compute_hfrs(subject)
        frailty_scores.append(frailty_score)

    # Convert to float just in case they are int
    ages = [float(a) for a in ages]
    frailty_scores = [float(f) for f in frailty_scores]

    # Create figure with three subplots
    fig = plt.figure(figsize=(15, 5))

    # 1. Histogram of age
    plt.subplot(1, 3, 1)
    plt.hist(ages, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Age Distribution')
    plt.xlabel('Age (years)')
    plt.ylabel('Number of subjects')
    plt.grid(True, alpha=0.3)

    # 2. Histogram of frailty score
    plt.subplot(1, 3, 2)
    plt.hist(frailty_scores, bins=200, color='salmon', edgecolor='black', alpha=0.7)
    plt.title('Frailty Score Distribution')
    plt.xlabel('Frailty Score')
    plt.ylabel('Number of subjects')
    plt.grid(True, alpha=0.3)

    # 3. Scatter plot: age vs frailty score
    plt.subplot(1, 3, 3)
    plt.scatter(ages, frailty_scores, color='purple', alpha=0.6)
    plt.title('Age vs Frailty Score')
    plt.xlabel('Age (years)')
    plt.ylabel('Frailty Score')
    plt.grid(True, alpha=0.3)

    # Optional: add correlation coefficient
    if len(ages) > 1:
        import numpy as np
        corr = np.corrcoef(ages, frailty_scores)[0, 1]
        plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt


# def plot_frailty_categories(subjects: dict) -> None:
#     """
#     Plots a bar chart showing the count of subjects in each of 4 frailty categories:
#     'low', 'intermediate', 'high', 'unknown (no diagnoses)'
#
#     Parameters:
#         subjects: dict where key is subject_id and value is a Subject object
#                   Each Subject must have .frailty_category attribute
#     """
#     # Define expected categories (normalized to lowercase)
#     expected = {
#         "low",
#         "intermediate",
#         "high",
#         "unknown (no diagnoses)"
#     }
#
#     # Initialize counts
#     counts = {cat: 0 for cat in expected}
#     unexpected = 0
#
#     # Count categories (case-insensitive)
#     for subject in subjects.values():
#         frailty_score, frailty_category = subject_module.compute_hfrs(subject)
#         cat = frailty_category.strip().lower()
#
#         # Normalize known variations
#         if cat == "unknown(no diagnoses)" or cat == "unknown" or cat == "no diagnoses":
#             cat = "unknown (no diagnoses)"
#         elif cat == "intermediary":
#             cat = "intermediate"
#
#         if cat in counts:
#             counts[cat] += 1
#         else:
#             unexpected += 1
#             print(f"Warning: Unrecognized frailty_category '{frailty_category}' ignored.")
#
#     # Prepare data for plotting in a logical order
#     ordered_categories = [
#         "low",
#         "intermediate",
#         "high",
#         "unknown (no diagnoses)"
#     ]
#     labels = ["Low", "Intermediate", "High", "Unknown\n(No Diagnoses)"]
#     values = [counts[cat] for cat in ordered_categories]
#
#     # Nice color palette (colorblind-friendly)
#     colors = ['#66c2a5', '#fc8d62', '#e78ac3', '#8da0cb']
#
#     # Create bar chart
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.2, alpha=0.85)
#
#     # Add count labels on top of bars
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.02,
#                  f'{int(height)}',
#                  ha='center', va='bottom', fontsize=12, fontweight='bold')
#
#     # Titles and labels
#     plt.title('Distribution of Frailty Categories (4-Level)', fontsize=16, pad=20)
#     plt.xlabel('Frailty Category', fontsize=12)
#     plt.ylabel('Number of Subjects', fontsize=12)
#
#     # Improve y-axis limit
#     max_count = max(values, default=0)
#     plt.ylim(0, max_count * 1.2)
#
#     # Light grid
#     plt.grid(axis='y', alpha=0.3, linestyle='--')
#
#     # Warn if there were unexpected values
#     if unexpected > 0:
#         plt.text(0.5, 0.95, f'{unexpected} unexpected categories ignored',
#                  transform=plt.gca().transAxes, ha='center', color='red', fontsize=10,
#                  bbox=dict(boxstyle="round,pad=0.3", facecolor="pink", alpha=0.5))
#
#     plt.tight_layout()
#     plt.show()

import matplotlib.pyplot as plt


def plot_frailty_categories(subjects: dict) -> None:
    """
    Stacked bar chart of frailty categories, colored by mortality outcome.
    Shows percentage of 'died' within each frailty category.

    Expected frailty categories:
        'low', 'intermediate', 'high', 'unknown (no diagnoses)'
    Expected mortality values:
        'survived', 'died'
    """
    # Define expected values (normalized)
    frailty_levels = [
        "low",
        "intermediate",
        "high",
        "unknown (no diagnoses)"
    ]
    mortality_states = ["survived", "died"]

    # Initialize counting structure: frailty -> mortality -> count
    counts = {level: {"survived": 0, "died": 0} for level in frailty_levels}
    unexpected_frailty = 0
    unexpected_mortality = 0

    # Count with normalization
    for subject in subjects.values():
        frailty_score, frailty_category = subject_module.compute_hfrs(subject)
        # Normalize frailty category
        f_raw = frailty_category.strip().lower()
        if f_raw in ["unknown(no diagnoses)", "unknown", "no diagnoses"]:
            f_norm = "unknown (no diagnoses)"
        elif f_raw == "intermediary":
            f_norm = "intermediate"
        else:
            f_norm = f_raw

        # Normalize mortality
        mortality = None
        if subject.inhosp_death_30day():
            mortality = 'died'
        else:
            mortality = 'survived'
        m_raw = mortality.strip().lower()
        m_norm = "died" if "die" in m_raw else "survived" if "surv" in m_raw else None

        if f_norm not in counts:
            unexpected_frailty += 1
            continue
        if m_norm not in mortality_states:
            unexpected_mortality += 1
            continue

        counts[f_norm][m_norm] += 1

    # Prepare data for stacking
    survived_counts = [counts[level]["survived"] for level in frailty_levels]
    died_counts = [counts[level]["died"] for level in frailty_levels]
    total_counts = [survived_counts[i] + died_counts[i] for i in range(len(frailty_levels))]

    # Labels with nice formatting
    labels = ["Low", "Intermediate", "High", "Unknown\n(No Diagnoses)"]

    # Colors: green for survived, red for died (colorblind-friendly)
    colors = ['#66c2a5', '#fc8d62']

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(11, 7))

    bars_survived = ax.bar(labels, survived_counts, label='Survived', color=colors[0], edgecolor='black', linewidth=1.2)
    bars_died = ax.bar(labels, died_counts, bottom=survived_counts, label='Died', color=colors[1], edgecolor='black',
                       linewidth=1.2)

    # Add percentage labels on each segment
    for i, (s, d, total) in enumerate(zip(survived_counts, died_counts, total_counts)):
        if total > 0:
            # Survived percentage
            if s > 0:
                ax.text(i, s / 2, f'{s}\n({s / total * 100:.1f}%)',
                        ha='center', va='center', fontweight='bold', color='white', fontsize=10)
            # Died percentage
            if d > 0:
                ax.text(i, s + d / 2, f'{d}\n({d / total * 100:.1f}%)',
                        ha='center', va='center', fontweight='bold', color='white', fontsize=10)
            # Total at top
            ax.text(i, s + d + max(total * 0.02, 1), f'N={total}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Titles and labels
    ax.set_title('Frailty Categories by Mortality Outcome\n(with % died within each category)',
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_ylabel('Number of Subjects', fontsize=12)
    ax.set_xlabel('Frailty Category', fontsize=12)

    # Legend
    ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=True, shadow=True)

    # Grid and layout
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(total_counts) * 1.25)

    # Warning text for data issues
    warning = []
    if unexpected_frailty:
        warning.append(f"{unexpected_frailty} unknown frailty categories")
    if unexpected_mortality:
        warning.append(f"{unexpected_mortality} invalid mortality values")
    if warning:
        ax.text(0.01, 0.01, "Warnings: " + "; ".join(warning),
                transform=ax.transAxes, color='red', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="pink", alpha=0.7))

    plt.tight_layout()
    plt.show()

def main():
    # filepath = '../inspire_subjects/survived/100004062.json'
    #
    # print(f"Reading subject from file {filepath}")
    # subject = Subject(None)
    # subject.fromJSON(filepath)
    # subject_id = subject.get_subject_id()
    #
    # frailty_score, frailty_category = subject_module.compute_hfrs(subject)
    # print(f"HFRS: frailty_score {frailty_score}  frailty_category {frailty_category}")

    parent_dir = '../inspire_subjects'
    #parent_dir = '../inspire_subjects_small'
    # subjects = read_all_subjects(parent_dir)

    stopwatch = Stopwatch()
    subjects = subject_module.read_subjects(parent_dir)
    print(f"Reading took {stopwatch.elapsedTime() / 60:.2f} minutes")

    plot_age_frailty_analysis( subjects )
    plot_frailty_categories( subjects )

if __name__ == "__main__":
    main()