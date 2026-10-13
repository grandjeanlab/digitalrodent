# Digital Rodent Partial Model Centiles - Alzheimer


``` python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pcntoolkit import NormativeModel, NormData


# ============================================================
# PATHS
# ============================================================

root = Path(
    "/home/traaffneu/dansch/Documents/digital_rodent/digitalrodent"
)

input_file = (
    root / "results" / "combined_dataset"
    / "Partial_Model" / "Partial_Model_dataframe.csv"
)

model_dir = (
    root / "results" / "normative_model"
    / "Model_Testing" / "Partial_Model"
)

plot_dir = model_dir / "centiles_svg_Alzheimer"
plot_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIGURE STYLE
# ============================================================

MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 6,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "svg.fonttype": "none"
})


# ============================================================
# COLOURS AND COMPONENTS
# ============================================================

BLUE = "#4682B4"
ORANGE = "#F4A460"
GREEN = "#6D9F8C"
DARK = "#39434B"

component_colors = {
    "comp2": BLUE,
    "comp10": ORANGE,
    "comp11": GREEN
}

component_labels = {
    "comp2": "Somatosensory cortex",
    "comp10": "DMN-associated cortex",
    "comp11": "Ventral hippocampus"
}

response_vars = ["comp2", "comp10", "comp11"]


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = NormativeModel.load(str(model_dir))

model_df = pd.read_csv(input_file)

Alzheimer = pd.read_csv(
    root / "data" / "Alzheimer" / "Alzheimer.csv"
)

Alzheimer = Alzheimer[
    ["Age(week)"] + response_vars
].copy()

for col in ["Age(week)"] + response_vars:
    Alzheimer[col] = pd.to_numeric(
        Alzheimer[col],
        errors="coerce"
    )

covariates = [str(x) for x in model.covariates]
batch_effects = []

print("Model loaded")
print("Responses:", response_vars)
print("Covariates:", covariates)
print("Alzheimer scans:", len(Alzheimer))
print("Saving to:", plot_dir)


# ============================================================
# REFERENCE PROFILE
# ============================================================

reference_pool = model_df.dropna(
    subset=covariates
).copy()

median_age = reference_pool["Age(week)"].median()

reference_index = (
    (reference_pool["Age(week)"] - median_age)
    .abs()
    .idxmin()
)

reference_row = reference_pool.loc[reference_index]

print("\nReference profile:")
print(reference_row[covariates])


def make_reference_dataframe(n):

    df = pd.DataFrame({
        col: [reference_row[col]] * n
        for col in covariates
    })

    for response in response_vars:
        df[response] = 0.0

    return df


# ============================================================
# NORMDATA AND CENTILES
# ============================================================

def to_normdata(name, dataframe):

    return NormData.from_dataframe(
        name=name,
        dataframe=dataframe,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
        remove_Nan=False
    )


def get_centiles(data):

    model.compute_centiles(
        data,
        centiles=[0.05, 0.25, 0.50, 0.75, 0.95],
        recompute=True
    )

    return data


def centile_values(data, component):

    return tuple(
        np.atleast_1d(
            data.centiles.sel(
                response_vars=component,
                centile=p
            ).values.squeeze()
        )
        for p in [0.05, 0.25, 0.50, 0.75, 0.95]
    )


# ============================================================
# PLOT STYLE
# ============================================================

def style_axis(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color(DARK)
    ax.spines["bottom"].set_color(DARK)

    ax.tick_params(colors=DARK, width=0.6)
    ax.xaxis.label.set_color(DARK)
    ax.yaxis.label.set_color(DARK)
    ax.title.set_color(DARK)


def add_legend(ax):

    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=6,
        fontsize=6,
        handlelength=1.5,
        columnspacing=1.0,
        borderaxespad=0
    )


def add_centile_shading(
    ax, x, c05, c25, c50, c75, c95, color
):

    ax.fill_between(
        x, c05, c25,
        color=color, alpha=0.08, zorder=2
    )

    ax.fill_between(
        x, c25, c50,
        color=color, alpha=0.20, zorder=2
    )

    ax.fill_between(
        x, c50, c75,
        color=color, alpha=0.20, zorder=2
    )

    ax.fill_between(
        x, c75, c95,
        color=color, alpha=0.08, zorder=2
    )

    ax.plot(
        x, c05,
        color=color, linestyle="--",
        linewidth=0.8, alpha=0.75,
        label="5th", zorder=3
    )

    ax.plot(
        x, c25,
        color=color, linestyle=":",
        linewidth=0.9, alpha=0.85,
        label="25th", zorder=3
    )

    ax.plot(
        x, c50,
        color=color, linewidth=1.5,
        label="50th", zorder=4
    )

    ax.plot(
        x, c75,
        color=color, linestyle=":",
        linewidth=0.9, alpha=0.85,
        label="75th", zorder=3
    )

    ax.plot(
        x, c95,
        color=color, linestyle="--",
        linewidth=0.8, alpha=0.75,
        label="95th", zorder=3
    )


def safe_name(text):

    return (
        text.replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
    )


# ============================================================
# CONTINUOUS CENTILE PLOTS
# ============================================================

continuous_variables = {
    "Age(week)": "Age (weeks)",
    "MRI": "MRI field strength (T)",
    "MRI T.R.": "TR"
}

for variable, xlabel in continuous_variables.items():

    if variable not in covariates:
        continue

    x_min = model_df[variable].min()
    x_max = model_df[variable].max()

    x_values = np.linspace(x_min, x_max, 150)

    synthetic_df = make_reference_dataframe(
        len(x_values)
    )

    synthetic_df[variable] = x_values

    centile_data = to_normdata(
        f"{variable}_centiles",
        synthetic_df
    )

    centile_data = get_centiles(centile_data)

    for component in response_vars:

        color = component_colors[component]

        c05, c25, c50, c75, c95 = centile_values(
            centile_data,
            component
        )

        fig, ax = plt.subplots(
            figsize=(85 * MM, 65 * MM)
        )

        # Original observations
        ax.scatter(
            model_df[variable],
            model_df[component],
            s=7,
            color=color,
            alpha=0.35,
            linewidth=0,
            zorder=1
        )

        # Original centiles
        add_centile_shading(
            ax,
            x_values,
            c05, c25, c50, c75, c95,
            color
        )

        # ====================================================
        # ALZHEIMER OBSERVATIONS - AGE ONLY
        # ====================================================

        if variable == "Age(week)":

            alz_points = Alzheimer.dropna(
                subset=["Age(week)", component]
            )

            ax.scatter(
                alz_points["Age(week)"],
                alz_points[component],
                s=22,
                color="crimson",
                alpha=0.9,
                edgecolors="white",
                linewidth=0.3,
                label=f"Alzheimer (n={len(alz_points)})",
                zorder=10
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("DR value")

        ax.set_title(
            component_labels[component],
            pad=18
        )

        add_legend(ax)
        style_axis(ax)

        fig.tight_layout()

        filename = (
            f"{safe_name(component_labels[component])}_"
            f"{safe_name(variable)}_centiles.svg"
        )

        fig.savefig(
            plot_dir / filename,
            format="svg",
            bbox_inches="tight"
        )

        print("Saved:", filename)

        plt.close(fig)


# ============================================================
# CATEGORICAL HELPERS
# ============================================================

def observed_binary_labels(column, mapping):

    return model_df[column].map(mapping)


def observed_from_dummies(
    dummy_cols,
    reference_label,
    prefix_to_remove
):

    labels = pd.Series(
        [reference_label] * len(model_df),
        index=model_df.index,
        dtype="object"
    )

    for dummy in dummy_cols:

        label = dummy.replace(
            prefix_to_remove, ""
        )

        labels.loc[
            model_df[dummy] == 1
        ] = label

    return labels


# ============================================================
# CATEGORICAL CENTILE PLOTS
# ============================================================

def categorical_centile_plot(
    name,
    labels,
    profiles,
    xlabel,
    observed_labels
):

    synthetic_df = make_reference_dataframe(
        len(labels)
    )

    for i, profile in enumerate(profiles):

        for col, value in profile.items():

            synthetic_df.loc[i, col] = value

    centile_data = to_normdata(
        f"{name}_centiles",
        synthetic_df
    )

    centile_data = get_centiles(centile_data)

    x = np.arange(len(labels))

    for component in response_vars:

        color = component_colors[component]

        c05, c25, c50, c75, c95 = centile_values(
            centile_data,
            component
        )

        width_mm = max(85, len(labels) * 18)

        fig, ax = plt.subplots(
            figsize=(width_mm * MM, 65 * MM)
        )

        rng = np.random.default_rng(42)

        # Observed scans
        for i, label in enumerate(labels):

            yvals = pd.to_numeric(
                model_df.loc[
                    observed_labels == label,
                    component
                ],
                errors="coerce"
            ).dropna()

            if len(yvals) > 0:

                jitter = rng.normal(
                    0, 0.06, len(yvals)
                )

                ax.scatter(
                    np.full(len(yvals), x[i]) + jitter,
                    yvals,
                    s=8,
                    color=color,
                    alpha=0.40,
                    linewidth=0,
                    zorder=1
                )

        # Centile shading and lines
        add_centile_shading(
            ax,
            x,
            c05, c25, c50, c75, c95,
            color
        )

        # Centile markers
        for i in range(len(labels)):

            ax.vlines(
                x[i],
                c05[i],
                c95[i],
                color=color,
                linewidth=0.8,
                alpha=0.8,
                zorder=4
            )

            ax.vlines(
                x[i],
                c25[i],
                c75[i],
                color=color,
                linewidth=3,
                alpha=0.60,
                zorder=5
            )

            ax.scatter(
                x[i],
                c50[i],
                s=20,
                color=color,
                edgecolor=DARK,
                linewidth=0.4,
                zorder=6
            )

        ax.set_xticks(x)

        ax.set_xticklabels(
            labels,
            rotation=35 if len(labels) > 4 else 0,
            ha="right" if len(labels) > 4 else "center"
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("DR value")

        ax.set_title(
            component_labels[component],
            pad=18
        )

        add_legend(ax)
        style_axis(ax)

        fig.tight_layout()

        filename = (
            f"{safe_name(component_labels[component])}_"
            f"{safe_name(name)}_centiles.svg"
        )

        fig.savefig(
            plot_dir / filename,
            format="svg",
            bbox_inches="tight"
        )

        print("Saved:", filename)

        plt.close(fig)


# ============================================================
# SEX
# ============================================================

if "Sex_encoded" in covariates:

    obs_labels = observed_binary_labels(
        "Sex_encoded",
        {0.0: "Female", 1.0: "Male"}
    )

    categorical_centile_plot(
        name="Sex",
        labels=["Female", "Male"],
        profiles=[
            {"Sex_encoded": 0.0},
            {"Sex_encoded": 1.0}
        ],
        xlabel="Sex",
        observed_labels=obs_labels
    )


# ============================================================
# COIL
# ============================================================

if "Coil_encoded" in covariates:

    obs_labels = observed_binary_labels(
        "Coil_encoded",
        {0.0: "RT", 1.0: "Cryo"}
    )

    categorical_centile_plot(
        name="Coil",
        labels=["RT", "Cryo"],
        profiles=[
            {"Coil_encoded": 0.0},
            {"Coil_encoded": 1.0}
        ],
        xlabel="Coil",
        observed_labels=obs_labels
    )


# ============================================================
# VENTILATION
# ============================================================

if "Ventilation_encoded" in covariates:

    obs_labels = observed_binary_labels(
        "Ventilation_encoded",
        {0.0: "No", 1.0: "Yes"}
    )

    categorical_centile_plot(
        name="Ventilation",
        labels=["No", "Yes"],
        profiles=[
            {"Ventilation_encoded": 0.0},
            {"Ventilation_encoded": 1.0}
        ],
        xlabel="Ventilation",
        observed_labels=obs_labels
    )


# ============================================================
# STRAIN
# ============================================================

strain_dummy_cols = [
    col for col in covariates
    if col.startswith("Rodent.strain_")
]

if strain_dummy_cols:

    strain_labels = ["C57BL/6"]

    strain_profiles = [{
        col: 0.0 for col in strain_dummy_cols
    }]

    for dummy in strain_dummy_cols:

        label = dummy.replace("Rodent.strain_", "")

        profile = {
            col: 0.0 for col in strain_dummy_cols
        }

        profile[dummy] = 1.0

        strain_labels.append(label)
        strain_profiles.append(profile)

    obs_labels = observed_from_dummies(
        strain_dummy_cols,
        "C57BL/6",
        "Rodent.strain_"
    )

    categorical_centile_plot(
        name="Rodent_strain",
        labels=strain_labels,
        profiles=strain_profiles,
        xlabel="Rodent strain",
        observed_labels=obs_labels
    )


# ============================================================
# fMRI SEQUENCE
# ============================================================

sequence_dummy_cols = [
    col for col in covariates
    if col.startswith("fMRI.sequence_")
]

if sequence_dummy_cols:

    sequence_labels = ["GE-EPI"]

    sequence_profiles = [{
        col: 0.0 for col in sequence_dummy_cols
    }]

    for dummy in sequence_dummy_cols:

        label = dummy.replace("fMRI.sequence_", "")

        profile = {
            col: 0.0 for col in sequence_dummy_cols
        }

        profile[dummy] = 1.0

        sequence_labels.append(label)
        sequence_profiles.append(profile)

    obs_labels = observed_from_dummies(
        sequence_dummy_cols,
        "GE-EPI",
        "fMRI.sequence_"
    )

    categorical_centile_plot(
        name="fMRI_sequence",
        labels=sequence_labels,
        profiles=sequence_profiles,
        xlabel="fMRI sequence",
        observed_labels=obs_labels
    )


# ============================================================
# ANESTHESIA
# ============================================================

anesthesia_dummy_cols = [
    col for col in covariates
    if col.startswith("Anesthesia_")
]

if anesthesia_dummy_cols:

    anesthesia_labels = ["Awake"]

    anesthesia_profiles = [{
        col: 0.0 for col in anesthesia_dummy_cols
    }]

    for dummy in anesthesia_dummy_cols:

        label = dummy.replace("Anesthesia_", "")

        profile = {
            col: 0.0 for col in anesthesia_dummy_cols
        }

        profile[dummy] = 1.0

        anesthesia_labels.append(label)
        anesthesia_profiles.append(profile)

    obs_labels = observed_from_dummies(
        anesthesia_dummy_cols,
        "Awake",
        "Anesthesia_"
    )

    categorical_centile_plot(
        name="Anesthesia",
        labels=anesthesia_labels,
        profiles=anesthesia_profiles,
        xlabel="Anesthesia",
        observed_labels=obs_labels
    )

# ============================================================
# ALZHEIMER Z-SCORES
# ============================================================

alz_df = make_reference_dataframe(len(Alzheimer))

alz_df["Age(week)"] = Alzheimer["Age(week)"].to_numpy()

for component in response_vars:
    alz_df[component] = Alzheimer[component].to_numpy()

alz_data = to_normdata(
    "Alzheimer_Zscores",
    alz_df
)

model.compute_zscores(alz_data)

z_results = Alzheimer.reset_index(drop=True).copy()

for component in response_vars:
    z_results[f"{component}_Z"] = (
        alz_data.Z
        .sel(response_vars=component)
        .values
        .flatten()
    )

print(z_results.to_string(index=False))

z_results.to_csv(
    plot_dir / "Alzheimer_Zscores.csv",
    index=False
)

# ============================================================
# DONE
# ============================================================

print("\nDONE")
print("SVG files saved to:", plot_dir)

print("\nFiles created:")
for file in sorted(plot_dir.glob("*.svg")):
    print(file.name)
```

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Model loaded
    Responses: ['comp2', 'comp10', 'comp11']
    Covariates: ['Age(week)', 'MRI', 'MRI T.R.', 'Sex_encoded', 'Coil_encoded', 'Ventilation_encoded', 'Rodent.strain_F1 C6/129P', 'Rodent.strain_ICR', 'fMRI.sequence_ME-EPI', 'fMRI.sequence_SE-EPI', 'Anesthesia_Dex', 'Anesthesia_Dexiso', 'Anesthesia_Hal', 'Anesthesia_Iso', 'Anesthesia_Med', 'Anesthesia_Mediso', 'Anesthesia_Propofol']
    Alzheimer scans: 22
    Saving to: /home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/results/normative_model/Model_Testing/Partial_Model/centiles_svg_Alzheimer

    Reference profile:
    Age(week)                     12.0
    MRI                            9.4
    MRI T.R.                    1500.0
    Sex_encoded                    1.0
    Coil_encoded                   0.0
    Ventilation_encoded            1.0
    Rodent.strain_F1 C6/129P       0.0
    Rodent.strain_ICR              0.0
    fMRI.sequence_ME-EPI           0.0
    fMRI.sequence_SE-EPI           0.0
    Anesthesia_Dex                 0.0
    Anesthesia_Dexiso              0.0
    Anesthesia_Hal                 0.0
    Anesthesia_Iso                 0.0
    Anesthesia_Med                 0.0
    Anesthesia_Mediso              0.0
    Anesthesia_Propofol            0.0
    Name: 154, dtype: object
    Process: 4027568 - 2026-10-13 03:08:23 - Dataset "Age(week)_centiles" created.
        - 150 observations
        - 150 unique subjects
        - 17 covariates
        - 3 response variables
        - 1 batch effects:
            dummy_batch_effect (1)
        
    Process: 4027568 - 2026-10-13 03:08:23 - Computing centiles for 3 response variables.
    Process: 4027568 - 2026-10-13 03:08:23 - Computing centiles for comp2.
    Process: 4027568 - 2026-10-13 03:08:23 - Computing centiles for comp11.
    Process: 4027568 - 2026-10-13 03:08:23 - Computing centiles for comp10.

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Somatosensory_cortex_Ageweek_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: DMN_associated_cortex_Ageweek_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Ventral_hippocampus_Ageweek_centiles.svg
    Process: 4027568 - 2026-10-13 03:08:24 - Dataset "MRI_centiles" created.
        - 150 observations
        - 150 unique subjects
        - 17 covariates
        - 3 response variables
        - 1 batch effects:
            dummy_batch_effect (1)
        
    Process: 4027568 - 2026-10-13 03:08:24 - Computing centiles for 3 response variables.
    Process: 4027568 - 2026-10-13 03:08:24 - Computing centiles for comp2.
    Process: 4027568 - 2026-10-13 03:08:24 - Computing centiles for comp11.
    Process: 4027568 - 2026-10-13 03:08:24 - Computing centiles for comp10.

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Somatosensory_cortex_MRI_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: DMN_associated_cortex_MRI_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Ventral_hippocampus_MRI_centiles.svg
    Process: 4027568 - 2026-10-13 03:08:25 - Dataset "MRI T.R._centiles" created.
        - 150 observations
        - 150 unique subjects
        - 17 covariates
        - 3 response variables
        - 1 batch effects:
            dummy_batch_effect (1)
        
    Process: 4027568 - 2026-10-13 03:08:25 - Computing centiles for 3 response variables.
    Process: 4027568 - 2026-10-13 03:08:25 - Computing centiles for comp2.
    Process: 4027568 - 2026-10-13 03:08:25 - Computing centiles for comp11.
    Process: 4027568 - 2026-10-13 03:08:25 - Computing centiles for comp10.

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Somatosensory_cortex_MRI_T_R__centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: DMN_associated_cortex_MRI_T_R__centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Ventral_hippocampus_MRI_T_R__centiles.svg
    Process: 4027568 - 2026-10-13 03:08:26 - Dataset "Sex_centiles" created.
        - 2 observations
        - 2 unique subjects
        - 17 covariates
        - 3 response variables
        - 1 batch effects:
            dummy_batch_effect (1)
        
    Process: 4027568 - 2026-10-13 03:08:26 - Computing centiles for 3 response variables.
    Process: 4027568 - 2026-10-13 03:08:26 - Computing centiles for comp2.
    Process: 4027568 - 2026-10-13 03:08:26 - Computing centiles for comp11.
    Process: 4027568 - 2026-10-13 03:08:26 - Computing centiles for comp10.

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Somatosensory_cortex_Sex_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: DMN_associated_cortex_Sex_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Ventral_hippocampus_Sex_centiles.svg
    Process: 4027568 - 2026-10-13 03:08:27 - Dataset "Coil_centiles" created.
        - 2 observations
        - 2 unique subjects
        - 17 covariates
        - 3 response variables
        - 1 batch effects:
            dummy_batch_effect (1)
        
    Process: 4027568 - 2026-10-13 03:08:27 - Computing centiles for 3 response variables.
    Process: 4027568 - 2026-10-13 03:08:27 - Computing centiles for comp2.
    Process: 4027568 - 2026-10-13 03:08:27 - Computing centiles for comp11.
    Process: 4027568 - 2026-10-13 03:08:27 - Computing centiles for comp10.

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Somatosensory_cortex_Coil_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: DMN_associated_cortex_Coil_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Ventral_hippocampus_Coil_centiles.svg
    Process: 4027568 - 2026-10-13 03:08:28 - Dataset "Ventilation_centiles" created.
        - 2 observations
        - 2 unique subjects
        - 17 covariates
        - 3 response variables
        - 1 batch effects:
            dummy_batch_effect (1)
        
    Process: 4027568 - 2026-10-13 03:08:28 - Computing centiles for 3 response variables.
    Process: 4027568 - 2026-10-13 03:08:28 - Computing centiles for comp2.
    Process: 4027568 - 2026-10-13 03:08:28 - Computing centiles for comp11.
    Process: 4027568 - 2026-10-13 03:08:28 - Computing centiles for comp10.

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Somatosensory_cortex_Ventilation_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: DMN_associated_cortex_Ventilation_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Ventral_hippocampus_Ventilation_centiles.svg
    Process: 4027568 - 2026-10-13 03:08:29 - Dataset "Rodent_strain_centiles" created.
        - 3 observations
        - 3 unique subjects
        - 17 covariates
        - 3 response variables
        - 1 batch effects:
            dummy_batch_effect (1)
        
    Process: 4027568 - 2026-10-13 03:08:29 - Computing centiles for 3 response variables.
    Process: 4027568 - 2026-10-13 03:08:29 - Computing centiles for comp2.
    Process: 4027568 - 2026-10-13 03:08:29 - Computing centiles for comp11.
    Process: 4027568 - 2026-10-13 03:08:29 - Computing centiles for comp10.

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Somatosensory_cortex_Rodent_strain_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: DMN_associated_cortex_Rodent_strain_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Ventral_hippocampus_Rodent_strain_centiles.svg
    Process: 4027568 - 2026-10-13 03:08:30 - Dataset "fMRI_sequence_centiles" created.
        - 3 observations
        - 3 unique subjects
        - 17 covariates
        - 3 response variables
        - 1 batch effects:
            dummy_batch_effect (1)
        
    Process: 4027568 - 2026-10-13 03:08:30 - Computing centiles for 3 response variables.
    Process: 4027568 - 2026-10-13 03:08:30 - Computing centiles for comp2.
    Process: 4027568 - 2026-10-13 03:08:30 - Computing centiles for comp11.
    Process: 4027568 - 2026-10-13 03:08:30 - Computing centiles for comp10.

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Somatosensory_cortex_fMRI_sequence_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: DMN_associated_cortex_fMRI_sequence_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Ventral_hippocampus_fMRI_sequence_centiles.svg
    Process: 4027568 - 2026-10-13 03:08:31 - Dataset "Anesthesia_centiles" created.
        - 8 observations
        - 8 unique subjects
        - 17 covariates
        - 3 response variables
        - 1 batch effects:
            dummy_batch_effect (1)
        
    Process: 4027568 - 2026-10-13 03:08:31 - Computing centiles for 3 response variables.
    Process: 4027568 - 2026-10-13 03:08:31 - Computing centiles for comp2.
    Process: 4027568 - 2026-10-13 03:08:31 - Computing centiles for comp11.
    Process: 4027568 - 2026-10-13 03:08:31 - Computing centiles for comp10.

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Somatosensory_cortex_Anesthesia_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: DMN_associated_cortex_Anesthesia_centiles.svg

    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.
    findfont: Font family 'Liberation Sans' not found.

    Saved: Ventral_hippocampus_Anesthesia_centiles.svg
    Process: 4027568 - 2026-10-13 03:08:32 - Dataset "Alzheimer_Zscores" created.
        - 22 observations
        - 22 unique subjects
        - 17 covariates
        - 3 response variables
        - 1 batch effects:
            dummy_batch_effect (1)
        
    Process: 4027568 - 2026-10-13 03:08:32 - Making predictions on 3 response variables.
    Process: 4027568 - 2026-10-13 03:08:32 - Computing z-scores for 3 response variables.
    Process: 4027568 - 2026-10-13 03:08:32 - Computing z-scores for comp2.
    Process: 4027568 - 2026-10-13 03:08:32 - Computing z-scores for comp11.
    Process: 4027568 - 2026-10-13 03:08:32 - Computing z-scores for comp10.
     Age(week)    comp2   comp10   comp11   comp2_Z  comp10_Z  comp11_Z
            13 0.088831 0.054319 0.055880 -2.670316 -2.325473 -2.109594
            26 0.088930 0.070310 0.061353 -1.810725 -1.415520 -1.473505
            13 0.106523 0.051530 0.061231 -2.343267 -2.374792 -2.071099
            26 0.067329 0.054323 0.065543 -2.144538 -1.617821 -1.450244
            13 0.120034 0.073478 0.058793 -2.093506 -1.986671 -2.088638
            13 0.072314 0.056144 0.060129 -2.975645 -2.293200 -2.079027
            26 0.052744 0.042228 0.056046 -2.369928 -1.770872 -1.502967
            13 0.083338 0.059990 0.060970 -2.771858 -2.225188 -2.072977
            13 0.119104 0.047799 0.051384 -2.110698 -2.440770 -2.141938
            13 0.049870 0.033541 0.050688 -3.390539 -2.692904 -2.146945
            26 0.056638 0.029763 0.049751 -2.309752 -1.928605 -1.537914
            13 0.073884 0.039724 0.051585 -2.946623 -2.583566 -2.140492
            26 0.140118 0.041709 0.068943 -1.019688 -1.777440 -1.431369
            13 0.078851 0.053674 0.059313 -2.854804 -2.336878 -2.084897
            26 0.113527 0.069109 0.049501 -1.430614 -1.430718 -1.539302
            13 0.089940 0.046223 0.057119 -2.649816 -2.468639 -2.100681
            26 0.073326 0.053779 0.049900 -2.051863 -1.624705 -1.537087
            13 0.085802 0.039858 0.054649 -2.726310 -2.581196 -2.118450
            26 0.118120 0.053166 0.098003 -1.359635 -1.632462 -1.270040
            13 0.094555 0.052234 0.064256 -2.564504 -2.362343 -2.049338
            13 0.057928 0.036661 0.069767 -3.241581 -2.637731 -2.009692
            26 0.108060 0.040900 0.056493 -1.515098 -1.787677 -1.500486

    DONE
    SVG files saved to: /home/traaffneu/dansch/Documents/digital_rodent/digitalrodent/results/normative_model/Model_Testing/Partial_Model/centiles_svg_Alzheimer

    Files created:
    DMN_associated_cortex_Ageweek_centiles.svg
    DMN_associated_cortex_Anesthesia_centiles.svg
    DMN_associated_cortex_Coil_centiles.svg
    DMN_associated_cortex_MRI_T_R__centiles.svg
    DMN_associated_cortex_MRI_centiles.svg
    DMN_associated_cortex_Rodent_strain_centiles.svg
    DMN_associated_cortex_Sex_centiles.svg
    DMN_associated_cortex_Ventilation_centiles.svg
    DMN_associated_cortex_fMRI_sequence_centiles.svg
    Somatosensory_cortex_Ageweek_centiles.svg
    Somatosensory_cortex_Anesthesia_centiles.svg
    Somatosensory_cortex_Coil_centiles.svg
    Somatosensory_cortex_MRI_T_R__centiles.svg
    Somatosensory_cortex_MRI_centiles.svg
    Somatosensory_cortex_Rodent_strain_centiles.svg
    Somatosensory_cortex_Sex_centiles.svg
    Somatosensory_cortex_Ventilation_centiles.svg
    Somatosensory_cortex_fMRI_sequence_centiles.svg
    Ventral_hippocampus_Ageweek_centiles.svg
    Ventral_hippocampus_Anesthesia_centiles.svg
    Ventral_hippocampus_Coil_centiles.svg
    Ventral_hippocampus_MRI_T_R__centiles.svg
    Ventral_hippocampus_MRI_centiles.svg
    Ventral_hippocampus_Rodent_strain_centiles.svg
    Ventral_hippocampus_Sex_centiles.svg
    Ventral_hippocampus_Ventilation_centiles.svg
    Ventral_hippocampus_fMRI_sequence_centiles.svg
