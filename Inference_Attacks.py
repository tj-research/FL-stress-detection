import os
import sys
import gc
import json
import warnings
import importlib.util

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import StratifiedKFold


# CONFIGURATION

EXISTING_FL_SCRIPT = "FedAvg_WESAD_Baseline+Defense.py"

RESULTS_DIR = "results"

ATTACK_RESULTS_DIR = os.path.join(
    RESULTS_DIR,
    "Privacy_Attacks_Publication"
)

MIA_RESULTS_DIR = os.path.join(
    ATTACK_RESULTS_DIR,
    "MIA"
)

AIA_RESULTS_DIR = os.path.join(
    ATTACK_RESULTS_DIR,
    "AIA"
)

LIA_RESULTS_DIR = os.path.join(
    ATTACK_RESULTS_DIR,
    "LIA"
)

FIGURES_DIR = os.path.join(
    ATTACK_RESULTS_DIR,
    "Figures"
)

RANDOM_STATE = 42
ATTACK_CV_FOLDS = 5
ATTACK_MAX_ITER = 3000
MAX_AIA_SAMPLES_PER_SUBJECT = 500

# Attack switches

RUN_LEARNED_MIA = True
RUN_AIA = False #false True
RUN_LEARNED_LIA = True
CREATE_FIGURES = True
RUN_SUBJECT_LEVEL_MIA = True

ATTRIBUTE_SOURCE = "groups"
ATTRIBUTE_ATTACK_TYPE = "Subject_Identity_Inference"

# MIA roles

MIA_MEMBER_ROLE = "Federated_Training_Client"
MIA_NONMEMBER_ROLE = "Test"

# Publication figure settings

FIG_DPI = 600

FIG_FORMATS = [
    "png",
    "pdf",
    "svg"
]

FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE = 15
AXIS_LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
LINE_WIDTH = 2.2
MEAN_LINE_WIDTH = 3.2
GRID_ALPHA = 0.25

# PUBLICATION FIGURE COLORS — MIA METHODS

MIA_COLORS = {
    "MIA_Loss": "#1f77b4",              # Blue
    "MIA_Confidence": "#d62728",       # Red
    "MIA_Entropy": "#2ca02c",          # Green
    "MIA_Correctness": "#9467bd",      # Purple
    "MIA_Learned_Logistic": "#ff7f0e"  # Orange
}

# PUBLICATION FIGURE COLORS — LIA METHODS

LIA_COLORS = {
    "LIA_Direct_Model_Output": "#D55E00",      # Orange
    "LIA_Learned_Output_Attack": "#CC79A7"     # Pink
}

# Publication colors

COLORS = {

    "MIA":
        "#0072B2",

    "AIA":
        "#D55E00",

    "LIA":
        "#009E73",

    "Random":
        "#666666",

    "Fold":
        "#56B4E9",

    "Mean":
        "#0072B2",

    "Train":
        "#0072B2",

    "Test":
        "#D55E00"
}


# DIRECTORY SETUP

for directory in [
    ATTACK_RESULTS_DIR,
    MIA_RESULTS_DIR,
    AIA_RESULTS_DIR,
    LIA_RESULTS_DIR,
    FIGURES_DIR
]:

    os.makedirs(
        directory,
        exist_ok=True
    )


# RANDOM GENERATOR

RNG = np.random.default_rng(
    RANDOM_STATE
)


# MATPLOTLIB PUBLICATION STYLE

def configure_publication_style():

    plt.rcParams.update({

        "font.family":
            FONT_FAMILY,

        "font.size":
            TICK_SIZE,

        "axes.titlesize":
            TITLE_SIZE,

        "axes.labelsize":
            AXIS_LABEL_SIZE,

        "axes.linewidth":
            1.1,

        "xtick.labelsize":
            TICK_SIZE,

        "ytick.labelsize":
            TICK_SIZE,

        "legend.fontsize":
            LEGEND_SIZE,

        "legend.frameon":
            False,

        "figure.dpi":
            FIG_DPI,

        "savefig.dpi":
            FIG_DPI,

        "savefig.bbox":
            "tight",

        "savefig.pad_inches":
            0.08,

        "axes.spines.top":
            False,

        "axes.spines.right":
            False
    })


configure_publication_style()


# SAVE FIGURE IN MULTIPLE PUBLICATION FORMATS

def save_publication_figure(
    fig,
    filename
):

    base = os.path.join(
        FIGURES_DIR,
        filename
    )

    for fmt in FIG_FORMATS:

        path = (
            f"{base}.{fmt}"
        )

        fig.savefig(
            path,
            dpi=FIG_DPI,
            bbox_inches="tight",
            facecolor="white"
        )

    print(
        f"Saved publication figure: {base}.[png/pdf/svg]"
    )


# LOAD EXISTING FEDERATED SCRIPT

def load_existing_federated_script():

    script_path = os.path.abspath(
        EXISTING_FL_SCRIPT
    )

    if not os.path.exists(
        script_path
    ):

        raise FileNotFoundError(
            "\nExisting FL script not found:\n"
            f"{script_path}\n"
        )

    module_name = (
        "existing_wesad_federated_module"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            script_path
        )
    )

    if (
        spec is None
        or
        spec.loader is None
    ):

        raise ImportError(
            "Unable to load federated-learning script."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


FL = load_existing_federated_script()


# RESULTS DIRECTORY FROM FL SCRIPT

if hasattr(
    FL,
    "RESULTS_DIR"
):

    RESULTS_DIR = FL.RESULTS_DIR

    ATTACK_RESULTS_DIR = os.path.join(
        RESULTS_DIR,
        "Privacy_Attacks_Publication"
    )

    MIA_RESULTS_DIR = os.path.join(
        ATTACK_RESULTS_DIR,
        "MIA"
    )

    AIA_RESULTS_DIR = os.path.join(
        ATTACK_RESULTS_DIR,
        "AIA"
    )

    LIA_RESULTS_DIR = os.path.join(
        ATTACK_RESULTS_DIR,
        "LIA"
    )

    FIGURES_DIR = os.path.join(
        ATTACK_RESULTS_DIR,
        "Figures"
    )

    for directory in [
        ATTACK_RESULTS_DIR,
        MIA_RESULTS_DIR,
        AIA_RESULTS_DIR,
        LIA_RESULTS_DIR,
        FIGURES_DIR
    ]:

        os.makedirs(
            directory,
            exist_ok=True
        )


# CHECK REQUIRED FUNCTIONS

required_functions = [
    "build_dataset",
    "predict_model",
    "build_model"
]

for function_name in required_functions:

    if not hasattr(
        FL,
        function_name
    ):

        raise AttributeError(
            f"Required function missing: "
            f"{function_name}()"
        )


# DATA LOADING

def load_dataset():

    print("\n" + "-" * 50)
    print(
        "LOADING WESAD DATASET"
    )
    print("=" * 100)

    (
        BVP,
        EDA,
        TEMP,
        ACC,
        y,
        groups
    ) = FL.build_dataset()

    BVP = np.asarray(BVP)

    EDA = np.asarray(EDA)

    TEMP = np.asarray(TEMP)

    ACC = np.asarray(ACC)

    y = np.asarray(
        y
    ).astype(
        np.int32
    )

    groups = np.asarray(
        groups
    )

    lengths = [
        len(BVP),
        len(EDA),
        len(TEMP),
        len(ACC),
        len(y),
        len(groups)
    ]

    if len(
        set(lengths)
    ) != 1:

        raise ValueError(
            "Dataset arrays have inconsistent lengths."
        )

    print(
        f"Windows: {len(y):,}"
    )

    print(
        f"Subjects: "
        f"{len(np.unique(groups))}"
    )

    print(
        f"Class 0: "
        f"{np.sum(y == 0):,}"
    )

    print(
        f"Class 1: "
        f"{np.sum(y == 1):,}"
    )

    return (
        BVP,
        EDA,
        TEMP,
        ACC,
        y,
        groups
    )


# LOAD FEDERATED SUBJECT ASSIGNMENTS

def load_subject_assignments():

    path = os.path.join(
        RESULTS_DIR,
        "Federated_Subject_Assignments.csv"
    )

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"Missing:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "Fold",
        "Role",
        "Subject"
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    df["Subject"] = (
        df["Subject"]
        .astype(str)
    )

    return df


# LOAD FEDERATED MODEL

def load_fold_model(
    fold
):

    model_path = os.path.join(
        RESULTS_DIR,
        f"Federated_Best_Global_Model_Fold_{fold}.keras"
    )

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Missing model:\n{model_path}"
        )

    print(
        f"Loading Fold {fold} model"
    )

    model = FL.tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model


# MODEL PROBABILITY PREDICTION

def predict_probabilities(
    model,
    BVP,
    EDA,
    TEMP,
    ACC
):

    probabilities = FL.predict_model(
        model,
        BVP,
        EDA,
        TEMP,
        ACC
    )

    probabilities = np.asarray(
        probabilities
    ).reshape(-1)

    probabilities = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7
    )

    return probabilities


# MODEL-DERIVED FEATURES

def probability_features(
    y_true,
    probabilities
):

    y_true = np.asarray(
        y_true
    ).astype(
        np.int32
    )

    probabilities = np.asarray(
        probabilities
    ).reshape(-1)

    probabilities = np.clip(
        probabilities,
        1e-7,
        1 - 1e-7
    )

    predictions = (
        probabilities >= 0.5
    ).astype(
        np.int32
    )

    loss = -(
        y_true * np.log(probabilities)
        +
        (1 - y_true)
        *
        np.log(1 - probabilities)
    )

    entropy = -(
        probabilities * np.log(probabilities)
        +
        (1 - probabilities)
        *
        np.log(1 - probabilities)
    )

    confidence = np.maximum(
        probabilities,
        1 - probabilities
    )

    margin = np.abs(
        probabilities - 0.5
    )

    correctness = (
        predictions == y_true
    ).astype(float)

    return pd.DataFrame({

        "Probability":
            probabilities,

        "Confidence":
            confidence,

        "Margin":
            margin,

        "Entropy":
            entropy,

        "Loss":
            loss,

        "Correct":
            correctness
    })



#  SAFE METRICS


def safe_auc(
    y_true,
    scores
):

    if len(
        np.unique(y_true)
    ) < 2:

        return np.nan

    return float(
        roc_auc_score(
            y_true,
            scores
        )
    )


def safe_pr_auc(
    y_true,
    scores
):

    if len(
        np.unique(y_true)
    ) < 2:

        return np.nan

    return float(
        average_precision_score(
            y_true,
            scores
        )
    )


def safe_tpr_at_fpr(
    y_true,
    scores,
    target_fpr
):

    if len(
        np.unique(y_true)
    ) < 2:

        return np.nan

    fpr, tpr, _ = roc_curve(
        y_true,
        scores
    )

    valid = np.where(
        fpr <= target_fpr
    )[0]

    if len(valid) == 0:

        return 0.0

    return float(
        np.max(
            tpr[valid]
        )
    )


# MIA METRICS

def mia_metrics(
    membership,
    scores,
    predictions
):

    cm = confusion_matrix(
        membership,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else np.nan
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else np.nan
    )

    return {

        "Accuracy":
            accuracy_score(
                membership,
                predictions
            ),

        "Balanced_Accuracy":
            balanced_accuracy_score(
                membership,
                predictions
            ),

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "Precision":
            precision_score(
                membership,
                predictions,
                zero_division=0
            ),

        "F1":
            f1_score(
                membership,
                predictions,
                zero_division=0
            ),

        "ROC_AUC":
            safe_auc(
                membership,
                scores
            ),

        "PR_AUC":
            safe_pr_auc(
                membership,
                scores
            ),

        "TPR_at_FPR_1pct":
            safe_tpr_at_fpr(
                membership,
                scores,
                0.01
            ),

        "TPR_at_FPR_5pct":
            safe_tpr_at_fpr(
                membership,
                scores,
                0.05
            ),

        "TPR_at_FPR_10pct":
            safe_tpr_at_fpr(
                membership,
                scores,
                0.10
            )
    }


# MIA ATTACK

def run_mia(
    BVP,
    EDA,
    TEMP,
    ACC,
    y,
    groups,
    assignments
):

    print("\n" + "-" * 50)
    print(
        "MEMBERSHIP INFERENCE ATTACK"
    )
    print("-" * 50)

    all_results = []

    roc_records = []

    pr_records = []

    subject_records = []

    group_strings = groups.astype(str)

    folds = sorted(
        assignments["Fold"].unique()
    )

    for fold in folds:

        print(
            f"\nMIA Fold {fold}"
        )

        fa = assignments[
            assignments["Fold"] == fold
        ]

        member_subjects = (
            fa[
                fa["Role"]
                ==
                MIA_MEMBER_ROLE
            ]["Subject"]
            .astype(str)
            .tolist()
        )

        nonmember_subjects = (
            fa[
                fa["Role"]
                ==
                MIA_NONMEMBER_ROLE
            ]["Subject"]
            .astype(str)
            .tolist()
        )

        if (
            not member_subjects
            or
            not nonmember_subjects
        ):

            continue

        member_idx = np.where(
            np.isin(
                group_strings,
                member_subjects
            )
        )[0]

        nonmember_idx = np.where(
            np.isin(
                group_strings,
                nonmember_subjects
            )
        )[0]

        if (
            len(member_idx) == 0
            or
            len(nonmember_idx) == 0
        ):

            continue

        model = load_fold_model(
            fold
        )

        member_prob = predict_probabilities(
            model,
            BVP[member_idx],
            EDA[member_idx],
            TEMP[member_idx],
            ACC[member_idx]
        )

        nonmember_prob = predict_probabilities(
            model,
            BVP[nonmember_idx],
            EDA[nonmember_idx],
            TEMP[nonmember_idx],
            ACC[nonmember_idx]
        )

        member_features = probability_features(
            y[member_idx],
            member_prob
        )

        nonmember_features = probability_features(
            y[nonmember_idx],
            nonmember_prob
        )

        member_features["Membership"] = 1
        nonmember_features["Membership"] = 0

        member_features["Subject"] = (
            group_strings[member_idx]
        )

        nonmember_features["Subject"] = (
            group_strings[nonmember_idx]
        )

        attack_data = pd.concat(
            [
                member_features,
                nonmember_features
            ],
            ignore_index=True
        )

        attack_data["Fold"] = fold

        attack_data.to_csv(
            os.path.join(
                MIA_RESULTS_DIR,
                f"MIA_Fold_{fold}_Scores.csv"
            ),
            index=False
        )

        membership = (
            attack_data["Membership"]
            .values
            .astype(int)
        )

        # HEURISTIC ATTACKS

        heuristic_attacks = {

            "Loss":
                -attack_data["Loss"].values,

            "Confidence":
                attack_data["Confidence"].values,

            "Entropy":
                -attack_data["Entropy"].values,

            "Correctness":
                attack_data["Correct"].values
        }

        for attack_name, scores in (
            heuristic_attacks.items()
        ):

            if attack_name == "Correctness":

                predictions = (
                    scores >= 0.5
                ).astype(int)

            else:

                threshold = np.median(
                    scores
                )

                predictions = (
                    scores >= threshold
                ).astype(int)

            metrics = mia_metrics(
                membership,
                scores,
                predictions
            )

            metrics["Fold"] = fold

            metrics["Attack"] = (
                f"MIA_{attack_name}"
            )

            all_results.append(
                metrics
            )

            # ROC data

            fpr, tpr, thresholds = roc_curve(
                membership,
                scores
            )

            roc_records.append(
                pd.DataFrame({

                    "Fold":
                        fold,

                    "Attack":
                        f"MIA_{attack_name}",

                    "FPR":
                        fpr,

                    "TPR":
                        tpr,

                    "Threshold":
                        thresholds
                })
            )

            # PR data

            precision, recall, pr_thresholds = (
                precision_recall_curve(
                    membership,
                    scores
                )
            )

            pr_records.append(
                pd.DataFrame({

                    "Fold":
                        fold,

                    "Attack":
                        f"MIA_{attack_name}",

                    "Precision":
                        precision,

                    "Recall":
                        recall
                })
            )

        # LEARNED MIA

        if RUN_LEARNED_MIA:

            feature_columns = [

                "Probability",

                "Confidence",

                "Margin",

                "Entropy",

                "Loss",

                "Correct"
            ]

            X_attack = (
                attack_data[
                    feature_columns
                ]
                .values
            )

            y_attack = (
                attack_data[
                    "Membership"
                ]
                .values
                .astype(int)
            )

            class_counts = np.bincount(
                y_attack
            )

            if len(class_counts) >= 2:

                min_count = (
                    class_counts.min()
                )

                n_splits = min(
                    ATTACK_CV_FOLDS,
                    min_count
                )

                if n_splits >= 2:

                    pipeline = Pipeline([

                        (
                            "scaler",
                            StandardScaler()
                        ),

                        (
                            "classifier",
                            LogisticRegression(
                                max_iter=ATTACK_MAX_ITER,
                                class_weight="balanced",
                                random_state=RANDOM_STATE
                            )
                        )
                    ])

                    attack_scores = np.zeros(
                        len(y_attack)
                    )

                    skf = StratifiedKFold(
                        n_splits=n_splits,
                        shuffle=True,
                        random_state=RANDOM_STATE
                    )

                    for train_pos, test_pos in (
                        skf.split(
                            X_attack,
                            y_attack
                        )
                    ):

                        local_model = clone(
                            pipeline
                        )

                        local_model.fit(
                            X_attack[train_pos],
                            y_attack[train_pos]
                        )

                        attack_scores[test_pos] = (
                            local_model
                            .predict_proba(
                                X_attack[test_pos]
                            )[:, 1]
                        )

                    # Primary metrics remain ROC-AUC/PR-AUC.

                    predictions = (
                        attack_scores >= 0.5
                    ).astype(int)

                    metrics = mia_metrics(
                        y_attack,
                        attack_scores,
                        predictions
                    )

                    metrics["Fold"] = fold

                    metrics["Attack"] = (
                        "MIA_Learned_Logistic"
                    )

                    all_results.append(
                        metrics
                    )

                    attack_data[
                        "Learned_MIA_Score"
                    ] = attack_scores

                    attack_data[
                        "Learned_MIA_Prediction"
                    ] = predictions

                    attack_data.to_csv(
                        os.path.join(
                            MIA_RESULTS_DIR,
                            f"MIA_Fold_{fold}_Complete.csv"
                        ),
                        index=False
                    )

                    fpr, tpr, thresholds = roc_curve(
                        y_attack,
                        attack_scores
                    )

                    roc_records.append(
                        pd.DataFrame({

                            "Fold":
                                fold,

                            "Attack":
                                "MIA_Learned_Logistic",

                            "FPR":
                                fpr,

                            "TPR":
                                tpr,

                            "Threshold":
                                thresholds
                        })
                    )

                    precision, recall, _ = (
                        precision_recall_curve(
                            y_attack,
                            attack_scores
                        )
                    )

                    pr_records.append(
                        pd.DataFrame({

                            "Fold":
                                fold,

                            "Attack":
                                "MIA_Learned_Logistic",

                            "Precision":
                                precision,

                            "Recall":
                                recall
                        })
                    )

        # SUBJECT LEVEL MIA

        if RUN_SUBJECT_LEVEL_MIA:

            for subject in sorted(
                attack_data["Subject"]
                .unique()
            ):

                subject_data = (
                    attack_data[
                        attack_data["Subject"]
                        ==
                        subject
                    ]
                )

                subject_records.append({

                    "Fold":
                        fold,

                    "Subject":
                        subject,

                    "Membership":
                        int(
                            subject
                            in
                            member_subjects
                        ),

                    "Mean_Loss":
                        subject_data[
                            "Loss"
                        ].mean(),

                    "Mean_Confidence":
                        subject_data[
                            "Confidence"
                        ].mean(),

                    "Mean_Entropy":
                        subject_data[
                            "Entropy"
                        ].mean(),

                    "Mean_Correctness":
                        subject_data[
                            "Correct"
                        ].mean()
                })

        del model

        FL.tf.keras.backend.clear_session()

        gc.collect()

    mia_df = pd.DataFrame(
        all_results
    )

    if mia_df.empty:

        raise RuntimeError(
            "No MIA results generated."
        )

    mia_df.to_csv(
        os.path.join(
            MIA_RESULTS_DIR,
            "MIA_All_Fold_Results.csv"
        ),
        index=False
    )

    # SUMMARY

    metric_columns = [

        "Accuracy",

        "Balanced_Accuracy",

        "Sensitivity",

        "Specificity",

        "Precision",

        "F1",

        "ROC_AUC",

        "PR_AUC",

        "TPR_at_FPR_1pct",

        "TPR_at_FPR_5pct",

        "TPR_at_FPR_10pct"
    ]

    summary_rows = []

    for attack in sorted(
        mia_df["Attack"].unique()
    ):

        subset = mia_df[
            mia_df["Attack"]
            ==
            attack
        ]

        row = {

            "Attack":
                attack,

            "N_Folds":
                len(subset)
        }

        for metric in metric_columns:

            values = pd.to_numeric(
                subset[metric],
                errors="coerce"
            )

            row[
                f"{metric}_Mean"
            ] = values.mean()

            row[
                f"{metric}_SD"
            ] = values.std(
                ddof=1
            )

        summary_rows.append(
            row
        )

    mia_summary = pd.DataFrame(
        summary_rows
    )

    mia_summary.to_csv(
        os.path.join(
            MIA_RESULTS_DIR,
            "MIA_Summary_Mean_SD.csv"
        ),
        index=False
    )

    if roc_records:

        roc_df = pd.concat(
            roc_records,
            ignore_index=True
        )

        roc_df.to_csv(
            os.path.join(
                MIA_RESULTS_DIR,
                "MIA_ROC_Curves.csv"
            ),
            index=False
        )

    else:

        roc_df = pd.DataFrame()

    if pr_records:

        pr_df = pd.concat(
            pr_records,
            ignore_index=True
        )

        pr_df.to_csv(
            os.path.join(
                MIA_RESULTS_DIR,
                "MIA_PR_Curves.csv"
            ),
            index=False
        )

    else:

        pr_df = pd.DataFrame()

    if subject_records:

        subject_df = pd.DataFrame(
            subject_records
        )

        subject_df.to_csv(
            os.path.join(
                MIA_RESULTS_DIR,
                "MIA_Subject_Level.csv"
            ),
            index=False
        )

    return (
        mia_df,
        mia_summary,
        roc_df,
        pr_df
    )


# SENSOR FEATURE EXTRACTION

def flatten_sensor_features(
    BVP,
    EDA,
    TEMP,
    ACC
):

    def summarize(
        x,
        prefix
    ):

        x = np.asarray(x)

        if x.ndim == 1:

            x = x.reshape(
                -1,
                1
            )

        x2 = x.reshape(
            x.shape[0],
            -1
        )

        return pd.DataFrame({

            f"{prefix}_mean":
                np.mean(
                    x2,
                    axis=1
                ),

            f"{prefix}_std":
                np.std(
                    x2,
                    axis=1
                ),

            f"{prefix}_min":
                np.min(
                    x2,
                    axis=1
                ),

            f"{prefix}_max":
                np.max(
                    x2,
                    axis=1
                ),

            f"{prefix}_median":
                np.median(
                    x2,
                    axis=1
                ),

            f"{prefix}_q25":
                np.percentile(
                    x2,
                    25,
                    axis=1
                ),

            f"{prefix}_q75":
                np.percentile(
                    x2,
                    75,
                    axis=1
                ),

            f"{prefix}_energy":
                np.mean(
                    x2 ** 2,
                    axis=1
                ),

            f"{prefix}_range":
                np.max(
                    x2,
                    axis=1
                )
                -
                np.min(
                    x2,
                    axis=1
                )
        })

    X = pd.concat(
        [
            summarize(
                BVP,
                "BVP"
            ),

            summarize(
                EDA,
                "EDA"
            ),

            summarize(
                TEMP,
                "TEMP"
            ),

            summarize(
                ACC,
                "ACC"
            )
        ],
        axis=1
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )

    X = X.fillna(0.0)

    return X


# GET ATTRIBUTE

def get_attribute_array(
    groups
):

    if hasattr(
        FL,
        ATTRIBUTE_SOURCE
    ):

        attribute = np.asarray(
            getattr(
                FL,
                ATTRIBUTE_SOURCE
            )
        )

        if len(attribute) == len(groups):

            return attribute

    if ATTRIBUTE_SOURCE == "groups":

        return np.asarray(
            groups
        )

    raise ValueError(
        f"Attribute '{ATTRIBUTE_SOURCE}' "
        "was not found in the FL module."
    )


# ATTRIBUTE / SUBJECT IDENTITY INFERENCE

def run_aia(
    BVP,
    EDA,
    TEMP,
    ACC,
    y,
    groups,
    assignments
):

    print("\n" + "-" * 50)
    print(
        "ATTRIBUTE / SUBJECT-IDENTITY INFERENCE"
    )
    print("-" * 50)

    attribute = get_attribute_array(
        groups
    )

    attribute = attribute.astype(
        str
    )

    X = flatten_sensor_features(
        BVP,
        EDA,
        TEMP,
        ACC
    )

    X_values = X.values.astype(
        np.float64
    )

    group_strings = groups.astype(
        str
    )

    fold_results = []

    prediction_records = []

    confusion_matrices = {}

    for fold in sorted(
        assignments["Fold"].unique()
    ):

        print(
            f"AIA Fold {fold}"
        )

        fa = assignments[
            assignments["Fold"] == fold
        ]

        test_subjects = (
            fa[
                fa["Role"] == "Test"
            ]["Subject"]
            .astype(str)
            .tolist()
        )

        # -------------------------------------------------------------

        if ATTRIBUTE_SOURCE == "groups":

            print(
                "  Subject-disjoint identity inference:"
            )

            print(
                "  skipped for multiclass LogisticRegression because "
                "test subjects are unseen classes."
            )

            continue

        test_mask = np.isin(
            group_strings,
            test_subjects
        )

        train_idx = np.where(
            ~test_mask
        )[0]

        test_idx = np.where(
            test_mask
        )[0]

        if (
            len(train_idx) == 0
            or
            len(test_idx) == 0
        ):

            continue

        selected_train = []

        selected_test = []

        for subject in np.unique(
            group_strings[train_idx]
        ):

            idx = train_idx[
                group_strings[train_idx]
                ==
                subject
            ]

            if len(idx) > MAX_AIA_SAMPLES_PER_SUBJECT:

                idx = RNG.choice(
                    idx,
                    size=MAX_AIA_SAMPLES_PER_SUBJECT,
                    replace=False
                )

            selected_train.extend(
                idx.tolist()
            )

        for subject in np.unique(
            group_strings[test_idx]
        ):

            idx = test_idx[
                group_strings[test_idx]
                ==
                subject
            ]

            if len(idx) > MAX_AIA_SAMPLES_PER_SUBJECT:

                idx = RNG.choice(
                    idx,
                    size=MAX_AIA_SAMPLES_PER_SUBJECT,
                    replace=False
                )

            selected_test.extend(
                idx.tolist()
            )

        selected_train = np.asarray(
            selected_train
        )

        selected_test = np.asarray(
            selected_test
        )

        X_train = X_values[
            selected_train
        ]

        X_test = X_values[
            selected_test
        ]

        a_train = attribute[
            selected_train
        ]

        a_test = attribute[
            selected_test
        ]

        known = np.isin(
            a_test,
            np.unique(a_train)
        )

        X_test = X_test[
            known
        ]

        a_test = a_test[
            known
        ]

        if (
            len(np.unique(a_train)) < 2
            or
            len(np.unique(a_test)) < 2
        ):

            continue

        model = Pipeline([

            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=ATTACK_MAX_ITER,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ])

        model.fit(
            X_train,
            a_train
        )

        predictions = model.predict(
            X_test
        )

        acc = accuracy_score(
            a_test,
            predictions
        )

        bal_acc = balanced_accuracy_score(
            a_test,
            predictions
        )

        macro_precision = precision_score(
            a_test,
            predictions,
            average="macro",
            zero_division=0
        )

        macro_recall = recall_score(
            a_test,
            predictions,
            average="macro",
            zero_division=0
        )

        macro_f1 = f1_score(
            a_test,
            predictions,
            average="macro",
            zero_division=0
        )

        fold_results.append({

            "Fold":
                fold,

            "Accuracy":
                acc,

            "Balanced_Accuracy":
                bal_acc,

            "Macro_Precision":
                macro_precision,

            "Macro_Recall":
                macro_recall,

            "Macro_F1":
                macro_f1,

            "N_Train":
                len(a_train),

            "N_Test":
                len(a_test),

            "N_Classes":
                len(np.unique(a_test))
        })

        prediction_records.append(
            pd.DataFrame({

                "Fold":
                    fold,

                "True_Attribute":
                    a_test,

                "Predicted_Attribute":
                    predictions
            })
        )

        classes = np.unique(
            np.concatenate(
                [
                    a_test,
                    predictions
                ]
            )
        )

        cm = confusion_matrix(
            a_test,
            predictions,
            labels=classes
        )

        confusion_matrices[
            fold
        ] = (
            classes,
            cm
        )

    aia_df = pd.DataFrame(
        fold_results
    )

    if aia_df.empty:

        aia_summary = pd.DataFrame()

        return (
            aia_df,
            aia_summary,
            confusion_matrices
        )

    aia_df.to_csv(
        os.path.join(
            AIA_RESULTS_DIR,
            "AIA_All_Fold_Results.csv"
        ),
        index=False
    )

    metrics = [

        "Accuracy",

        "Balanced_Accuracy",

        "Macro_Precision",

        "Macro_Recall",

        "Macro_F1"
    ]

    summary_rows = []

    for metric in metrics:

        values = aia_df[
            metric
        ]

        summary_rows.append({

            "Metric":
                metric,

            "Mean":
                values.mean(),

            "SD":
                values.std(
                    ddof=1
                )
        })

    aia_summary = pd.DataFrame(
        summary_rows
    )

    aia_summary.to_csv(
        os.path.join(
            AIA_RESULTS_DIR,
            "AIA_Summary_Mean_SD.csv"
        ),
        index=False
    )

    if prediction_records:

        pd.concat(
            prediction_records,
            ignore_index=True
        ).to_csv(
            os.path.join(
                AIA_RESULTS_DIR,
                "AIA_All_Predictions.csv"
            ),
            index=False
        )

    return (
        aia_df,
        aia_summary,
        confusion_matrices
    )


# LABEL INFERENCE ATTACK

def run_lia(
    BVP,
    EDA,
    TEMP,
    ACC,
    y,
    groups,
    assignments
):

    print("\n" + "-" * 50)
    print(
        "LABEL INFERENCE ATTACK"
    )
    print("-" * 50)

    results = []

    roc_records = []

    pr_records = []

    prediction_records = []

    group_strings = groups.astype(
        str
    )

    for fold in sorted(
        assignments["Fold"].unique()
    ):

        print(
            f"LIA Fold {fold}"
        )

        fa = assignments[
            assignments["Fold"] == fold
        ]

        train_subjects = (
            fa[
                fa["Role"]
                ==
                MIA_MEMBER_ROLE
            ]["Subject"]
            .astype(str)
            .tolist()
        )

        test_subjects = (
            fa[
                fa["Role"]
                ==
                MIA_NONMEMBER_ROLE
            ]["Subject"]
            .astype(str)
            .tolist()
        )

        if (
            not train_subjects
            or
            not test_subjects
        ):

            continue

        train_idx = np.where(
            np.isin(
                group_strings,
                train_subjects
            )
        )[0]

        test_idx = np.where(
            np.isin(
                group_strings,
                test_subjects
            )
        )[0]

        if (
            len(train_idx) == 0
            or
            len(test_idx) == 0
        ):

            continue

        model = load_fold_model(
            fold
        )

        train_prob = predict_probabilities(
            model,
            BVP[train_idx],
            EDA[train_idx],
            TEMP[train_idx],
            ACC[train_idx]
        )

        test_prob = predict_probabilities(
            model,
            BVP[test_idx],
            EDA[test_idx],
            TEMP[test_idx],
            ACC[test_idx]
        )

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        # DIRECT MODEL OUTPUT

        direct_pred = (
            test_prob >= 0.5
        ).astype(int)

        results.append({

            "Fold":
                fold,

            "Attack":
                "LIA_Direct_Model_Output",

            "Accuracy":
                accuracy_score(
                    y_test,
                    direct_pred
                ),

            "Balanced_Accuracy":
                balanced_accuracy_score(
                    y_test,
                    direct_pred
                ),

            "Sensitivity":
                recall_score(
                    y_test,
                    direct_pred,
                    zero_division=0
                ),

            "Specificity":
                recall_score(
                    y_test,
                    direct_pred,
                    pos_label=0,
                    zero_division=0
                ),

            "Precision":
                precision_score(
                    y_test,
                    direct_pred,
                    zero_division=0
                ),

            "F1":
                f1_score(
                    y_test,
                    direct_pred,
                    zero_division=0
                ),

            "ROC_AUC":
                safe_auc(
                    y_test,
                    test_prob
                ),

            "PR_AUC":
                safe_pr_auc(
                    y_test,
                    test_prob
                )
        })

        fpr, tpr, thresholds = roc_curve(
            y_test,
            test_prob
        )

        roc_records.append(
            pd.DataFrame({

                "Fold":
                    fold,

                "Attack":
                    "LIA_Direct_Model_Output",

                "FPR":
                    fpr,

                "TPR":
                    tpr,

                "Threshold":
                    thresholds
            })
        )

        precision, recall, _ = (
            precision_recall_curve(
                y_test,
                test_prob
            )
        )

        pr_records.append(
            pd.DataFrame({

                "Fold":
                    fold,

                "Attack":
                    "LIA_Direct_Model_Output",

                "Precision":
                    precision,

                "Recall":
                    recall
            })
        )

        # LEARNED LIA

        if RUN_LEARNED_LIA:

            train_features = probability_features(
                y_train,
                train_prob
            )

            test_features = probability_features(
                y_test,
                test_prob
            )

            feature_columns = [

                "Probability",

                "Confidence",

                "Margin",

                "Entropy"
            ]

            X_train = (
                train_features[
                    feature_columns
                ]
                .values
            )

            X_test = (
                test_features[
                    feature_columns
                ]
                .values
            )

            attack_model = Pipeline([

                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "classifier",
                    LogisticRegression(
                        max_iter=ATTACK_MAX_ITER,
                        class_weight="balanced",
                        random_state=RANDOM_STATE
                    )
                )
            ])

            attack_model.fit(
                X_train,
                y_train
            )

            attack_probability = (
                attack_model
                .predict_proba(
                    X_test
                )[:, 1]
            )

            attack_prediction = (
                attack_probability >= 0.5
            ).astype(int)

            results.append({

                "Fold":
                    fold,

                "Attack":
                    "LIA_Learned_Output_Attack",

                "Accuracy":
                    accuracy_score(
                        y_test,
                        attack_prediction
                    ),

                "Balanced_Accuracy":
                    balanced_accuracy_score(
                        y_test,
                        attack_prediction
                    ),

                "Sensitivity":
                    recall_score(
                        y_test,
                        attack_prediction,
                        zero_division=0
                    ),

                "Specificity":
                    recall_score(
                        y_test,
                        attack_prediction,
                        pos_label=0,
                        zero_division=0
                    ),

                "Precision":
                    precision_score(
                        y_test,
                        attack_prediction,
                        zero_division=0
                    ),

                "F1":
                    f1_score(
                        y_test,
                        attack_prediction,
                        zero_division=0
                    ),

                "ROC_AUC":
                    safe_auc(
                        y_test,
                        attack_probability
                    ),

                "PR_AUC":
                    safe_pr_auc(
                        y_test,
                        attack_probability
                    )
            })

            fpr, tpr, thresholds = roc_curve(
                y_test,
                attack_probability
            )

            roc_records.append(
                pd.DataFrame({

                    "Fold":
                        fold,

                    "Attack":
                        "LIA_Learned_Output_Attack",

                    "FPR":
                        fpr,

                    "TPR":
                        tpr,

                    "Threshold":
                        thresholds
                })
            )

            precision, recall, _ = (
                precision_recall_curve(
                    y_test,
                    attack_probability
                )
            )

            pr_records.append(
                pd.DataFrame({

                    "Fold":
                        fold,

                    "Attack":
                        "LIA_Learned_Output_Attack",

                    "Precision":
                        precision,

                    "Recall":
                        recall
                })
            )

            prediction_records.append(
                pd.DataFrame({

                    "Fold":
                        fold,

                    "Subject":
                        group_strings[
                            test_idx
                        ],

                    "True_Label":
                        y_test,

                    "Model_Probability":
                        test_prob,

                    "Direct_Prediction":
                        direct_pred,

                    "Attack_Probability":
                        attack_probability,

                    "Attack_Prediction":
                        attack_prediction
                })
            )

        del model

        FL.tf.keras.backend.clear_session()

        gc.collect()

    lia_df = pd.DataFrame(
        results
    )

    if lia_df.empty:

        return (
            lia_df,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )

    lia_df.to_csv(
        os.path.join(
            LIA_RESULTS_DIR,
            "LIA_All_Fold_Results.csv"
        ),
        index=False
    )

    metrics = [

        "Accuracy",

        "Balanced_Accuracy",

        "Sensitivity",

        "Specificity",

        "Precision",

        "F1",

        "ROC_AUC",

        "PR_AUC"
    ]

    summary_rows = []

    for attack in sorted(
        lia_df["Attack"].unique()
    ):

        subset = lia_df[
            lia_df["Attack"]
            ==
            attack
        ]

        row = {

            "Attack":
                attack,

            "N_Folds":
                len(subset)
        }

        for metric in metrics:

            values = subset[
                metric
            ]

            row[
                f"{metric}_Mean"
            ] = values.mean()

            row[
                f"{metric}_SD"
            ] = values.std(
                ddof=1
            )

        summary_rows.append(
            row
        )

    lia_summary = pd.DataFrame(
        summary_rows
    )

    lia_summary.to_csv(
        os.path.join(
            LIA_RESULTS_DIR,
            "LIA_Summary_Mean_SD.csv"
        ),
        index=False
    )

    roc_df = pd.DataFrame()

    pr_df = pd.DataFrame()

    if roc_records:

        roc_df = pd.concat(
            roc_records,
            ignore_index=True
        )

        roc_df.to_csv(
            os.path.join(
                LIA_RESULTS_DIR,
                "LIA_ROC_Curves.csv"
            ),
            index=False
        )

    if pr_records:

        pr_df = pd.concat(
            pr_records,
            ignore_index=True
        )

        pr_df.to_csv(
            os.path.join(
                LIA_RESULTS_DIR,
                "LIA_PR_Curves.csv"
            ),
            index=False
        )

    if prediction_records:

        pd.concat(
            prediction_records,
            ignore_index=True
        ).to_csv(
            os.path.join(
                LIA_RESULTS_DIR,
                "LIA_All_Predictions.csv"
            ),
            index=False
        )

    return (
        lia_df,
        lia_summary,
        roc_df,
        pr_df
    )


# PUBLICATION FIGURE: MIA ROC

def plot_mia_roc(
    roc_df,
    mia_summary
):

    if roc_df.empty:

        return

    fig, ax = plt.subplots(
        figsize=(7.2, 6.0)
    )

    attacks = [
        a
        for a in sorted(
            roc_df["Attack"].unique()
        )
    ]

    for attack in attacks:

        subset = roc_df[
            roc_df["Attack"]
            ==
            attack
        ]

        # Use a different publication colour for each MIA method
        color = MIA_COLORS.get(
            attack,
            COLORS["MIA"]
        )

        if attack == "MIA_Confidence":
            label = "Confidence"

        elif attack == "MIA_Loss":
            label = "Loss"

        elif attack == "MIA_Entropy":
            label = "Entropy"

        elif attack == "MIA_Correctness":
            label = "Correctness"

        elif attack == "MIA_Learned_Logistic":
            label = "Learned MIA"

        else:
            label = attack

        for fold in sorted(
            subset["Fold"].unique()
        ):

            f = subset[
                subset["Fold"]
                ==
                fold
            ]

            ax.plot(
                f["FPR"],
                f["TPR"],
                color=color,
                alpha=0.12,
                linewidth=1.0
            )

        # A simple representative pooled-style visualization
        # Interpolate each fold onto a common FPR grid.

        grid = np.linspace(
            0,
            1,
            501
        )

        interpolated = []

        for fold in sorted(
            subset["Fold"].unique()
        ):

            f = subset[
                subset["Fold"]
                ==
                fold
            ].sort_values(
                "FPR"
            )

            x = f["FPR"].values
            y = f["TPR"].values

            x_unique, indices = np.unique(
                x,
                return_index=True
            )

            y_unique = y[
                indices
            ]

            interpolated.append(
                np.interp(
                    grid,
                    x_unique,
                    y_unique
                )
            )

        mean_tpr = np.mean(
            interpolated,
            axis=0
        )

        ax.plot(
            grid,
            mean_tpr,
            color=color,
            linewidth=MEAN_LINE_WIDTH,
            label=label
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=COLORS["Random"],
        linewidth=1.5,
        label="Random"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Membership Inference Attack: ROC Curves",
        fontweight="bold"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.xaxis.set_major_locator(
        MultipleLocator(0.2)
    )

    ax.yaxis.set_major_locator(
        MultipleLocator(0.2)
    )

    ax.grid(
        linestyle="--",
        alpha=GRID_ALPHA
    )

    ax.legend(
        loc="lower right"
    )

    fig.tight_layout()

    save_publication_figure(
        fig,
        "Figure_1_MIA_ROC"
    )

    plt.close(fig)


# PUBLICATION FIGURE: MIA PR=

def plot_mia_pr(
    pr_df
):

    if pr_df.empty:

        return

    fig, ax = plt.subplots(
        figsize=(7.2, 6.0)
    )

    attacks = sorted(
        pr_df["Attack"].unique()
    )

    for attack in attacks:

        subset = pr_df[
            pr_df["Attack"]
            ==
            attack
        ]

        if attack == "MIA_Confidence":
            label = "Confidence"

        elif attack == "MIA_Loss":
            label = "Loss"

        elif attack == "MIA_Entropy":
            label = "Entropy"

        elif attack == "MIA_Correctness":
            label = "Correctness"

        elif attack == "MIA_Learned_Logistic":
            label = "Learned MIA"

        else:
            label = attack

        # Use a different publication colour for each MIA method
        color = MIA_COLORS.get(
            attack,
            COLORS["MIA"]
        )

        for fold in sorted(
            subset["Fold"].unique()
        ):

            f = subset[
                subset["Fold"]
                ==
                fold
            ]

            ax.plot(
                f["Recall"],
                f["Precision"],
                color=color,
                alpha=0.12,
                linewidth=1.0
            )

        grid = np.linspace(
            0,
            1,
            501
        )

        curves = []

        for fold in sorted(
            subset["Fold"].unique()
        ):

            f = subset[
                subset["Fold"]
                ==
                fold
            ].sort_values(
                "Recall"
            )

            recall = f[
                "Recall"
            ].values

            precision = f[
                "Precision"
            ].values

            recall_unique, indices = (
                np.unique(
                    recall,
                    return_index=True
                )
            )

            precision_unique = (
                precision[indices]
            )

            curves.append(
                np.interp(
                    grid,
                    recall_unique,
                    precision_unique
                )
            )

        mean_curve = np.mean(
            curves,
            axis=0
        )

        ax.plot(
            grid,
            mean_curve,
            color=color,
            linewidth=MEAN_LINE_WIDTH,
            label=label
        )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Membership Inference Attack: Precision–Recall",
        fontweight="bold"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        linestyle="--",
        alpha=GRID_ALPHA
    )

    ax.legend(
        loc="lower left"
    )

    fig.tight_layout()

    save_publication_figure(
        fig,
        "Figure_2_MIA_PR"
    )

    plt.close(fig)

# PUBLICATION FIGURE: MIA PERFORMANCE

def plot_mia_performance(
    mia_df
):

    metric = "ROC_AUC"

    methods = sorted(
        mia_df["Attack"].unique()
    )

    means = []

    sds = []

    labels = []

    for method in methods:

        values = mia_df[
            mia_df["Attack"]
            ==
            method
        ][metric]

        means.append(
            values.mean()
        )

        sds.append(
            values.std(
                ddof=1
            )
        )

        labels.append(
            method.replace(
                "MIA_",
                ""
            ).replace(
                "_",
                " "
            )
        )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    x = np.arange(
        len(labels)
    )

    bar_colors = [
        MIA_COLORS.get(
            method,
            COLORS["MIA"]
        )
        for method in methods
    ]

    bars = ax.bar(
        x,
        means,
        yerr=sds,
        capsize=4,
        color=bar_colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.7
    )

    ax.axhline(
        0.5,
        linestyle="--",
        color=COLORS["Random"],
        linewidth=1.5,
        label="Random guessing"
    )

    for bar, mean in zip(
        bars,
        means
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            mean + 0.025,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=25,
        ha="right"
    )

    ax.set_ylabel(
        "ROC-AUC"
    )

    ax.set_title(
        "Membership Inference Performance",
        fontweight="bold"
    )

    ax.set_ylim(
        0.4,
        1.0
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=GRID_ALPHA
    )

    ax.legend()

    fig.tight_layout()

    save_publication_figure(
        fig,
        "Figure_3_MIA_Performance"
    )

    plt.close(fig)

# PUBLICATION FIGURE: AIA CONFUSION MATRIX

def plot_aia_confusion_matrices(
    confusion_matrices
):

    if not confusion_matrices:

        return

    # Use the first available valid fold.

    fold = sorted(
        confusion_matrices.keys()
    )[0]

    classes, cm = (
        confusion_matrices[fold]
    )

    cm_normalized = (
        cm
        /
        np.maximum(
            cm.sum(
                axis=1,
                keepdims=True
            ),
            1
        )
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 6.0)
    )

    im = ax.imshow(
        cm_normalized,
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Proportion"
    )

    ax.set_xticks(
        np.arange(
            len(classes)
        )
    )

    ax.set_yticks(
        np.arange(
            len(classes)
        )
    )

    ax.set_xticklabels(
        classes,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        classes
    )

    ax.set_xlabel(
        "Predicted Attribute"
    )

    ax.set_ylabel(
        "True Attribute"
    )

    ax.set_title(
        f"Attribute Inference Confusion Matrix "
        f"(Fold {fold})",
        fontweight="bold"
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            value = cm_normalized[
                i,
                j
            ]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > 0.5
                    else "black"
                ),
                fontsize=9
            )

    fig.tight_layout()

    save_publication_figure(
        fig,
        "Figure_4_AIA_Confusion_Matrix"
    )

    plt.close(fig)


# PUBLICATION FIGURE: AIA PERFORMANCE

def plot_aia_performance(
    aia_df
):

    if aia_df.empty:

        return

    metrics = [

        "Accuracy",

        "Balanced_Accuracy",

        "Macro_F1"
    ]

    labels = [

        "Accuracy",

        "Balanced Accuracy",

        "Macro-F1"
    ]

    means = []

    sds = []

    for metric in metrics:

        values = aia_df[
            metric
        ]

        means.append(
            values.mean()
        )

        sds.append(
            values.std(
                ddof=1
            )
        )

    fig, ax = plt.subplots(
        figsize=(7.5, 5.5)
    )

    x = np.arange(
        len(metrics)
    )

    bars = ax.bar(
        x,
        means,
        yerr=sds,
        capsize=4,
        color=COLORS["AIA"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.7
    )

    ax.axhline(
        0.5,
        linestyle="--",
        color=COLORS["Random"],
        linewidth=1.5
    )

    for bar, mean in zip(
        bars,
        means
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            mean + 0.025,
            f"{mean:.3f}",
            ha="center",
            fontsize=9
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Performance"
    )

    ax.set_title(
        "Attribute Inference Performance",
        fontweight="bold"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=GRID_ALPHA
    )

    fig.tight_layout()

    save_publication_figure(
        fig,
        "Figure_5_AIA_Performance"
    )

    plt.close(fig)


# PUBLICATION FIGURE: LIA ROC

def plot_lia_roc(
    roc_df
):

    if roc_df.empty:

        return

    fig, ax = plt.subplots(
        figsize=(7.2, 6.0)
    )

    for attack in sorted(
        roc_df["Attack"].unique()
    ):

        subset = roc_df[
            roc_df["Attack"]
            ==
            attack
        ]

        if attack == "LIA_Direct_Model_Output":

            label = "Direct model output"

        else:

            label = "Learned label attack"

        for fold in sorted(
            subset["Fold"].unique()
        ):

            f = subset[
                subset["Fold"]
                ==
                fold
            ]

            ax.plot(
                f["FPR"],
                f["TPR"],
                color=LIA_COLORS.get(
                    attack,
                    COLORS["LIA"]
                ),
                alpha=0.15,
                linewidth=1
            )

        grid = np.linspace(
            0,
            1,
            501
        )

        curves = []

        for fold in sorted(
            subset["Fold"].unique()
        ):

            f = subset[
                subset["Fold"]
                ==
                fold
            ].sort_values(
                "FPR"
            )

            x = f["FPR"].values
            y = f["TPR"].values

            x_unique, idx = np.unique(
                x,
                return_index=True
            )

            y_unique = y[
                idx
            ]

            curves.append(
                np.interp(
                    grid,
                    x_unique,
                    y_unique
                )
            )

        mean_curve = np.mean(
            curves,
            axis=0
        )

        ax.plot(
            grid,
            mean_curve,
            color=LIA_COLORS.get(
                attack,
                COLORS["LIA"]
            ),
            linewidth=MEAN_LINE_WIDTH,
            label=label
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color=COLORS["Random"],
        linewidth=1.5,
        label="Random"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Label Inference Attack: ROC",
        fontweight="bold"
    )

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        linestyle="--",
        alpha=GRID_ALPHA
    )

    ax.legend(
        loc="lower right"
    )

    fig.tight_layout()

    save_publication_figure(
        fig,
        "Figure_6_LIA_ROC"
    )

    plt.close(fig)


# PUBLICATION FIGURE: LIA PERFORMANCE

def plot_lia_performance(
    lia_df
):

    if lia_df.empty:

        return

    methods = sorted(
        lia_df["Attack"].unique()
    )

    fig, ax = plt.subplots(
        figsize=(8.0, 5.5)
    )

    x = np.arange(
        len(methods)
    )

    means = []

    sds = []

    labels = []

    for method in methods:

        values = lia_df[
            lia_df["Attack"]
            ==
            method
        ]["ROC_AUC"]

        means.append(
            values.mean()
        )

        sds.append(
            values.std(
                ddof=1
            )
        )

        if "Direct" in method:

            labels.append(
                "Direct model output"
            )

        else:

            labels.append(
                "Learned output attack"
            )

    bars = ax.bar(
        x,
        means,
        yerr=sds,
        capsize=4,
        color=COLORS["LIA"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.7
    )

    ax.axhline(
        0.5,
        linestyle="--",
        color=COLORS["Random"],
        linewidth=1.5,
        label="Random"
    )

    for bar, mean in zip(
        bars,
        means
    ):

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            mean + 0.025,
            f"{mean:.3f}",
            ha="center",
            fontsize=9
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=15,
        ha="right"
    )

    ax.set_ylabel(
        "ROC-AUC"
    )

    ax.set_title(
        "Label Inference Performance",
        fontweight="bold"
    )

    ax.set_ylim(
        0.4,
        1.0
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=GRID_ALPHA
    )

    ax.legend()

    fig.tight_layout()

    save_publication_figure(
        fig,
        "Figure_7_LIA_Performance"
    )

    plt.close(fig)


# OVERALL PRIVACY RISK SUMMARY

def create_privacy_summary(
    mia_df,
    aia_df,
    lia_df
):

    rows = []

    # MIA

    if (
        mia_df is not None
        and
        not mia_df.empty
    ):

        for attack in sorted(
            mia_df["Attack"].unique()
        ):

            subset = mia_df[
                mia_df["Attack"]
                ==
                attack
            ]

            rows.append({

                "Attack_Type":
                    "Membership Inference",

                "Method":
                    attack,

                "Primary_Metric":
                    "ROC_AUC",

                "Mean":
                    subset[
                        "ROC_AUC"
                    ].mean(),

                "SD":
                    subset[
                        "ROC_AUC"
                    ].std(
                        ddof=1
                    )
            })

    # AIA

    if (
        aia_df is not None
        and
        not aia_df.empty
    ):

        rows.append({

            "Attack_Type":
                "Attribute Inference",

            "Method":
                "Learned Attribute Attack",

            "Primary_Metric":
                "Balanced Accuracy",

            "Mean":
                aia_df[
                    "Balanced_Accuracy"
                ].mean(),

            "SD":
                aia_df[
                    "Balanced_Accuracy"
                ].std(
                    ddof=1
                )
        })

    # LIA

    if (
        lia_df is not None
        and
        not lia_df.empty
    ):

        for attack in sorted(
            lia_df["Attack"].unique()
        ):

            subset = lia_df[
                lia_df["Attack"]
                ==
                attack
            ]

            rows.append({

                "Attack_Type":
                    "Label Inference",

                "Method":
                    attack,

                "Primary_Metric":
                    "ROC_AUC",

                "Mean":
                    subset[
                        "ROC_AUC"
                    ].mean(),

                "SD":
                    subset[
                        "ROC_AUC"
                    ].std(
                        ddof=1
                    )
            })

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        os.path.join(
            ATTACK_RESULTS_DIR,
            "Privacy_Attack_Primary_Metrics.csv"
        ),
        index=False
    )

    return summary


# PUBLICATION FIGURE: OVERALL PRIVACY RISK

def plot_overall_privacy_risk(
    summary
):

    if summary.empty:

        return

    labels = (
        summary["Attack_Type"]
        +
        "\n"
        +
        summary["Method"]
        .str.replace(
            "MIA_",
            "",
            regex=False
        )
        .str.replace(
            "LIA_",
            "",
            regex=False
        )
        .str.replace(
            "_",
            " ",
            regex=False
        )
    )

    means = (
        summary["Mean"]
        .astype(float)
        .values
    )

    sds = (
        summary["SD"]
        .astype(float)
        .values
    )

    colors = []

    for attack_type in (
        summary["Attack_Type"]
    ):

        if attack_type == "Membership Inference":

            colors.append(
                COLORS["MIA"]
            )

        elif attack_type == "Attribute Inference":

            colors.append(
                COLORS["AIA"]
            )

        else:

            colors.append(
                COLORS["LIA"]
            )

    fig, ax = plt.subplots(
        figsize=(10.0, 6.0)
    )

    x = np.arange(
        len(labels)
    )

    bars = ax.bar(
        x,
        means,
        yerr=sds,
        capsize=4,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.7
    )

    ax.axhline(
        0.5,
        linestyle="--",
        color=COLORS["Random"],
        linewidth=1.5,
        label="Random baseline"
    )

    for bar, mean in zip(
        bars,
        means
    ):

        if np.isfinite(mean):

            ax.text(
                bar.get_x()
                +
                bar.get_width() / 2,
                mean + 0.025,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right"
    )

    ax.set_ylabel(
        "Primary attack performance"
    )

    ax.set_title(
        "Privacy Attack Performance Summary",
        fontweight="bold"
    )

    ax.set_ylim(
        0,
        1
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=GRID_ALPHA
    )

    ax.legend()

    fig.tight_layout()

    save_publication_figure(
        fig,
        "Figure_8_Overall_Privacy_Risk"
    )

    plt.close(fig)


# DATASET SUMMARY

def create_dataset_summary(
    y,
    groups
):

    summary = pd.DataFrame({

        "Quantity": [

            "Number of windows",

            "Number of subjects",

            "Class 0 windows",

            "Class 1 windows",

            "Class 1 prevalence"
        ],

        "Value": [

            len(y),

            len(
                np.unique(
                    groups
                )
            ),

            int(
                np.sum(
                    y == 0
                )
            ),

            int(
                np.sum(
                    y == 1
                )
            ),

            float(
                np.mean(
                    y == 1
                )
            )
        ]
    })

    summary.to_csv(
        os.path.join(
            ATTACK_RESULTS_DIR,
            "Dataset_Summary.csv"
        ),
        index=False
    )

    return summary


#  SAVE EXPERIMENT CONFIGURATION

def save_configuration():

    configuration = {

        "Existing_FL_Script":
            os.path.abspath(
                EXISTING_FL_SCRIPT
            ),

        "Results_Directory":
            os.path.abspath(
                RESULTS_DIR
            ),

        "Attack_Results_Directory":
            os.path.abspath(
                ATTACK_RESULTS_DIR
            ),

        "Random_State":
            RANDOM_STATE,

        "Attack_CV_Folds":
            ATTACK_CV_FOLDS,

        "Attack_Max_Iterations":
            ATTACK_MAX_ITER,

        "Attribute_Source":
            ATTRIBUTE_SOURCE,

        "Attribute_Attack_Type":
            ATTRIBUTE_ATTACK_TYPE,

        "MIA_Member_Role":
            MIA_MEMBER_ROLE,

        "MIA_Nonmember_Role":
            MIA_NONMEMBER_ROLE,

        "Run_Learned_MIA":
            RUN_LEARNED_MIA,

        "Run_AIA":
            RUN_AIA,

        "Run_Learned_LIA":
            RUN_LEARNED_LIA,

        "Figure_DPI":
            FIG_DPI,

        "Figure_Formats":
            FIG_FORMATS
    }

    with open(
        os.path.join(
            ATTACK_RESULTS_DIR,
            "Privacy_Attack_Configuration.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            configuration,
            f,
            indent=4
        )


# MAIN

def main():

    print("\n")
    print("-" * 50)
    print(
        "PUBLICATION-READY WESAD PRIVACY ATTACK EVALUATION"
    )
    print("-" * 50)

    save_configuration()

    # DATA

    (
        BVP,
        EDA,
        TEMP,
        ACC,
        y,
        groups
    ) = load_dataset()

    create_dataset_summary(
        y,
        groups
    )

    # ASSIGNMENTS

    assignments = (
        load_subject_assignments()
    )

    # MIA

    (
        mia_df,
        mia_summary,
        mia_roc_df,
        mia_pr_df
    ) = run_mia(
        BVP,
        EDA,
        TEMP,
        ACC,
        y,
        groups,
        assignments
    )

    # AIA

    if RUN_AIA:

        (
            aia_df,
            aia_summary,
            aia_confusion
        ) = run_aia(
            BVP,
            EDA,
            TEMP,
            ACC,
            y,
            groups,
            assignments
        )

    else:

        aia_df = pd.DataFrame()

        aia_summary = pd.DataFrame()

        aia_confusion = {}

    # LIA

    (
        lia_df,
        lia_summary,
        lia_roc_df,
        lia_pr_df
    ) = run_lia(
        BVP,
        EDA,
        TEMP,
        ACC,
        y,
        groups,
        assignments
    )

    # OVERALL SUMMARY

    privacy_summary = (
        create_privacy_summary(
            mia_df,
            aia_df,
            lia_df
        )
    )

    # FIGURES

    if CREATE_FIGURES:

        print("\n")
        print("-" * 50)
        print(
            "CREATING PUBLICATION FIGURES"
        )
        print("-" * 50)

        plot_mia_roc(
            mia_roc_df,
            mia_summary
        )

        plot_mia_pr(
            mia_pr_df
        )

        plot_mia_performance(
            mia_df
        )

        plot_aia_confusion_matrices(
            aia_confusion
        )

        plot_aia_performance(
            aia_df
        )

        plot_lia_roc(
            lia_roc_df
        )

        plot_lia_performance(
            lia_df
        )

        plot_overall_privacy_risk(
            privacy_summary
        )


    # PRINT RESULTS

    print("\n")
    print("-" * 50)
    print(
        "MIA SUMMARY"
    )
    print("-" * 50)

    print(
        mia_summary.round(4).to_string(
            index=False
        )
    )

    if not aia_summary.empty:

        print("\n")
        print("-" * 50)
        print(
            "AIA SUMMARY"
        )
        print("-" * 50)

        print(
            aia_summary.round(4).to_string(
                index=False
            )
        )

    print("\n")
    print("-" * 50)
    print(
        "LIA SUMMARY"
    )
    print("-" * 50)

    if not lia_summary.empty:

        print(
            lia_summary.round(4).to_string(
                index=False
            )
        )

    print("\n")
    print("-" * 50)
    print(
        "PRIMARY PRIVACY METRICS"
    )
    print("-" * 50)

    if not privacy_summary.empty:

        print(
            privacy_summary.round(4).to_string(
                index=False
            )
        )

    print("\n")
    print("-" * 50)
    print(
        "PRIVACY ATTACK ANALYSIS COMPLETED"
    )
    print("-" * 50)

    print(
        "\nResults:"
    )

    print(
        os.path.abspath(
            ATTACK_RESULTS_DIR
        )
    )

    print(
        "\nPublication figures:"
    )

    print(
        os.path.abspath(
            FIGURES_DIR
        )
    )

    print("\nFigures generated:")

    print(
        "1. Figure_1_MIA_ROC"
    )

    print(
        "2. Figure_2_MIA_PR"
    )

    print(
        "3. Figure_3_MIA_Performance"
    )

    print(
        "4. Figure_4_AIA_Confusion_Matrix"
    )

    print(
        "5. Figure_5_AIA_Performance"
    )

    print(
        "6. Figure_6_LIA_ROC"
    )

    print(
        "7. Figure_7_LIA_Performance"
    )

    print(
        "8. Figure_8_Overall_Privacy_Risk"
    )

    return {

        "MIA":
            mia_df,

        "MIA_Summary":
            mia_summary,

        "AIA":
            aia_df,

        "AIA_Summary":
            aia_summary,

        "LIA":
            lia_df,

        "LIA_Summary":
            lia_summary,

        "Privacy_Summary":
            privacy_summary
    }


# EXECUTION

if __name__ == "__main__":

    results = main()