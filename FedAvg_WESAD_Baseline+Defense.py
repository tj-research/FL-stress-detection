# IMPORTS

import os
import pickle
import random
import warnings
import gc

import numpy as np
import pandas as pd

from scipy import signal

import tensorflow as tf

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns


warnings.filterwarnings("ignore")


# CONFIGURATION
DATASET_PATH = (
    r"directory to the dataset"
)

SUBJECTS = [
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "S9",
    "S10",
    "S11",
    "S12",
    "S13",
    "S14",
    "S15",
    "S16",
    "S17"
]

# WINDOW CONFIGURATION
WINDOW_SECONDS = 60
STEP_SECONDS = 30

# ORIGINAL WESAD SAMPLING FREQUENCIES
FS_BVP = 64
FS_EDA = 4
FS_TEMP = 4
FS_ACC = 32
FS_LABEL = 700

# TARGET SAMPLING FREQUENCIES
TARGET_FS_BVP = 16
TARGET_FS_ACC = 16
TARGET_FS_EDA = 4
TARGET_FS_TEMP = 4

# FEDERATED LEARNING CONFIGURATION
NUM_ROUNDS = 15
LOCAL_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.0003

# EARLY STOPPING
FEDERATED_EARLY_STOPPING_PATIENCE = 2

# THRESHOLD
DEFAULT_THRESHOLD = 0.50

# CROSS-VALIDATION
N_FOLDS = 5
RANDOM_STATE = 42

# PRIVACY DEFENSE
# Options:
#     "none"
#     "clip"
#     "clip_dp"

PRIVACY_MODE = "none" # none = no defense mechanisms (Baseline), clip = Baseline + CLIP_NORM, clip_dp = Baseline + CLIP_NORM + NOISE_MULTIPLIER
CLIP_NORM = 0.5 # do not change this
NOISE_MULTIPLIER = 0.05 #(0.05, 0.5)
privacy_rng = np.random.default_rng(RANDOM_STATE)

# OUTPUT DIRECTORY
RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# REPRODUCIBILITY

os.environ["PYTHONHASHSEED"] = str(
    RANDOM_STATE
)

random.seed(
    RANDOM_STATE
)

np.random.seed(
    RANDOM_STATE
)

tf.random.set_seed(
    RANDOM_STATE
)

# GPU CONFIGURATION

print("\n" + "-" * 50)
print("GPU CONFIGURATION")
print("-" * 50)

print(
    "TensorFlow version:",
    tf.__version__
)

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print(
        f"GPU(s) detected: {len(gpus)}"
    )

    for i, gpu in enumerate(gpus):

        print(
            f"  GPU {i}: {gpu}"
        )

        try:

            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

        except RuntimeError as e:

            print(
                "Memory growth warning:",
                e
            )

else:

    print(
        "WARNING: No GPU detected."
    )

    print(
        "TensorFlow will run on CPU."
    )


print(
    "\nAvailable TensorFlow devices:"
)

print(
    tf.config.list_physical_devices()
)

print(
    "\nGPU devices:"
)

print(
    tf.config.list_physical_devices("GPU")
)

print("-" * 50)


# SIGNAL CLEANING

def clean_signal(x):

    x = np.asarray(
        x,
        dtype=np.float32
    ).flatten()

    if len(x) == 0:
        return x

    x[
        ~np.isfinite(x)
    ] = np.nan

    if np.all(
        np.isnan(x)
    ):

        return np.zeros(
            len(x),
            dtype=np.float32
        )

    s = pd.Series(x)

    s = (
        s
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )

    return s.values.astype(
        np.float32
    )


# BANDPASS FILTER

def butter_bandpass(
    x,
    fs,
    lowcut,
    highcut,
    order=3
):

    x = clean_signal(x)

    if len(x) < 20:
        return x

    nyquist = fs / 2.0

    low = lowcut / nyquist
    high = highcut / nyquist

    low = max(low, 1e-5)
    high = min(high, 0.99)

    if low >= high:
        return x

    try:

        b, a = signal.butter(
            order,
            [low, high],
            btype="band"
        )

        return signal.filtfilt(
            b,
            a,
            x
        ).astype(np.float32)

    except Exception:

        return x


# LOWPASS FILTER

def butter_lowpass(
    x,
    fs,
    cutoff,
    order=3
):

    x = clean_signal(x)

    if len(x) < 20:
        return x

    nyquist = fs / 2.0

    normalized_cutoff = (
        cutoff / nyquist
    )

    normalized_cutoff = min(
        normalized_cutoff,
        0.95
    )

    if normalized_cutoff <= 0:
        return x

    try:

        b, a = signal.butter(
            order,
            normalized_cutoff,
            btype="low"
        )

        return signal.filtfilt(
            b,
            a,
            x
        ).astype(np.float32)

    except Exception:

        return x

# ANTI-ALIAS LOWPASS

def anti_alias_lowpass(
    x,
    original_fs,
    target_fs
):

    cutoff = 0.45 * target_fs

    return butter_lowpass(
        x,
        original_fs,
        cutoff,
        order=4
    )


# RESAMPLE SIGNAL

def resample_signal(
    x,
    original_fs,
    target_fs
):

    x = clean_signal(x)

    if len(x) == 0:

        return np.zeros(
            0,
            dtype=np.float32
        )

    if original_fs == target_fs:

        return x.astype(
            np.float32
        )

    duration = (
        len(x) /
        original_fs
    )

    target_length = int(
        round(
            duration *
            target_fs
        )
    )

    if target_length <= 0:

        return np.zeros(
            0,
            dtype=np.float32
        )

    if target_fs < original_fs:

        x = anti_alias_lowpass(
            x,
            original_fs,
            target_fs
        )

    if len(x) < 2:

        if len(x) == 0:

            return np.zeros(
                target_length,
                dtype=np.float32
            )

        return np.repeat(
            x,
            target_length
        ).astype(np.float32)

    old_time = (
        np.arange(len(x)) /
        original_fs
    )

    new_time = (
        np.arange(target_length) /
        target_fs
    )

    new_time = np.minimum(
        new_time,
        old_time[-1]
    )

    return np.interp(
        new_time,
        old_time,
        x
    ).astype(np.float32)


# PER-WINDOW ROBUST NORMALIZATION

def normalize_channel(x):

    x = np.asarray(
        x,
        dtype=np.float32
    )

    median = np.median(x)

    q25 = np.percentile(
        x,
        25
    )

    q75 = np.percentile(
        x,
        75
    )

    iqr = q75 - q25

    if iqr < 1e-6:

        std = np.std(x)

        if std < 1e-6:

            return np.zeros_like(
                x,
                dtype=np.float32
            )

        return (
            (x - np.mean(x))
            /
            (std + 1e-6)
        ).astype(np.float32)

    return (
        (x - median)
        /
        (iqr + 1e-6)
    ).astype(np.float32)


# FIND SUBJECT FILE

def find_subject_file(subject):

    possible_paths = [

        os.path.join(
            DATASET_PATH,
            subject,
            f"{subject}.pkl"
        ),

        os.path.join(
            DATASET_PATH,
            f"{subject}.pkl"
        ),

        os.path.join(
            DATASET_PATH,
            subject,
            f"{subject}.pickle"
        )
    ]

    for path in possible_paths:

        if os.path.isfile(path):

            return path

    return None

# LOAD WESAD SUBJECT

def load_subject(subject):

    file_path = find_subject_file(
        subject
    )

    if file_path is None:

        print(
            f"[WARNING] {subject}: file not found."
        )

        return None

    print(
        f"\nLoading {subject}"
    )

    with open(
        file_path,
        "rb"
    ) as f:

        data = pickle.load(
            f,
            encoding="latin1"
        )

    wrist = data["signal"]["wrist"]

    bvp = clean_signal(
        wrist["BVP"]
    )

    eda = clean_signal(
        wrist["EDA"]
    )

    temp = clean_signal(
        wrist["TEMP"]
    )

    acc = np.asarray(
        wrist["ACC"],
        dtype=np.float32
    )

    labels = np.asarray(
        data["label"]
    ).flatten()

    if acc.ndim == 1:

        if len(acc) % 3 != 0:

            raise ValueError(
                f"{subject}: ACC cannot be reshaped."
            )

        acc = acc.reshape(-1, 3)

    elif acc.ndim == 2:

        if acc.shape[1] == 3:

            pass

        elif acc.shape[0] == 3:

            acc = acc.T

        else:

            raise ValueError(
                f"{subject}: unexpected ACC shape "
                f"{acc.shape}"
            )

    else:

        raise ValueError(
            f"{subject}: unexpected ACC dimensions."
        )

    return {
        "BVP": bvp,
        "EDA": eda,
        "TEMP": temp,
        "ACC": acc,
        "label": labels
    }


# WINDOW LABEL

def get_window_label(
    labels,
    start_time,
    end_time
):

    start_sample = int(
        start_time * FS_LABEL
    )

    end_sample = int(
        end_time * FS_LABEL
    )

    start_sample = max(
        0,
        start_sample
    )

    end_sample = min(
        len(labels),
        end_sample
    )

    if end_sample <= start_sample:

        return None

    window_labels = labels[
        start_sample:end_sample
    ]

    stress_count = np.sum(
        window_labels == 2
    )

    nonstress_count = np.sum(
        np.isin(
            window_labels,
            [1, 3]
        )
    )

    total_relevant = (
        stress_count +
        nonstress_count
    )

    if total_relevant == 0:

        return None

    dominance = (
        max(
            stress_count,
            nonstress_count
        )
        /
        total_relevant
    )

    if dominance < 0.80:

        return None

    if stress_count > nonstress_count:

        return 1

    return 0


# EXTRACT MULTI-RATE WINDOW

def extract_wrist_window(
    data,
    start_time,
    end_time
):


    # BVP


    bvp_start = int(
        start_time * FS_BVP
    )

    bvp_end = int(
        end_time * FS_BVP
    )

    bvp = data["BVP"][
        bvp_start:bvp_end
    ]

    bvp = butter_bandpass(
        bvp,
        FS_BVP,
        0.5,
        8.0
    )

    bvp = resample_signal(
        bvp,
        FS_BVP,
        TARGET_FS_BVP
    )


    # EDA


    eda_start = int(
        start_time * FS_EDA
    )

    eda_end = int(
        end_time * FS_EDA
    )

    eda = data["EDA"][
        eda_start:eda_end
    ]

    eda = butter_lowpass(
        eda,
        FS_EDA,
        1.0
    )

    eda = resample_signal(
        eda,
        FS_EDA,
        TARGET_FS_EDA
    )


    # TEMP


    temp_start = int(
        start_time * FS_TEMP
    )

    temp_end = int(
        end_time * FS_TEMP
    )

    temp = data["TEMP"][
        temp_start:temp_end
    ]

    temp = butter_lowpass(
        temp,
        FS_TEMP,
        1.0
    )

    temp = resample_signal(
        temp,
        FS_TEMP,
        TARGET_FS_TEMP
    )


    # ACC


    acc_start = int(
        start_time * FS_ACC
    )

    acc_end = int(
        end_time * FS_ACC
    )

    acc = data["ACC"][
        acc_start:acc_end
    ]

    if len(acc) == 0:

        return None

    acc_x = clean_signal(
        acc[:, 0]
    )

    acc_y = clean_signal(
        acc[:, 1]
    )

    acc_z = clean_signal(
        acc[:, 2]
    )

    acc_x = butter_lowpass(
        acc_x,
        FS_ACC,
        6.0
    )

    acc_y = butter_lowpass(
        acc_y,
        FS_ACC,
        6.0
    )

    acc_z = butter_lowpass(
        acc_z,
        FS_ACC,
        6.0
    )

    acc_x = resample_signal(
        acc_x,
        FS_ACC,
        TARGET_FS_ACC
    )

    acc_y = resample_signal(
        acc_y,
        FS_ACC,
        TARGET_FS_ACC
    )

    acc_z = resample_signal(
        acc_z,
        FS_ACC,
        TARGET_FS_ACC
    )


    # EXPECTED LENGTHS


    expected_bvp = (
        WINDOW_SECONDS *
        TARGET_FS_BVP
    )

    expected_eda = (
        WINDOW_SECONDS *
        TARGET_FS_EDA
    )

    expected_temp = (
        WINDOW_SECONDS *
        TARGET_FS_TEMP
    )

    expected_acc = (
        WINDOW_SECONDS *
        TARGET_FS_ACC
    )

    signals = [
        bvp,
        eda,
        temp,
        acc_x,
        acc_y,
        acc_z
    ]

    expected_lengths = [
        expected_bvp,
        expected_eda,
        expected_temp,
        expected_acc,
        expected_acc,
        expected_acc
    ]


    # QUALITY CHECK

    for x, expected in zip(
        signals,
        expected_lengths
    ):

        if len(x) < int(
            expected * 0.90
        ):

            return None

    # EXACT LENGTH

    final_signals = []

    for x, expected in zip(
        signals,
        expected_lengths
    ):

        if len(x) >= expected:

            x = x[:expected]

        else:

            if len(x) == 0:

                return None

            x = np.pad(
                x,
                (
                    0,
                    expected - len(x)
                ),
                mode="edge"
            )

        final_signals.append(x)

    bvp = final_signals[0]
    eda = final_signals[1]
    temp = final_signals[2]
    acc_x = final_signals[3]
    acc_y = final_signals[4]
    acc_z = final_signals[5]

    # NORMALIZATION

    bvp = normalize_channel(bvp)
    eda = normalize_channel(eda)
    temp = normalize_channel(temp)
    acc_x = normalize_channel(acc_x)
    acc_y = normalize_channel(acc_y)
    acc_z = normalize_channel(acc_z)

    # RETURN

    return {

        "BVP": bvp[:, None].astype(
            np.float32
        ),

        "EDA": eda[:, None].astype(
            np.float32
        ),

        "TEMP": temp[:, None].astype(
            np.float32
        ),

        "ACC": np.stack(
            [
                acc_x,
                acc_y,
                acc_z
            ],
            axis=-1
        ).astype(
            np.float32
        )
    }


# BUILD DATASET

def build_dataset():

    bvp_list = []
    eda_list = []
    temp_list = []
    acc_list = []
    y_list = []
    groups_list = []

    print("\n")
    print("-" * 50)
    print(
        "BUILDING MULTI-RATE WESAD WRIST DATASET"
    )
    print("-" * 50)

    for subject in SUBJECTS:

        data = load_subject(
            subject
        )

        if data is None:
            continue

        duration = (
            len(data["BVP"]) /
            FS_BVP
        )

        print(
            f"{subject} duration: "
            f"{duration:.2f} sec"
        )

        subject_count = 0
        subject_stress = 0
        subject_nonstress = 0

        start_time = 0.0

        while (
            start_time +
            WINDOW_SECONDS
            <= duration
        ):

            end_time = (
                start_time +
                WINDOW_SECONDS
            )

            label = get_window_label(
                data["label"],
                start_time,
                end_time
            )

            if label is None:

                start_time += STEP_SECONDS
                continue

            window = extract_wrist_window(
                data,
                start_time,
                end_time
            )

            if window is None:

                start_time += STEP_SECONDS
                continue

            bvp_list.append(
                window["BVP"]
            )

            eda_list.append(
                window["EDA"]
            )

            temp_list.append(
                window["TEMP"]
            )

            acc_list.append(
                window["ACC"]
            )

            y_list.append(
                label
            )

            groups_list.append(
                subject
            )

            subject_count += 1

            if label == 1:

                subject_stress += 1

            else:

                subject_nonstress += 1

            start_time += STEP_SECONDS

        print(
            f"{subject}: "
            f"{subject_count} valid windows | "
            f"non-stress={subject_nonstress} | "
            f"stress={subject_stress}"
        )


    # ARRAYS

    if len(y_list) == 0:

        raise RuntimeError(
            "No valid WESAD windows were created. "
            "Check DATASET_PATH and SUBJECTS."
        )

    BVP = np.asarray(
        bvp_list,
        dtype=np.float32
    )

    EDA = np.asarray(
        eda_list,
        dtype=np.float32
    )

    TEMP = np.asarray(
        temp_list,
        dtype=np.float32
    )

    ACC = np.asarray(
        acc_list,
        dtype=np.float32
    )

    y = np.asarray(
        y_list,
        dtype=np.int32
    )

    groups = np.asarray(
        groups_list
    )

    print("\n")
    print("-" * 50)
    print("FINAL MULTI-RATE DATASET")
    print("-" * 50)

    print("BVP shape:", BVP.shape)
    print("EDA shape:", EDA.shape)
    print("TEMP shape:", TEMP.shape)
    print("ACC shape:", ACC.shape)
    print("y shape:", y.shape)
    print(
        "Subjects:",
        np.unique(groups)
    )

    for cls in [0, 1]:

        count = np.sum(
            y == cls
        )

        name = (
            "Non-Stress"
            if cls == 0
            else
            "Stress"
        )

        print(
            f"{name}: {count}"
        )

    return (
        BVP,
        EDA,
        TEMP,
        ACC,
        y,
        groups
    )


# ATTENTION LAYER

@tf.keras.utils.register_keras_serializable(
    package="WESAD"
)
class AttentionLayer(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )

    def build(
        self,
        input_shape
    ):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(
                input_shape[-1],
                1
            ),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )

        super().build(
            input_shape
        )

    def call(
        self,
        inputs
    ):

        score = tf.tanh(
            tf.matmul(
                inputs,
                self.W
            ) + self.b
        )

        weights = tf.nn.softmax(
            score,
            axis=1
        )

        context = (
            inputs *
            weights
        )

        context = tf.reduce_sum(
            context,
            axis=1
        )

        return context


# BUILD MODEL

def build_model():

    bvp_input = tf.keras.Input(
        shape=(
            WINDOW_SECONDS *
            TARGET_FS_BVP,
            1
        ),
        name="BVP"
    )

    eda_input = tf.keras.Input(
        shape=(
            WINDOW_SECONDS *
            TARGET_FS_EDA,
            1
        ),
        name="EDA"
    )

    temp_input = tf.keras.Input(
        shape=(
            WINDOW_SECONDS *
            TARGET_FS_TEMP,
            1
        ),
        name="TEMP"
    )

    acc_input = tf.keras.Input(
        shape=(
            WINDOW_SECONDS *
            TARGET_FS_ACC,
            3
        ),
        name="ACC"
    )

    # BVP

    bvp = tf.keras.layers.Conv1D(
        32,
        7,
        padding="same",
        activation="relu"
    )(bvp_input)

    bvp = tf.keras.layers.LayerNormalization()(bvp)
    bvp = tf.keras.layers.MaxPooling1D(2)(bvp)
    bvp = tf.keras.layers.Dropout(0.20)(bvp)

    bvp = tf.keras.layers.Conv1D(
        64,
        5,
        padding="same",
        activation="relu"
    )(bvp)

    bvp = tf.keras.layers.LayerNormalization()(bvp)
    bvp = tf.keras.layers.MaxPooling1D(2)(bvp)
    bvp = tf.keras.layers.Dropout(0.20)(bvp)

    bvp = tf.keras.layers.Conv1D(
        96,
        3,
        padding="same",
        activation="relu"
    )(bvp)

    bvp = tf.keras.layers.LayerNormalization()(bvp)

    bvp = tf.keras.layers.GlobalAveragePooling1D()(bvp)

    # EDA

    eda = tf.keras.layers.Conv1D(
        32,
        5,
        padding="same",
        activation="relu"
    )(eda_input)

    eda = tf.keras.layers.LayerNormalization()(eda)
    eda = tf.keras.layers.MaxPooling1D(2)(eda)
    eda = tf.keras.layers.Dropout(0.20)(eda)

    eda = tf.keras.layers.Conv1D(
        64,
        5,
        padding="same",
        activation="relu"
    )(eda)

    eda = tf.keras.layers.LayerNormalization()(eda)
    eda = tf.keras.layers.MaxPooling1D(2)(eda)

    eda = tf.keras.layers.Conv1D(
        96,
        3,
        padding="same",
        activation="relu"
    )(eda)

    eda = tf.keras.layers.LayerNormalization()(eda)

    eda = tf.keras.layers.GlobalAveragePooling1D()(eda)

    # TEMP

    temp = tf.keras.layers.Conv1D(
        16,
        5,
        padding="same",
        activation="relu"
    )(temp_input)

    temp = tf.keras.layers.LayerNormalization()(temp)
    temp = tf.keras.layers.MaxPooling1D(2)(temp)

    temp = tf.keras.layers.Conv1D(
        32,
        5,
        padding="same",
        activation="relu"
    )(temp)

    temp = tf.keras.layers.LayerNormalization()(temp)

    temp = tf.keras.layers.GlobalAveragePooling1D()(temp)

    # ACC

    acc = tf.keras.layers.Conv1D(
        32,
        7,
        padding="same",
        activation="relu"
    )(acc_input)

    acc = tf.keras.layers.LayerNormalization()(acc)
    acc = tf.keras.layers.MaxPooling1D(2)(acc)
    acc = tf.keras.layers.Dropout(0.20)(acc)

    acc = tf.keras.layers.Conv1D(
        64,
        5,
        padding="same",
        activation="relu"
    )(acc)

    acc = tf.keras.layers.LayerNormalization()(acc)
    acc = tf.keras.layers.MaxPooling1D(2)(acc)

    acc = tf.keras.layers.Conv1D(
        96,
        3,
        padding="same",
        activation="relu"
    )(acc)

    acc = tf.keras.layers.LayerNormalization()(acc)

    acc = tf.keras.layers.GlobalAveragePooling1D()(acc)

    # FUSION

    x = tf.keras.layers.Concatenate(
        name="multi_rate_fusion"
    )([
        bvp,
        eda,
        temp,
        acc
    ])

    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(x)

    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dropout(
        0.35
    )(x)

    # TEMPORAL REPRESENTATION

    x = tf.keras.layers.RepeatVector(8)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.20
        )
    )(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            32,
            return_sequences=True,
            dropout=0.20
        )
    )(x)

    # ATTENTION

    x = AttentionLayer(
        name="attention"
    )(x)

    # CLASSIFIER

    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Dropout(
        0.35
    )(x)

    x = tf.keras.layers.Dense(
        64,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.25
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="stress_probability"
    )(x)

    model = tf.keras.Model(
        inputs=[
            bvp_input,
            eda_input,
            temp_input,
            acc_input
        ],
        outputs=outputs
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(
                name="accuracy"
            ),
            tf.keras.metrics.AUC(
                name="auc"
            ),
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            )
        ]
    )

    return model


# CLASS WEIGHTS

def get_class_weights(y):

    classes, counts = np.unique(
        y,
        return_counts=True
    )

    total = len(y)
    n_classes = len(classes)

    weights = {}

    for cls, count in zip(
        classes,
        counts
    ):

        weights[int(cls)] = (
            total /
            (
                n_classes *
                count
            )
        )

    weights.setdefault(0, 1.0)
    weights.setdefault(1, 1.0)

    return weights


# METRICS

def calculate_metrics(
    y_true,
    y_prob,
    threshold=0.50
):

    y_pred = (
        y_prob >= threshold
    ).astype(np.int32)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    ppv = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0.0
    )

    metrics = {

        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred
            ),

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "PPV":
            ppv,

        "NPV":
            npv,

        "F1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
    }

    try:

        metrics["ROC_AUC"] = (
            roc_auc_score(
                y_true,
                y_prob
            )
        )

    except Exception:

        metrics["ROC_AUC"] = np.nan

    try:

        metrics["PR_AUC"] = (
            average_precision_score(
                y_true,
                y_prob
            )
        )

    except Exception:

        metrics["PR_AUC"] = np.nan

    return (
        metrics,
        y_pred,
        cm
    )


# THRESHOLD OPTIMIZATION

def find_best_threshold(
    y_true,
    y_prob
):

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )

    best_threshold = (
        DEFAULT_THRESHOLD
    )

    best_score = -np.inf

    for threshold in thresholds:

        y_pred = (
            y_prob >= threshold
        ).astype(np.int32)

        score = (
            balanced_accuracy_score(
                y_true,
                y_pred
            )
        )

        if score > best_score:

            best_score = score
            best_threshold = threshold

    return (
        float(best_threshold),
        float(best_score)
    )


# FEDAVG

def fedavg(
    client_weights,
    client_sample_counts
):

    if len(client_weights) == 0:

        raise RuntimeError(
            "FedAvg received zero client updates."
        )

    if len(client_weights) != len(
        client_sample_counts
    ):

        raise RuntimeError(
            "Client weights and sample counts "
            "do not match."
        )

    total_samples = np.sum(
        client_sample_counts
    )

    if total_samples <= 0:

        raise RuntimeError(
            "Total federated training samples is zero."
        )

    averaged_weights = []

    for weight_index in range(
        len(client_weights[0])
    ):

        weighted_sum = None

        for client_index in range(
            len(client_weights)
        ):

            weight = client_weights[
                client_index
            ][weight_index]

            sample_count = (
                client_sample_counts[
                    client_index
                ]
            )

            contribution = (
                weight.astype(np.float32)
                *
                sample_count
            )

            if weighted_sum is None:

                weighted_sum = contribution

            else:

                weighted_sum += contribution

        averaged_weight = (
            weighted_sum /
            total_samples
        )

        averaged_weights.append(
            averaged_weight.astype(
                np.float32
            )
        )

    return averaged_weights


# CLIENT UPDATE DEFENSE

def defend_client_update(
    client_weights,
    global_weights,
    mode="none",
    clip_norm=1.0,
    noise_multiplier=0.0,
    rng=None
):

    if rng is None:

        rng = np.random.default_rng(
            RANDOM_STATE
        )

    deltas = []

    for client_w, global_w in zip(
        client_weights,
        global_weights
    ):

        delta = (
            client_w.astype(np.float32)
            -
            global_w.astype(np.float32)
        )

        deltas.append(delta)

    # ORIGINAL NORM

    squared_norm = 0.0

    for delta in deltas:

        squared_norm += np.sum(
            np.square(
                delta.astype(np.float64)
            )
        )

    original_norm = float(
        np.sqrt(squared_norm)
    )

    # CLIPPING

    if mode in [
        "clip",
        "clip_dp"
    ]:

        if original_norm > clip_norm:

            scaling_factor = (
                clip_norm /
                (original_norm + 1e-12)
            )

        else:

            scaling_factor = 1.0

        deltas = [
            delta * scaling_factor
            for delta in deltas
        ]

    else:

        scaling_factor = 1.0

    # GAUSSIAN NOISE

    if mode == "clip_dp":

        noise_std = (
            noise_multiplier *
            clip_norm
        )

        for i in range(
            len(deltas)
        ):

            noise = rng.normal(
                loc=0.0,
                scale=noise_std,
                size=deltas[i].shape
            ).astype(np.float32)

            deltas[i] += noise

    elif mode not in [
        "none",
        "clip"
    ]:

        raise ValueError(
            f"Unknown privacy mode: {mode}"
        )

    # CONVERT TO WEIGHTS

    defended_weights = []

    for global_w, delta in zip(
        global_weights,
        deltas
    ):

        defended_weights.append(
            (
                global_w.astype(
                    np.float32
                )
                +
                delta
            ).astype(
                np.float32
            )
        )

    # DEFENDED NORM

    defended_squared_norm = 0.0

    for defended_w, global_w in zip(
        defended_weights,
        global_weights
    ):

        delta = (
            defended_w.astype(
                np.float64
            )
            -
            global_w.astype(
                np.float64
            )
        )

        defended_squared_norm += np.sum(
            delta ** 2
        )

    defended_norm = float(
        np.sqrt(
            defended_squared_norm
        )
    )

    return (
        defended_weights,
        original_norm,
        defended_norm,
        scaling_factor
    )


# PRIVACY UPDATE FEATURES

def extract_privacy_update_features(
    client_weights,
    global_weights,
    client_id,
    round_number
):

    records = {}

    records["Client"] = client_id
    records["Round"] = round_number

    total_squared = 0.0
    all_delta_values = []

    for client_w, global_w in zip(
        client_weights,
        global_weights
    ):

        delta = (
            client_w.astype(np.float64)
            -
            global_w.astype(np.float64)
        )

        total_squared += np.sum(
            delta ** 2
        )

        all_delta_values.append(
            delta.ravel()
        )

    all_delta_values = np.concatenate(
        all_delta_values
    )

    records["total_update_l2"] = float(
        np.sqrt(total_squared)
    )

    records["total_update_mean"] = float(
        np.mean(all_delta_values)
    )

    records["total_update_std"] = float(
        np.std(all_delta_values)
    )

    records["total_update_abs_mean"] = float(
        np.mean(
            np.abs(
                all_delta_values
            )
        )
    )

    # Final Dense layer
    # Keras stores kernel and bias as the final two arrays.

    final_kernel_delta = (
        client_weights[-2].astype(np.float64)
        -
        global_weights[-2].astype(np.float64)
    )

    final_bias_delta = (
        client_weights[-1].astype(np.float64)
        -
        global_weights[-1].astype(np.float64)
    )

    records["final_kernel_l2"] = float(
        np.linalg.norm(
            final_kernel_delta
        )
    )

    records["final_kernel_mean"] = float(
        np.mean(
            final_kernel_delta
        )
    )

    records["final_kernel_std"] = float(
        np.std(
            final_kernel_delta
        )
    )

    records["final_kernel_abs_mean"] = float(
        np.mean(
            np.abs(
                final_kernel_delta
            )
        )
    )

    records["final_bias_delta"] = float(
        np.linalg.norm(
            final_bias_delta
        )
    )

    return records


# PREDICTION

def predict_model(
    model,
    BVP,
    EDA,
    TEMP,
    ACC
):

    probabilities = model.predict(
        {
            "BVP": BVP,
            "EDA": EDA,
            "TEMP": TEMP,
            "ACC": ACC
        },
        batch_size=BATCH_SIZE,
        verbose=0
    ).ravel()

    return probabilities


# EVALUATION

def evaluate_model(
    model,
    BVP,
    EDA,
    TEMP,
    ACC,
    y
):

    probabilities = predict_model(
        model,
        BVP,
        EDA,
        TEMP,
        ACC
    )

    loss = tf.keras.losses.binary_crossentropy(
        tf.convert_to_tensor(
            y.astype(np.float32)
        ),
        tf.convert_to_tensor(
            probabilities.astype(np.float32)
        )
    )

    loss = float(
        tf.reduce_mean(
            loss
        ).numpy()
    )

    return (
        loss,
        probabilities
    )


# CONFUSION MATRIX

def plot_confusion_matrix(
    cm,
    title,
    filename
):

    plt.figure(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=[
            "Non-Stress",
            "Stress"
        ],
        yticklabels=[
            "Non-Stress",
            "Stress"
        ],
        annot_kws={
            "fontsize": 15
        }
    )

    plt.xlabel(
        "Predicted Class"
    )

    plt.ylabel(
        "True Class"
    )

    plt.title(
        title,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# FOLD ROC

def plot_fold_roc(
    y_true,
    y_prob,
    fold
):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    filename = os.path.join(
        RESULTS_DIR,
        f"Federated_Fold_{fold}_ROC_Curve.png"
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        color="#1565C0",
        linewidth=2.5,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        label="Random classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"Federated Fold {fold} ROC Curve",
        fontweight="bold"
    )

    plt.grid(alpha=0.25)

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return (
        fpr,
        tpr,
        roc_auc
    )


#  OVERALL ROC

def plot_overall_roc(
    all_true,
    all_prob
):

    fpr, tpr, _ = roc_curve(
        all_true,
        all_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.plot(
        fpr,
        tpr,
        color="#1565C0",
        linewidth=3,
        label=(
            f"Overall Out-of-Fold ROC "
            f"(AUC = {roc_auc:.4f})"
        )
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        label="Random classifier"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Overall Federated WESAD ROC Curve",
        fontweight="bold"
    )

    plt.grid(alpha=0.25)

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "Federated_Overall_Out_of_Fold_ROC_Curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    pd.DataFrame({
        "False_Positive_Rate": fpr,
        "True_Positive_Rate": tpr
    }).to_csv(
        os.path.join(
            RESULTS_DIR,
            "Federated_Overall_ROC_Data.csv"
        ),
        index=False
    )

    return (
        fpr,
        tpr,
        roc_auc
    )



#  MEAN FOLD ROC


def plot_mean_fold_roc(
    fold_roc_data
):

    mean_fpr = np.linspace(
        0,
        1,
        101
    )

    interpolated_tprs = []
    fold_aucs = []

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd"
    ]

    plt.figure(
        figsize=(8, 7)
    )

    for i, data in enumerate(
        fold_roc_data
    ):

        fpr = data["fpr"]
        tpr = data["tpr"]
        fold_auc = data["auc"]

        fold_aucs.append(
            fold_auc
        )

        plt.plot(
            fpr,
            tpr,
            color=colors[
                i % len(colors)
            ],
            linewidth=1.5,
            alpha=0.60,
            label=(
                f"Fold {i + 1} "
                f"(AUC={fold_auc:.3f})"
            )
        )

        interp_tpr = np.interp(
            mean_fpr,
            fpr,
            tpr
        )

        interp_tpr[0] = 0.0

        interpolated_tprs.append(
            interp_tpr
        )

    mean_tpr = np.mean(
        interpolated_tprs,
        axis=0
    )

    mean_tpr[-1] = 1.0

    mean_auc = auc(
        mean_fpr,
        mean_tpr
    )

    std_tpr = np.std(
        interpolated_tprs,
        axis=0
    )

    tpr_upper = np.minimum(
        mean_tpr + std_tpr,
        1
    )

    tpr_lower = np.maximum(
        mean_tpr - std_tpr,
        0
    )

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="black",
        linewidth=3,
        label=(
            f"Mean ROC "
            f"(AUC={mean_auc:.4f})"
        )
    )

    plt.fill_between(
        mean_fpr,
        tpr_lower,
        tpr_upper,
        color="gray",
        alpha=0.20,
        label="±1 SD"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Five-Fold Federated ROC Curves",
        fontweight="bold"
    )

    plt.grid(alpha=0.25)

    plt.legend(
        loc="lower right",
        fontsize=9
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "Federated_Mean_FiveFold_ROC_Curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return (
        mean_auc,
        np.mean(fold_aucs),
        np.std(
            fold_aucs,
            ddof=1
        )
    )


#  PRECISION-RECALL

def plot_precision_recall(
    all_true,
    all_prob
):

    precision, recall, _ = (
        precision_recall_curve(
            all_true,
            all_prob
        )
    )

    pr_auc = average_precision_score(
        all_true,
        all_prob
    )

    prevalence = np.mean(
        all_true
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.plot(
        recall,
        precision,
        color="#C62828",
        linewidth=3,
        label=f"AP = {pr_auc:.4f}"
    )

    plt.axhline(
        prevalence,
        linestyle="--",
        color="gray",
        label=f"Random baseline ({prevalence:.3f})"
    )

    plt.xlabel(
        "Recall / Sensitivity"
    )

    plt.ylabel(
        "Precision / PPV"
    )

    plt.title(
        "Federated WESAD Precision-Recall Curve",
        fontweight="bold"
    )

    plt.grid(alpha=0.25)

    plt.legend(
        loc="lower left"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "Federated_Overall_Precision_Recall_Curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return pr_auc


# LEARNING CURVES

def plot_average_learning_curves(
    histories
):

    if len(histories) == 0:

        return pd.DataFrame()

    max_rounds = max(
        len(h["round"])
        for h in histories
    )

    n_folds = len(
        histories
    )

    train_accuracy = np.full(
        (n_folds, max_rounds),
        np.nan
    )

    val_accuracy = np.full(
        (n_folds, max_rounds),
        np.nan
    )

    train_loss = np.full(
        (n_folds, max_rounds),
        np.nan
    )

    val_loss = np.full(
        (n_folds, max_rounds),
        np.nan
    )

    for i, history in enumerate(
        histories
    ):

        n = len(
            history["round"]
        )

        train_accuracy[
            i,
            :n
        ] = history[
            "train_accuracy"
        ]

        val_accuracy[
            i,
            :n
        ] = history[
            "val_accuracy"
        ]

        train_loss[
            i,
            :n
        ] = history[
            "train_loss"
        ]

        val_loss[
            i,
            :n
        ] = history[
            "val_loss"
        ]

    mean_train_accuracy = np.nanmean(
        train_accuracy,
        axis=0
    )

    mean_val_accuracy = np.nanmean(
        val_accuracy,
        axis=0
    )

    mean_train_loss = np.nanmean(
        train_loss,
        axis=0
    )

    mean_val_loss = np.nanmean(
        val_loss,
        axis=0
    )

    std_train_accuracy = np.nanstd(
        train_accuracy,
        axis=0
    )

    std_val_accuracy = np.nanstd(
        val_accuracy,
        axis=0
    )

    std_train_loss = np.nanstd(
        train_loss,
        axis=0
    )

    std_val_loss = np.nanstd(
        val_loss,
        axis=0
    )

    rounds = np.arange(
        1,
        max_rounds + 1
    )

    # ACCURACY

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        rounds,
        mean_train_accuracy,
        color="#1565C0",
        linewidth=2.5,
        label="Federated Client Training Accuracy"
    )

    plt.plot(
        rounds,
        mean_val_accuracy,
        color="#C62828",
        linewidth=2.5,
        label="Global Validation Accuracy"
    )

    plt.fill_between(
        rounds,
        np.maximum(
            0,
            mean_train_accuracy -
            std_train_accuracy
        ),
        np.minimum(
            1,
            mean_train_accuracy +
            std_train_accuracy
        ),
        color="#1565C0",
        alpha=0.12
    )

    plt.fill_between(
        rounds,
        np.maximum(
            0,
            mean_val_accuracy -
            std_val_accuracy
        ),
        np.minimum(
            1,
            mean_val_accuracy +
            std_val_accuracy
        ),
        color="#C62828",
        alpha=0.12
    )

    plt.xlabel(
        "Communication Round"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "Federated Client Training vs Validation Accuracy",
        fontweight="bold"
    )

    plt.ylim(
        0,
        1
    )

    plt.grid(alpha=0.25)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "Average_Federated_Training_vs_Validation_Accuracy.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # LOSS

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        rounds,
        mean_train_loss,
        color="#1565C0",
        linewidth=2.5,
        label="Federated Client Training Loss"
    )

    plt.plot(
        rounds,
        mean_val_loss,
        color="#C62828",
        linewidth=2.5,
        label="Global Validation Loss"
    )

    plt.fill_between(
        rounds,
        np.maximum(
            0,
            mean_train_loss -
            std_train_loss
        ),
        mean_train_loss +
        std_train_loss,
        color="#1565C0",
        alpha=0.12
    )

    plt.fill_between(
        rounds,
        np.maximum(
            0,
            mean_val_loss -
            std_val_loss
        ),
        mean_val_loss +
        std_val_loss,
        color="#C62828",
        alpha=0.12
    )

    plt.xlabel(
        "Communication Round"
    )

    plt.ylabel(
        "Binary Cross-Entropy Loss"
    )

    plt.title(
        "Federated Training vs Validation Loss",
        fontweight="bold"
    )

    plt.grid(alpha=0.25)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "Average_Federated_Training_vs_Validation_Loss.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    curve_df = pd.DataFrame({

        "Communication_Round":
            rounds,

        "Training_Accuracy_Mean":
            mean_train_accuracy,

        "Training_Accuracy_SD":
            std_train_accuracy,

        "Validation_Accuracy_Mean":
            mean_val_accuracy,

        "Validation_Accuracy_SD":
            std_val_accuracy,

        "Training_Loss_Mean":
            mean_train_loss,

        "Training_Loss_SD":
            std_train_loss,

        "Validation_Loss_Mean":
            mean_val_loss,

        "Validation_Loss_SD":
            std_val_loss
    })

    curve_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "Average_Federated_Learning_Curves.csv"
        ),
        index=False
    )

    return curve_df


# FEDERATED 5-FOLD EXPERIMENT

def run_federated_5fold(
    BVP,
    EDA,
    TEMP,
    ACC,
    y,
    groups
):

    print("\n")
    print("=" * 110)
    print(
        "FEDERATED WESAD STRESS DETECTION"
    )
    print(
        "5-FOLD SUBJECT-INDEPENDENT CROSS-VALIDATION"
    )
    print("=" * 110)

    print(
        f"Communication rounds: {NUM_ROUNDS}"
    )

    print(
        f"Local epochs: {LOCAL_EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        "FedAvg: sample-weighted"
    )

    print(
        f"Privacy defense: {PRIVACY_MODE}"
    )

    print(
        f"Clip norm: {CLIP_NORM}"
    )

    print(
        f"Noise multiplier: {NOISE_MULTIPLIER}"
    )

    outer_cv = StratifiedGroupKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_results = []
    histories = []
    fold_roc_data = []
    fold_subject_records = []

    all_true = []
    all_pred = []
    all_prob = []

    # FOLD LOOP

    for fold, (
        development_idx,
        test_idx
    ) in enumerate(
        outer_cv.split(
            BVP,
            y,
            groups
        ),
        start=1
    ):

        print("\n")
        print("=" * 110)
        print(
            f"FOLD {fold} / {N_FOLDS}"
        )
        print("=" * 110)

        # DEVELOPMENT / TEST

        y_dev = y[
            development_idx
        ]

        groups_dev = groups[
            development_idx
        ]

        y_test = y[
            test_idx
        ]

        groups_test = groups[
            test_idx
        ]

        test_subjects = np.unique(
            groups_test
        )

        # INNER TRAIN / VALIDATION

        inner_cv = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=100 + fold
        )

        inner_train_pos, val_pos = next(
            inner_cv.split(
                development_idx,
                y_dev,
                groups_dev
            )
        )

        train_indices = (
            development_idx[
                inner_train_pos
            ]
        )

        val_indices = (
            development_idx[
                val_pos
            ]
        )

        y_train = y[
            train_indices
        ]

        y_val = y[
            val_indices
        ]

        groups_train = groups[
            train_indices
        ]

        groups_val = groups[
            val_indices
        ]

        train_subjects = np.unique(
            groups_train
        )

        val_subjects = np.unique(
            groups_val
        )

        # SUBJECT INDEPENDENCE

        assert not (
            set(train_subjects)
            &
            set(val_subjects)
        )

        assert not (
            set(train_subjects)
            &
            set(test_subjects)
        )

        assert not (
            set(val_subjects)
            &
            set(test_subjects)
        )

        print(
            "\nFederated training clients:"
        )

        print(
            list(train_subjects)
        )

        print(
            "\nValidation subjects:"
        )

        print(
            list(val_subjects)
        )

        print(
            "\nTest subjects:"
        )

        print(
            list(test_subjects)
        )

        # SAVE SUBJECT ASSIGNMENTS

        for subject in train_subjects:

            fold_subject_records.append({
                "Fold":
                    fold,
                "Role":
                    "Federated_Training_Client",
                "Subject":
                    subject
            })

        for subject in val_subjects:

            fold_subject_records.append({
                "Fold":
                    fold,
                "Role":
                    "Validation",
                "Subject":
                    subject
            })

        for subject in test_subjects:

            fold_subject_records.append({
                "Fold":
                    fold,
                "Role":
                    "Test",
                "Subject":
                    subject
            })

        # CLIENT DATA

        client_data = {}

        for subject in train_subjects:

            mask = (
                groups[
                    train_indices
                ]
                ==
                subject
            )

            client_indices = (
                train_indices[
                    mask
                ]
            )

            client_y = y[
                client_indices
            ]

            client_data[
                subject
            ] = {

                "BVP":
                    BVP[
                        client_indices
                    ],

                "EDA":
                    EDA[
                        client_indices
                    ],

                "TEMP":
                    TEMP[
                        client_indices
                    ],

                "ACC":
                    ACC[
                        client_indices
                    ],

                "y":
                    client_y
            }

            print(
                f"Client {subject}: "
                f"{len(client_y)} windows | "
                f"class0={np.sum(client_y == 0)} | "
                f"class1={np.sum(client_y == 1)}"
            )

        # VALIDATION DATA

        val_BVP = BVP[
            val_indices
        ]

        val_EDA = EDA[
            val_indices
        ]

        val_TEMP = TEMP[
            val_indices
        ]

        val_ACC = ACC[
            val_indices
        ]

        # TEST DATA

        test_BVP = BVP[
            test_idx
        ]

        test_EDA = EDA[
            test_idx
        ]

        test_TEMP = TEMP[
            test_idx
        ]

        test_ACC = ACC[
            test_idx
        ]

        # GLOBAL MODEL

        global_model = build_model()

        if fold == 1:

            print("\n")
            print("=" * 90)
            print(
                "GLOBAL MODEL ARCHITECTURE"
            )
            print("=" * 90)

            global_model.summary()

        global_weights = [
            np.copy(w)
            for w in global_model.get_weights()
        ]

        if len(global_weights) == 0:

            raise RuntimeError(
                "Global model returned zero weights."
            )

        # PRIVACY AUDIT

        privacy_update_records = []

        # HISTORY

        fold_history = {

            "round": [],

            "train_loss": [],

            "train_accuracy": [],

            "val_loss": [],

            "val_accuracy": [],

            "val_balanced_accuracy": [],

            "val_sensitivity": [],

            "val_specificity": [],

            "val_f1": [],

            "val_auc": [],

            "threshold": []
        }

        # BEST MODEL

        best_val_balanced_accuracy = -np.inf

        best_round = 0

        best_threshold = (
            DEFAULT_THRESHOLD
        )

        best_global_weights = None

        rounds_without_improvement = 0

        # COMMUNICATION ROUNDS

        for round_number in range(
            1,
            NUM_ROUNDS + 1
        ):

            print("\n")
            print("-" * 110)
            print(
                f"COMMUNICATION ROUND "
                f"{round_number}/{NUM_ROUNDS}"
            )
            print("-" * 110)

            client_weights = []
            client_sample_counts = []
            client_losses = []
            client_accuracies = []

            # CLIENT TRAINING

            for client_number, subject in enumerate(
                train_subjects,
                start=1
            ):

                print("\n")
                print(
                    f"CLIENT {client_number}/"
                    f"{len(train_subjects)} "
                    f"- SUBJECT {subject}"
                )

                client_BVP = (
                    client_data[
                        subject
                    ]["BVP"]
                )

                client_EDA = (
                    client_data[
                        subject
                    ]["EDA"]
                )

                client_TEMP = (
                    client_data[
                        subject
                    ]["TEMP"]
                )

                client_ACC = (
                    client_data[
                        subject
                    ]["ACC"]
                )

                client_y = (
                    client_data[
                        subject
                    ]["y"]
                )

                client_sample_count = (
                    len(client_y)
                )

                if client_sample_count == 0:

                    continue

                # LOCAL MODEL

                local_model = build_model()

                local_model.set_weights(
                    global_weights
                )

                # LOCAL OPTIMIZER

                local_model.compile(

                    optimizer=
                    tf.keras.optimizers.Adam(
                        learning_rate=
                        LEARNING_RATE,
                        clipnorm=1.0
                    ),

                    loss=
                    "binary_crossentropy",

                    metrics=[
                        "accuracy"
                    ]
                )

                # LOCAL CLASS WEIGHTS

                class_weights = (
                    get_class_weights(
                        client_y
                    )
                )

                # LOCAL TRAINING

                history = local_model.fit(

                    {
                        "BVP":
                            client_BVP,

                        "EDA":
                            client_EDA,

                        "TEMP":
                            client_TEMP,

                        "ACC":
                            client_ACC
                    },

                    client_y,

                    epochs=
                    LOCAL_EPOCHS,

                    batch_size=
                    BATCH_SIZE,

                    class_weight=
                    class_weights,

                    shuffle=True,

                    verbose=0
                )

                client_loss = float(
                    history.history[
                        "loss"
                    ][-1]
                )

                client_accuracy = float(
                    history.history[
                        "accuracy"
                    ][-1]
                )

                print(
                    f"Samples: "
                    f"{client_sample_count}"
                )

                print(
                    f"Local loss: "
                    f"{client_loss:.4f}"
                )

                print(
                    f"Local accuracy: "
                    f"{client_accuracy:.4f}"
                )

                # CLIENT WEIGHTS

                trained_client_weights = (
                    local_model.get_weights()
                )

                # DEFENSE

                (
                    defended_client_weights,
                    original_norm,
                    defended_norm,
                    scaling_factor
                ) = defend_client_update(

                    trained_client_weights,

                    global_weights,

                    mode=
                    PRIVACY_MODE,

                    clip_norm=
                    CLIP_NORM,

                    noise_multiplier=
                    NOISE_MULTIPLIER,

                    rng=
                    privacy_rng
                )

                print(
                    f"Original update norm: "
                    f"{original_norm:.6f}"
                )

                print(
                    f"Defended update norm: "
                    f"{defended_norm:.6f}"
                )

                print(
                    f"Clip scaling factor: "
                    f"{scaling_factor:.6f}"
                )

                client_weights.append(
                    defended_client_weights
                )

                client_sample_counts.append(
                    client_sample_count
                )

                client_losses.append(
                    client_loss
                )

                client_accuracies.append(
                    client_accuracy
                )

                privacy_update_records.append({

                    "Round":
                        round_number,

                    "Subject":
                        subject,

                    "Samples":
                        client_sample_count,

                    "Original_Update_Norm":
                        original_norm,

                    "Defended_Update_Norm":
                        defended_norm,

                    "Clip_Scaling_Factor":
                        scaling_factor,

                    "Defense_Mode":
                        PRIVACY_MODE,

                    "Clip_Norm":
                        CLIP_NORM,

                    "Noise_Multiplier":
                        NOISE_MULTIPLIER
                })

                del local_model

                tf.keras.backend.clear_session()

                gc.collect()

            # CHECK CLIENTS

            if len(client_weights) == 0:

                raise RuntimeError(
                    "No federated clients produced an update."
                )

            # FEDAVG

            global_weights = fedavg(
                client_weights,
                client_sample_counts
            )

            global_model.set_weights(
                global_weights
            )

            # TRAINING STATISTICS

            total_samples = np.sum(
                client_sample_counts
            )

            weighted_train_loss = (
                np.sum(
                    np.asarray(
                        client_losses
                    )
                    *
                    np.asarray(
                        client_sample_counts
                    )
                )
                /
                total_samples
            )

            weighted_train_accuracy = (
                np.sum(
                    np.asarray(
                        client_accuracies
                    )
                    *
                    np.asarray(
                        client_sample_counts
                    )
                )
                /
                total_samples
            )

            # VALIDATION

            (
                val_loss,
                val_prob
            ) = evaluate_model(

                global_model,

                val_BVP,
                val_EDA,
                val_TEMP,
                val_ACC,

                y_val
            )

            (
                threshold,
                threshold_score
            ) = find_best_threshold(
                y_val,
                val_prob
            )

            (
                val_metrics,
                val_pred,
                val_cm
            ) = calculate_metrics(
                y_val,
                val_prob,
                threshold
            )

            # HISTORY

            fold_history[
                "round"
            ].append(
                round_number
            )

            fold_history[
                "train_loss"
            ].append(
                weighted_train_loss
            )

            fold_history[
                "train_accuracy"
            ].append(
                weighted_train_accuracy
            )

            fold_history[
                "val_loss"
            ].append(
                val_loss
            )

            fold_history[
                "val_accuracy"
            ].append(
                val_metrics[
                    "accuracy"
                ]
            )

            fold_history[
                "val_balanced_accuracy"
            ].append(
                val_metrics[
                    "balanced_accuracy"
                ]
            )

            fold_history[
                "val_sensitivity"
            ].append(
                val_metrics[
                    "sensitivity"
                ]
            )

            fold_history[
                "val_specificity"
            ].append(
                val_metrics[
                    "specificity"
                ]
            )

            fold_history[
                "val_f1"
            ].append(
                val_metrics[
                    "F1"
                ]
            )

            fold_history[
                "val_auc"
            ].append(
                val_metrics[
                    "ROC_AUC"
                ]
            )

            fold_history[
                "threshold"
            ].append(
                threshold
            )

            # ROUND RESULTS

            print("\nROUND RESULTS")

            print(
                f"Train loss: "
                f"{weighted_train_loss:.4f}"
            )

            print(
                f"Train accuracy: "
                f"{weighted_train_accuracy:.4f}"
            )

            print(
                f"Validation loss: "
                f"{val_loss:.4f}"
            )

            print(
                f"Validation accuracy: "
                f"{val_metrics['accuracy']:.4f}"
            )

            print(
                f"Validation balanced accuracy: "
                f"{val_metrics['balanced_accuracy']:.4f}"
            )

            print(
                f"Validation sensitivity: "
                f"{val_metrics['sensitivity']:.4f}"
            )

            print(
                f"Validation specificity: "
                f"{val_metrics['specificity']:.4f}"
            )

            print(
                f"Validation F1: "
                f"{val_metrics['F1']:.4f}"
            )

            print(
                f"Validation ROC-AUC: "
                f"{val_metrics['ROC_AUC']:.4f}"
            )

            print(
                f"Selected threshold: "
                f"{threshold:.2f}"
            )

            # MODEL SELECTION

            improvement = (
                val_metrics[
                    "balanced_accuracy"
                ]
                >
                best_val_balanced_accuracy
                +
                1e-4
            )

            if improvement:

                best_val_balanced_accuracy = (
                    val_metrics[
                        "balanced_accuracy"
                    ]
                )

                best_round = (
                    round_number
                )

                best_threshold = (
                    threshold
                )

                best_global_weights = [
                    np.copy(w)
                    for w in global_weights
                ]

                rounds_without_improvement = 0

                print(
                    "\nNew best global model."
                )

            else:

                rounds_without_improvement += 1

                print(
                    f"\nNo improvement: "
                    f"{rounds_without_improvement}/"
                    f"{FEDERATED_EARLY_STOPPING_PATIENCE}"
                )

            # EARLY STOPPING

            if (
                rounds_without_improvement
                >=
                FEDERATED_EARLY_STOPPING_PATIENCE
            ):

                print(
                    "\nFederated early stopping."
                )

                break

        # RESTORE BEST MODEL

        if best_global_weights is None:

            raise RuntimeError(
                "No best global model was saved."
            )

        global_model.set_weights(
            best_global_weights
        )

        # FINAL VALIDATION

        (
            final_val_loss,
            final_val_prob
        ) = evaluate_model(

            global_model,

            val_BVP,
            val_EDA,
            val_TEMP,
            val_ACC,

            y_val
        )

        (
            best_threshold,
            final_threshold_score
        ) = find_best_threshold(
            y_val,
            final_val_prob
        )

        # FINAL TEST

        test_probabilities = predict_model(

            global_model,

            test_BVP,
            test_EDA,
            test_TEMP,
            test_ACC
        )

        (
            metrics,
            test_predictions,
            test_cm
        ) = calculate_metrics(

            y_test,

            test_probabilities,

            best_threshold
        )

        print("\n")
        print("=" * 90)
        print(
            f"FOLD {fold} TEST PERFORMANCE"
        )
        print("=" * 90)

        for name, value in metrics.items():

            print(
                f"{name:25s}: "
                f"{value:.4f}"
            )

        # ROC

        (
            fold_fpr,
            fold_tpr,
            fold_auc
        ) = plot_fold_roc(

            y_test,

            test_probabilities,

            fold
        )

        fold_roc_data.append({

            "fpr":
                fold_fpr,

            "tpr":
                fold_tpr,

            "auc":
                fold_auc
        })

        # CONFUSION MATRIX

        plot_confusion_matrix(

            test_cm,

            title=(
                f"Federated Fold {fold} "
                "Confusion Matrix"
            ),

            filename=os.path.join(

                RESULTS_DIR,

                f"Federated_Fold_{fold}_"
                "Confusion_Matrix.png"
            )
        )

        # CLASSIFICATION REPORT

        print("\n")

        print(
            classification_report(

                y_test,

                test_predictions,

                target_names=[
                    "Non-Stress",
                    "Stress"
                ],

                digits=4,

                zero_division=0
            )
        )

        # SAVE PRIVACY DATA

        pd.DataFrame(
            privacy_update_records
        ).to_csv(

            os.path.join(

                RESULTS_DIR,

                f"Federated_Fold_{fold}_"
                "Privacy_Update_Features.csv"
            ),

            index=False
        )

        # SAVE HISTORY

        pd.DataFrame(
            fold_history
        ).to_csv(

            os.path.join(

                RESULTS_DIR,

                f"Federated_Fold_{fold}_"
                "Learning_History.csv"
            ),

            index=False
        )

        # SAVE MODEL

        global_model.save(

            os.path.join(

                RESULTS_DIR,

                f"Federated_Best_Global_Model_"
                f"Fold_{fold}.keras"
            )
        )

        # SAVE FOLD RESULT

        fold_result = {

            "Fold":
                fold,

            "Best_Round":
                best_round,

            "Threshold":
                best_threshold
        }

        fold_result.update(
            metrics
        )

        fold_results.append(
            fold_result
        )

        histories.append(
            fold_history
        )

        # OUT-OF-FOLD

        all_true.extend(
            y_test.tolist()
        )

        all_pred.extend(
            test_predictions.tolist()
        )

        all_prob.extend(
            test_probabilities.tolist()
        )

        # CLEAN

        del global_model

        tf.keras.backend.clear_session()

        gc.collect()

    # RESULTS DATAFRAME

    results_df = pd.DataFrame(
        fold_results
    )

    print("\n")
    print("=" * 110)
    print(
        "FEDERATED 5-FOLD TEST RESULTS"
    )
    print("=" * 110)

    print(
        results_df.round(4)
    )

    # MEAN ± SD

    metric_names = [

        "accuracy",

        "balanced_accuracy",

        "sensitivity",

        "specificity",

        "precision",

        "PPV",

        "NPV",

        "F1",

        "ROC_AUC",

        "PR_AUC"
    ]

    summary_rows = []

    for metric in metric_names:

        mean = results_df[
            metric
        ].mean()

        std = results_df[
            metric
        ].std(
            ddof=1
        )

        summary_rows.append({

            "Metric":
                metric,

            "Mean":
                mean,

            "SD":
                std,

            "Mean_percent":
                mean * 100,

            "SD_percent":
                std * 100
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    print("\n")
    print(
        "FEDERATED MEAN ± SD"
    )

    for _, row in (
        summary_df.iterrows()
    ):

        print(
            f"{row['Metric']:25s}: "
            f"{row['Mean']:.4f} ± "
            f"{row['SD']:.4f}"
        )

    # OUT-OF-FOLD ARRAYS

    all_true = np.asarray(
        all_true,
        dtype=np.int32
    )

    all_pred = np.asarray(
        all_pred,
        dtype=np.int32
    )

    all_prob = np.asarray(
        all_prob,
        dtype=np.float32
    )

    # OVERALL METRICS

    overall_cm = confusion_matrix(

        all_true,

        all_pred,

        labels=[0, 1]
    )

    tn, fp, fn, tp = (
        overall_cm.ravel()
    )

    overall_specificity = (

        tn / (tn + fp)

        if (tn + fp) > 0

        else 0.0
    )

    overall_metrics = {

        "accuracy":
            accuracy_score(
                all_true,
                all_pred
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                all_true,
                all_pred
            ),

        "sensitivity":
            recall_score(
                all_true,
                all_pred,
                zero_division=0
            ),

        "specificity":
            overall_specificity,

        "PPV":
            precision_score(
                all_true,
                all_pred,
                zero_division=0
            ),

        "NPV":
            (
                tn / (tn + fn)
                if (tn + fn) > 0
                else 0.0
            ),

        "F1":
            f1_score(
                all_true,
                all_pred,
                zero_division=0
            ),

        "ROC_AUC":
            roc_auc_score(
                all_true,
                all_prob
            ),

        "PR_AUC":
            average_precision_score(
                all_true,
                all_prob
            )
    }

    print("\n")
    print("=" * 110)
    print(
        "OVERALL FEDERATED OUT-OF-FOLD PERFORMANCE"
    )
    print("=" * 110)

    for name, value in (
        overall_metrics.items()
    ):

        print(
            f"{name:25s}: "
            f"{value:.4f}"
        )

    # OVERALL CONFUSION MATRIX

    plot_confusion_matrix(

        overall_cm,

        title=(
            "Overall Federated "
            "Out-of-Fold Confusion Matrix"
        ),

        filename=os.path.join(

            RESULTS_DIR,

            "Overall_Federated_Out_of_Fold_"
            "Confusion_Matrix.png"
        )
    )

    # CLASSIFICATION REPORT

    print("\n")

    print(
        "OVERALL FEDERATED CLASSIFICATION REPORT"
    )

    print(
        classification_report(

            all_true,

            all_pred,

            target_names=[
                "Non-Stress",
                "Stress"
            ],

            digits=4,

            zero_division=0
        )
    )

    # OVERALL ROC

    (
        overall_fpr,
        overall_tpr,
        overall_roc_auc
    ) = plot_overall_roc(

        all_true,

        all_prob
    )

    # MEAN FOLD ROC

    (
        mean_fold_auc,
        fold_auc_mean,
        fold_auc_std
    ) = plot_mean_fold_roc(

        fold_roc_data
    )


    # PR CURVE

    overall_pr_auc_curve = (
        plot_precision_recall(

            all_true,

            all_prob
        )
    )

    # LEARNING CURVES

    curve_df = (
        plot_average_learning_curves(
            histories
        )
    )

    # SAVE RESULTS

    results_df.to_csv(

        os.path.join(

            RESULTS_DIR,

            "Federated_5Fold_Test_Results.csv"
        ),

        index=False
    )

    summary_df.to_csv(

        os.path.join(

            RESULTS_DIR,

            "Federated_5Fold_Mean_SD_Results.csv"
        ),

        index=False
    )

    np.save(

        os.path.join(

            RESULTS_DIR,

            "Federated_All_Test_Labels.npy"
        ),

        all_true
    )

    np.save(

        os.path.join(

            RESULTS_DIR,

            "Federated_All_Test_Predictions.npy"
        ),

        all_pred
    )

    np.save(

        os.path.join(

            RESULTS_DIR,

            "Federated_All_Test_Probabilities.npy"
        ),

        all_prob
    )

    # SUBJECT ASSIGNMENTS

    pd.DataFrame(
        fold_subject_records
    ).to_csv(

        os.path.join(

            RESULTS_DIR,

            "Federated_Subject_Assignments.csv"
        ),

        index=False
    )

    # OVERALL METRICS

    pd.DataFrame({

        "Metric":
            list(
                overall_metrics.keys()
            ),

        "Value":
            list(
                overall_metrics.values()
            )

    }).to_csv(

        os.path.join(

            RESULTS_DIR,

            "Federated_Overall_Out_of_Fold_"
            "Metrics.csv"
        ),

        index=False
    )

    # ROC SUMMARY

    pd.DataFrame({

        "Metric": [

            "Overall_Out_of_Fold_ROC_AUC",

            "Mean_Fold_ROC_AUC",

            "SD_Fold_ROC_AUC"
        ],

        "Value": [

            overall_roc_auc,

            fold_auc_mean,

            fold_auc_std
        ]

    }).to_csv(

        os.path.join(

            RESULTS_DIR,

            "Federated_ROC_Summary.csv"
        ),

        index=False
    )

    # FINAL SUMMARY

    print("\n")
    print("=" * 50)
    print(
        "FINAL FEDERATED LEARNING SUMMARY"
    )
    print("=" * 50)

    print(
        f"\nMaximum communication rounds: "
        f"{NUM_ROUNDS}"
    )

    print(
        f"Local epochs per round: "
        f"{LOCAL_EPOCHS}"
    )

    print(
        f"Maximum local epochs/client: "
        f"{NUM_ROUNDS * LOCAL_EPOCHS}"
    )

    print(
        f"\nAccuracy: "
        f"{results_df['accuracy'].mean() * 100:.2f}% "
        f"± "
        f"{results_df['accuracy'].std(ddof=1) * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{results_df['balanced_accuracy'].mean() * 100:.2f}% "
        f"± "
        f"{results_df['balanced_accuracy'].std(ddof=1) * 100:.2f}%"
    )

    print(
        f"Sensitivity: "
        f"{results_df['sensitivity'].mean() * 100:.2f}% "
        f"± "
        f"{results_df['sensitivity'].std(ddof=1) * 100:.2f}%"
    )

    print(
        f"Specificity: "
        f"{results_df['specificity'].mean() * 100:.2f}% "
        f"± "
        f"{results_df['specificity'].std(ddof=1) * 100:.2f}%"
    )

    print(
        f"PPV: "
        f"{results_df['PPV'].mean() * 100:.2f}% "
        f"± "
        f"{results_df['PPV'].std(ddof=1) * 100:.2f}%"
    )

    print(
        f"NPV: "
        f"{results_df['NPV'].mean() * 100:.2f}% "
        f"± "
        f"{results_df['NPV'].std(ddof=1) * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{results_df['F1'].mean() * 100:.2f}% "
        f"± "
        f"{results_df['F1'].std(ddof=1) * 100:.2f}%"
    )

    print(
        f"ROC-AUC: "
        f"{results_df['ROC_AUC'].mean():.4f} "
        f"± "
        f"{results_df['ROC_AUC'].std(ddof=1):.4f}"
    )

    print(
        f"PR-AUC: "
        f"{results_df['PR_AUC'].mean():.4f} "
        f"± "
        f"{results_df['PR_AUC'].std(ddof=1):.4f}"
    )

    print(
        f"\nOverall out-of-fold ROC-AUC: "
        f"{overall_roc_auc:.4f}"
    )

    print(
        f"Mean fold ROC-AUC: "
        f"{fold_auc_mean:.4f} ± "
        f"{fold_auc_std:.4f}"
    )

    print(
        f"\nPrivacy defense: "
        f"{PRIVACY_MODE}"
    )

    print(
        f"Clip norm: "
        f"{CLIP_NORM}"
    )

    print(
        f"Noise multiplier: "
        f"{NOISE_MULTIPLIER}"
    )

    print(
        "\nResults saved to:"
    )

    print(
        os.path.abspath(
            RESULTS_DIR
        )
    )

    print("\n")
    print("=" * 110)
    print(
        "FEDERATED EXPERIMENT COMPLETED"
    )
    print("=" * 110)

    return (
        results_df,
        summary_df,
        histories,
        overall_metrics
    )


# MAIN

if __name__ == "__main__":

    print("\n")
    print("=" * 110)

    print(
        "WESAD WRIST STRESS DETECTION"
    )

    print(
        "MULTI-RATE FEDERATED CNN-BiLSTM-ATTENTION"
    )

    print(
        "SAMPLE-WEIGHTED FEDAVG"
    )

    print(
        "CLIENT UPDATE DEFENSE"
    )

    print(
        "5-FOLD SUBJECT-INDEPENDENT CV"
    )

    print("=" * 110)

    # BUILD DATASET

    (
        BVP,
        EDA,
        TEMP,
        ACC,
        y,
        groups
    ) = build_dataset()

    # AVAILABLE SUBJECTS

    available_subjects = np.unique(
        groups
    )

    print(
        "\nAvailable subjects:"
    )

    print(
        list(available_subjects)
    )

    print(
        f"\nNumber of available subjects: "
        f"{len(available_subjects)}"
    )

    if "S12" not in available_subjects:

        print(
            "\nS12 is unavailable and has "
            "automatically been excluded."
        )

    # MINIMUM SUBJECT CHECK

    if len(available_subjects) < 10:

        raise RuntimeError(
            "Too few subjects are available "
            "for reliable 5-fold subject-independent CV."
        )

    # CLASS CHECK

    if len(
        np.unique(y)
    ) < 2:

        raise RuntimeError(
            "Both stress and non-stress classes "
            "must be present."
        )


    # RUN

    (
        results_df,
        summary_df,
        histories,
        overall_metrics
    ) = run_federated_5fold(

        BVP,
        EDA,
        TEMP,
        ACC,
        y,
        groups
    )

    print("\n")
    print("-" * 50)
    print("DONE")
    print("-" * 50)
