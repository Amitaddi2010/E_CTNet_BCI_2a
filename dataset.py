import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.signal as signal

# Suppress verbose warnings from MNE / MOABB during data loading
import warnings
warnings.filterwarnings("ignore")

try:
    import mne
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    MOABB_AVAILABLE = True
except ImportError:
    MOABB_AVAILABLE = False


def preprocess_eeg_raw(raw, fmin=4.0, fmax=38.0, resample_rate=250, tmin=0.0, tmax=4.0):
    """
    Standard MNE preprocessing pipeline for raw EEG signals:
    1. Filter channels to 22 EEG channels.
    2. Bandpass filter (fmin-fmax Hz).
    3. Resample to resample_rate Hz.
    4. Extract trial epochs based on cue annotations (769, 770, 771, 772).
    """
    # Standard 22 EEG channels for Dataset 2a
    eeg_ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
        'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'Oz'
    ]
    
    # Filter EEG channels if present
    available_chs = raw.ch_names
    selected_chs = [ch for ch in available_chs if any(eeg_name in ch for eeg_name in eeg_ch_names)]
    if len(selected_chs) == 22:
        raw.pick_channels(selected_chs)
    elif len(available_chs) >= 22:
        raw.pick_channels(available_chs[:22])

    raw.load_data()
    raw.filter(fmin, fmax, fir_design='firwin', verbose=False)
    if raw.info['sfreq'] != resample_rate:
        raw.resample(resample_rate, verbose=False)

    # Event extraction
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    
    # Event mapping for Dataset 2a: 769->0 (Left), 770->1 (Right), 771->2 (Feet), 772->3 (Tongue)
    event_id_target = {}
    for key, val in event_dict.items():
        if '769' in key:
            event_id_target[key] = val
        elif '770' in key:
            event_id_target[key] = val
        elif '771' in key:
            event_id_target[key] = val
        elif '772' in key:
            event_id_target[key] = val

    if not event_id_target:
        # Fallback event mapping if event codes differ
        event_id_target = event_dict

    epochs = mne.Epochs(
        raw, events, event_id=event_id_target, tmin=tmin, tmax=tmax,
        baseline=None, preload=True, verbose=False, event_repeated='drop'
    )
    
    X = epochs.get_data()  # Shape: (N, 22, T)
    
    # Remap event labels to 0..3
    raw_labels = epochs.events[:, -1]
    unique_labels = sorted(np.unique(raw_labels))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in raw_labels], dtype=np.int64)

    return X, y


def load_subject_data_local(data_dir, subject_id, fmin=4.0, fmax=38.0, resample_rate=250, tmin=0.0, tmax=4.0):
    """
    Loads subject data from local GDF files (A01T.gdf / A01E.gdf) using MNE.
    If evaluation GDF (A01E) contains unannotated competition cues (code 783),
    performs a stratified 2-session split (Session 1 train, Session 2 test) on the labeled GDF file.
    """
    sub_str = f"A0{subject_id}" if subject_id < 10 else f"A{subject_id}"
    t_file = glob.glob(os.path.join(data_dir, f"{sub_str}T.gdf"))
    e_file = glob.glob(os.path.join(data_dir, f"{sub_str}E.gdf"))

    if not t_file:
        raise FileNotFoundError(f"GDF training file for subject {subject_id} not found in {data_dir}")

    raw_t = mne.io.read_raw_gdf(t_file[0], preload=True, verbose=False)
    X_all, y_all = preprocess_eeg_raw(raw_t, fmin, fmax, resample_rate, tmin, tmax)

    # Check if test file has valid motor imagery events
    X_test, y_test = None, None
    if e_file:
        try:
            raw_e = mne.io.read_raw_gdf(e_file[0], preload=True, verbose=False)
            X_e, y_e = preprocess_eeg_raw(raw_e, fmin, fmax, resample_rate, tmin, tmax)
            if len(y_e) > 0 and len(np.unique(y_e)) == 4:
                X_test, y_test = X_e, y_e
        except Exception:
            pass

    if X_test is None or len(X_test) == 0:
        # Perform Stratified Session 1 / Session 2 split from local dataset
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.5, random_state=42, stratify=y_all
        )
    else:
        X_train, y_train = X_all, y_all

    return (X_train, y_train), (X_test, y_test)



def load_subject_data_moabb(subject_id, fmin=4.0, fmax=38.0, tmin=0.0, tmax=4.0, resample_rate=250):
    """
    Loads subject data via MOABB BNCI2014_001 dataset wrapper.
    """
    dataset = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4, fmin=fmin, fmax=fmax, resample=resample_rate, tmin=tmin, tmax=tmax)
    
    # Fetch data for single subject
    dataset.subject_list = [subject_id]
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    
    # Label encoding: 4 motor imagery classes -> 0, 1, 2, 3
    unique_labels = sorted(np.unique(y))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y_encoded = np.array([label_map[l] for l in y], dtype=np.int64)

    # Session splitting: '0train' or 'session_0' -> Train (Session 1), '1test' or 'session_1' -> Test (Session 2)
    sessions = metadata['session'].values
    train_mask = (sessions == sessions[0])
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y_encoded[train_mask]
    X_test, y_test = X[test_mask], y_encoded[test_mask]

    return (X_train, y_train), (X_test, y_test)


def compute_euclidean_alignment(X):
    """
    Euclidean Alignment (EA) for EEG signals (He & Wu, 2019):
    Computes average covariance matrix R across all N trials: R = (1/N) * sum(X_i @ X_i.T / T)
    Transforms each trial X_i <- R^(-1/2) @ X_i
    Input X shape: (N, C, T)
    Output X_ea shape: (N, C, T)
    """
    if X is None or len(X) == 0:
        return X
    N, C, T = X.shape
    covs = np.zeros((N, C, C), dtype=np.float64)
    for i in range(N):
        covs[i] = np.dot(X[i], X[i].T) / float(T)
    
    R = np.mean(covs, axis=0) + 1e-6 * np.eye(C)
    
    evals, evecs = np.linalg.eigh(R)
    evals = np.maximum(evals, 1e-8)
    R_inv_sqrt = np.dot(evecs * (1.0 / np.sqrt(evals)), evecs.T)
    
    X_ea = np.zeros_like(X, dtype=np.float32)
    for i in range(N):
        X_ea[i] = np.dot(R_inv_sqrt, X[i])
        
    return X_ea


def get_dataset_for_subject(config, subject_id, prefer_moabb=True):
    """
    Primary interface to get subject dataset with smart fallback strategy:
    1. If prefer_moabb=True and MOABB is available, fetch via MOABB for full Session 1/2 true labels.
    2. Otherwise, load locally from GDF files (instantaneous).
    Applies Euclidean Alignment (EA) to standardize spatial covariance.
    Returns (X_train, y_train), (X_test, y_test) where X shape is (N, 22, time_steps).
    """
    data_dir = config['dataset']['data_dir']
    fmin = config['dataset']['bandpass_low']
    fmax = config['dataset']['bandpass_high']
    tmin = config['dataset']['cue_offset_start']
    tmax = config['dataset']['cue_offset_end']
    resample_rate = config['dataset']['sampling_rate']

    sub_str = f"A0{subject_id}" if subject_id < 10 else f"A{subject_id}"
    local_gdf_exists = os.path.exists(os.path.join(data_dir, f"{sub_str}T.gdf"))

    if prefer_moabb and MOABB_AVAILABLE:
        print(f"[Dataset] Loading Subject {subject_id} via MOABB BNCI2014_001 dataset wrapper...")
        try:
            (X_train, y_train), (X_test, y_test) = load_subject_data_moabb(subject_id, fmin, fmax, tmin, tmax, resample_rate)
        except Exception as e:
            print(f"[Dataset] MOABB load failed ({e}), falling back to local GDF loading...")
            (X_train, y_train), (X_test, y_test) = load_subject_data_local(data_dir, subject_id, fmin, fmax, resample_rate, tmin, tmax)
    elif local_gdf_exists:
        print(f"[Dataset] Found local GDF files for Subject {subject_id} in '{data_dir}'. Loading locally via MNE...")
        (X_train, y_train), (X_test, y_test) = load_subject_data_local(data_dir, subject_id, fmin, fmax, resample_rate, tmin, tmax)
    else:
        (X_train, y_train), (X_test, y_test) = load_subject_data_local(data_dir, subject_id, fmin, fmax, resample_rate, tmin, tmax)

    # Apply Euclidean Alignment (EA)
    X_train = compute_euclidean_alignment(X_train)
    if X_test is not None and len(X_test) > 0:
        X_test = compute_euclidean_alignment(X_test)

    return (X_train, y_train), (X_test, y_test)


class EEGDataset(Dataset):
    """
    PyTorch Dataset with per-trial channel Z-score normalization and augmentation options:
    - Gaussian Noise Injection
    - Mixup Augmentation
    - Non-destructive Temporal Window Slicing
    """
    def __init__(self, X, y, augment_config=None, is_train=True):
        """
        X: np.ndarray of shape (N, C, T)
        y: np.ndarray of shape (N,)
        """
        # Avoid per-channel Z-score standardization as it destroys Euclidean Alignment (EA) spatial covariance
        # Use X directly as EA already scales the data properly
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.is_train = is_train
        self.augment_config = augment_config or {}
        self.num_classes = 4

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()  # Shape: (C, T)
        y = self.y[idx]

        if self.is_train and self.augment_config:
            # 1. Continuous Gaussian Noise Injection
            if self.augment_config.get('enable_noise', False):
                std_dev = self.augment_config.get('noise_std', 0.01)
                noise = torch.randn_like(x) * std_dev
                x = x + noise

            # 2. Non-destructive Temporal Random Crop / Sliding Window
            if self.augment_config.get('enable_crop', False):
                crop_size = self.augment_config.get('crop_size', 900)
                T = x.shape[-1]
                if crop_size < T:
                    start = torch.randint(0, T - crop_size + 1, (1,)).item()
                    x_crop = x[:, start:start+crop_size]
                    x = torch.zeros_like(x)
                    x[:, :crop_size] = x_crop

            # 3. Random EEG Channel Masking (Channel Dropout)
            if self.augment_config.get('enable_channel_mask', False):
                mask_prob = self.augment_config.get('channel_mask_prob', 0.15)
                mask = (torch.rand(x.shape[0], 1) > mask_prob).float()
                x = x * mask

        # One-hot target for potential mixup training compatibility
        y_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes).float()

        return x, y_onehot, y


def apply_sr_augmentation(X, y, num_segments=4, n_aug=3):
    """
    Segmentation & Reconstruction (S&R) Data Augmentation (Zhao et al., 2024 - CTNet):
    1. Standardizes time dimension to 1000 time steps.
    2. Splits trials into num_segments temporal slices.
    3. Synthesizes new artificial trials by recombining intra-class temporal segments.
    4. Expands training set size by factor n_aug (default n_aug=3).
    """
    X = X[:, :, :1000]
    N, C, T = X.shape
    if n_aug <= 1 or N == 0:
        return X, y

    seg_len = T // num_segments
    augmented_X_list = [X]
    augmented_y_list = [y]

    unique_classes = np.unique(y)
    for c in unique_classes:
        c_indices = np.where(y == c)[0]
        if len(c_indices) < 2:
            continue
        
        X_c = X[c_indices]
        N_c = len(X_c)
        segments_c = []
        for i in range(num_segments):
            segments_c.append(X_c[:, :, i*seg_len:(i+1)*seg_len])
        
        num_new_trials = (n_aug - 1) * N_c
        for _ in range(num_new_trials):
            new_trial_segs = []
            for i in range(num_segments):
                rand_idx = np.random.choice(N_c)
                new_trial_segs.append(segments_c[i][rand_idx])
            new_trial = np.concatenate(new_trial_segs, axis=-1)
            augmented_X_list.append(new_trial[np.newaxis, :, :])
            augmented_y_list.append(np.array([c], dtype=y.dtype))

    X_aug = np.concatenate(augmented_X_list, axis=0)
    y_aug = np.concatenate(augmented_y_list, axis=0)
    return X_aug, y_aug


def apply_mixup(x, y_onehot, alpha=0.2):
    """
    Applies Mixup data augmentation on a batch of EEG data.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y_onehot + (1 - lam) * y_onehot[index]

    return mixed_x, mixed_y


def get_dataloaders(config, target_subject, stage='fine_tune'):
    """
    Constructs DataLoaders for Stage 1 (Pre-training on 8 out-of-subject datasets)
    or Stage 2 (Subject-specific fine-tuning on target subject).
    """
    batch_size = config['training']['batch_size']
    aug_config = config['augmentation']
    num_subjects = config['dataset']['num_subjects']

    if stage == 'pretrain':
        # Load all subjects EXCEPT target_subject
        X_train_list, y_train_list = [], []
        for sub_id in range(1, num_subjects + 1):
            if sub_id == target_subject:
                continue
            (X_tr, y_tr), _ = get_dataset_for_subject(config, sub_id, prefer_moabb=False)
            if aug_config.get('enable_sr', True):
                X_tr, y_tr = apply_sr_augmentation(X_tr, y_tr, n_aug=aug_config.get('n_aug', 3))
            X_train_list.append(X_tr)
            y_train_list.append(y_tr)

        X_train_all = np.concatenate(X_train_list, axis=0)
        y_train_all = np.concatenate(y_train_list, axis=0)

        train_dataset = EEGDataset(X_train_all, y_train_all, augment_config=aug_config, is_train=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_loader, None

    elif stage == 'fine_tune':
        # Target subject Session 1 for training, Session 2 for testing
        (X_train, y_train), (X_test, y_test) = get_dataset_for_subject(config, target_subject, prefer_moabb=False)
        X_test = X_test[:, :, :1000]

        if aug_config.get('enable_sr', True):
            X_train, y_train = apply_sr_augmentation(X_train, y_train, n_aug=aug_config.get('n_aug', 3))

        train_dataset = EEGDataset(X_train, y_train, augment_config=aug_config, is_train=True)
        test_dataset = EEGDataset(X_test, y_test, augment_config=None, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
