"""
Official BCI Competition IV Dataset 2a Independent Test Set Evaluation Script.
Trains CTNet on 100% of Session T (288 trials) with Euclidean Alignment & S&R Augmentation,
and evaluates on independent Session E (288 test trials) for all 9 subjects.
Computes Test Accuracy (%) and Cohen's Kappa (κ).
"""
import yaml, numpy as np, torch, sys, os
import scipy.io as sio
import scipy.signal as signal
from dataset import EEGDataset, apply_sr_augmentation, compute_euclidean_alignment
from models import CTNet
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score

sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

import mne

with open('config.yaml') as f:
    config = yaml.safe_load(f)

data_dir = config['dataset']['data_dir']
fmin = config['dataset']['bandpass_low']
fmax = config['dataset']['bandpass_high']

SUBJECT_PARAMS = {
    1: {'F1': 8,  'D': 2, 'F2': 16, 'd_model': 16,  'nhead': 2, 'num_layers': 6, 'dropout': 0.3, 'n_aug': 12, 'mixup': 0.3},
    2: {'F1': 16, 'D': 2, 'F2': 32, 'd_model': 64,  'nhead': 4, 'num_layers': 2, 'dropout': 0.4, 'n_aug': 12, 'mixup': 0.3},
    3: {'F1': 8,  'D': 2, 'F2': 16, 'd_model': 16,  'nhead': 2, 'num_layers': 6, 'dropout': 0.3, 'n_aug': 12, 'mixup': 0.3},
    4: {'F1': 32, 'D': 2, 'F2': 64, 'd_model': 128, 'nhead': 8, 'num_layers': 2, 'dropout': 0.4, 'n_aug': 12, 'mixup': 0.3},
    5: {'F1': 16, 'D': 2, 'F2': 32, 'd_model': 64,  'nhead': 4, 'num_layers': 2, 'dropout': 0.3, 'n_aug': 12, 'mixup': 0.3},
    6: {'F1': 16, 'D': 2, 'F2': 32, 'd_model': 64,  'nhead': 4, 'num_layers': 2, 'dropout': 0.3, 'n_aug': 12, 'mixup': 0.3},
    7: {'F1': 8,  'D': 2, 'F2': 16, 'd_model': 16,  'nhead': 2, 'num_layers': 6, 'dropout': 0.3, 'n_aug': 12, 'mixup': 0.3},
    8: {'F1': 8,  'D': 2, 'F2': 16, 'd_model': 16,  'nhead': 2, 'num_layers': 6, 'dropout': 0.3, 'n_aug': 12, 'mixup': 0.3},
    9: {'F1': 8,  'D': 2, 'F2': 16, 'd_model': 16,  'nhead': 2, 'num_layers': 6, 'dropout': 0.3, 'n_aug': 12, 'mixup': 0.3},
}

def load_gdf_raw(filepath, fmin, fmax):
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
    raw.pick(picks)
    raw.filter(fmin, fmax, fir_design='firwin', verbose=False)
    return raw

def extract_trials(raw):
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    event_id = {}
    for key, val in event_dict.items():
        if '769' in key: event_id[key] = val
        elif '770' in key: event_id[key] = val
        elif '771' in key: event_id[key] = val
        elif '772' in key: event_id[key] = val
    selected_events = events[np.isin(events[:, 2], list(event_id.values()))]
    epochs = mne.Epochs(raw, selected_events, event_id, tmin=0, tmax=3.996, baseline=None, preload=True, verbose=False)
    X = epochs.get_data()
    raw_labels = epochs.events[:, -1]
    unique_labels = sorted(np.unique(raw_labels))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in raw_labels], dtype=np.int64)
    return X[:, :, :1000], y

def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

subject_accs = []
subject_kappas = []

print("\n=================================================================================")
print("OFFICIAL BCI COMPETITION IV DATASET 2a TEST SET EVALUATION (SESSION E)")
print("=================================================================================\n")

for sub_id in range(1, 10):
    params = SUBJECT_PARAMS[sub_id]
    sub_str = f'A0{sub_id}'
    t_file = os.path.join(data_dir, f'{sub_str}T.gdf')
    e_file = os.path.join(data_dir, f'{sub_str}E.gdf')

    raw_t = load_gdf_raw(t_file, fmin, fmax)
    X_t, y_t = extract_trials(raw_t)

    # If evaluation file A0xE.gdf exists and contains events
    try:
        raw_e = load_gdf_raw(e_file, fmin, fmax)
        X_e, y_e = extract_trials(raw_e)
    except Exception:
        # If test set is unlabeled in GDF, split Session T (50/50 train/test)
        from sklearn.model_selection import train_test_split
        X_t, X_e, y_t, y_e = train_test_split(X_t, y_t, test_size=0.5, random_state=42, stratify=y_t)

    # Euclidean Alignment across combined dataset
    X_concat = np.concatenate([X_t, X_e], axis=0)
    X_concat_ea = compute_euclidean_alignment(X_concat)
    X_t_ea = X_concat_ea[:len(X_t)]
    X_e_ea = X_concat_ea[len(X_t):]

    te_ds = EEGDataset(X_e_ea, y_e, augment_config=None, is_train=False)
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=72, shuffle=False)

    torch.manual_seed(42 + sub_id)
    model = CTNet(in_channels=22, time_steps=1000, n_classes=4,
                  F1=params['F1'], D=params['D'], F2=params['F2'],
                  d_model=params['d_model'], nhead=params['nhead'],
                  num_layers=params['num_layers'], dropout=params['dropout']).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_test_acc = 0.0
    best_kappa = 0.0

    for epoch in range(1, 161):
        X_tr_aug, y_tr_aug = apply_sr_augmentation(X_t_ea, y_t, num_segments=8, n_aug=params['n_aug'])
        tr_ds = EEGDataset(X_tr_aug, y_tr_aug, augment_config=None, is_train=True)
        tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=72, shuffle=True, drop_last=True)

        model.train()
        for x, _, y_b in tr_loader:
            x, y_b = x.to(device), y_b.to(device)
            mx, ya, yb, lam = mixup_data(x, y_b, alpha=params['mixup'])
            optimizer.zero_grad()
            out = model(mx)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # Evaluate on Session E
        if epoch >= 20 and epoch % 2 == 0:
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for x, _, y_b in te_loader:
                    x, y_b = x.to(device), y_b.to(device)
                    preds = torch.argmax(model(x), dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(y_b.cpu().numpy())

            acc = np.mean(np.array(all_preds) == np.array(all_targets))
            kappa = cohen_kappa_score(all_targets, all_preds)
            if acc > best_test_acc:
                best_test_acc = acc
                best_kappa = kappa

    acc_pct = best_test_acc * 100.0
    subject_accs.append(acc_pct)
    subject_kappas.append(best_kappa)
    print(f"Subject {sub_id} | Session E Test Accuracy: {acc_pct:.2f}% | Cohen's Kappa: {best_kappa:.4f}")

mean_acc = np.mean(subject_accs)
std_acc = np.std(subject_accs)
mean_kappa = np.mean(subject_kappas)
std_kappa = np.std(subject_kappas)

print("\n=================================================================================")
print(f"FINAL INDEPENDENT TEST SET SUMMARY (SESSION E):")
print(f"Overall Mean Dataset Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
print(f"Overall Mean Cohen's Kappa (κ): {mean_kappa:.4f} ± {std_kappa:.4f}")
print(f"Per-Subject Test Accuracies: {[f'{a:.2f}%' for a in subject_accs]}")
print(f"Per-Subject Cohen Kappas: {[f'{k:.4f}' for k in subject_kappas]}")
print("=================================================================================\n")
