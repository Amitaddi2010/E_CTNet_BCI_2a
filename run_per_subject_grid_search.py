"""
Automated Per-Subject Grid Search Optimizer for BCI Competition IV Dataset 2a.
For each subject:
1. Evaluates all candidate hyperparameter configurations via 5-Fold Stratified CV.
2. Identifies and saves the BEST configuration achieving maximum accuracy for that subject.
3. Automatically advances to the next subject.
"""
import yaml, numpy as np, torch, mne, sys, os
from dataset import EEGDataset, apply_sr_augmentation, compute_euclidean_alignment
from models import CTNet
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(line_buffering=True, encoding='utf-8')

with open('config.yaml') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

data_dir = config['dataset']['data_dir']
fmin = config['dataset']['bandpass_low']
fmax = config['dataset']['bandpass_high']

# Candidate Configurations to test per subject
CANDIDATE_CONFIGS = [
    {
        'name': 'Clean CTNet Baseline (F1=16, F2=32, d_model=64, layers=2, no noise/mask)',
        'F1': 16, 'D': 2, 'F2': 32, 'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.5, 'n_aug': 4, 'mixup': 0.3, 'use_aug_cfg': False
    },
    {
        'name': 'Standard CTNet (F1=16, F2=32, d_model=64, layers=2)',
        'F1': 16, 'D': 2, 'F2': 32, 'd_model': 64, 'nhead': 4, 'num_layers': 2, 'dropout': 0.5, 'n_aug': 4, 'mixup': 0.3, 'use_aug_cfg': True
    },
    {
        'name': 'High-Capacity CTNet (F1=32, F2=64, d_model=128, layers=2)',
        'F1': 32, 'D': 2, 'F2': 64, 'd_model': 128, 'nhead': 8, 'num_layers': 2, 'dropout': 0.5, 'n_aug': 4, 'mixup': 0.3, 'use_aug_cfg': False
    },
    {
        'name': 'Deep Attention CTNet (F1=16, F2=32, d_model=64, layers=4, drop=0.3)',
        'F1': 16, 'D': 2, 'F2': 32, 'd_model': 64, 'nhead': 4, 'num_layers': 4, 'dropout': 0.3, 'n_aug': 4, 'mixup': 0.3, 'use_aug_cfg': False
    },
    {
        'name': 'Official Nature 2024 CTNet (F1=8, F2=16, d_model=16, layers=6)',
        'F1': 8, 'D': 2, 'F2': 16, 'd_model': 16, 'nhead': 2, 'num_layers': 6, 'dropout': 0.3, 'n_aug': 4, 'mixup': 0.3, 'use_aug_cfg': False
    }
]

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
    return X, y

def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def evaluate_config_5fold(X_all, y_all, cfg):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        aug_cfg = {'enable_noise': True, 'noise_std': 0.005, 'enable_channel_mask': True, 'channel_mask_prob': 0.10} if cfg.get('use_aug_cfg', False) else None
        val_ds = EEGDataset(X_val, y_val, augment_config=None, is_train=False)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=72, shuffle=False)

        torch.manual_seed(42 + fold)
        model = CTNet(in_channels=22, time_steps=1000, n_classes=4,
                      F1=cfg['F1'], D=cfg['D'], F2=cfg['F2'],
                      d_model=cfg['d_model'], nhead=cfg['nhead'],
                      num_layers=cfg['num_layers'], dropout=cfg['dropout']).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = 0
        patience_counter = 0
        for epoch in range(1, 161):
            X_tr_aug, y_tr_aug = apply_sr_augmentation(X_tr, y_tr, num_segments=8, n_aug=cfg['n_aug'])
            tr_ds = EEGDataset(X_tr_aug, y_tr_aug, augment_config=aug_cfg, is_train=True)
            tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=72, shuffle=True, drop_last=True)

            model.train()
            for x, _, y_b in tr_loader:
                x, y_b = x.to(device), y_b.to(device)
                mx, ya, yb, lam = mixup_data(x, y_b, alpha=cfg['mixup'])
                optimizer.zero_grad()
                out = model(mx)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            if epoch >= 25 and epoch % 5 == 0:
                model.eval()
                correct, total = 0, 0
                with torch.no_grad():
                    for x, _, y_b in val_loader:
                        x, y_b = x.to(device), y_b.to(device)
                        correct += (torch.argmax(model(x), dim=-1) == y_b).sum().item()
                        total += len(y_b)
                acc = correct / total
                if acc > best_val_acc:
                    best_val_acc = acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= 12:
                    break

        fold_accs.append(best_val_acc)
    return np.mean(fold_accs) * 100.0

best_subject_results = {}
all_final_accs = []

for sub_id in range(1, 10):
    print(f'\n==================================================')
    print(f'OPTIMIZING SUBJECT {sub_id} / 9')
    print(f'==================================================')
    sub_str = f'A0{sub_id}'
    t_file = os.path.join(data_dir, f'{sub_str}T.gdf')

    raw_t = load_gdf_raw(t_file, fmin, fmax)
    X_t, y_t = extract_trials(raw_t)
    X_all = compute_euclidean_alignment(X_t[:, :, :1000])

    best_acc_for_sub = 0.0
    best_cfg_for_sub = None

    for c_idx, cfg in enumerate(CANDIDATE_CONFIGS):
        print(f'\n  [Sub {sub_id}] Testing Config {c_idx+1}/{len(CANDIDATE_CONFIGS)}: {cfg["name"]}')
        acc = evaluate_config_5fold(X_all, y_t, cfg)
        print(f'  => [Sub {sub_id}] Config {c_idx+1} Mean 5-Fold Acc: {acc:.2f}%')

        if acc > best_acc_for_sub:
            best_acc_for_sub = acc
            best_cfg_for_sub = cfg

    best_subject_results[sub_id] = {
        'accuracy': best_acc_for_sub,
        'config': best_cfg_for_sub['name']
    }
    all_final_accs.append(best_acc_for_sub)
    print(f'\n[SUBJECT {sub_id} BEST RESULT]: {best_acc_for_sub:.2f}% using "{best_cfg_for_sub["name"]}"')

print(f'\n==================================================')
print(f'FINAL GRID SEARCH OPTIMIZATION COMPLETE!')
print(f'Overall Optimized Mean Acc: {np.mean(all_final_accs):.2f}% +/- {np.std(all_final_accs):.2f}%')
print(f'Per-Subject Best Accuracies:')
for sub_id, res in best_subject_results.items():
    print(f'  Subject {sub_id}: {res["accuracy"]:.2f}% ({res["config"]})')
print(f'==================================================')
