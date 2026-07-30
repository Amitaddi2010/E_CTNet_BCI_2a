"""
Optimized 5-Fold Stratified CV on local GDF Session T data.
Uses: EA, S&R augmentation (n_seg=8, n_aug=4), Mixup, CosineAnnealingWarmRestarts,
      Gradient Clipping, Early Stopping, Label Smoothing.
"""
import yaml, numpy as np, torch, mne, sys
from dataset import EEGDataset, apply_sr_augmentation, compute_euclidean_alignment
from models import CTNet
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import os

sys.stdout.reconfigure(line_buffering=True)

with open('config.yaml') as f:
    config = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

data_dir = config['dataset']['data_dir']
fmin = config['dataset']['bandpass_low']
fmax = config['dataset']['bandpass_high']


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
    epochs = mne.Epochs(raw, selected_events, event_id, tmin=0, tmax=3.996,
                         baseline=None, preload=True, verbose=False)
    X = epochs.get_data()  # (N, 22, 1000)
    raw_labels = epochs.events[:, -1]
    unique_labels = sorted(np.unique(raw_labels))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in raw_labels], dtype=np.int64)
    return X, y


def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


all_sub_accs = []
for sub_id in range(1, 10):
    print(f'\n--- Subject {sub_id} ---')
    sub_str = f'A0{sub_id}'
    t_file = os.path.join(data_dir, f'{sub_str}T.gdf')

    raw_t = load_gdf_raw(t_file, fmin, fmax)
    X_t, y_t = extract_trials(raw_t)
    print(f'  Session T: {X_t.shape[0]} trials, {X_t.shape[1]}ch, {X_t.shape[2]}t')

    X_all = X_t[:, :, :1000]
    y_all = y_t

    X_all = compute_euclidean_alignment(X_all)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]

        val_ds = EEGDataset(X_val, y_val, augment_config=None, is_train=False)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=72, shuffle=False)

        torch.manual_seed(42 + fold)
        from models import CTNet
        model = CTNet(in_channels=22, time_steps=1000, n_classes=4, F1=8, D=2, F2=16,
                      d_model=16, nhead=2, num_layers=6, dropout=0.3).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=2)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        best_val_acc = 0
        patience_counter = 0
        aug_cfg = {'enable_noise': True, 'noise_std': 0.005, 'enable_channel_mask': True, 'channel_mask_prob': 0.10}

        for epoch in range(1, 201):
            # Dynamic S&R Augmentation
            X_tr_aug, y_tr_aug = apply_sr_augmentation(X_tr, y_tr, num_segments=8, n_aug=4)
            tr_ds = EEGDataset(X_tr_aug, y_tr_aug, augment_config=aug_cfg, is_train=True)
            tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=72, shuffle=True, drop_last=True)

            model.train()
            for x, _, y_b in tr_loader:
                x, y_b = x.to(device), y_b.to(device)
                mx, ya, yb, lam = mixup_data(x, y_b, alpha=0.3)
                optimizer.zero_grad()
                out = model(mx)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            if epoch >= 30 and epoch % 5 == 0:
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
                if patience_counter >= 16:
                    break

        fold_accs.append(best_val_acc)
        print(f'  Fold {fold+1}/5 | Best Val Acc: {best_val_acc*100:.2f}%')

    sub_mean = np.mean(fold_accs) * 100.0
    all_sub_accs.append(sub_mean)
    print(f'  => Subject {sub_id} 5-Fold Acc: {sub_mean:.2f}%')

print(f'\n======================================')
print(f'Overall 5-Fold CV Mean Acc: {np.mean(all_sub_accs):.2f}% +/- {np.std(all_sub_accs):.2f}%')
print(f'Per-Subject Accs: {[f"{a:.2f}" for a in all_sub_accs]}')
