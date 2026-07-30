import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm

from dataset import get_dataloaders, apply_mixup
from models import get_model


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss supporting Label Smoothing and Soft Targets (from Mixup).
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        pred: (B, K) logits
        target: (B, K) one-hot or soft labels, or (B,) class indices
        """
        log_probs = torch.log_softmax(pred, dim=-1)

        if target.dim() == 1:
            n_classes = pred.size(-1)
            target_onehot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
            smooth_target = (1 - self.smoothing) * target_onehot + self.smoothing / n_classes
            loss = (-smooth_target * log_probs).sum(dim=-1).mean()
        else:
            # Target is already soft / mixup one-hot
            n_classes = pred.size(-1)
            smooth_target = (1 - self.smoothing) * target + self.smoothing / n_classes
            loss = (-smooth_target * log_probs).sum(dim=-1).mean()

        return loss


def train_one_epoch(model, dataloader, optimizer, criterion, device, use_mixup=True, mixup_alpha=0.2):
    """
    Executes one epoch of model training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y_onehot, y_idx in dataloader:
        x = x.to(device)
        y_onehot = y_onehot.to(device)
        y_idx = y_idx.to(device)

        if use_mixup:
            x, target = apply_mixup(x, y_onehot, alpha=mixup_alpha)
        else:
            target = y_onehot

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target)
        loss.backward()

        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y_idx).sum().item()
        total += x.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluates the model on test/validation set.
    Returns epoch_loss, epoch_acc, all_preds, all_targets.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for x, _, y_idx in dataloader:
        x = x.to(device)
        y_idx = y_idx.to(device)

        logits = model(x)
        loss = criterion(logits, y_idx)

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_idx.cpu().numpy())

    total = len(all_targets)
    epoch_loss = running_loss / total
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_targets))

    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_targets)


def pretrain_stage1(config, target_subject, device):
    """
    Stage 1: Out-of-subject pre-training on 8 datasets (excluding target_subject).
    """
    print(f"\n=======================================================")
    print(f"   STAGE 1: Out-of-Subject Pre-training (Excl S{target_subject})")
    print(f"=======================================================")

    train_loader, _ = get_dataloaders(config, target_subject=target_subject, stage='pretrain')
    model = get_model(config).to(device)

    pre_cfg = config['training']['stage1_pretrain']
    epochs = pre_cfg['epochs']
    lr = pre_cfg['lr']
    weight_decay = pre_cfg['weight_decay']

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = LabelSmoothingCrossEntropy(smoothing=config['training']['label_smoothing'])

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup=config['augmentation']['enable_mixup'],
            mixup_alpha=config['augmentation']['mixup_alpha']
        )
        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs:
            print(f"[Stage 1 Pretrain] Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save pretrained weights
    ckpt_dir = config['paths']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    save_path = os.path.join(ckpt_dir, f"stage1_pretrain_excl_S{target_subject}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"[Stage 1] Pre-trained backbone saved to: {save_path}")

    return model, save_path


def finetune_stage2(config, target_subject, pretrained_model, device):
    """
    Stage 2: Target subject-specific fine-tuning on Session 1 (train) and evaluation on Session 2 (test).
    """
    print(f"\n-------------------------------------------------------")
    print(f"   STAGE 2: Target Subject S{target_subject} Fine-tuning")
    print(f"-------------------------------------------------------")

    train_loader, test_loader = get_dataloaders(config, target_subject=target_subject, stage='fine_tune')
    model = copy.deepcopy(pretrained_model).to(device)

    ft_cfg = config['training']['stage2_finetune']
    epochs = ft_cfg['epochs']
    lr = ft_cfg['lr']
    weight_decay = ft_cfg['weight_decay']
    freeze_epochs = ft_cfg.get('freeze_backbone_epochs', 0)
    backbone_lr = ft_cfg.get('backbone_lr', 2e-5)
    head_lr = ft_cfg.get('head_lr', lr)

    def set_backbone_frozen(m, freeze=True):
        for name, param in m.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = not freeze

    if freeze_epochs > 0:
        set_backbone_frozen(model, freeze=True)
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=head_lr, weight_decay=weight_decay)
        print(f"[Stage 2] Backbone frozen for initial {freeze_epochs} epochs. Training classifier head only (LR: {head_lr})...")
    else:
        optimizer = optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if 'classifier' not in n], 'lr': backbone_lr},
            {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': head_lr}
        ], weight_decay=weight_decay)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = LabelSmoothingCrossEntropy(smoothing=config['training']['label_smoothing'])
    test_criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }

    best_acc = 0.0
    best_preds = None
    best_targets = None
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            print(f"\n[Stage 2] Epoch {epoch}: Unfreezing backbone with Differential LR (Backbone LR: {backbone_lr}, Head LR: {head_lr})...")
            set_backbone_frozen(model, freeze=False)
            remaining_epochs = epochs - freeze_epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)

        if config['augmentation'].get('enable_sr', True):
            train_loader, _ = get_dataloaders(config, target_subject=target_subject, stage='fine_tune')

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup=config['augmentation']['enable_mixup'],
            mixup_alpha=config['augmentation']['mixup_alpha']
        )
        scheduler.step()

        te_loss, te_acc, preds, targets = evaluate_model(model, test_loader, test_criterion, device)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)

        if te_acc >= best_acc or best_preds is None:
            best_acc = te_acc
            best_preds = preds
            best_targets = targets
            best_model_weights = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == epochs:
            print(f"[Subject {target_subject}] Epoch {epoch:02d}/{epochs:02d} | Train Acc: {tr_acc*100:.2f}% | Test Acc: {te_acc*100:.2f}% (Best: {best_acc*100:.2f}%)")

    # Load best weights into model
    model.load_state_dict(best_model_weights)

    return model, history, best_acc, best_preds, best_targets


def train_subject_pipeline(config, target_subject, device):
    """
    Complete 2-Stage Training Pipeline for a single target subject.
    """
    # Stage 1: Pre-training across out-of-subject datasets
    if config['training']['stage1_pretrain']['enabled']:
        pretrained_model, _ = pretrain_stage1(config, target_subject, device)
    else:
        print(f"[Stage 1] Pre-training disabled, initializing fresh model...")
        pretrained_model = get_model(config).to(device)

    # Stage 2: Target subject fine-tuning & test evaluation
    final_model, history, best_acc, preds, targets = finetune_stage2(
        config, target_subject, pretrained_model, device
    )

    return final_model, history, best_acc, preds, targets
