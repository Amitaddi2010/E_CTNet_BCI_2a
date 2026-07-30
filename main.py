import os
import sys
import yaml
import argparse
import numpy as np
import torch

# Ensure UTF-8 stdout encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


from dataset import get_dataloaders
from train import train_subject_pipeline
from evaluate import compute_metrics, generate_markdown_table, generate_latex_table
from visualize import plot_confusion_matrix, plot_tsne_features, plot_training_curves


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="BCI Competition IV - Dataset 2a PyTorch Research Framework")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml file")
    parser.add_argument("--subjects", type=int, nargs="+", default=None, help="Subjects to train/evaluate (e.g. 1 2 ... 9). Default: all 9 subjects")
    parser.add_argument("--dry_run", action="store_true", help="Quick dry-run test mode (few epochs for code verification)")
    args = parser.parse_args()

    # 1. Load Configuration
    config = load_config(args.config)

    # Override epochs if dry_run flag is set
    if args.dry_run:
        print("\n[Dry Run Mode] Reducing epochs for rapid verification...")
        config['training']['stage1_pretrain']['epochs'] = 2
        config['training']['stage2_finetune']['epochs'] = 2

    # Set device
    device_str = config['training'].get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    print(f"[Hardware Setup] Active Computation Device: {device}")

    # Set random seed
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Directories
    output_dir = config['paths']['output_dir']
    figures_dir = config['paths']['figures_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    subjects_to_run = args.subjects or list(range(1, config['dataset']['num_subjects'] + 1))
    class_names = config['dataset']['class_labels']
    model_name = config['models'].get('active_model', 'EEGConformerATCNet')

    print(f"\n==========================================================")
    print(f" Starting BCI Competition IV - Dataset 2a Benchmark Execution")
    print(f" Model Architecture: {model_name}")
    print(f" Target Subjects   : {subjects_to_run}")
    print(f" Output Directory  : {output_dir}")
    print(f" Figures Directory : {figures_dir}")
    print(f"==========================================================\n")

    subject_results = {}
    all_y_true = []
    all_y_pred = []
    last_subject_data = None

    for sub_id in subjects_to_run:
        print(f"\n>>>> Starting Pipeline for Subject {sub_id}/{len(subjects_to_run)} <<<<")
        
        # Run 2-Stage Training Pipeline
        model, history, best_acc, preds, targets = train_subject_pipeline(config, sub_id, device)

        # Compute Metrics
        acc, kappa, cm = compute_metrics(targets, preds)
        subject_results[sub_id] = {
            'accuracy': acc,
            'kappa': kappa,
            'confusion_matrix': cm
        }

        all_y_true.extend(targets)
        all_y_pred.extend(preds)

        print(f"[Subject {sub_id} Summary] Accuracy: {acc:.2f}% | Cohen's Kappa (kappa): {kappa:.4f}")

        # 1. Save Training/Loss Curves
        curve_path = os.path.join(figures_dir, f"training_curves_S{sub_id}.png")
        plot_training_curves(history, sub_id, save_path=curve_path)

        # 2. Save Subject Confusion Matrix
        cm_path = os.path.join(figures_dir, f"confusion_matrix_S{sub_id}.png")
        plot_confusion_matrix(targets, preds, class_names, title=f"Subject {sub_id} Confusion Matrix", save_path=cm_path)

        # Save last subject data loaders for t-SNE plot
        _, test_loader = get_dataloaders(config, target_subject=sub_id, stage='fine_tune')
        last_subject_data = (model, test_loader)

    # 3. Generate Aggregate Confusion Matrix
    print(f"\n[Visualization] Generating aggregated 300 DPI confusion matrix across subjects...")
    agg_cm_path = os.path.join(figures_dir, "confusion_matrix_aggregate.png")
    plot_confusion_matrix(
        all_y_true, all_y_pred, class_names,
        title=f"Aggregated 4-Class Confusion Matrix ({model_name})", save_path=agg_cm_path
    )

    # 4. Generate Latent Space t-SNE Visualization
    if last_subject_data is not None:
        last_model, last_loader = last_subject_data
        tsne_path = os.path.join(figures_dir, "tsne_features.png")
        print(f"[Visualization] Computing t-SNE latent feature manifold...")
        plot_tsne_features(
            last_model, last_loader, class_names, device,
            title=f"Latent Bottleneck Feature Manifold ({model_name})", save_path=tsne_path
        )

    # 5. Export Publication Tables (Markdown & LaTeX)
    md_path = os.path.join(output_dir, "results_table.md")
    tex_path = os.path.join(output_dir, "results_table.tex")
    
    md_str = generate_markdown_table(subject_results, output_file=md_path)
    tex_str = generate_latex_table(subject_results, model_name=model_name, output_file=tex_path)

    print(f"\n==========================================================")
    print(f" Execution Completed Successfully!")
    print(f" Summary of Results:")
    print(f"==========================================================")
    print(md_str)


if __name__ == "__main__":
    main()
