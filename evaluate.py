import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


def compute_metrics(y_true, y_pred):
    """
    Computes classification performance metrics:
    - Top-1 Accuracy (%)
    - Cohen's Kappa (kappa)
    - Confusion Matrix (4x4)
    """
    acc = accuracy_score(y_true, y_pred) * 100.0
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    return acc, kappa, cm


def generate_markdown_table(subject_results, output_file=None):
    """
    Generates a cleanly formatted Markdown summary table of per-subject and aggregate results.
    """
    md = []
    md.append("# BCI Competition IV - Dataset 2a Experimental Results Summary\n")
    md.append("| Subject | Train Session (S1) | Test Session (S2) | Accuracy (%) | Cohen's Kappa (κ) |")
    md.append("| :---: | :---: | :---: | :---: | :---: |")

    accs = []
    kappas = []

    for sub_id in sorted(subject_results.keys()):
        res = subject_results[sub_id]
        acc = res['accuracy']
        kappa = res['kappa']
        accs.append(acc)
        kappas.append(kappa)
        md.append(f"| **Subject {sub_id}** | 288 trials | 288 trials | **{acc:.2f}%** | **{kappa:.4f}** |")

    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_kappa, std_kappa = np.mean(kappas), np.std(kappas)

    md.append("| :--- | :--- | :--- | :--- | :--- |")
    md.append(f"| **Mean ± Std** | - | - | **{mean_acc:.2f}% ± {std_acc:.2f}%** | **{mean_kappa:.4f} ± {std_kappa:.4f}** |\n")

    md_str = "\n".join(md)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_str)
        print(f"[Results] Markdown summary table saved to: {output_file}")

    return md_str


def generate_latex_table(subject_results, model_name="EEGConformerATCNet", output_file=None):
    """
    Generates publication-ready IEEE/Springer LaTeX code for results table.
    """
    tex = []
    tex.append("% Publication Table for IEEE/Springer BCI Journal Submissions")
    tex.append("\\begin{table}[htbp]")
    tex.append("  \\centering")
    tex.append("  \\caption{Classification Performance of " + model_name + " on BCI Competition IV Dataset 2a (4-Class Motor Imagery).}")
    tex.append("  \\label{tab:bci_results}")
    tex.append("  \\begin{tabular}{ccccc}")
    tex.append("    \\toprule")
    tex.append("    \\textbf{Subject} & \\textbf{Train Trials} & \\textbf{Test Trials} & \\textbf{Accuracy (\\%)} & \\textbf{Cohen's Kappa ($\\kappa$)} \\\\")
    tex.append("    \\midrule")

    accs = []
    kappas = []

    for sub_id in sorted(subject_results.keys()):
        res = subject_results[sub_id]
        acc = res['accuracy']
        kappa = res['kappa']
        accs.append(acc)
        kappas.append(kappa)
        tex.append(f"    Subject {sub_id} & 288 & 288 & {acc:.2f} & {kappa:.4f} \\\\")

    mean_acc, std_acc = np.mean(accs), np.std(accs)
    mean_kappa, std_kappa = np.mean(kappas), np.std(kappas)

    tex.append("    \\midrule")
    tex.append(f"    \\textbf{{Mean $\\pm$ Std}} & -- & -- & \\textbf{{{mean_acc:.2f} $\\pm$ {std_acc:.2f}}} & \\textbf{{{mean_kappa:.4f} $\\pm$ {std_kappa:.4f}}} \\\\")
    tex.append("    \\bottomrule")
    tex.append("  \\end{tabular}")
    tex.append("\\end{table}")

    tex_str = "\n".join(tex)

    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(tex_str)
        print(f"[Results] LaTeX publication table saved to: {output_file}")

    return tex_str
