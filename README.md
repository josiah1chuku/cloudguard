# CloudGuard

**A Multi-Stage Machine Learning Pipeline for Detecting Azure Policy Governance Gaps and Deployment-Triggered Vulnerability Emergence in Cloud Security Environments**

[![IEEE HiPC 2026](https://img.shields.io/badge/IEEE-HiPC%202026-blue)](https://hipc.org)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![scikit-learn 1.9.0](https://img.shields.io/badge/scikit--learn-1.9.0-orange)](https://scikit-learn.org)
[![XGBoost 3.2.0](https://img.shields.io/badge/XGBoost-3.2.0-red)](https://xgboost.readthedocs.io)
[![PyTorch 2.12.0](https://img.shields.io/badge/PyTorch-2.12.0-EE4C2C)](https://pytorch.org)

> Josiah Chuku · Dr. Liu Jinwei  
> Florida Agricultural and Mechanical University  
> Submitted to IEEE HiPC 2026 — Bengaluru, India · December 16–19, 2026

---

## Overview

CloudGuard is a three-stage machine learning pipeline that systematically detects Azure Policy governance gaps using exclusively public cloud security datasets. No organizational telemetry is required.

Cloud misconfiguration is the leading preventable cause of enterprise security incidents (IBM X-Force 2024: 15% of all breaches, average cost $4.88M). Microsoft Azure Policy's Deny-mode enforcement is the primary prevention mechanism — but organizational deployments routinely leave policy assignments in Audit-only mode, creating governance vacuums that persist undetected until a breach occurs.

CloudGuard addresses this through three sequential stages:

| Stage | Model | Task | F1 | ROC-AUC |
|-------|-------|------|----|---------|
| 1 | Isolation Forest | Unsupervised policy gap screening | 0.8103 | 0.9684 |
| 2 | Random Forest + SMOTE | Supervised compliance classification | 0.8688 | 0.9785 |
| 3 | CNN-LSTM | Temporal compliance drift detection | 0.6301 | 0.8952 |

---

## Novel Contributions

### Two novel features
- **Policy Coverage Ratio (PCR)** — quantifies the proportion of resource types with active Deny-mode policy coverage. PCR = 0 indicates a governance vacuum; PCR = 1 indicates full Deny-mode coverage.
- **Deployment-to-Vulnerability Lag (DVL)** — measures elapsed time between resource deployment and first associated vulnerability detection. DVL ≈ 0 confirms resources were deployed into ungoverned environments.

### Feature importance findings
`enforcement_mode` (0.3733) and PCR (0.3590) together account for **73.2%** of predictive signal — confirming that Deny-mode enforcement presence is the single most diagnostic indicator of governance health.

| Feature | MDI Importance | Rank |
|---------|---------------|------|
| enforcement_mode | 0.3733 | 1 |
| PCR | 0.3590 | 2 |
| DVL | 0.1275 | 3 |
| vuln_count_30d | 0.1042 | 4 |
| policy_age_days | 0.0259 | 5 |
| scope_level | 0.0101 | 6 |
| resource_type_flag | 0.0000 | 7 — excluded |

### Remediation framework
A four-tier remediation framework maps each misconfiguration to a specific Azure built-in policy definition ID, raising PCR from 0.00 to 0.83 using 8 policy assignments (Table XI in paper).

---

## Repository Structure

```
cloudguard/
├── src/
│   ├── data/
│   │   └── download_datasets.py      # Downloads Azure Policy, CloudSploit, NVD
│   ├── features/
│   │   └── build_features.py         # Computes PCR, DVL, all 7 features
│   ├── models/
│   │   ├── stage1_isolation_forest.py  # Stage 1: iForest training + evaluation
│   │   └── stage2_classifier.py        # Stage 2: XGBoost + RF with SMOTE
│   ├── evaluation/
│   └── utils/
│       └── helpers.py                # Logging, file checks, dataset summaries
├── notebooks/
│   └── stage3_cnn_lstm.ipynb         # Stage 3: CNN-LSTM (run on Google Colab T4)
├── data/
│   ├── raw/                          # Downloaded source data (not tracked in git)
│   ├── processed/                    # cloudguard_dataset.csv (6,674 records)
│   └── splits/                       # train.csv (5,672) / test.csv (1,002)
├── outputs/
│   ├── models/                       # Trained model files (.pkl, .pt)
│   ├── results/                      # Metrics JSON files
│   └── figures/                      # Confusion matrices, feature importance plots
├── paper/
│   ├── main.tex                      # IEEE HiPC 2026 LaTeX submission
│   └── references.bib                # BibTeX references (27 peer-reviewed sources)
├── config.py                         # All hyperparameters and file paths
├── run_pipeline.py                   # End-to-end pipeline runner
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Dataset

Built entirely from public sources — no organizational telemetry required:

| Source | Records | Role |
|--------|---------|------|
| [Azure Policy GitHub](https://github.com/Azure/azure-policy) | 5,108 built-in definitions | Feature extraction |
| [CloudSploit](https://github.com/aquasecurity/cloudsploit) | 976 Azure plugins | Label generation |
| [NIST NVD](https://nvd.nist.gov) | CVE records 2019–2024 | DVL computation |

500 Azure subscription deployments are simulated across three governance profiles:
- Well-governed (25%): 70–100% Deny enforcement
- Partially governed (40%): 30–70% Deny enforcement
- Poorly governed (35%): 0–15% Deny enforcement

Final dataset: **6,674 records**, 85% compliant / 15% non-compliant (85/15 stratified split).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cloudguard.git
cd cloudguard

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# Activate (Git Bash / Linux / Mac)
source venv/Scripts/activate

# Install dependencies
pip install --prefer-binary -r requirements.txt

# Install PyTorch (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Running the Pipeline

### Full pipeline
```bash
python run_pipeline.py
```

### Individual stages
```bash
# Stage 0: Download datasets (~90 minutes, downloads 5,108 policies + 976 plugins)
python run_pipeline.py --stage download

# Stage 0.5: Feature engineering (builds 6,674-record dataset)
python src/features/build_features.py

# Stage 1: Isolation Forest
python src/models/stage1_isolation_forest.py

# Stage 2: XGBoost + Random Forest with SMOTE
python src/models/stage2_classifier.py

# Stage 3: CNN-LSTM (Google Colab — requires T4 GPU)
# Upload notebooks/stage3_cnn_lstm.ipynb to colab.research.google.com
# Runtime → Change runtime type → T4 GPU → Run all
```

---

## Experimental Results

### Stage 1 — Isolation Forest (Table V)

| Model | F1 | Recall | Precision | ROC-AUC |
|-------|-----|--------|-----------|---------|
| **iForest + PCR/DVL** | **0.8103** | **0.8400** | **0.7826** | **0.9684** |
| iForest (no PCR/DVL) | 0.651 | 0.693 | 0.614 | 0.771 |
| One-Class SVM | 0.612 | 0.543 | 0.700 | 0.741 |
| Local Outlier Factor | 0.589 | 0.571 | 0.608 | 0.718 |

### Stage 2 — Supervised Classification (Table VI)

| Model | F1 | Recall | Precision | MCC |
|-------|-----|--------|-----------|-----|
| **RF + SMOTE** | **0.8688** | **0.9267** | **0.8176** | **0.8463** |
| XGBoost + SMOTE | 0.8527 | 0.9067 | 0.8047 | 0.8270 |
| XGBoost (no SMOTE) | 0.791 | 0.803 | 0.779 | 0.751 |
| Logistic Regression | 0.691 | 0.711 | 0.672 | 0.624 |

### Stage 3 — CNN-LSTM Temporal Drift (Table VIII)

| Model | F1 | ROC-AUC | Notes |
|-------|-----|---------|-------|
| **CNN-LSTM** | **0.6301** | **0.8952** | 63 epochs, T4 GPU |
| MLP baseline | 0.2694 | 0.5961 | Non-temporal ablation |
| **Improvement** | **+36.1pp** | **+29.9pp** | Temporal modeling value |

---

## Remediation Framework (Table XI)

CloudGuard closes the detect-to-fix loop with a four-tier remediation framework ranked by MDI feature importance, and a **three-path closed-loop pipeline** for existing resources:

**Path 1 — Azure Policy remediation tasks (fastest, zero downtime)**
```bash
az policy remediation create \
  --name "CloudGuard-Remediation" \
  --policy-assignment "CloudGuard-SecurityBenchmark" \
  --resource-discovery-mode ReEvaluateCompliance
```

**Path 2 — Azure CLI bulk configuration updates**
```bash
# Fix Storage Accounts (PCR +0.08)
az storage account update \
  --default-action Deny \
  --allow-blob-public-access false \
  --https-only true \
  --min-tls-version TLS1_2

# Fix Key Vaults (PCR +0.05)
az keyvault update \
  --enable-soft-delete true \
  --enable-purge-protection true
```

**Path 3 — IaC import-and-fix (ongoing governance)**
```bash
# Import existing resources into Terraform
aztfexport resource-group <rg-name> --output-dir ./cloudguard-import/

# Apply CloudGuard secure defaults ranked by MDI importance
# Then apply to bring existing resources into compliance
terraform apply -auto-approve
```

All three paths are applied in order. All remediation priority is determined by the Stage 2 Random Forest MDI rankings (Table VII).

---

## CI/CD Integration

Block pull requests that reduce PCR below 0.80:

```yaml
# .github/workflows/cloudguard.yml
name: CloudGuard IaC Security Gate
on:
  pull_request:
    paths: ['**.tf', '**.bicep']
jobs:
  cloudguard-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: CloudGuard PCR check
        run: |
          pip install cloudguard-cli
          cloudguard scan --path . --pcr-threshold 0.80 --fail-on-reduce
```

---

## Paper

The full IEEE HiPC 2026 paper is available in `paper/`:
- `paper/main.tex` — IEEEtran LaTeX source
- `paper/references.bib` — 27 peer-reviewed BibTeX references

To compile locally:
```bash
pdflatex paper/main.tex
bibtex main
pdflatex paper/main.tex
pdflatex paper/main.tex
```

Or upload to [Overleaf](https://overleaf.com) — New Project → Upload Project → select `paper/`.

---

## Configuration

All hyperparameters are in `config.py`:

```python
# Stage 1 — Isolation Forest
IFOREST_N_ESTIMATORS = 200
IFOREST_CONTAMINATION = 0.15    # Expected misconfiguration rate (IBM X-Force 2024)
IFOREST_MAX_SAMPLES = 256

# Stage 2 — Supervised Classification
XGBOOST_CONFIG = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'scale_pos_weight': 6.0,    # Handles 85/15 class imbalance
}
RANDOM_FOREST_CONFIG = {
    'n_estimators': 500,
    'max_features': 'sqrt',
    'class_weight': 'balanced_subsample',
}

# Stage 3 — CNN-LSTM
SEQUENCE_LENGTH = 30            # 30-day compliance windows
PREDICTION_HORIZON = 7          # 7-day ahead prediction
LSTM_UNITS_1 = 128
CONV_FILTERS = 64
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001
```

---

## Requirements

```
Python >= 3.14
scikit-learn >= 1.9.0
xgboost >= 3.2.0
imbalanced-learn >= 0.14.2
torch >= 2.12.0
pandas >= 3.0.3
numpy >= 2.4.6
matplotlib >= 3.11.0
seaborn >= 0.13.2
requests >= 2.34.2
tqdm >= 4.68.2
python-dotenv >= 1.2.2
```

---

## Citation

If you use CloudGuard in your research, please cite:

```bibtex
@inproceedings{chuku2026cloudguard,
  title     = {CloudGuard: A Multi-Stage Machine Learning Pipeline for Detecting 
               Azure Policy Governance Gaps and Deployment-Triggered Vulnerability 
               Emergence in Cloud Security Environments},
  author    = {Chuku, Josiah and Jinwei, Liu},
  booktitle = {Proceedings of the 33rd IEEE International Conference on 
               High Performance Computing, Data, and Analytics (HiPC)},
  year      = {2026},
  address   = {Bengaluru, India},
  publisher = {IEEE}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Datasets are publicly available from:
- [Microsoft Azure Policy GitHub](https://github.com/Azure/azure-policy)
- [Aqua Security CloudSploit](https://github.com/aquasecurity/cloudsploit)
- [NIST National Vulnerability Database](https://nvd.nist.gov)

No organizational telemetry was used in this research.
