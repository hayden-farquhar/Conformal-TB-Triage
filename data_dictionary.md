# Data Dictionary

## splits.parquet

The split manifest defining which images belong to which split and their labels.
Located at `data/processed/splits.parquet`.

| Variable | Type | Description |
|----------|------|-------------|
| `patient_id` | string | Unique patient/image identifier |
| `filename` | string | Original image filename |
| `file_path` | string | Original file path (relative to dataset root) |
| `dataset` | string | Source dataset: `shenzhen`, `montgomery`, `tbx11k`, `pakistan`, `chexpert` |
| `label` | string | Original label from source dataset (e.g., `active_tb`, `healthy`, `sick_non_tb`, `latent_tb`, `unknown`) |
| `tb_binary` | string | Harmonised binary label: `tb_positive`, `tb_negative`, or `unknown` |
| `split` | string | Data split: `calibration`, `dev`, `test`, `ext_pakistan`, `distractor` |
| `sex` | string | Patient sex (where available): `M`, `F`, or `unknown` |
| `age` | float | Patient age in years (where available; NaN if missing) |

**Split allocation:**

| Split | Datasets | n | Purpose |
|-------|----------|---|---------|
| `calibration` | Shenzhen + Montgomery | 800 | Conformal calibration + probe training |
| `dev` | TBX11K (30%) | 3,510 | Probe hyperparameter selection |
| `test` | TBX11K (70%) | 8,191 | Primary evaluation |
| `ext_pakistan` | Pakistan | 3,008 | External validation |
| `distractor` | CheXpert | 5,000 | Non-TB specificity evaluation |

**Label mapping:**

| tb_binary | Includes |
|-----------|----------|
| `tb_positive` | TBX11K: active_tb + latent_tb; Shenzhen/Montgomery/Pakistan: TB |
| `tb_negative` | TBX11K: healthy + sick_non_tb; Shenzhen/Montgomery/Pakistan: normal; CheXpert: all |
| `unknown` | TBX11K: unlabelled images (3,302) |

## Embedding parquet files

One file per foundation model (e.g., `rad_dino.parquet`). Available from Zenodo.

| Variable | Type | Description |
|----------|------|-------------|
| `patient_id` | string | Matches `splits.parquet` |
| `dataset` | string | Source dataset |
| `tb_binary` | string | Harmonised binary label |
| `split` | string | Data split |
| `emb_0` ... `emb_{d-1}` | float32 | Embedding dimensions (d varies by model) |

**Embedding dimensions:**

| Model | d | File |
|-------|---|------|
| RAD-DINO | 768 | `rad_dino.parquet` |
| BiomedCLIP | 512 | `biomedclip.parquet` |
| torchxrayvision | 1,024 | `torchxrayvision.parquet` |
| DINOv2-B | 768 | `dinov2.parquet` |

## Result CSV files

All result tables are in `outputs/tables/`. Each file corresponds to one or
more pre-registered analyses (referenced by section number from the OSF
pre-registration).

| File | Pre-registration section | Description |
|------|--------------------------|-------------|
| `conformal_results.csv` | §5.2.5 | APS, RAPS, Mondrian results for all embedding x probe combinations |
| `improved_conformal_results.csv` | §5.3.3, §5.4.20 | Enhanced pipeline with weighted CP and safeguards |
| `crc_results.csv` | §5.3.1 | Conformal Risk Control (direct FNR bounding) |
| `ltt_results.csv` | §5.3.2 | Learn Then Test (joint sensitivity + specificity control) |
| `who_tpp_alignment.csv` | §5.3.4 | WHO TPP operating characteristics at multiple prevalences |
| `commercial_comparison.csv` | §5.3.5 | Comparison to published CAD4TB, qXR, Lunit performance |
| `triage_results.csv` | §5.3.7 | Three-tier triage allocation by model and method |
| `decision_curves.csv` | §5.3.8 | Net benefit at threshold probabilities 0.01-0.50 |
| `referral_cascade.csv` | §5.3.10 | Xpert utilisation modelling across 4 scenarios and 6 prevalences |
| `recalibration_lodo.csv` | §5.3.11 | Leave-one-dataset-out recalibration simulation |
| `computational_cost.csv` | §5.3.13 | Model size, embedding dimensions, storage footprint |
| `bootstrap_cis.csv` | §5.4.8 | 95% bootstrap CIs for primary metrics |
| `calset_sensitivity.csv` | §5.4.1 | Coverage vs calibration set size |
| `image_degradation.csv` | §5.4.2 | AUROC and coverage under resolution/JPEG/noise/brightness degradation |
| `tta_results.csv` | §5.4.3 | Test-time augmentation comparison |
| `dimensionality_reduction.csv` | §5.4.4 | PCA dimensionality reduction impact |
| `probe_sensitivity.csv` | §5.4.5 | Probe regularisation sensitivity |
| `raps_sensitivity.csv` | §5.4.6 | RAPS lambda hyperparameter sweep |
| `multiple_testing.csv` | §5.4.9 | Bonferroni and BH-corrected comparisons |
| `seed_stability.csv` | §5.4.10 | Results across 5 random seeds |
| `meta_coverage.csv` | §5.4.11 | TB coverage across 200 resplits |
| `label_noise.csv` | §5.4.12 | Coverage under label corruption |
| `latent_tb_exclusion.csv` | §5.4.13 | Impact of latent TB label definition |
| `prevalence_mismatch.csv` | §5.4.14 | Coverage vs calibration TB prevalence |
| `geographic_gap.csv` | §5.4.15 | WHO high-burden country representation |
| `lung_segmentation.csv` | §5.4.17 | Whole-image vs lung-only vs mediastinal-only |
| `shortcut_detection.csv` | §5.4.18 | Dataset-origin vs TB classification AUROC |
| `nontb_confusion.csv` | §5.4.19 | False positive rate by non-TB category |
| `clinical_futility.csv` | §5.4.21 | Singleton fraction across alpha levels |
| `fused_embeddings.csv` | §5.2.2, H6 | Fused embedding discrimination results |
| `calibration_metrics.csv` | §5.2.3 | ECE, MCE, Brier before/after calibration |
| `calibration_bins.csv` | §5.2.3 | Reliability diagram bin data |
| `reference_baselines.csv` | §5.2.4 | Random, prevalence, radiomics baselines |
| `efficiency_curves.csv` | §5.2.5E | Coverage vs set size across alpha grid |
| `roc_curves.csv` | §5.2.2, Fig 2 | ROC curve coordinates per model |
| `pr_curves.csv` | §5.2.2 | Precision-recall curve coordinates |
| `probe_ensemble.csv` | §5.5.5 | Probe ensemble comparison |
| `multiclass_analysis.csv` | §5.5.1 | Four-class conformal (TBX11K) |
| `selective_prediction.csv` | §5.5.2 | Abstention analysis |
| `adaptive_conformal.csv` | §5.5.3 | Adaptive conformal inference |
| `shap_feature_importance.csv` | §5.5.4 | Top-50 SHAP values for RAD-DINO XGBoost probe |
| `concordance.csv` | §5.5.6 | Inter-method agreement |
| `empty_sets.csv` | §5.5.7 | Empty prediction set analysis |
| `cal_conformal_interaction.csv` | §5.5.8 | Calibration method x conformal method interaction |
| `cross_model_disagreement.csv` | §5.5.9 | Inter-model prediction agreement |
| `drift_monitoring.csv` | §5.5.10 | MMD drift detection simulation |
| `cost_sensitive.csv` | §5.5.11 | Cost-sensitive conformal at varying FN/FP ratios |
| `venn_abers.csv` | §5.5.12 | Venn-ABERS probability intervals per patient |
| `cross_conformal.csv` | §5.5.13 | Cross-conformal (CV+) results |
| `lodo_simulation.csv` | §5.5.14 | Leave-one-dataset-out deployment simulation |
| `perturbation_stability.csv` | §5.5.15 | Prediction set stability under image perturbation |
| `tsne_rad_dino.csv` | §5.3.6, sFig 2 | t-SNE coordinates for RAD-DINO embeddings |
| `tsne_biomedclip.csv` | §5.3.6, sFig 2 | t-SNE coordinates for BiomedCLIP embeddings |
