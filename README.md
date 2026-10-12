# Automated Anomaly Detection in 5G Network Logs Using Natural Language Processing

**Module:** COS700 Research Project  
**Author:** Hendré Beyer (u26846188)  
**Supervisor:** Sheunesu Makura

## Overview

Forensic analysis of 5G core logs still relies heavily on keyword filtering, which misses abnormal
events that don't contain alarm words. This project is a lightweight, unsupervised Python prototype
that learns what normal 5G core logs look like and flags log lines that don't fit, using TF-IDF
feature extraction and K-Means clustering. It is benchmarked against traditional keyword filtering on
Accuracy, Precision, Recall, F1-score and runtime.

## Objectives

- Use TF-IDF to turn semi-structured 5G core logs into feature vectors.
- Use unsupervised clustering (K-Means) to model normal network behaviour and flag structural deviations.
- Benchmark the prototype against traditional keyword filtering on detection quality and computational efficiency.

## How it works

1. **Preprocessing:** regex replaces the values that change on every line (timestamps, IMSIs, IP addresses,
   hex codes, IDs) with placeholders, so each line becomes a stable *template*.
2. **Feature extraction:** TF-IDF (unigrams and bigrams) turns each template into a vector.
3. **Clustering:** K-Means (k = 15) groups similar lines. In the main mode the clusters are learned from
   normal traffic only.
4. **Detection:** each line is scored by its distance to its cluster centre. Lines further out than a set
   cut-off are flagged as anomalies, and a SOC-style report is produced.

## Data

The logs were generated in an isolated lab: the OpenAirInterface 5G core (AMF, SMF, UPF) running in Docker
in an Ubuntu VM, with UERANSIM simulating the gNodeB and UEs. There is one normal capture and one attack
capture (an invalid-IMSI registration storm), with the attack start and stop times recorded in
`attack_timeline.log`. The log files are not in this repository; see [data/README.md](data/README.md).

## Repository structure

```
COS700-5G-Log-Anomaly-Detection/
├── main.py                      # Runs the detection pipeline and writes the SOC report
├── src/
│   ├── preprocessing/log_parser.py      # Loads logs, regex cleaning into templates
│   ├── modeling/anomaly_detector.py     # TF-IDF, K-Means, centroid-distance scoring
│   ├── reporting/summary_report.py      # SOC-style text report
│   └── evaluation/
│       ├── keyword_baseline.py          # Keyword-filtering baseline
│       ├── ground_truth.py              # Attack labels (from the attack timeline)
│       └── benchmark.py                 # Accuracy, Precision, Recall, F1, runtime, sweep
├── scripts/
│   ├── run_benchmark.py                 # Prototype vs keyword filtering (+ threshold sweep)
│   ├── rate_diagnostic.py               # Log volume over time vs model score (figure)
│   ├── run_combined_attack_logs.py      # Pipeline on the three attack logs combined
│   └── build_ground_truth.py            # Optional template-level label file
├── data/                        # Log files (not in git, see data/README.md)
├── results/                     # Generated output (not in git)
├── HOW_TO_RUN.md                # Setup and all commands
└── requirements.txt
```

## Running it

See **[HOW_TO_RUN.md](HOW_TO_RUN.md)** for setup and every command. Quick start (Windows PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

python main.py --log amf_attack_logs.txt --train-on-normal   # detect + SOC report
python scripts/run_benchmark.py                              # benchmark vs keyword filtering
```

## Tech stack

Python 3.10+ with pandas, scikit-learn and matplotlib (matplotlib is only used for the rate diagnostic figure).
