# How to Run the Prototype

All commands are for **Windows PowerShell** and are run from the project folder (the one containing `main.py`).

---

## 1. One-time setup

You need **Python 3.10 or newer** (developed on 3.11).

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, allow local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Activate the venv every time you open a new terminal (you'll see `(venv)` at the start of the prompt).
If the box-drawing characters in the output look broken, also run:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

## 2. Put the log files in `data/`

The logs are not in git. Copy them into `data/` so it contains:

| File | What it is |
|------|------------|
| `amf_normal_logs.txt` | Normal AMF capture |
| `smf_normal_logs.txt` | Normal SMF capture |
| `upf_normal_logs.txt` | Normal UPF capture |
| `amf_attack_logs.txt` | AMF captured during the attack |
| `smf_attack_logs.txt` | SMF captured during the attack |
| `upf_attack_logs.txt` | UPF captured during the attack |
| `attack_timeline.log` | Attack start/stop times (needed for the benchmark) |

File names matter:
- the network function (AMF/SMF/UPF) is read from the name, so keep `amf`, `smf` or `upf` in it;
- files ending in `_attack_logs.txt` are attack captures; every other `.txt` counts as normal traffic;
- only `.txt` files in `data/` itself are loaded. Subfolders (e.g. older captures) are ignored.

## 3. Run the detection pipeline

```powershell
python main.py --log amf_attack_logs.txt --train-on-normal   # main mode: learn normal first, then score the attack log
python main.py --log amf_attack_logs.txt                     # fit the model on the attack log itself
python main.py                                               # default file: amf_normal_logs.txt
python main.py --log all                                     # every .txt in data/ combined (normal + attack)
python scripts/run_combined_attack_logs.py                   # only the three attack logs combined
```

Options for `main.py`:

| Option | Default | Meaning |
|--------|---------|---------|
| `--log FILE` | `amf_normal_logs.txt` | Log file in `data/`, or `all` |
| `--train-on-normal` | off | Learn "normal" from the normal log(s) of the same network function first, then score `--log` against it. Use it on attack logs. |
| `--clusters N` | 15 | Number of K-Means clusters |
| `--outlier-percent P` | 1.0 | The cut-off. Without `--train-on-normal`: flag the furthest P% of events. With it: flag events further out than all but P% of the normal logs. |
| `--auto-clusters` | off | Choose k (5–20) automatically with the silhouette score |

Each run writes four files to `results/<name>/`, where `<name>` is the log file name (plus `_vs_normal`
with `--train-on-normal`), or `combined_5g_core` / `combined_attack_logs`:

| File | Contents |
|------|----------|
| `<name>_summary.txt` | The SOC-style report (also printed in the terminal) |
| `<name>_anomalies.csv` | Only the flagged events, with distance and proposed threat category |
| `<name>_scored_events.csv` | Every event with its cluster and distance |
| `<name>_cluster_distribution.csv` | How many events landed in each cluster |

Everything in `results/` can be regenerated, so it's safe to delete.

## 4. Run the benchmark against keyword filtering

Compares three methods on the attack logs (AMF, SMF, UPF and all three combined):

1. keyword filtering (a grep for words like `error`, `fail`, `reject`, `warning`);
2. the TF-IDF model fitted on the attack log itself;
3. the TF-IDF model trained on the normal logs first.

```powershell
python scripts/run_benchmark.py                          # cut-off 1%
python scripts/run_benchmark.py --outlier-percent 5      # another cut-off
python scripts/run_benchmark.py --sweep                  # all cut-offs from 0.5% to 40%
```

The attack labels come from `data/attack_timeline.log`: a line counts as attack if it was logged during the
recorded attack window **and** its template mostly occurs inside that window (this keeps background
heartbeats out). No manual labelling is needed.

Output in `results/benchmark/`:

| File | Contents |
|------|----------|
| `benchmark_report.txt` | Accuracy, Precision, Recall, F1, runtime and Average Precision per dataset |
| `benchmark_comparison.csv` | The same numbers as a table |
| `benchmark_<dataset>_events.csv` | Every line with its label and each method's flag and score |
| `benchmark_sweep_report.txt` | (`--sweep`) every method at every cut-off |
| `benchmark_sweep.csv` | (`--sweep`) the same as a table |
| `benchmark_pr_curve.csv` | (`--sweep`) full precision-recall curve per model, for plotting |

Other options: `--clusters N`, `--repeats N` (timing repeats, median reported; default 5).
A normal run takes about 20 seconds and `--sweep` about 30 seconds on a laptop.

### Optional: template-level labels

The benchmark can also use one label per template instead of the timeline:

```powershell
python scripts/build_ground_truth.py                 # creates data/ground_truth_labels.csv
python scripts/run_benchmark.py --labels templates
```

Each row gets a suggested label; type `1` or `0` in the `label` column to override it. Until every row is
labelled by hand the report is marked PROVISIONAL. Re-running `build_ground_truth.py` keeps labels already typed.

## 5. Rate diagnostic

```powershell
python scripts/rate_diagnostic.py
```

Counts how often each log template appears per 10 seconds in the attack capture, and plots that next to
the model's score for every AMF line. Output in `results/rate_diagnostic/`:

| File | Contents |
|------|----------|
| `rate_vs_wording.png` | The figure: log volume over time vs the model's scores |
| `template_rate_over_time.csv` | Count per template per 10 s bucket |
| `amf_template_rate_vs_score.csv` | Per AMF template: normal rate, attack-window rate and model score |

Use `--bucket-seconds N` to change the bucket width.

## 6. Where things live

| Path | Role |
|------|------|
| `main.py` | Entry point for the detection pipeline |
| `src/preprocessing/log_parser.py` | Loads logs and turns lines into templates with regex |
| `src/modeling/anomaly_detector.py` | TF-IDF, K-Means, centroid-distance scoring, threat labels |
| `src/reporting/summary_report.py` | The SOC-style text report |
| `src/evaluation/keyword_baseline.py` | The keyword list (`ALERT_KEYWORDS`) and filter |
| `src/evaluation/ground_truth.py` | Attack labels from the timeline (and the optional template labels) |
| `src/evaluation/benchmark.py` | Metrics, runtime, sweep and report tables |
| `scripts/` | Benchmark, rate diagnostic, combined attack run, label file builder |

## Troubleshooting

- **`ModuleNotFoundError: No module named 'src'`**: run commands from the project folder, not from inside `src/` or `scripts/`.
- **`ModuleNotFoundError: pandas / sklearn / matplotlib`**: the venv isn't active, or run `pip install -r requirements.txt` again.
- **`Log file not found`**: check the file is in `data/` and spelled exactly. The error lists the files it found.
- **`No ATTACK_*_START / ATTACK_*_STOP pairs found`**: `data/attack_timeline.log` is missing or has no attack marks.
- **`... templates that aren't in the label file`** (only with `--labels templates`): re-run `python scripts/build_ground_truth.py`.
