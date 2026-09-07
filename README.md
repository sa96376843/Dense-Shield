# Dense Shield

Official research artifacts for **DenseShield: A Density-Centric Unified Defense against Evasion, Backdoors, and Concept Drift in Malware Detection**.

Repository: <https://github.com/sa96376843/Dense-Shield>

DenseShield combines sparse feature compression (SFC), density-boosted adversarial training (DBAT), density-guided post-training purification (D-PUR), and drift-adaptive minimal-label adaptation (DMLA) for feature-based malware detectors.

## Repository contents

```text
code/
  denseshield.py          Core methods, models, attacks, and evaluation utilities
  run_experiments.py      Main experiments and reported metrics
  generate_figures.py     Code-generated empirical plots
  ember_proc.py           EMBER 2018 JSONL preprocessing utility
  results_summary.json    Cached numerical results used by the plotting script
dataset/
  drebin215.csv           DREBIN-215 feature table
  ClaMP_Integrated-5184.csv
                          ClaMP-5184 feature table
```

The empirical plots are generated from numerical experiment outputs.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the experiments

Run commands from the `code` directory so the relative dataset paths resolve correctly.

```bash
cd code
python run_experiments.py
python generate_figures.py
```

`run_experiments.py` reads the two included feature tables from `../dataset/` and writes `results_summary.json`. The EMBER 2018 raw JSONL files are not redistributed here. After obtaining the public release, place it under `dataset/ember2018/` and run `python ember_proc.py`, or provide explicit `--ember-dir` and `--output-dir` paths.

## Data provenance

- DREBIN-215 feature table: [Malware-Hunter/SBSeg23-XAI](https://github.com/Malware-Hunter/SBSeg23-XAI)
- ClaMP-5184 feature table: [urwithajit9/ClaMP](https://github.com/urwithajit9/ClaMP)
- EMBER 2018: [Elastic EMBER](https://github.com/elastic/ember)

The included CSV files contain extracted feature vectors and labels, not malware binaries. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for attribution and reuse notes.

## License and citation

No blanket license is asserted over third-party datasets. Their upstream terms and attribution requirements continue to apply. The authors' code is shared for research reproducibility; a formal code license can be added by the authors if broader reuse rights are intended.

If you use these artifacts, please cite the DenseShield paper. Final bibliographic metadata will be added after publication.
