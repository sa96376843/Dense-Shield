# Data provenance and reuse notice

This repository redistributes two compact, tabular feature datasets solely to support reproducibility of the DenseShield experiments. They do not contain malware binaries.

## DREBIN-215

- File: `dataset/drebin215.csv`
- Upstream repository: <https://github.com/Malware-Hunter/SBSeg23-XAI>
- Upstream artifact: `drebin215.csv.zip`

## ClaMP-5184

- File: `dataset/ClaMP_Integrated-5184.csv`
- Upstream repository: <https://github.com/urwithajit9/ClaMP>
- Upstream artifact: `dataset/ClaMP_Integrated-5184.csv`

## EMBER 2018

EMBER 2018 is referenced by the experiments but is not included in this repository. Obtain it from the official Elastic EMBER project and process its JSONL files with `code/ember_proc.py`.

## Reuse terms

The upstream repositories did not present an explicit dataset license in the materials inspected when this package was prepared. Consequently, this repository does not apply a new license to these third-party data files or imply ownership of them. Users must verify and comply with the upstream terms and provide the original attribution. If an upstream maintainer requests a change in redistribution, replace the corresponding CSV with a documented download/preprocessing step.

