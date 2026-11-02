# AI-Assisted Cyber & IoT Threat-Detection and Contextual Risk-Prioritization System

An end-to-end machine learning pipeline and interactive Security Operations Center (SOC) triage web dashboard that detects known and unseen zero-day cyber attacks from structured network and IoT telemetry while suppressing false positives through contextual risk scoring and human-in-the-loop active learning.

## Key Empirical Results (3,600 Held-Out Chronological Test Flows)

| Pipeline Stage | Overall Accuracy | Precision | Overall Recall | F1-Score | Known Recall | Zero-Day Recall | False Positives | Total Alerts (P1 Queue) | Latency (ms/flow) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **1. Deterministic Baseline** ([Sigma](https://sigmahq.io/) / [YARA](https://virustotal.github.io/yara/) + $Z > 1.75$) | 88.00% | 40.3% | 62.3% | 48.9% | 100.0% | 10.7% | 307 | 514 | 0.0012 ms | 
| **2. Supervised Only** ([XGBoost](https://xgboost.readthedocs.io/) / [LightGBM](https://lightgbm.readthedocs.io/)) | 95.97% | 96.5% | 58.4% | 72.8% | 100.0% | 1.4% | 7 | 201 | 0.0012 ms | 
| **3. Unsupervised Only** (Subspace AE / [Isolation Forest](https://ieeexplore.ieee.org/document/4781136)) | 93.86% | 60.0% | 100.0% | 75.0% | 100.0% | 100.0% | 221 | 553 | 0.0017 ms | 
| **4. Raw Hybrid Ensemble** (Supervised + Unsupervised) | 95.31% | 66.3% | 100.0% | 79.7% | 100.0% | 100.0% | 169 | 501 | 0.0019 ms | 
| **5. Context-Prioritized AI (Proposed)** | **99.72%** | **97.1%** | **100.0%** | **98.5%** | **100.0%** | **100.0%** | **10** | **342 (111 P1)** | **0.0020 ms** | 

* **96.7% False-Positive Reduction**: Suppresses false alarms from `307` (deterministic baseline) to `10` via diurnal business-hour thresholding (`08:00–18:00`) and active-learning administrative burst suppression.

* **78.4% Immediate Triage Queue Reduction**: Concentrates immediate analyst attention on `111` high-criticality **P1 Immediate Triage** alerts ($\text{Risk} \ge 50$) out of `514` raw baseline alerts.

* **77.3% Faster Triage (MTTT)**: Attaches local [SHAP](https://github.com/shap/shap) additive feature attributions to every alert, reducing Mean Time to Triage from `18.5` minutes to `4.2` minutes.

## Repository Files (Upload Order)

1. `README.md` — Main project overview, empirical ablation table, and embedded visual diagrams.

2. `01_system_architecture.png` — 4-Stage pipeline architecture diagram.

3. `02_model_benchmarks.png` — Detection quality and false-positive suppression benchmark chart.

4. `03_shap_and_soc_funnel.png` — Local [SHAP](https://github.com/shap/shap) feature attribution and SOC triage funnel chart.

5. `app.py` — Standalone [Gradio](https://gradio.app/) SOC Analyst Web Dashboard (ready for local execution or permanent 24/7 deployment on [Hugging Face Spaces](https://huggingface.co/spaces)).

6. `requirements.txt` — Python package dependencies for local, Colab, and Hugging Face Spaces environments.

7. `ai_threat_detection_pipeline.py` — Headless CLI benchmark script that runs all 4 pipeline stages and outputs the JSON evaluation report.

8. `AI_Threat_Detection_Colab_Gradio.ipynb` — Interactive [Google Colab](https://colab.research.google.com/) notebook with step-by-step markdown documentation, plots, and the embedded Gradio web app.

9. `ai_threat_detection_research_paper.tex` (and `ai_threat_detection_system_guide.pdf`) — Formal academic research paper and technical implementation blueprint.

## Quickstart Guide

### 1. Run Locally

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the CLI benchmark report
python3 ai_threat_detection_pipeline.py

# Launch the interactive Gradio SOC Web Dashboard at http://127.0.0.1:7860
python3 app.py

```

### 2. Deploy Permanently for Free on Hugging Face Spaces (24/7 Web Link)

1. Create a new Space at [Hugging Face Spaces](https://huggingface.co/spaces) and select **Gradio** as the SDK.

2. Upload `app.py` and `requirements.txt` to your Space repository.

3. Hugging Face automatically builds and hosts your live SOC dashboard on a permanent public URL.

## Benchmark Telemetry & Reference Sources

* **Datasets**: [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset), [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), and [BoT-IoT](https://research.unsw.edu.au/projects/bot-iot-dataset).

* **Telemetry & Threat Intelligence**: [Zeek](https://docs.zeek.org/), [Suricata](https://suricata.io/), [AWS CloudTrail](https://docs.aws.amazon.com/cloudtrail/), [Sigma](https://sigmahq.io/), [YARA](https://virustotal.github.io/yara/), [VirusTotal](https://docs.virustotal.com/docs/what-is-yara), and [MISP](https://www.misp-project.org/).

* **Machine Learning & Streaming**: [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), [Isolation Forest](https://ieeexplore.ieee.org/document/4781136), [SHAP](https://github.com/shap/shap), [ONNX Runtime](https://onnxruntime.ai/docs/), [Apache Kafka](https://kafka.apache.org/), and [Apache Flink](https://flink.apache.org/).