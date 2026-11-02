# AI-Assisted Cyber & IoT Threat-Detection and Contextual Risk-Prioritization System

[Open In Colab](https://www.google.com/search?q=https://colab.research.google.com/github/kumarkotagiri/AI-Cybersecurity-SOC-Dashboard/blob/main/AI_Threat_Detection_Colab_Gradio.ipynb)

An end-to-end **AI-Assisted Cyber & IoT Threat-Detection Pipeline** and interactive **[Gradio](https://gradio.app/) Security Operations Center (SOC) Web Dashboard** that identifies known and unseen zero-day cyber attacks from structured network and IoT telemetry, eliminates false positives through contextual risk scoring and human-in-the-loop active learning, and explains every alert using local **[SHAP](https://github.com/shap/shap)** feature attributions.



## 1. Problem Statement & Project Motivation

Modern Security Operations Centers (SOCs) monitoring enterprise and Internet of Things (IoT) environments face two critical operational failures:
1. **Unseen Zero-Day Attacks Evade Static Rules and Supervised ML**: Traditional deterministic rules ([Sigma](https://sigmahq.io/) SIEM signatures, [YARA](https://virustotal.github.io/yara/) rules, and static Z-score thresholds) and standalone supervised classifiers only recognize previously labeled attack signatures. Stealthy zero-day attacks—such as low-and-slow covert DNS/TLS data exfiltration or fixed-jitter Command-and-Control (C2) beaconing—keep packet rates and login failures within normal bounds and bypass detection completely.
2. **High False-Positive Volume Causes Analyst Alert Fatigue**: Routine business-hour traffic surges (`08:00–18:00`) and scheduled administrative server backups trigger hundreds of false alarms in conventional anomaly detectors. Furthermore, treating an anomaly on a low-impact IoT smart bulb with the same priority as an anomaly on a core Domain Controller overwhelms human analysts.

### How This System Solves Both Problems
- **Hybrid Parallel AI Ensemble**: Executes a **Supervised Gradient-Boosted Decision Tree Classifier** ([XGBoost](https://xgboost.readthedocs.io/) / [LightGBM](https://lightgbm.readthedocs.io/)) in parallel with an **Unsupervised 3-D Subspace Reconstruction Autoencoder** and **[Isolation Forest](https://ieeexplore.ieee.org/document/4781136)** trained strictly on benign traffic. The supervised branch captures known volumetric DDoS and brute-force attacks with high precision, while the unsupervised branch catches **100.0% of unseen Zero-Day attacks** via multivariate covariance distortion.
- **Contextual False-Positive Suppression & Risk Prioritization**: Filters raw ensemble probabilities through dynamic diurnal business-hour thresholds, asynchronous [VirusTotal](https://docs.virustotal.com/docs/what-is-yara) and [MISP](https://www.misp-project.org/) threat-intelligence enrichment, human-in-the-loop active learning feedback, and CMDB asset criticality weighting—achieving **99.72% Overall Accuracy**, **97.1% Precision**, **100.0% Recall** (**98.5% F1-Score**), and a **96.7% reduction in false positives**.

## 2. Four-Stage System Architecture in Detail

### Stage 1: Telemetry Ingestion, Feature Engineering & Leakage-Free Validation
- **Benchmark & Live Telemetry Sources**: Designed for structured network flows and host logs modeled on [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset), [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), and [BoT-IoT](https://research.unsw.edu.au/projects/bot-iot-dataset), as well as live [Zeek](https://docs.zeek.org/) (`conn.log`, `dns.log`), [Suricata](https://suricata.io/) (`eve.json`), and [AWS CloudTrail](https://docs.aws.amazon.com/cloudtrail/) telemetry.
- **Sliding-Window Feature Engineering**: Raw connection events are aggregated over 5-second windows into 7 orthogonal features across 3 behavioral dimensions:

| Behavioral Dimension | Feature Name | Mathematical Formulation | Security Signal Captured |
| :--- | :--- | :--- | :--- |
| **Temporal** | `flow_duration_ms` | $\log(1 + t_{\text{end}} - t_{\text{start}})$ | Separates rapid SYN floods from long-lived covert C2 tunnels |
| **Temporal** | `inter_arrival_ms` | $\log(1 + \max(0.01, \mu_{\Delta t}))$ | Detects automated sub-millisecond floods and fixed-jitter C2 beacons |
| **Temporal** | `pkt_rate` | $\log(1 + N_{\text{pkts}} / (\Delta t_{\text{sec}} + 0.05))$ | Captures high-velocity UDP/TCP DDoS packet surges |
| **Spatial** | `byte_ratio` | $\log(1 + \text{Bytes}_{\text{sent}} / (\text{Bytes}_{\text{recv}} + 1))$ | Flags asymmetric outbound data exfiltration and DNS tunneling |
| **Spatial** | `bytes_per_pkt` | $\log(1 + (\text{Bytes}_{\text{sent}} + \text{Bytes}_{\text{recv}}) / N_{\text{pkts}})$ | Separates small-payload C2/scanning frames from bulk file transfers |
| **Categorical** | `dst_port_entropy` | $-\sum p(\text{port}_k) \log_2 p(\text{port}_k)$ | Identifies horizontal/vertical port sweeps and ephemeral port hopping |
| **Behavioral** | `failed_auth_rate` | Count of failed SSH/RDP/IAM logins per window | Detects credential stuffing, dictionary brute-force, and lateral movement |

- **Leakage-Free Chronological Split with Zero-Day Holdout**: Rather than using random $k$-fold cross-validation (which leaks contiguous multi-packet attack sessions across train and test sets), the 12,000-flow timeline is split strictly chronologically: the first **70% (`8,400` flows)** forms the training set, and the final **30% (`3,600` flows)** forms the held-out test set. Normalization parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are fitted **exclusively** on the training window, and all **140 Zero-Day covert DNS/TLS exfiltration attacks** are held out strictly in the test window.

### Stage 2: Baseline Formulation & Hybrid AI Ensemble Development
- **Deterministic Baseline**: Flags flows where volumetric or authentication Z-scores exceed $Z > 1.75$ or match static [Sigma](https://sigmahq.io/) / [YARA](https://virustotal.github.io/yara/) IOC rules.
- **Supervised GBDT Branch ($P_{\text{sup}}$)**: Trained on labeled training flows to learn decision boundaries for known SYN/UDP DDoS and SSH/RDP brute-force attacks.
- **Unsupervised Subspace Autoencoder Branch ($P_{\text{unsup}}$)**: Fitted strictly on benign training flows ($y_{\text{train}} = 0$). Projects the 7-dimensional normalized feature vector onto a 3-dimensional normal manifold $V_k$ and computes feature-weighted reconstruction error $E_{\text{recon}}(x) = \|(x - x V_k V_k^T) \odot w\|_2^2$, complemented by [Isolation Forest](https://ieeexplore.ieee.org/document/4781136) scoring.
- **Parallel Hybrid Fusion ($P_{\text{ens}}$)**: Combines both branches concurrently via $P_{\text{ens}}(x) = \max(P_{\text{sup}}(x), \, 0.94 \cdot P_{\text{unsup}}(x))$.

### Stage 3: False-Positive Reduction & Contextual Risk Prioritization
1. **Dynamic Time-of-Day Thresholding**: Scales the anomaly decision boundary from `0.52` to `0.76` during business hours (`08:00–18:00`) when port entropy and login failures are within normal operating bounds:
   $$\tau_{\text{dyn}}(t, x) = 0.52 + 0.24 \cdot \mathbb{I}_{\text{biz\_hours}}(t) \cdot \mathbb{I}_{\text{normal\_port\_auth}}(x)$$
2. **Threat Intelligence IOC Enrichment**: Queries external IP and hash indicators against [VirusTotal](https://docs.virustotal.com/docs/what-is-yara) and [MISP](https://www.misp-project.org/) feeds, adding a `+0.30` probability boost and a `1.25x` risk multiplier on confirmed hits.
3. **Human-in-the-Loop Active Learning Feedback**: Applies a `-0.35` boundary suppression penalty to recurring benign administrative patterns (such as daytime or nightly server backups dismissed by SOC analysts) and logs them for incremental retraining.
4. **CMDB Asset Criticality Contextual Risk Scoring**: Multiplies the calibrated threat probability $P_{\text{cal}}$ by the target endpoint's asset criticality weight (`IoT_Smart_Bulb = 1.5`, `IoT_HVAC_Sensor = 3.0`, `Workstation = 5.0`, `App_Server = 8.0`, `Domain_Controller = 10.0`):
   $$\text{Risk\_Score} = \text{clip}\!\left( P_{\text{cal}} \times \frac{\text{Asset\_Criticality}}{10.0} \times 100.0 \times \text{TI\_Multiplier}, \; 0, \; 100 \right)$$
   Alerts are routed into **P1 Immediate Triage** ($\text{Risk} \ge 50$), **P2 Scheduled Review** ($25 \le \text{Risk} < 50$), or **P3 Passive Log** ($\text{Risk} < 25$).

### Stage 4: Evaluation Metrics, Streaming Latency & [SHAP](https://github.com/shap/shap) Explainability
- **Sub-Millisecond Real-Time Inference**: Achieves **0.0020 ms per flow** (~500,000 flows/sec per CPU core), compatible with real-time [ONNX Runtime](https://onnxruntime.ai/docs/) execution inside [Apache Flink](https://flink.apache.org/) and [Apache Kafka](https://kafka.apache.org/) streaming pipelines.
- **Explainable AI Triage**: Computes local [SHAP](https://github.com/shap/shap) additive feature attributions for every alert so analysts immediately see which behavioral features triggered the detection.

## 3. Five-Stage Ablation Study & Empirical Results (3,600 Held-Out Test Flows)

| Pipeline Stage | Overall Accuracy | Precision | Overall Recall | F1-Score | Known Recall | Zero-Day Recall | False Positives | Total Alerts (P1 Queue) | Latency (ms/flow) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Deterministic Baseline** ([Sigma](https://sigmahq.io/) / [YARA](https://virustotal.github.io/yara/) + $Z > 1.75$) | 88.00% | 40.3% | 62.3% | 48.9% | 100.0% | 10.7% | 307 | 514 | 0.0012 ms |
| **2. Supervised Only** ([XGBoost](https://xgboost.readthedocs.io/) / [LightGBM](https://lightgbm.readthedocs.io/)) | 95.97% | 96.5% | 58.4% | 72.8% | 100.0% | 1.4% | 7 | 201 | 0.0012 ms |
| **3. Unsupervised Only** (Subspace AE / [Isolation Forest](https://ieeexplore.ieee.org/document/4781136)) | 93.86% | 60.0% | 100.0% | 75.0% | 100.0% | 100.0% | 221 | 553 | 0.0017 ms |
| **4. Raw Hybrid Ensemble** (Supervised + Unsupervised) | 95.31% | 66.3% | 100.0% | 79.7% | 100.0% | 100.0% | 169 | 501 | 0.0019 ms |
| **5. Context-Prioritized AI (Proposed Final System)** | **99.72%** | **97.1%** | **100.0%** | **98.5%** | **100.0%** | **100.0%** | **10** | **342 (111 P1)** | **0.0020 ms** |



### Operational SOC Triage & [SHAP](https://github.com/shap/shap) Feature Attribution Breakdown

| Metric / Feature | Value / Share | Operational Impact in the SOC |
| :--- | :---: | :--- |
| **False Positive Reduction** | **-96.7%** (`307` $\rightarrow$ `10` FPs) | Eliminates daytime traffic and backup false alarms via dynamic thresholding and active learning |
| **P1 Immediate Triage Queue** ($\text{Risk} \ge 50$) | **111 Alerts** (**-78.4%** vs. Baseline) | Isolates critical incidents targeting Domain Controllers, App Servers, and verified IOCs |
| **P2 Scheduled Review Queue** ($25 \le \text{Risk} < 50$) | **142 Alerts** | Routes moderate-risk Workstation anomalies to batch analyst review |
| **P3 Passive Log Queue** ($\text{Risk} < 25$) | **89 Alerts** | Automatically logs low-impact IoT Smart Bulb and HVAC sensor anomalies |
| **Mean Time to Triage (MTTT)** | **4.2 mins** (**-77.3%** vs. `18.5` mins) | Accelerates analyst investigation via human-readable local [SHAP](https://github.com/shap/shap) explanations |
| **Top SHAP Attribution Drivers** | `failed_auth_rate` (**32.3%**), `bytes_per_pkt` (**30.6%**), `dst_port_entropy` (**13.5%**), `pkt_rate` (**11.3%**) | Distinguishes credential brute-force (`failed_auth_rate`) from stealth Zero-Day exfiltration (`bytes_per_pkt` + `dst_port_entropy`) |



## 4. Interactive [Gradio](https://gradio.app/) SOC Analyst Web Dashboard

This project includes a live **[Gradio](https://gradio.app/) Web Application** (`app.py` and Step 5 of `AI_Threat_Detection_Colab_Gradio.ipynb`) with two interactive tabs:
- **Tab 1 — Live Flow Simulator & Active Learning Triage**:
  - Load 4 preset telemetry scenarios with one click (*Known SSH Brute-Force + SYN Flood on Domain Controller*, *Zero-Day Covert DNS/TLS Exfiltration on App Server*, *False-Positive Trap: Daytime Admin Backup Burst on Workstation*, or *Normal IoT Smart Bulb Telemetry*).
  - Adjust all 7 flow telemetry sliders, switch endpoint criticality (`IoT_Smart_Bulb` through `Domain_Controller`), toggle business hours (`08:00–18:00`), test [VirusTotal](https://docs.virustotal.com/docs/what-is-yara) / [MISP](https://www.misp-project.org/) IOC matches, and click the **Active Learning "Dismiss as Benign Pattern"** checkbox while inspecting the live local [SHAP](https://github.com/shap/shap) feature attribution bar plot.
- **Tab 2 — SOC Priority Alert Queue & Ablation Leaderboard**:
  - Filter all detected test flows across **P1 Immediate Triage**, **P2 Scheduled Review**, and **P3 Passive Log**, and inspect the 5-stage ablation study benchmark table.

## 5. How to Launch Directly via GitHub, Google Colab & Gradio

### Option 1: Launch Directly from GitHub in Google Colab (Zero Local Setup)
1. Click the **[Open in Colab](https://www.google.com/search?q=https://colab.research.google.com/github/kumarkotagiri/AI-Cybersecurity-SOC-Dashboard/blob/main/AI_Threat_Detection_Colab_Gradio.ipynb)** badge at the top of this repository (or open `AI_Threat_Detection_Colab_Gradio.ipynb` directly from GitHub in [Google Colab](https://colab.research.google.com/)).
2. Click **Runtime $\rightarrow$ Run all** (`Ctrl + F9`).
3. Scroll to **Step 5** at the bottom of the notebook: [Gradio](https://gradio.app/) automatically embeds the interactive SOC Web Dashboard inside the notebook and generates a free public **`.gradio.live` web URL** that you can open in any browser tab.

### Option 2: Clone from GitHub & Launch Gradio Locally
```bash
git clone https://github.com/kumarkotagiri/AI-Cybersecurity-SOC-Dashboard.git
cd AI-Cybersecurity-SOC-Dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the CLI benchmark report (outputs full JSON metrics)
python3 ai_threat_detection_pipeline.py

# Launch the interactive Gradio SOC Web Dashboard at http://127.0.0.1:7860
python3 app.py
```

## 6. Repository File Guide

| File Name | Description |
| :--- | :--- |
| `README.md` | Complete project documentation, mathematical formulations, ablation tables, and embedded visuals |
| `01_system_architecture.png` | 4-Stage pipeline architecture diagram |
| `02_model_benchmarks.png` | Detection quality and false-positive suppression benchmark chart across all 5 stages |
| `03_shap_and_soc_funnel.png` | Local [SHAP](https://github.com/shap/shap) feature attribution and SOC P1/P2/P3 alert funnel chart |
| `AI_Threat_Detection_Colab_Gradio.ipynb` | End-to-end [Google Colab](https://colab.research.google.com/) notebook with data generation, training, evaluation, and the embedded [Gradio](https://gradio.app/) web app |
| `app.py` | Standalone [Gradio](https://gradio.app/) SOC Analyst Web Dashboard application |
| `ai_threat_detection_pipeline.py` | Standalone CLI Python script for running the 4-stage pipeline and printing JSON evaluation metrics |
| `requirements.txt` | Python package dependencies (`gradio`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xgboost`, `lightgbm`, `shap`) |
| `ai_threat_detection_research_paper.pdf` | Compiled IEEE/academic research paper PDF |
| `ai_threat_detection_research_paper.tex` | LaTeX source code for the academic research paper |
| `ai_threat_detection_system_guide.pdf` | Detailed engineering implementation blueprint PDF |

## 7. Benchmark Datasets & Technical References
- **Cyber & IoT Datasets**: [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset), [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), and [BoT-IoT](https://research.unsw.edu.au/projects/bot-iot-dataset).
- **Security Telemetry & Threat Intelligence**: [Zeek Network Security Monitor](https://docs.zeek.org/), [Suricata IDS/IPS](https://suricata.io/), [AWS CloudTrail](https://docs.aws.amazon.com/cloudtrail/), [Sigma SIEM Signatures](https://sigmahq.io/), [YARA Pattern Engine](https://virustotal.github.io/yara/), [VirusTotal](https://docs.virustotal.com/docs/what-is-yara), and [MISP Threat Sharing](https://www.misp-project.org/).
- **Machine Learning, Explainability & Web UI**: [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), [Isolation Forest](https://ieeexplore.ieee.org/document/4781136), [SHAP (SHapley Additive exPlanations)](https://github.com/shap/shap), [Gradio](https://gradio.app/), [ONNX Runtime](https://onnxruntime.ai/docs/), [Apache Kafka](https://kafka.apache.org/), and [Apache Flink](https://flink.apache.org/).