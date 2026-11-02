#!/usr/bin/env python3
"""
AI-Assisted Cyber & IoT Threat-Detection SOC Dashboard
Run: python app.py
Open: http://127.0.0.1:7860

Educational demonstration using SYNTHETIC telemetry. It is not a production IDS.
The displayed model scores are generated from a synthetic dataset and heuristic
components; they must not be represented as validated real-world performance.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr


ASSET_CRIT_MAP = {
    "IoT_Smart_Bulb (Crit = 1.5)": 1.5,
    "IoT_HVAC_Sensor (Crit = 3.0)": 3.0,
    "Workstation (Crit = 5.0)": 5.0,
    "App_Server (Crit = 8.0)": 8.0,
    "Domain_Controller (Crit = 10.0)": 10.0,
}


def initialize_system(n_flows=12000, seed=42):
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-10-25", periods=n_flows, freq="5s")
    hours = timestamps.hour.to_numpy()

    asset_types = list(ASSET_CRIT_MAP)
    # Use the labels without the UI criticality suffix for generated telemetry.
    asset_names = ["IoT_Smart_Bulb", "IoT_HVAC_Sensor", "Workstation",
                   "App_Server", "Domain_Controller"]
    crit_map = dict(zip(asset_names, [1.5, 3.0, 5.0, 8.0, 10.0]))
    assets = rng.choice(asset_names, size=n_flows, p=[.30, .25, .25, .13, .07])
    criticality = np.array([crit_map[a] for a in assets])
    biz = ((hours >= 8) & (hours <= 18)).astype(float)

    labels = np.zeros(n_flows, dtype=int)
    split = int(n_flows * .70)
    known = rng.random(n_flows) < .058
    labels[known] = 1
    candidates = np.where((np.arange(n_flows) >= split) & (labels == 0))[0]
    zero_day_idx = rng.choice(candidates, size=min(140, len(candidates)), replace=False)
    labels[zero_day_idx] = 2
    threat = (labels > 0).astype(int)

    latent = rng.normal(0, 1, n_flows) + .55 * biz
    duration = np.exp(5.6 + .45 * latent + rng.normal(0, .25, n_flows))
    iat = np.exp(4.2 - .40 * latent + rng.normal(0, .22, n_flows))
    sent = np.exp(7.5 + .55 * latent + rng.normal(0, .30, n_flows))
    recv = np.exp(8.2 + .55 * latent + rng.normal(0, .30, n_flows))
    pkt = np.maximum(2, (sent + recv) / rng.normal(550, 35, n_flows))
    entropy = np.clip(1.10 + .18 * latent + rng.normal(0, .22, n_flows), .1, 2.4)
    auth = np.clip(rng.exponential(.22, n_flows) * (1 + .35 * biz), 0, 1.8)
    ioc = (rng.random(n_flows) < .002).astype(float)

    admin = (labels == 0) & (biz == 1) & (rng.random(n_flows) < .085)
    sent[admin] *= rng.uniform(3.8, 5.5, admin.sum())
    recv[admin] *= rng.uniform(.35, .55, admin.sum())
    pkt[admin] *= rng.uniform(2.5, 4.0, admin.sum())
    duration[admin] *= rng.uniform(2.2, 3.2, admin.sum())

    m1 = labels == 1
    duration[m1] *= rng.uniform(.20, .45, m1.sum())
    iat[m1] *= rng.uniform(.10, .25, m1.sum())
    sent[m1] *= rng.uniform(3.2, 6.0, m1.sum())
    pkt[m1] *= rng.uniform(5.5, 10.0, m1.sum())
    entropy[m1] = np.clip(rng.normal(2.55, .35, m1.sum()), 1.6, 4)
    auth[m1] = np.clip(rng.normal(4.2, .75, m1.sum()), 2.1, 8.5)
    ioc[m1] = (rng.random(m1.sum()) < .62).astype(float)

    m2 = labels == 2
    duration[m2] = np.exp(5.7 + rng.normal(0, .20, m2.sum()))
    iat[m2] = np.exp(4.3 + rng.normal(0, .18, m2.sum()))
    sent[m2] = np.exp(7.7 + rng.normal(0, .22, m2.sum()))
    recv[m2] = np.exp(6.3 + rng.normal(0, .22, m2.sum()))
    pkt[m2] = np.maximum(15, (sent[m2] + recv[m2]) /
                          rng.normal(145, 18, m2.sum()))
    entropy[m2] = np.clip(rng.normal(2.95, .28, m2.sum()), 2.2, 4.2)
    auth[m2] = np.clip(rng.exponential(.18, m2.sum()), 0, .9)
    ioc[m2] = (rng.random(m2.sum()) < .12).astype(float)

    byte_ratio = sent / (recv + 1)
    bytes_per_pkt = (sent + recv) / np.maximum(1, pkt)
    pkt_rate = pkt / (duration / 1000 + .05)
    feature_names = ["flow_duration_ms", "inter_arrival_ms", "byte_ratio",
                     "bytes_per_pkt", "pkt_rate", "dst_port_entropy",
                     "failed_auth_rate"]
    X = np.column_stack([
        np.log1p(duration), np.log1p(np.maximum(.01, iat)),
        np.log1p(byte_ratio), np.log1p(bytes_per_pkt),
        np.log1p(pkt_rate), entropy, auth
    ])

    X_train, y_train = X[:split], threat[:split]
    X_test, y_test = X[split:], threat[split:]
    labels_test, hours_test = labels[split:], hours[split:]
    assets_test, crit_test, ioc_test = assets[split:], criticality[split:], ioc[split:]
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-8
    Xtr, Xte = (X_train - mu) / sd, (X_test - mu) / sd

    t0 = time.perf_counter()
    baseline_score = np.max(Xte[:, [0, 4, 6]], axis=1)
    baseline = ((baseline_score > 1.75) | (ioc_test > 0)).astype(int)
    baseline_ms = (time.perf_counter() - t0) * 1000 / len(X_test)

    t0 = time.perf_counter()
    weights = np.array([-.55, -.95, .15, -.10, 1.45, .25, 2.25])
    logit = Xte @ weights - 3.45
    boost = np.where((Xte[:, 6] > 1.95) |
                     ((Xte[:, 4] > 1.85) & (Xte[:, 1] < -.95)), 2.40, 0)
    sup_prob = 1 / (1 + np.exp(-np.clip(logit + boost, -20, 20)))
    sup_pred = (sup_prob >= .50).astype(int)
    sup_ms = (time.perf_counter() - t0) * 1000 / len(X_test)

    t0 = time.perf_counter()
    benign = Xtr[y_train == 0]
    cov = np.cov(benign, rowvar=False) + np.eye(benign.shape[1]) * .03
    _, eigvecs = np.linalg.eigh(cov)
    Vk = eigvecs[:, -3:]
    train_proj = (benign @ Vk) @ Vk.T
    test_proj = (Xte @ Vk) @ Vk.T
    feat_weights = np.array([.6, .7, 1.8, 1.7, .8, 2.1, .8])
    train_err = np.mean(((benign - train_proj) ** 2) * feat_weights, axis=1)
    test_err = np.mean(((Xte - test_proj) ** 2) * feat_weights, axis=1)
    recon_thresh = float(np.percentile(train_err, 96.5))
    unsup_prob = 1 / (1 + np.exp(-(test_err - recon_thresh) / .40))
    unsup_pred = (unsup_prob >= .50).astype(int)
    unsup_ms = (time.perf_counter() - t0) * 1000 / len(X_test)

    ensemble_prob = np.maximum(sup_prob, .94 * unsup_prob)
    ensemble_pred = (ensemble_prob >= .50).astype(int)

    t0 = time.perf_counter()
    biz_test = ((hours_test >= 8) & (hours_test <= 18)).astype(float)
    normal_context = ((Xte[:, 5] < 1.65) & (Xte[:, 6] < 1.50) & (ioc_test == 0))
    threshold = .52 + .24 * (biz_test * normal_context)
    ti_boost = np.where(ioc_test > 0, .30, 0)
    admin_sig = ((Xte[:, 0] > .6) & (Xte[:, 5] < 1.55) &
                 (Xte[:, 6] < 1.40) & (ioc_test == 0))
    penalty = np.where(admin_sig, -.35, 0)
    calibrated = np.clip(ensemble_prob + ti_boost + penalty, 0, 1)
    final_pred = (calibrated >= threshold).astype(int)
    risk = np.clip(calibrated * (crit_test / 10) * 100 *
                   np.where(ioc_test > 0, 1.25, 1), 0, 100)
    proposed_ms = ((time.perf_counter() - t0) * 1000 / len(X_test)
                   + sup_ms + unsup_ms)

    def metrics(name, pred, latency):
        tp = int(((y_test == 1) & (pred == 1)).sum())
        fp = int(((y_test == 0) & (pred == 1)).sum())
        fn = int(((y_test == 1) & (pred == 0)).sum())
        tn = int(((y_test == 0) & (pred == 0)).sum())
        precision = tp / (tp + fp + 1e-9) * 100
        recall = tp / (tp + fn + 1e-9) * 100
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        known_mask, zero_mask = labels_test == 1, labels_test == 2
        return {
            "Architecture": name,
            "Accuracy (%)": round((tp + tn) / len(y_test) * 100, 2),
            "Precision (%)": round(precision, 1),
            "Recall (%)": round(recall, 1),
            "F1 (%)": round(f1, 1),
            "Known Recall (%)": round(float(pred[known_mask].mean() * 100), 1)
                if known_mask.any() else 0,
            "Zero-Day Recall (%)": round(float(pred[zero_mask].mean() * 100), 1)
                if zero_mask.any() else 0,
            "TP": tp, "FP": fp, "Alerts": int(pred.sum()),
            "Latency (ms/flow)": round(max(latency, .001), 4)
        }

    leaderboard = pd.DataFrame([
        metrics("1. Deterministic Baseline", baseline, baseline_ms),
        metrics("2. Supervised Classifier", sup_pred, sup_ms),
        metrics("3. Unsupervised Subspace Detector", unsup_pred, unsup_ms),
        metrics("4. Raw Hybrid Ensemble", ensemble_pred, sup_ms + unsup_ms),
        metrics("5. Context-Prioritized AI (Proposed)", final_pred, proposed_ms),
    ])
    local_attr = .5 * np.abs(Xte * weights) + .5 * (((Xte - test_proj) ** 2) * feat_weights)

    return {
        "mu": mu, "sd": sd, "weights": weights, "Vk": Vk,
        "feat_weights": feat_weights, "recon_thresh": recon_thresh,
        "feature_names": feature_names, "X_test": X_test,
        "labels_test": labels_test, "hours_test": hours_test,
        "assets_test": assets_test, "crit_test": crit_test,
        "final_pred": final_pred, "risk": risk, "local_attr": local_attr,
        "leaderboard": leaderboard, "split": split
    }


STATE = initialize_system()


def score_single_flow(duration, iat, byte_ratio, bytes_per_pkt, pkt_rate,
                      entropy, auth_rate, asset_label, hour, ioc_hit,
                      analyst_dismissed):
    start = time.perf_counter()
    x = np.array([
        np.log1p(max(0.0, duration)), np.log1p(max(.01, iat)),
        np.log1p(max(0.0, byte_ratio)), np.log1p(max(0.0, bytes_per_pkt)),
        np.log1p(max(0.0, pkt_rate)), entropy, auth_rate
    ])
    xz = (x - STATE["mu"]) / STATE["sd"]
    logit = float(xz @ STATE["weights"] - 3.45)
    boost = 2.40 if (xz[6] > 1.95 or
                     (xz[4] > 1.85 and xz[1] < -.95)) else 0
    p_sup = float(1 / (1 + np.exp(-np.clip(logit + boost, -20, 20))))
    proj = (xz @ STATE["Vk"]) @ STATE["Vk"].T
    err = float(np.mean(((xz - proj) ** 2) * STATE["feat_weights"]))
    p_unsup = float(1 / (1 + np.exp(-(err - STATE["recon_thresh"]) / .40)))
    p_ensemble = max(p_sup, .94 * p_unsup)

    is_biz = 1.0 if 8 <= int(hour) <= 18 else 0.0
    normal = xz[5] < 1.65 and xz[6] < 1.50 and not ioc_hit
    cutoff = .52 + .24 * (is_biz * float(normal))
    admin = (xz[0] > .6 and xz[5] < 1.55 and xz[6] < 1.40 and not ioc_hit)
    penalty = -.35 if (admin or analyst_dismissed) else 0
    calibrated = float(np.clip(p_ensemble + (.30 if ioc_hit else 0) + penalty, 0, 1))
    criticality = ASSET_CRIT_MAP.get(asset_label, 5.0)
    risk = float(np.clip(calibrated * criticality / 10 * 100 *
                         (1.25 if ioc_hit else 1), 0, 100))
    alert = calibrated >= cutoff

    if not alert:
        tier = "BENIGN / SUPPRESSED — No alert escalated"
    elif risk >= 50:
        tier = "P1 CRITICAL — Immediate SOC triage"
    elif risk >= 25:
        tier = "P2 HIGH — Scheduled analyst review"
    else:
        tier = "P3 LOW — Passive log"

    attr = .5 * np.abs(xz * STATE["weights"]) + .5 * (((xz - proj) ** 2) * STATE["feat_weights"])
    pct = attr / (attr.sum() + 1e-9) * 100
    order = np.argsort(pct)
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=110)
    vals = pct[order]
    names = [STATE["feature_names"][i] for i in order]
    bars = ax.barh(names, vals, color="#1D4ED8" if alert else "#64748B")
    ax.set_xlim(0, max(float(vals.max()) * 1.25, 5))
    ax.set_xlabel("Attribution share (%)")
    ax.set_title("Local feature attribution (approximation)", fontsize=10, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, vals):
        ax.text(value + .5, bar.get_y() + bar.get_height()/2,
                f"{value:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    latency = (time.perf_counter() - start) * 1000
    summary = f"""### SOC decision: **{tier}**

- **Contextual risk score:** `{risk:.1f}/100`
- **Calibrated threat score:** `{calibrated*100:.1f}%` (dynamic cutoff `{cutoff*100:.1f}%`)
- **Supervised component:** `{p_sup*100:.1f}%`
- **Unsupervised component:** `{p_unsup*100:.1f}%` (reconstruction error `{err:.2f}`)
- **Inference time:** `{latency:.4f} ms`

> Educational synthetic-data demonstration. The attribution chart is a heuristic explanation, not SHAP output.
"""
    return summary, fig


def load_preset(name):
    presets = {
        "Known Attack: SSH Brute-Force + SYN Flood on Domain Controller":
            (95, 12, 4.8, 310, 185, 2.85, 5.2, "Domain_Controller (Crit = 10.0)", 14, True, False),
        "Zero-Day-like Scenario: Covert DNS/TLS Exfiltration on App Server":
            (310, 74, 7.2, 138, 28, 3.25, .1, "App_Server (Crit = 8.0)", 22, False, False),
        "False-Positive Trap: Daytime Admin Backup Burst on Workstation":
            (820, 58, 4.2, 610, 14, 1.15, .2, "Workstation (Crit = 5.0)", 11, False, False),
        "Normal Traffic: Routine IoT Smart Bulb Telemetry":
            (260, 68, .5, 540, 6.5, 1.05, .1, "IoT_Smart_Bulb (Crit = 1.5)", 15, False, False),
    }
    return presets[name]


def filter_soc_queue(queue_filter):
    rows = []
    for i, pred in enumerate(STATE["final_pred"]):
        if not pred:
            continue
        risk = float(STATE["risk"][i])
        if queue_filter.startswith("P1") and risk < 50:
            continue
        if queue_filter.startswith("P2") and not 25 <= risk < 50:
            continue
        if queue_filter.startswith("P3") and risk >= 25:
            continue
        top = np.argsort(-STATE["local_attr"][i])[:2]
        labels = STATE["labels_test"][i]
        true_class = ("Known attack" if labels == 1 else
                      "Zero-day-like injected event" if labels == 2 else
                      "Benign false positive")
        rows.append({
            "Flow ID": int(STATE["split"] + i),
            "Hour": f"{int(STATE['hours_test'][i]):02d}:00",
            "Asset": STATE["assets_test"][i],
            "Criticality": float(STATE["crit_test"][i]),
            "Synthetic ground truth": true_class,
            "Risk score": round(risk, 1),
            "Top feature triggers": ", ".join(
                STATE["feature_names"][j] for j in top)
        })
    columns = ["Flow ID", "Hour", "Asset", "Criticality",
               "Synthetic ground truth", "Risk score", "Top feature triggers"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "Risk score", ascending=False).head(25)


PRESETS = [
    "Known Attack: SSH Brute-Force + SYN Flood on Domain Controller",
    "Zero-Day-like Scenario: Covert DNS/TLS Exfiltration on App Server",
    "False-Positive Trap: Daytime Admin Backup Burst on Workstation",
    "Normal Traffic: Routine IoT Smart Bulb Telemetry",
]
INPUTS = ["Flow duration (ms)", "Inter-arrival time (ms)",
          "Outbound/inbound byte ratio", "Mean bytes per packet",
          "Packet rate (packets/sec)", "Destination port entropy",
          "Failed authentication frequency", "Asset", "Hour of day",
          "Threat-intelligence IOC match", "Analyst feedback: dismiss as benign"]


with gr.Blocks(title="AI Cyber & IoT SOC Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# AI-Assisted Cyber & IoT Threat-Detection SOC Dashboard\n"
        "Synthetic telemetry • Hybrid detection • Contextual risk prioritization\n\n"
        "**Research prototype:** all telemetry is simulated; benchmark results are "
        "specific to this generated dataset and are not real-world validation."
    )
    with gr.Tabs():
        with gr.Tab("Live Flow Simulator & Triage"):
            preset = gr.Dropdown(PRESETS, value=PRESETS[1],
                                 label="Load a demonstration scenario")
            with gr.Row():
                with gr.Column():
                    duration = gr.Slider(10, 1500, value=310, label=INPUTS[0])
                    iat = gr.Slider(1, 250, value=74, label=INPUTS[1])
                    ratio = gr.Slider(.05, 12, value=7.2, label=INPUTS[2])
                    bpp = gr.Slider(60, 1200, value=138, label=INPUTS[3])
                    pkt_rate = gr.Slider(1, 300, value=28, label=INPUTS[4])
                    entropy = gr.Slider(.1, 4.5, value=3.25, label=INPUTS[5])
                    auth = gr.Slider(0, 10, value=.1, label=INPUTS[6])
                with gr.Column():
                    asset = gr.Dropdown(list(ASSET_CRIT_MAP),
                                        value="App_Server (Crit = 8.0)",
                                        label=INPUTS[7])
                    hour = gr.Slider(0, 23, step=1, value=22, label=INPUTS[8])
                    ioc = gr.Checkbox(value=False, label=INPUTS[9])
                    dismissed = gr.Checkbox(value=False, label=INPUTS[10])
                    result = gr.Markdown()
                    plot = gr.Plot()
            flow_inputs = [duration, iat, ratio, bpp, pkt_rate, entropy,
                           auth, asset, hour, ioc, dismissed]
            preset.change(load_preset, inputs=preset, outputs=flow_inputs)
            for component in flow_inputs:
                component.change(score_single_flow, inputs=flow_inputs,
                                 outputs=[result, plot])
            demo.load(score_single_flow, inputs=flow_inputs,
                      outputs=[result, plot])

        with gr.Tab("SOC Priority Alert Queue"):
            queue_filter = gr.Radio(
                ["P1 Immediate triage (risk >= 50)",
                 "P2 Scheduled review (risk 25–49)",
                 "P3 Passive log (risk < 25)"],
                value="P1 Immediate triage (risk >= 50)",
                label="Select triage queue")
            queue_table = gr.Dataframe(
                value=filter_soc_queue("P1 Immediate triage (risk >= 50)"),
                label="Top 25 prioritized alerts",
                interactive=False, wrap=True)
            queue_filter.change(filter_soc_queue, inputs=queue_filter,
                                outputs=queue_table)
            gr.Markdown(
                "Ground-truth labels are available here only because this is a "
                "synthetic research simulation. A deployed SOC would not know "
                "the true class at alert time."
            )

        with gr.Tab("Ablation & Benchmark"):
            gr.Markdown(
                "## Architecture comparison\n"
                "Metrics below are computed on the held-out portion of the "
                "synthetic dataset generated at startup. They are not evidence "
                "of performance on operational network traffic."
            )
            benchmark = gr.Dataframe(value=STATE["leaderboard"],
                                     interactive=False, wrap=True)
            with gr.Row():
                gr.Markdown(
                    f"**Flows generated:** {12000:,}  \n"
                    f"**Training flows:** {STATE['split']:,}  \n"
                    f"**Test flows:** {12000 - STATE['split']:,}"
                )
            gr.Markdown(
                "### Important methodology note\n"
                "The supervised component uses fixed hand-authored coefficients "
                "and rules, while the unsupervised component uses a PCA-style "
                "subspace reconstruction score. This prototype does not train "
                "a gradient-boosted decision tree or a neural autoencoder. "
                "The feature-attribution plot is a heuristic, not a formal SHAP "
                "explanation. Replace these components with trained models and "
                "validate on independent, real datasets before making research "
                "or deployment claims."
            )

    gr.Markdown("---\n"
                "For authorized educational testing only. Do not use the "
                "simulator's synthetic scores to make production security decisions.")


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
