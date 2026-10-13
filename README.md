# Cloudify

A yellow-and-black Streamlit dashboard with **Scheduling** and **Hardware** only. Bundled, frozen Random Forest, LSTM Autoencoder, and One-Class SVM models from the parent `research_improved` experiment (seed 42). No training, API keys, or parent directory is required.

## Run locally

Use Python **3.12**:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

On Windows, `start.bat` also launches using the project virtual environment.

## Deploy on Streamlit Community Cloud

1. Push this entire repository, including `models/` and `examples/`, to a GitHub repository named `cloudify`.
2. At https://share.streamlit.io select **Create app**, choose the repository and `main` branch, and set the entry point to **app.py**.
3. In **Advanced settings**, select **Python 3.12**, then deploy. No secrets are needed.

Official instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

Dependencies and Streamlit theme are included. TensorFlow is loaded lazily when evaluating a sequence. Models use CPU inference, limited to one thread by default. The online app requires a normal package installation on its first deployment.

## How to use

- **Scheduling:** configure task metadata; values persist while switching views.
- **Hardware:** set CPI and memory accesses per instruction, then evaluate manual hardware with Random Forest and One-Class SVM.
- Click **Try bundled hardware example** to immediately evaluate all three models without an upload. Download the example CSV or upload 30–1000 chronological rows with `metric_01` through `metric_38`.
- Choose a primary model and download evaluation results as JSON.

## Model scope and provenance

Artifacts come from `20260930T155937Z_research_improved`, seed 42. Manual inference uses the parent's Google Cluster preprocessing, Random Forest and genuine One-Class SVM. Unexposed fields use fitted training medians. Scheduling identifiers were excluded by the parent pipeline and do not influence these scores.

Sequence inference uses the parent's SMD machine-3-1 preprocessing and all three frozen models. Causal rolling means and deltas match the original preprocessing. The LSTM uses the final 30-row window; tabular models score the same final endpoint. An uploaded sequence starts a new rolling context. Arbitrary production counters are not interchangeable with the SMD feature schema.

The LSTM is not applicable to the Google sample (at most four observations per task). Model thresholds are the original saved thresholds; One-Class SVM is a comparison baseline, with no intentionally degraded scores. Model ranking depends on dataset and metric; choosing these three algorithms does not assert that the two selected ones outperform SVM everywhere. Full saved evaluation metrics are bundled in `models/`.

This is functional offline scenario evaluation, not a live provider connection or remote job scheduler. Scheduling configuration is session-scoped and does not dispatch cloud jobs. Uploaded telemetry is session-only and is not written to disk. The example consists of 60 real SMD machine-3-1 observations.

Parent SMD license note: upstream OmniAnomaly repository contains MIT LICENSE, with no separate dataset-specific license identified. Google-derived models retain the parent sample's provenance (Kaggle listing CC BY-NC-SA 4.0; upstream Google trace CC BY 4.0). Review upstream terms for your intended redistribution.
