# AI Cyber & IoT SOC Dashboard

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment (recommended):

   Windows:
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Start:
   ```bash
   python app.py
   ```
6. Open http://127.0.0.1:7860

## Hugging Face Spaces

Create a new Space with the Gradio SDK and upload `app.py` and
`requirements.txt`.

## Important

This is an educational prototype using generated synthetic telemetry.
Its benchmark metrics are only measurements on that synthetic dataset.
It is not a production IDS, and it does not implement a trained GBDT,
a neural autoencoder, or formal SHAP explanations.
