# Predictive Maintenance: Multi-Label Failure Classifier

This is an advanced predictive maintenance tool that doesn't just predict *if* a failure will happen, but diagnoses the **specific type** of failure.

This app uses a `MultiOutputClassifier` built inside a `scikit-learn` `Pipeline` to predict the individual risk for 4 distinct failure modes:
* **HDF** (Heat Dissipation Failure)
* **PWF** (Power Failure)
* **OSF** (Overstrain Failure)
* **TWF** (Tool Wear Failure)

For each failure type, the app applies a unique, custom-tuned threshold (e.g., `TWF: 0.17`, `PWF: 0.45`) to make a decision and then generates a *separate* SHAP plot to explain the drivers for that specific mode.



## 🚀 App Features

* **Multi-Target Prediction:** Provides 4 distinct predictions from a single set of inputs.
* **Custom Threshold Dictionary:** Applies a different, optimized threshold for each failure type, drastically improving accuracy (especially for rare failures like `TWF`).
* **Per-Target Explainability (XAI):** Generates 4 separate SHAP waterfall plots (in expanders) to answer questions like:
    * "What features are driving the `HDF` risk?"
    * "What features are driving the `TWF` risk?"
* **Manual Feature Engineering:** The app first creates `TempDiff`, `Power [W]`, and `OverstrainMetric` before feeding the data to the model's pipeline.

## 🛠️ Tech Stack

* **`streamlit`**
* **`pandas`**
* **`numpy`**
* **`scikit-learn`** (for `Pipeline`, `MultiOutputClassifier`, etc.)
* **`shap`**
* **`matplotlib`**

## 🏃 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [your-github-repo-url]
    cd [your-repo-name]
    ```

2.  **Install dependencies:**
    Make sure you have the pinned `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```
    (Or the robust version: `python -m streamlit run app.py`)
