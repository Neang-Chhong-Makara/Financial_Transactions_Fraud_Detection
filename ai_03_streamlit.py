import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import random
import os
import sys
import io

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# --- Force UTF-8 encoding for Windows / MLflow safety ---
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass  # Streamlit’s I/O capture may block reconfiguration

# Page setup
st.set_page_config(page_title="Financial Transactions Fraud Detection", layout="wide")
st.title("📊 Assignment: Financial Transactions Fraud Detection")

st.markdown("##### -`INSTITUTION`: Sunrise Institute")
st.markdown("##### -`LECTURER`: KIV Sith Vothy")
st.markdown("##### -`SUBJECT`: Data Science (AI)-G10 T4")
st.markdown("##### -`SHIFT`: DSS-G10 (Weekend-Morning)")
st.markdown("##### -`STUDENT NAME`: Neang Chhong Makara")
st.markdown("##### -`SUBMITTED DATE`: 25-Oct-2025")


# Show Clear button in top right
clear_spacer, clear_col = st.columns([9, 1])
with clear_col:
    if st.button("🔁 Clear", use_container_width=True):
        # Clear session state variables
        st.session_state.df = None
        st.session_state.predictions_done = False
        st.session_state.selected_models = []

        # Clear additional prediction-related state
        st.session_state.predicted_df = None

        # Safely remove any other optional keys
        for key in ["predicted_df", "selected_models"]:
            if key in st.session_state:
                del st.session_state[key]

        # Rerun the app to reflect the cleared state
        st.rerun()


# Template download
st.markdown("### 🧾 Need a sample template?")
with open("dataset/sample_input_template.csv", "rb") as file:
    sample_csv = file.read()

st.download_button(
    label="📄 Download Sample CSV Template",
    data=sample_csv,
    file_name="dataset/sample_input_template.csv",
    mime="text/csv"
)

# Upload CSV file
uploaded_file = st.file_uploader("📂 Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state["original_df"] = df.copy()  # Save original for reruns
    except Exception as e:
        st.error(f"❌ Failed to read CSV file: {e}")
        st.stop()

    st.subheader("🔍 Uploaded Data Preview")
    st.dataframe(df.head())

    st.subheader("📈 Basic Statistics: Numerical & Categorical Features Preview")

    num_cols = df.select_dtypes(include=['number']).columns
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns

    if not num_cols.empty:
        st.write("**🔢 Numerical Features Summary:**")
        st.dataframe(df[num_cols].describe().transpose())
    else:
        st.write("No numerical columns found.")

    if not cat_cols.empty:
        st.write("**🔠 Categorical Features Summary:**")
        st.dataframe(df[cat_cols].describe(include='all').transpose())
    else:
        st.write("No categorical columns found.")

    X_test = df.copy()

    # === Cast Integer Columns to Float64 ===
    int_cols = X_test.select_dtypes(include=["int", "int64"]).columns.tolist()
    st.write("Detected integer columns converted to float64:", int_cols)

    if int_cols:
        X_test[int_cols] = X_test[int_cols].astype("float64")
    # MLflow is warning to avoid schema enforcement errors during inference, cast those columns to float64.

    # Model selector
    model_dict = {
        '1_Logistic_Regression': '1_Logistic_Regression_model.joblib',
        '2_SVM': '2_SVC_model.joblib',
        '3_Decision_Tree': '3_Decision_Tree_model.joblib',
        '4_Random_Forest': '4_Random_Forest_model.joblib',
        '5_XGBoost': '5_XGBoost_model.joblib'
    }

    selected_models = st.multiselect(
        "🧠 Select model(s) for prediction:",
        options=list(model_dict.keys()),
        default=["1_Logistic_Regression"]
    )

    # --- Helper: safe MLflow logging wrapper ---
    def safe_log_param(key, value):
        try:
            mlflow.log_param(key, str(value))
        except Exception as e:
            st.warning(f"⚠️ Skipped logging param {key}: {e}")

    def safe_log_params(params):
        for k, v in params.items():
            safe_log_param(k, v)

    if st.button("🔍 Run Prediction"):

        # --------------Start Custom Mlflow -------------------------------------
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        
        # Set experiment name
        experiment_name = "03_Financial_Fraud_Detection__Streamlit"
        mlflow.set_experiment(experiment_name)
        
        # Get experiment ID
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        
        # Count existing runs
        runs = client.search_runs(experiment_ids=[experiment_id])
        run_count = len(runs)
        
        # Optional: if you want to group runs by 5s
        if run_count >= 5:
            run_group = run_count // 5
        else:
            run_group = 0
        
        run_number = run_group + 1
        run_name = f"Run_{run_number:03d}"  # e.g., Run_001, Run_002
        # --------------End Custom Mlflow -------------------------------------

        for model_name in selected_models:
            model_file = model_dict[model_name]
            try:
                model_path = os.path.join('models', model_file)

                if not os.path.exists(model_path):
                    st.error(f"❌ Model file not found: {model_file}")
                    continue

                model = joblib.load(model_path)
                model_label = model_name.replace(" ", "_")

                # Check required features
                if hasattr(model, "feature_names_in_"):
                    missing = set(model.feature_names_in_) - set(X_test.columns)
                    if missing:
                        st.error(f"❌ {model_name} is missing required columns: {missing}")
                        continue

                pred_col = f'{model_label}_Prediction'
                proba_col = f'{model_label}_predict_proba'

                # -------------- Start Run Mlflow -------------------------------------
                safe_run_name = (run_name + "__" + model_name).encode('ascii', 'ignore').decode()
                with mlflow.start_run(run_name=safe_run_name):
                    safe_log_param("model_file", model_file)
                    safe_log_params(model.get_params())

                    preds = model.predict(X_test)

                    # === Convert 5_XGBoost predictions (1, 0) to (True/False) ===
                    if model_name == "5_XGBoost":
                        preds = np.where(preds == 1, True, False)

                    # === Predict class labels ===
                    df[pred_col] = preds

                    # === Predict probability of fraud (class 1) ===
                    if hasattr(model, "predict_proba"):
                        df[proba_col] = model.predict_proba(X_test)[:, 1]
                        mlflow.log_metric("avg_pred_proba", np.mean(df[proba_col]))
                    else:
                        df[proba_col] = np.nan  # If predict_proba is unavailable

                    # === Log Model with Signature ===
                    signature = infer_signature(X_test, preds)
                    mlflow.sklearn.log_model(model, artifact_path=f"{model_name}_model", signature=signature)
        
                    # === Save intermediate prediction file ===
                    temp_output = f"{model_name}_predictions.csv"
                    df[[pred_col, proba_col]].to_csv(temp_output, index=False, encoding='utf-8')
                    mlflow.log_artifact(temp_output)
                  
                    # === Clean Up ===
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
        
                    st.success(f"✅ {model_name} predictions completed and logged to MLflow.")
                # -------------- End Run Mlflow -------------------------------------

            except Exception as e:
                st.error(f"❌ Error with {model_name}: {e}")

        # Store results in session state for persistence
        st.session_state["predicted_df"] = df.copy()
        st.session_state["selected_models"] = selected_models

# -------------------------------------------
# Display prediction results from session state (after rerun, download, etc.)
# -------------------------------------------
if "predicted_df" in st.session_state and "selected_models" in st.session_state:
    df = st.session_state["predicted_df"]
    selected_models = st.session_state["selected_models"]

    st.subheader("📋 Prediction Output Preview")

    # Pastel palette
    PASTEL_COLORS = [
        "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF",
        "#D7BAFF", "#FFBAED", "#C2F0FC", "#C6FFDD", "#FFF5BA"
    ]
    random.shuffle(PASTEL_COLORS)

    model_colors = {}
    for i, model_name in enumerate(selected_models):
        model_label = model_name.replace(" ", "_")
        model_colors[model_label] = PASTEL_COLORS[i % len(PASTEL_COLORS)]

    highlight_cols = {}
    for model_label, color in model_colors.items():
        highlight_cols[f"{model_label}_Prediction"] = color
        highlight_cols[f"{model_label}_predict_proba"] = color

    def highlight_new_cols(col):
        color = highlight_cols.get(col.name)
        return [f'background-color: {color}' if color else '' for _ in col]

    st.dataframe(df.head(100).style.apply(highlight_new_cols, axis=0))

    # Color legend
    st.markdown("✅ Model Output Color Legend")
    legend_html = """<div style='display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px'>"""
    for model_label, color in model_colors.items():
        display_name = model_label.replace("_", " ")
        legend_html += f"<div style='background-color:{color}; padding:8px 14px; border-radius:5px; font-size:90%; "
        legend_html += f"font-weight:500;'>{display_name}</div>"
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    # ------------------- Charts -------------------
    st.subheader("📊 Prediction Summary Charts")

    for model_name in selected_models:
        model_label = model_name.replace(" ", "_")
        pred_col = f"{model_label}_Prediction"

        if pred_col in df.columns:
            pred_counts = df[pred_col].value_counts().sort_index()
            labels = ['Not Fraud' if i == 0 else 'Fraud' for i in pred_counts.index]
            values = pred_counts.values
            total = int(values.sum())

            pred_df = pd.DataFrame({
                'Prediction': labels,
                'Count': values
            })

            col1, col2 = st.columns(2)

            with col1:
                fig_bar = px.bar(
                    pred_df,
                    x='Prediction',
                    y='Count',
                    color='Prediction',
                    color_discrete_map={'Not Fraud': 'green', 'Fraud': 'red'},
                    text='Count',
                    title=f"{model_name} - Prediction Distribution (Total: {total:,})"
                )
                fig_bar.update_layout(xaxis_title="Prediction", yaxis_title="Count")
                fig_bar.update_traces(textposition='outside')
                st.plotly_chart(fig_bar, use_container_width=True, config={"staticPlot": True})

            with col2:
                fig_pie = px.pie(
                    pred_df,
                    names='Prediction',
                    values='Count',
                    color='Prediction',
                    color_discrete_map={'Not Fraud': 'green', 'Fraud': 'red'},
                    title=f"{model_name} - Fraud Pie Chart (Total: {total:,})"
                )
                fig_pie.update_traces(textinfo='label+percent+value')
                st.plotly_chart(fig_pie, use_container_width=True)

    # ------------------- Download -------------------
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction Results as CSV",
        data=csv,
        file_name="Financial-Transactions-Fraud-Detection_Results.csv",
        mime='text/csv'
    )

elif uploaded_file is None:
    st.info("📂 Please upload a CSV file to begin.")
