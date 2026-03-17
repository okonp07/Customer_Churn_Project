from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.modeling import (
    DATA_PATH,
    DEFAULT_THRESHOLD,
    MODEL_PATH,
    load_bundle,
    load_dataset,
    risk_band,
    score_records,
    train_and_persist,
)

st.set_page_config(
    page_title="Churn Signal Studio",
    page_icon=":crystal_ball:",
    layout="wide",
    initial_sidebar_state="expanded",
)


PURPLE = "#a855f7"
PURPLE_BRIGHT = "#d8b4fe"
ASH = "#c9cbd3"
ASH_SOFT = "#9da3b4"
BLACK = "#05010d"
SURFACE = "#14111d"
SURFACE_ALT = "#221a2f"

SAMPLE_PROFILES = {
    "Portfolio median": {
        "CreditScore": 652,
        "Geography": "France",
        "Gender": "Male",
        "Age": 39,
        "Tenure": 5,
        "Balance": 97198.54,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 100193.91,
    },
    "High churn risk": {
        "CreditScore": 487,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 52,
        "Tenure": 2,
        "Balance": 148500.0,
        "NumOfProducts": 1,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 72500.0,
    },
    "Likely to stay": {
        "CreditScore": 731,
        "Geography": "France",
        "Gender": "Male",
        "Age": 33,
        "Tenure": 7,
        "Balance": 0.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 121800.0,
    },
}


def inject_styles() -> None:
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

            :root {{
                --purple: {PURPLE};
                --purple-bright: {PURPLE_BRIGHT};
                --ash: {ASH};
                --ash-soft: {ASH_SOFT};
                --black: {BLACK};
                --surface: {SURFACE};
                --surface-alt: {SURFACE_ALT};
                --border: rgba(201, 203, 211, 0.14);
            }}

            html, body, [class*="css"] {{
                font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
            }}

            .stApp {{
                color: var(--ash);
                background:
                    radial-gradient(circle at top left, rgba(168, 85, 247, 0.30), transparent 28%),
                    radial-gradient(circle at top right, rgba(88, 28, 135, 0.35), transparent 24%),
                    linear-gradient(180deg, #05010d 0%, #0b0812 45%, #17121e 100%);
            }}

            [data-testid="stSidebar"] {{
                background:
                    linear-gradient(180deg, rgba(34, 26, 47, 0.96) 0%, rgba(10, 8, 15, 0.98) 100%);
                border-right: 1px solid var(--border);
            }}

            .block-container {{
                padding-top: 2rem;
                padding-bottom: 2.5rem;
            }}

            .hero-shell {{
                padding: 1.5rem 1.6rem;
                border-radius: 24px;
                border: 1px solid var(--border);
                background:
                    linear-gradient(135deg, rgba(12, 10, 18, 0.92), rgba(34, 26, 47, 0.86)),
                    linear-gradient(135deg, rgba(168, 85, 247, 0.2), rgba(5, 1, 13, 0.2));
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.30);
            }}

            .hero-kicker {{
                text-transform: uppercase;
                letter-spacing: 0.18rem;
                font-size: 0.78rem;
                color: var(--purple-bright);
                margin-bottom: 0.5rem;
            }}

            .hero-title {{
                font-size: 2.5rem;
                line-height: 1.05;
                margin: 0;
                color: white;
            }}

            .hero-copy {{
                margin-top: 0.8rem;
                max-width: 42rem;
                font-size: 1rem;
                color: var(--ash);
            }}

            .glass-card {{
                padding: 1.1rem 1.2rem;
                border-radius: 22px;
                background: rgba(20, 17, 29, 0.72);
                border: 1px solid var(--border);
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
            }}

            .metric-chip {{
                display: inline-block;
                margin-top: 0.8rem;
                margin-right: 0.55rem;
                padding: 0.45rem 0.8rem;
                border-radius: 999px;
                background: rgba(168, 85, 247, 0.12);
                border: 1px solid rgba(216, 180, 254, 0.18);
                color: var(--ash);
                font-size: 0.85rem;
            }}

            .section-note {{
                color: var(--ash-soft);
                font-size: 0.92rem;
                margin-bottom: 0.8rem;
            }}

            div[data-testid="stMetric"] {{
                background: rgba(20, 17, 29, 0.76);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 1rem;
            }}

            div[data-testid="stMetric"] label {{
                color: var(--ash-soft);
            }}

            div[data-testid="stMetricValue"] {{
                color: white;
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: 0.75rem;
            }}

            .stTabs [data-baseweb="tab"] {{
                border-radius: 999px;
                padding: 0.6rem 1rem;
                color: var(--ash);
                background: rgba(20, 17, 29, 0.7);
                border: 1px solid var(--border);
            }}

            .stTabs [aria-selected="true"] {{
                background: linear-gradient(135deg, rgba(168, 85, 247, 0.95), rgba(109, 40, 217, 0.95));
                color: white;
            }}

            .stButton > button, .stDownloadButton > button {{
                border-radius: 999px;
                border: none;
                background: linear-gradient(135deg, #8b5cf6, #6d28d9);
                color: white;
                padding: 0.7rem 1.15rem;
                font-weight: 700;
            }}

            .stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {{
                color: var(--ash);
            }}

            .stDataFrame {{
                border: 1px solid var(--border);
                border-radius: 18px;
                overflow: hidden;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_artifacts() -> dict:
    if Path(MODEL_PATH).exists():
        try:
            return load_bundle()
        except Exception:
            # Rebuild the artifact if the serialized model can't be loaded
            # in the current deployment environment.
            return train_and_persist()
    return train_and_persist()


@st.cache_data(show_spinner=False)
def reference_data() -> pd.DataFrame:
    return load_dataset(DATA_PATH)


def metric_card(label: str, value: str, delta: str | None = None) -> None:
    st.metric(label, value, delta=delta)


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def recommendation_block(probability: float) -> str:
    band = risk_band(probability)
    if band == "High":
        return (
            "Open a save play immediately: trigger a personal outreach, present a retention offer, "
            "and review inactivity plus product concentration."
        )
    if band == "Medium":
        return (
            "Watch this account closely: use proactive nudges, product education, and a short-term check-in "
            "before churn intent hardens."
        )
    return (
        "This customer looks relatively stable. Keep engagement warm with loyalty messaging and monitor for "
        "future behavior shifts."
    )


def probability_gauge(probability: float, threshold: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"color": "white", "size": 32}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": ASH_SOFT},
                "bar": {"color": PURPLE},
                "bgcolor": SURFACE,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#1f2937"},
                    {"range": [40, 70], "color": "#4338ca"},
                    {"range": [70, 100], "color": "#7e22ce"},
                ],
                "threshold": {
                    "line": {"color": PURPLE_BRIGHT, "width": 5},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
            title={"text": "Churn Probability", "font": {"color": ASH}},
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=10),
        height=300,
    )
    return figure


def importance_chart(feature_importance: list[dict]) -> go.Figure:
    importance_frame = pd.DataFrame(feature_importance).sort_values("importance")
    figure = go.Figure(
        go.Bar(
            x=importance_frame["importance"],
            y=importance_frame["feature"],
            orientation="h",
            marker=dict(
                color=importance_frame["importance"],
                colorscale=[
                    [0.0, "#3b0764"],
                    [0.4, "#7e22ce"],
                    [1.0, "#d8b4fe"],
                ],
            ),
        )
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ASH),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Relative importance",
        yaxis_title="",
        height=420,
    )
    return figure


def hero_section(bundle: dict, data: pd.DataFrame) -> None:
    metrics = bundle["metrics"]
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-kicker">Purple / Black / Ash deployment</div>
            <h1 class="hero-title">Churn Signal Studio</h1>
            <p class="hero-copy">
                A production-ready Streamlit front end for customer churn scoring, batch uploads,
                and model visibility. The underlying pipeline removes identifier leakage and serves a
                persisted model artifact for faster startup.
            </p>
            <span class="metric-chip">Holdout ROC-AUC {metrics['roc_auc']:.3f}</span>
            <span class="metric-chip">5-fold CV ROC-AUC {metrics['cv_roc_auc_mean']:.3f}</span>
            <span class="metric-chip">Portfolio churn rate {data['Exited'].mean() * 100:.1f}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_single_record(profile: dict) -> pd.DataFrame:
    return pd.DataFrame([profile])


def main() -> None:
    inject_styles()
    bundle = load_artifacts()
    data = reference_data()
    metrics = bundle["metrics"]
    defaults = bundle["input_defaults"]

    with st.sidebar:
        st.markdown("### Control Deck")
        threshold = st.slider(
            "Decision threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(metrics.get("threshold", DEFAULT_THRESHOLD)),
            step=0.01,
        )
        starter_profile = st.selectbox(
            "Starter profile",
            options=list(SAMPLE_PROFILES.keys()),
            index=0,
        )
        st.markdown(
            '<p class="section-note">Use the starter profile to prefill the form, then fine-tune any customer field.</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="glass-card">
                <strong>{bundle['model_name']}</strong><br>
                <span class="section-note">Artifact: {Path(MODEL_PATH).name}</span><br>
                <span class="section-note">Built from {bundle['dataset_summary']['rows']:,} customer records</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    preset = {**defaults, **SAMPLE_PROFILES[starter_profile]}

    hero_section(bundle, data)
    st.write("")

    top_row = st.columns(4)
    with top_row[0]:
        metric_card("Holdout ROC-AUC", f"{metrics['roc_auc']:.3f}")
    with top_row[1]:
        metric_card("Cross-val ROC-AUC", f"{metrics['cv_roc_auc_mean']:.3f}", f"+/- {metrics['cv_roc_auc_std']:.3f}")
    with top_row[2]:
        metric_card("Accuracy", format_percent(metrics["accuracy"]))
    with top_row[3]:
        metric_card("Recall", format_percent(metrics["recall"]))

    tab_score, tab_batch, tab_model = st.tabs(
        ["Single Customer Scoring", "Batch CSV Scoring", "Model Room"]
    )

    with tab_score:
        left_col, right_col = st.columns([1.2, 0.8], gap="large")
        with left_col:
            st.markdown("### Profile Builder")
            st.markdown(
                '<p class="section-note">Shape a customer profile and score churn likelihood instantly.</p>',
                unsafe_allow_html=True,
            )
            with st.form("score_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    credit_score = st.slider("Credit Score", 350, 850, int(preset["CreditScore"]))
                    age = st.slider("Age", 18, 92, int(preset["Age"]))
                    tenure = st.slider("Tenure", 0, 10, int(preset["Tenure"]))
                    geography = st.selectbox(
                        "Geography",
                        options=bundle["dataset_summary"]["geographies"],
                        index=bundle["dataset_summary"]["geographies"].index(str(preset["Geography"])),
                    )
                    gender = st.radio(
                        "Gender",
                        options=bundle["dataset_summary"]["genders"],
                        index=bundle["dataset_summary"]["genders"].index(str(preset["Gender"])),
                        horizontal=True,
                    )
                with col_b:
                    balance = st.number_input(
                        "Balance",
                        min_value=0.0,
                        max_value=250898.09,
                        value=float(preset["Balance"]),
                        step=500.0,
                    )
                    estimated_salary = st.number_input(
                        "Estimated Salary",
                        min_value=0.0,
                        max_value=200000.0,
                        value=float(preset["EstimatedSalary"]),
                        step=500.0,
                    )
                    num_products = st.select_slider(
                        "Number of Products",
                        options=[1, 2, 3, 4],
                        value=int(preset["NumOfProducts"]),
                    )
                    has_card = st.radio(
                        "Has Credit Card",
                        options=[1, 0],
                        index=0 if int(preset["HasCrCard"]) == 1 else 1,
                        format_func=lambda value: "Yes" if value == 1 else "No",
                        horizontal=True,
                    )
                    is_active = st.radio(
                        "Is Active Member",
                        options=[1, 0],
                        index=0 if int(preset["IsActiveMember"]) == 1 else 1,
                        format_func=lambda value: "Active" if value == 1 else "Inactive",
                        horizontal=True,
                    )
                submitted = st.form_submit_button("Predict churn risk")

            profile = {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_card,
                "IsActiveMember": is_active,
                "EstimatedSalary": estimated_salary,
            }

            if submitted:
                st.session_state["latest_profile"] = profile

        with right_col:
            latest_profile = st.session_state.get("latest_profile", preset)
            result = score_records(build_single_record(latest_profile), bundle, threshold=threshold).iloc[0]
            probability = float(result["churn_probability"])
            st.markdown("### Risk Readout")
            st.plotly_chart(probability_gauge(probability, threshold), use_container_width=True)

            summary_left, summary_right = st.columns(2)
            with summary_left:
                metric_card("Risk band", result["risk_band"])
            with summary_right:
                metric_card("Prediction", result["prediction"])

            st.markdown(
                f"""
                <div class="glass-card">
                    <strong>Recommended action</strong>
                    <p class="hero-copy" style="font-size:0.98rem; margin-top:0.6rem;">
                        {recommendation_block(probability)}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.dataframe(pd.DataFrame([latest_profile]), use_container_width=True, hide_index=True)

    with tab_batch:
        st.markdown("### Batch Scoring")
        st.markdown(
            '<p class="section-note">Upload a CSV with raw customer features and download a scored file with probability, risk band, and prediction.</p>',
            unsafe_allow_html=True,
        )
        st.code(
            ", ".join(bundle["feature_columns"]),
            language="text",
        )
        uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_frame = pd.read_csv(uploaded_file)
                scored_frame = score_records(batch_frame, bundle, threshold=threshold)
                st.success(f"Scored {len(scored_frame):,} customer records.")
                st.dataframe(scored_frame.head(50), use_container_width=True)
                st.download_button(
                    "Download scored CSV",
                    data=scored_frame.to_csv(index=False).encode("utf-8"),
                    file_name="churn_scored_output.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(str(exc))

    with tab_model:
        st.markdown("### Model Room")
        st.markdown(
            '<p class="section-note">A transparent snapshot of what is powering the app and how it was trained.</p>',
            unsafe_allow_html=True,
        )
        model_col, chart_col = st.columns([0.9, 1.1], gap="large")
        with model_col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <strong>Pipeline notes</strong>
                    <p class="hero-copy" style="font-size:0.98rem; margin-top:0.6rem;">
                        The production pipeline drops RowNumber, CustomerId, and Surname, imputes missing values,
                        one-hot encodes Geography and Gender, and scores with a class-weighted Random Forest.
                    </p>
                    <p class="hero-copy" style="font-size:0.98rem;">
                        This avoids the notebook's identifier leakage and skips synthetic category generation from SMOTE.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            summary_frame = pd.DataFrame(
                {
                    "Metric": [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1",
                        "Holdout ROC-AUC",
                        "5-fold CV ROC-AUC",
                    ],
                    "Value": [
                        f"{metrics['accuracy']:.3f}",
                        f"{metrics['precision']:.3f}",
                        f"{metrics['recall']:.3f}",
                        f"{metrics['f1']:.3f}",
                        f"{metrics['roc_auc']:.3f}",
                        f"{metrics['cv_roc_auc_mean']:.3f} +/- {metrics['cv_roc_auc_std']:.3f}",
                    ],
                }
            )
            st.dataframe(summary_frame, use_container_width=True, hide_index=True)
        with chart_col:
            st.plotly_chart(
                importance_chart(bundle["feature_importance"]),
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
