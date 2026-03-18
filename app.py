from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
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
PROFILE_IMAGE_PATH = Path(__file__).resolve().parent / "assets" / "pic1.png"

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

PAGE_LABELS = {
    "score": "Single Customer Scoring",
    "batch": "Batch CSV Scoring",
    "model": "Model Room",
    "eda": "EDA Lab",
    "about": "About",
}
PAGE_HELP = {
    "score": "Open the guided scoring workspace to estimate churn risk for one customer at a time.",
    "batch": "Open the batch workspace to upload a CSV and score many customer records at once.",
    "model": "Inspect how the production model was trained, evaluated, and which features matter most.",
    "eda": "Explore interactive charts that explain who churns and where portfolio risk clusters.",
    "about": "Read what the project does, how to use the app, and who built it.",
}
DEFAULT_PAGE = "score"
APP_ENTRYPOINT = "streamlit_app.py"


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

            .about-chip-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.6rem;
                margin-bottom: 1rem;
            }}

            .about-chip {{
                display: inline-block;
                padding: 0.42rem 0.85rem;
                border-radius: 999px;
                background: rgba(168, 85, 247, 0.12);
                border: 1px solid rgba(216, 180, 254, 0.18);
                color: var(--ash);
                font-size: 0.86rem;
            }}

            .footer-note {{
                margin-top: 2rem;
                padding-top: 1rem;
                border-top: 1px solid var(--border);
                color: var(--ash-soft);
                text-align: center;
                font-size: 0.84rem;
                letter-spacing: 0.04rem;
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


def normalize_page(value: str | None) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else DEFAULT_PAGE
    page = str(value or DEFAULT_PAGE)
    if page not in PAGE_LABELS:
        return DEFAULT_PAGE
    return page


def query_target(page: str) -> dict[str, str]:
    return {"page": normalize_page(page)}


def render_page_link(
    page: str,
    label: str | None = None,
    *,
    help_text: str | None = None,
    width: str = "stretch",
    disabled: bool = False,
) -> None:
    st.page_link(
        APP_ENTRYPOINT,
        label=label or PAGE_LABELS[page],
        help=help_text or PAGE_HELP[page],
        width=width,
        disabled=disabled,
        query_params=query_target(page),
    )


def style_figure(figure: go.Figure, height: int = 360) -> go.Figure:
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ASH),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        height=height,
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor="rgba(201, 203, 211, 0.08)",
        zeroline=False,
    )
    figure.update_yaxes(
        showgrid=True,
        gridcolor="rgba(201, 203, 211, 0.08)",
        zeroline=False,
    )
    return figure


def churn_mix_chart(data: pd.DataFrame) -> go.Figure:
    chart_data = (
        data["Exited"]
        .map({0: "Retained", 1: "Churned"})
        .value_counts()
        .rename_axis("status")
        .reset_index(name="customers")
    )
    figure = px.pie(
        chart_data,
        names="status",
        values="customers",
        hole=0.68,
        color="status",
        color_discrete_map={"Retained": "#6b7280", "Churned": PURPLE},
    )
    figure.update_traces(
        texttemplate="%{label}<br>%{percent}",
        hovertemplate="%{label}: %{value}<extra></extra>",
    )
    return style_figure(figure, height=340)


def geography_churn_chart(data: pd.DataFrame) -> go.Figure:
    chart_data = (
        data.groupby(["Geography", "Exited"])
        .size()
        .reset_index(name="customers")
        .assign(status=lambda frame: frame["Exited"].map({0: "Retained", 1: "Churned"}))
    )
    figure = px.bar(
        chart_data,
        x="Geography",
        y="customers",
        color="status",
        barmode="group",
        color_discrete_map={"Retained": "#4b5563", "Churned": PURPLE},
    )
    figure.update_layout(yaxis_title="Customers", xaxis_title="")
    return style_figure(figure, height=360)


def age_distribution_chart(data: pd.DataFrame) -> go.Figure:
    chart_data = data.assign(status=data["Exited"].map({0: "Retained", 1: "Churned"}))
    figure = px.box(
        chart_data,
        x="status",
        y="Age",
        color="status",
        points="outliers",
        color_discrete_map={"Retained": "#6b7280", "Churned": PURPLE},
    )
    figure.update_layout(xaxis_title="", yaxis_title="Age")
    return style_figure(figure, height=360)


def product_churn_chart(data: pd.DataFrame) -> go.Figure:
    chart_data = (
        data.groupby("NumOfProducts")["Exited"]
        .mean()
        .reset_index(name="churn_rate")
    )
    figure = px.bar(
        chart_data,
        x="NumOfProducts",
        y="churn_rate",
        text_auto=".1%",
        color="churn_rate",
        color_continuous_scale=["#312e81", "#7e22ce", "#d8b4fe"],
    )
    figure.update_layout(
        xaxis_title="Products held",
        yaxis_title="Churn rate",
        coloraxis_showscale=False,
    )
    figure.update_yaxes(tickformat=".0%")
    return style_figure(figure, height=360)


def balance_salary_chart(data: pd.DataFrame) -> go.Figure:
    sample = data.sample(min(1200, len(data)), random_state=42).copy()
    sample["status"] = sample["Exited"].map({0: "Retained", 1: "Churned"})
    figure = px.scatter(
        sample,
        x="Balance",
        y="EstimatedSalary",
        color="status",
        size="Age",
        hover_data=["Geography", "Gender", "CreditScore", "Tenure", "NumOfProducts"],
        color_discrete_map={"Retained": "#6b7280", "Churned": PURPLE},
        opacity=0.75,
    )
    figure.update_layout(
        xaxis_title="Balance",
        yaxis_title="Estimated salary",
    )
    return style_figure(figure, height=440)


def hero_section(bundle: dict, data: pd.DataFrame) -> None:
    metrics = bundle["metrics"]
    st.markdown(
        f"""
        <div class="hero-shell">
            <h1 class="hero-title">Churn Signal Studio</h1>
            <p class="hero-copy">
                A production-ready Streamlit front end for customer churn scoring, batch uploads,
                and model visibility. The underlying pipeline removes identifier leakage and serves a
                persisted model artifact for faster startup.
            </p>
            <p class="section-note" style="margin-top: 0.9rem; margin-bottom: 0;">
                Use the live links below to jump into model diagnostics or the EDA Lab.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chip_targets = [
        (
            "model",
            f"Holdout ROC-AUC {metrics['roc_auc']:.3f}",
            "Open Model Room to inspect holdout performance, feature importance, and pipeline notes.",
        ),
        (
            "model",
            f"5-fold CV ROC-AUC {metrics['cv_roc_auc_mean']:.3f}",
            "Open Model Room to review cross-validation stability and how the model generalizes.",
        ),
        (
            "eda",
            f"Portfolio churn rate {data['Exited'].mean() * 100:.1f}%",
            "Open EDA Lab to explore churn patterns, segment differences, and portfolio behavior.",
        ),
    ]
    button_row = st.columns(3, gap="small")
    for idx, (page, label, help_text) in enumerate(chip_targets):
        with button_row[idx]:
            render_page_link(page, label=label, help_text=help_text)


def build_single_record(profile: dict) -> pd.DataFrame:
    return pd.DataFrame([profile])


def render_score_page(bundle: dict, threshold: float, preset: dict) -> None:
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
                credit_score = st.slider(
                    "Credit Score",
                    350,
                    850,
                    int(preset["CreditScore"]),
                    help="Set the customer's credit score. Lower scores can raise churn risk in the model.",
                )
                age = st.slider(
                    "Age",
                    18,
                    92,
                    int(preset["Age"]),
                    help="Choose the customer's age. Older age bands behave differently in the training data.",
                )
                tenure = st.slider(
                    "Tenure",
                    0,
                    10,
                    int(preset["Tenure"]),
                    help="How many years the customer has stayed with the bank.",
                )
                geography = st.selectbox(
                    "Geography",
                    options=bundle["dataset_summary"]["geographies"],
                    index=bundle["dataset_summary"]["geographies"].index(str(preset["Geography"])),
                    help="Select the customer's country segment. Geography is an important churn signal.",
                )
                gender = st.radio(
                    "Gender",
                    options=bundle["dataset_summary"]["genders"],
                    index=bundle["dataset_summary"]["genders"].index(str(preset["Gender"])),
                    horizontal=True,
                    help="Pick the customer's gender as represented in the source dataset.",
                )
            with col_b:
                balance = st.number_input(
                    "Balance",
                    min_value=0.0,
                    max_value=250898.09,
                    value=float(preset["Balance"]),
                    step=500.0,
                    help="Set the customer's current account balance.",
                )
                estimated_salary = st.number_input(
                    "Estimated Salary",
                    min_value=0.0,
                    max_value=200000.0,
                    value=float(preset["EstimatedSalary"]),
                    step=500.0,
                    help="Estimated annual salary for the customer.",
                )
                num_products = st.select_slider(
                    "Number of Products",
                    options=[1, 2, 3, 4],
                    value=int(preset["NumOfProducts"]),
                    help="How many banking products the customer currently holds.",
                )
                has_card = st.radio(
                    "Has Credit Card",
                    options=[1, 0],
                    index=0 if int(preset["HasCrCard"]) == 1 else 1,
                    format_func=lambda value: "Yes" if value == 1 else "No",
                    horizontal=True,
                    help="Indicate whether the customer has a credit card.",
                )
                is_active = st.radio(
                    "Is Active Member",
                    options=[1, 0],
                    index=0 if int(preset["IsActiveMember"]) == 1 else 1,
                    format_func=lambda value: "Active" if value == 1 else "Inactive",
                    horizontal=True,
                    help="Tell the model whether the customer is active or inactive.",
                )
            submitted = st.form_submit_button(
                "Predict churn risk",
                help="Score the profile and estimate the likelihood that this customer will churn.",
            )

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


def render_batch_page(bundle: dict, threshold: float) -> None:
    st.markdown("### Batch Scoring")
    st.markdown(
        '<p class="section-note">Upload a CSV with raw customer features and download a scored file with probability, risk band, and prediction.</p>',
        unsafe_allow_html=True,
    )
    st.code(
        ", ".join(bundle["feature_columns"]),
        language="text",
    )
    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        help="Upload a CSV containing the model input columns so the app can score many customers at once.",
    )
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
                help="Download the scored dataset with churn probability, risk band, and prediction columns.",
            )
        except Exception as exc:
            st.error(str(exc))


def render_model_page(bundle: dict, metrics: dict) -> None:
    st.markdown("### Model Room")
    st.markdown(
        '<p class="section-note">A transparent snapshot of what is powering the app and how it was trained.</p>',
        unsafe_allow_html=True,
    )
    model_col, chart_col = st.columns([0.9, 1.1], gap="large")
    with model_col:
        st.markdown(
            """
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


def render_eda_page(data: pd.DataFrame) -> None:
    st.markdown("### EDA Lab")
    st.markdown(
        '<p class="section-note">Explore who churns, where risk clusters, and which portfolio traits deserve the next business experiment.</p>',
        unsafe_allow_html=True,
    )

    summary_row = st.columns(4)
    with summary_row[0]:
        metric_card("Customers", f"{len(data):,}")
    with summary_row[1]:
        metric_card("Churn rate", format_percent(data["Exited"].mean()))
    with summary_row[2]:
        metric_card("Median age", f"{int(data['Age'].median())}")
    with summary_row[3]:
        metric_card("Median balance", f"${data['Balance'].median():,.0f}")

    row_one = st.columns(2, gap="large")
    with row_one[0]:
        st.plotly_chart(churn_mix_chart(data), use_container_width=True)
    with row_one[1]:
        st.plotly_chart(geography_churn_chart(data), use_container_width=True)

    row_two = st.columns(2, gap="large")
    with row_two[0]:
        st.plotly_chart(age_distribution_chart(data), use_container_width=True)
    with row_two[1]:
        st.plotly_chart(product_churn_chart(data), use_container_width=True)

    st.plotly_chart(balance_salary_chart(data), use_container_width=True)

    insights = [
        "Customers in Germany churn at a visibly higher rate than the France and Spain segments.",
        "Older customers show a wider spread and higher upper-tail churn pattern than younger cohorts.",
        "Accounts with fewer products carry more churn pressure, which supports cross-sell and engagement plays.",
        "Balance alone is not the story; churn also clusters around inactivity and product concentration.",
    ]
    insight_markup = "".join(f"<li>{item}</li>" for item in insights)
    st.markdown(
        f"""
        <div class="glass-card">
            <strong>EDA readout</strong>
            <p class="hero-copy" style="font-size:0.98rem; margin-top:0.6rem;">
                This page is meant to turn the model back into business intuition. Use it to frame retention experiments,
                segment reviews, and stakeholder conversations before you move into scoring.
            </p>
            <ul class="hero-copy" style="margin-top:0.3rem;">
                {insight_markup}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_about_page(bundle: dict, data: pd.DataFrame) -> None:
    stack_items = [
        "Python",
        "Streamlit",
        "BigQuery",
        "Supabase",
        "Hugging Face",
        "PySpark",
        "SQL",
        "ML",
        "LLMs",
        "Transformers",
    ]
    stack_markup = "".join(f'<span class="about-chip">{item}</span>' for item in stack_items)
    project_lines = [
        "Risk scoring systems",
        "AB tests",
        "AI-powered dashboards",
        "Retrieval-Augmented Generation (RAG) pipelines",
        "Predictive analytics solutions",
        "LLM based Solutions",
    ]
    project_markup = "".join(f"<li>{item}</li>" for item in project_lines)
    st.markdown("### About This Project")
    intro_col, guide_col = st.columns([1.05, 0.95], gap="large")
    with intro_col:
        st.markdown(
            f"""
            <div class="glass-card">
                <strong style="font-size: 1.25rem; color: white;">Customer churn prediction for proactive retention</strong>
                <p class="hero-copy" style="margin-top: 0.65rem;">
                    This project predicts whether a bank customer is likely to churn, using demographic and financial
                    attributes from <code>Churn_Modelling.csv</code>. The target variable is <code>Exited</code>,
                    where <code>1</code> means the customer churned and <code>0</code> means the customer stayed.
                </p>
                <p class="hero-copy">
                    The goal is to help teams act before churn happens: identify at-risk customers, prioritize outreach,
                    and support better retention decisions with data instead of guesswork.
                </p>
                <p class="hero-copy">
                    The production app keeps the original project intent from the README, but turns it into a usable tool:
                    one-customer scoring, batch CSV scoring, model diagnostics, and EDA for stakeholder storytelling.
                </p>
                <p class="section-note" style="margin-top: 1rem; margin-bottom: 0.2rem;">Key data signals used</p>
                <p class="hero-copy" style="margin-top: 0.2rem;">
                    Credit score, geography, gender, age, tenure, balance, number of products, credit card ownership,
                    activity status, and estimated salary.
                </p>
                <p class="hero-copy" style="margin-top: 0.9rem;">
                    The current model was trained on <strong>{bundle['dataset_summary']['rows']:,}</strong> customer records,
                    with an observed churn rate of <strong>{data['Exited'].mean() * 100:.1f}%</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with guide_col:
        st.markdown(
            """
            <div class="glass-card">
                <strong style="font-size: 1.25rem; color: white;">How to use the app</strong>
                <ol class="hero-copy" style="margin-top: 0.7rem; padding-left: 1.2rem;">
                    <li>Use <strong>Single Customer Scoring</strong> when you want to assess one customer profile manually.</li>
                    <li>Use <strong>Batch CSV Scoring</strong> when you have a file of many customers to score in one pass.</li>
                    <li>Use <strong>Model Room</strong> to understand model quality, validation performance, and feature importance.</li>
                    <li>Use <strong>EDA Lab</strong> to explore who churns, where risk clusters, and what patterns deserve business action.</li>
                    <li>Adjust the decision threshold when you want stricter or looser churn alerts.</li>
                </ol>
                <p class="hero-copy" style="margin-top: 0.9rem;">
                    Hover over links, buttons, and form controls throughout the app to see quick explanations of what each action does.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("### About the Author")
    image_col, content_col = st.columns([0.38, 0.62], gap="large")
    with image_col:
        if PROFILE_IMAGE_PATH.exists():
            st.image(
                str(PROFILE_IMAGE_PATH),
                use_container_width=True,
                caption="Okon Prince",
            )
    with content_col:
        st.markdown(
            f"""
            <div class="glass-card">
                <strong style="font-size: 1.35rem; color: white;">Okon Prince</strong>
                <p class="hero-copy" style="font-size: 1.05rem; margin-top: 0.6rem;">AI Engineer &amp; Data Scientist</p>
                <p class="hero-copy">
                    I design and deploy end-to-end data systems — from raw data to production-ready intelligence.
                </p>
                <p class="section-note" style="margin-top: 1rem; margin-bottom: 0.2rem;">Core stack</p>
                <div class="about-chip-row">{stack_markup}</div>
                <p class="section-note" style="margin-top: 0.8rem; margin-bottom: 0.3rem;">Projects span</p>
                <ul class="hero-copy" style="margin-top: 0.2rem;">
                    {project_markup}
                </ul>
                <p class="hero-copy" style="margin-top: 1rem;">
                    <strong>I believe:</strong> models are trained, systems are engineered, impact is delivered.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-note">© Okon Prince, 2026</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_styles()
    bundle = load_artifacts()
    data = reference_data()
    metrics = bundle["metrics"]
    defaults = bundle["input_defaults"]

    active_page = normalize_page(st.query_params.get("page", DEFAULT_PAGE))

    threshold = float(metrics.get("threshold", DEFAULT_THRESHOLD))
    starter_profile = list(SAMPLE_PROFILES.keys())[0]

    with st.sidebar:
        st.markdown("### Control Deck")
        st.caption(f"Current workspace: {PAGE_LABELS[active_page]}")
        for page_key in PAGE_LABELS:
            render_page_link(
                page_key,
                help_text=PAGE_HELP[page_key],
                disabled=page_key == active_page,
            )
        st.write("")

        if active_page in {"score", "batch"}:
            threshold = st.slider(
                "Decision threshold",
                min_value=0.10,
                max_value=0.90,
                value=float(metrics.get("threshold", DEFAULT_THRESHOLD)),
                step=0.01,
                help="Set the probability cutoff used to classify a customer as likely to churn.",
            )

        if active_page == "score":
            starter_profile = st.selectbox(
                "Starter profile",
                options=list(SAMPLE_PROFILES.keys()),
                index=0,
                help="Load a sample customer profile into the form so you can start from a realistic scenario.",
            )
            st.markdown(
                '<p class="section-note">Use the starter profile to prefill the form, then fine-tune any customer field.</p>',
                unsafe_allow_html=True,
            )
        elif active_page == "eda":
            st.markdown(
                '<p class="section-note">The EDA Lab turns the raw bank portfolio into segment patterns and risk storylines.</p>',
                unsafe_allow_html=True,
            )
        elif active_page == "about":
            st.markdown(
                '<p class="section-note">Read the project overview, usage guide, and the author profile in one place.</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="section-note">Use the workspace links above to move across the app. Hover them to see what each page is for.</p>',
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

    if active_page == "score":
        render_score_page(bundle, threshold, preset)
    elif active_page == "batch":
        render_batch_page(bundle, threshold)
    elif active_page == "model":
        render_model_page(bundle, metrics)
    elif active_page == "eda":
        render_eda_page(data)
    else:
        render_about_page(bundle, data)

    render_footer()


if __name__ == "__main__":
    main()
