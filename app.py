"""
Used Cars Price Predictor & Market Explorer
-------------------------------------------
Streamlit app built on top of the EDA + Modeling notebooks.

Run:   streamlit run app.py
Needs: model.joblib  (saved from the Modeling notebook)
       data/cleaned_cars.csv  (or cleaned_cars.csv next to this file)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Used Cars Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        div[data-testid="stMetric"] {
            background: rgba(37, 99, 235, 0.07);
            border: 1px solid rgba(37, 99, 235, 0.25);
            padding: 14px 18px;
            border-radius: 12px;
        }
        .hero {
            background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
            padding: 28px 34px; border-radius: 16px; margin-bottom: 18px;
        }
        .hero h1 {color: white; margin: 0; font-size: 2.1rem;}
        .hero p  {color: #bfdbfe; margin: 6px 0 0 0; font-size: 1.05rem;}
        .hero .author {color: #e0e7ff; font-size: 0.95rem; margin-top: 14px;
                       padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.25);}
        .author-card {
            border: 1px solid rgba(37, 99, 235, 0.3); border-radius: 12px;
            padding: 12px 14px; background: rgba(37, 99, 235, 0.06);
        }
        .author-card b {font-size: 1.0rem;}
        .footer {text-align: center; opacity: 0.7; font-size: 0.85rem; padding: 10px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE = Path(__file__).parent
CURRENT_YEAR = 2026

# ---- Author / ownership details (edit these) --------------------------------
AUTHOR_NAME = "Rahma Ashraf Ali Faragallah"
AUTHOR_TITLE = "Data Scientist & AI Engineer"
LINKEDIN_URL = ""   # e.g. "https://www.linkedin.com/in/your-profile"
GITHUB_URL = ""     # e.g. "https://github.com/your-username"
# -----------------------------------------------------------------------------
LUXURY_BRANDS = {"Mercedes-Benz", "BMW", "Audi"}
STATE_NAMES = {
    "CA": "California", "NY": "New York", "FL": "Florida", "TX": "Texas",
    "MI": "Michigan", "IL": "Illinois", "GA": "Georgia", "PA": "Pennsylvania",
    "OH": "Ohio", "NC": "North Carolina", "Unknown": "Unknown",
}
DEFAULT_METRICS = [
    {"Model": "XGBoost", "R2 ($)": 0.984, "MAE ($)": 913, "RMSE ($)": 1639},
    {"Model": "Random Forest", "R2 ($)": 0.962, "MAE ($)": 1542, "RMSE ($)": 2502},
    {"Model": "Linear Regression", "R2 ($)": 0.736, "MAE ($)": 3046, "RMSE ($)": 6609},
]
DEFAULT_IMPORTANCE = {
    "Age": 25.4, "Model": 23.9, "Mileage": 15.6, "Is_Luxury": 9.8,
    "Service_History": 6.9, "Accident_History": 5.0, "Location": 2.1,
    "Color": 2.1, "Mileage_per_Year": 1.6, "Owners": 1.4,
}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    path = BASE / "model.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data(show_spinner="Loading data...")
def load_data():
    for p in (BASE / "data" / "cleaned_cars.csv", BASE / "cleaned_cars.csv"):
        if p.exists():
            return pd.read_csv(p)
    return None


bundle = load_bundle()
df = load_data()

if bundle is None or df is None:
    st.error(
        "Missing files. Place **model.joblib** and **cleaned_cars.csv** "
        "(or `data/cleaned_cars.csv`) next to `app.py`."
    )
    st.stop()

pipe = bundle["model"]
FEATURES = bundle["features"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def spec_defaults(model: str, fuel: str) -> dict:
    """Typical technical specs for a model + fuel type (median of the dataset)."""
    sub = df[(df["Model"] == model) & (df["Fuel_Type"] == fuel)]
    if len(sub) < 3:
        sub = df[df["Model"] == model]
    out = {c: float(sub[c].median()) for c in ["Engine_Size", "Horsepower", "Torque", "Fuel_Efficiency"]}
    if fuel == "Electric":
        out["Engine_Size"] = 0.0
    return out


def build_frame(rows: list) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    return frame[FEATURES]


def predict(frame: pd.DataFrame) -> np.ndarray:
    return np.clip(np.expm1(pipe.predict(frame)), 500, None)


def error_pct(price: float) -> float:
    edges, errs = bundle["edges"], bundle["typical_error_pct"]
    idx = int(np.searchsorted(edges, price, side="right") - 1)
    return float(errs[min(max(idx, 0), len(errs) - 1)])


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def show(fig):
    """Render a Plotly figure full-width (works on old and new Streamlit versions)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(frame: pd.DataFrame):
    try:
        st.dataframe(frame, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(frame, hide_index=True, use_container_width=True)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚗 Used Cars Analytics")
    st.caption("End-to-end data analysis project: EDA, feature engineering, XGBoost and SHAP.")
    st.divider()
    st.metric("Cars analysed", f"{len(df):,}")
    st.metric("Brands", df["Make"].nunique())
    st.metric("Models", df["Model"].nunique())
    st.divider()
    links = []
    if LINKEDIN_URL:
        links.append(f"[LinkedIn]({LINKEDIN_URL})")
    if GITHUB_URL:
        links.append(f"[GitHub]({GITHUB_URL})")
    st.markdown(
        f"""<div class="author-card">👩‍💻 <b>{AUTHOR_NAME}</b><br>
        <span style="opacity:0.8">{AUTHOR_TITLE}</span></div>""",
        unsafe_allow_html=True,
    )
    if links:
        st.markdown(" · ".join(links))

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>🚗 Used Cars Price Predictor</h1>
        <p>Estimate a fair price, explore the market and understand what drives car prices.</p>
        <div class="author">Created by <b>{AUTHOR_NAME}</b> · {AUTHOR_TITLE}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_pred, tab_market, tab_model, tab_about = st.tabs(
    ["💰 Price Predictor", "📊 Market Explorer", "🧠 Model Insights", "ℹ️ About"]
)

# ============================================================================
# TAB 1: PRICE PREDICTOR
# ============================================================================
with tab_pred:
    left, right = st.columns([1.05, 1.25], gap="large")

    with left:
        st.subheader("Car details")

        c1, c2 = st.columns(2)
        make = c1.selectbox("Brand", sorted(df["Make"].unique()))
        models = sorted(df.loc[df["Make"] == make, "Model"].unique())
        model = c2.selectbox("Model", models)
        model_df = df[df["Model"] == model]

        c3, c4 = st.columns(2)
        year = c3.slider("Year", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].median()))
        owners = c4.slider("Previous owners", int(df["Owners"].min()), int(df["Owners"].max()), 2)

        age = CURRENT_YEAR - year
        mileage = st.number_input(
            "Mileage (miles)", min_value=0, max_value=int(df["Mileage"].max() * 1.2),
            value=int(model_df["Mileage"].median()), step=1000,
        )

        c5, c6 = st.columns(2)
        fuels = sorted(df["Fuel_Type"].unique())
        fuel = c5.selectbox("Fuel type", fuels, index=fuels.index(model_df["Fuel_Type"].mode()[0]))
        trans_opts = sorted(df["Transmission"].dropna().unique())
        transmission = c6.selectbox("Transmission", trans_opts, index=trans_opts.index(model_df["Transmission"].mode()[0]))

        c7, c8 = st.columns(2)
        accident = c7.selectbox("Accident history", ["No", "Yes", "Unknown"])
        service_opts = ["Full Service", "Partial Service", "No Service", "Unknown"]
        service = c8.selectbox("Service history", service_opts)

        with st.expander("Advanced specifications"):
            defaults = spec_defaults(model, fuel)
            k = f"{model}_{fuel}"
            a1, a2 = st.columns(2)
            body_opts = sorted(df["Body_Type"].unique())
            body = a1.selectbox("Body type", body_opts, index=body_opts.index(model_df["Body_Type"].mode()[0]), key=f"body_{model}")
            drive_opts = sorted(df["Drivetrain"].unique())
            drivetrain = a2.selectbox("Drivetrain", drive_opts, index=drive_opts.index(model_df["Drivetrain"].mode()[0]), key=f"drive_{model}")

            b1, b2 = st.columns(2)
            engine = b1.number_input("Engine size (L)", 0.0, 8.0, round(defaults["Engine_Size"], 1), 0.1, key=f"eng_{k}")
            hp = b2.number_input("Horsepower", 0.0, 800.0, float(round(defaults["Horsepower"])), 5.0, key=f"hp_{k}")

            b3, b4 = st.columns(2)
            torque = b3.number_input("Torque", 0.0, 800.0, float(round(defaults["Torque"])), 5.0, key=f"tq_{k}")
            eff = b4.number_input("Fuel efficiency", 0.0, 160.0, float(round(defaults["Fuel_Efficiency"], 1)), 0.5, key=f"fe_{k}")

            colors = sorted(df["Color"].unique())
            color = b1.selectbox("Color", colors, index=colors.index("Unknown") if "Unknown" in colors else 0)
            locs = sorted(df["Location"].unique())
            location = b2.selectbox("Location", locs, format_func=lambda s: f"{s} · {STATE_NAMES.get(s, s)}",
                                    index=locs.index("Unknown") if "Unknown" in locs else 0)

    # ---- one car → features
    car = {
        "Make": make, "Model": model, "Fuel_Type": fuel, "Transmission": transmission,
        "Engine_Size": engine, "Mileage": mileage, "Horsepower": hp, "Torque": torque,
        "Owners": owners, "Accident_History": accident, "Service_History": service,
        "Color": color, "Body_Type": body, "Drivetrain": drivetrain, "Fuel_Efficiency": eff,
        "Location": location, "Age": age, "Mileage_per_Year": mileage / max(age, 1),
        "Is_Luxury": int(make in LUXURY_BRANDS),
    }
    price = float(predict(build_frame([car]))[0])
    err = error_pct(price)
    low, high = max(500.0, price * (1 - err / 100)), price * (1 + err / 100)

    with right:
        st.subheader("Estimated price")
        st.metric("Predicted selling price", fmt_money(price))
        st.caption(f"Likely range: **{fmt_money(low)} – {fmt_money(high)}**  (3 out of 4 similar predictions fall within ±{err:.0f}%)")

        m1, m2 = st.columns(2)
        model_median = float(model_df["Selling_Price"].median())
        delta = (price / model_median - 1) * 100
        m1.metric(f"Median {model} price", fmt_money(model_median), f"{delta:+.0f}% vs this car", delta_color="off")
        pct = float((df["Selling_Price"] < price).mean() * 100)
        m2.metric("Market position", f"Top {100 - pct:.0f}%" if pct >= 50 else f"Bottom {pct:.0f}%",
                  help="Where this price sits among all cars in the dataset")

        if price <= 600:
            st.info("This car sits at the **$500 price floor**: old, high-mileage cars in the data often sell at the minimum price.")

        # ---- depreciation curve for this exact car
        st.markdown("##### How this car's value changes with age")
        ages = list(range(2, 21))
        mpy = mileage / max(age, 1)
        rows = [{**car, "Age": a, "Mileage": mpy * a, "Mileage_per_Year": mpy} for a in ages]
        curve = predict(build_frame(rows))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ages, y=curve, mode="lines", name="Predicted price",
                                 line=dict(color="#2563eb", width=3), fill="tozeroy",
                                 fillcolor="rgba(37,99,235,0.10)"))
        fig.add_trace(go.Scatter(x=[max(age, 2)], y=[price], mode="markers", name="Your car",
                                 marker=dict(size=14, color="#ef4444", line=dict(color="white", width=2))))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Car age (years)", yaxis_title="Price ($)",
                          legend=dict(orientation="h", y=1.12))
        show(fig)
        st.caption("Assumes the same yearly mileage and number of owners as entered.")

# ============================================================================
# TAB 2: MARKET EXPLORER
# ============================================================================
with tab_market:
    st.subheader("Explore the market")

    f1, f2, f3 = st.columns(3)
    all_makes = sorted(df["Make"].unique())
    all_fuels = sorted(df["Fuel_Type"].unique())
    sel_makes = f1.multiselect("Brand", all_makes, default=all_makes)
    sel_fuels = f2.multiselect("Fuel type", all_fuels, default=all_fuels)
    a_min, a_max = int(df["Age"].min()), int(df["Age"].max())
    sel_age = f3.slider("Car age (years)", a_min, a_max, (a_min, a_max))
    hide_floor = st.checkbox("Hide cars at the $500 price floor", value=False)

    view = df[df["Make"].isin(sel_makes) & df["Fuel_Type"].isin(sel_fuels) & df["Age"].between(*sel_age)]
    if hide_floor:
        view = view[view["Selling_Price"] > 500]

    if view.empty:
        st.warning("No cars match these filters.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Cars", f"{len(view):,}")
        k2.metric("Median price", fmt_money(view["Selling_Price"].median()))
        k3.metric("Average age", f"{view['Age'].mean():.1f} yrs")
        k4.metric("Average mileage", f"{view['Mileage'].mean():,.0f}")

        g1, g2 = st.columns(2)
        with g1:
            order = view.groupby("Make")["Selling_Price"].median().sort_values(ascending=False).index
            fig = px.box(view, x="Make", y="Selling_Price", color="Make",
                         category_orders={"Make": list(order)}, title="Price by brand")
            fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=50, b=10),
                              yaxis_title="Price ($)", xaxis_title="")
            show(fig)

        with g2:
            sample = view.sample(min(len(view), 2000), random_state=1)
            fig = px.scatter(sample, x="Age", y="Selling_Price", color="Fuel_Type", opacity=0.6,
                             hover_data=["Make", "Model", "Mileage", "Owners"], title="Price vs age")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="Price ($)")
            show(fig)

        g3, g4 = st.columns(2)
        with g3:
            nf = view[view["Selling_Price"] > 500].copy()
            nf["Age group"] = pd.cut(nf["Age"], [0, 5, 10, 15, 22], labels=["0-5", "6-10", "11-15", "16+"])
            grp = nf.groupby(["Age group", "Accident_History"], observed=True)["Selling_Price"].median().reset_index()
            fig = px.bar(grp, x="Age group", y="Selling_Price", color="Accident_History", barmode="group",
                         color_discrete_map={"No": "#2ca02c", "Unknown": "#9ca3af", "Yes": "#d62728"},
                         title="Accident effect within each age group")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="Median price ($)")
            show(fig)

        with g4:
            top = (view.groupby("Model")["Selling_Price"].median().sort_values(ascending=False).head(10)
                   .reset_index().sort_values("Selling_Price"))
            fig = px.bar(top, x="Selling_Price", y="Model", orientation="h", title="Top 10 models by median price",
                         color="Selling_Price", color_continuous_scale="Blues")
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10), coloraxis_showscale=False,
                              xaxis_title="Median price ($)", yaxis_title="")
            show(fig)

# ============================================================================
# TAB 3: MODEL INSIGHTS
# ============================================================================
with tab_model:
    st.subheader("How good is the model?")
    metrics = pd.DataFrame(bundle.get("metrics") or DEFAULT_METRICS)
    best = metrics.sort_values("MAE ($)").iloc[0]

    q1, q2, q3 = st.columns(3)
    q1.metric("Best model", str(best["Model"]))
    q2.metric("R² (price in $)", f"{best['R2 ($)']:.3f}")
    q3.metric("Average error (MAE)", fmt_money(best["MAE ($)"]))

    show_table(metrics)

    st.subheader("What drives the price?")
    imp = pd.Series(bundle.get("importance") or DEFAULT_IMPORTANCE).sort_values()
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", color=imp.values,
                 color_continuous_scale="Blues", labels={"x": "Share of total SHAP impact (%)", "y": ""})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    show(fig)

    st.markdown(
        """
        **Key takeaways**
        - **Age, model and mileage** drive most of the price. Brand tier (luxury or not) comes next.
        - Service and accident history have a real but smaller effect.
        - Color and location contribute very little.
        - The price range shown in the predictor reflects the model's typical error for that price level:
          cheap cars have a larger *relative* error than expensive ones.
        """
    )

# ============================================================================
# TAB 4: ABOUT
# ============================================================================
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
        This app is the final stage of an end-to-end data analysis project on **5,500 used cars**.

        **Workflow**
        1. **Data audit & cleaning:** missing values imputed by model and fuel type; no rows removed.
        2. **EDA:** Matplotlib, Seaborn, Plotly and Altair to explore price drivers.
        3. **Feature engineering:** car age, yearly mileage, luxury tier, log-transformed price.
        4. **Modeling:** Linear Regression → Random Forest → **XGBoost** (R² = 0.984).
        5. **Explainability:** SHAP to verify the model relies on sensible factors.

        **Good to know**
        - About 22% of the cars sell at a **$500 price floor**, so predictions never go below it.
        - The model is trained on this dataset only; treat results as estimates, not appraisals.

        **Tech stack:** Python · pandas · scikit-learn · XGBoost · SHAP · Plotly · Streamlit
        """
    )

    st.subheader("Author")
    st.markdown(f"**{AUTHOR_NAME}**  \n{AUTHOR_TITLE}")
    if LINKEDIN_URL:
        st.markdown(f"🔗 [LinkedIn]({LINKEDIN_URL})")
    if GITHUB_URL:
        st.markdown(f"💻 [GitHub]({GITHUB_URL})")

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.divider()
st.markdown(
    f'<div class="footer">© {CURRENT_YEAR} {AUTHOR_NAME} · {AUTHOR_TITLE} · All rights reserved</div>',
    unsafe_allow_html=True,
)
