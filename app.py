"""
EcoCast - app.py
Coral Bleaching Prediction Dashboard

Run:
    streamlit run app.py

Dependencies:
    pip install streamlit pandas numpy statsmodels xgboost plotly matplotlib seaborn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import io
from bathymetric_map import plot_bathymetric_overview
from ml_engine import (
    load_and_preprocess,
    forecast_sst,
    compute_dhw,
    generate_nla,
    REGION_MMM,
    DHW_THRESHOLDS,
)
from agent_action import ActionAgent, ChatAgent

st.set_page_config(
    page_title="EcoCast - Coral Bleaching Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header { font-size:2.2rem; font-weight:700; color:#0d6ea1; margin-bottom:0; }
.sub-header { font-size:0.95rem; color:#6c757d; margin-top:0; margin-bottom:1.5rem; }
.metric-card { background:#f0f7ff; border-radius:10px; padding:1rem 1.2rem; border-left:4px solid #0d6ea1; }
.alert-critical { border-left-color:#dc3545 !important; background:#fff5f5 !important; }
.alert-high { border-left-color:#fd7e14 !important; background:#fff8f0 !important; }
.alert-medium { border-left-color:#ffc107 !important; background:#fffdf0 !important; }
.alert-low { border-left-color:#28a745 !important; background:#f0fff4 !important; }
.section-title { font-size:1.15rem; font-weight:600; color:#333; border-bottom:2px solid #e0e0e0; padding-bottom:0.3rem; margin-top:1.5rem; margin-bottom:0.8rem; }
.agent-badge { display:inline-block; background:linear-gradient(135deg,#0d6ea1,#1a1a2e); color:white; padding:3px 10px; border-radius:12px; font-size:0.75rem; font-weight:600; letter-spacing:0.05em; margin-left:8px; vertical-align:middle; }
.dispatch-box { background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:1rem 1.2rem; font-family:'Courier New',monospace; font-size:0.82rem; color:#7ee787; line-height:1.8; }
.letter-box { background:#fafafa; border:1px solid #e0e0e0; border-left:4px solid #0d6ea1; border-radius:6px; padding:1.2rem 1.5rem; font-family:'Georgia',serif; font-size:0.88rem; line-height:1.8; color:#222; white-space:pre-wrap; }
.nla-box { background:#1a1a2e; color:#e0e0e0; border-radius:10px; padding:1.2rem 1.5rem; font-family:monospace; font-size:0.9rem; line-height:1.7; }
</style>
""", unsafe_allow_html=True)

REGION_COORDS = {
    "Great Barrier Reef": (-18.2871, 147.6992),
    "Coral Sea": (-17.0, 152.0),
    "Indonesia": (-2.5, 118.0),
    "Philippines": (12.8797, 121.7740),
    "Coral Triangle": (-5.0, 125.0),
    "Maldives": (3.2028, 73.2207),
    "Red Sea": (22.0, 38.0),
    "Caribbean": (15.0, -75.0),
}

REGION_REEF_POINTS = {
    "Great Barrier Reef": [
        {"name": "Ribbon Reef 10", "lat": -14.78, "lon": 145.65},
        {"name": "Osprey Reef", "lat": -13.88, "lon": 146.57},
        {"name": "Agincourt Reef", "lat": -15.95, "lon": 145.82},
        {"name": "Lady Elliot Island", "lat": -24.11, "lon": 152.71},
        {"name": "Heron Island", "lat": -23.44, "lon": 151.91},
    ],
    "Coral Sea": [
        {"name": "Holmes Reef", "lat": -16.47, "lon": 147.87},
        {"name": "Bougainville Reef", "lat": -15.49, "lon": 147.10},
        {"name": "Flinders Reef", "lat": -17.72, "lon": 148.44},
    ],
    "Indonesia": [
        {"name": "Raja Ampat", "lat": -0.23, "lon": 130.52},
        {"name": "Komodo NP", "lat": -8.56, "lon": 119.52},
        {"name": "Bunaken", "lat": 1.64, "lon": 124.76},
        {"name": "Banda Sea", "lat": -4.52, "lon": 129.89},
        {"name": "Wakatobi", "lat": -5.47, "lon": 123.60},
    ],
    "Philippines": [
        {"name": "Tubbataha Reef", "lat": 8.94, "lon": 119.91},
        {"name": "Apo Island", "lat": 9.08, "lon": 123.27},
        {"name": "Coron Bay", "lat": 11.99, "lon": 120.20},
    ],
    "Coral Triangle": [
        {"name": "Bird's Head", "lat": -0.87, "lon": 131.15},
        {"name": "Togean Islands", "lat": -0.37, "lon": 121.93},
        {"name": "Cenderawasih", "lat": -2.65, "lon": 135.42},
    ],
    "Maldives": [
        {"name": "North Male Atoll", "lat": 4.71, "lon": 73.46},
        {"name": "Ari Atoll", "lat": 3.86, "lon": 72.85},
        {"name": "Baa Atoll", "lat": 5.07, "lon": 73.02},
    ],
    "Red Sea": [
        {"name": "Ras Mohammed", "lat": 27.73, "lon": 34.25},
        {"name": "Farasan Banks", "lat": 17.01, "lon": 41.79},
        {"name": "Dahab", "lat": 28.49, "lon": 34.52},
    ],
    "Caribbean": [
        {"name": "Mesoamerican Reef", "lat": 17.25, "lon": -87.80},
        {"name": "Buck Island", "lat": 17.79, "lon": -64.62},
        {"name": "Jardines de la Reina", "lat": 20.85, "lon": -79.05},
    ],
}


def generate_synthetic_ts(region, n_days=365):
    mmm = REGION_MMM.get(region, 28.5)
    np.random.seed(42)
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq="D")
    t = np.arange(n_days)
    sst = (mmm
           + 1.2 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
           + 0.002 * t
           + np.random.normal(0, 0.25, n_days)
           + np.where((t > 200) & (t < 280), 0.8, 0))
    return pd.Series(sst, index=dates, name="sst")


def plot_sst_forecast(ts, forecast_df):
    fig = go.Figure()
    if isinstance(ts, pd.DataFrame):
        ts = ts["sst"]
    hist = ts.tail(90)
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Historical SST",
                             line=dict(color="#0d6ea1", width=2), mode="lines"))
    fig.add_trace(go.Scatter(
        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
        y=list(forecast_df["upper_ci"]) + list(forecast_df["lower_ci"][::-1]),
        fill="toself", fillcolor="rgba(255,99,71,0.15)",
        line=dict(color="rgba(255,255,255,0)"), name="90% CI"))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["sst_forecast"],
                             name="Forecast SST",
                             line=dict(color="#ff6347", width=2.5, dash="dash"), mode="lines"))
    mmm = REGION_MMM.get(st.session_state.get("region", "Indonesia"), 28.5)
    fig.add_hline(y=mmm, line_dash="dot", line_color="#ffc107",
                  annotation_text=f"MMM ({mmm}C)", annotation_position="bottom right")
    fig.update_layout(title="Sea Surface Temperature - Historical & Forecast",
                      xaxis_title="Date", yaxis_title="SST (C)",
                      template="plotly_white", height=380,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_dhw_timeline(dhw_df):
    colors = dhw_df["dhw"].apply(lambda d:
        "#dc3545" if d >= DHW_THRESHOLDS["Alert 2"] else
        "#fd7e14" if d >= DHW_THRESHOLDS["Alert 1"] else
        "#ffc107" if d >= DHW_THRESHOLDS["Warning"] else
        "#17becf" if d >= DHW_THRESHOLDS["Watch"] else "#28a745")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dhw_df.index, y=dhw_df["dhw"], marker_color=colors, name="DHW",
                         hovertemplate="<b>%{x|%d %b %Y}</b><br>DHW: %{y:.2f} C-weeks<br><extra></extra>"))
    for label, threshold, color in [
        ("Watch (2)", DHW_THRESHOLDS["Watch"], "#17becf"),
        ("Warning (4)", DHW_THRESHOLDS["Warning"], "#ffc107"),
        ("Alert 1 (8)", DHW_THRESHOLDS["Alert 1"], "#fd7e14"),
        ("Alert 2 (12)", DHW_THRESHOLDS["Alert 2"], "#dc3545"),
    ]:
        fig.add_hline(y=threshold, line_dash="dot", line_color=color, opacity=0.7,
                      annotation_text=label, annotation_position="right")
    fig.update_layout(title="Degree Heating Weeks (DHW) - 30-Day Forecast",
                      xaxis_title="Date", yaxis_title="DHW (C-weeks)",
                      template="plotly_white", height=350, showlegend=False,
                      margin=dict(l=40, r=80, t=60, b=40))
    return fig


def plot_bleaching_risk(dhw_df, selected_day):
    risk = dhw_df.iloc[selected_day]["bleaching_risk"] * 100
    color = ("#dc3545" if risk >= 85 else "#fd7e14" if risk >= 60 else
             "#ffc107" if risk >= 35 else "#28a745")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=risk,
        title={"text": f"Bleaching Probability<br><sub>{dhw_df.index[selected_day].strftime('%d %b %Y')}</sub>"},
        number={"suffix": "%", "font": {"size": 36}},
        gauge={"axis": {"range": [0, 100], "ticksuffix": "%"}, "bar": {"color": color},
               "steps": [{"range": [0, 35], "color": "#e8f5e9"}, {"range": [35, 60], "color": "#fff9c4"},
                          {"range": [60, 85], "color": "#ffe0b2"}, {"range": [85, 100], "color": "#ffebee"}],
               "threshold": {"line": {"color": "#333", "width": 3}, "thickness": 0.75, "value": risk}}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig

@st.cache_data
def get_bathymetric_map(region, selected_date, rows_json):
    df_sites = pd.read_json(rows_json) if rows_json else None
    return plot_bathymetric_overview(
        region=region,
        df_sites=df_sites,
        selected_date=selected_date,
    )
def sidebar():
    st.sidebar.markdown("## EcoCast")
    st.sidebar.markdown("*Coral Bleaching Prediction System*")
    st.sidebar.markdown("---")
    region = st.sidebar.selectbox("Select Monitoring Region", options=list(REGION_MMM.keys()), index=2)
    horizon = st.sidebar.slider("Forecast Horizon (days)", min_value=14, max_value=60, value=30, step=7)
    backend = st.sidebar.radio("Model Backend", options=["auto", "arima", "xgboost"], index=0, horizontal=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Upload coral survey CSV (Donner et al. format)", type=["csv"])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Agent Settings")
    api_key = st.sidebar.text_input("Claude API Key", type="password", placeholder="sk-ant-...")
    agent_mode = st.sidebar.toggle("Enable Action Agent", value=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data source:** Donner et al. Global Coral Survey\n\n**DHW formula:** NOAA Coral Reef Watch\n\n**Built by:** Nguyen Trieu Vy Thu")
    return region, horizon, backend, uploaded_file, api_key, agent_mode


def main():
    st.markdown('<p class="main-header">EcoCast</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">End-to-End Coral Bleaching Prediction · SST Forecasting · Degree Heating Weeks · Reef Monitoring · <span style="color:#dc3545;font-weight:600;">Agentic Alerting</span></p>', unsafe_allow_html=True)

    region, horizon, backend, uploaded_file, api_key, agent_mode = sidebar()
    st.session_state["region"] = region

    if "chat_agent" not in st.session_state or st.session_state.get("agent_api_key") != api_key:
        st.session_state["chat_agent"] = ChatAgent(api_key=api_key or None)
        st.session_state["agent_api_key"] = api_key
    if "action_agent" not in st.session_state:
        st.session_state["action_agent"] = ActionAgent(api_key=api_key or None)
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    with st.spinner("Loading and processing data..."):
        if uploaded_file is not None:
            file_key = f"{uploaded_file.name}_{uploaded_file.size}_{region}"
            if st.session_state.get("loaded_file_key") != file_key:
                try:
                    file_bytes = uploaded_file.read()
                    ts = load_and_preprocess(io.BytesIO(file_bytes), region)
                    if len(ts) < 30:
                        st.warning(f"Only {len(ts)} records found. Using synthetic data.")
                        ts = generate_synthetic_ts(region)
                        data_mode = "Demo (Synthetic)"
                    else:
                        data_mode = "Real Data (uploaded)"
                except Exception as e:
                    st.warning(f"Could not parse CSV ({e}). Using synthetic data.")
                    ts = generate_synthetic_ts(region)
                    data_mode = "Demo (Synthetic)"
                st.session_state["cached_ts"] = ts
                st.session_state["cached_data_mode"] = data_mode
                st.session_state["loaded_file_key"] = file_key
            else:
                ts = st.session_state["cached_ts"]
                data_mode = st.session_state["cached_data_mode"]
        else:
            ts = generate_synthetic_ts(region)
            data_mode = "Demo (Synthetic)"
            st.session_state.pop("loaded_file_key", None)
            st.session_state.pop("cached_ts", None)

    with st.spinner(f"Forecasting SST ({backend} backend)..."):
        forecast_key = f"{st.session_state.get('loaded_file_key','synthetic')}_{region}_{horizon}_{backend}"
        if st.session_state.get("forecast_key") != forecast_key:
            try:
                forecast_df = forecast_sst(ts, horizon=horizon, backend=backend)
                dhw_df = compute_dhw(ts, forecast_df, region)
                nla = generate_nla(region, dhw_df, forecast_df)
            except Exception as e:
                st.error(f"Forecast failed: {e}")
                return
            st.session_state["forecast_df"] = forecast_df
            st.session_state["dhw_df"] = dhw_df
            st.session_state["nla"] = nla
            st.session_state["forecast_key"] = forecast_key
        else:
            forecast_df = st.session_state["forecast_df"]
            dhw_df = st.session_state["dhw_df"]
            nla = st.session_state["nla"]

    peak_risk = dhw_df["bleaching_risk"].max()
    alert_class = ("alert-critical" if peak_risk >= 0.85 else "alert-high" if peak_risk >= 0.60 else
                   "alert-medium" if peak_risk >= 0.35 else "alert-low")
    st.markdown(f'<div class="metric-card {alert_class}"><h3 style="margin:0">{nla["headline"]}</h3><p style="margin:6px 0 0 0;color:#555;">{nla["summary"]}</p></div>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Region", region)
    col2.metric("MMM Baseline", f"{REGION_MMM[region]}C")
    col3.metric("Avg Forecast SST", f"{forecast_df['sst_forecast'].mean():.2f}C")
    col4.metric("Peak DHW", f"{dhw_df['dhw'].max():.1f} C-wks")
    col5.metric("Max Bleach Risk", f"{dhw_df['bleaching_risk'].max()*100:.1f}%")
    st.markdown(f"<small>Data mode: **{data_mode}**</small>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-title">SST Forecast & Thermal Stress Analysis</p>', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns([2, 1])
    with chart_col1:
        st.plotly_chart(plot_sst_forecast(ts, forecast_df), use_container_width=True)
    with chart_col2:
        day_slider = st.slider("Select forecast day for gauge", min_value=0,
                               max_value=len(dhw_df)-1, value=len(dhw_df)-1, format="Day %d")
        st.plotly_chart(plot_bleaching_risk(dhw_df, day_slider), use_container_width=True)
    st.plotly_chart(plot_dhw_timeline(dhw_df), use_container_width=True)

    st.markdown('<p class="section-title">Reef Bleaching Alert Map</p>', unsafe_allow_html=True)
    map_day = st.slider("Scroll through forecast timeline", min_value=0,
                        max_value=len(dhw_df)-1, value=len(dhw_df)-1,
                        format="Day %d", key="map_slider")
    selected_date = dhw_df.index[map_day].strftime("%d %b %Y")
    st.caption(f"Showing bleaching risk forecast for: **{selected_date}**")

    reef_points = REGION_REEF_POINTS.get(region, [])
    target_row = dhw_df.iloc[map_day]
    rows = []
    for pt in reef_points:
        local_risk = min(1.0, target_row["bleaching_risk"] * np.random.uniform(0.7, 1.3))
        rows.append({
            "Reef Site": pt["name"],
            "Lat": pt["lat"],
            "Lon": pt["lon"],
            "SST (C)": round(float(target_row["sst"]) + np.random.uniform(-0.4, 0.4), 2),
            "Bleach Risk (%)": round(local_risk * 100, 1),
            "Alert": ("Critical" if local_risk >= 0.85 else "High" if local_risk >= 0.60 else
                      "Moderate" if local_risk >= 0.35 else "Low"),
        })

    df_map = None
    if rows:
        df_map = pd.DataFrame(rows)
        st.dataframe(df_map, hide_index=True)
        fig_map = go.Figure(go.Scattergeo(
            lat=df_map["Lat"], lon=df_map["Lon"],
            text=df_map["Reef Site"],
            mode="markers+text", textposition="top center",
            marker=dict(size=12, color="#0d6ea1"),
        ))
        fig_map.update_geos(fitbounds="locations", showland=True, landcolor="#f0f7ff",
                            showocean=True, oceancolor="#cce5ff", showcoastlines=True)
        fig_map.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown('<p class="section-title">Natural Language Analysis (NLA)</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="nla-box">'
        f'<b>HEADLINE:</b> {nla["headline"]}<br><br>'
        f'<b>ANALYSIS:</b> {nla["summary"]}<br><br>'
        f'<b>TREND:</b> {nla["trend"]}<br><br>'
        f'<b>ACTION:</b> {nla["action"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-title">Bathymetric Overview</p>', unsafe_allow_html=True)
    st.caption("NOAA-style bathymetric map with reef site bleaching risk overlay.")
    if df_map is not None:
        bathy_png = get_bathymetric_map(
            region=region,
            selected_date=selected_date,
            rows_json=df_map.to_json(),
        )
        st.image(bathy_png, use_container_width=True)
    else:
        st.info("No reef site data available for this region.")

    st.markdown('<p class="section-title">Export Data</p>', unsafe_allow_html=True)
    export_df = dhw_df.copy()
    export_df["sst_forecast"] = forecast_df["sst_forecast"]
    export_df["lower_ci_90"] = forecast_df["lower_ci"]
    export_df["upper_ci_90"] = forecast_df["upper_ci"]
    export_df["region"] = region
    export_df = export_df.reset_index()
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.download_button(
            label="Download CSV (Power BI / Tableau ready)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"ecocast_{region.lower().replace(' ','_')}_{datetime.today().strftime('%Y%m%d')}.csv",
            mime="text/csv")
    with ex_col2:
        st.expander("Preview forecast table").dataframe(
            export_df[["date", "sst_forecast", "dhw", "bleaching_risk", "alert_level"]].head(15),
            width="stretch")

    st.markdown('<p class="section-title">Action Agent<span class="agent-badge">AGENTIC</span></p>', unsafe_allow_html=True)
    if agent_mode:
        agent_key = f"agent_{st.session_state.get('forecast_key','default')}"
        if st.session_state.get("agent_result_key") != agent_key:
            with st.spinner("Action Agent evaluating critical thresholds..."):
                agent_result = st.session_state["action_agent"].evaluate_and_dispatch(region, dhw_df, forecast_df)
            st.session_state["agent_result"] = agent_result
            st.session_state["agent_result_key"] = agent_key
        else:
            agent_result = st.session_state["agent_result"]

        if agent_result["triggered"]:
            st.error(f"**CRITICAL ALERT TRIGGERED** - {region} | Bleaching risk >= 85% sustained for **{agent_result['critical_days']} days**")
            a_col1, a_col2 = st.columns([3, 2])
            with a_col1:
                st.markdown("**Auto-Drafted Emergency Letter**")
                st.markdown(f'<div class="letter-box">{agent_result["letter"]}</div>', unsafe_allow_html=True)
                st.download_button(label="Download Alert Letter (.txt)",
                                   data=agent_result["letter"].encode("utf-8"),
                                   file_name=f"ecocast_alert_{region.replace(' ','_')}_{datetime.today().strftime('%Y%m%d')}.txt",
                                   mime="text/plain")
            with a_col2:
                st.markdown("**Dispatch Log**")
                email_s = agent_result["dispatch_status"]["email"]
                slack_s = agent_result["dispatch_status"]["slack"]
                dispatch_log = (f"[{agent_result['dispatch_status']['timestamp_utc']}]\n"
                                f"STATUS : ALERT DISPATCHED\nREGION : {region}\n"
                                f"RISK   : {agent_result['peak_risk']*100:.1f}%\n"
                                f"DHW    : {dhw_df['dhw'].max():.1f} C-weeks\n\n"
                                f"EMAIL  : {email_s['status']} -> {email_s['to']}\n"
                                f"SLACK  : {slack_s['status']}")
                st.markdown(f'<div class="dispatch-box">{dispatch_log}</div>', unsafe_allow_html=True)
                st.markdown("**Evidence Table**")
                st.markdown(agent_result["data_table"])
        else:
            st.success(f"**No critical alert triggered** - {region} | Peak risk: **{agent_result['peak_risk']*100:.1f}%** ({agent_result['critical_days']} days >= 85% threshold, need 14) | {agent_result['alert_level']}")
            st.caption("Action Agent is monitoring. Alert will auto-trigger if bleaching probability exceeds 85% for 14+ consecutive forecast days.")
    else:
        st.info("Action Agent is disabled.")

    st.markdown("---")
    st.markdown('<p class="section-title">EcoCast Chat Agent<span class="agent-badge">CONVERSATIONAL</span></p>', unsafe_allow_html=True)
    st.caption("Ask me anything about coral bleaching risk, reef regions, or request a mitigation strategy.")

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if not st.session_state["chat_messages"]:
        st.markdown("**Try asking:**")
        suggestion_cols = st.columns(3)
        suggestions = [
            f"Check 30-day bleaching risk for {region}",
            "Compare Indonesia vs Philippines thermal stress",
            "Draft a mitigation strategy for Great Barrier Reef",
        ]
        for col, sug in zip(suggestion_cols, suggestions):
            if col.button(sug, use_container_width=True):
                st.session_state["pending_prompt"] = sug
                st.rerun()

    active_prompt = None

    if "pending_prompt" in st.session_state:
        active_prompt = st.session_state.pop("pending_prompt")
        st.session_state["chat_messages"].append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.write(active_prompt)
    elif user_input := st.chat_input("Ask EcoCast Agent..."):
        active_prompt = user_input
        st.session_state["chat_messages"].append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.write(active_prompt)

    if active_prompt:
        with st.chat_message("assistant"):
            with st.spinner("Running inference and composing response..."):
                reply = st.session_state["chat_agent"].chat(active_prompt)
            st.write(reply)
        st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state["chat_messages"]:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state["chat_messages"] = []
            st.session_state["chat_agent"].clear_history()
            st.rerun()

    st.markdown("---")
    st.markdown("<small>EcoCast v2.0 · Built on NOAA DHW methodology · Data: Donner et al. Global Coral Reef Survey · Agentic layer: Anthropic Claude API · Author: Nguyen Trieu Vy Thu, HCMUS Data Science</small>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()