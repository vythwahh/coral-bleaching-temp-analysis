# EcoCast — Coral Bleaching Prediction System

> An end-to-end predictive analytics system for coral reef thermal stress monitoring, combining time-series forecasting, NOAA-compliant Degree Heating Week computation, and agentic LLM alerting.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

EcoCast originated as a Year-2 exploratory analysis of coral bleaching patterns across Indonesia and the Philippines (2019–2020). The original script produced static Matplotlib/Seaborn visualisations from a filtered CSV. This repository represents a complete rebuild into a production-grade predictive system.

The system ingests historical coral survey data, forecasts sea surface temperature (SST) for a 30-day horizon, computes accumulated thermal stress using the NOAA Degree Heating Weeks (DHW) methodology, and surfaces results through an interactive Streamlit dashboard with an agentic alerting layer powered by the Claude API.

---

## Architecture

```
global_bleaching_environmental.csv
           │
           ▼
   load_and_preprocess()        # ETL: normalize, filter, resample, ffill
           │
           ▼
     forecast_sst()             # ARIMA(5,1,0) → XGBoost lag-14 → Seasonal Naive
           │
           ▼
     compute_dhw()              # NOAA DHW: hotspot accumulation, 12-week rolling window
           │
           ▼
    generate_nla()              # Sigmoid risk mapping + Natural Language Analysis
           │
        ┌──┴──────────────┐
        ▼                 ▼
 ActionAgent          ChatAgent
 (alert letters)   (conversational)
        │                 │
        └──────┬──────────┘
               ▼
        Streamlit Dashboard
   (charts · map · NLA · export)
```

---

## Features

### ML Forecasting Engine (`ml_engine.py`)
- **ARIMA(5,1,0)** — primary backend for SST time-series forecasting with 90% confidence intervals
- **XGBoost** with 14-day lag features — fallback with recursive multi-step forecasting
- **Seasonal Naive** — emergency fallback ensuring the app never crashes
- **NOAA DHW computation** — rolling 84-day hotspot accumulation (hotspot = SST − MMM, counted only when ≥ 1°C above MMM)
- **Logistic bleaching risk mapping** — calibrated to NOAA alert thresholds (Watch/Warning/Alert 1/Alert 2)
- **Natural Language Analysis** — automated expert-register text generation from forecast outputs
- **Kelvin → Celsius auto-detection** — handles both raw Kelvin and Celsius input datasets

### Agentic Layer (`agent_action.py`)
- **ActionAgent** — monitors DHW forecasts and auto-triggers when bleaching probability ≥ 85% sustained for ≥ 14 consecutive days; drafts scientifically rigorous emergency letters via Claude API; simulates SMTP email and Slack webhook dispatch
- **ChatAgent** — conversational interface with on-demand ML inference; detects region names and intent from natural language; maintains multi-turn conversation history; RAG-style context injection (live forecast data → LLM prompt)

### Streamlit Dashboard (`app.py`)
- SST historical + 30-day forecast chart with 90% CI band
- Degree Heating Weeks bar chart with NOAA alert threshold annotations
- Bleaching probability gauge (day-selectable via slider)
- Reef site table with SST and risk per location
- Interactive Scattergeo map (Plotly, no API token required)
- NLA summary panel
- CSV export (Power BI / Tableau ready)
- Full agentic alert panel with dispatch log and letter preview
- Conversational chat agent with suggested prompts

---

## Dataset

**Donner et al. Global Coral Bleaching Database**
- Source: Biological and Chemical Oceanography Data Management Office (BCO-DMO)
- Coverage: Global reef survey sites, 1980s–2019
- Key fields used: `Temperature_Mean` (Kelvin), `Percent_Bleaching`, `Depth_m`, `Country_Name`, `Date`

---

## Quick Start

```bash
# Install dependencies
pip install streamlit pandas numpy statsmodels xgboost plotly

# Run dashboard
streamlit run app.py
```

Demo mode runs immediately without uploading data — synthetic SST series are generated automatically. Upload `global_bleaching_environmental.csv` via the sidebar to use real survey data.

To enable LLM-powered responses, add a Claude API key (`sk-ant-...`) in the Agent Settings panel.

---

## Project Structure

```
EcoCast/
├── ml_engine.py          # Forecasting pipeline + DHW + NLA
├── agent_action.py       # ActionAgent + ChatAgent (LLM layer)
├── app.py                # Streamlit dashboard
├── coralproject.py       # Original Year-2 EDA script (static charts)
├── charts/               # EDA output visualisations
│   ├── Bleaching at 0-5m (2019-2020).png
│   ├── Bleaching over time (0-5m).png
│   ├── Bleaching trend vs sea temperature 0-5m (2019-2020).png
│   ├── Coral bleaching 0-5m 2019-2020.png
│   ├── Distribution of mean see temperature 0-5m 2019-2020.png
│   ├── Monthly average bleaching 0-5m 2019.png
│   └── bleaching by location 0-5m 2019-2020.png
└── global_bleaching_environmental.csv
```

---

## Technical Notes

**Why ARIMA over a deep learning model?**
With ~365 daily observations per region after filtering, ARIMA(5,1,0) outperforms RNN/LSTM in both stability and interpretability. SST exhibits high autocorrelation and clear seasonal structure — exactly the conditions where ARIMA excels.
**The Claude API integration** is implemented but not validated end-to-end due to API access constraints. In demo mode, the chat agent returns template-based responses derived from live ML inference outputs. To enable full LLM-powered responses, add a valid sk-ant-... key to the Agent Settings panel. The agentic layer architecture (prompt construction, context injection, multi-turn history) is production-ready and **can be swapped to any OpenAI-compatible API (Gemini, Groq, OpenAI)** by modifying the endpoint and model string in agent_action.py.

**DHW calibration**
MMM baselines are approximate region-level values. For production use, pixel-level NOAA CoralTemp climatology (5 km resolution) should replace the hardcoded constants in `REGION_MMM`.

**Agentic dispatch**
Email and Slack dispatch are currently simulated (logged to console). To enable real dispatch, configure SMTP credentials and a Slack Incoming Webhook URL in `agent_action.py`.

---

## Background

This project was built as a personal side project alongside a Data Science degree at the University of Science, VNU-HCM (HCMUS). The long-term motivation is applying data engineering and machine learning to ocean conservation — specifically supporting early warning systems for coral reef monitoring organisations such as NOAA Coral Reef Watch and Global Fishing Watch.

---

## References

- Hughes et al. (2017). Global warming and recurrent mass bleaching of corals. *Nature*, 543, 373–377.
- Liu et al. (2006). Reef-scale thermal stress monitoring of coral ecosystems. *Remote Sensing*, 6(11), 11579–11606.
- Donner et al. (2017). Global Coral Bleaching Database. BCO-DMO.
- NOAA Coral Reef Watch: https://coralreefwatch.noaa.gov

---

*Author: Nguyễn Triệu Vy Thư · HCMUS Data Science · 2026*
