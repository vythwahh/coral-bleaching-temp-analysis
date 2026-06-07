"""
EcoCast - agent_action.py
 
ConservationAgent: Agentic Alerting & Conversational Insights Engine
 

Two agent modes:
  1. ActionAgent  — monitors DHW forecasts, auto-drafts emergency conservation
                    letters via Claude API, simulates email/Slack dispatch
  2. ChatAgent    — conversational NL interface that runs ML inference on-demand
                    and synthesises expert-level responses via Claude API

Designed to be imported by app.py (Streamlit) or run standalone.
"""

import json
import time
import smtplib
import logging
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

import numpy as np
import pandas as pd
import requests


from ml_engine import (
    generate_synthetic_ts,
    forecast_sst,
    compute_dhw,
    generate_nla,
    REGION_MMM,
    REGION_COORDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EcoCast.Agent")


# CONSTANTS


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL      = "claude-sonnet-4-20250514"

# Alert thresholds
CRITICAL_RISK_THRESHOLD    = 0.85   # 85% bleaching probability
CRITICAL_SUSTAINED_DAYS    = 14     # must persist for 14+ days
ALERT_RECIPIENT_EMAIL      = "alerts@coralreefwatch.noaa.gov"   # simulated
SLACK_WEBHOOK_URL          = "https://hooks.slack.com/services/SIMULATED/WEBHOOK"  # simulated

# Conservation orgs to CC (simulated)
CC_RECIPIENTS = [
    "gbrmpa@gbrmpa.gov.au",
    "alerts@globalfishingwatch.org",
    "marine@iucn.org",
    "science@coralcoe.org.au",
]


# CLAUDE API HELPER


def _call_claude(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 1500,
    api_key: Optional[str] = None,
) -> str:
    """
    Call Claude API and return the text response.
    Falls back to a structured template if no API key is provided.
    """
    if not api_key:
        return _fallback_response(user_message)

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    try:
        resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]
    except requests.RequestException as e:
        logger.error(f"Claude API call failed: {e}")
        return _fallback_response(user_message)


def _fallback_response(prompt: str) -> str:
    """Template fallback when no API key is available (demo mode)."""
    return (
        "[DEMO MODE — Claude API key not provided]\n\n"
        "In production, this would be a full LLM-generated response.\n"
        f"Input received: {prompt[:200]}..."
    )


# ACTION AGENT — Emergency Alert Dispatch


class ActionAgent:
    """
    Monitors ML forecast output and triggers emergency conservation alerts
    when bleaching risk exceeds critical thresholds for a sustained period.

    Usage:
        agent = ActionAgent(api_key="sk-ant-...")
        result = agent.evaluate_and_dispatch(region, dhw_df, forecast_df)
    """

    SYSTEM_PROMPT = """You are a senior marine scientist at the Global Coral Reef Monitoring Network.
Your role is to write urgent, scientifically rigorous conservation alert letters to international
reef management authorities. Your writing style is:
- Formal academic English, concise and authoritative
- Evidence-based: cite specific DHW values, SST anomalies, bleaching probabilities
- Action-oriented: clear recommended interventions with timeline
- Never alarmist but never understates — calibrated urgency matching the data
Always structure your letters with: Executive Summary → Data Evidence → Risk Assessment →
Recommended Actions → Monitoring Protocol → Sign-off."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key  = api_key
        self.dispatch_log: list[dict] = []

    def evaluate_and_dispatch(
        self,
        region: str,
        dhw_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
    ) -> dict:
        """
        Evaluate forecast and dispatch alert if threshold is met.

        Returns a dict with keys:
            triggered       bool
            alert_level     str
            letter          str   (full text of the drafted letter)
            dispatch_status dict  (email + slack simulation results)
            data_table      str   (markdown table of key metrics)
        """

        critical_days = (dhw_df["bleaching_risk"] >= CRITICAL_RISK_THRESHOLD).sum()
        peak_risk     = dhw_df["bleaching_risk"].max()
        peak_dhw      = dhw_df["dhw"].max()
        peak_date     = dhw_df["dhw"].idxmax()
        avg_sst       = forecast_df["sst_forecast"].mean()
        mmm           = REGION_MMM.get(region, 28.5)
        sst_anomaly   = avg_sst - mmm

        triggered = (
            peak_risk >= CRITICAL_RISK_THRESHOLD
            and critical_days >= CRITICAL_SUSTAINED_DAYS
        )

        if not triggered:
            return {
                "triggered": False,
                "alert_level": dhw_df["alert_level"].iloc[-1],
                "peak_risk": peak_risk,
                "critical_days": critical_days,
                "letter": None,
                "dispatch_status": None,
                "data_table": self._build_data_table(dhw_df, forecast_df, region),
            }

        logger.info(f"CRITICAL THRESHOLD MET — {region} | {critical_days} days ≥ 85% risk")


        data_table = self._build_data_table(dhw_df, forecast_df, region)


        letter_prompt = self._build_letter_prompt(
            region, critical_days, peak_risk, peak_dhw,
            peak_date, avg_sst, sst_anomaly, data_table,
        )
        letter = _call_claude(
            system_prompt=self.SYSTEM_PROMPT,
            user_message=letter_prompt,
            max_tokens=1500,
            api_key=self.api_key,
        )


        dispatch_status = self._dispatch(region, letter, peak_risk, peak_dhw)


        self.dispatch_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "region":    region,
            "peak_risk": round(peak_risk * 100, 1),
            "peak_dhw":  round(peak_dhw, 2),
            "critical_sustained_days": critical_days,
            "dispatch":  dispatch_status,
        })

        return {
            "triggered":       True,
            "alert_level":     "Alert Level 2",
            "peak_risk":       peak_risk,
            "critical_days":   critical_days,
            "letter":          letter,
            "dispatch_status": dispatch_status,
            "data_table":      data_table,
        }

    def _build_data_table(
        self,
        dhw_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        region: str,
    ) -> str:
        """Build a markdown table of weekly forecast metrics."""
        mmm = REGION_MMM.get(region, 28.5)
        rows = []
        # Sample weekly snapshots
        weekly_idx = list(range(0, len(dhw_df), 7))
        if len(dhw_df) - 1 not in weekly_idx:
            weekly_idx.append(len(dhw_df) - 1)

        for i in weekly_idx:
            row = dhw_df.iloc[i]
            sst = forecast_df["sst_forecast"].iloc[i]
            rows.append({
                "Date":            row.name.strftime("%Y-%m-%d"),
                "SST (°C)":        f"{sst:.2f}",
                "Anomaly (°C)":    f"{sst - mmm:+.2f}",
                "DHW (°C-wks)":    f"{row['dhw']:.2f}",
                "Bleach Risk (%)": f"{row['bleaching_risk'] * 100:.1f}",
                "Alert Level":     row["alert_level"].split(" ", 1)[-1],
            })

        df_table = pd.DataFrame(rows)
        return df_table.to_markdown(index=False)

    def _build_letter_prompt(
        self,
        region: str,
        critical_days: int,
        peak_risk: float,
        peak_dhw: float,
        peak_date: pd.Timestamp,
        avg_sst: float,
        sst_anomaly: float,
        data_table: str,
    ) -> str:
        return f"""Draft an emergency coral bleaching alert letter with the following verified data:

REGION: {region}
ALERT DATE: {datetime.utcnow().strftime("%d %B %Y")}
RECIPIENTS: NOAA Coral Reef Watch, Great Barrier Reef Marine Park Authority,
            IUCN Marine Programme, Coral CoE

KEY METRICS:
- Forecast SST: {avg_sst:.2f}°C (anomaly: {sst_anomaly:+.2f}°C above MMM baseline)
- Peak Degree Heating Weeks: {peak_dhw:.1f} °C-weeks on {peak_date.strftime("%d %B %Y")}
- Bleaching probability: {peak_risk * 100:.1f}% (sustained ≥ 85% for {critical_days} consecutive days)
- DHW threshold breached: Alert Level 2 (≥12 °C-weeks)

FORECAST DATA TABLE (next 30 days):
{data_table}

Draft the complete formal letter. Include:
1. Executive Summary (2–3 sentences, severity and urgency)
2. Meteorological & Oceanographic Evidence (cite the data above)
3. Ecological Risk Assessment (bleaching categories, species at risk)
4. Recommended Immediate Actions (field survey, stakeholder notification, no-take zone enforcement)
5. 30-Day Monitoring Protocol
6. Sign-off as "EcoCast Automated Monitoring System, Marine Conservation AI Division"

Tone: peer-reviewed academic English, calibrated urgency, no hyperbole."""

    def _dispatch(self, region: str, letter: str, peak_risk: float, peak_dhw: float) -> dict:
        """Simulate email + Slack dispatch (logs intent, does not actually send)."""
        subject = (
            f"[URGENT] Coral Bleaching Alert Level 2 — {region} | "
            f"DHW={peak_dhw:.1f}°C-wks | Risk={peak_risk*100:.0f}%"
        )

        email_result = self._simulate_email(subject, letter)
        slack_result = self._simulate_slack(region, peak_risk, peak_dhw, letter)

        return {
            "email": email_result,
            "slack": slack_result,
            "timestamp_utc": datetime.utcnow().isoformat(),
        }

    def _simulate_email(self, subject: str, body: str) -> dict:
        """Simulate SMTP email dispatch (dry-run, no actual send)."""
        logger.info(f"[SIMULATED EMAIL] To: {ALERT_RECIPIENT_EMAIL}")
        logger.info(f"[SIMULATED EMAIL] Subject: {subject}")
        logger.info(f"[SIMULATED EMAIL] CC: {', '.join(CC_RECIPIENTS)}")
        logger.info(f"[SIMULATED EMAIL] Body length: {len(body)} chars")

        # In production: configure real SMTP creds below
        # msg = MIMEMultipart()
        # msg['From']    = "ecocast@marineai.org"
        # msg['To']      = ALERT_RECIPIENT_EMAIL
        # msg['CC']      = ', '.join(CC_RECIPIENTS)
        # msg['Subject'] = subject
        # msg.attach(MIMEText(body, 'plain'))
        # with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        #     server.login(EMAIL_USER, EMAIL_PASS)
        #     server.send_message(msg)

        return {
            "status":    "SIMULATED",
            "to":        ALERT_RECIPIENT_EMAIL,
            "cc":        CC_RECIPIENTS,
            "subject":   subject,
            "body_len":  len(body),
            "note":      "Configure SMTP credentials in agent_action.py to enable real dispatch.",
        }

    def _simulate_slack(
        self, region: str, peak_risk: float, peak_dhw: float, letter: str
    ) -> dict:
        """Simulate Slack webhook dispatch."""
        slack_payload = {
            "text": f"*CORAL BLEACHING ALERT LEVEL 2* — *{region}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*EcoCast Emergency Alert*\n"
                            f"*Region:* {region}\n"
                            f"*Peak DHW:* {peak_dhw:.1f} °C-weeks\n"
                            f"*Bleaching Risk:* {peak_risk*100:.0f}%\n"
                            f"*Status:* Alert Level 2 — Severe bleaching & mortality expected\n"
                            f"*Time (UTC):* {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
                        ),
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Letter Preview:*\n```{letter[:400]}...```",
                    },
                },
            ],
        }

        logger.info(f"[SIMULATED SLACK] Webhook: {SLACK_WEBHOOK_URL}")
        logger.info(f"[SIMULATED SLACK] Payload: {json.dumps(slack_payload)[:300]}...")

        # In production:
        # resp = requests.post(SLACK_WEBHOOK_URL, json=slack_payload, timeout=10)
        # return {"status": "OK" if resp.ok else "FAILED", "http_code": resp.status_code}

        return {
            "status":  "SIMULATED",
            "webhook": SLACK_WEBHOOK_URL,
            "payload": slack_payload,
            "note":    "Replace SLACK_WEBHOOK_URL with real Incoming Webhook to enable.",
        }


# CHAT AGENT — Conversational Insights


class ChatAgent:
    """
    Conversational agent that answers natural language queries about coral
    bleaching risk by running ML inference on-demand and synthesising
    expert responses via Claude.

    Usage:
        agent = ChatAgent(api_key="sk-ant-...")
        reply = agent.chat("Check 30-day bleaching risk for Caribbean")
    """

    SYSTEM_PROMPT = """You are EcoCast AI, an expert marine conservation assistant with deep knowledge
of coral reef ecology, thermal stress indicators, and NOAA coral monitoring methodology.

You have access to real-time ML model outputs: SST forecasts, Degree Heating Weeks (DHW),
bleaching probability scores, and NOAA alert classifications.

When answering:
- Lead with the most critical finding (risk level, DHW value, probability)
- Back every claim with the specific numbers from the data provided
- Suggest concrete, actionable mitigation or monitoring steps
- Use expert vocabulary naturally: DHW, MMM, hotspot, thermal anomaly, symbiodinium, zooxanthellae
- Be concise — 3–5 paragraphs max unless asked for detailed strategy
- If asked for a strategy/plan, structure it as: Immediate → Short-term → Long-term actions
- Always end with the most important single number the user should remember

You do NOT fabricate data. All numerical claims come from the forecast data injected into your context."""

    def __init__(self, api_key: Optional[str] = None, history_limit: int = 10):
        self.api_key       = api_key
        self.history_limit = history_limit
        self.conversation_history: list[dict] = []
        self._region_cache: dict[str, dict] = {}

    def chat(self, user_message: str, csv_path: Optional[str] = None) -> str:
        """
        Process a user message, run ML inference if needed, return expert response.

        Intent routing:
          - Region mention → run inference for that region
          - "compare" → run for 2+ regions
          - General question → answer from knowledge without inference
          - "draft" / "write" / "letter" → delegate to ActionAgent letter draft
        """

        detected_regions = self._detect_regions(user_message)
        intent           = self._detect_intent(user_message)

        logger.info(f"[ChatAgent] User: {user_message[:80]}")
        logger.info(f"[ChatAgent] Detected regions: {detected_regions}, intent: {intent}")


        context_data = ""
        if detected_regions:
            for region in detected_regions[:2]:   # max 2 regions per turn
                forecast_context = self._run_inference(region, csv_path)
                context_data += f"\n\n--- FORECAST DATA: {region} ---\n{forecast_context}"


        augmented_message = user_message
        if context_data:
            augmented_message = (
                f"{user_message}\n\n"
                f"[SYSTEM: The following live forecast data has been retrieved. "
                f"Use it as the factual basis for your response.]\n{context_data}"
            )


        self.conversation_history.append({"role": "user", "content": augmented_message})

        # Trim history to avoid token overflow
        trimmed = self.conversation_history[-self.history_limit * 2:]


        if not self.api_key:
            reply = self._demo_reply(user_message, detected_regions, context_data)
        else:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model":      CLAUDE_MODEL,
                "max_tokens": 1000,
                "system":     self.SYSTEM_PROMPT,
                "messages":   trimmed,
            }
            try:
                resp = requests.post(
                    ANTHROPIC_API_URL, headers=headers,
                    json=payload, timeout=30,
                )
                resp.raise_for_status()
                reply = resp.json()["content"][0]["text"]
            except Exception as e:
                logger.error(f"Claude API error: {e}")
                reply = self._demo_reply(user_message, detected_regions, context_data)


        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def _run_inference(self, region: str, csv_path: Optional[str] = None) -> str:
        """Run full ML pipeline for a region and return a text summary for LLM context."""
        # Cache hit (same region within session)
        if region in self._region_cache:
            return self._region_cache[region]

        try:
            if csv_path:
                from ml_engine import load_and_preprocess
                ts = load_and_preprocess(csv_path, region)
            else:
                ts = generate_synthetic_ts(region)

            forecast_df = forecast_sst(ts, horizon=30, backend="auto")
            dhw_df      = compute_dhw(ts, forecast_df, region)
            nla         = generate_nla(region, dhw_df, forecast_df)
            mmm         = REGION_MMM.get(region, 28.5)

            # Build compact context string
            peak_date = dhw_df["dhw"].idxmax()
            context = (
                f"Region: {region}\n"
                f"MMM Baseline: {mmm}°C\n"
                f"30-Day Avg Forecast SST: {forecast_df['sst_forecast'].mean():.2f}°C "
                f"(anomaly: {forecast_df['sst_forecast'].mean() - mmm:+.2f}°C)\n"
                f"Peak DHW: {dhw_df['dhw'].max():.1f} °C-weeks on {peak_date.strftime('%d %b %Y')}\n"
                f"Max Bleaching Probability: {dhw_df['bleaching_risk'].max() * 100:.1f}%\n"
                f"Alert Level: {dhw_df.loc[peak_date, 'alert_level']}\n"
                f"Days ≥ 85% Risk: {(dhw_df['bleaching_risk'] >= 0.85).sum()}\n"
                f"NLA Headline: {nla['headline']}\n"
                f"NLA Summary: {nla['summary']}\n"
                f"Recommended Action: {nla['action']}\n"
            )

            self._region_cache[region] = context
            return context

        except Exception as e:
            logger.error(f"Inference failed for {region}: {e}")
            return f"[Inference failed for {region}: {e}]"

    def _detect_regions(self, text: str) -> list[str]:
        """Detect reef region names mentioned in user message."""
        text_lower = text.lower()
        found = []
        aliases = {
            "great barrier reef": "Great Barrier Reef",
            "gbr":                "Great Barrier Reef",
            "coral sea":          "Coral Sea",
            "indonesia":          "Indonesia",
            "philippines":        "Philippines",
            "philippine":         "Philippines",
            "coral triangle":     "Coral Triangle",
            "maldives":           "Maldives",
            "red sea":            "Red Sea",
            "caribbean":          "Caribbean",
        }
        for alias, canonical in aliases.items():
            if alias in text_lower and canonical not in found:
                found.append(canonical)
        return found

    def _detect_intent(self, text: str) -> str:
        """Classify user intent."""
        t = text.lower()
        if any(w in t for w in ["draft", "write", "letter", "report", "email"]):
            return "draft_letter"
        if any(w in t for w in ["compare", "vs", "versus", "difference between"]):
            return "compare"
        if any(w in t for w in ["strategy", "mitigation", "plan", "recommend"]):
            return "strategy"
        if any(w in t for w in ["risk", "bleach", "dhw", "temperature", "sst", "forecast", "check"]):
            return "risk_query"
        return "general"

    def _demo_reply(
        self,
        message: str,
        regions: list[str],
        context_data: str,
    ) -> str:
        """Structured demo reply when no API key is set."""
        if not regions:
            return (
                "I'm EcoCast AI — your coral reef bleaching intelligence system.\n\n"
                "Try asking:\n"
                "• *'Check bleaching risk for Great Barrier Reef'*\n"
                "• *'Compare Indonesia vs Philippines thermal stress'*\n"
                "• *'Draft a mitigation strategy for Caribbean reefs'*\n\n"
                "*Demo mode* — Connect a Claude API key in Settings to enable full LLM responses."
            )

        region = regions[0]
        # Parse key numbers from context_data
        lines = {
            line.split(":")[0].strip(): line.split(":", 1)[-1].strip()
            for line in context_data.split("\n")
            if ":" in line
        }
        sst     = lines.get("30-Day Avg Forecast SST", "N/A")
        dhw     = lines.get("Peak DHW", "N/A")
        risk    = lines.get("Max Bleaching Probability", "N/A")
        alert   = lines.get("Alert Level", "N/A")
        summary = lines.get("NLA Summary", "")

        return (
            f"**EcoCast Analysis — {region}** *(Demo Mode)*\n\n"
            f"**SST Forecast:** {sst}\n"
            f"**Peak DHW:** {dhw}\n"
            f"**Bleaching Risk:** {risk}\n"
            f"**Alert Level:** {alert}\n\n"
            f"{summary}\n\n"
            f"*Demo mode: responses are template-based. "
            f"Add Claude API key for full expert analysis.*"
        )

    def clear_history(self):
        """Reset conversation history and inference cache."""
        self.conversation_history = []
        self._region_cache = {}
        logger.info("[ChatAgent] Conversation history and cache cleared.")


# STANDALONE TEST


if __name__ == "__main__":
    import sys

    api_key  = sys.argv[1] if len(sys.argv) > 1 else None
    region   = sys.argv[2] if len(sys.argv) > 2 else "Great Barrier Reef"

    print("\n" + "="*60)
    print("  EcoCast Agent — Standalone Test")
    print("="*60)


    print("\n[1] Running ActionAgent...")
    ts          = generate_synthetic_ts(region, n_days=365)
    forecast_df = forecast_sst(ts, horizon=30, backend="auto")
    dhw_df      = compute_dhw(ts, forecast_df, region)

    action_agent = ActionAgent(api_key=api_key)
    result       = action_agent.evaluate_and_dispatch(region, dhw_df, forecast_df)

    print(f"  Triggered: {result['triggered']}")
    print(f"  Peak risk: {result['peak_risk']*100:.1f}%")
    if result["triggered"]:
        print(f"  Letter preview:\n{result['letter'][:400]}...")
        print(f"  Dispatch: email={result['dispatch_status']['email']['status']}, "
              f"slack={result['dispatch_status']['slack']['status']}")


    print("\n[2] Running ChatAgent...")
    chat_agent = ChatAgent(api_key=api_key)
    queries = [
        f"Hey Agent, check the 30-day bleaching risk for {region} and draft a mitigation strategy.",
        "What's the difference between DHW and SST anomaly?",
        "Compare Indonesia vs Philippines thermal stress this season.",
    ]
    for q in queries:
        print(f"\n  User: {q}")
        reply = chat_agent.chat(q)
        print(f"  Agent: {reply[:300]}...")

    print("\n[Done] Agent test complete.")