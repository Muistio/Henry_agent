#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Henry AI advisor -demo (Streamlit) — siistitty ilman admin-valikkoa
- Onboarding chatissa: "Hei, kukas sinä olet ja miten voin auttaa?" → personoitu sävy ja fokus
- Hero-avatar + freesi header
- CV-koukku: sidotaan vastaukset Henryn taustaan
- KPI-taulukko + AI governance -kaavio (automaattisesti kun viestissä pyydetään KPI/governance)
- Chat-loki tietokantaan reaaliajassa:
    * Supabase Postgres (pooled, 6543, sslmode=require) jos DATABASE_URL toimii
    * muutoin SQLite (/mount/data/chatlogs.db)
- Yhteys-CTA: mailto, Calendly (lähettää koko transkriptin)
- API-avain vain secrets/env – ei koskaan UI:ssa
"""

import os
import io
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import re

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

# ========= Perusasetukset =========

APP_NAME = "Tutustu Henryn CV:seen 🤖"
DEFAULT_MODEL = "gpt-4o-mini"   # nopea ja edullinen

# Kirjoituskelpoinen polku myös Streamlit Cloudissa
if os.path.exists("/mount/data"):
    DB_DIR = "/mount/data"
else:
    DB_DIR = os.getcwd()
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "chatlogs.db")

# --- Database URL: prioriteetti Secrets -> ENV + siivous + lähteen tunnistus ---

DATABASE_URL = st.secrets.get("DATABASE_URL", "") or os.getenv("DATABASE_URL", "")

def _clean_db_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip().strip('"').strip("'")
    # Estä placeholderin käyttö → pakota SQLiteen
    if "db.xxxxx.supabase.co" in u:
        return ""
    return u

DATABASE_URL = _clean_db_url(DATABASE_URL)

def _db_source() -> str:
    if st.secrets.get("DATABASE_URL", ""):
        return "secrets"
    if os.getenv("DATABASE_URL", ""):
        return "env"
    return "missing"

def _safe_dbu(mask_target: str) -> str:
    try:
        u = urlparse(mask_target)
        host = u.hostname or "?"
        port = u.port or "?"
        return f"{host}:{port}"
    except Exception:
        return "?"

# ========= Henryn tausta & persona =========

ABOUT_ME = """
Nimi: Henry
Rooli-identiteetti: AI-osaaja ja dataohjautuva markkinointistrategi (10+ vuotta), CRM-admin (HubSpot, Salesforce), Python-harrastaja ja sijoittamista harrastava.
Asuinmaat: Suomi, Saksa, Kiina. Harrastaa myös kuntosalia, uintia ja saunomista. Juo kahvin mustana.

- Data analytics and management
- Hubspot & Salesforce CRM
- Business development
- Event organizing
- Start-up background spiced up with corporate experience
- I find it rewarding to work amidst diverse international cultures
- My three childhood buddies and I run our own watch brand, Rohje (rohje.com). #Shopify

Työkokemus:
- Tällä hetkellä opintovapaalla viimeistelemässä International Business -gradua. Samalla sivuaineena strateginen analyysi ja makrotalous.

- Gofore Oyj (2020–2025): Marketing strategist
  • Dataohjautuvat markkinointistrategiat ja tekoäly
  • Myynnin ja konsulttitiimien tuki: kohdennus, segmentointi
  • Brändistrategiat yritysostoissa (4 kpl viime vuosina)
  • Marketing automation ja ABM-strategia
  • HubSpot & Salesforce integraatio ja ylläpito
  • LLM-koulutuksia ja AI-kehitystä

- Airbus (2018–2020): Marketing manager
  • Viestinnän ja myynnin linjaus liiketoimintatavoitteisiin
  • Tapahtumatuotanto
  • Kampanja-analytiikka (EU–LATAM)
  • Mission-critical IoT -konseptointi
  • Verkkosivuprojektit (esim. airbusfinland.com)

- Rohje Oy (2018–): Co-founder (sivuprojekti)
  • Kellobrändin rakentaminen alusta
  • Datalähtöinen kasvu, Shopify-optimoitu e-commerce
  • Google Ads & social, KPI-seuranta (CAC, ROAS)
  • “Finnish watch” -hakutermin kärkisijoitukset, valikoimaan mm. Stockmann

- Telia (2017): Marketing specialist (sijaisuus)
  • B2B-myyntiverkoston markkinoinnin kehitys, tapahtumat, B2B-some

- Digi Electronics, Shenzhen (2017): Marketing assistant (harjoittelu)
  • Adwords, Analytics, Smartly; Liveagent; valittu tiimin “employee of the quarter”

- Jyväskylä Entrepreneurship Society (2014–2016): Hallituksen pj (2015)
  • Spotlight-startup-tapahtuman käynnistäminen, laaja sidosryhmäverkosto

- Keski-Suomen Pelastuslaitos (2010–2017): VPK-palomies
  • Stressin hallinta, kriittinen viestintä (TETRA), kurssit: ensiapu, vaaralliset aineet, ym.

Koulutus:
- KTM, Jyväskylän yliopisto (2019–)
- Tradenomi, JAMK (2015–2018)
- Energia-ala opintoja, JAMK (2013–2015)

Kielet:
- Suomi (äidinkieli), Englanti (C1), Saksa (B1), Ruotsi (A1)

AI & data -osaamisen kohokohdat:
- Python-projektit: tuotetietojen haku, markkinakatsaus, kilpailijavertailu
- Liiketoimintalähtöinen AI: tunnistan arvokohteet, vien idean tuotantoon ja koulutan käyttäjät
- Google Cloud data/AI -tuntemus, Microsoft Copilot/Graph-integraatiot
- AI governance ja EU AI Act -näkökulma käytännön tekemiseen (riskit, kontrollit, selitettävyys)

Miksi POP Pankki:
- Haluan tuoda teknologista kehitystä perinteiselle toimialalle.
- Kehitän konkreettisia, mitattavia AI-ratkaisuja (asiakaspalvelu Copilot, ennustava analytiikka, riskienhallinta) ja pysyvät prosessit (monitorointi, MLOps/LLMOps).
"""

JOB_AD_SUMMARY = """
POP AI Advisor vastaa pankkiryhmän AI-kehityksen suunnittelusta ja koordinoinnista,
AI-ratkaisujen suunnittelusta ja mallinnuksesta, ennustavan analytiikan kehittämisestä, AI-käytäntöjen
juurruttamisesta, prosessi- ja data-analyysistä, Data- ja Tekoälystrategian tukemisesta sekä sisäisestä
AI-asiantuntijuudesta ja koulutuksesta.
"""

PERSONA = (
    "Olen Henry. "
    "Puhun minä-muodossa luonnollisesti ja napakasti — bisneslähtöisesti, mutta sopivalla huumorilla. "
    "Annan konkreettisia askelmerkkejä (30/60/90 pv), määrittelen KPI:t ja huomioin AI-governancen (EU AI Act) tarvittaessa. "
    "Vältän hypeä ja perustelen riskit sekä hyödyt. Hyödynnän ABOUT_ME ja roolivaatimukset."
)

# ========= Yleisö / personointi =========

AUDIENCE_PRESETS = {
    "rekrytoija": {
        "tone": "selkeä ja napakka, liiketoimintalähtöinen",
        "focus": [
            "proof-of-value 2–4 viikossa",
            "mitattavat KPI:t ja riskienhallinta",
            "sidosryhmäkommunikaatio ja koulutus"
        ]
    },
    "tiiminvetäjä": {
        "tone": "ratkaisu- ja toimeenpanolähtöinen",
        "focus": [
            "30/60/90 päivän suunnitelma",
            "resursointi, backlog ja arkkitehtuuri",
            "MLOps/LLMOps, monitorointi ja kustannukset"
        ]
    },
    "data engineer / analyst": {
        "tone": "tekninen mutta selkeä, käytännönläheinen",
        "focus": [
            "datan lähteet, skeemat, laadunvarmistus",
            "selitettävyys, drift, eval/testaus",
            "pipelines, versiointi, CI/CD"
        ]
    },
    "kollega": {
        "tone": "rentohko, yhteistyötä korostava",
        "focus": [
            "yhteiset työskentelytavat ja työkalut",
            "sisäinen RAG, playbookit, tiedonjakaminen",
            "koulutus ja enablement"
        ]
    },
    "media": {
        "tone": "ytimekäs ja ymmärrettävä",
        "focus": [
            "vaikutus asiakkaisiin ja yhteiskuntaan",
            "läpinäkyvyys ja vastuullisuus",
            "konkreettiset esimerkit ja tulokset"
        ]
    },
    "muu": {
        "tone": "neutraali ja selkeä",
        "focus": [
            "tarpeen kartoitus",
            "sopivan syvyystason valinta",
            "seuraavat askeleet"
        ]
    },
}

def build_audience_block(audience: str, name: str = "", company: str = "") -> str:
    key = (audience or "muu").lower().strip()
    if key not in AUDIENCE_PRESETS:
        key = "muu"
    p = AUDIENCE_PRESETS[key]
    who = audience
    if company:
        who += f" @ {company}"
    if name:
        who += f" ({name})"
    focus_bullets = "\n".join([f"- {f}" for f in p["focus"]])
    return (
        "KÄYTTÄJÄPROFIILI:\n"
        f"- Rooli: {who}\n"
        f"- Sävytaso: {p['tone']}\n"
        "- Korosta vastauksissa erityisesti:\n"
        f"{focus_bullets}\n"
        "Mukauta esimerkit ja KPI:t tälle yleisölle sopiviksi.\n"
    )

def classify_profile(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["rekry", "rekrytoija", "recruiter", "hiring"]):
        return "rekrytoija"
    if any(w in t for w in ["lead", "vetäjä", "esihenkilö", "manager", "tiiminvetäjä", "team lead"]):
        return "tiiminvetäjä"
    if any(w in t for w in ["data engineer", "analyyt", "analyst", "ml", "mlops", "pipeline"]):
        return "data engineer / analyst"
    if any(w in t for w in ["toimittaja", "media", "lehti", "press"]):
        return "media"
    if any(w in t for w in ["kollega", "työkaveri", "internal", "sisäinen"]):
        return "kollega"
    return "muu"

def extract_name_company(text: str) -> tuple[str, str]:
    name = ""
    m = re.search(r"\bolen\s+([A-ZÅÄÖ][a-zåäö]+(?:\s+[A-ZÅÄÖ][a-zåäö]+)?)", text)
    if m:
        name = m.group(1).strip()
    company = ""
    m2 = re.search(r"\b(yrityksestä|firmasta|talosta|yhtiöstä|company|from)\s+([A-Z0-9][\w&\-\s]{1,40})", text, re.I)
    if m2:
        company = m2.group(2).strip()
    return name, company

# ========= Pikatyökalut =========

def bullets_ai_opportunities() -> str:
    return "\n".join([
        "1) Asiakaspalvelu Copilot: summaus, vastaus-ehdotukset, CRM-kirjaus.",
        "2) Fraud score (rules+ML): signaalifuusio, SHAP-seuranta.",
        "3) AML alert triage: priorisointi + tutkintamuistion runko.",
        "4) Ennustava luotonanto: PD/LGD + selitettävyys-paneeli.",
        "5) Tietopyyntöjen automaatio: ohjattu haku, audit-logi.",
        "6) Sisäinen RAG-haku: ohjeet, prosessit, mallidokit.",
    ])

def bullets_ai_governance() -> str:
    return "\n".join([
        "• Data governance: omistajuus, laatu, säilytys, DPIA tarpeen mukaan.",
        "• Mallien elinkaari: versiointi, hyväksyntä, monitorointi (drift/bias).",
        "• Selitettävyys: SHAP/LIME tai policy, milloin vaaditaan.",
        "• EU AI Act: luokitus, kontrollit, rekisteröinti tarvittaessa.",
        "• Riskienhallinta: human-in-the-loop, fallback, vaikutusarvio.",
        "• Tietoturva & pääsynhallinta: salaisuudet, auditointi.",
    ])

# ========= OpenAI =========

def get_api_key() -> str:
    try:
        v = st.secrets.get("OPENAI_API_KEY", "")
        if v:
            return v
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")

def get_client() -> Optional[OpenAI]:
    key = get_api_key()
    if not key:
        return None
    try:
        return OpenAI(api_key=key, timeout=30.0)
    except Exception:
        return None

def call_chat(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content

# ========= Avatar =========

def get_avatar_url() -> str:
    direct = st.secrets.get("GITHUB_AVATAR_URL", "")
    if direct:
        return direct
    user = st.secrets.get("GITHUB_USERNAME", "")
    if user:
        return f"https://github.com/{user}.png?size=240"
    return "https://api.dicebear.com/7.x/thumbs/svg?seed=Henry"

# ========= DB: SQLite oletus, Supabase PG jos saatavilla =========

def _sqlite_conn():
    return sqlite3.connect(DB_PATH)

def _pg_conn():
    import psycopg2  # vaatii psycopg2-binary
    return psycopg2.connect(DATABASE_URL)

def _use_postgres() -> bool:
    """Päätä kerran per sessio käytetäänkö Postgresta."""
    global DATABASE_URL
    if not (DATABASE_URL and (DATABASE_URL.startswith("postgres://") or DATABASE_URL.startswith("postgresql://"))):
        st.session_state.use_postgres = False
        return False
    if "use_postgres" in st.session_state:
        return bool(st.session_state.use_postgres)
    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=6, sslmode="require")
        conn.close()
        st.session_state.use_postgres = True
        st.info(f"Tietokanta: Postgres ({_safe_dbu(DATABASE_URL)}) • lähde: {_db_source()}")
    except Exception as e:
        st.session_state.use_postgres = False
        st.warning(f"Postgres ei käytettävissä ({e}); lukitaan SQLiteen täksi sessioksi.")
    return bool(st.session_state.use_postgres)

def init_db():
    if _use_postgres():
        try:
            import psycopg2
            with _pg_conn() as conn:
                with conn.cursor() as c:
                    c.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id SERIAL PRIMARY KEY,
                        user_id TEXT,
                        started_at TIMESTAMP,
                        ended_at TIMESTAMP,
                        consent BOOLEAN DEFAULT TRUE,
                        user_agent TEXT
                    );
                    """)
                    c.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        conversation_id INTEGER REFERENCES conversations(id),
                        role TEXT,
                        content TEXT,
                        ts TIMESTAMP
                    );
                    """)
                conn.commit()
            return
        except Exception as e:
            st.session_state.use_postgres = False
            st.warning(f"PG-init epäonnistui ({e}); siirrytään SQLiteen.")
    # SQLite
    with _sqlite_conn() as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            started_at TEXT,
            ended_at TEXT,
            consent INTEGER DEFAULT 1,
            user_agent TEXT
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            ts TEXT,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
        """)
        conn.commit()

def start_conversation(user_id: str, user_agent: str) -> int:
    now = datetime.utcnow().isoformat()
    if _use_postgres():
        try:
            with _pg_conn() as conn:
                with conn.cursor() as c:
                    c.execute(
                        "INSERT INTO conversations (user_id, started_at, consent, user_agent) VALUES (%s, NOW(), %s, %s) RETURNING id",
                        (user_id, True, user_agent[:200] if user_agent else None)
                    )
                    conv_id = c.fetchone()[0]
                conn.commit()
            return int(conv_id)
        except Exception as e:
            st.session_state.use_postgres = False
            st.warning(f"PG-insert epäonnistui ({e}); siirrytään SQLiteen.")
    with _sqlite_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO conversations (user_id, started_at, consent, user_agent) VALUES (?, ?, ?, ?)",
            (user_id, now, 1, user_agent[:200] if user_agent else None)
        )
        conv_id = c.lastrowid
        conn.commit()
    return int(conv_id)

def save_message(conversation_id: int, role: str, content: str):
    now = datetime.utcnow().isoformat()
    if _use_postgres():
        try:
            with _pg_conn() as conn:
                with conn.cursor() as c:
                    c.execute(
                        "INSERT INTO messages (conversation_id, role, content, ts) VALUES (%s, %s, %s, NOW())",
                        (conversation_id, role, content)
                    )
                conn.commit()
            return
        except Exception as e:
            st.session_state.use_postgres = False
            st.warning(f"PG-msg epäonnistui ({e}); siirrytään SQLiteen.")
    with _sqlite_conn() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now)
        )
        conn.commit()

def fetch_messages(conversation_id: int) -> List[Dict[str, Any]]:
    if _use_postgres():
        try:
            with _pg_conn() as conn:
                with conn.cursor() as c:
                    c.execute("""
                    SELECT role, content, ts FROM messages
                    WHERE conversation_id = %s
                    ORDER BY id ASC
                    """, (conversation_id,))
                    rows = c.fetchall()
            return [{"role": r[0], "content": r[1], "ts": r[2].isoformat() if r[2] else ""} for r in rows]
        except Exception as e:
            st.session_state.use_postgres = False
            st.warning(f"PG-fetch epäonnistui ({e}); siirrytään SQLiteen.")
    with _sqlite_conn() as conn:
        c = conn.cursor()
        c.execute("""
        SELECT role, content, ts FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """, (conversation_id,))
        rows = c.fetchall()
    return [{"role": r[0], "content": r[1], "ts": r[2]} for r in rows]

# ========= Wow-efekti: KPI & Governance =========

def render_kpi_table():
    data = [
        ("Asiakaspalvelu Copilot", "TTFR (time-to-first-response)", "90 s", "≤ 30 s", "LLM-luonnosvastaukset + tietopohja"),
        ("Asiakaspalvelu Copilot", "CSAT", "3.9 / 5", "≥ 4.3 / 5", "sävy & faktat kohdilleen"),
        ("Sisäinen RAG-haku", "Osumatarkkuus (nDCG@5)", "—", "≥ 0.85", "prosessidokit lähteiksi"),
        ("Fraud score", "Precision @ k", "—", "+15–25 %", "rules + ML, SHAP-seuranta"),
        ("AML triage", "Käsittelyaika / alert", "—", "−30–50 %", "LLM tiivistää & ehdottaa"),
        ("Luotonanto", "PD AUC", "—", "≥ 0.78", "selitettävyys-paneeli"),
    ]
    df = pd.DataFrame(data, columns=["Alue", "Mittari", "Nykytila", "Tavoite", "Huomio"])
    st.subheader("Ehdotetut KPI:t")
    st.dataframe(df, use_container_width=True)

def render_governance_flow():
    dot = r"""
    digraph G {
      rankdir=LR;
      node [shape=box, style="rounded,filled", color="#444444", fillcolor="#f5f5f5"];
      edge [color="#888888"];
      A [label="Käyttötapaus & riskiluokitus\n(EU AI Act)"];
      B [label="Data governance\n(omistajuus • laatu • DPIA)"];
      C [label="Mallikehitys\n(MLOps/LLMOps)"];
      D [label="Validoi & hyväksy\n(kriteerit, fairness, selitettävyys)"];
      E [label="Pilotointi\n(SLA/KPI seuranta)"];
      F [label="Tuotanto\n(drift, kustannus, audit trail)"];
      A -> B -> C -> D -> E -> F;
    }
    """
    st.subheader("AI governance – prosessi")
    st.graphviz_chart(dot, use_container_width=True)

def detect_intents(text: str) -> set[str]:
    t = text.lower()
    intents = set()
    if any(w in t for w in ["kpi", "mittari", "sla", "ttfr", "tavoite", "tavoitteet"]):
        intents.add("kpi")
    if any(w in t for w in ["governance", "ai act", "risk", "selitettävyys", "audit", "valvonta"]):
        intents.add("gov")
    return intents

# ========= CV-koukku =========

CV_HOOKS = {
    ("hubspot", "salesforce", "crm"): [
        "Olen rakentanut ja ylläpitänyt HubSpot–Salesforce-integraatioita, joten CRM-prosessit ovat tuttua maastoa.",
        "Kohdennuksen ja ICP-segmentoinnin tein Goforella myynnin ja markkinoinnin yhteiseksi kieleksi."
    ],
    ("rag", "tietopohja", "tietohaku", "ohje", "dokumentaatio"): [
        "Olen tehnyt sisäisiä RAG-konsepteja: kuratoidut ohje- ja prosessilähteet pitävät vastaukset faktoissa."
    ],
    ("fraud", "aml", "rahanpesu", "riskimalli"): [
        "Sääntöpohjaisen ja ML-pohjaisen riskipisteytyksen yhdistäminen on tuttua – selitettävyys (SHAP) mukaan alusta asti."
    ],
    ("governance", "ai act", "eettinen", "selitettävyys"): [
        "Tuon AI governance -periaatteet käytäntöön: riskiluokitus, hyväksymiskriteerit ja audit trail sisäänrakennettuna."
    ],
    ("copilot", "asiakaspalvelu", "service", "sla"): [
        "Asiakaspalvelun Copilotissa fokusoin TTFR-parannukseen ja sävy/fakta-laatuun – mittarit ja hyväksymiskriteerit ensin."
    ],
    ("tapahtuma", "event", "international", "messu"): [
        "Airbus-tausta ja kansainvälinen tapahtumatuotanto auttavat viemään AI-pilotit myös kentälle esiteltäviksi."
    ],
    ("kansainvälinen", "international", "culture"): [
        "Olen työskennellyt Suomessa, Saksassa ja Kiinassa – monikulttuurinen yhteistyö sujuu."
    ],
}

def build_cv_hook(user_query: str) -> str:
    q = user_query.lower()
    picked: list[str] = []
    for keys, lines in CV_HOOKS.items():
        if any(k in q for k in keys):
            picked.extend(lines[:1])
    if not picked:
        picked = ["Agentti Henry:"]
    return " ".join(picked[:2])

# ========= Yhteys-CTA =========

CONTACT_EMAIL = st.secrets.get("CONTACT_EMAIL", "")
CALENDLY_URL = st.secrets.get("CALENDLY_URL", "")

def transcript_text(conversation_id: int) -> str:
    msgs = fetch_messages(conversation_id)
    lines = []
    for m in msgs:
        lines.append(f"[{m['ts']}] {m['role'].upper()}: {m['content']}")
    return "\n".join(lines)

def render_connect_cta(last_user_msg: str = ""):
    st.markdown("### Ota yhteys")
    cols = st.columns(2)
    with cols[0]:
        if CONTACT_EMAIL:
            subject = "Hei Henry – jatketaan juttua"
            body = f"Moi Henry,%0D%0A%0D%0AAsiani: {last_user_msg[:200]}%0D%0A%0D%0ATerveisin, {st.session_state.get('audience_name','')}"
            st.link_button("📧 Sähköposti", f"mailto:{CONTACT_EMAIL}?subject={subject}&body={body}")
    with cols[1]:
        if CALENDLY_URL:
            st.link_button("📅 Varaa aika", CALENDLY_URL)

def wants_connect(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in [
        "ota yhteys", "yhdistä", "voitko välittää", "soita", "mailaa", "sähköposti", "varaa aika", "tapaaminen", "connect"
    ])

# ========= UI =========

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    initial_sidebar_state="collapsed",
    layout="wide",
)

# CSS viimeistely
st.markdown("""
<style>
.hero {
  display: flex; flex-direction: column; align-items: center; text-align: center;
  gap: 14px; padding: 28px; margin: 6px 0 14px 0;
  border-radius: 18px; border: 1px solid rgba(120,120,120,0.2);
  background: linear-gradient(180deg, rgba(150,150,150,0.06), rgba(120,120,120,0.04));
}
.hero img { width: 120px; height: 120px; border-radius: 50%; box-shadow: 0 6px 24px rgba(0,0,0,0.15); object-fit: cover; }
.hero h1 { font-size: 1.6rem; margin: 0; }
.hero p { margin: 0; opacity: 0.85; }
.footer-note { opacity:0.7; font-size: 0.9rem; margin-top: 8px;}
</style>
""", unsafe_allow_html=True)

# Hero
def get_avatar_url() -> str:
    direct = st.secrets.get("GITHUB_AVATAR_URL", "")
    if direct:
        return direct
    user = st.secrets.get("GITHUB_USERNAME", "")
    if user:
        return f"https://github.com/{user}.png?size=240"
    return "https://api.dicebear.com/7.x/thumbs/svg?seed=Henry"

avatar_url = get_avatar_url()
st.markdown(
    f"""
<div class="hero">
  <img src="{avatar_url}" alt="Henry avatar" />
  <h1>Tutustu Henryn CV:seen</h1>
  <p>Data- ja AI-vetoista markkinointia, CRM-kehitystä ja käytännön tekemistä. Kysy mitä vain! ✨</p>
  <div class="footer-note">Demo tallentaa keskustelut anonyymisti tietokantaan.</div>
</div>
""",
    unsafe_allow_html=True,
)

# Status-sivupalkki (vain OpenAI API -info)
with st.sidebar:
    st.subheader("Status")
    if get_client():
        st.info("Henry-agentti linjoilla: ✅")
    else:
        st.warning("API-yhteys puuttuu: lisää OPENAI_API_KEY Secretsiin.")

# DB init
init_db()

# Anonyymi user_id
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user-{os.urandom(4).hex()}"

# Aloita uusi conversation
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = start_conversation(st.session_state.user_id, user_agent="")

# Viestipinon ja profiilitietojen alustus
if "messages" not in st.session_state:
    st.session_state.messages = []  # tyhjä pino aluksi
if "profile_text" not in st.session_state:
    st.session_state.profile_text = None
if "audience" not in st.session_state:
    st.session_state.audience = None
if "audience_name" not in st.session_state:
    st.session_state.audience_name = ""
if "audience_company" not in st.session_state:
    st.session_state.audience_company = ""
if "system_built" not in st.session_state:
    st.session_state.system_built = False  # rakennetaan vasta kun profiili on saatu

# Ensimmäinen tervehdys jos pino tyhjä
if not st.session_state.messages:
    greeting = "Hei, kukas sinä olet ja miten voin auttaa? 😊"
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    save_message(st.session_state.conversation_id, "assistant", greeting)

# Näytä koko historia (emme oleta system-viestiä indeksissä 0)
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
user_msg = st.chat_input("Kirjoita tähän…")
if user_msg:
    # käyttäjän viesti talteen ja ruutuun
    st.session_state.messages.append({"role": "user", "content": user_msg})
    save_message(st.session_state.conversation_id, "user", user_msg)
    with st.chat_message("user"):
        st.markdown(user_msg)

    # Rakennetaan system-prompt ensimmäisen esittäytymisen perusteella
    if not st.session_state.system_built:
        st.session_state.profile_text = user_msg
        aud = classify_profile(user_msg)
        name, company = extract_name_company(user_msg)
        st.session_state.audience = aud
        st.session_state.audience_name = name
        st.session_state.audience_company = company

        aud_block = build_audience_block(aud, name, company)
        system_prompt = (
            f"{PERSONA}\n\n"
            f"{aud_block}\n"
            f"ABOUT_ME:\n{ABOUT_ME.strip()}\n\n"
            f"ROOLIN TIIVISTELMÄ:\n{JOB_AD_SUMMARY.strip()}\n\n"
            "Kun sinulta kysytään ideoita tai etenemistä, tarjoa:\n"
            "- lyhyet ratkaisuehdotukset (mitä toteutetaan, millä teknologioilla)\n"
            "- KPI-ehdotukset ja hyväksymiskriteerit\n"
            "- 30/60/90 päivän askelmerkit\n"
            "- AI governance -näkökulmat (EU AI Act, riskit, kontrollit)\n"
        )
        # lisää system-viesti pinoon ALKUUN
        st.session_state.messages.insert(0, {"role": "system", "content": system_prompt})
        save_message(st.session_state.conversation_id, "system", system_prompt)
        st.session_state.system_built = True

    # OpenAI-vastaus
    client = get_client()
    if client:
        try:
            reply_text = call_chat(client, st.session_state.messages)
        except Exception as e:
            st.error(f"OpenAI-virhe: {e.__class__.__name__}")
            reply_text = (
                f"Kiitos! Backend ei vastaa juuri nyt. Tässä suuntaviivat:\n\n"
                f"{bullets_ai_opportunities()}\n\n{bullets_ai_governance()}"
            )
    else:
        reply_text = (
            f"API-avain puuttuu. Tässä suuntaviivat:\n\n"
            f"{bullets_ai_opportunities()}\n\n{bullets_ai_governance()}"
        )

    # CV-koukku vastauksen alkuun
    hook = build_cv_hook(user_msg)
    reply_text = f"_{hook}_\n\n{reply_text}"

    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    save_message(st.session_state.conversation_id, "assistant", reply_text)

    with st.chat_message("assistant"):
        st.markdown(reply_text)
        # intent-pohjaiset visut
        intents = detect_intents(user_msg)
        if "kpi" in intents:
            render_kpi_table()
        if "gov" in intents:
            render_governance_flow()
        # Yhteys
        if wants_connect(user_msg):
            # Yhteys vain pyydettäessä
         if wants_connect(user_msg):
          st.info("Hienoa! Tässä suorat yhteystavat ↓")
          render_connect_cta(user_msg)
