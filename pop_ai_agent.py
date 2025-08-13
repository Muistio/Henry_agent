#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Henry AI advisor -demo (Streamlit) — chat + SQLite-loki + admin-näkymä
- Vain chatti (ei RAGia eikä tiedostonlatausta)
- Kaikkien käyttäjien keskustelut talteen SQLiteen palvelinpuolella (ilman erillistä kysymistä)
- Admin-näkymä salasanalla: listaus, haku, JSON/CSV-lataus
- API-avain vain palvelimella (secrets/env), ei koskaan UI:ssa
- Malli: gpt-4o-mini (nopea ja edullinen), ei UI-valintaa
- Sivupalkki on oletuksena piilotettu (collapsed)
"""

import os
import io
import csv
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI

# -------------------------------
# Tietokannan polku (kirjoituskelpoinen myös Streamlit Cloudissa)
# -------------------------------
if os.path.exists("/mount/data"):
    DB_DIR = "/mount/data"
else:
    DB_DIR = os.getcwd()  # paikallisesti nykyinen työhakemisto
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "chatlogs.db")

# -------------------------------
# Perusasetukset
# -------------------------------
APP_NAME = "Botti Henry 🤖"
DEFAULT_MODEL = "gpt-4o-mini"  # nopea ja edullinen

# -------------------------------
# Henryn tausta & persona
# -------------------------------
ABOUT_ME = """
Nimi: Henry
Rooli-identiteetti: AI-osaaja ja dataohjautuva markkinointistrategi (10+ vuotta), CRM-admin (HubSpot, Salesforce), Python-harrastaja ja sijoittamista harrastava.
Asuinmaat: Suomi, Saksa, Kiina.

Työkokemus:
- Gofore Oyj (2020–): Marketing strategist
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
  • Altisti kriittiselle viestinnälle (TETRA), kurssit: ensiapu, vaaralliset aineet, ym.

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
    "Annan konkreettisia askelmerkkejä (30/60/90 pv), määrittelen KPI:t ja huomioin AI-governancen (EU AI Act). "
    "Vältän hypeä ja perustelen riskit sekä hyödyt. Käytän alla olevaa taustaa (ABOUT_ME) ja roolin vaatimuksia."
)

# -------------------------------
# Pikatools-tekstit (vastausten tueksi)
# -------------------------------
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

# -------------------------------
# OpenAI: avain vain palvelimella
# -------------------------------
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

# -------------------------------
# SQLite apurit
# -------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
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
    conn.close()

def start_conversation(user_id: str, user_agent: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (user_id, started_at, consent, user_agent) VALUES (?, ?, ?, ?)",
        (user_id, datetime.utcnow().isoformat(), 1, user_agent[:200] if user_agent else None)
    )
    conv_id = c.lastrowid
    conn.commit()
    conn.close()
    return conv_id

def save_message(conversation_id: int, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (conversation_id, role, content, ts) VALUES (?, ?, ?, ?)",
        (conversation_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def fetch_conversations(limit: int = 200, search_user: str = "") -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if search_user:
        c.execute("""
        SELECT id, user_id, started_at, ended_at, consent, user_agent
        FROM conversations
        WHERE user_id LIKE ?
        ORDER BY id DESC LIMIT ?
        """, (f"%{search_user}%", limit))
    else:
        c.execute("""
        SELECT id, user_id, started_at, ended_at, consent, user_agent
        FROM conversations
        ORDER BY id DESC LIMIT ?
        """, (limit,))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "user_id": r[1],
            "started_at": r[2],
            "ended_at": r[3],
            "consent": bool(r[4]),
            "user_agent": r[5],
        } for r in rows
    ]

def fetch_messages(conversation_id: int) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    SELECT role, content, ts FROM messages
    WHERE conversation_id = ?
    ORDER BY id ASC
    """, (conversation_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "ts": r[2]} for r in rows]

# -------------------------------
# Paikallinen fallback-vastaus
# -------------------------------
def local_demo_response(user_query: str) -> str:
    plan = (
        "### 30/60/90 päivän suunnitelma\n"
        "- **30 pv**: Kartoitus (käyttötapaukset, datalähteet), nopea POC (asiakaspalvelu Copilot tai sisäinen RAG), governance-periaatteet ja hyväksymiskriteerit.\n"
        "- **60 pv**: POC → pilotiksi, mittarit (SLA/CSAT/TTFR/fraud-precision), monitorointi (drift/bias), dokumentaatio ja koulutus.\n"
        "- **90 pv**: Skaalaus (lisätiimit/prosessit), kustannus/vaikutusanalyysi, backlogin priorisointi, tuotantoprosessi (MLOps/LLMOps).\n"
    )
    return (
        "#### Paikallinen demotila (ei OpenAI-vastauksia)\n"
        "OpenAI-kutsu ei ole käytettävissä (avain/kiintiö/verkko). Alla suuntaviivat:\n\n"
        f"**Pyyntö:** {user_query}\n\n"
        "#### Pankin AI-mahdollisuudet\n"
        f"{bullets_ai_opportunities()}\n\n"
        "#### AI governance – muistilista\n"
        f"{bullets_ai_governance()}\n\n"
        f"{plan}"
        "Pyydä syventämään jotakin osa-aluetta tai antamaan konkreettiset KPI:t ja hyväksymiskriteerit."
    )

# -------------------------------
# Chat-vastaus OpenAI:lla
# -------------------------------
def call_chat(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🤖", initial_sidebar_state="collapsed")
st.title(APP_NAME)
st.caption("Keskustele 'Henry'-agentin kanssa ja tutustu minuun")

# 1) DB init
init_db()

# 2) Anonyymi käyttäjä-ID
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user-{os.urandom(4).hex()}"

# 3) Aloita uusi conversation (tallennus aina päällä)
if "conversation_id" not in st.session_state:
    user_agent = ""  # Streamlit ei anna UA:ta suoraan
    st.session_state.conversation_id = start_conversation(st.session_state.user_id, user_agent)

# 4) Sivupalkki (oletus collapsed): status + admin
with st.sidebar:
    st.subheader("Asetukset")
    # API-yhteyden tila
    if get_client():
        st.info("API-yhteys: ✅ käytössä")
    else:
        st.warning("API-yhteys: ❌ ei avainta")

    st.markdown("---")
    st.subheader("Admin")
    admin_pw = st.text_input("Admin-salasana", type="password", help="Aseta Streamlit Secrets → ADMIN_PASSWORD")
    admin_ok = (admin_pw and st.secrets.get("ADMIN_PASSWORD", "") == admin_pw)

    if admin_ok:
        st.success("Admin-näkymä käytössä")
        q = st.text_input("Hae käyttäjän tunnisteella (optional)")
        limit = st.number_input("Kuinka monta keskustelua näytetään", min_value=10, max_value=2000, value=200, step=10)
        convs = fetch_conversations(limit=int(limit), search_user=q or "")
        st.write(f"Löytyi {len(convs)} keskustelua")
        for conv in convs:
            with st.expander(f"ID {conv['id']} • {conv['user_id']} • {conv['started_at']} • consent={conv['consent']}"):
                msgs = fetch_messages(conv["id"])
                for m in msgs:
                    st.markdown(f"**{m['role']}** · _{m['ts']}_\n\n{m['content']}")
                # Lataukset
                json_data = json.dumps(msgs, ensure_ascii=False, indent=2)
                st.download_button(
                    label="⬇️ Lataa JSON",
                    data=json_data.encode("utf-8"),
                    file_name=f"conversation_{conv['id']}.json",
                    mime="application/json",
                    key=f"dl_json_{conv['id']}"
                )
                csv_buf = io.StringIO()
                writer = csv.writer(csv_buf)
                writer.writerow(["role", "content", "ts"])
                for m in msgs:
                    writer.writerow([m["role"], m["content"], m["ts"]])
                st.download_button(
                    label="⬇️ Lataa CSV",
                    data=csv_buf.getvalue().encode("utf-8"),
                    file_name=f"conversation_{conv['id']}.csv",
                    mime="text/csv",
                    key=f"dl_csv_{conv['id']}"
                )

# 5) System-prompt
if "messages" not in st.session_state:
    system_prompt = (
        f"{PERSONA}\n\n"
        f"ABOUT_ME:\n{ABOUT_ME.strip()}\n\n"
        f"ROOLIN TIIVISTELMÄ:\n{JOB_AD_SUMMARY.strip()}\n\n"
        "Kun sinulta kysytään ideoita tai etenemistä, tarjoa:\n"
        "- lyhyet ratkaisuehdotukset (mitä toteutetaan, millä teknologioilla)\n"
        "- KPI-ehdotukset ja hyväksymiskriteerit\n"
        "- 30/60/90 päivän askelmerkit\n"
        "- AI governance -näkökulmat (EU AI Act, riskit, kontrollit)\n"
    )
    st.session_state.messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]
    # tallenna system-viesti
    save_message(st.session_state.conversation_id, "system", system_prompt)

# 6) Näytä historia (ilman system-viestiä)
for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 7) Chat input
user_msg = st.chat_input("Kysy Henryltä roolista, demoista tai projekteista…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    save_message(st.session_state.conversation_id, "user", user_msg)

    with st.chat_message("user"):
        st.markdown(user_msg)

    client = get_client()
    if client:
        try:
            reply_text = call_chat(client, st.session_state.messages)
        except Exception as e:
            st.error(f"OpenAI-virhe: {e.__class__.__name__}: {e}")
            reply_text = local_demo_response(user_msg)
    else:
        reply_text = local_demo_response(user_msg)

    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    save_message(st.session_state.conversation_id, "assistant", reply_text)

    with st.chat_message("assistant"):
        st.markdown(reply_text)
