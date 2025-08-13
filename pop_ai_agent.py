#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Henry AI advisor -demo (Streamlit)

Kevyt demo, jossa "Henry"-agentti keskustelee POP Pankkikeskuksen AI Advisor -roolista.
- Ei dokumenttien latausta / RAG:ia – vain chatti
- Persona + ABOUT_ME + työpaikkailmoituksen tiivistelmä syötetään system-promptiin
- API-avain: sivupalkista, ympäristömuuttujasta tai Streamlit Cloud Secretsista
- Turvallinen fallback: jos OpenAI ei ole käytettävissä, näytetään paikallinen demovastaus
"""

import os
import streamlit as st
from typing import List, Dict, Any, Optional
from openai import OpenAI

# -------------------------------
# Perusasetukset
# -------------------------------
APP_NAME = "Henry AI advisor -demo"
DEFAULT_MODEL = "gpt-4o-mini"

# -------------------------------
# Henryn tausta (ABOUT_ME)
# -------------------------------
ABOUT_ME = """
Nimi: Henry
Rooli-identiteetti: AI-osaaja ja dataohjautuva markkinointistrategi (10+ vuotta), CRM-admin (HubSpot, Salesforce), Python-harrastaja ja sijoittaja.
Asuinmaat: Suomi, Saksa, Kiina.

Työkokemus:
- Gofore Oyj (2020–): Marketing strategist
  • Dataohjautuvat markkinointistrategiat ja Looker Studio -dashboardit
  • Myynnin ja konsulttitiimien tuki: kohdennus, ICP, segmentointi
  • Brändistrategiat yritysostoissa (4 kpl viime vuosina)
  • Marketing automation ja ABM-strategia
  • HubSpot & Salesforce integraatio ja ylläpito

- Airbus (2018–2020): Marketing manager
  • Viestinnän ja myynnin linjaus liiketoimintatavoitteisiin
  • Kampanja-analytiikka (EU–LATAM), tapahtumat
  • Mission-critical IoT -konseptointi
  • Verkkosivuprojektit (esim. airbusfinland.com)

- Rohje Oy (2018–): Co-founder (sivuprojekti)
  • Datalähtöinen kasvu, Shopify-optimoitu e-commerce
  • Google Ads & social, KPI-seuranta (CAC, ROAS)
  • “Finnish watch” -hakutermin kärkisijoitukset, valikoimaan mm. Stockmann

- Telia (2017): Marketing specialist (sijaisuus)
  • B2B-myyntiverkoston markkinoinnin kehitys, tapahtumat, B2B-some

- Digi Electronics, Shenzhen (2017): Marketing assistant (harjoittelu)
  • Adwords, Analytics, Smartly; Liveagent; valittu tiimin “employee of the quarter”

- Jyväskylä Entrepreneurship Society (2014–2016): Hallituksen pj (2015)
  • Spotlight-startup-tapahtuman käynnistäminen, laaja sidosryhmäverkosto

- Invivian (2023–): Investor (oma yhtiö)
  • Python-sijoitusanalytiikka (markkinadataskriptit, salkkuseuranta)

- Keski-Suomen Pelastuslaitos (2010–2017): VPK-palomies
  • Altisti kriittiselle viestinnälle (TETRA), kurssit: ensiapu, vaaralliset aineet, ym.

Koulutus:
- KTM, Jyväskylän yliopisto (2019–)
- Tradenomi, JAMK (2015–2018)
- Energia-ala opintoja, JAMK (2013–2015)
- Varusmiespalvelus: F/A-18 Hornet -mekaanikko (Ilmavoimat)

Kielet:
- Suomi (äidinkieli), Englanti (C1), Saksa (B1), Ruotsi (A1)

AI & data -osaamisen kohokohdat:
- Python-projektit: automatisoitu kaupankäynti (IBKR API), ML + sääntöpohja yhdistellen
- Liiketoimintalähtöinen AI: tunnistan arvokohteet, vien idean tuotantoon ja koulutan käyttäjät
- Google Cloud data/AI -tuntemus, Microsoft Copilot/Graph-integraatiot
- AI governance ja EU AI Act -näkökulma käytännön tekemiseen (riskit, kontrollit, selitettävyys)

Miksi POP Pankki:
- Haluan tuoda perinteiselle toimialalle konkreettisia, mitattavia AI-ratkaisuja (asiakaspalvelu Copilot, AML/fraud-käsittelyn tehostus, sisäinen RAG, ennustava analytiikka) ja rakentaa pysyvät prosessit (MLOps/LLMOps, monitorointi, audit trail).
"""

# -------------------------------
# Työpaikkailmoituksen tiivistelmä
# -------------------------------
JOB_AD_SUMMARY = """
POP Pankkikeskuksen AI Advisor vastaa pankkiryhmän AI-kehityksen suunnittelusta ja koordinoinnista,
AI-ratkaisujen suunnittelusta ja mallinnuksesta, ennustavan analytiikan kehittämisestä, AI-käytäntöjen
juurruttamisesta, prosessi- ja data-analyysistä, Data- ja Tekoälystrategian tukemisesta sekä sisäisestä
AI-asiantuntijuudesta ja koulutuksesta. Eduksi: AI governance ja EU AI Act -osaaminen.
"""

# -------------------------------
# Persona / toimintatapa
# -------------------------------
PERSONA = (
    "Olen Henry – haen POP Pankkikeskuksen AI Advisor -rooliin. "
    "Vastaan minä-muodossa, napakasti ja bisneslähtöisesti. "
    "Annan konkreettisia askelmerkkejä (30/60/90 pv), määrittelen KPI:t ja huomioin AI-governancen (EU AI Act). "
    "Vältän hypeä ja perustelen riskit sekä hyödyt. Käytän alla olevaa taustaa (ABOUT_ME) ja roolin vaatimuksia."
)

# -------------------------------
# Pikatools-tekstit
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
# Avainhaku: sivupalkki, env, secrets
# -------------------------------
def get_api_key() -> str:
    # 1) sivupalkin syöte session statesta
    v = st.session_state.get("OPENAI_API_KEY_INPUT", "")
    if v:
        return v
    # 2) ympäristömuuttuja
    v = os.getenv("OPENAI_API_KEY", "")
    if v:
        return v
    # 3) streamlit secrets
    try:
        v = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        v = ""
    return v

def get_client() -> Optional[OpenAI]:
    key = get_api_key()
    if not key:
        return None
    try:
        return OpenAI(api_key=key)
    except Exception:
        return None

# -------------------------------
# Fallback-vastaus jos API ei käytettävissä
# -------------------------------
def local_demo_response(user_query: str) -> str:
    plan = (
        "### 30/60/90 päivän suunnitelma\n"
        "- **30 pv**: Kartoitus (käyttötapaukset, datalähteet), nopea POC (asiakaspalvelu Copilot tai sisäinen RAG), "
        "governance-periaatteet ja hyväksymiskriteerit.\n"
        "- **60 pv**: POC → pilotiksi, mittarit (SLA/CSAT/TTFR/fraud-precision), monitorointi (drift/bias), "
        "dokumentaatio ja koulutus.\n"
        "- **90 pv**: Skaalaus (lisätiimit/prosessit), kustannus/vaikutusanalyysi, backlogin priorisointi, "
        "tuotantoprosessi (MLOps/LLMOps).\n"
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
# Streamlit UI
# -------------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🤖")
st.title(APP_NAME)

with st.sidebar:
    st.subheader("Asetukset")
    api_key_input = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    st.session_state["OPENAI_API_KEY_INPUT"] = api_key_input
    model = st.text_input("Chat-malli", value=DEFAULT_MODEL)
    key_status = "✅ avain löytyi" if get_api_key() else "❌ avain puuttuu"
    st.info(f"Avain: {key_status}")

st.caption("Keskustele 'Henry'-agentin kanssa tästä AI Advisor -roolista.")

# Viestipino
if "messages" not in st.session_state:
    # Alusta system-prompt: persona + about + job ad + pikatyökalut
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

# Näytä historia (ilman system-viestiä)
for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Syöte
user_msg = st.chat_input("Kysy Henryltä roolista, demoista tai projekteista…")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    client = get_client()
    reply_text = None

    if client:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=st.session_state.messages,
                temperature=0.3,
            )
            reply_text = resp.choices[0].message.content
        except Exception:
            # Älä kaada demoa – käytä paikallista fallbackia
            st.warning("OpenAI-chat ei ole käytettävissä (avain/kiintiö/verkko). Näytetään paikallinen demovastaus.")
            reply_text = local_demo_response(user_msg)
    else:
        # Ei avainta → paikallinen demo
        reply_text = local_demo_response(user_msg)

    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    with st.chat_message("assistant"):
        st.markdown(reply_text)
