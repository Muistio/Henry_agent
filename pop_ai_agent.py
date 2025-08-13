#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Henry AI advisor -demo (Streamlit) — turvallinen, nano-ensisijainen + fallbackit

- Ei dokumenttien latausta / RAG:ia – vain chatti
- Persona + ABOUT_ME + työpaikkailmoituksen tiivistelmä system-promptissa
- API-avain luetaan VAIN palvelimelta: Streamlit Secrets tai ympäristömuuttuja
- gpt-5-nano ensisijainen (ei temperature-paramia), fallback: gpt-4o, gpt-4o-mini (temperature=0.3)
- Sivupalkissa kevyt diagnostiikka (ei näytä avainta)
"""

import os
from typing import List, Dict, Any, Optional

import streamlit as st
from openai import OpenAI

APP_NAME = "Botti Henry 🤖"
DEFAULT_MODEL = "gpt-4.1-mini"  # ensisijainen, halpa malli

# ===== Henryn tausta (ABOUT_ME) =====
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

# ===== Työpaikkailmoituksen tiivistelmä =====
JOB_AD_SUMMARY = """
POP Pankkikeskuksen AI Advisor vastaa pankkiryhmän AI-kehityksen suunnittelusta ja koordinoinnista,
AI-ratkaisujen suunnittelusta ja mallinnuksesta, ennustavan analytiikan kehittämisestä, AI-käytäntöjen
juurruttamisesta, prosessi- ja data-analyysistä, Data- ja Tekoälystrategian tukemisesta sekä sisäisestä
AI-asiantuntijuudesta ja koulutuksesta. Eduksi: AI governance ja EU AI Act -osaaminen.
"""

# ===== Persona =====
PERSONA = (
    "Olen Henry – haen POP Pankkikeskuksen AI Advisor -rooliin. "
    "Puhun minä-muodossa luonnollisesti ja napakasti. Puhun bisneslähtöisesti mutta huumorilla. Käytän luontevasti mutta niukasti emojia. "
    "Annan konkreettisia askelmerkkejä (30/60/90 pv), määrittelen KPI:t ja huomioin AI-governancen (EU AI Act). "
    "Vältän hypeä ja perustelen riskit sekä hyödyt. Käytän alla olevaa taustaa (ABOUT_ME) ja roolin vaatimuksia."
)

# ===== Pikatools-tekstit =====
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

# ===== Avaimen luku vain palvelinpuolelta =====
def get_api_key() -> str:
    try:
        v = st.secrets.get("OPENAI_API_KEY", "")
        if v:
            return v
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY", "")

def get_api_key_source() -> str:
    try:
        if st.secrets.get("OPENAI_API_KEY", ""):
            return "secrets"
    except Exception:
        pass
    if os.getenv("OPENAI_API_KEY", ""):
        return "env"
    return "missing"

def get_client() -> Optional[OpenAI]:
    key = get_api_key()
    if not key:
        return None
    try:
        return OpenAI(api_key=key, timeout=30.0)
    except Exception:
        return None

# ===== Paikallinen fallback-vastaus =====
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

# ===== Mallin varamoodi (nano ilman temperaturea) =====
def try_chat_with_fallbacks(client: OpenAI, base_model: str, messages: List[Dict[str, str]]) -> str:
    # Järjestys: base_model -> gpt-4o -> gpt-4o-mini
    candidates = []
    if base_model:
        candidates.append(base_model)
    for alt in ("gpt-4o", "gpt-4o-mini"):
        if alt not in candidates:
            candidates.append(alt)

    last_err = None
    for m in candidates:
        try:
            kwargs = {"model": m, "messages": messages}
         
            if not m.startswith("gpt-5-mini"):
                kwargs["temperature"] = 0.3
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            st.error(f"Chat-virhe mallilla `{m}`: {e.__class__.__name__}: {e}")
            continue
    raise last_err if last_err else RuntimeError("Tuntematon virhe chat-kutsussa")

# ===== UI =====
st.set_page_config(page_title=APP_NAME, page_icon="🤖")
st.title(APP_NAME)
st.caption("Keskustele 'Henry'-agentin kanssa tästä AI Advisor -roolista.")

with st.sidebar:
    st.subheader("Asetukset")
    st.markdown("---")
    # Diagnostiikka: mistä avain löytyy (ei näytä avainta)
    src = get_api_key_source()
    if src == "secrets":
        st.info("Botti-Henry linjoilla: ✅ ")
    elif src == "env":
        st.info("API-yhteys: ✅ (Environment)")
    else:
        st.warning("API-yhteys: ❌ ei avainta")

# Viestipino
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
    if client:
        try:
            reply_text = try_chat_with_fallbacks(client, DEFAULT_MODEL, st.session_state.messages)
        except Exception:
            st.warning("OpenAI-chat ei toiminut varamalleillakaan → näytetään paikallinen demovastaus.")
            reply_text = local_demo_response(user_msg)
    else:
        reply_text = local_demo_response(user_msg)

    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    with st.chat_message("assistant"):
        st.markdown(reply_text)
