#!/usr/bin/env python3
"""
Pop AI Advisor – Personal Agent Demo (Streamlit)
------------------------------------------------
A lightweight, self-contained Streamlit app that lets POP Pankki recruiters
chat with an AI agent modeled after you. It demonstrates:
  • Persona + conversation memory
  • RAG over your CV/cover letter + this job ad text
  • Small tool functions (project ideas, roadmap bullets, AI governance checklists)

Quick start
-----------
1) pip install -U streamlit openai pypdf numpy tiktoken
2) export OPENAI_API_KEY=...  (or set in the sidebar)
3) streamlit run pop_ai_agent.py

Notes
-----
- Embeddings: OpenAI `text-embedding-3-small` (cheap, good enough for demo)
- Chat model: gpt-4o-mini (swap if you prefer)
- All docs stay in memory; no external services. For a longer doc set, replace the
  in-memory store with a vector DB (FAISS/Chroma).
- The app ships with the POP job ad embedded so the agent can discuss the role.

Security & privacy
------------------
- No documents are uploaded off your machine beyond the OpenAI API calls you trigger.
- Redact personal data in uploads if you share the live demo link.
"""

import io
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# -------------------------------
# Config
# -------------------------------
DEFAULT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
APP_NAME = "POP AI Advisor – Henry Agent"

# The job ad text (Finnish) – pasted here to ground the agent
ABOUT_ME = """
ABOUT ME
Data-driven marketing strategist with 10+ years of experience turning data into
actionable business. Background spans B2B consulting, global tech and retail
entrepreneurship. Passionate investor.
Lived in Finland Germany and China.
WORK EXPERIENCE
Gofore Oyj Helsinki, Finland
Marketing strategist
01/11/2020 – Current
• Designed and executed data-driven marketing strategies for clients,
optimizing campaigns via Looker Studio dashboards
• Partnered with sales and consultant teams to align targeting strategies
with market and audience insights
• Brand strategies for mergers and acquisitions (4 acquisitions in the
past years)
• Marketing automation and account based marketing strategy
• Hubspot & Salesforce integration and administration
Gofore Plc is a digital transformation consultancy with more than 1500
impact-driven employees across Finland, Germany, Spain, and Estonia with a
revenue of more than 180M€.
Airbus Helsinki, Finland
Marketing manager
01/03/2018 – 01/11/2020
At Airbus I used to work with the following topics:
• Worked with technical teams and sales to align messaging with
strategic business goals
• Applied data analysis to evaluate European-LATAM wide campaign
performance
• Event management
• Designing the mission-critical Internet of Things concept
• Website projects e.g. Airbusfinland.com
Airbus Defence and Space is a division of Airbus responsible for defence and
aerospace products and services.
Airbus pioneers sustainable aerospace for a safe and united world. In
commercial aircraft, Airbus oﬀers modern and fuel-eﬃcient airliners and
associated services. Airbus is also a European leader in defence and security
and one of the world's leading space businesses.
Rohje Oy
Co-founder - Side hustle
2018 – Current
Co-founded Rohje, a Finnish watch brand with international sales, driving
growth through data-led marketing and Shopify e-commerce optimisation.
Managed digital ad campaigns (Google Ads, social media), tracking KPIs
including CAC and ROAS.
Combined creative storytelling with measurable performance metrics to
sustain growth in competitive markets.
Rohje has gained the pole position in Finnish Google searches for “Finnish
watch” and for example Zalando.com and the largest department store in
Finland, Stockmann has taken it on it's catalogue.
Telia Helsinki, Finland
Marketing specialist
01/04/2017 – 09/2017
Substitute role for maternity leave over the summer 2017.
• Telia’s nationwide B2B sales network’s marketing development
• Telia brand compatible B2B event management
• Telia Finland’s B2B-Social media management
Digi Electronics Ltd Shenzhen, China
Marketing assistant
01/01/2017 – 01/04/2017
3 months marketing internship in Shenzhen, China.
• Marketing planning and developing: Adwords, Analytics, Smartly
• Digital customer service: Liveagent
“Henry’s contribution to our company exceeded our expectations. He was a liked colleague amongst all team members.
He was selected as the employee of the quarter within his team. We are thankful to Henry’s proactive approach to
developing our shop front and marketing activities with a lot of improvement ideas.” Ville Majanen – Digi Electronics
Ltd.
Jyväskylä entrepreneurship Society
Chairman of the board
2014 – 2016
For 2015 I was promoted to be the chairman of the board. As a president and chairman my responsibility was to
coordinate the board and keep the society active and growing. This is a position where I got familiar with Finnish
start-up scene and the people behind it. We invented Spotlight event (start-up event in Jyväskylä which is still
running) and I got to known a lot of people who run main stakeholders in Finnish start-up ecosystem.
Invivian (Own company) Helsinki, Finland
Investor
2023 – Current
Python investment analytics platform – Built automated scripts for market data collection, analysis, and portfolio
tracking, improving decision-making speed and accuracy.
Keski-Suomen Pelastuslaitos / Middle Finland Rescue center Muurame, Finland
Firefighter
2010 – 01/01/2017
Volunteer firefighter for seven years in Muurame.
This experience brought me to the world of critical communications. We had Airbus TETRA-radios in operational
use.
Courses:
• Hazardious materials
• Driving
• First aid courses
• Firefighting courses
• Marine rescuing
• etc
EDUCATION AND TRAINING
01/08/2019 – CURRENT Jyväskylä, Finland
Master of Science in Economics and Business Administration University of Jyväskylä
Address Seminaarinkatu 15, 40100, Jyväskylä, Finland Website https://www.jyu.fi/jsbe/en
01/08/2015 – 01/08/2018 Jyväskylä, Finland
Bachelor of Business Administration Jyväskylä University of Applied Sciences
Address Rajakatu 35, 40200, Jyväskylä, Finland Website https://www.jamk.fi/
2013 – 2015 Jyväskylä
Energy engineering Jyväskylä University of Applied Sciences
Address Rajakatu 35, 40200, Jyväskylä Website https://www.jamk.fi/
2012 – 2013
Military service in as F/A-18 Hornet mechanic Finnish Air force
LANGUAGE SKILLS
MOTHER TONGUE(S): Finnish
Other language(s):
English
Listening C1
Spoken production C1
Reading C1
Spoken interaction C1
Writing C1
German
Listening B1
Spoken production B1
Reading B1
Spoken interaction B1
Writing B1
Swedish
Listening A1
Reading A1
Levels: A1 and A2: Basic user; B1 and B2: Independent user; C1 and C2: Proficient user
VOLUNTEERING
2021
United Nations department of safety and security - BSAFE
United Nations department of safety and security - BSAFE Course
Trainer for "first aid fire-extinguishing" - courses
Finnish: Alkusammutuskouluttaja
Kuopio, Finland
Civil crisis management courses: USAR / Camp PDT coursesCivil crisis management courses completed:
• Information Management
• USAR PDT
• Camp PDT
• Basic training course
• United Nations - Basic security in the field courses 1 and 2
"""

JOB_TEXT = """
Haemme POP Pankkikeskukseen: AI Advisor (AI-asiantuntijaa)

AI Advisorin keskeiset tehtävät:
- Suunnittelet ja koordinoit pankkiryhmän tekoälykehitystä tiiviissä yhteistyössä pankkien kanssa.
- Suunnittelet AI-ratkaisuja yhdessä muiden asiantuntijoiden kanssa sekä luot ja testaat itse tekoälymalleja.
- Kehität edistyksellistä, ennustavaa analytiikkaa yhdessä tiimin analyytikkojen kanssa.
- Jalostat AI-kehittämiskäytäntöjä osaksi organisaation toimintamalleja.
- Tutkit ja analysoit prosesseista ja datasta esiin uusia kehitysmahdollisuuksia.
- Tuet POP Pankki -ryhmän Data- ja Tekoälystrategiaa.
- Toimit sisäisenä AI-asiantuntijana ja kouluttajana.

Odotukset:
- Seuraat AI-teknologioiden kehittymistä ja sovellat niitä käytäntöön.
- Tunnistat hyödyntämismahdollisuuksia datasta ja prosesseista.
- Ymmärrät ML/AI:n ja osaat toteuttaa ratkaisuja palveluihin ja prosesseihin.
- Kokemusta edistyksellisestä analytiikasta.
- Tunnet Googlen Data- ja AI-teknologiat sekä Microsoft Copilotin ja sinulla on kokemusta niiden hyödyntämisestä.
- Tunnet kehittämismenetelmiä; yhteistyö sujuu sidosryhmien kanssa.
- Hallitset projektityöskentelyn; hyvät vuorovaikutus- ja koulutustaidot.
- Eduksi: AI governance, regulaatio (EU AI Act) ja niiden soveltaminen käytäntöön.
"""

# A concise persona/system prompt for the agent
PERSONA = f"""
Sinä olet "Henry Agent" – hakija Pop Pankkikeskuksen AI Advisor -rooliin.
Tyyli: selkeä, ratkaisu- ja bisneslähtöinen, rento mutta jämäkkä. Vastaa suomeksi, ellei käyttäjä
vaihda kieltä. Vastaa konkreettisesti ja ehdota askelmerkkejä.

Profiili pähkinänkuoressa:
- Tausta: marketing-strategia Goforella, CRM-admin (HubSpot, Salesforce), data & analytiikka.
- Rakentanut Python-pohjaisia AI- ja analytiikkatyökaluja (mm. automatisoitu kaupankäynti IBKR API:n päällä),
  yhdistää sääntöpohjan ja ML:n, tekee tuotantokelpoista koodia.
- Vahvuus: tunnistaa AI:n liiketoiminta-arvon, vie idean tuotantoon, kouluttaa käyttäjät, rakentaa prosessit.
- Tuntemus: Google Cloudin data/AI, Microsoft Copilot/Graph-integraatiot, AI governance -periaatteet,
  ja EU AI Act -näkökulma käytännön projektityöhön.

Tehtäväsi:
- Keskustele roolista, kyvykkyyksistä ja konkreettisista ratkaisu-ideoista POPin ympäristöön.
- Hyödynnä annettuja dokumentteja (CV, cover letter, job ad) faktapohjana.
- Kerro selkeitä vaiheita: toimintasuunnitelmat (30/60/90 pv), käyttökelpoisia
  aiheita (esim. asiakaspalvelun Copilot, fraud-scoren malli, AML-työn tehostus), ja riskit & governance.
- Jos kysytään demoista, ideoi nopeita proof-of-concepteja, joilla voidaan validoida arvo 2–4 viikossa.
- Vältä liiallista hypeä; tarjoa mitattavia mittareita ja hyväksymiskriteerejä.
"""

# --- Simple in-memory vector store -----------------------------------------
@dataclass
class Chunk:
    doc_id: str
    text: str
    vector: np.ndarray
    meta: Dict[str, Any]

class MiniStore:
    def __init__(self):
        self.chunks: List[Chunk] = []

    def add_doc(self, doc_id: str, text: str, embedder, meta: Dict[str, Any]):
        for piece in split_into_chunks(text, 1200):
            vec = embedder(piece)
            self.chunks.append(Chunk(doc_id=doc_id, text=piece, vector=vec, meta=meta))

    def search(self, query: str, embedder, k: int = 5):
        if not self.chunks:
            return []
        qv = embedder(query)
        scores = []
        for ch in self.chunks:
            s = cosine_sim(qv, ch.vector)
            scores.append((s, ch))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scores[:k]]

# --- Embedding helpers ------------------------------------------------------
_client_cache = {}

def get_client(api_key: str | None = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Use the sidebar to set it.")
    if key in _client_cache:
        return _client_cache[key]
    cli = OpenAI(api_key=key)
    _client_cache[key] = cli
    return cli


def embed_text(text: str, client: OpenAI) -> np.ndarray:
    text = text.replace("\n", " ")
    emb = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
    return np.array(emb, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


def split_into_chunks(text: str, max_chars: int) -> List[str]:
    text = " ".join(text.split())
    out, buf = [], []
    count = 0
    for token in text.split(" "):
        if count + len(token) + 1 > max_chars:
            out.append(" ".join(buf))
            buf, count = [token], len(token)
        else:
            buf.append(token)
            count += len(token) + 1
    if buf:
        out.append(" ".join(buf))
    return out

# --- PDF/Text ingestion -----------------------------------------------------

def read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(parts)

# --- Tooling shortcuts the agent can call implicitly via instruction --------

def tool_bullets_ai_opportunities() -> str:
    return (
        "\n".join([
            "1) Asiakaspalvelu Copilot: Summarisoi, ehdottaa vastauksia, kirjaa CRM:ään automaattisesti.",
            "2) Sääntöpohjainen + ML-fraud score: Reaaliaikainen signaalifuusio, SHAP-seuranta.",
            "3) AML alert triage: LLM priorisoi, generoi tutkinnan muistion rungon.",
            "4) Luotonannon ennustava analytiikka: PD/LGD-mallit + selitettävyyspaneeli.",
            "5) Tietopyyntöjen automaatio: LLM ohjaa hakemaan oikeista järjestelmistä, audit-logi.",
            "6) Sisäinen tiedonhaku (RAG): Ohjeet, prosessit, mallit – valvotut lähteet.",
        ])
    )


def tool_ai_governance_checklist() -> str:
    return (
        "\n".join([
            "• Data governance: omistajuus, laatukriteerit, säilytys, DPIA tarvittaessa.",
            "• Mallien elinkaari: versiointi, hyväksyntä, monitorointi (drift, bias, suorituskyky).",
            "• Mallien selitettävyys: SHAP/LIME tai policy, milloin selityksiä vaaditaan.",
            "• EU AI Act -luokitus ja kontrollit, rekisteröinti tarvittaessa.",
            "• Riskienhallinta: man-in-the-loop, fallback, virheen vaikutusanalyysi.",
            "• Tietoturva ja pääsynhallinta: avaimet, salaisuudet, auditointi.",
        ])
    )

# --- UI ---------------------------------------------------------------------

st.set_page_config(page_title=APP_NAME, page_icon="🤖")
st.title("Pop AI advisor – Henry agent")

with st.sidebar:
    st.subheader("Asetukset")
    api_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    model = st.text_input("Chat-malli", value=DEFAULT_MODEL)
    st.markdown("---")
    st.caption("Lisää CV/cover letter PDF:nä tai tekstinä. Ne indeksoidaan paikallisesti.")
    up_files = st.file_uploader("Lisää dokumentteja (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    st.markdown("---")
    if st.button("Tyhjennä keskustelu"):
        st.session_state.pop("messages", None)

# Prepare OpenAI client lazily
client: OpenAI | None = None
if api_key:
    try:
        client = get_client(api_key)
    except Exception as e:
        st.error(str(e))

# Build (or reuse) the store
if "store" not in st.session_state:
    st.session_state.store = MiniStore()
if "bootstrapped" not in st.session_state:
    st.session_state.bootstrapped = False

store: MiniStore = st.session_state.store

# Bootstrap with the job ad once
if client and not st.session_state.bootstrapped:
    def emb(txt: str):
        return embed_text(txt, client)
    store.add_doc("job_ad", JOB_TEXT, emb, {"source": "job_ad"})
    store.add_doc("about_me", ABOUT_ME, emb, {"source": "about_me"})
    st.session_state.bootstrapped = True

# Ingest uploads
if client and up_files:
    def emb(txt: str):
        return embed_text(txt, client)
    for f in up_files:
        if f.type == "application/pdf":
            text = read_pdf(io.BytesIO(f.getvalue()))
        else:
            text = f.getvalue().decode("utf-8", errors="ignore")
        store.add_doc(f.name, text, emb, {"source": f.name})
    st.success(f"Indeksoitu {len(up_files)} tiedosto(a).")

# Conversation state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper to build RAG context
def build_context(query: str) -> str:
    if not client or not store.chunks:
        return ""
    def emb(txt: str):
        return embed_text(txt, client)
    hits = store.search(query, emb, k=5)
    ctx = []
    for h in hits:
        tag = h.meta.get("source", "doc")
        ctx.append(f"[Lähde: {tag}]\n{h.text}")
    return "\n\n".join(ctx)

# Chat input
user_text = st.chat_input("Kysy roolista, demoista tai projekteista…")
if user_text and client:
    st.session_state.messages.append({"role": "user", "content": user_text})

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"]) 

# Respond
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and client:
    query = st.session_state.messages[-1]["content"]
    context = build_context(query)

    sys_prompt = PERSONA + "\n\n" + (
        "Konteksti (RAG, tiivistä ja lainaa vain tarpeen mukaan):\n" + context if context else ""
    ) + "\n\nKäytettävissä olevat pikat työkalut:\n" + tool_bullets_ai_opportunities() + "\n\n" + tool_ai_governance_checklist()

    # Call Chat Completions
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        ],
        temperature=0.3,
    )
    answer = resp.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# Footer tips
st.markdown("---")
st.subheader("Mitä tämä demo näyttää")
st.markdown(
    """
- Keskusteltava agentti, joka tuntee työpaikkailmoituksen ja omat dokumenttisi.
- RAG: agentti hakee vastauksiin pätkiä CV:stäsi/coverista ja tästä job adista.
- Valmiit työkalulistat: *mahdollisuudet pankissa* ja *AI governance -tarkistuslista*.
- Helppo jatkokehitys: lisää omat *tools()*-funktiot ja kutsu niitä ohjeilla personassa.
"""
)
