#!/usr/bin/env python3
"""
Pop AI Advisor – Personal Agent Demo (Streamlit)

Kevyt Streamlit-sovellus, jolla POP Pankin rekry voi jutella AI-agentin kanssa.
- Persona + keskustelumuisti
- RAG työpaikkailmoituksesta + (valinnaisesti) ladatuista tiedostoista
- Pienet työkalut (AI-ideat, governance-checklist)
- Turvalliset fallbackit (ei kaadu vaikka API ei olisi käytettävissä)
"""

import io
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable

import numpy as np
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader

# -------------------------------
# Perusasetukset
# -------------------------------
DEFAULT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
APP_NAME = "Pop AI advisor – agent"

# Henkilökohtaisia tietoja
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
  • “Finnish watch” hakutermin kärkisijoitukset, valikoimaan mm. Stockmann

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
- Haluan tuoda perinteiselle toimialalle konkreettisia, mitattavia AI-ratkaisuja (asiakaspalvelu Copilot, AML/ fraud-käsittelyn tehostus, sisäinen RAG, ennustava analytiikka) ja rakentaa pysyvät prosessit (MLOps/LLMOps, monitorointi, audit trail).
"""


# Työpaikkailmoitus (ydin)
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

# Persona / system-prompt lyhyenä
PERSONA = (
    "Olen Henry – haen POP Pankkikeskuksen AI Advisor -rooliin. "
    "Puhun suomea ja vastaan napakasti, liiketoimintaa edistävästi ja konkreettisilla askelmerkeillä. "
    "Käytän minä-muotoa, koska agentin tarkoitus on edustaa minua hakijana. "
    "Hyödynnän alla olevaa taustakuvausta (ABOUT_ME) ja keskustelussa ladattuja dokumentteja (CV/cover/job ad). "
    "Priorisoin ratkaisuehdotuksia, hyväksymiskriteerejä (KPI), aikatauluja (30/60/90 pv). "
    "Vältän hypeä, perustelen hyödyt ja riskit, ja annan konkreettiset seuraavat stepit."
)


# -------------------------------
# In-memory "vektorikauppa"
# -------------------------------
@dataclass
class Chunk:
    doc_id: str
    text: str
    vector: np.ndarray
    meta: Dict[str, Any]

class MiniStore:
    def __init__(self):
        self.chunks: List[Chunk] = []

    def add_doc(self, doc_id: str, text: str, embedder: Optional[Callable[[str], np.ndarray]], meta: Dict[str, Any]):
        for piece in split_into_chunks(text, 1200):
            vec = None
            if embedder is not None:
                try:
                    vec = embedder(piece)
                except Exception:
                    vec = None
            if vec is None:
                # Fallback vektori, jotta appi ei kaadu quota-/avainongelmissa
                vec = np.zeros(8, dtype=np.float32)
            self.chunks.append(Chunk(doc_id=doc_id, text=piece, vector=vec, meta=meta))

    def search(self, query: str, embedder: Optional[Callable[[str], np.ndarray]], k: int = 5):
        if not self.chunks:
            return []
        qv = None
        if embedder is not None:
            try:
                qv = embedder(query)
            except Exception:
                qv = None

        if qv is not None and np.linalg.norm(qv) > 0:
            # Embedding-haku
            scores = [(cosine_sim(qv, ch.vector), ch) for ch in self.chunks]
        else:
            # Fallback: yksinkertainen avainsanapisteytys
            qwords = [w for w in query.lower().split() if len(w) > 2]
            def kw_score(txt: str):
                lt = txt.lower()
                return sum(lt.count(w) for w in qwords) if qwords else 0
            scores = [(kw_score(ch.text), ch) for ch in self.chunks]

        scores.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scores[:k]]

# -------------------------------
# OpenAI-apurit (avaimen haku)
# -------------------------------
_client_cache: Dict[str, OpenAI] = {}

def _get_api_key_from_anywhere() -> str:
    # 1) Sidebarin syöte (tallennamme sen session stateen alempana)
    ki = st.session_state.get("OPENAI_API_KEY_INPUT", "")
    # 2) Ympäristömuuttuja
    if not ki:
        ki = os.getenv("OPENAI_API_KEY", "")
    # 3) Streamlit Secrets (Cloud)
    if not ki:
        try:
            ki = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            ki = ""
    return ki

def get_client(api_key: Optional[str] = None) -> OpenAI:
    key = api_key or _get_api_key_from_anywhere()
    if not key:
        raise RuntimeError("OPENAI_API_KEY ei ole asetettu. Lisää se sivupalkissa, ympäristömuuttujana tai Streamlit Secretsiin.")
    if key in _client_cache:
        return _client_cache[key]
    cli = OpenAI(api_key=key)
    _client_cache[key] = cli
    return cli

def embed_text(text: str, client: OpenAI) -> np.ndarray:
    text = text.replace("\n", " ")
    try:
        emb = client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding
        return np.array(emb, dtype=np.float32)
    except Exception:
        # Fallback nollavektori → ei kaatumista quota-virheissä
        st.session_state["embed_fallback"] = True
        return np.zeros(8, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)

def split_into_chunks(text: str, max_chars: int) -> List[str]:
    text = " ".join(text.split())
    out: List[str] = []
    buf: List[str] = []
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

def read_pdf(file: io.BytesIO) -> str:
    reader = PdfReader(file)
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(parts)

# -------------------------------
# Pienet työkalut (bullets)
# -------------------------------
def tool_bullets_ai_opportunities() -> str:
    return "\n".join([
        "1) Asiakaspalvelu Copilot: summaus, vastaus-ehdotukset, CRM-kirjaus.",
        "2) Fraud score (rules+ML): signaalifuusio, SHAP-seuranta.",
        "3) AML alert triage: priorisointi + tutkintamuistion runko.",
        "4) Ennustava luotonanto: PD/LGD + selitettävyys-paneeli.",
        "5) Tietopyyntöjen automaatio: ohjattu haku, audit-logi.",
        "6) Sisäinen RAG-haku: ohjeet, prosessit, mallidokit.",
    ])

def tool_ai_governance_checklist() -> str:
    return "\n".join([
        "• Data governance: omistajuus, laatu, säilytys, DPIA tarpeen mukaan.",
        "• Mallien elinkaari: versiointi, hyväksyntä, monitorointi (drift/bias).",
        "• Selitettävyys: SHAP/LIME tai policy, milloin vaaditaan.",
        "• EU AI Act: luokitus, kontrollit, rekisteröinti tarvittaessa.",
        "• Riskienhallinta: human-in-the-loop, fallback, vaikutusarvio.",
        "• Tietoturva & pääsynhallinta: salaisuudet, auditointi.",
    ])

# -------------------------------
# Safe chat + local fallback
# -------------------------------
def local_demo_response(query: str, context: str) -> str:
    bullets = tool_bullets_ai_opportunities()
    gov = tool_ai_governance_checklist()
    plan = (
        "### 30/60/90 päivän suunnitelma\n"
        "- **30 pv**: Kartoitus (käyttötapaukset, datalähteet), nopea POC (asiakaspalvelu Copilot tai sisäinen RAG), "
        "governance-periaatteet ja hyväksymiskriteerit.\n"
        "- **60 pv**: POC → pilotiksi, mittarit (SLA/CSAT/TTFR/fraud-precision), monitorointi (drift/bias), "
        "dokumentaatio ja koulutus.\n"
        "- **90 pv**: Skaalaus (lisätiimit/prosessit), kustannus/vaikutusanalyysi, backlogin priorisointi, "
        "tuotantoprosessi (MLOps/LLMOps).\n"
    )
    ctx_note = f"> **Konteksti (poimintoja):**\n{context[:1000]}\n\n" if context else ""
    return (
        "#### Paikallinen demotila (ei OpenAI-vastauksia)\n"
        "OpenAI-kutsu ei ole käytettävissä (avain/kiintiö/verkko). Alla ehdotuksia demoa varten:\n\n"
        f"{ctx_note}"
        "#### Pankin AI-mahdollisuudet\n"
        f"{bullets}\n\n"
        "#### AI governance – muistilista\n"
        f"{gov}\n\n"
        f"{plan}"
        "Pyydä täydentämään yksityiskohdat tai lataamaan dokumentteja (PDF/TXT), niin demo viittaa niihin RAG-haulla."
    )

def safe_chat_completion(client: OpenAI, model: str, messages: list, temperature: float = 0.3):
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    except Exception:
        # Älä kaada sovellusta, kerro mitä tapahtui ja palauta None
        st.warning("OpenAI-chat ei ole käytettävissä (avain/kiintiö/verkko). Näytetään paikallinen demovastaus.")
        return None

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🤖")
st.title(APP_NAME)


# Client (jos avain on saatavilla)
client: Optional[OpenAI] = None
try:
    key_try = _get_api_key_from_anywhere()
    if key_try:
        client = get_client(key_try)
except Exception as e:
    st.error(str(e))

# Store ja bootstrap
if "store" not in st.session_state:
    st.session_state.store = MiniStore()
if "bootstrapped" not in st.session_state:
    st.session_state.bootstrapped = False

store: MiniStore = st.session_state.store

# Bootstrap: lisätään job ad (+ tyhjä ABOUT_ME)
if not st.session_state.bootstrapped:
    emb_fn = (lambda txt: embed_text(txt, client)) if client else None
    store.add_doc("job_ad", JOB_TEXT, emb_fn, {"source": "job_ad"})
    if ABOUT_ME.strip():
        store.add_doc("about_me", ABOUT_ME, emb_fn, {"source": "about_me"})
    st.session_state.bootstrapped = True

# Banneri, jos embeddings fallback on päällä
if st.session_state.get("embed_fallback"):
    st.info("Embeddings ei käytettävissä (avain/kiintiö). Haku toimii avainsanoilla, chatilla on paikallinen fallback.")

# Uploadit
if up_files:
    emb_fn = (lambda txt: embed_text(txt, client)) if client else None
    for f in up_files:
        text = read_pdf(io.BytesIO(f.getvalue())) if f.type == "application/pdf" else f.getvalue().decode("utf-8", errors="ignore")
        store.add_doc(f.name, text, emb_fn, {"source": f.name})
    st.success(f"Indeksoitu {len(up_files)} tiedosto(a).")

# Keskustelutila
if "messages" not in st.session_state:
    st.session_state.messages = []

def build_context(query: str) -> str:
    emb_fn = (lambda txt: embed_text(txt, client)) if client else None
    hits = store.search(query, emb_fn, k=5)
    ctx_parts: List[str] = []
    for h in hits:
        tag = h.meta.get("source", "doc")
        ctx_parts.append(f"[Lähde: {tag}]\n{h.text}")
    return "\n\n".join(ctx_parts)

user_text = st.chat_input("Kysy roolista, demoista tai projekteista…")
if user_text:
    st.session_state.messages.append({"role": "user", "content": user_text})

# Historia
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Vastaus (chat: API tai paikallinen fallback)
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    query = st.session_state.messages[-1]["content"]
    context = build_context(query)
    sys_prompt = (
        PERSONA
        + ("\n\nKonteksti (tiivistä, lainaa maltilla):\n" + context if context else "")
        + "\n\nPikatyökalut:\n"
        + tool_bullets_ai_opportunities()
        + "\n\nGovernance-checklist:\n"
        + tool_ai_governance_checklist()
    )

    if client is None:
        answer = local_demo_response(query, context)
    else:
        resp = safe_chat_completion(
            client=client,
            model=model,
            messages=[{"role": "system", "content": sys_prompt}] + st.session_state.messages,
            temperature=0.3,
        )
        if resp is None:
            answer = local_demo_response(query, context)
        else:
            answer = resp.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

# Footer
st.markdown("---")
st.subheader("Mitä tämä demo näyttää")
st.markdown(
    "- Keskusteltava agentti, joka tuntee työpaikkailmoituksen.\n"
    "- RAG-haku job adista ja (valinnaisesti) ladatuista dokumenteista.\n"
    "- Valmiit AI-ideat ja AI governance -tarkistuslista.\n"
    "- Turvalliset fallbackit, ettei appi kaadu vaikka embeddings- tai chat-quota loppuisi."
)
