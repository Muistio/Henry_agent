# Pop AI advisor – Henry agent (Streamlit)

Kevyt demo, jonka avulla POP Pankin rekry voi keskustella "Henry Agentin" kanssa
AI Advisor -roolista, konkreettisista AI-ideoista pankkiympäristössä sekä AI-governancesta.

## Mitä demo näyttää
- Persona: sinut mallinnettu agentiksi (CV/tausta lyhyesti personassa).
- RAG: agentti hyödyntää työpaikkailmoitusta sekä lisättyjä CV/cover PDF/TXT -tiedostoja.
- Pankkikohtaiset pikatyökalut: asiakaspalvelun Copilot, fraud score, AML triage, sisäinen RAG.
- AI governance -tarkistuslista: EU AI Act -kulma, riskit, monitorointi, selitettävyys.

## Pika-aloitus paikallisesti
```bash
pip install -U -r requirements.txt
export OPENAI_API_KEY=your_key_here
streamlit run pop_ai_agent.py
```

## Web-käyttöönotto (valitse yksi)

### 1) Streamlit Community Cloud (helpoin)
1. Vie kansio GitHubiin (esim. repo `pop-ai-advisor-agent`).
2. Mene https://streamlit.io/cloud → New app → valitse repo, haara ja tiedosto `pop_ai_agent.py`.
3. Aseta `Secrets` / `Advanced settings` kohdassa ympäristömuuttuja: `OPENAI_API_KEY=xxxxx`.
4. Deploy. Jaa syntyvä URL hakemuksessa.

### 2) Hugging Face Spaces (Streamlit)
1. Luo Space → valitse *Streamlit*.
2. Lisää repo-tiedostot (`pop_ai_agent.py`, `requirements.txt`, `README.md`).
3. Aseta *Variables and secrets* → lisää `OPENAI_API_KEY` salaisuutena.
4. Build käynnistyy automaattisesti → saat julkisen URL:n.

### 3) Railway / Render (Python web service)
- Start-komento: `streamlit run pop_ai_agent.py --server.port $PORT --server.address 0.0.0.0`
- Lisää `OPENAI_API_KEY` ympäristömuuttujaksi.
- Ota automaattinen deploy käyttöön GitHub-kytkennällä.

## Turvallisuus ja tietosuoja
- Dokumentit indeksoidaan muistissa, eivät tallennu palvelimeen (poislukien OpenAI API -kyselyt).
- Poista/redusoi henkilötiedot dokumenteista ennen julkista jakoa.

## Demo-kysymyksiä
- "Mitä nopeita 2–4 viikon POC-ideoita tekisit POP Pankille?"
- "Miten EU AI Act vaikuttaa tähän käyttöön? Mitä kontrollit edellyttävät?"
- "Laadi 30/60/90 päivän etenemissuunnitelma AI Advisor -rooliin."
