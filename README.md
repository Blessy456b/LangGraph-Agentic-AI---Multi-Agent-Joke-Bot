# 🤖 Multi-Agent LangGraph Joke Bot

A **LangGraph-based Agentic AI system** that generates, critiques, scores, and rewrites jokes using the **Groq LLM** (LLaMA 3.1–70B).  
Built to demonstrate multi-agent orchestration, memory persistence, and adaptive humor scoring.

---

## 🧩 Features

- 🧠 **Joke Generator Agent** — Creates witty, original jokes from topics.
- 🤔 **Critic Agent** — Evaluates humor quality and assigns a 0–10 score.
- 😂 **HumorScore Agent** — Routes jokes for rewriting if score < 7.
- 😇 **Rewriter Agent** — Makes jokes family-friendly.
- 🧱 **Memory Agent** — Stores previously told jokes to avoid repetition.
- 🔄 **Agentic Workflow** — Built on `LangGraph`’s stateful graph-based orchestration.

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/multi-agent-joke-bot.git
cd multi-agent-joke-bot
python -m venv joke_env
source joke_env/bin/activate
pip install -r requirements.txt
```
Set your Groq API key:
```bash
Copy code
export GROQ_API_KEY="your_api_key_here"
```

## ▶️ Run the Bot
```bash

cd src
python multi_agent_joke_bot.py
```
Example interaction:
```

😂 Welcome to the Multi-Agent LangGraph Joke Bot (with Memory + HumorScore)!
Type a topic for a joke or 'exit' to quit.

Enter a topic: airplane
🧠 Original Joke: Why did the airplane go to therapy? Because it had turbulent emotions.
😂 Humor Score: 7
🤔 Critic’s Review: Clever wordplay, relatable setup, mild chuckle potential.
😇 Family-Friendly Version: Already suitable for all audiences.
🧩 Joke Memory: Tracks all previous jokes.
```
## 🧱 Architecture
Each agent is a LangGraph node:

css

[ tell_joke ] → [ critic ] → [ rewriter ] → [ memory ] → END

- tell_joke: Calls LLM for creative joke generation

- critic: Evaluates and scores joke humor

- rewriter: Makes joke family-safe if needed

- memory: Stores joke history

## 💡 Future Extensions
- 🤖 Add an Audience Feedback Agent (simulate laughter level)

- 📈 Track long-term humor performance

- 🧬 Plug in open-weight models via Ollama or vLLM

## 🪪 License
MIT License © 2025 Blessy Thomas
