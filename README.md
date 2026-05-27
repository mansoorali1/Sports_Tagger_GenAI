---
title: MatchLens AI
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---


# MatchLens AI — Sports Commentary Auto-Tagger & Highlight Generator

End-to-end sports commentary pipeline combining NLP classification
and LLM-powered summarization to automate match highlight generation
for sports streaming platforms like DAZN.

## What It Does

Paste raw football commentary text → get instant multi-format match highlights:

- **Full Summary** — 150-200 word match narrative for editorial teams
- **Executive Summary** — 2-3 sentences for match result cards
- **Push Notification** — under 120 characters for mobile alerts
- **Key Moments** — structured JSON for VOD chapter markers

## Architecture
Raw Commentary Text
↓
Text Segmentation (regex)
↓
TF-IDF + Linear SVM Classifier (11 event classes)
↓
Impact Filtering (Goals, Cards, Penalties)
↓
LLaMA-3.1 via Groq (multi-format generation)
↓
Post-generation Guardrail (fact verification)
↓
Full Summary | Push Notification | Key Moments JSON

## Key Findings

| Model | F1 Weighted | F1 Macro | Notes |
|---|---|---|---|
| TF-IDF + LR | 0.9992 | 0.9897 | Keyword exploitation |
| TF-IDF + SVM | 0.9993 | 0.9901 | Strongest classical |
| BERT fine-tuned | 0.9997 | 0.9971 | Match-level split |

**Key insight:** Classical TF-IDF models match BERT performance on this
dataset because sports commentary contains explicit lexical triggers.
Transformer advantage is marginal overall but meaningful for ambiguous
events like red cards.

## Tech Stack

- **Classification:** TF-IDF + LinearSVC (scikit-learn)
- **LLM:** LLaMA-3.1-8B-Instant via Groq API
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Experiment Tracking:** MLflow
- **Deployment:** Docker + Hugging Face Spaces
- **CI/CD:** GitHub Actions

## Data

European football commentary dataset — 941,009 events across 9,074 matches
from 5 leagues (2011-2017). Events.csv + ginf.csv from Kaggle
