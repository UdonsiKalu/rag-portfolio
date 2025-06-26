Retrieval-Augmented Generation (RAG) Portfolio
This repository showcases a collection of domain-adapted RAG agents designed for real-world applications in healthcare, claims analysis, and intelligent document reasoning. Each project demonstrates a unique retrieval and generation architecture tailored for specific tasks — from CMS policy interpretation to clinical Q&A and risk stratification.

Included Projects
#	Project	Description
1	Policy-Guided Claim Denial Agent (RAP)	Analyzes claim submissions and provides CMS-aligned denial rationales and appeal suggestions
2	CMS Q&A Bot	Classic RAG setup to answer questions from Medicare coverage documents
3	Payer Policy Fusion Bot	Aligns and compares policy language across multiple payers using hybrid retrieval
4	RAG + Re-Ranker (w/ Eval)	Combines semantic search with reranking and evaluation tools for retrieval precision
5	Clinical ChatBot (Conversational RAG)	Multi-turn chatbot for navigating clinical policies and documentation context
6	CPT Risk Extractor (RHE)	Extracts billing risk and flags inconsistencies from CPT-code-based patterns
7	RAG Decision Agent	Orchestrates multiple tools with a controller to route retrieval, reasoning, and action
Each module is self-contained with its own demo app, RAG pipeline, and documentation.

Features
Modular project layout

FAISS- and hybrid-based retrievals

Prompt engineering with CMS-aligned constraints

Streamlit demos for each agent

Local and cloud-ready deployment

Readiness for evaluation and productionization

Navigation
bash
projects/
├── 1-policy-denial-agent/
├── 2-cms-qa-bot/
├── ...
└── 7-rag-decision-agent/
Each folder includes:

streamlit_app.py: launchable app

src/: pipeline logic, retrievers, models

data/: indexes or policy source files

README.md: use case + setup guide

Setup (for local use)
bash
git clone https://github.com/yourname/rag-portfolio.git
cd projects/1-policy-denial-agent
pip install -r requirements.txt
streamlit run streamlit_app.py

License & Contact
This work is open-sourced for educational and demonstration purposes. For inquiries, collaboration, or deployment support: [your.email@domain]