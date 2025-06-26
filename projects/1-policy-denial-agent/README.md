policy-denial-agent/README.md
markdown
# Policy-Guided Claim Denial Agent (RAP)

A retrieval-augmented generation (RAG) system that interprets individual medical claims in real time and identifies Medicare-aligned denial justifications and appeal pathways — directly from CMS policy guidance.

---

![app screenshot](D:\GitHub\rag-portfolio\projects\1-policy-denial-agent\images\denial_analyzer.PNG)


##  Overview

This app serves as the foundation of a broader healthcare-focused RAG portfolio. It couples fast semantic retrieval over indexed CMS coverage documents with a local or cloud-hosted LLM to generate natural-language rationale for claim denials and appeal logic.

**Key features:**
- Medicare-aligned claim interpretation
- Modifier-sensitive CPT + ICD mapping
- FAISS-based dense retrieval from CMS policy documents
- Streamlit UI for single or batch JSON claims
- Token-efficient prompt wrapping for fast LLM inference

---

##  Technologies Used

- **Streamlit** for rapid UI
- **FAISS** vector search (dense embeddings)
- **SentenceTransformers / HuggingFace** for query embedding
- **Custom prompt engineering** to guide compliant reasoning
- Compatible with local inference (e.g., Mistral, LLaMA) or hosted APIs

---

##  Getting Started

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
Paste a claim directly or upload a .json/.jsonl file with multiple claims.

 Sample Input
json
{
  "cpt_code": "99213",
  "diagnosis": "E11.9",
  "modifiers": ["25"],
  "payer": "Medicare"
}
 Project Layout
1-policy-denial-agent/
├── streamlit_app.py         # Main interface
├── requirements.txt
├── src/
│   ├── analyzer.py          # Core RAG pipeline
│   ├── retriever.py         # FAISS-based search logic
│   ├── prompts/             # Prompt templates
│   └── utils.py
├── data/
│   ├── cms_index.faiss
│   └── cms_docs.pkl
├── assets/                  # (optional: icons, policy PDFs, etc.)
└── README.md
 Performance Notes
Embedding runs only on the query — the CMS documents are pre-indexed

Supports small models (e.g. quantized Mistral) for efficient local inference

Progress bar and streamed results in Streamlit enhance UX responsiveness

Designed for modular expansion into claim auditing, payer fusion, or risk stratification

 Privacy & Fair Use
No PHI is processed. Example inputs are synthetic or anonymized. This project is intended for educational, research, and demo use only.

 Author
Developed by Udonsi Contact: [udonsik@yahoo.com]


