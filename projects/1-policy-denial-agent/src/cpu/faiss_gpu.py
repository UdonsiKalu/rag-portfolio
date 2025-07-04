import os
import sys
import faiss
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaLLM
from langchain_community.cache import SQLiteCache

# Make sure to install these before running:
# pip install langchain-ollama faiss-cpu langchain_nomic rank_bm25 pypdf

# =================================
# STRICT CPU-ONLY CONFIGURATION
# =================================
# os.environ["OLLAMA_NO_CUDA"] = "1"        # Force CPU for Ollama if needed
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA globally
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Suppress certain library warnings

print("Running in GPU Mode")
print(f"FAISS version: {faiss.__version__}")
print(f"Detected GPUs: {faiss.get_num_gpus()}")

sys.modules["faiss"] = faiss  # Patch for langchain compatibility

# =================================
# CONSTANTS
# =================================
#CMS_MANUAL_PATH = "/media/udonsi-kalu/New Volume/denials/denials/manuals/"
#FAISS_INDEX_PATH = "cms_manuals_faiss_index"
CMS_MANUAL_PATH = "/media/udonsi-kalu/New Volume/GitHub/rag-portfolio/projects/1-policy-denial-agent/data/cms_manuals_rag/"
FAISS_INDEX_PATH = "/media/udonsi-kalu/New Volume/GitHub/rag-portfolio/projects/1-policy-denial-agent/data/cms_manuals_faiss_index"
PRIORITY_CHAPTERS = {
    "Chapter 1": "General Billing Requirements",
    "Chapter 12": "Physicians Nonphysician Practitioners",
    "Chapter 23": "Fee Schedule Administration",
    "Chapter 30": "Financial Liability Protections"
}

# =================================
# DOCUMENT PROCESSING
# =================================
def load_and_chunk_manuals():
    if not os.path.exists(CMS_MANUAL_PATH):
        raise FileNotFoundError(f"CMS manuals directory not found at {CMS_MANUAL_PATH}")

    loaders = []
    for chap, title in PRIORITY_CHAPTERS.items():
        filename = f"{chap} - {title}.pdf"
        path = os.path.join(CMS_MANUAL_PATH, filename)
        if os.path.exists(path):
            loaders.append(PyPDFLoader(path))
        else:
            print(f"Warning: {filename} not found")

    if not loaders:
        raise ValueError("No CMS manual PDFs found")

    # Configure splitters with fixed regex
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "Chapter"), ("##", "Section"), ("###", "Subsection"), ("####", "PolicyNumber")
    ])
    content_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,  # Reduced for CPU memory
        chunk_overlap=2000,
        separators=["\n\n", "\n", r"(?<=\. )", " "]
    )

    chunks = []
    for loader in loaders:
        try:
            docs = loader.load()
            for doc in docs:
                headers = header_splitter.split_text(doc.page_content)
                chunks.extend(content_splitter.split_documents(headers))
        except Exception as e:
            print(f"Error loading {loader.file_path}: {e}")

    return chunks

# =================================
# VECTOR STORE & RETRIEVAL
# =================================
def create_or_load_vector_store(chunks):
    # Embeddings on GPU device if available
    embedder = NomicEmbeddings(
        model="nomic-embed-text-v1",
        inference_mode="local",
        device="cuda"  # Use "cpu" if GPU is not available
    )

    if os.path.exists(FAISS_INDEX_PATH):
        print(f"Loading existing FAISS index from {FAISS_INDEX_PATH} and moving to GPU")
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings=embedder,
            allow_dangerous_deserialization=True
        )
        # Move to GPU if available
        if faiss.get_num_gpus() > 0:
            vectorstore.index = faiss.index_cpu_to_all_gpus(vectorstore.index)
    else:
        print("Creating new FAISS index (GPU if available)...")
        print(f"Indexing {len(chunks)} processed chunks into FAISS...")
        vectorstore = FAISS.from_documents(chunks, embedder)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print("FAISS index created and saved successfully.")

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 2

    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 25}
    )

    ensemble = EnsembleRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[0.4, 0.6]
    )

    return {"vectorstore": vectorstore, "retriever": ensemble}

# =================================
# LLM CHAIN SETUP
# =================================
def setup_llm_chain(retriever):
    prompt = ChatPromptTemplate.from_template("""
    As a Medicare billing expert, analyze this claim for denial risks using ONLY the provided CMS rules.

    You MUST respond with VALID JSON containing these EXACT fields:
    {{{{ 
        "risk_score": (an integer between 0 and 100, categorized as follows:
            - 0-30: Low Risk,
            - 31-70: Moderate Risk,
            - 71-100: High Risk),
        "potential_denial_reasons": [an array of strings describing ALL possible reasons based on CMS rules],
        "required_corrections": [an array of actionable steps necessary to prevent denial],
        "appeal_excerpts": [an array of strings, where each string is a complete CMS policy quote.
                             If there are no valid quotes, the array must contain a single element: "No relevant policy found."]
    }}}}

    **IMPORTANT:** Your response MUST start with {{ and end with }} — no extra text.
    **IMPORTANT:** Your entire response must be valid JSON. Every element, including all policy quotes, must be enclosed in quotation marks with no additional text.

    <CMS_RULES>
    {context}
    </CMS_RULES>

    Claim Details:
    - CPT: {cpt_code}
    - Diagnosis: {diagnosis} (Ensure diagnosis is relevant to the CPT code)
    - Modifiers: {modifiers} (Check whether modifiers justify billing separately)
    - Payer: {payer} (Consider payer-specific rules if applicable)

    **Response Format Must Be Valid JSON Only:**
    """)

    llm = OllamaLLM(
        model="llama3:8b-instruct-q4_0",
        temperature=0,
        num_ctx=4096,
        num_thread=4,
        top_k=30
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

# =================================
# MAIN ANALYZER CLASS
# =================================
class CMSDenialAnalyzer:
    def __init__(self):
        print("Initializing CMS Denial Analyzer (GPU mode if available)...")
        
        print("1. Loading and chunking manuals...")
        chunks = load_and_chunk_manuals()
        print(f"   Processed {len(chunks)} policy chunks")

        print("2. Building retrieval system...")
        self.retrieval = create_or_load_vector_store(chunks)

        print("3. Setting up LLM chain...")
        self.chain = setup_llm_chain(self.retrieval["retriever"])
        print("System ready\n" + "="*50)

    def analyze_claim(self, claim_data):
        query = self._generate_query(claim_data)
        
        print(f"\nAnalyzing claim: CPT {claim_data['cpt_code']}...")

        retrieved_docs = self.retrieval["retriever"].invoke(claim_data["cpt_code"])
        
        filtered_docs = [
            doc for doc in retrieved_docs if "modifier" in doc.page_content.lower() 
            or "billing" in doc.page_content.lower() or "e/m" in doc.page_content.lower()
        ]

        print("\n=== Retrieved CMS Context ===")
        for i, doc in enumerate(retrieved_docs):
            print(f"Document {i+1}:\n{doc.page_content}\n")

        # Robust LLM call with JSON parsing
        response = self.chain.invoke({
            "input": query,
            "context": filtered_docs,
            "cpt_code": claim_data["cpt_code"],
            "diagnosis": claim_data["diagnosis"],
            "modifiers": claim_data.get("modifiers", []),
            "payer": claim_data.get("payer", "Medicare")
        })

        print("Raw LLM chain response:", response)

        raw_answer = response.get("answer")
        if not raw_answer:
            raise RuntimeError("LLM response missing 'answer' field or empty")

        try:
            parsed_answer = json.loads(raw_answer)
        except json.JSONDecodeError as e:
            print("Error parsing LLM JSON output:", e)
            print("Raw LLM output:", raw_answer)
            raise RuntimeError("Failed to parse LLM output as JSON")

        return parsed_answer

    def _generate_query(self, claim):
        if claim["cpt_code"].startswith("99"):
            return f"Evaluation and Management coding rules for CPT {claim['cpt_code']}"
        elif any(m in claim.get("modifiers", []) for m in ["-59", "-25"]):
            return f"Modifier {claim['modifiers'][0]} documentation requirements"
        else:
            return f"Medicare coverage criteria for {claim['cpt_code']} with {claim['diagnosis']}"

# =================================
# ENTRY POINT
# =================================
if __name__ == "__main__":
    try:
        analyzer = CMSDenialAnalyzer()

        test_claim = {
            "cpt_code": "99214",
            "diagnosis": "Z79.899",
            "modifiers": [],
            "payer": "Medicare"
        }

        result = analyzer.analyze_claim(test_claim)

        print("\n=== Analysis Results ===")
        print(f"Risk Score: {result.get('risk_score', 'N/A')}%")
        print("\nPotential Denial Reasons:")
        print("- " + "\n- ".join(result.get("potential_denial_reasons", [])))
        print("\nRecommended Actions:")
        print("- " + "\n- ".join(result.get("required_corrections", [])))
        
        print("\nRelevant CMS Policies:")
        for i, doc in enumerate(result.get("source_documents", [])[:2], 1):
            print(f"\n[{i}] Page {doc.metadata.get('page', '?')}:")
            print(doc.page_content)

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify Ollama is running: `ollama serve`")
        print("2. Check required PDFs exist in:", CMS_MANUAL_PATH)
        print("3. Try these commands if issues persist:")
        print("   - `ollama pull llama3:8b-instruct-q4_0`")
        print("   - `pip install --upgrade faiss-cpu langchain-nomic langchain-ollama rank_bm25 pypdf`")
