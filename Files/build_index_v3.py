# build_index_v2.py – Commented version
# -------------------------------------------------------------
# This module builds a simple Retrieval‑Augmented Generation (RAG)
# index for a collection of Markdown files that contain speeches.
# Each speech file is assumed to have a YAML front‑matter block
# with metadata (e.g. speaker name, date) followed by sections like
# "## Full Text", "### Thought Process", grading information, and
# rubric evidence.
#
# The core workflow is:
#   1. Load every speech matching a glob pattern (default: "Speech*.md").
#   2. Parse out the metadata, full text, thought‑process notes, grade,
#      and rubric evidence from each file.
#   3. Encode the full text of each speech with a sentence‑transformer
#      model; store the embeddings in a FAISS index (inner‑product /
#      cosine similarity variant).
#   4. Provide a retrieval method that, given a query string, returns
#      the *k* most similar speeches together with their metadata and
#      similarity score.
#   5. Offer a convenience method that formats the retrieved speeches
#      as training examples that can be fed into downstream LLM prompts.
#
# NOTE: The code purposefully avoids any writing or updating of the
# speech files themselves; it only reads and parses them.
# -------------------------------------------------------------

import re  # Regular‑expression utilities
import json  # (Imported but currently unused – could be handy for future serialization)
from glob import glob  # File‑pattern matching, e.g. "Speech*.md"

import numpy as np  # Numerical arrays for stacking embeddings
from sentence_transformers import SentenceTransformer  # Embedding model
import faiss  # Facebook AI Similarity Search – fast vector index


class PopulismRAG:
    """Minimal RAG helper tailored to analysing political speeches.

    The class wraps five high‑level responsibilities:

    * **Data Ingestion**   – `load_speeches` reads Markdown files and
      extracts useful structured fields.
    * **Index Construction** – `build_index` builds a FAISS cosine‑similarity
      index (implemented with `IndexFlatIP` over *L2‑normalised* vectors).
    * **Retrieval**        – `retrieve_similar_speeches` performs k‑NN
      search to surface the most relevant speeches for a given query.
    * **Formatting**       – `format_training_examples` pretty‑prints the
      retrieved speeches as prompt‑ready examples (thought process, grade,
      and abbreviated rubric evidence).

    The default embedding model is **nomic‑ai/nomic‑embed‑text‑v1.5**
    loaded from HuggingFace with `trust_remote_code=True`.
    """

    def __init__(self):
        """Instantiate the embedding model and prepare empty containers."""
        # Load the model once. Using `normalize_embeddings=True` later so
        # that cosine similarity equals inner product.
        self.model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )

        # A list that will hold dictionaries, one per speech.
        self.speeches = []

        # This will become a FAISS index after `build_index` is called.
        self.index = None

    # ------------------------------------------------------------------
    # Pre‑processing helpers
    # ------------------------------------------------------------------

    def load_speeches(self, pattern: str = "Speech*.md"):
        """Read Markdown files matching *pattern* and populate `self.speeches`.

        Parameters
        ----------
        pattern : str, optional
            A glob pattern (wildcards allowed). Defaults to "Speech*.md".
        """
        speech_files = glob(pattern)  # e.g. ["Speech001.md", "Speech002.md", ...]

        for fname in speech_files:
            # ------------------
            # 1) Read file text
            # ------------------
            with open(fname, encoding="utf-8") as f:
                content = f.read()

            # ------------------------------
            # 2) Extract YAML front‑matter
            # ------------------------------
            yaml_match = re.search(r"(?s)^---\n(.*?)\n---", content)
            metadata = {}
            if yaml_match:
                # Parse each "key: value" line inside front‑matter.
                for line in yaml_match.group(1).split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip().strip('"')

            # -----------------------------------------
            # 3) Extract individual content sections
            # -----------------------------------------
            speech_data = {
                "file": fname,
                "metadata": metadata,
                "full_text": self._extract_section(content, "## Full Text"),
                "thought_process": self._extract_section(
                    content, "### Thought Process"
                ),
                "grade": self._extract_grade(content),
                "rubric_evidence": self._extract_rubric_evidence(content),
            }

            # Append to master list
            self.speeches.append(speech_data)

    # -----------------------
    # Content‑parsing helpers
    # -----------------------

    def _extract_section(self, content: str, header: str) -> str:
        """Return the text that follows *header* up to the next header or EOF."""
        pattern = rf"{re.escape(header)}\n(.*?)(?=\n##|\n###|\Z)"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_grade(self, content: str):
        """Find a numerical grade like `Score: 18.5` and return it as *float*."""
        grade_match = re.search(r"Score:\s*(\d+\.?\d*)", content)
        return float(grade_match.group(1)) if grade_match else None

    def _extract_rubric_evidence(self, content: str):
        """Collect quoted evidence under each rubric subsection.

        Returns
        -------
        dict
            Mapping from rubric name (string) to evidence (string).
        """
        evidence = {}
        # Regex captures: (1) rubric name, (2) evidence block (stops before next rubric/header)
        rubric_pattern = (
            r"#### Rubric \d+: (.*?)\n.*?\*\*Evidence\*\*:\n> (.*?)(?=\n####|\n###|\Z)"
        )
        matches = re.findall(rubric_pattern, content, re.DOTALL)
        for rubric_name, quotes in matches:
            evidence[rubric_name] = quotes.strip()
        return evidence

    # -----------------------
    # Embedding & indexing
    # -----------------------

    def build_index(self):
        """Encode *full text* of each speech and create a FAISS index."""
        embeddings = []  # Collect each speech vector here

        for speech in self.speeches:
            # Concatenate a static instruction prefix – improves retrieval quality
            text = speech["full_text"]
            emb = self.model.encode(
                f"search_document: {text}", normalize_embeddings=True
            )
            embeddings.append(emb)

        # Stack into a (n_samples × dim) matrix
        embedding_matrix = np.vstack(embeddings)
        d = embedding_matrix.shape[1]  # Embedding dimensionality

        # Inner‑product index (fast, no training step) – because vectors
        # are L2‑normalised, IP is equivalent to cosine similarity.
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embedding_matrix)

    # ---------------
    #  k‑NN retrieval
    # ---------------

    def retrieve_similar_speeches(self, query_text: str, top_k: int = 3):
        """Return the *top_k* most similar speeches to *query_text*.

        The function embeds the query, runs a FAISS search, and enriches
        the returned speech dictionaries with a `similarity_score` field
        (float, cosine similarity in [‑1, 1]).
        """
        # Encode query with same prefix used during indexing
        query_emb = self.model.encode(
            f"search_query: {query_text}", normalize_embeddings=True
        )

        # Perform k‑NN search (vectors already on GPU/CPU according to model)
        D, I = self.index.search(np.array([query_emb]), top_k)

        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.speeches):
                # Copy to avoid mutating original dict
                speech = self.speeches[idx].copy()
                speech["similarity_score"] = float(score)
                results.append(speech)

        return results

    # -----------------------------
    #  Pretty‑printing for LLM use
    # -----------------------------

    def format_training_examples(self, speeches):
        """Render retrieved speeches in a prompt‑friendly textual format."""
        examples = []

        for speech in speeches:
            example = f"""
=== Training Example: {speech['metadata'].get('speaker', 'Unknown')} (Score: {speech['grade']}) ===

THOUGHT PROCESS:
{speech['thought_process']}

GRADE: {speech['grade']}

KEY EVIDENCE:
"""
            # Append each rubric and first 200 characters of evidence
            for rubric, evidence in speech["rubric_evidence"].items():
                if evidence:
                    example += f"\n{rubric}:\n{evidence[:200]}...\n"

            examples.append(example)

        # Join with blank lines; keeps prompt clean
        return "\n\n".join(examples)


# ----------------------------------------------------------------------
# Example usage – these lines run immediately on import. If you would
# rather make this import‑safe, you could wrap them in `if __name__ ==
# '__main__':`, but per the original specification we leave them as‑is.
# ----------------------------------------------------------------------

# Initialize RAG helper
# rag = PopulismRAG()

# Load all speech Markdown files in the current directory matching pattern
# rag.load_speeches()

# Build vector index for downstream retrievals
# rag.build_index()
