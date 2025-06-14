# Global Populism Database (GPD) – LLM Replication

This repository recreates Team Populism’s Global Populism Database (GPD) workflow with modern Large Language Models (LLMs). We fully reproduce the original holistic-grading training process—complete with theoretical primers, detailed rubric, and anchor speeches—and then layer a Retrieval-Augmented Generation (RAG) loop on top. The system embeds the anchor speeches, retrieves the closest examples for any new text, and prompts an LLM to deliver a 0.0–2.0 populism score with supporting quotes and reasoning.


---

## What This Project Does

1. **Embed & Index** training speeches using a Sentence‑Transformer model and store them in a FAISS cosine‑similarity index.
2. **Retrieve** the *k* nearest, already‑scored speeches for any new speech (Retrieval‑Augmented Generation, RAG).
3. **Prompt** an LLM with (i) theoretical training, (ii) the populism rubric, and (iii) the retrieved anchors to produce:
   * a 0.0 – 2.0 populism score (rounded down to the tenth),
   * supporting quotes, and
   * category‑by‑category reasoning.

---

## Repository Contents

| File | Role |
|------|------|
| `build_index_v3.py` | Python helper that builds / queries the FAISS index |
| `RAG_GPD.R` | R driver: constructs the full prompt and calls an Ollama‑hosted LLM |
| `Speech_*.md` | Training speeches with YAML front‑matter and existing scores |
| `rubrics-definition.yaml` | Structured YAML version of the populism rubric |

---

## Quick Start

1. **Download** all project files—`build_index_v3.py`, `RAG_GPD.R`, and the `Speech_*.md` (plus any accompanying YAML) files—into the **same folder**.
2. **Open R** in your favourite IDE (e.g., RStudio, VS Code) and open `RAG_GPD.R`.
3. **Paste the speech** you want to score into the line:
   ```r
   new_speech <- "..."
   ```
4. **Run the script.** It automatically handles the Python calls via *reticulate*, builds/queries the index, and prints the populism score with supporting quotes.

---

## License
MIT License.

## Contact
Eduardo Tamaki · eduardo@tamaki.ai


