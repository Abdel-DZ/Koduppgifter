# %% [markdown]
# 1. Skapa en chattbot som svarar på frågor utifrån något dokument som du själv väljer.

# %% [markdown]
# 1. 0 IMPORTER + API‑KEY

# %%
# ---------------------------------------------------------
# IMPORTER
# ---------------------------------------------------------
import os
import numpy as np
import polars as pl
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# API-KEY (GROQ)
# ---------------------------------------------------------
api_key = ""
client = Groq(api_key=api_key)


# %% [markdown]
# 1. 1 LÄS IN DOKUMENT

# %%
# ---------------------------------------------------------
# 1. LÄS IN DOKUMENT (TVÅ PDF-FILER)
# ---------------------------------------------------------
paths = [
    "6.1 AF TOJOS PLAST.pdf",
    "6.2.1 MARK BYGG RAMBESKRIVNNG.pdf"
]

text = ""
for p in paths:
    reader = PdfReader(p)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

print(f"Totala antal tecken: {len(text)}")


# %% [markdown]
# 1. 2 CHUNKING

# %%
# ---------------------------------------------------------
# 2. CHUNKING
# ---------------------------------------------------------
chunks = []
n = 1800
overlap = 300

for i in range(0, len(text), n - overlap):
    chunks.append(text[i:i + n])

print(f"Antal chunks: {len(chunks)}")


# %% [markdown]
# 1. 3 EMBEDDINGS

# %%
# ---------------------------------------------------------
# 3. EMBEDDINGS (LOCAL SENTENCE-TRANSFORMER)
# ---------------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embedding(text):
    return model.encode(text).tolist()

# Skapa embeddings för alla chunks
chunk_embeddings = [create_embedding(chunk) for chunk in chunks]
print(f"Embeddings skapade: {len(chunk_embeddings)}")


# %% [markdown]
# 1. 4 SEMANTISK SÖKNING

# %%
# ---------------------------------------------------------
# 4. SEMANTISK SÖKNING
# ---------------------------------------------------------
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def semantic_search(query, chunks, embeddings, k=5):
    query_embedding = create_embedding(query)
    scores = []

    for i, emb in enumerate(embeddings):
        score = cosine_similarity(query_embedding, emb)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in scores[:k]]
    return [chunks[i] for i in top_indices]


# %% [markdown]
# 1. 5 SYSTEM PROMPT

# %%
# ---------------------------------------------------------
# 5. SYSTEM PROMPT
# ---------------------------------------------------------
system_prompt = """Jag kommer ställa dig en fråga, och jag vill att du svarar
baserat bara på kontexten jag skickar med, och ingen annan information.
Om det inte finns nog med information i kontexten för att svara på frågan,
säg "Det vet jag inte". Försök inte att gissa.
Formulera dig enkelt och dela upp svaret i fina stycken."""


# %% [markdown]
# 1. 6 GENERERA USER PROMPT

# %%
# ---------------------------------------------------------
# 6. GENERERA USER PROMPT
# ---------------------------------------------------------
def generate_user_prompt(query):
    context = "\n".join(semantic_search(query, chunks, chunk_embeddings))
    return f"Frågan är: {query}\n\nHär är kontexten:\n{context}"


# %% [markdown]
# 1. 7 GENERERA SVAR

# %%
# ---------------------------------------------------------
# 7. GENERERA SVAR (GROQ CHAT)
# ---------------------------------------------------------
def generate_response(system_prompt, user_message, model="llama-3.1-8b-instant"):
    full_prompt = generate_user_prompt(user_message)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )
    return resp.choices[0].message.content


# %% [markdown]
# 1. 8 CHATT‑LOOP

# %%
# ---------------------------------------------------------
# 8. CHATT-LOOP
# ---------------------------------------------------------
print("\n*** Groq RAG chat ***")
print("Skriv <Exit> för att avsluta.\n")

while True:
    prompt = input("User: ")
    if prompt.lower() == "exit":
        break
    else:
        answer = generate_response(system_prompt, prompt)
        print("Robot:", answer, "\n")


# %% [markdown]
# 1. 9 VECTOR STORE

# %%
# ---------------------------------------------------------
# 9. VECTOR STORE (för att spara embeddings)
# ---------------------------------------------------------
class VectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []
    
    def add_item(self, text, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})
    
    def save(self, filename="embeddings.parquet"):
        df = pl.DataFrame({
            "vectors": self.vectors,
            "texts": self.texts,
            "metadata": self.metadata
        })
        df.write_parquet(filename)
        print(f"Embeddings sparade till {filename}")
    
    def load(self, filename="embeddings.parquet"):
        df = pl.read_parquet(filename)
        self.vectors = df["vectors"].to_list()
        self.texts = df["texts"].to_list()
        self.metadata = df["metadata"].to_list()
        print(f"Embeddings laddade från {filename}")

vector_store = VectorStore()
for i, chunk in enumerate(chunks):
    vector_store.add_item(chunk, chunk_embeddings[i], {"index": i})
vector_store.save()


# %% [markdown]
# 1. 10 EVALUERING

# %%
# ---------------------------------------------------------
# 10. EVALUERING
# ---------------------------------------------------------
validation_data = [
    {
        "question": "Vad handlar dokumenten om?",
        "ideal_answer": "Dokument om byggprojekt och plastmaterial."
    }
]

evaluation_system_prompt = """Du är ett utvärderingssystem.
Poäng: 1 = korrekt, 0.5 = delvis, 0 = fel.
Motivera kort."""

query = validation_data[0]["question"]
ai_answer = generate_response(system_prompt, query)

evaluation_prompt = f"""Fråga: {query}
AI-svar: {ai_answer}
Ideal-svar: {validation_data[0]['ideal_answer']}"""

evaluation = generate_response(evaluation_system_prompt, evaluation_prompt)
print("\nUtvärdering:")
print(evaluation)
