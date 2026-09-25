!pip install transformers faiss-cpu datasets

!pip install -q sentence-transformers faiss-cpu

import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

"""## FIR Dataset"""

import pandas as pd

df = pd.read_excel(
    "Synthetic data.xlsx"
)

print(df.head())

print(df.columns)


# =========================================================
# LOAD BGE-M3 MODEL
# =========================================================

model = SentenceTransformer(
    "BAAI/bge-m3"
)

print("\nMODEL LOADED ")

# =========================================================
# PREPARE DOCUMENTS (WITH METADATA AUGMENTATION)
# =========================================================

documents = []
for idx, row in df.iterrows():
    # Dataset ke unique metadata columns ko text ke sath merge kar rahe hain
    metadata_prefix = f"[Category: {row['Crime_category']} | Keywords: {row['Trigger_words']} | Sections: {row['Applied_Sections']}] "

    # BGE-M3 ka rule: passage prefix lagana lazmi hai
    full_text = "passage: " + metadata_prefix + str(row["Text"])
    documents.append(full_text)

print(f"Prepared {len(documents)} augmented documents for embedding ")

# =========================================================
# CREATE EMBEDDINGS
# =========================================================

embeddings = model.encode(

    documents,

    batch_size=16,

    show_progress_bar=True,

    normalize_embeddings=True
)

embeddings = np.array(

    embeddings,

    dtype="float32"
)

print("\nEmbeddings Shape:", embeddings.shape)

# =========================================================
# CREATE FAISS INDEX
# =========================================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    embeddings
)

print("\nFAISS INDEX CREATED ")

# =========================================================
# SAVE EMBEDDINGS
# =========================================================

np.save(
    "embeddings.npy",
    embeddings
)

print("Embeddings Saved ")

# =========================================================
# SAVE FAISS INDEX
# =========================================================

faiss.write_index(
    index,
    "faiss_index.bin"
)

print("FAISS Index Saved ")

# =========================================================
# LOAD EMBEDDINGS
# =========================================================

embeddings = np.load(
    "embeddings.npy"
)

print("Embeddings Loaded ")

# =========================================================
# LOAD FAISS INDEX
# =========================================================

index = faiss.read_index(
    "faiss_index.bin"
)

print("FAISS Index Loaded ")

# =========================================================

query = """
بحوالہ ریٹ نمبر 45 بوقت 0440 بجے مورخہ 18/01/2026 روز نامچہ عام کے تحت روانہ شدہ من [اہلکار] ہمراہ ملازمان [اہلکار 1، اہلکار 2] بسواری موبائل سرکاری [گاڑی نمبر] مصروف گشت و انسداد جرائم تھا کہ جب [مقام وقوعہ] پہنچا تو دو [موٹرسائیکلوں] پر سوار چار مسلح [ملزمان] کو مشکوک حالت میں دیکھا۔ پولیس کو دیکھ کر ملزمان نے پولیس پارٹی پر جان سے مارنے کی نیت سے فائرنگ شروع کر دی، جس پر پولیس نے بھی دفاعی فائرنگ کی۔ فائرنگ کے تبادلے کے بعد دو [ملزمان] (1) [ملزم 1] اور (2) [ملزم 2] زخمی حالت میں گرفتار کر لیے گئے جبکہ ان کے دو دیگر ساتھی اندھیرے کا فائدہ اٹھاتے ہوئے فرار ہو گئے۔ گرفتار ملزمان کے قبضے سے دو عدد [اسلحہ] معہ لوڈ میگزین اور ایک [موٹرسائیکل] [گاڑی نمبر] میکر سپر اسٹار برآمد ہوئی جس کے کاغذات وہ پیش نہ کر سکے۔ موقع سے چلے ہوئے خول بھی برآمد کیے گئے۔ زخمی ملزمان کو علاج کیلئے [ہسپتال] روانہ کیا گیا۔ ملزمان کا یہ فعل کار سرکار میں مداخلت اور پولیس پر فائرنگ کے زمرے میں آتا ہے۔

"""

# =========================================================
# QUERY EMBEDDING
# =========================================================

query_embedding = model.encode(

    ["query: " + query],

    normalize_embeddings=True
)

query_embedding = np.array(

    query_embedding,

    dtype="float32"
)

# =========================================================
# SEARCH TOP 5
# =========================================================

D, I = index.search(

    query_embedding,

    k=5
)

# =========================================================
# RETRIEVE DOCUMENTS
# =========================================================

retrieved_docs = [

    {

        "text":
        df.iloc[idx]["Text"],

        "sections":
        df.iloc[idx]["Applied_Sections"],

        "trigger_words":
        df.iloc[idx]["Trigger_words"],

        "category":
        df.iloc[idx]["Crime_category"],

        "similarity":
        D[0][i]
    }

    for i, idx in enumerate(I[0])
]

# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\nTOP 5 SIMILAR FIRs \n")

for i, doc in enumerate(retrieved_docs, 1):

    print("="*80)

    print(f"RESULT {i}")

    print("\nFIR:")
    print(doc["text"])

    print("\nTrigger Words:")
    print(doc["trigger_words"])

    print("\nApplied Sections:")
    print(doc["sections"])

    print("\nCrime Category:")
    print(doc["category"])

    print("\nSimilarity Score:")
    print(round(float(doc["similarity"]),4))

    print("="*80)

"""#**Generation**"""

!pip install google-generativeai

import google.generativeai as genai

from google.colab import userdata

#GOOGLE_API_KEY=userdata.get('GOOGLE_API_KEY')

#genai.configure(api_key=GOOGLE_API_KEY)

genai.configure(api_key="AIzaSyAfBf8IHScuYO1VUPblbx3WtZgkWBCkkYc")

genai.configure(api_key="AIzaSyAfBf8IHScuYO1VUPblbx3WtZgkWBCkkYc")

for m in genai.list_models():
  # Print models that support the generateContent method
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

#llm_model = "models/gemini-2.5-flash"
llm_model = "models/gemma-4-31b-it"

genai_model = genai.GenerativeModel(llm_model)

#context_docs = [doc["text"] for doc in retrieved_docs]
# ============================================
# BUILD CONTEXT FROM RETRIEVED FIRs
# ============================================

context = ""

for i, doc in enumerate(retrieved_docs, 1):

    context += f"""

Similar FIR {i}

FIR:
{doc['text'][:700]}

Applied Sections:
{doc['sections']}

Crime Category:
{doc['category']}

-----------------------------------
"""

#prompt = f"Query: {query}\nContext: {retrieved_docs}\nAnswer:"
#print(prompt)

# ============================================
# FINAL PROMPT
# ============================================

prompt = f"""

آپ ایک ماہر پاکستانی قانونی معاون ہیں۔

Similar FIRs صرف معاون معلومات کیلئے ہیں۔
ان کی دفعات کو blindly copy نہ کریں۔

موجودہ FIR کا الگ قانونی تجزیہ کریں۔

━━━━━━━━━━━━━━━━━━
موجودہ FIR:
━━━━━━━━━━━━━━━━━━

{query}

━━━━━━━━━━━━━━━━━━
Similar FIRs:
━━━━━━━━━━━━━━━━━━

{context}

━━━━━━━━━━━━━━━━━━
صرف اسی فارمیٹ میں جواب دیں:
━━━━━━━━━━━━━━━━━━

متعلقہ دفعات:
[صرف دفعات]

ٹرگر الفاظ:
[اہم الفاظ]

جرم کی قسم:
[category]

قانونی جواز:
[مختصر وجہ]

"""

# ============================================
# PRINT PROMPT
# ============================================

print(prompt)

#response = genai_model.generate_content(prompt)
#print(response.text)
# ============================================
# GENERATE RESPONSE
# ============================================

response = genai_model.generate_content(prompt)

# ============================================
# OUTPUT
# ============================================

print("\nLLM RESPONSE\n")

print(response.text)

#prompt = f"Query: {query}\nContext: {retrieved_docs}\nAnswer in a sentence:"
#response = genai_model.generate_content(prompt)
#print(response.text)
# ============================================
# LEGAL RAG PROMPT
# ============================================

prompt = f"""

You are an expert Pakistani legal assistant.

Analyze the FIR complaint carefully
using the retrieved similar FIRs.

==================================================

QUERY FIR:

{query}

==================================================

RETRIEVED SIMILAR FIRs:

{retrieved_docs}

==================================================

Answer ONLY in this format:

Relevant Sections:
<sections>

Trigger Words:
<trigger words>

Crime Category:
<category>

Legal Reasoning:
<reasoning>

"""

# ============================================
# GENERATE RESPONSE
# ============================================

response = genai_model.generate_content(prompt)

# ============================================
# OUTPUT
# ============================================

print("\nLLM RESPONSE\n")

print(response.text)

#using a different prompt (an example of QA)
#prompt = f"Query: Describe about the Eifel Tower.\nContext: {retrieved_docs}\nAnswer in a sentence:"
#response = genai_model.generate_content(prompt)
#print(response.text)
# ============================================
# ANOTHER TEST QUERY
# ============================================

query2 = """
دفتر جاتے ہوئے شہری سے اسلحہ کے زور پر
موٹر سائیکل، موبائل اور نقدی چھین لی گئی۔
"""

prompt2 = f"""

You are an expert Pakistani legal assistant.

Analyze the FIR complaint carefully
using the retrieved similar FIRs.

==================================================

QUERY FIR:

{query2}

==================================================

RETRIEVED SIMILAR FIRs:

{retrieved_docs}

==================================================

Answer ONLY in this format:

Relevant Sections:
<sections>

Trigger Words:
<trigger words>

Crime Category:
<category>

Legal Reasoning:
<reasoning>

"""

response2 = genai_model.generate_content(prompt2)

print("\nSECOND QUERY RESPONSE\n")

print(response2.text)

"""*BAAI*/bge-m3 - Testing

"""

# =========================================================
# INSTALL
# =========================================================

!pip install -q sentence-transformers faiss-cpu openpyxl

# =========================================================
# LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import faiss
import re

from sentence_transformers import SentenceTransformer

# =========================================================
# LOAD TEST FILE
# =========================================================

test_df = pd.read_csv(
    "200split/FIR_Test.csv"
)

print("Total Test FIRs:", len(test_df))

# =========================================================
# LOAD MAIN DATABASE
# =========================================================

df = pd.read_excel(
    "Synthetic_data.xlsx"
)

print("Total FIR Database:", len(df))

# =========================================================
# LOAD MODEL FROM HUGGINGFACE
# =========================================================

embedding_model = SentenceTransformer(
    "BAAI/bge-m3"
)

print("Model Loaded ")

# =========================================================
# LOAD SAVED EMBEDDINGS
# =========================================================

embeddings = np.load(
    "embeddings.npy"
).astype("float32")

print("Saved Embeddings Loaded ")

# =========================================================
# LOAD SAVED FAISS INDEX
# =========================================================

index = faiss.read_index(
    "faiss_index.bin"
)

print("Saved FAISS Index Loaded ")

# =========================================================
# HELPER FUNCTION
# =========================================================

def extract_sections(text):

    sections = re.findall(
        r'\d+[A-Z\-]*',
        str(text)
    )

    return sorted(list(set(sections)))

# =========================================================
# RESULTS STORAGE
# =========================================================

results = []

# =========================================================
# TESTING LOOP
# =========================================================

for idx, row in test_df.iterrows():

    query = str(row["Text"])

    actual_sections = str(
        row["Applied_Sections"]
    )

    # =====================================================
    # QUERY EMBEDDING ONLY
    # =====================================================

    query_embedding = embedding_model.encode(

        ["query: " + query],

        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    # =====================================================
    # SEARCH TOP 3
    # =====================================================

    D, I = index.search(
        query_embedding,
        k=3
    )

    # =====================================================
    # RETRIEVE SECTIONS
    # =====================================================

    retrieved_sections = []

    for doc_idx in I[0]:

        retrieved_sections.append(

            str(
                df.iloc[doc_idx]["Applied_Sections"]
            )
        )

    combined_sections = " / ".join(
        retrieved_sections
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    predicted_sections = extract_sections(
        combined_sections
    )

    actual_clean = extract_sections(
        actual_sections
    )

    exact_match = int(

        set(predicted_sections)

        ==

        set(actual_clean)
    )

    # =====================================================
    # SAVE RESULT
    # =====================================================

    results.append({

        "Query_FIR":
        query,

        "Actual_Sections":
        actual_sections,

        "Predicted_Sections":
        ", ".join(predicted_sections),

        "Retrieved_Sections":
        combined_sections,

        "Similarity_1":
        round(float(D[0][0]),4),

        "Similarity_2":
        round(float(D[0][1]),4),

        "Similarity_3":
        round(float(D[0][2]),4),

        "Exact_Match":
        exact_match
    })

    print(f"Processed: {idx+1}/{len(test_df)}")

# =========================================================
# RESULTS
# =========================================================

results_df = pd.DataFrame(results)

accuracy = results_df[
    "Exact_Match"
].mean()

print("\n===================================")
print("RAG EVALUATION COMPLETE ")
print("===================================")

print(f"\nExact Match Accuracy: {accuracy:.4f}")

# =========================================================
# SAVE RESULTS
# =========================================================

save_path = "rag_test_results.xlsx"

results_df.to_excel(
    save_path,
    index=False
)

print(f"\nResults Saved At:\n{save_path}")

import pandas as pd

df = pd.read_excel("rag_test_results.xlsx")

df_75 = df.head(75)

total = len(df_75)
correct = df_75["Exact_Match"].sum()
accuracy = (correct / total) * 100

print("="*50)
print(f"Total Test Cases : {total}")
print(f"Correct Matches  : {correct}")
print(f"Wrong Matches    : {total - correct}")
print(f"Accuracy         : {accuracy:.2f}%")
print("="*50)

wrong_cases = df_75[df_75["Exact_Match"] == 0]

print("\nWRONG PREDICTIONS:\n")

for i, row in wrong_cases.iterrows():
    print("="*80)
    print(f"Query: {row['Query_FIR'][:300]}...")
    print(f"Actual Section     : {row['Actual_Sections']}")
    print(f"Predicted Section  : {row['Predicted_Sections']}")
    print(f"Retrieved Sections : {row['Retrieved_Sections']}")
    print(f"Similarity Scores  : {row['Similarity_1']}, "
          f"{row['Similarity_2']}, {row['Similarity_3']}")
    print("="*80)

# =========================================================
# INSTALL
# =========================================================

!pip install -q sentence-transformers faiss-cpu openpyxl

# =========================================================
# LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer

# =========================================================
# LOAD TEST FILE
# =========================================================

test_df = pd.read_csv(
    "200split/FIR_Test.csv"
)

print("Total Test FIRs:", len(test_df))

# =========================================================
# LOAD MAIN DATABASE
# =========================================================

df = pd.read_excel(
    "Synthetic_data.xlsx"
)

print("Total FIR Database:", len(df))

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

embedding_model = SentenceTransformer(
    "BAAI/bge-m3"
)

print("Embedding Model Loaded ")

# =========================================================
# LOAD SAVED EMBEDDINGS
# =========================================================

embeddings = np.load(
    "embeddings.npy"
).astype("float32")

print("Saved Embeddings Loaded ")

# =========================================================
# LOAD SAVED FAISS INDEX
# =========================================================

index = faiss.read_index(
    "faiss_index.bin"
)

print("Saved FAISS Index Loaded ")

# =========================================================
# RESULTS STORAGE
# =========================================================

results = []

# =========================================================
# TESTING LOOP
# =========================================================

for idx, row in test_df.iterrows():

    query = str(row["Text"])

    actual_section = str(
        row["Applied_Sections"]
    )

    # =====================================================
    # QUERY EMBEDDING
    # =====================================================

    query_embedding = embedding_model.encode(

        ["query: " + query],

        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    # =====================================================
    # TOP-3 SEARCH
    # =====================================================

    D, I = index.search(
        query_embedding,
        k=3
    )

    # =====================================================
    # RETRIEVED RESULT 1
    # =====================================================

    fir1 = ""
    sec1 = ""
    sim1 = ""

    if len(I[0]) > 0:

        fir1 = str(
            df.iloc[I[0][0]]["Text"]
        )

        sec1 = str(
            df.iloc[I[0][0]]["Applied_Sections"]
        )

        sim1 = round(
            float(D[0][0]),
            4
        )

    # =====================================================
    # RETRIEVED RESULT 2
    # =====================================================

    fir2 = ""
    sec2 = ""
    sim2 = ""

    if len(I[0]) > 1:

        fir2 = str(
            df.iloc[I[0][1]]["Text"]
        )

        sec2 = str(
            df.iloc[I[0][1]]["Applied_Sections"]
        )

        sim2 = round(
            float(D[0][1]),
            4
        )

    # =====================================================
    # RETRIEVED RESULT 3
    # =====================================================

    fir3 = ""
    sec3 = ""
    sim3 = ""

    if len(I[0]) > 2:

        fir3 = str(
            df.iloc[I[0][2]]["Text"]
        )

        sec3 = str(
            df.iloc[I[0][2]]["Applied_Sections"]
        )

        sim3 = round(
            float(D[0][2]),
            4
        )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results.append({

        "Query_FIR":
        query,

        "Actual_Section":
        actual_section,

        # TOP-1
        "Retrieved_FIR_1":
        fir1,

        "Retrieved_Section_1":
        sec1,

        "Similarity_1":
        sim1,

        # TOP-2
        "Retrieved_FIR_2":
        fir2,

        "Retrieved_Section_2":
        sec2,

        "Similarity_2":
        sim2,

        # TOP-3
        "Retrieved_FIR_3":
        fir3,

        "Retrieved_Section_3":
        sec3,

        "Similarity_3":
        sim3
    })

    print(
        f"Processed {idx+1}/{len(test_df)}"
    )

# =========================================================
# CREATE DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)

# =========================================================
# SAVE RESULTS
# =========================================================

save_path = (
    ""
    "top3_retrieval_results.xlsx"
)

results_df.to_excel(
    save_path,
    index=False
)

# =========================================================
# DONE
# =========================================================

print("\n===================================")
print("TOP-3 RETRIEVAL COMPLETE ")
print("===================================")

print("\nSaved File:")

print(save_path)

print("\nTotal Records:")

print(len(results_df))

# =========================================================
# FINAL HYBRID RAG + LLM PIPELINE
# USING SAVED FAISS + SAVED EMBEDDINGS
# ONLY QUERY EMBEDDING WILL BE GENERATED
# =========================================================

import re
import time
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

embedding_model = SentenceTransformer(
    "BAAI/bge-m3"
)

print("Embedding Model Loaded ")

# =========================================================
# LOAD FAISS INDEX
# =========================================================

index = faiss.read_index(
    "faiss_index.bin"
)

print("FAISS Index Loaded ")

# =========================================================
# LOAD METADATA
# =========================================================

metadata_df = pd.read_excel(
    "Synthetic_data.xlsx"
)

metadata = metadata_df.to_dict("records")

print("Metadata Loaded ")

# =========================================================
# LOAD TEST FILE
# =========================================================

test_df = pd.read_csv(
    "200split/FIR_Test.csv"
)

# only 75 FIRs
test_df = test_df.head(75)

print("Test File Loaded ")
print("Total FIRs:", len(test_df))

# =========================================================
# NORMALIZE FUNCTION
# =========================================================

def normalize_section(sec):

    sec = str(sec).upper()

    sec = sec.replace("PPC", "")
    sec = sec.replace("CNSA", "")
    sec = sec.replace("ACT", "")

    sec = re.sub(r'[^0-9A-Z/(), ]', '', sec)

    sec = re.sub(r'\s+', ' ', sec)

    return sec.strip()

# =========================================================
# SAFE LLM FUNCTION
# =========================================================

def ask_llm(prompt, retries=3):

    for attempt in range(retries):

        try:

            response = genai_model.generate_content(
                prompt
            )

            return response.text.strip()

        except Exception as e:

            print(f"Retry {attempt+1}: {e}")

            time.sleep(5)

    return "FAILED"

# =========================================================
# MAIN LOOP
# =========================================================

results = []

for idx, row in test_df.iterrows():

    print(f"\nProcessing {idx+1}/75")

    # =====================================================
    # CURRENT FIR
    # =====================================================

    query = str(row["Text"])

    actual_section = normalize_section(
        row["Applied_Sections"]
    )

    # =====================================================
    # GENERATE QUERY EMBEDDING ONLY
    # =====================================================

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    # =====================================================
    # FAISS SEARCH
    # =====================================================

    D, I = index.search(
        query_embedding,
        5
    )

    # =====================================================
    # RETRIEVED RESULTS
    # =====================================================

    retrieved_sections = []

    retrieved_firs = []

    similarity_scores = []

    for score, idx2 in zip(D[0], I[0]):

        if idx2 < len(metadata):

            sec = str(
                metadata[idx2]["Applied_Sections"]
            )

            fir_text = str(
                metadata[idx2]["Text"]
            )

            retrieved_sections.append(sec)

            retrieved_firs.append(
                fir_text[:700]
            )

            similarity_scores.append(
                round(float(score), 4)
            )

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    retrieved_sections = list(
        dict.fromkeys(retrieved_sections)
    )

    # =====================================================
    # RAG TOP-1 PREDICTION
    # =====================================================

    rag_prediction = normalize_section(
        retrieved_sections[0]
    )

    # =====================================================
    # LLM PROMPT
    # =====================================================

    prompt = f"""
You are a senior Pakistani criminal law expert.

Task:
Analyze the FIR independently and determine the most relevant legal sections.

Important Rules:

1. Read the FIR carefully.
2. Perform legal reasoning yourself.
3. Retrieved FIRs are only supporting references.
4. Ignore retrieved sections if they are not legally relevant.
5. Prefer exact PPC, CNSA, SAA, TIP Act sections.
6. Return only sections supported by facts in the FIR.
7. Do not invent sections.
8. If retrieved cases conflict with FIR facts, ignore them.

==================================================
CURRENT FIR
===========

{query}

==================================================
TOP RETRIEVED SECTIONS
======================

{retrieved_sections}

==================================================
SIMILAR FIR REFERENCES
======================

{retrieved_firs}

==================================================

First identify:

* Trigger words
* Crime type
* Criminal act
* Weapon usage (if any)
* Property involved
* Victim details
* Relevant legal elements

Then determine applicable sections.

Return EXACTLY in the following format:

Relevant Sections:
[sections only]

Trigger Words:
[keywords only]

Crime Category:
[crime type]

Legal Reasoning:
[2-3 sentence explanation]
"""
    # =====================================================
    # LLM RESPONSE
    # =====================================================

    llm_response = ask_llm(prompt)

    # =====================================================
    # STORE RESULTS
    # =====================================================

    results.append({

        "Query_FIR": query,

        "Actual_Sections": actual_section,

        "RAG_Prediction": rag_prediction,

        "Retrieved_Sections":
        " | ".join(retrieved_sections),

        "Similarity_Scores":
        ", ".join([
            str(x)
            for x in similarity_scores
        ]),

        "LLM_Response": llm_response
    })

# =========================================================
# SAVE FINAL RESULTS
# =========================================================

results_df = pd.DataFrame(results)

save_path = "final_hybrid_rag_llm_results.xlsx"

results_df.to_excel(
    save_path,
    index=False
)

# =========================================================
# DONE
# =========================================================

print("\n======================================")
print("FINAL FILE SAVED ")
print("======================================")

print(save_path)

print("\nTotal Processed:", len(results_df))

import pandas as pd
import re

df = pd.read_excel("top3_retrieval_results.xlsx")

def extract_sections(text):
    if pd.isna(text) or str(text) == '0':
        return set()

    # 1. Clean string
    text = str(text).lower().replace(" ", "")

    # 2. Extract basic tokens
    tokens = re.findall(r'\d+[a-z]?|cnsa|saa|tip', text)

    # 3. Intelligent Expansion (Mapping)
    # Agar model ne '381a' diya to hum '381' (base) bhi add kar denge
    final_set = set()
    for t in tokens:
        final_set.add(t) # Original token
        # Agar token '381a' hai to uska base '381' bhi add kar do
        base_match = re.match(r'^(\d+)', t)
        if base_match:
            final_set.add(base_match.group(1))

    return final_set
def jaccard(a, b):
    # Ab 'a' aur 'b' dono single values hain, series nahi
    if pd.isna(b) or str(b) == '0':
        return 0

    A = extract_sections(a)
    B = extract_sections(b)

    union = A | B
    if len(union) == 0:
        return 0
    return round(len(A & B) / len(union), 4)

# --- SAHI TARIQA: axis=1 ka istemal ---
for i in [1, 2, 3]:
    col_name = f"Retrieved_Section_{i}"
    if col_name in df.columns:
        # Row-by-row apply karein
        df[f"Jaccard_{i}"] = df.apply(
            lambda row: jaccard(row["Actual_Section"], row[col_name]),
            axis=1
        )
    else:
        df[f"Jaccard_{i}"] = 0

# Best Jaccard calculation
df["Best_Jaccard"] = df[["Jaccard_1", "Jaccard_2", "Jaccard_3"]].max(axis=1)

# Results
print("Best Jaccard:", df["Best_Jaccard"].mean())

# Save
df.to_excel(
    "rag_test_results_jaccard.xlsx",
    index=False
)

import pandas as pd

df = pd.read_excel(
    "rag_test_results_jaccard.xlsx"
)

df.head()

import pandas as pd
import re

# Load file
df = pd.read_excel("top3_retrieval_results.xlsx")

# -----------------------------
# Extract Sections
# -----------------------------
def extract_sections(text):
    if pd.isna(text) or str(text).strip() == "0":
        return set()

    text = str(text).lower().replace(" ", "")

    tokens = re.findall(r'\d+[a-z]?|cnsa|saa|tip', text)

    final_set = set()

    for t in tokens:
        final_set.add(t)

        # Add base section (381a -> 381)
        m = re.match(r'^(\d+)', t)
        if m:
            final_set.add(m.group(1))

    return final_set

# -----------------------------
# Create universe of all labels
# -----------------------------
all_labels = set()

for col in ["Actual_Section",
            "Retrieved_Section_1",
            "Retrieved_Section_2",
            "Retrieved_Section_3"]:

    if col in df.columns:
        df[col].apply(lambda x: all_labels.update(extract_sections(x)))

all_labels = sorted(all_labels)

# -----------------------------
# Hamming Loss Function
# -----------------------------
def hamming_loss(actual, predicted):

    A = extract_sections(actual)
    B = extract_sections(predicted)

    if len(all_labels) == 0:
        return 0

    mismatches = 0

    for label in all_labels:
        actual_bit = label in A
        pred_bit = label in B

        if actual_bit != pred_bit:
            mismatches += 1

    return round(mismatches / len(all_labels), 4)

# -----------------------------
# Calculate Hamming Loss
# -----------------------------
for i in [1, 2, 3]:

    col = f"Retrieved_Section_{i}"

    if col in df.columns:
        df[f"Hamming_{i}"] = df.apply(
            lambda row: hamming_loss(
                row["Actual_Section"],
                row[col]
            ),
            axis=1
        )
    else:
        df[f"Hamming_{i}"] = 1

# -----------------------------
# Best Hamming Loss
# (Lower is Better)
# -----------------------------
df["Best_Hamming"] = df[
    ["Hamming_1", "Hamming_2", "Hamming_3"]
].min(axis=1)

print("Average Best Hamming Loss:",
      round(df["Best_Hamming"].mean(), 4))

# -----------------------------
# Save Result
# -----------------------------
df.to_excel(
    "rag_test_results_hamming.xlsx",
    index=False
)

import pandas as pd
import re

# Load file
df = pd.read_excel("top3_retrieval_results.xlsx")


def extract_sections(text):
    if pd.isna(text) or str(text).strip() == "0":
        return set()

    text = str(text).lower().replace(" ", "")

    tokens = re.findall(r'\d+[a-z]?|cnsa|saa|tip', text)

    final_set = set()

    for t in tokens:

        # Add base section
        base_match = re.match(r'^(\d+)', t)
        if base_match:
            final_set.add(base_match.group(1))

    return final_set


def is_relevant(actual, retrieved):
    if pd.isna(retrieved) or str(retrieved).strip() == "0":
        return 0

    A = extract_sections(actual)
    B = extract_sections(retrieved)

    # At least one common PPC section
    if len(A & B) > 0:
        return 1
    else:
        return 0


# Calculate Hit for each retrieved FIR
for i in [1, 2, 3]:
    col = f"Retrieved_Section_{i}"

    if col in df.columns:
        df[f"Hit_{i}"] = df.apply(
            lambda row: is_relevant(
                row["Actual_Section"],
                row[col]
            ),
            axis=1
        )
    else:
        df[f"Hit_{i}"] = 0


# Top-3 Hit
df["Top3_Hit"] = df[["Hit_1", "Hit_2", "Hit_3"]].max(axis=1)

# Top-3 Retrieval Accuracy
top3_accuracy = df["Top3_Hit"].mean()

print(f"Top-3 Retrieval Accuracy: {top3_accuracy:.4f}")

# Save
df.to_excel(
    "rag_test_results_top3_accuracy.xlsx",
    index=False
)