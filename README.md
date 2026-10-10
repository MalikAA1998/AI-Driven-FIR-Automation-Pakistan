# AI-Driven FIR Automation for Digital Policing in Pakistan

This project presents an AI-driven framework for identifying relevant Pakistan Penal Code (PPC) sections from Urdu First Information Reports (FIRs). The framework combines semantic retrieval with retrieval-supported legal reasoning to assist in automated analysis of criminal complaints.

## Overview

Urdu FIRs contain important legal information, but their unstructured nature makes automated legal analysis challenging. This project develops a retrieval-supported framework that processes Urdu FIR narratives, retrieves semantically similar cases, and uses a small language model (SLM) to recommend relevant PPC sections with contextual legal reasoning.

## Dataset

The dataset contains:

* **200 original Urdu FIRs**
* **125 original FIRs used for training**
* **75 original FIRs retained as the test set**
* **3,054 synthetically generated FIRs**
* **3,179 FIRs in the final training dataset** (125 original training FIRs + 3,054 synthetic FIRs)

Synthetic FIRs were generated using relevant PPC sections and crime categories represented in the original FIR dataset. The generated cases introduce new factual scenarios and linguistic variations while maintaining consistency between the FIR narrative, trigger words, crime category, and relevant PPC sections.

## Methodology

The overall framework consists of the following stages:

1. Urdu FIR text extraction and preprocessing
2. Data anonymization
3. Metadata extraction, including crime category, trigger words, and applied PPC sections
4. Synthetic FIR generation
5. Semantic embedding using multilingual embedding models
6. Comparison of embedding models for Urdu FIR retrieval
7. Semantic retrieval of relevant FIR cases
8. Retrieval-supported legal reasoning using an SLM
9. PPC section recommendation
10. Quantitative and qualitative evaluation

## Embedding Models

The project evaluates multiple multilingual embedding models for Urdu FIR retrieval:

* all-MiniLM-L6-v2
* Paraphrase-Multilingual-MiniLM-L12-v2
* BGE-M3

BGE-M3 achieved the strongest retrieval performance among the evaluated models and was selected for subsequent retrieval experiments.

## Retrieval

Semantic retrieval is used to identify legally similar FIR cases from the available dataset. Dense embeddings are generated for FIR narratives and indexed for similarity-based retrieval.

The retrieval system is evaluated using:

* Jaccard Similarity
* Hamming Loss
* Top-3 Retrieval Accuracy

## Retrieval-Supported Legal Reasoning

Retrieved FIRs are used as supporting evidence for an SLM-based legal reasoning process. The model analyzes the current FIR together with relevant retrieved evidence and recommends applicable PPC sections rather than directly copying sections from retrieved cases.

## Evaluation

The framework evaluates both semantic retrieval and legal reasoning. Quantitative evaluation uses Jaccard Similarity, Hamming Loss, Top-3 Retrieval Accuracy, and Exact Match Accuracy. Qualitative case studies are also used to examine the reasoning behind PPC recommendations.

## Privacy

Original FIRs may contain sensitive personal and legal information. Therefore, raw original FIR documents and personally identifiable information should not be uploaded to a public repository. Publicly released data should be anonymized, synthetic, or otherwise approved for sharing.

## Repository Structure

```text
DATA/       Dataset and sample data
Script/     Data processing, embedding, retrieval, and evaluation scripts
PPC/        PPC reference data
Result/     Retrieval and reasoning results
Prompt/     OCR, anonymization and Synthetic Data 
```

## Technologies

* Python
* Pandas
* Sentence Transformers
* BGE-M3
* FAISS
* Large Language Models / Small Language Models
* Jaccard Similarity
* Hamming Loss

## Research Context

This repository accompanies research on AI-driven FIR automation for digital policing in Pakistan, focusing on Urdu legal text processing, semantic retrieval, and retrieval-supported legal reasoning.

