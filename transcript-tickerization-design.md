# Transcript Tickerization Benchmark

This is a finance-specific version of Named Entity Recognition (NER) and entity normalization (company → ticker mapping) problem.

# Labeling Design

The best practice is separate the problem into two steps:

### Step 1 — Detect company mentions (entity extraction)

Identify company names mentioned in text:

- “Apple”
- “Microsoft”
- “Nvidia”
- “Google”

### Step 2 — Normalize to a canonical ticker

Map:

- Apple → AAPL
- Microsoft → MSFT
- Alphabet / Google → GOOGL
- Nvidia → NVDA

This separation reduces hallucination and improves explainability. We can also deal with ticker changes more flexibly later on.

# Labeled (Tickerized) Dataset

The most general format to support financial ML:

```
{
  "doc_id": "...",
  "entities": [
    {
      "ticker": "NVDA",
      "company_name": "Nvidia",
      "evidence_text": "Nvidia is dominating the AI chip market",
      "confidence": 0.93
    }
  ]
}
```

This accomplishes several goals:

- It is auditable
- It can be verified
- It supports re-scoring later

# Prompt Best Practices Across LLM Providers

You want prompts that are:

✔ Model-agnostic  
✔ Non-chain-of-thought dependent  
✔ Output-constrained  
✔ Deterministic  
✔ Schema-enforced

## Core Prompt Engineering Principles

### 1\. Provide Explicit Schema

Models behave better when given exact JSON schema.

### 2\. Avoid Open-Ended Instructions

Say:

> Only return valid JSON. Do not include commentary.

### 3\. Define Edge Cases

Specify how to handle:

- Ambiguous names
- Partial references
- Ticker-only mentions

### 4\. Avoid Asking for "Reasoning"

Don't request:

> Explain why you chose this ticker.

That:

- Increases cost
- Increases variability
- Reduces stability across model versions

Instead:

- Ask for evidence spans
- Keep output structured

## Simple Standard Prompt (Production Ready)

Below is a clean prompt that works across OpenAI, Anthropic, Mistral, Gemini, etc.

### Simple Multi-Ticker Extraction Prompt

```
You are a financial entity extraction system.

Task:
Given a text transcript, identify all publicly traded companies that are explicitly mentioned.

Rules:
1. Include a company only if it is clearly referenced.
2. Include the text containing the reference as evidence.
3. If a company name appears, map it to its correct stock ticker symbol.
4. Ignore generic sector references (e.g., "tech stocks").
5. If the reference is ambiguous, include it only if context clearly refers to the company.
6. Do not guess.

Output Requirements:
- Return only valid JSON.
- Do not include commentary.
- Use the following format:

{
  "tickers": [
    {
      "ticker": "STRING",
      "company_name": "STRING",
      "evidence_text": "STRING is mentioned",
      "confidence": FLOAT BETWEEN 0 AND 1
    }
  ]
}

Transcript:
"""
{TRANSCRIPT_TEXT}
"""
```

# Human Triage

You are essentially describing a **triage labeling system**:

1. LLM A labels
2. LLM B labels
3. Disagreements go to human adjudication

Now the real challenge is:

> How should humans assign labels and confidence in a principled, reproducible way?

Let’s break this down properly.

## First: What Is the Human Actually Deciding?

For ticker labeling, the human is deciding:

> “Does this transcript segment clearly refer to Company X (ticker T)?”

This is a **binary semantic judgment problem**, not a probabilistic one in the human’s head.

Humans do not naturally think in calibrated probabilities like 0.73.

So best practice is:

- Humans assign a **categorical decision**
- System converts that into a numeric confidence score

## Best Practice: Use Tiered Confidence Buckets (Not Free-Form Probabilities)

Instead of asking a human:

> “Give a probability from 0 to 1.”

You give them structured categories.

### Standard 4-Level Schema (Recommended)

| Label | Meaning | Interpretation |
| --- | --- | --- |
| 3 – Certain | Explicit mention of company or ticker | Clear reference |
| 2 – High Confidence | Strong contextual reference | Very likely |
| 1 – Weak Mention | Ambiguous or indirect | Possible |
| 0 – Not Mentioned | Does not refer to company | No  |

This prevents:

- Random subjective numbers
- Inconsistent probability scales across annotators
- Drift over time

## Then Convert Categories to Numeric Confidence

Internally you can map:

```
3 → 0.95
2 → 0.80
1 → 0.55
0 → 0.05
```

You calibrate this mapping later using validation data.

This is how high-quality NLP datasets are built.

# Technical Notes

- It is suggested to segment docs into ~200–500 token chunks to:
  - Limit context overflow
  - Manage false positives from long discussions
- A manageable list of candidate or universe tickers really helps.