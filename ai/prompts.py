"""Prompt templates for Gemma AI explanations."""

# Finding explanation prompt template
FINDING_PROMPT_TEMPLATE = """You are a data analyst explaining findings to a non-technical business user.

Finding: {finding_type} — {title}
Description: {description}

Evidence (exact numbers from analysis):
{evidence_json}

Dataset Context:
- Rows: {n_rows:,}
- Columns: {n_cols}
- Quality Score: {quality_score}/100

Rules:
1. Use ONLY the evidence numbers provided above - do not hallucinate or add outside knowledge
2. Write 2-3 sentences maximum
3. Use plain English, no technical jargon
4. End with ONE actionable suggestion
5. Be specific - reference actual values from the evidence

Explanation:"""

# Follow-up question prompt template
FOLLOWUP_PROMPT_TEMPLATE = """You are a data analyst answering a user's question about their dataset.

User Question: {question}

Dataset Info:
- Columns: {columns}
- Shape: {shape}
- Key Findings: {findings_summary}

Answer the question using only the data context provided. Be concise (2-3 sentences), use plain English, and reference specific numbers when possible. If you cannot answer from the given context, say so.

Answer:"""

# Chapter summary prompt template
CHAPTER_SUMMARY_PROMPT_TEMPLATE = """You are a data analyst writing a chapter summary for a data story.

Chapter: {chapter_title}
Findings in this chapter:
{findings_list}

Dataset Context:
- Rows: {n_rows:,}
- Columns: {n_cols}

Write a 3-4 sentence executive summary of this chapter's key insights for a non-technical reader. Focus on the most surprising or actionable findings. End with one clear recommendation.

Summary:"""

# Anomaly explanation prompt (more detailed for anomalies)
ANOMALY_PROMPT_TEMPLATE = """You are a data analyst explaining an anomaly to a business user.

Anomaly Type: {anomaly_type}
Column: {column}
Method: {method}
Key Statistics: {stats_json}

Explain what this anomaly means in plain English (2-3 sentences). What should the user investigate? Be specific about the numbers.

Explanation:"""