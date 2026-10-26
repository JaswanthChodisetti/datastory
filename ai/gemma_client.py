"""Gemma API client with retry logic and fallback."""

import os
import json
import time
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import streamlit as st


@dataclass
class GemmaConfig:
    """Configuration for Gemma API."""
    provider: str = "google_ai_studio"  # google_ai_studio, vertex_ai, together_ai, ollama
    api_key: str = ""
    base_url: str = ""
    model: str = "gemma-2-9b-it"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


class GemmaClient:
    """Client for interacting with Gemma API."""

    def __init__(self, config: Optional[GemmaConfig] = None):
        self.config = config or self._load_config()
        self.session = requests.Session()
        self._setup_session()

    def _load_config(self) -> GemmaConfig:
        """Load configuration from environment/Streamlit secrets."""
        config = GemmaConfig()

        # Try Streamlit secrets first
        try:
            if hasattr(st, 'secrets'):
                if 'gemma' in st.secrets:
                    config.api_key = st.secrets['gemma'].get('api_key', '')
                    config.provider = st.secrets['gemma'].get('provider', config.provider)
                    config.base_url = st.secrets['gemma'].get('base_url', config.base_url)
                    config.model = st.secrets['gemma'].get('model', config.model)
        except Exception:
            pass

        # Fallback to environment variables
        config.api_key = os.getenv('GEMMA_API_KEY', config.api_key)
        config.provider = os.getenv('GEMMA_PROVIDER', config.provider)
        config.base_url = os.getenv('GEMMA_BASE_URL', config.base_url)
        config.model = os.getenv('GEMMA_MODEL', config.model)

        # Set default base URLs based on provider
        if not config.base_url:
            config.base_url = self._get_default_base_url(config.provider)

        return config

    def _get_default_base_url(self, provider: str) -> str:
        """Get default base URL for provider."""
        urls = {
            'google_ai_studio': 'https://generativelanguage.googleapis.com/v1beta',
            'vertex_ai': '',  # Requires project/region
            'together_ai': 'https://api.together.xyz/v1',
            'ollama': 'http://localhost:11434/v1',
        }
        return urls.get(provider, '')

    def _setup_session(self):
        """Setup session with headers."""
        if self.config.provider == 'google_ai_studio':
            self.session.headers.update({
                'Content-Type': 'application/json',
                'x-goog-api-key': self.config.api_key
            })
        elif self.config.provider in ['together_ai', 'ollama']:
            self.session.headers.update({
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.config.api_key}'
            })

    def is_configured(self) -> bool:
        """Check if client is properly configured."""
        if not self.config.api_key:
            return False
        if self.config.provider == 'ollama':
            return True  # Ollama doesn't need API key
        return bool(self.config.api_key and self.config.base_url)

    def explain_finding(self, finding: Dict[str, Any], dataset_summary: Dict[str, Any]) -> str:
        """
        Generate explanation for a finding using Gemma.

        Args:
            finding: The finding dictionary with type, title, description, evidence
            dataset_summary: Profile summary of the dataset

        Returns:
            AI-generated explanation string
        """
        if not self.is_configured():
            return self._fallback_explanation(finding)

        prompt = self._build_finding_prompt(finding, dataset_summary)

        for attempt in range(self.config.max_retries):
            try:
                response = self._call_api(prompt)
                if response:
                    return response.strip()
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    return self._fallback_explanation(finding, str(e))
                time.sleep(self.config.retry_delay * (attempt + 1))

        return self._fallback_explanation(finding)

    def explain_followup(self, context: Dict[str, Any]) -> str:
        """Generate answer for follow-up question."""
        if not self.is_configured():
            return "AI explanations require Gemma API configuration. Please add your API key to Streamlit secrets or environment variables."

        prompt = self._build_followup_prompt(context)

        for attempt in range(self.config.max_retries):
            try:
                response = self._call_api(prompt)
                if response:
                    return response.strip()
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    return f"Error generating response: {e}"
                time.sleep(self.config.retry_delay * (attempt + 1))

        return "Failed to generate response after retries."

    def _call_api(self, prompt: str) -> Optional[str]:
        """Make API call to Gemma."""
        if self.config.provider == 'google_ai_studio':
            return self._call_google_ai_studio(prompt)
        elif self.config.provider == 'together_ai':
            return self._call_together_ai(prompt)
        elif self.config.provider == 'ollama':
            return self._call_ollama(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")

    def _call_google_ai_studio(self, prompt: str) -> Optional[str]:
        """Call Google AI Studio (Gemini) API."""
        url = f"{self.config.base_url}/models/{self.config.model}:generateContent"

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 500,
                "topP": 0.9
            }
        }

        response = self.session.post(url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()

        data = response.json()
        if 'candidates' in data and data['candidates']:
            return data['candidates'][0]['content']['parts'][0]['text']
        return None

    def _call_together_ai(self, prompt: str) -> Optional[str]:
        """Call Together.ai API."""
        url = f"{self.config.base_url}/chat/completions"

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500
        }

        response = self.session.post(url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()

        data = response.json()
        if 'choices' in data and data['choices']:
            return data['choices'][0]['message']['content']
        return None

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama local API."""
        url = f"{self.config.base_url}/chat/completions"

        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500,
            "stream": False
        }

        response = self.session.post(url, json=payload, timeout=self.config.timeout)
        response.raise_for_status()

        data = response.json()
        if 'choices' in data and data['choices']:
            return data['choices'][0]['message']['content']
        # Alternative response format for some Ollama versions
        if 'message' in data:
            return data['message']['content']
        return None

    def _build_finding_prompt(self, finding: Dict[str, Any], dataset_summary: Dict[str, Any]) -> str:
        """Build prompt for finding explanation."""
        evidence_json = json.dumps(finding.get('evidence', {}), indent=2)

        prompt = f"""You are a data analyst explaining findings to a non-technical business user.

Finding: {finding.get('type', 'finding')} — {finding.get('title', 'Analysis Result')}
Description: {finding.get('description', 'No description provided')}

Evidence (exact numbers from analysis):
{evidence_json}

Dataset Context:
- Rows: {dataset_summary.get('shape', [0, 0])[0]:,}
- Columns: {dataset_summary.get('shape', [0, 0])[1]}
- Quality Score: {dataset_summary.get('quality_score', {}).get('overall', 'N/A')}/100

Rules:
1. Use ONLY the evidence numbers provided above - do not hallucinate or add outside knowledge
2. Write 2-3 sentences maximum
3. Use plain English, no technical jargon
4. End with ONE actionable suggestion
5. Be specific - reference actual values from the evidence

Explanation:"""

        return prompt

    def _build_followup_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for follow-up question."""
        prompt = f"""You are a data analyst answering a user's question about their dataset.

User Question: {context.get('question', '')}

Dataset Info:
- Columns: {', '.join(context.get('columns', []))}
- Shape: {context.get('profile', {}).get('shape', 'Unknown')}
- Key Findings: {context.get('findings_summary', {})}

Answer the question using only the data context provided. Be concise (2-3 sentences), use plain English, and reference specific numbers when possible. If you cannot answer from the given context, say so.

Answer:"""

        return prompt

    def _fallback_explanation(self, finding: Dict[str, Any], error: str = None) -> str:
        """Generate template-based explanation when AI is unavailable."""
        finding_type = finding.get('type', 'finding')
        evidence = finding.get('evidence', {})

        templates = {
            'trend': self._template_trend,
            'correlation': self._template_correlation,
            'outlier_iqr': self._template_outlier,
            'outlier_zscore': self._template_outlier,
            'sudden_change': self._template_sudden_change,
            'segment': self._template_segment,
            'segment_2d': self._template_segment,
            'categorical_numeric': self._template_categorical_numeric,
            'quality_issue': self._template_quality,
        }

        template_func = templates.get(finding_type, self._template_generic)
        return template_func(evidence, error)

    def _template_trend(self, ev: Dict, error: str) -> str:
        direction = ev.get('trend_direction', 'upward')
        pct = ev.get('change_pct', 0)
        col = ev.get('numeric_column', 'the metric')
        dt = ev.get('datetime_column', 'time')
        return (f"{col} shows a {direction} trend over {dt}, changing by {pct:.1f}% across the period. "
                f"The trend is statistically significant (p={ev.get('mann_kendall_p', 0):.3f}). "
                f"Consider investigating what drove this change.")

    def _template_correlation(self, ev: Dict, error: str) -> str:
        c1 = ev.get('column_1', 'X')
        c2 = ev.get('column_2', 'Y')
        corr = ev.get('correlation', 0)
        direction = 'increases' if corr > 0 else 'decreases'
        return (f"{c1} and {c2} have a {ev.get('direction', 'positive')} {ev.get('relationship_type', 'linear')} "
                f"relationship (r={corr:.3f}). When {c1} goes up, {c2} tends to {direction}. "
                f"Explore whether this relationship is causal or driven by a third factor.")

    def _template_outlier(self, ev: Dict, error: str) -> str:
        col = ev.get('column', 'the metric')
        count = ev.get('outlier_count', 0)
        pct = ev.get('outlier_percentage', 0)
        method = ev.get('method', 'statistical')
        return (f"Found {count} outliers ({pct:.1f}%) in {col} using {method} method. "
                f"These values fall outside the expected range and may indicate data entry errors, "
                f"rare events, or important exceptions. Review the flagged rows to decide if they should be kept or corrected.")

    def _template_sudden_change(self, ev: Dict, error: str) -> str:
        col = ev.get('numeric_column', 'the metric')
        pct = ev.get('percent_change', 0)
        direction = ev.get('direction', 'change')
        date = ev.get('date_after', 'a specific point')
        return (f"{col} had a sudden {direction} of {pct:.1f}% at {date}. "
                f"This abrupt shift is unusual compared to normal variation. "
                f"Investigate external events or data collection changes around this time.")

    def _template_segment(self, ev: Dict, error: str) -> str:
        num_col = ev.get('numeric_column', 'the metric')
        cat_col = ev.get('categorical_column', 'category')
        seg_val = ev.get('segment_value', 'a segment')
        pct = ev.get('pct_difference', 0)
        direction = 'higher' if pct > 0 else 'lower'
        return (f"The segment {cat_col}={seg_val} has {abs(pct):.1f}% {direction} {num_col} than average. "
                f"This segment (n={ev.get('segment_count', 0)}) stands out from the rest. "
                f"Consider what makes this segment different and whether to target or investigate it.")

    def _template_categorical_numeric(self, ev: Dict, error: str) -> str:
        num_col = ev.get('numeric_column', 'the metric')
        cat_col = ev.get('categorical_column', 'category')
        max_g = ev.get('max_group', 'top group')
        min_g = ev.get('min_group', 'bottom group')
        diff = ev.get('difference_pct', 0)
        return (f"{num_col} varies significantly across {cat_col} categories (p={ev.get('p_value', 0):.4f}). "
                f"{max_g} averages {diff:.0f}% higher than {min_g}. "
                f"This suggests {cat_col} is an important factor for {num_col}.")

    def _template_quality(self, ev: Dict, error: str) -> str:
        issue = ev.get('issue_type', 'quality issue')
        col = ev.get('column', 'a column')
        if issue == 'high_missingness':
            pct = ev.get('missing_percentage', 0)
            return (f"{col} has {pct:.1f}% missing values. This could bias analysis results. "
                    f"Consider imputation strategies or understand why data is missing.")
        elif issue == 'duplicates':
            pct = ev.get('duplicate_percentage', 0)
            return (f"Found {ev.get('duplicate_count', 0)} duplicate rows ({pct:.1f}%). "
                    f"Deduplication recommended before analysis.")
        return f"Data quality issue detected in {col}: {issue}."

    def _template_generic(self, ev: Dict, error: str) -> str:
        if error:
            return f"Analysis complete. AI explanation unavailable ({error}). Key evidence: {ev}"
        return f"Analysis complete. Key evidence: {ev}. Consider reviewing the numbers above for insights."