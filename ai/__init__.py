"""AI package - Groq integration and prompt templates."""
from .groq_client import GroqClient, GroqConfig
from .prompts import FINDING_PROMPT_TEMPLATE, FOLLOWUP_PROMPT_TEMPLATE

__all__ = ['GroqClient', 'GroqConfig', 'FINDING_PROMPT_TEMPLATE', 'FOLLOWUP_PROMPT_TEMPLATE']