"""DataStory AI - Turn messy CSV data into interactive data stories."""

__version__ = "0.1.0"
__author__ = "DataStory AI Team"

# Main exports for easy importing
from datastory.analysis import (
    profile_dataframe,
    get_profile_summary_text,
    detect_trends,
    detect_correlations,
    detect_anomalies,
    detect_segments
)
from datastory.ai import GroqClient
from datastory.story import (
    CHAPTERS,
    render_chapter_tabs,
    render_followup_chat
)
from datastory.utils import (
    load_csv_from_upload,
    get_column_info,
    generate_executive_summary
)

__all__ = [
    'profile_dataframe',
    'get_profile_summary_text',
    'detect_trends',
    'detect_correlations',
    'detect_anomalies',
    'detect_segments',
    'GroqClient',
    'CHAPTERS',
    'render_chapter_tabs',
    'render_followup_chat',
    'load_csv_from_upload',
    'get_column_info',
    'generate_executive_summary',
]