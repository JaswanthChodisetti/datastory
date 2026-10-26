"""Analysis package - statistical detection modules."""
from .profile import profile_dataframe, get_profile_summary_text
from .trends import detect_trends
from .correlations import detect_correlations
from .anomalies import detect_anomalies
from .segments import detect_segments

__all__ = [
    'profile_dataframe',
    'get_profile_summary_text',
    'detect_trends',
    'detect_correlations',
    'detect_anomalies',
    'detect_segments',
]