from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Finding:
    """
    The core contract for a deterministic data insight.
    Ensures every AI claim is traceable to a specific calculation and raw data.
    """
    id: str                      # Unique slug (e.g., 'trend-revenue-q3')
    title: str                   # Short, punchy headline
    category: str                # Trend / Anomaly / Correlation / Quality / Segment
    metric: str                  # The calculated value (e.g., '+22%', 'r=0.85')
    formula: str                 # The math used (e.g., '(current - prev) / prev')
    supporting_records: List[str] # Stable identifiers (hashes/IDs) of rows that prove this
    visual_type: str            # LineChart / BarChart / ScatterPlot / Table
    observation: str             # The deterministic fact (e.g., 'Revenue increased by 22%')
    hypothesis: Optional[str] = None # The AI's interpretation (generated later)

# Sensor Thresholds
THRESHOLDS = {
    "trend_change": 0.15,       # 15% change triggers a trend finding
    "z_score": 3.0,             # 3 standard deviations for anomalies
    "correlation": 0.7,         # |r| > 0.7 for strong correlation
    "category_gap": 0.20,       # 20% difference between groups
    "quality_missing": 0.05,    # 5% missing data triggers alert
}
