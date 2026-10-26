"""Tests for the analysis module."""

import pandas as pd
import numpy as np
import pytest
from datastory.analysis import profile_dataframe
from datastory.analysis.anomalies import detect_anomalies
from datastory.analysis.trends import detect_trends, mann_kendall_test
from datastory.analysis.correlations import detect_correlations, detect_categorical_numeric_relationships
from datastory.analysis.segments import detect_segments


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'value': np.random.normal(100, 15, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'score': np.random.uniform(0, 100, n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n)
    })
    # Add some outliers
    df.loc[0, 'value'] = 500
    df.loc[1, 'value'] = -100
    # Add some missing values
    df.loc[5:7, 'score'] = np.nan
    return df


@pytest.fixture
def subscription_df():
    """Create a subscription-like DataFrame for testing."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    df = pd.DataFrame({
        'signup_date': dates,
        'monthly_charge': np.random.choice([19.99, 29.99, 49.99], n),
        'tenure_months': np.random.randint(1, 24, n),
        'age': np.random.randint(18, 65, n),
        'plan': np.random.choice(['Basic', 'Standard', 'Premium'], n),
        'churned': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n)
    })
    return df


class TestProfiler:
    """Tests for profiler module."""

    def test_profile_dataframe_basic(self, sample_df):
        """Test basic profiling works."""
        profile = profile_dataframe(sample_df)

        assert 'shape' in profile
        assert profile['shape'][0] == 100
        assert profile['shape'][1] == 5
        assert 'columns' in profile
        assert 'missing' in profile
        assert 'missing_pct' in profile
        assert 'duplicates' in profile
        assert 'quality_score' in profile
        assert 'numeric_summary' in profile
        assert 'categorical_summary' in profile

    def test_profile_quality_score(self, sample_df):
        """Test quality score calculation."""
        profile = profile_dataframe(sample_df)
        assert 'overall' in profile['quality_score']
        assert 'grade' in profile['quality_score']
        assert 0 <= profile['quality_score']['overall'] <= 100


class TestAnomalies:
    """Tests for anomaly detection."""

    def test_detect_anomalies(self, sample_df):
        """Test anomaly detection."""
        profile = profile_dataframe(sample_df)
        findings = detect_anomalies(sample_df, profile)

        assert isinstance(findings, list)
        # Should find outliers in 'value' column
        outlier_findings = [f for f in findings if f['type'] in ('outlier_iqr', 'outlier_zscore')]
        assert len(outlier_findings) >= 1


class TestTrends:
    """Tests for trend detection."""

    def test_detect_trends(self, sample_df):
        """Test trend detection with datetime."""
        profile = profile_dataframe(sample_df)
        findings = detect_trends(sample_df, profile)

        assert isinstance(findings, list)
        # Findings may be empty if no significant trends, but should not error

    def test_mann_kendall_test(self):
        """Test Mann-Kendall trend test function."""
        # Increasing trend
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = mann_kendall_test(data)
        assert 'tau' in result
        assert 'p_value' in result
        assert result['tau'] > 0
        assert result['p_value'] < 0.05

        # No trend
        data = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        result = mann_kendall_test(data)
        assert result['tau'] == 0
        assert result['p_value'] == 1.0


class TestCorrelations:
    """Tests for correlation detection."""

    def test_detect_correlations(self, sample_df):
        """Test numeric correlation detection."""
        profile = profile_dataframe(sample_df)
        findings = detect_correlations(sample_df, profile)

        assert isinstance(findings, list)
        # Each finding should have the right structure
        for f in findings:
            assert f['type'] == 'correlation'
            assert 'evidence' in f
            assert 'correlation' in f['evidence']

    def test_detect_categorical_numeric_relationships(self, subscription_df):
        """Test categorical-numeric relationship detection."""
        profile = profile_dataframe(subscription_df)
        findings = detect_categorical_numeric_relationships(subscription_df, profile)

        assert isinstance(findings, list)
        for f in findings:
            assert f['type'] == 'categorical_numeric'
            assert 'evidence' in f
            assert 'p_value' in f['evidence']


class TestSegments:
    """Tests for segment discovery."""

    def test_detect_segments(self, subscription_df):
        """Test segment discovery."""
        profile = profile_dataframe(subscription_df)
        findings = detect_segments(subscription_df, profile)

        assert isinstance(findings, list)
        for f in findings:
            assert f['type'] in ('segment', 'segment_2d')
            assert 'evidence' in f


class TestCodeSnippets:
    """Tests for code snippets."""

    def test_get_code_snippet(self):
        """Test getting code snippets for finding types."""
        from datastory.analysis.code_snippets import get_code_snippet, FINDING_CODE_MAP

        # Test known finding types
        for finding_type in FINDING_CODE_MAP.keys():
            snippet = get_code_snippet(finding_type)
            assert isinstance(snippet, str)
            assert len(snippet) > 0
            assert 'import' in snippet.lower() or '#' in snippet

    def test_unknown_finding_type(self):
        """Test unknown finding type returns default message."""
        from datastory.analysis.code_snippets import get_code_snippet

        snippet = get_code_snippet('unknown_type')
        assert 'No code snippet available' in snippet


class TestExport:
    """Tests for export module."""

    def test_export_to_markdown(self, sample_df):
        """Test markdown export."""
        from datastory.export import export_to_markdown

        profile = profile_dataframe(sample_df)
        findings = {
            'big_picture': [],
            'opportunities': [],
            'anomalies': [],
            'actions': []
        }
        metadata = {'rows': 100, 'cols': 5, 'encoding': 'utf-8', 'delimiter': ','}

        markdown = export_to_markdown(findings, profile, metadata, sample_df)

        assert isinstance(markdown, str)
        assert 'DataStory AI Report' in markdown
        assert 'Dataset Overview' in markdown

    def test_export_to_html(self, sample_df):
        """Test HTML export."""
        from datastory.export import export_to_html

        profile = profile_dataframe(sample_df)
        findings = {
            'big_picture': [],
            'opportunities': [],
            'anomalies': [],
            'actions': []
        }
        metadata = {'rows': 100, 'cols': 5, 'encoding': 'utf-8', 'delimiter': ','}

        html = export_to_html(findings, profile, metadata, sample_df)

        assert isinstance(html, str)
        assert '<html' in html.lower()
        assert 'DataStory AI Report' in html

    def test_export_findings_json(self, sample_df):
        """Test JSON export."""
        from datastory.export import export_findings_json

        profile = profile_dataframe(sample_df)
        findings = {
            'big_picture': [],
            'opportunities': [],
            'anomalies': [],
            'actions': []
        }
        metadata = {'rows': 100, 'cols': 5, 'encoding': 'utf-8', 'delimiter': ','}

        json_str = export_findings_json(findings, profile, metadata)

        assert isinstance(json_str, str)
        import json
        data = json.loads(json_str)
        assert 'generated_at' in data
        assert 'profile' in data
        assert 'findings' in data