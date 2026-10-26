"""Chapter definitions and rendering for the 4-chapter data story."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
from story.evidence import format_evidence_table, format_finding_evidence
from analysis.code_snippets import get_code_snippet


# Chapter configuration
CHAPTERS = [
    {
        'id': 'big_picture',
        'title': 'Chapter 1: The Big Picture',
        'icon': '📊',
        'description': 'Overview, key metrics, and data shape'
    },
    {
        'id': 'opportunities',
        'title': 'Chapter 2: Hidden Opportunities',
        'icon': '🔍',
        'description': 'Trends, correlations, and segments worth exploring'
    },
    {
        'id': 'anomalies',
        'title': 'Chapter 3: Unexpected Anomalies',
        'icon': '⚠️',
        'description': 'Outliers, sudden changes, and data quality issues'
    },
    {
        'id': 'actions',
        'title': 'Chapter 4: What You Should Do Next',
        'icon': '🎯',
        'description': 'Actionable recommendations with evidence'
    }
]


def render_chapter_tabs(findings: Dict[str, List[Dict]], profile: Dict[str, Any], df: pd.DataFrame):
    """Render the 4 chapter tabs with findings."""
    tabs = st.tabs([f"{c['icon']} {c['title']}" for c in CHAPTERS])

    for i, (tab, chapter) in enumerate(zip(tabs, CHAPTERS)):
        with tab:
            render_chapter(chapter['id'], findings.get(chapter['id'], []), profile, df)


def render_chapter(chapter_id: str, chapter_findings: List[Dict], profile: Dict[str, Any], df: pd.DataFrame):
    """Render a single chapter with its findings."""
    chapter_config = next(c for c in CHAPTERS if c['id'] == chapter_id)

    st.markdown(f"### {chapter_config['icon']} {chapter_config['title']}")
    st.markdown(f"*{chapter_config['description']}*")
    st.divider()

    if chapter_id == 'big_picture':
        render_big_picture(profile, df)
    elif not chapter_findings:
        st.info("No significant findings in this chapter. Run analysis to discover insights.")
    else:
        for j, finding in enumerate(chapter_findings):
            render_finding_card(finding, chapter_id, j, df)


def render_big_picture(profile: Dict[str, Any], df: pd.DataFrame):
    """Render the Big Picture chapter with overview charts."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", f"{profile['shape'][0]:,}")
    with col2:
        st.metric("Columns", profile['shape'][1])
    with col3:
        st.metric("Quality Score", f"{profile['quality_score']['overall']}/100")
    with col4:
        st.metric("Duplicates", f"{profile['duplicates']:,}")

    st.divider()

    # Data types distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Column Types")
        type_counts = {}
        for col, sem_type in [(c['Column'], c['Semantic Type']) for c in profile['columns']]:
            type_counts[sem_type] = type_counts.get(sem_type, 0) + 1

        fig = px.pie(
            values=list(type_counts.values()),
            names=list(type_counts.keys()),
            title="Column Semantic Types",
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True, height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Missing Values by Column")
        missing_df = pd.DataFrame({
            'Column': list(profile['missing'].keys()),
            'Missing %': list(profile['missing_pct'].values())
        }).sort_values('Missing %', ascending=True)

        fig = px.bar(
            missing_df,
            x='Missing %',
            y='Column',
            orientation='h',
            title="Missing Value Percentage",
            color='Missing %',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # Numeric distributions
    if profile['numeric_summary']:
        st.markdown("#### Numeric Column Distributions")
        numeric_cols = list(profile['numeric_summary'].keys())
        selected_col = st.selectbox("Select column to visualize", numeric_cols, key="bigpic_numeric")

        if selected_col:
            col1, col2 = st.columns(2)
            with col1:
                # Histogram
                fig = px.histogram(df, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Box plot
                fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    # Categorical distributions
    if profile['categorical_summary']:
        st.markdown("#### Top Categorical Values")
        cat_cols = list(profile['categorical_summary'].keys())
        selected_cat = st.selectbox("Select categorical column", cat_cols, key="bigpic_categorical")

        if selected_cat:
            value_counts = df[selected_cat].value_counts().head(15)
            fig = px.bar(
                x=value_counts.values,
                y=value_counts.index,
                orientation='h',
                title=f"Top 15 Values in {selected_cat}",
                labels={'x': 'Count', 'y': selected_cat}
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)


def render_finding_card(finding: Dict[str, Any], chapter_id: str, index: int, df: pd.DataFrame):
    """Render a single finding as an expandable card with evidence."""
    finding_type = finding.get('type', 'finding')
    title = finding.get('title', f'{finding_type.title()} #{index + 1}')
    description = finding.get('description', '')
    evidence = finding.get('evidence', {})
    ai_explanation = finding.get('ai_explanation', '')
    chart_data = finding.get('chart_data')
    chart_type = finding.get('chart_type')

    with st.expander(f"**{title}**", expanded=(index == 0)):
        # AI Explanation
        if ai_explanation:
            st.markdown(f"**🤖 AI Explanation:** {ai_explanation}")
            st.divider()

        # Description
        if description:
            st.markdown(f"**Finding:** {description}")

        # Evidence
        if evidence:
            st.markdown("**📋 Evidence:**")
            format_finding_evidence(evidence)

        # Chart
        if chart_data is not None and chart_type:
            render_finding_chart(chart_data, chart_type, finding.get('chart_config', {}))

        # Show Calculation - THE KEY DIFFERENTIATOR
        code_snippet = get_code_snippet(finding_type)
        if code_snippet and code_snippet.strip() != '# No code snippet available for this finding type':
            with st.expander("🔍 Show Calculation", expanded=False):
                st.markdown("**Exact Python code used to compute this finding:**")
                st.code(code_snippet, language='python')
                st.caption("Copy this code to verify, reproduce, or extend the analysis.")

        # Raw evidence table (collapsed)
        if evidence:
            with st.expander("View Raw Evidence Table"):
                format_evidence_table(evidence)


def render_finding_chart(chart_data: Dict, chart_type: str, config: Dict):
    """Render a Plotly chart based on chart_data and type."""
    try:
        if chart_type == 'scatter':
            fig = px.scatter(
                chart_data,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                size=config.get('size'),
                hover_data=config.get('hover_data'),
                title=config.get('title', 'Scatter Plot'),
                trendline=config.get('trendline')
            )

        elif chart_type == 'line':
            fig = px.line(
                chart_data,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                title=config.get('title', 'Line Chart')
            )

        elif chart_type == 'bar':
            fig = px.bar(
                chart_data,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                title=config.get('title', 'Bar Chart'),
                orientation=config.get('orientation', 'v')
            )

        elif chart_type == 'heatmap':
            fig = px.imshow(
                chart_data,
                title=config.get('title', 'Heatmap'),
                color_continuous_scale=config.get('colorscale', 'RdBu'),
                aspect='auto'
            )

        elif chart_type == 'box':
            fig = px.box(
                chart_data,
                x=config.get('x'),
                y=config.get('y'),
                color=config.get('color'),
                title=config.get('title', 'Box Plot')
            )

        else:
            st.warning(f"Unknown chart type: {chart_type}")
            return

        fig.update_layout(height=config.get('height', 350))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not render chart: {e}")


def render_followup_chat(df: pd.DataFrame, profile: Dict, findings: Dict, gemma_client=None):
    """Render the follow-up chat interface."""
    st.markdown("---")
    st.markdown("### 💬 Ask a Follow-up Question")
    st.caption("Ask anything about your data - trends, specific values, comparisons, etc.")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            if msg.get('evidence'):
                with st.expander("Evidence"):
                    format_evidence_table(msg['evidence'])

    # Chat input
    if prompt := st.chat_input("Ask about your data..."):
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        # Generate response (placeholder - would integrate with Gemma)
        with st.chat_message('assistant'):
            with st.spinner("Analyzing..."):
                response, evidence = answer_followup_question(prompt, df, profile, findings, gemma_client)
                st.markdown(response)
                if evidence:
                    with st.expander("Evidence"):
                        format_evidence_table(evidence)

        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response,
            'evidence': evidence
        })


def answer_followup_question(question: str, df: pd.DataFrame, profile: Dict, findings: Dict, gemma_client=None) -> tuple:
    """
    Answer a follow-up question about the data.
    Returns (response_text, evidence_dict).
    """
    # Simple keyword-based routing for demo
    question_lower = question.lower()

    # If Gemma client available, use it
    if gemma_client:
        try:
            context = {
                'question': question,
                'profile': profile,
                'findings_summary': {k: len(v) for k, v in findings.items()},
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict()
            }
            response = gemma_client.explain_followup(context)
            return response, {}
        except Exception as e:
            pass  # Fall through to rule-based

    # Rule-based fallback
    if any(kw in question_lower for kw in ['mean', 'average', 'avg']):
        return answer_aggregation_question(question, df, 'mean')
    elif any(kw in question_lower for kw in ['max', 'maximum', 'highest']):
        return answer_aggregation_question(question, df, 'max')
    elif any(kw in question_lower for kw in ['min', 'minimum', 'lowest']):
        return answer_aggregation_question(question, df, 'min')
    elif any(kw in question_lower for kw in ['correlation', 'correlate', 'related']):
        return answer_correlation_question(question, df)
    elif any(kw in question_lower for kw in ['trend', 'trending', 'over time', 'time series']):
        return answer_trend_question(question, df, profile)
    elif any(kw in question_lower for kw in ['outlier', 'anomal', 'unusual', 'strange']):
        return answer_anomaly_question(question, df)
    else:
        return (
            "I can help you explore your data! Try asking about:\n"
            "- Averages, maximums, minimums of numeric columns\n"
            "- Correlations between columns\n"
            "- Trends over time\n"
            "- Outliers or unusual values\n\n"
            "For example: 'What's the average sales by region?' or 'Show me the trend of revenue over time.'",
            {}
        )


def answer_aggregation_question(question: str, df: pd.DataFrame, agg_type: str) -> tuple:
    """Answer aggregation questions."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        return "No numeric columns found for aggregation.", {}

    # Try to find mentioned column
    mentioned_col = None
    for col in numeric_cols:
        if col.lower() in question.lower():
            mentioned_col = col
            break

    if not mentioned_col:
        mentioned_col = numeric_cols[0]

    if agg_type == 'mean':
        value = df[mentioned_col].mean()
        label = "Average"
    elif agg_type == 'max':
        value = df[mentioned_col].max()
        label = "Maximum"
    else:
        value = df[mentioned_col].min()
        label = "Minimum"

    evidence = {
        'column': mentioned_col,
        'operation': agg_type,
        'value': float(value),
        'sample_size': int(df[mentioned_col].count())
    }

    return f"The {label.lower()} of **{mentioned_col}** is **{value:,.2f}** (based on {evidence['sample_size']:,} non-null values).", evidence


def answer_correlation_question(question: str, df: pd.DataFrame) -> tuple:
    """Answer correlation questions."""
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.shape[1] < 2:
        return "Need at least 2 numeric columns for correlation analysis.", {}

    corr_matrix = numeric_df.corr()
    # Find strongest correlation (excluding self-correlation)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pair = corr_pairs[0] if corr_pairs else None

    if top_pair:
        evidence = {
            'column_1': top_pair[0],
            'column_2': top_pair[1],
            'correlation': float(top_pair[2]),
            'method': 'pearson'
        }
        direction = "positive" if top_pair[2] > 0 else "negative"
        return (
            f"The strongest correlation is between **{top_pair[0]}** and **{top_pair[1]}** "
            f"({direction}, r = {top_pair[2]:.3f}). This means as one increases, the other "
            f"{'tends to increase' if top_pair[2] > 0 else 'tends to decrease'}.",
            evidence
        )

    return "No significant correlations found.", {}


def answer_trend_question(question: str, df: pd.DataFrame, profile: Dict) -> tuple:
    """Answer trend questions."""
    datetime_cols = profile.get('datetime_columns', [])
    numeric_cols = list(profile.get('numeric_summary', {}).keys())

    if not datetime_cols or not numeric_cols:
        return "Need a datetime column and a numeric column to analyze trends.", {}

    # Use first datetime and first numeric column
    dt_col = datetime_cols[0]
    num_col = numeric_cols[0]

    # Simple trend: correlate time with value
    try:
        df_sorted = df.sort_values(dt_col)
        time_numeric = pd.to_datetime(df_sorted[dt_col]).astype('int64') // 10**9
        correlation = time_numeric.corr(df_sorted[num_col])

        evidence = {
            'datetime_column': dt_col,
            'numeric_column': num_col,
            'correlation_with_time': float(correlation),
            'trend_direction': 'increasing' if correlation > 0 else 'decreasing'
        }

        direction = "increasing" if correlation > 0 else "decreasing"
        strength = "strong" if abs(correlation) > 0.5 else "moderate" if abs(correlation) > 0.3 else "weak"

        return (
            f"Over time ({dt_col}), **{num_col}** shows a **{strength} {direction}** trend "
            f"(correlation with time: {correlation:.3f}).",
            evidence
        )
    except Exception as e:
        return f"Could not compute trend: {e}", {}


def answer_anomaly_question(question: str, df: pd.DataFrame) -> tuple:
    """Answer anomaly/outlier questions."""
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        return "No numeric columns for anomaly detection.", {}

    # Use IQR method on first numeric column
    col = numeric_cols[0]
    series = df[col].dropna()
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_count = len(outliers)

    evidence = {
        'column': col,
        'method': 'IQR',
        'lower_bound': float(lower),
        'upper_bound': float(upper),
        'outlier_count': int(outlier_count),
        'outlier_percentage': float(outlier_count / len(df) * 100)
    }

    if outlier_count > 0:
        example = outliers.iloc[0][col]
        return (
            f"Found **{outlier_count} outliers** in **{col}** ({evidence['outlier_percentage']:.1f}% of data) "
            f"using the IQR method. Bounds: [{lower:.2f}, {upper:.2f}]. "
            f"Example outlier: {example:.2f}.",
            evidence
        )
    else:
        return f"No outliers detected in **{col}** using IQR method.", evidence