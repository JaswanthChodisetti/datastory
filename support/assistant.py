"""Floating Support Assistant Widget for DataTale AI."""

import streamlit as st
from typing import List, Dict, Any
from support.responses import get_support_response

def render_support_assistant():
    """Renders the TaleMate AI floating support widget."""

    # 1. Session State Initialization
    if 'support_open' not in st.session_state:
        st.session_state.support_open = False
    if 'support_messages' not in st.session_state:
        st.session_state.support_messages = [
            {"role": "assistant", "content": (
                "Hi! I'm TaleMate, your DataTale AI assistant. I can help you explore your data, "
                "understand insights, verify evidence, and create better data stories. "
                "What can I help you with today?"
            )}
        ]

    # 2. CSS for the floating widget
    st.markdown("""
    <style>
        /* The Floating Launcher */
        .support-launcher {
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .support-launcher:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        }
        .support-launcher-icon {
            font-size: 30px;
            color: white;
        }

        /* The Chat Panel */
        .support-panel {
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 350px;
            max-height: 600px;
            background: #1e293b;
            border-radius: 16px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            display: flex;
            flex-direction: column;
            z-index: 9998;
            overflow: hidden;
            font-family: 'Inter', sans-serif;
            border: 1px solid #334155;
        }
        .support-panel-header {
            background: #0f172a;
            padding: 15px 20px;
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid #334155;
        }
        .support-panel-title {
            font-weight: 600;
            font-size: 1.1rem;
            margin: 0;
        }
        .support-panel-subtitle {
            font-size: 0.75rem;
            color: #94a3b8;
            margin: 0;
        }
        .support-panel-body {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: 400px;
        }
        .support-msg {
            max-width: 85%;
            padding: 10px 14px;
            border-radius: 12px;
            font-size: 0.9rem;
            line-height: 1.4;
            margin-bottom: 8px;
        }
        .support-msg-user {
            align-self: flex-end;
            background: #4f46e5;
            color: white;
            border-bottom-right-radius: 2px;
        }
        .support-msg-assistant {
            align-self: flex-start;
            background: #334155;
            color: #f1f5f9;
            border-bottom-left-radius: 2px;
        }
        .support-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
        }
        .support-btn {
            background: #334155;
            color: #cbd5e1;
            border: 1px solid #475569;
            border-radius: 20px;
            padding: 6px 12px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: background 0.2s;
            text-align: center;
        }
        .support-btn:hover {
            background: #475569;
            color: white;
        }
        .support-input-area {
            padding: 15px;
            background: #0f172a;
            border-top: 1px solid #334155;
            display: flex;
            gap: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # 3. Floating Launcher
    # We use a custom HTML element that clicks a hidden Streamlit button
    st.markdown(f"""
    <div class="support-launcher" onclick="window.parent.document.querySelector('button[aria-label=\'Toggle Support Assistant\']').click()">
        <span class="support-launcher-icon">💬</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Toggle Support Assistant", key="support_toggle_btn", help="Toggle Support Assistant"):
        st.session_state.support_open = not st.session_state.support_open
        st.rerun()

    st.markdown('<style>.stButton > button[aria-label="Toggle Support Assistant"] { display: none; }</style>', unsafe_allow_html=True)

    # 4. Expanded Chat Panel
    if st.session_state.support_open:
        # Use a container that we can style as the panel
        # We use an empty container to isolate the panel styling
        with st.container():
            st.markdown('<div class="support-panel">', unsafe_allow_html=True)

            # Header
            st.markdown("""
            <div class="support-panel-header">
                <div>
                    <p class="support-panel-title">TaleMate AI</p>
                    <p class="support-panel-subtitle">Your DataTale support assistant</p>
                </div>
                <div style="cursor:pointer; color: #94a3b8;" onclick="window.parent.document.querySelector('button[aria-label=\'Toggle Support Assistant\']').click()">
                    ✖
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Chat Body
            st.markdown('<div class="support-panel-body">', unsafe_allow_html=True)

            # Quick Actions
            st.markdown('<div class="support-actions">', unsafe_allow_html=True)

            # Note: Since we can't put st.button inside raw HTML strings,
            # we use a simple list of buttons. Streamlit will render them
            # following the HTML, so we just need to ensure they are visually
            # contained.

            st.markdown('</div>', unsafe_allow_html=True)

            # Render actual Streamlit buttons for quick actions
            # We wrap them in a div via markdown just to help with layout if needed
            # But it's easier to just use st.columns or similar

            q_cols = st.columns(2)
            actions = [
                ("Upload & analyze data", "upload"),
                ("Explain an insight", "insight"),
                ("Verify the evidence", "evidence"),
                ("Export my data story", "export")
            ]

            for idx, (label, key) in enumerate(actions):
                with q_cols[idx % 2]:
                    if st.button(label, key=f"sup_act_{key}", use_container_width=True):
                        response = get_support_response(label, st.session_state)
                        st.session_state.support_messages.append({"role": "user", "content": label})
                        st.session_state.support_messages.append({"role": "assistant", "content": response})
                        st.rerun()

            # Render Messages
            for msg in st.session_state.support_messages:
                role_class = "support-msg-user" if msg["role"] == "user" else "support-msg-assistant"
                st.markdown(f'<div class="support-msg {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Input Area
            st.markdown('<div class="support-input-area">', unsafe_allow_html=True)

            # Use a form for the input to handle Enter key
            with st.form("support_chat_form", clear_on_submit=True):
                cols = st.columns([0.8, 0.2])
                with cols[0]:
                    user_input = st.text_input("Message TaleMate...", label_visibility="collapsed", placeholder="Type your question here...")
                with cols[1]:
                    submit = st.form_submit_button("Send", use_container_width=True)

                if submit and user_input:
                    response = get_support_response(user_input, st.session_state)
                    st.session_state.support_messages.append({"role": "user", "content": user_input})
                    st.session_state.support_messages.append({"role": "assistant", "content": response})
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
