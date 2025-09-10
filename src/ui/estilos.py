import streamlit as st

def aplicar_config_e_css():
    st.set_page_config(page_title="Meta Ads — Paridade + Funil", page_icon="📊", layout="wide")
    st.markdown("""
    <style>
    .small-muted { color:#6b7280; font-size:12px; }
    .kpi-card { padding:14px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#fff; }
    .kpi-card .big-number { font-size:28px; font-weight:700; color:#000 !important; }
    </style>
    """, unsafe_allow_html=True)
