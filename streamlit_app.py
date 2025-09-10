import streamlit as st
import pandas as pd
import numpy as np
import requests, json, time
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
APP_TZ = ZoneInfo("America/Sao_Paulo")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============== Config & Estilos ===============
st.set_page_config(page_title="Meta Ads — Paridade + Funil", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .small-muted { color:#6b7280; font-size:12px; }
    .kpi-card { padding:14px 16px; border:1px solid #e5e7eb; border-radius:14px; background:#fff; }
    .kpi-card .big-number { font-size:28px; font-weight:700; color:#000 !important; }
    </style>
    """,
    unsafe_allow_html=True
)
