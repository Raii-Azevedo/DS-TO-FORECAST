"""Identidade visual Artefact aplicada ao Streamlit."""

from __future__ import annotations

import streamlit as st

from .config import APP, BRAND


def inject_css() -> None:
    """Injeta a folha de estilo da aplicação (chamar uma vez, no topo)."""
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
        <style>
          :root {{
            --navy: {BRAND.navy};
            --pink: {BRAND.pink};
            --ink: {BRAND.ink};
            --muted: {BRAND.muted};
            --line: {BRAND.line};
            --surface: {BRAND.surface};
            --canvas: {BRAND.canvas};
          }}

          html, body, [class*="css"], .stApp {{
            font-family: {BRAND.font};
            color: var(--ink);
          }}
          .stApp {{ background: var(--canvas); }}
          #MainMenu, footer {{ visibility: hidden; }}
          .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }}

          /* ---------- Cabeçalho ---------- */
          .af-hero {{
            background: linear-gradient(115deg, var(--navy) 0%, #2A2A6E 55%, var(--pink) 160%);
            border-radius: 16px;
            padding: 30px 34px;
            color: #fff;
            margin-bottom: 26px;
          }}
          .af-hero h1 {{
            font-size: 1.85rem; font-weight: 700; margin: 0 0 6px 0;
            letter-spacing: -0.02em; color: #fff;
          }}
          .af-hero p {{ margin: 0; font-size: 0.98rem; opacity: 0.82; font-weight: 300; }}
          .af-tag {{
            display: inline-block; font-size: 0.68rem; font-weight: 700;
            letter-spacing: 0.16em; text-transform: uppercase;
            color: var(--pink); margin-bottom: 12px;
          }}

          /* ---------- Cartões de KPI ---------- */
          .af-kpi {{
            background: var(--surface); border: 1px solid var(--line);
            border-radius: 12px; padding: 18px 20px; height: 100%;
            border-top: 3px solid var(--pink);
          }}
          .af-kpi .label {{
            font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
            text-transform: uppercase; color: var(--muted); margin-bottom: 6px;
          }}
          .af-kpi .value {{
            font-size: 1.55rem; font-weight: 700; color: var(--navy);
            line-height: 1.15; letter-spacing: -0.02em;
          }}
          .af-kpi .caption {{ font-size: 0.76rem; color: var(--muted); margin-top: 4px; }}

          /* ---------- Seções ---------- */
          .af-section {{
            font-size: 1.05rem; font-weight: 700; color: var(--navy);
            margin: 30px 0 4px 0; padding-left: 11px;
            border-left: 3px solid var(--pink);
          }}
          .af-section-sub {{
            font-size: 0.84rem; color: var(--muted);
            margin: 0 0 14px 14px; font-weight: 300;
          }}

          .af-card {{
            background: var(--surface); border: 1px solid var(--line);
            border-radius: 12px; padding: 20px 22px;
          }}

          /* ---------- Componentes Streamlit ---------- */
          .stButton > button, .stDownloadButton > button {{
            background: var(--navy); color: #fff; border: none;
            border-radius: 8px; padding: 0.5rem 1.15rem;
            font-weight: 500; transition: background 0.15s ease;
          }}
          .stButton > button:hover, .stDownloadButton > button:hover {{
            background: var(--pink); color: #fff;
          }}
          [data-testid="stSidebar"] {{
            background: var(--surface); border-right: 1px solid var(--line);
          }}
          [data-testid="stSidebar"] h2 {{
            font-size: 0.98rem; color: var(--navy); font-weight: 700;
          }}
          [data-testid="stFileUploaderDropzone"] {{
            border: 1.5px dashed var(--line); border-radius: 12px;
            background: var(--surface);
          }}
          .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--line); }}
          .stTabs [data-baseweb="tab"] {{ font-weight: 500; color: var(--muted); }}
          .stTabs [aria-selected="true"] {{ color: var(--navy); }}
          .stTabs [data-baseweb="tab-highlight"] {{ background: var(--pink); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    """Cabeçalho da aplicação."""
    st.markdown(
        f"""
        <div class="af-hero">
          <div class="af-tag">{APP.owner}</div>
          <h1>{APP.title}</h1>
          <p>{APP.subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    """Título de seção com a régua rosa da marca."""
    st.markdown(f'<div class="af-section">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="af-section-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, caption: str = "") -> None:
    """Cartão de indicador. Usar dentro de uma coluna."""
    st.markdown(
        f"""
        <div class="af-kpi">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
