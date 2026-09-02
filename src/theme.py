"""Identidade visual Artefact aplicada ao Streamlit.

IMPORTANTE: o Streamlit passa o conteúdo de `st.markdown` por um parser
CommonMark antes de renderizar o HTML. Duas armadilhas derrubam o CSS:

1. Linhas indentadas com 4+ espaços viram bloco de código e o CSS aparece
   como texto na tela.
2. Uma linha em branco encerra o bloco HTML, e o resto vaza como parágrafo.

Por isso as strings deste módulo são montadas sem indentação e sem linhas
em branco, e a fonte é carregada com `@import` dentro do próprio `<style>`
(em vez de uma tag `<link>` separada, que abriria um bloco HTML frágil).
"""

from __future__ import annotations

import streamlit as st

from .config import APP, BRAND

_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2"
    "?family=Roboto:wght@300;400;500;700&display=swap');"
)


def _css() -> str:
    """Monta a folha de estilo como linhas isoladas, sem indentação."""
    rules = [
        # ---------- Tokens ----------
        f":root {{ --pink: {BRAND.pink}; --ink: {BRAND.ink}; --ink-strong: {BRAND.ink_strong};"
        f" --muted: {BRAND.muted}; --line: {BRAND.line}; --surface: {BRAND.surface};"
        f" --surface-alt: {BRAND.surface_alt}; --canvas: {BRAND.canvas}; }}",
        # ---------- Base ----------
        f"html, body, [class*='css'], .stApp {{ font-family: {BRAND.font}; color: var(--ink); }}",
        ".stApp { background: var(--canvas); }",
        "#MainMenu, footer { visibility: hidden; }",
        ".block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }",
        "h1, h2, h3, h4, h5, h6 { color: var(--ink-strong); }",
        "p, span, label, li, div { color: inherit; }",
        # ---------- Cabeçalho ----------
        f".af-hero {{ background: linear-gradient(115deg, {BRAND.navy} 0%, {BRAND.navy_soft} 45%,"
        f" #3A2A63 75%, {BRAND.pink} 190%); border: 1px solid var(--line); border-radius: 16px;"
        " padding: 30px 34px; margin-bottom: 26px; }",
        ".af-hero h1 { font-size: 1.85rem; font-weight: 700; margin: 0 0 6px 0;"
        " letter-spacing: -0.02em; color: #fff; }",
        ".af-hero p { margin: 0; font-size: 0.98rem; color: #C9C9E0; font-weight: 300; }",
        ".af-tag { display: inline-block; font-size: 0.68rem; font-weight: 700;"
        " letter-spacing: 0.16em; text-transform: uppercase; color: var(--pink); margin-bottom: 12px; }",
        # ---------- KPIs ----------
        ".af-kpi { background: var(--surface); border: 1px solid var(--line); border-radius: 12px;"
        " padding: 18px 20px; height: 100%; border-top: 3px solid var(--pink); }",
        ".af-kpi .label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;"
        " text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }",
        ".af-kpi .value { font-size: 1.55rem; font-weight: 700; color: var(--ink-strong);"
        " line-height: 1.15; letter-spacing: -0.02em; }",
        ".af-kpi .caption { font-size: 0.76rem; color: var(--muted); margin-top: 4px; }",
        # ---------- Seções ----------
        ".af-section { font-size: 1.05rem; font-weight: 700; color: var(--ink-strong);"
        " margin: 30px 0 4px 0; padding-left: 11px; border-left: 3px solid var(--pink); }",
        ".af-section-sub { font-size: 0.84rem; color: var(--muted);"
        " margin: 0 0 14px 14px; font-weight: 300; }",
        # ---------- Sidebar: a aplicação não usa barra lateral ----------
        "[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'],"
        " [data-testid='collapsedControl'] { display: none !important; }",
        "[data-testid='stCaptionContainer'], [data-testid='stCaptionContainer'] p"
        " { color: var(--muted) !important; }",
        # ---------- Campos de formulário (a origem do texto ilegível) ----------
        "[data-baseweb='select'] > div { background: var(--surface-alt) !important;"
        " border-color: var(--line) !important; color: var(--ink) !important; }",
        "[data-baseweb='select'] div, [data-baseweb='select'] span"
        " { color: var(--ink) !important; }",
        "[data-baseweb='select'] svg { fill: var(--muted); }",
        "[data-baseweb='popover'] [role='listbox'] { background: var(--surface-alt) !important;"
        " border: 1px solid var(--line); }",
        "[data-baseweb='popover'] li { color: var(--ink) !important; }",
        "[data-baseweb='popover'] li:hover { background: var(--line) !important; }",
        "input, textarea { background: var(--surface-alt) !important; color: var(--ink) !important; }",
        "[data-testid='stWidgetLabel'] p { color: var(--ink) !important; font-weight: 500; }",
        "[data-baseweb='checkbox'] div { background: var(--surface-alt); }",
        "[data-testid='stSliderTickBarMin'], [data-testid='stSliderTickBarMax']"
        " { color: var(--muted) !important; }",
        "[data-testid='stToggle'] label p { color: var(--ink) !important; }",
        # ---------- Uploader ----------
        "[data-testid='stFileUploaderDropzone'] { border: 1.5px dashed var(--line);"
        " border-radius: 12px; background: var(--surface-alt); }",
        "[data-testid='stFileUploaderDropzone'] * { color: var(--ink) !important; }",
        "[data-testid='stFileUploaderDropzone'] small { color: var(--muted) !important; }",
        # ---------- Botões ----------
        ".stButton > button, .stDownloadButton > button { background: var(--pink); color: #16162E;"
        " border: none; border-radius: 8px; padding: 0.5rem 1.15rem; font-weight: 600; }",
        ".stButton > button:hover, .stDownloadButton > button:hover"
        " { background: #FF7FA3; color: #16162E; }",
        "[data-testid='stFileUploaderDropzone'] button { background: transparent;"
        " color: var(--ink) !important; border: 1px solid var(--line); }",
        # ---------- Abas ----------
        ".stTabs [data-baseweb='tab-list'] { gap: 4px; border-bottom: 1px solid var(--line); }",
        ".stTabs [data-baseweb='tab'] { font-weight: 500; color: var(--muted); }",
        ".stTabs [aria-selected='true'] { color: var(--ink-strong); }",
        ".stTabs [data-baseweb='tab-highlight'] { background: var(--pink); }",
        # ---------- Tabelas e métricas ----------
        "[data-testid='stDataFrame'] { border: 1px solid var(--line); border-radius: 10px; }",
        "[data-testid='stMetricValue'] { color: var(--ink-strong); }",
        "[data-testid='stMetricLabel'] p { color: var(--muted) !important; }",
        "[data-testid='stExpander'] { border: 1px solid var(--line); border-radius: 10px;"
        " background: var(--surface); }",
        "[data-testid='stExpander'] summary { color: var(--ink) !important; }",
    ]
    return "<style>\n" + _FONT_IMPORT + "\n" + "\n".join(rules) + "\n</style>"


def inject_css() -> None:
    """Injeta a folha de estilo da aplicação (chamar uma vez, no topo)."""
    st.markdown(_css(), unsafe_allow_html=True)


def hero() -> None:
    """Cabeçalho da aplicação."""
    st.markdown(
        f'<div class="af-hero">'
        f'<div class="af-tag">{APP.owner}</div>'
        f"<h1>{APP.title}</h1>"
        f"<p>{APP.subtitle}</p>"
        f"</div>",
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
        f'<div class="af-kpi">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="caption">{caption}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
