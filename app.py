# -*- coding: utf-8 -*-
"""
Tanager-1 Physiology & Management Dashboard  ·  FABLE5 redesign
================================================================
Hyperspectral (426 bands) lot-scale physiology, biochemistry, uncertainty,
red-edge metrics and management zoning over Mato Grosso, Brazil.

Deploy: gunicorn app:server   ·   Data: ./data   ·   Map: tokenless carto basemap
"""
from __future__ import annotations

import json
from pathlib import Path

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html

# ----------------------------------------------------------------------
# Paths & data
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IMG = "assets/img"


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


with open(DATA / "lotes_dashboard.geojson", encoding="utf-8") as f:
    GEOJSON = json.load(f)

df_idx = _read("06_estadisticas_por_lote.csv")
df_bio = _read("07_bioquimica_por_lote.csv")
df_spec = pd.read_csv(DATA / "09_perfiles_todos_lotes.csv", index_col="id_lote")
df_pca = _read("09_pca_scores.csv")
df_vip = _read("10_vip_scores.csv")
df_het = _read("12_heterogeneidad_lotes.csv")
df_reip = _read("13_reip_por_lote.csv")
df_clust = _read("14_clustering_lotes.csv").rename(columns={"Unnamed: 0": "id_lote"})
df_unc = _read("15_incertidumbre_por_lote.csv")
df_red = _read("16_rededge_metricas_por_lote.csv")
df_anom = _read("17_anomalias_lotes.csv")
df_sub = _read("18_subzonas_por_lote.csv")
df_exp = _read("19_explicacion_lotes.csv")

wl_nm = df_spec.columns.astype(float).values
lotes = sorted(df_spec.index.tolist())


def _read_md(local: str, fallback: Path) -> str:
    p = DATA / local
    if p.exists():
        return p.read_text(encoding="utf-8")
    if fallback.exists():
        return fallback.read_text(encoding="utf-8")
    return "_Document not bundled with deployment._"


brief_text = _read_md("brief.md", ROOT.parent / "TECHNICAL_SCIENTIFIC_BRIEF.md")
comp_text = _read_md("competition.md", ROOT.parent / "OUTPUTS" / "20_competition_summary.md")

# ----------------------------------------------------------------------
# Master table (one row per lot)
# ----------------------------------------------------------------------
df_master = (
    df_idx[["id_lote", "NDRE_mean", "CIre_mean", "PRI_mean", "WBI_mean", "NDVI_mean", "REIP_mean", "NDWI_mean"]]
    .merge(df_bio[["id_lote", "Cab_est_ugcm2", "N_foliar_rel", "H2O_foliar_rel",
                   "Efic_fotosint_rel", "Biomasa_rel", "Estres_car_rel"]], on="id_lote", how="left")
    .merge(df_het[["id_lote", "hetero_score", "NDRE_cv", "WBI_cv", "REIP_cv"]], on="id_lote", how="left")
    .merge(df_reip[["id_lote", "REIP_range"]], on="id_lote", how="left")
    .merge(df_pca[["id_lote", "PC1", "PC2", "PC3"]], on="id_lote", how="left")
    .merge(df_clust[["id_lote", "cluster_name"]], on="id_lote", how="left")
    .merge(df_unc[["id_lote", "uncertainty_score", "unc_visible_mean", "unc_rededge_mean",
                   "unc_nir_mean", "unc_swir_mean", "su_rededge_proxy", "su_swir_proxy"]], on="id_lote", how="left")
    .merge(df_red[["id_lote", "re_slope_max", "re_slope_wl_nm", "re_area_680_760", "re_contrast_750_680"]], on="id_lote", how="left")
    .merge(df_anom[["id_lote", "anomaly_score", "anomaly_class", "anomaly_flags"]], on="id_lote", how="left")
    .merge(df_sub[["id_lote", "subzone_critica_pct", "subzone_transicion_pct", "subzone_alta_pct",
                   "score_mean", "dominant_subzone"]], on="id_lote", how="left")
    .merge(df_exp[["id_lote", "interpretation_short", "management_recommendation"]], on="id_lote", how="left")
)


def norm01(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else s * 0.0


# Composite management-priority index (higher = needs more attention)
_attn = pd.concat([
    1 - norm01(df_master["Cab_est_ugcm2"]),
    1 - norm01(df_master["N_foliar_rel"]),
    1 - norm01(df_master["H2O_foliar_rel"]),
    norm01(df_master["subzone_critica_pct"]),
    norm01(df_master["anomaly_score"]),
    norm01(df_master["hetero_score"]),
], axis=1)
df_master["priority_index"] = (100 * _attn.mean(axis=1)).round(1)

# ----------------------------------------------------------------------
# Variable registries
# ----------------------------------------------------------------------
# label -> column (numeric, lot-level)
NUM_VARS = {
    "NDRE": "NDRE_mean", "CIre": "CIre_mean", "PRI": "PRI_mean", "WBI": "WBI_mean",
    "NDVI": "NDVI_mean", "NDWI": "NDWI_mean", "REIP (nm)": "REIP_mean",
    "Cab estimated (µg/cm²)": "Cab_est_ugcm2", "N relative": "N_foliar_rel",
    "Water relative": "H2O_foliar_rel", "Photo-efficiency": "Efic_fotosint_rel",
    "Biomass relative": "Biomasa_rel", "Low-stress (carot.)": "Estres_car_rel",
    "Heterogeneity": "hetero_score", "REIP range": "REIP_range",
    "Red-edge slope": "re_slope_max", "Red-edge area": "re_area_680_760",
    "Uncertainty": "uncertainty_score", "Anomaly score": "anomaly_score",
    "Critical subzone %": "subzone_critica_pct", "Priority index": "priority_index",
    "PC1": "PC1", "PC2": "PC2", "PC3": "PC3",
}
COL2LABEL = {v: k for k, v in NUM_VARS.items()}
# variables where HIGH value means worse / more attention -> reversed colour scale
REVERSE = {"uncertainty_score", "anomaly_score", "hetero_score", "subzone_critica_pct",
           "REIP_range", "NDRE_cv", "WBI_cv", "REIP_cv", "priority_index",
           "unc_visible_mean", "unc_rededge_mean", "unc_nir_mean", "unc_swir_mean"}

ZONE_COLORS = {"Zona A": "#37e2b0", "Zona B": "#ff8c6b", "Zona C": "#6ec1ff", "Zona D": "#f4a261"}
SPECTRAL_REGIONS = [
    (376, 500, "#4f8fdf", "Blue"), (500, 680, "#37c98f", "Green"),
    (680, 750, "#ff8c6b", "Red-edge"), (750, 1300, "#9fb3c8", "NIR"),
    (1300, 1800, "#f4a261", "SWIR-1"), (1800, 2499, "#b07a55", "SWIR-2"),
]
WATER_BANDS = [(1340, 1460), (1790, 1970)]

# theme tokens (mirrors style.css)
C_BG, C_BG2, C_CARD, C_BORDER = "#0b1018", "#0f1622", "#141d2b", "#26313f"
C_TEXT, C_MUTED, C_ACCENT, C_ACCENT2 = "#eaf1f8", "#93a6bd", "#6ec1ff", "#37e2b0"

app = dash.Dash(__name__, suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
app.title = "Tanager-1 Lab · MR"
server = app.server

# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def fmt(v, nd=3):
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{nd}f}"
    return str(v)


def dark(fig, title=None, h=None, legend=True):
    fig.update_layout(
        paper_bgcolor=C_CARD, plot_bgcolor=C_BG2, font=dict(color=C_TEXT, family="Inter"),
        margin=dict(t=46 if title else 18, b=40, l=20, r=20),
        title=dict(text=title, font=dict(size=14, color=C_ACCENT)) if title else None,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER) if legend else None,
        colorway=px.colors.qualitative.Set2,
    )
    fig.update_xaxes(gridcolor=C_BORDER, zerolinecolor=C_BORDER)
    fig.update_yaxes(gridcolor=C_BORDER, zerolinecolor=C_BORDER)
    if h:
        fig.update_layout(height=h)
    return fig


def card(*children, **style):
    return html.Div(className="card", style=style or None, children=list(children))


def img_card(title, name):
    return html.Div(className="card", children=[html.H4(title), html.Img(src=f"/{IMG}/{name}")])


def row(*children, gap="16px"):
    return html.Div(style={"display": "flex", "gap": gap, "flexWrap": "wrap"}, children=list(children))


def dd(_id, options, value, multi=False):
    return dcc.Dropdown(id=_id, options=options, value=value, multi=multi, clearable=False)


def var_options():
    return [{"label": k, "value": v} for k, v in NUM_VARS.items()]


def priority_chip(v):
    cls = "chip-red" if v >= 66 else "chip-amber" if v >= 33 else "chip-green"
    return html.Span(f"{v:.0f}", className=f"chip {cls}")


def data_bars(df, column, color=C_ACCENT):
    """Conditional in-cell bar background for a DataTable numeric column."""
    styles = []
    col = df[column].astype(float)
    lo, hi = col.min(), col.max()
    if not np.isfinite(lo) or hi == lo:
        return styles
    for i in range(1, 101):
        bound_lo = lo + (i - 1) / 100 * (hi - lo)
        bound_hi = lo + i / 100 * (hi - lo)
        pct = i
        styles.append({
            "if": {"filter_query": f"{{{column}}} >= {bound_lo}" + (f" && {{{column}}} < {bound_hi}" if i < 100 else ""),
                   "column_id": column},
            "background": (f"linear-gradient(90deg, {color}33 0%, {color}33 {pct}%, "
                           f"transparent {pct}%, transparent 100%)"),
        })
    return styles


ODD_ROW = [{"if": {"row_index": "odd"}, "backgroundColor": "#101825"}]
TABLE_KW = dict(
    page_size=14, sort_action="native", filter_action="native",
    style_table={"overflowX": "auto"},
    style_cell={"backgroundColor": C_CARD, "color": C_TEXT, "border": f"1px solid {C_BORDER}",
                "fontSize": "12px", "fontFamily": "JetBrains Mono, monospace", "padding": "6px 10px"},
    style_header={"backgroundColor": C_BG2, "color": C_ACCENT, "fontWeight": "700", "border": f"1px solid {C_BORDER}"},
)

# ----------------------------------------------------------------------
# Header + KPIs (dynamic)
# ----------------------------------------------------------------------
n_anom = int((df_master["anomaly_class"].isin(["ATIPICO", "MUY_ATIPICO"])).sum())
n_high_prio = int((df_master["priority_index"] >= 66).sum())
kpis = [
    ("Lots analysed", f"{len(df_master)}", "66 agricultural lots"),
    ("Spectral bands", "426", "376–2499 nm · 30 m"),
    ("Mean Cab", f"{df_master['Cab_est_ugcm2'].mean():.1f}", "µg/cm² (relative proxy)"),
    ("High-priority lots", f"{n_high_prio}", "priority index ≥ 66"),
    ("Atypical lots", f"{n_anom}", "anomaly flagged"),
    ("Management zones", f"{df_master['cluster_name'].nunique()}", "k-means clusters"),
]

header = html.Div(className="app-header", children=[
    html.H1("Tanager-1 · Physiology & Management Intelligence"),
    html.P("Hyperspectral lot-scale inference over Mato Grosso, Brazil — pigment, function, water & structure "
           "from a single VNIR–SWIR acquisition.", className="sub"),
    html.Div(className="app-badges", children=[
        html.Span("Tanager-1", className="badge on"), html.Span("426 bands", className="badge"),
        html.Span("30 m", className="badge"), html.Span("EPSG:32721", className="badge"),
        html.Span("2025-05-01", className="badge"), html.Span("physiology-informed", className="badge"),
    ]),
])

kpi_strip = html.Div(className="kpi-row", children=[
    html.Div(className="kpi", children=[
        html.P(label, className="label"), html.P(value, className="value"), html.P(sub, className="delta")
    ]) for label, value, sub in kpis
])

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
app.layout = html.Div(style={"backgroundColor": C_BG, "minHeight": "100vh"}, children=[
    header,
    kpi_strip,
    dcc.Download(id="dl-master"),
    dcc.Tabs(className="dash-tabs", children=[

        # ---- TAB 1 · Overview map ----
        dcc.Tab(label="🗺 Overview", className="tab", selected_className="tab--selected", children=[
            html.Div(style={"padding": "16px 28px"}, children=[
                row(
                    card(
                        row(
                            html.Div(style={"flex": "2", "minWidth": "240px"}, children=[
                                html.Label("Variable", style={"fontSize": "12px", "color": C_MUTED}),
                                dd("map-var", var_options(), "NDRE_mean")]),
                            html.Div(style={"flex": "1", "minWidth": "160px"}, children=[
                                html.Label("Basemap", style={"fontSize": "12px", "color": C_MUTED}),
                                dd("map-base", [{"label": "Dark", "value": "carto-darkmatter"},
                                                {"label": "Light", "value": "carto-positron"},
                                                {"label": "Streets", "value": "open-street-map"}],
                                   "carto-darkmatter")]),
                        ),
                        dcc.Graph(id="map-fig", style={"height": "560px"}, config={"displaylogo": False}),
                        **{"flex": "3", "minWidth": "440px"},
                    ),
                    card(html.H4("Lot inspector"), html.Div(id="lot-panel"),
                         **{"flex": "1", "minWidth": "300px"}),
                ),
                card(html.H4("Management-priority leaderboard (top 15)"),
                     dcc.Graph(id="prio-bar", style={"height": "420px"}, config={"displaylogo": False}),
                     html.P("Composite of low chlorophyll/N/water + high anomaly, heterogeneity and critical-subzone "
                            "share. A scouting shortlist, not a verdict.", className="section-note")),
            ])
        ]),

        # ---- TAB 2 · Spectral explorer ----
        dcc.Tab(label="📈 Spectral", className="tab", selected_className="tab--selected", children=[
            html.Div(style={"padding": "16px 28px"}, children=[
                row(
                    card(
                        html.Label("Lots (max 6)", style={"fontSize": "12px", "color": C_MUTED}),
                        dd("spec-lotes", [{"label": l, "value": l} for l in lotes],
                           ["A23", "A56", "A65"], multi=True),
                        html.Div(style={"marginTop": "10px"}, children=[
                            dcc.RadioItems(id="spec-mode", inline=True,
                                options=[{"label": " Reflectance", "value": "raw"},
                                         {"label": " Min–max normalised", "value": "norm"},
                                         {"label": " 1st derivative", "value": "deriv"}],
                                value="raw", labelStyle={"marginRight": "18px", "color": C_TEXT})]),
                        dcc.Graph(id="spec-fig", style={"height": "520px"}, config={"displaylogo": False}),
                        **{"flex": "2", "minWidth": "440px"},
                    ),
                    card(html.H4("Red-edge zoom (680–760 nm)"),
                         dcc.Graph(id="spec-re", style={"height": "300px"}, config={"displaylogo": False}),
                         html.Img(src=f"/{IMG}/08_diferencia_espectral.png", style={"marginTop": "10px"}),
                         **{"flex": "1", "minWidth": "320px"}),
                ),
            ])
        ]),

        # ---- TAB 3 · Biochemistry ----
        dcc.Tab(label="🧪 Biochemistry", className="tab", selected_className="tab--selected", children=[
            html.Div(style={"padding": "16px 28px"}, children=[
                row(
                    card(html.H4("Compare two lots"),
                         row(html.Div(style={"flex": "1"}, children=[
                                html.Label("Lot A", style={"fontSize": "12px", "color": C_MUTED}),
                                dd("bio-a", [{"label": l, "value": l} for l in lotes], "A23")]),
                             html.Div(style={"flex": "1"}, children=[
                                html.Label("Lot B", style={"fontSize": "12px", "color": C_MUTED}),
                                dd("bio-b", [{"label": l, "value": l} for l in lotes], "A65")]), gap="10px"),
                         dcc.Graph(id="bio-radar", style={"height": "430px"}, config={"displaylogo": False}),
                         **{"flex": "1", "minWidth": "360px"}),
                    card(html.H4("Biochemical state heatmap"),
                         html.Img(src=f"/{IMG}/07_heatmap_bioquimica.png"),
                         **{"flex": "1", "minWidth": "360px"}),
                ),
                card(html.H4("Biochemical table"),
                     dash_table.DataTable(
                         id="bio-table",
                         columns=[{"name": c, "id": c} for c in
                                  ["id_lote", "Cab_est_ugcm2", "N_foliar_rel", "H2O_foliar_rel",
                                   "Efic_fotosint_rel", "Biomasa_rel", "Estres_car_rel"]],
                         data=df_bio.round(2).to_dict("records"),
                         style_data_conditional=(
                             ODD_ROW +
                             data_bars(df_bio, "Cab_est_ugcm2", C_ACCENT2) +
                             data_bars(df_bio, "N_foliar_rel", C_ACCENT) +
                             data_bars(df_bio, "H2O_foliar_rel", "#6ec1ff")),
                         **TABLE_KW)),
            ])
        ]),

        # ---- TAB 4 · Relationships ----
        dcc.Tab(label="🔗 Relationships", className="tab", selected_className="tab--selected", children=[
            html.Div(style={"padding": "16px 28px"}, children=[
                row(
                    card(html.H4("Bivariate explorer"),
                         row(html.Div(style={"flex": "1"}, children=[html.Label("X", style={"fontSize": "12px", "color": C_MUTED}), dd("xy-x", var_options(), "N_foliar_rel")]),
                             html.Div(style={"flex": "1"}, children=[html.Label("Y", style={"fontSize": "12px", "color": C_MUTED}), dd("xy-y", var_options(), "Cab_est_ugcm2")]), gap="10px"),
                         row(html.Div(style={"flex": "1"}, children=[html.Label("Colour", style={"fontSize": "12px", "color": C_MUTED}), dd("xy-c", var_options(), "uncertainty_score")]),
                             html.Div(style={"flex": "1"}, children=[html.Label("Size", style={"fontSize": "12px", "color": C_MUTED}),
                                 dd("xy-s", [{"label": "— none —", "value": "none"}] + var_options(), "subzone_critica_pct")]), gap="10px"),
                         dcc.Graph(id="xy-fig", style={"height": "470px"}, config={"displaylogo": False}),
                         **{"flex": "3", "minWidth": "460px"}),
                    card(html.H4("Correlation matrix"),
                         dcc.Graph(id="corr-fig", style={"height": "560px"}, config={"displaylogo": False}),
                         **{"flex": "2", "minWidth": "360px"}),
                ),
            ])
        ]),

        # ---- TAB 5 · Rankings & distributions ----
        dcc.Tab(label="📊 Rankings", className="tab", selected_className="tab--selected", children=[
            html.Div(style={"padding": "16px 28px"}, children=[
                card(html.H4("Ranking & distribution"),
                     row(html.Div(style={"flex": "1", "minWidth": "220px"}, children=[
                            html.Label("Variable", style={"fontSize": "12px", "color": C_MUTED}),
                            dd("rank-var", var_options(), "Cab_est_ugcm2")]),
                         html.Div(style={"flex": "1", "minWidth": "220px"}, children=[
                            html.Label("Highlight lot", style={"fontSize": "12px", "color": C_MUTED}),
                            dd("rank-lote", [{"label": l, "value": l} for l in lotes], "A23")]), gap="12px"),
                     row(dcc.Graph(id="rank-bar", style={"height": "560px", "flex": "2", "minWidth": "420px"}, config={"displaylogo": False}),
                         dcc.Graph(id="rank-hist", style={"height": "560px", "flex": "1", "minWidth": "300px"}, config={"displaylogo": False}))),
            ])
        ]),

        # ---- TAB 6 · PCA & zoning ----
        dcc.Tab(label="🧬 PCA & Zoning", className="tab", selected_className="tab--selected", children=[
            html.Div(style={"padding": "16px 28px"}, children=[
                row(
                    card(html.Label("Colour by", style={"fontSize": "12px", "color": C_MUTED}),
                         dcc.RadioItems(id="pca-color", inline=True, value="cluster_name",
                            options=[{"label": " Cluster", "value": "cluster_name"}, {"label": " NDRE", "value": "NDRE_mean"},
                                     {"label": " Cab", "value": "Cab_est_ugcm2"}, {"label": " Anomaly", "value": "anomaly_score"},
                                     {"label": " Priority", "value": "priority_index"}],
                            labelStyle={"marginRight": "16px", "color": C_TEXT}),
                         dcc.Graph(id="pca-fig", style={"height": "470px"}, config={"displaylogo": False}),
                         **{"flex": "1", "minWidth": "440px"}),
                    img_card("PCA & clustering fingerprint", "14_fingerprint_clustering.png"),
                ),
                img_card("PCA loadings by spectral region", "09_pca_espectral.png"),
            ])
        ]),

        # ---- TAB 7 · PLSR & VIP ----
        dcc.Tab(label="🎯 PLSR & VIP", className="tab", selected_className="tab--selected", children=[
            html.Div(style={"padding": "16px 28px"}, children=[
                row(
                    card(html.Label("VIP variable", style={"fontSize": "12px", "color": C_MUTED}),
                         dd("vip-var", [{"label": c.replace("VIP_", ""), "value": c}
                                        for c in df_vip.columns if c.startswith("VIP_")],
                            next(c for c in df_vip.columns if c.startswith("VIP_"))),
                         dcc.Graph(id="vip-fig", style={"height": "440px"}, config={"displaylogo": False}),
                         **{"flex": "1", "minWidth": "440px"}),
                    img_card("PLSR overview", "10_plsr_bioquimica.png"),
                ),
                img_card("Pixel-level PLSR map", "10_mapa_plsr_pixeles.png"),
            ])
        ]),

        # ---- TAB 8 · Spatial management ----
        dcc.Tab(label="🛰 Spatial", className="tab", selected_className="tab--selected", children=[
            html.Div(style={"padding": "16px 28px"}, children=[
                row(img_card("REIP gradient (N proxy)", "13_reip_gradiente_N.png"),
                    img_card("Heterogeneity", "12_heterogeneidad_lotes.png")),
                row(img_card("Management subzones", "18_subzonas_manejo.png"),
                    img_card("RGB & lot quality", "03_RGB_calidad_lotes.png")),
            ])
        ]),

        # ---- TAB 9 · Quality & uncertainty ----
        dcc.Tab(label="🔬 Quality", className="tab", selected_className="tab--selected", children=[
            html.Div(style={"padding": "16px 28px"}, children=[
                row(card(html.H4("Uncertainty score by lot"),
                         dcc.Graph(id="unc-bar", style={"height": "470px"}, config={"displaylogo": False}),
                         **{"flex": "1", "minWidth": "420px"}),
                    card(html.H4("Red-edge vs SWIR uncertainty"),
                         dcc.Graph(id="unc-scatter", style={"height": "470px"}, config={"displaylogo": False}),
                         **{"flex": "1", "minWidth": "420px"})),
                card(html.H4("Uncertainty table"),
                     dash_table.DataTable(
                         columns=[{"name": c, "id": c} for c in
                                  ["id_lote", "uncertainty_score", "uncertainty_class", "unc_rededge_mean",
                                   "unc_swir_mean", "su_rededge_proxy", "su_swir_proxy"]],
                         data=df_unc.round(5).to_dict("records"), style_data_conditional=ODD_ROW, **TABLE_KW)),
            ])
        ]),

        # ---- TAB 10 · Priority & brief ----
        dcc.Tab(label="📋 Priority & Brief", className="tab", selected_className="tab--selected", children=[
            html.Div(style={"padding": "16px 28px"}, children=[
                card(row(html.H4("Field-scouting priority table", style={"flex": "1"}),
                         html.Button("⬇ Download master CSV", id="btn-dl", className="btn-dl")),
                     dash_table.DataTable(
                         id="prio-table",
                         columns=[{"name": n, "id": i} for n, i in [
                             ("Lot", "id_lote"), ("Priority", "priority_index"), ("Cab", "Cab_est_ugcm2"),
                             ("N rel", "N_foliar_rel"), ("Water rel", "H2O_foliar_rel"),
                             ("Crit. subzone %", "subzone_critica_pct"), ("Anomaly", "anomaly_class"),
                             ("Zone", "cluster_name"), ("Dominant subzone", "dominant_subzone")]],
                         data=df_master.sort_values("priority_index", ascending=False).round(2).to_dict("records"),
                         style_data_conditional=(
                             ODD_ROW +
                             data_bars(df_master, "priority_index", "#ff8c6b") +
                             [{"if": {"filter_query": "{anomaly_class} = 'MUY_ATIPICO'", "column_id": "anomaly_class"},
                               "color": "#ff6b6b", "fontWeight": "700"},
                              {"if": {"filter_query": "{anomaly_class} = 'ATIPICO'", "column_id": "anomaly_class"},
                               "color": "#ffb454"}]),
                         **TABLE_KW)),
                row(
                    card(html.Label("Lot", style={"fontSize": "12px", "color": C_MUTED}),
                         dd("explain-lote", [{"label": l, "value": l} for l in lotes], "A23"),
                         html.Div(id="explain-panel", style={"marginTop": "14px"}),
                         **{"flex": "1", "minWidth": "320px"}),
                    card(html.H4("Technical scientific brief"), dcc.Markdown(brief_text),
                         html.Hr(style={"borderColor": C_BORDER}),
                         html.H4("Competition summary"), dcc.Markdown(comp_text),
                         **{"flex": "2", "minWidth": "480px"}),
                ),
            ])
        ]),
    ]),
    html.Div(style={"padding": "18px 34px", "color": C_MUTED, "fontSize": "12px",
                    "borderTop": f"1px solid {C_BORDER}"},
             children="Physiology-informed inference · proxies are scene-relative unless field-calibrated · "
                      "built with Dash & Plotly · Tanager-1 Open Data."),
])


# ======================================================================
# Callbacks
# ======================================================================
@app.callback(Output("map-fig", "figure"), Output("lot-panel", "children"),
              Input("map-var", "value"), Input("map-base", "value"), Input("map-fig", "clickData"))
def update_map(var_col, basemap, click_data):
    label = COL2LABEL.get(var_col, var_col)
    fig = px.choropleth_mapbox(
        df_master, geojson=GEOJSON, locations="id_lote", featureidkey="properties.id_lote",
        color=var_col, color_continuous_scale="RdYlGn_r" if var_col in REVERSE else "RdYlGn",
        mapbox_style=basemap, center={"lat": -15.45, "lon": -55.02}, zoom=10, opacity=0.82,
        hover_name="id_lote",
        hover_data={"cluster_name": True, "anomaly_class": True, "priority_index": ":.0f",
                    "subzone_critica_pct": ":.1f", var_col: ":.3f"})
    fig.update_layout(paper_bgcolor=C_CARD, font_color=C_TEXT, margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      coloraxis_colorbar=dict(title=label, thickness=12, len=0.8))
    lot_id = "A23"
    if click_data and click_data.get("points"):
        lot_id = click_data["points"][0].get("location", "A23")
    r = df_master[df_master["id_lote"] == lot_id].iloc[0]

    def kv(k, v):
        return html.Div(className="kv", children=[html.Span(k, className="k"), html.Span(v, className="v")])

    a_cls = {"MUY_ATIPICO": "chip-red", "ATIPICO": "chip-amber"}.get(r["anomaly_class"], "chip-green")
    panel = html.Div(children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px"}, children=[
            html.H4(lot_id, style={"margin": 0, "color": C_ACCENT}),
            html.Span(r["anomaly_class"], className=f"chip {a_cls}"),
            html.Span(r["cluster_name"], className="chip chip-blue")]),
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "8px", "margin": "8px 0"},
                 children=[html.Span("Priority", className="k"), priority_chip(r["priority_index"])]),
        html.Hr(style={"borderColor": C_BORDER}),
        kv("NDRE", fmt(r["NDRE_mean"])), kv("REIP", f"{fmt(r['REIP_mean'], 1)} nm"),
        kv("Cab (µg/cm²)", fmt(r["Cab_est_ugcm2"], 1)), kv("N relative", fmt(r["N_foliar_rel"], 1)),
        kv("Water relative", fmt(r["H2O_foliar_rel"], 1)), kv("Biomass rel.", fmt(r["Biomasa_rel"], 1)),
        kv("Heterogeneity", fmt(r["hetero_score"], 1)), kv("Uncertainty", fmt(r["uncertainty_score"], 4)),
        kv("Critical subzone", f"{fmt(r['subzone_critica_pct'], 1)} %"),
        kv("Dominant subzone", fmt(r["dominant_subzone"])),
        html.Hr(style={"borderColor": C_BORDER}),
        html.P("Interpretation", style={"margin": "6px 0 2px", "color": C_ACCENT, "fontWeight": 600}),
        html.P(fmt(r["interpretation_short"]), style={"lineHeight": "1.55", "fontSize": "13px"}),
        html.P("Recommendation", style={"margin": "8px 0 2px", "color": C_ACCENT2, "fontWeight": 600}),
        html.P(fmt(r["management_recommendation"]), style={"lineHeight": "1.55", "fontSize": "13px"}),
    ])
    return fig, panel


@app.callback(Output("prio-bar", "figure"), Input("map-var", "value"))
def update_prio(_):
    d = df_master.sort_values("priority_index", ascending=False).head(15)
    colors = ["#ff6b6b" if v >= 66 else "#ffb454" if v >= 33 else "#37e2b0" for v in d["priority_index"]]
    fig = go.Figure(go.Bar(x=d["id_lote"], y=d["priority_index"], marker_color=colors,
                           text=d["priority_index"], textposition="outside",
                           customdata=np.stack([d["anomaly_class"], d["cluster_name"]], axis=-1),
                           hovertemplate="<b>%{x}</b><br>Priority %{y:.0f}<br>%{customdata[0]} · %{customdata[1]}<extra></extra>"))
    fig.update_yaxes(range=[0, 105], title="Priority index")
    return dark(fig, h=None, legend=False)


@app.callback(Output("spec-fig", "figure"), Input("spec-lotes", "value"), Input("spec-mode", "value"))
def update_spec(selected, mode):
    selected = (selected or [])[:6]
    fig = go.Figure()
    title = {"raw": "Surface reflectance", "norm": "Min–max normalised", "deriv": "1st derivative dR/dλ"}[mode]

    def transform(y):
        if mode == "norm":
            rng = np.nanmax(y) - np.nanmin(y)
            return (y - np.nanmin(y)) / rng if rng > 0 else y
        if mode == "deriv":
            return np.gradient(y, wl_nm)
        return y

    if mode != "deriv":
        fig.add_trace(go.Scatter(x=wl_nm, y=transform(df_spec.mean(axis=0).values), mode="lines",
                                 name="Scene mean", line=dict(color="#9fb3c8", width=2, dash="dash")))
    colors = px.colors.qualitative.Set2
    for i, lid in enumerate(selected):
        if lid in df_spec.index:
            fig.add_trace(go.Scatter(x=wl_nm, y=transform(df_spec.loc[lid].values), mode="lines",
                                     name=lid, line=dict(color=colors[i % len(colors)], width=2.4)))
    for lo, hi in WATER_BANDS:
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#1d3557", opacity=0.20, line_width=0)
    for lo, hi, col, txt in SPECTRAL_REGIONS:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=col, opacity=0.05, line_width=0)
        fig.add_annotation(x=(lo + hi) / 2, y=1.03, xref="x", yref="paper", text=txt, showarrow=False,
                           font=dict(size=9, color=col))
    fig.update_xaxes(title="Wavelength (nm)", range=[376, 2499])
    fig.update_yaxes(title=title)
    return dark(fig, title=f"Spectral signatures · {title}")


@app.callback(Output("spec-re", "figure"), Input("spec-lotes", "value"))
def update_spec_re(selected):
    selected = (selected or [])[:6]
    m = (wl_nm >= 680) & (wl_nm <= 760)
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, lid in enumerate(selected):
        if lid in df_spec.index:
            fig.add_trace(go.Scatter(x=wl_nm[m], y=df_spec.loc[lid].values[m], mode="lines",
                                     name=lid, line=dict(color=colors[i % len(colors)], width=2.4)))
    fig.update_xaxes(title="nm", range=[680, 760])
    fig.update_yaxes(title="SR")
    return dark(fig, legend=False)


@app.callback(Output("bio-radar", "figure"), Input("bio-a", "value"), Input("bio-b", "value"))
def update_radar(la, lb):
    cols = ["Cab_est_ugcm2", "N_foliar_rel", "H2O_foliar_rel", "Efic_fotosint_rel", "Biomasa_rel", "Estres_car_rel"]
    labels = ["Chlorophyll", "N", "Water", "Efficiency", "Biomass", "Low stress"]
    vals = df_bio[cols].astype(float).values
    mn, mx = np.nanmin(vals, axis=0), np.nanmax(vals, axis=0)

    def scaled(lid):
        rr = df_bio[df_bio["id_lote"] == lid]
        if rr.empty:
            return None
        r0 = rr.iloc[0]
        out = [(float(r0[c]) - mn[i]) / (mx[i] - mn[i]) * 100 if mx[i] > mn[i] else 0 for i, c in enumerate(cols)]
        return out + [out[0]]

    theta = labels + [labels[0]]
    fig = go.Figure()
    mean_scaled = [(float(np.nanmean(df_bio[c])) - mn[i]) / (mx[i] - mn[i]) * 100 if mx[i] > mn[i] else 0
                   for i, c in enumerate(cols)]
    fig.add_trace(go.Scatterpolar(r=mean_scaled + [mean_scaled[0]], theta=theta, name="Scene mean",
                                  line=dict(color="#94a3b8", dash="dash"), opacity=0.5))
    for lid, color in [(la, C_ACCENT), (lb, "#ff8c6b")]:
        s = scaled(lid)
        if s:
            fig.add_trace(go.Scatterpolar(r=s, theta=theta, fill="toself", name=lid,
                                          line=dict(color=color, width=3), opacity=0.45))
    fig.update_layout(paper_bgcolor=C_CARD, font_color=C_TEXT, title=dict(text=f"Fingerprint · {la} vs {lb}", font=dict(color=C_ACCENT, size=14)),
                      polar=dict(bgcolor=C_BG2, radialaxis=dict(range=[0, 100], gridcolor=C_BORDER, tickfont=dict(color=C_MUTED)),
                                 angularaxis=dict(gridcolor=C_BORDER, tickfont=dict(color=C_TEXT))),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


@app.callback(Output("xy-fig", "figure"), Input("xy-x", "value"), Input("xy-y", "value"),
              Input("xy-c", "value"), Input("xy-s", "value"))
def update_xy(x, y, c, s):
    d = df_master.copy()
    size = None
    if s and s != "none":
        d["_size"] = norm01(d[s]) * 26 + 6
        size = "_size"
    fig = px.scatter(d, x=x, y=y, color=c, size=size, text="id_lote",
                     color_continuous_scale="RdYlGn_r" if c in REVERSE else "RdYlGn",
                     hover_name="id_lote", hover_data={"cluster_name": True, "anomaly_class": True})
    fig.update_traces(textposition="top center", textfont=dict(size=8, color=C_MUTED),
                      marker=dict(line=dict(width=0.6, color="#0b1018")))
    fig.update_xaxes(title=COL2LABEL.get(x, x))
    fig.update_yaxes(title=COL2LABEL.get(y, y))
    fig.update_layout(coloraxis_colorbar=dict(title=COL2LABEL.get(c, c), thickness=12))
    return dark(fig, title=f"{COL2LABEL.get(y, y)} vs {COL2LABEL.get(x, x)}")


@app.callback(Output("corr-fig", "figure"), Input("xy-x", "value"))
def update_corr(_):
    cols = ["NDRE_mean", "CIre_mean", "PRI_mean", "WBI_mean", "NDVI_mean", "REIP_mean",
            "Cab_est_ugcm2", "N_foliar_rel", "H2O_foliar_rel", "Biomasa_rel",
            "hetero_score", "re_slope_max", "uncertainty_score", "anomaly_score",
            "subzone_critica_pct", "priority_index"]
    labels = [COL2LABEL.get(c, c.replace("_mean", "").replace("_", " ")) for c in cols]
    corr = df_master[cols].astype(float).corr().values
    fig = go.Figure(go.Heatmap(z=corr, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu_r",
                               colorbar=dict(title="r", thickness=12),
                               hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>"))
    fig.update_layout(paper_bgcolor=C_CARD, plot_bgcolor=C_CARD, font=dict(color=C_TEXT, size=9),
                      margin=dict(t=46, l=10, r=10, b=10),
                      title=dict(text="Pearson correlation across lots", font=dict(color=C_ACCENT, size=14)))
    fig.update_xaxes(tickangle=45)
    return fig


@app.callback(Output("rank-bar", "figure"), Output("rank-hist", "figure"),
              Input("rank-var", "value"), Input("rank-lote", "value"))
def update_rank(var, lote):
    label = COL2LABEL.get(var, var)
    rev = var in REVERSE
    d = df_master.sort_values(var, ascending=rev)
    colors = [C_ACCENT if l == lote else "#33414f" for l in d["id_lote"]]
    bar = go.Figure(go.Bar(y=d["id_lote"], x=d[var], orientation="h", marker_color=colors,
                           hovertemplate="<b>%{y}</b><br>" + label + " = %{x:.3f}<extra></extra>"))
    bar.update_layout(height=560)
    bar.update_yaxes(title=None, tickfont=dict(size=7), autorange="reversed")
    bar.update_xaxes(title=label)
    dark(bar, title=f"Ranking · {label}", legend=False)

    vals = df_master[var].astype(float).dropna()
    hist = go.Figure(go.Histogram(x=vals, nbinsx=18, marker_color="#2a3b52", marker_line_color=C_BORDER, marker_line_width=1))
    lv = df_master.loc[df_master["id_lote"] == lote, var]
    if not lv.empty and np.isfinite(lv.iloc[0]):
        hist.add_vline(x=float(lv.iloc[0]), line_color=C_ACCENT, line_width=2.5,
                       annotation_text=lote, annotation_font_color=C_ACCENT)
    hist.add_vline(x=float(vals.mean()), line_color="#ff8c6b", line_dash="dash",
                   annotation_text="mean", annotation_font_color="#ff8c6b")
    hist.update_xaxes(title=label)
    hist.update_yaxes(title="lots")
    dark(hist, title="Distribution", legend=False)
    return bar, hist


@app.callback(Output("pca-fig", "figure"), Input("pca-color", "value"))
def update_pca(color_col):
    common = dict(x="PC1", y="PC2", text="id_lote", hover_data=["cluster_name", "Cab_est_ugcm2", "anomaly_score"])
    if color_col == "cluster_name":
        fig = px.scatter(df_master, color="cluster_name", color_discrete_map=ZONE_COLORS, **common)
    else:
        fig = px.scatter(df_master, color=color_col,
                         color_continuous_scale="RdYlGn_r" if color_col in REVERSE else "RdYlGn", **common)
    fig.update_traces(textposition="top center", textfont=dict(size=8, color=C_MUTED), marker=dict(size=12, line=dict(width=0.6, color="#0b1018")))
    fig.add_hline(y=0, line_dash="dash", line_color=C_BORDER)
    fig.add_vline(x=0, line_dash="dash", line_color=C_BORDER)
    return dark(fig, title="PCA lot space (PC1 vs PC2)")


@app.callback(Output("vip-fig", "figure"), Input("vip-var", "value"))
def update_vip(col):
    wl = df_vip["wavelength_nm"].values
    vip = df_vip[col].values
    fig = go.Figure(go.Bar(x=wl, y=vip, width=5,
                           marker_color=np.where(vip >= 1.0, "#ff8c6b", "#4f8fdf")))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#ffffff")
    for lo, hi in WATER_BANDS:
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#1d3557", opacity=0.20, line_width=0)
    for idx in np.argsort(vip)[-5:][::-1]:
        fig.add_annotation(x=wl[idx], y=vip[idx], text=f"{wl[idx]:.0f}", showarrow=True, arrowhead=1, arrowcolor="#ff8c6b", font=dict(color=C_TEXT, size=9))
    fig.update_xaxes(title="Wavelength (nm)", range=[376, 2499])
    fig.update_yaxes(title="VIP score")
    return dark(fig, title=f"VIP profile · {col.replace('VIP_', '')}", legend=False)


@app.callback(Output("unc-bar", "figure"), Output("unc-scatter", "figure"), Input("map-var", "value"))
def update_unc(_):
    d = df_unc.sort_values("uncertainty_score")
    bar = go.Figure(go.Bar(y=d["id_lote"], x=d["uncertainty_score"], orientation="h", marker_color="#4f8fdf",
                           hovertemplate="<b>%{y}</b><br>score %{x:.4f}<extra></extra>"))
    bar.update_yaxes(tickfont=dict(size=7), autorange="reversed")
    bar.update_xaxes(title="Uncertainty score (lower = steadier)")
    dark(bar, legend=False)
    sc = px.scatter(df_unc, x="unc_rededge_mean", y="unc_swir_mean", color="uncertainty_score",
                    color_continuous_scale="Viridis", text="id_lote", hover_name="id_lote")
    sc.update_traces(textposition="top center", textfont=dict(size=7, color=C_MUTED), marker=dict(size=11, line=dict(width=0.5, color="#0b1018")))
    sc.update_xaxes(title="Red-edge uncertainty")
    sc.update_yaxes(title="SWIR uncertainty")
    dark(sc)
    return bar, sc


@app.callback(Output("explain-panel", "children"), Input("explain-lote", "value"))
def update_explain(lid):
    rr = df_master[df_master["id_lote"] == lid]
    if rr.empty:
        return html.P("Lot not found.")
    r = rr.iloc[0]
    flags = [f for f in str(r["anomaly_flags"]).split(", ") if f and f != "sin_alerta_mayor"]
    chips = ([html.Span(f.replace("_", " "), className="chip chip-amber") for f in flags]
             or [html.Span("no major alert", className="chip chip-green")])
    return html.Div(children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px"}, children=[
            html.H4(lid, style={"margin": 0, "color": C_ACCENT}), priority_chip(r["priority_index"])]),
        html.P("Flags", style={"margin": "10px 0 4px", "color": C_MUTED, "fontSize": "12px"}),
        html.Div(chips),
        html.Hr(style={"borderColor": C_BORDER}),
        html.Div(className="kv", children=[html.Span("Diagnosis", className="k"), html.Span(fmt(r["anomaly_class"]), className="v")]),
        html.Div(className="kv", children=[html.Span("Critical subzone", className="k"), html.Span(f"{fmt(r['subzone_critica_pct'], 1)} %", className="v")]),
        html.Div(className="kv", children=[html.Span("Red-edge slope", className="k"), html.Span(fmt(r["re_slope_max"], 4), className="v")]),
        html.Div(className="kv", children=[html.Span("Uncertainty", className="k"), html.Span(fmt(r["uncertainty_score"], 4), className="v")]),
        html.P("Interpretation", style={"margin": "10px 0 2px", "color": C_ACCENT, "fontWeight": 600}),
        html.P(fmt(r["interpretation_short"]), style={"lineHeight": "1.6", "fontSize": "13px"}),
        html.P("Recommendation", style={"margin": "8px 0 2px", "color": C_ACCENT2, "fontWeight": 600}),
        html.P(fmt(r["management_recommendation"]), style={"lineHeight": "1.6", "fontSize": "13px"}),
    ])


@app.callback(Output("dl-master", "data"), Input("btn-dl", "n_clicks"), prevent_initial_call=True)
def download_master(_):
    return dcc.send_data_frame(df_master.round(4).to_csv, "tanager1_lot_master.csv", index=False)


if __name__ == "__main__":
    app.run(debug=True, port=8050)
