# -*- coding: utf-8 -*-
"""
Public dashboard entrypoint for Render/GitHub.
Uses only tabular/vector data so deployment does not depend on extra image assets.
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


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BRIEF_MD = ROOT / "TECHNICAL_SCIENTIFIC_BRIEF.md"
COMP_MD = ROOT / "competition_summary.md"

with open(DATA / "lotes_dashboard.geojson", encoding="utf-8") as f:
    GEOJSON = json.load(f)

df_idx = pd.read_csv(DATA / "06_estadisticas_por_lote.csv")
df_bio = pd.read_csv(DATA / "07_bioquimica_por_lote.csv")
df_spec = pd.read_csv(DATA / "09_perfiles_todos_lotes.csv", index_col="id_lote")
df_pca = pd.read_csv(DATA / "09_pca_scores.csv")
df_vip = pd.read_csv(DATA / "10_vip_scores.csv")
df_het = pd.read_csv(DATA / "12_heterogeneidad_lotes.csv")
df_reip = pd.read_csv(DATA / "13_reip_por_lote.csv")
df_clust = pd.read_csv(DATA / "14_clustering_lotes.csv").rename(columns={"Unnamed: 0": "id_lote"})
df_unc = pd.read_csv(DATA / "15_incertidumbre_por_lote.csv")
df_red = pd.read_csv(DATA / "16_rededge_metricas_por_lote.csv")
df_anom = pd.read_csv(DATA / "17_anomalias_lotes.csv")
df_sub = pd.read_csv(DATA / "18_subzonas_por_lote.csv")
df_exp = pd.read_csv(DATA / "19_explicacion_lotes.csv")

wl_nm = df_spec.columns.astype(float).values
lotes = sorted(df_spec.index.tolist())

brief_text = BRIEF_MD.read_text(encoding="utf-8") if BRIEF_MD.exists() else "Brief not found."
comp_text = COMP_MD.read_text(encoding="utf-8") if COMP_MD.exists() else "Competition summary not found."

vip_columns = [c for c in df_vip.columns if c.startswith("VIP_")]

df_master = (
    df_idx[["id_lote", "NDRE_mean", "CIre_mean", "PRI_mean", "WBI_mean", "NDVI_mean", "REIP_mean", "NDWI_mean"]]
    .merge(
        df_bio[
            [
                "id_lote",
                "Cab_est_ugcm2",
                "N_foliar_rel",
                "H2O_foliar_rel",
                "Efic_fotosint_rel",
                "Biomasa_rel",
                "Estres_car_rel",
            ]
        ],
        on="id_lote",
        how="left",
    )
    .merge(df_het[["id_lote", "hetero_score", "NDRE_cv", "WBI_cv", "REIP_cv"]], on="id_lote", how="left")
    .merge(df_reip[["id_lote", "REIP_range"]], on="id_lote", how="left")
    .merge(df_pca[["id_lote", "PC1", "PC2", "PC3"]], on="id_lote", how="left")
    .merge(df_clust[["id_lote", "cluster_name"]], on="id_lote", how="left")
    .merge(
        df_unc[
            [
                "id_lote",
                "uncertainty_score",
                "unc_visible_mean",
                "unc_rededge_mean",
                "unc_nir_mean",
                "unc_swir_mean",
                "su_rededge_proxy",
                "su_swir_proxy",
            ]
        ],
        on="id_lote",
        how="left",
    )
    .merge(
        df_red[["id_lote", "re_slope_max", "re_slope_wl_nm", "re_area_680_760", "re_contrast_750_680"]],
        on="id_lote",
        how="left",
    )
    .merge(df_anom[["id_lote", "anomaly_score", "anomaly_class", "anomaly_flags"]], on="id_lote", how="left")
    .merge(
        df_sub[
            [
                "id_lote",
                "subzone_critica_pct",
                "subzone_transicion_pct",
                "subzone_alta_pct",
                "score_mean",
                "dominant_subzone",
            ]
        ],
        on="id_lote",
        how="left",
    )
    .merge(df_exp[["id_lote", "interpretation_short", "management_recommendation"]], on="id_lote", how="left")
)

MAP_VARS = {
    "NDRE": "NDRE_mean",
    "CIre": "CIre_mean",
    "PRI": "PRI_mean",
    "WBI": "WBI_mean",
    "REIP (nm)": "REIP_mean",
    "Cab estimada": "Cab_est_ugcm2",
    "N foliar relativo": "N_foliar_rel",
    "H2O foliar relativo": "H2O_foliar_rel",
    "Biomasa relativa": "Biomasa_rel",
    "Heterogeneidad": "hetero_score",
    "Incertidumbre": "uncertainty_score",
    "Pendiente red-edge": "re_slope_max",
    "Subzona critica (%)": "subzone_critica_pct",
    "Anomalia": "anomaly_score",
}

ZONE_COLORS = {
    "Zona A": "#2a9d8f",
    "Zona B": "#e76f51",
    "Zona C": "#457b9d",
    "Zona D": "#f4a261",
}

C_DARK = "#0f1720"
C_CARD = "#17212b"
C_BORDER = "#2a3442"
C_TEXT = "#ecf3fa"
C_MUTED = "#9fb3c8"
C_ACCENT = "#6ec1ff"
C_WARN = "#e76f51"
C_OK = "#2a9d8f"

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Tanager-1 Dashboard"


def fmt(v: object, nd: int = 3) -> str:
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{nd}f}"
    return str(v)


def quantile_rank(series: pd.Series, value: float) -> str:
    q20, q80 = np.nanpercentile(series, [20, 80])
    if value <= q20:
        return "low"
    if value >= q80:
        return "high"
    return "mid"


card_style = {
    "backgroundColor": C_CARD,
    "border": f"1px solid {C_BORDER}",
    "borderRadius": "10px",
    "padding": "16px",
    "marginBottom": "16px",
}

table_style = {
    "overflowX": "auto",
}

cell_style = {
    "backgroundColor": C_CARD,
    "color": C_TEXT,
    "border": f"1px solid {C_BORDER}",
    "fontSize": "12px",
    "whiteSpace": "normal",
    "height": "auto",
}

header_style = {
    "backgroundColor": C_DARK,
    "color": C_ACCENT,
    "fontWeight": "bold",
    "border": f"1px solid {C_BORDER}",
}

top_cab = df_bio.nlargest(5, "Cab_est_ugcm2")[["id_lote", "Cab_est_ugcm2"]]
top_critical = df_sub.nlargest(5, "subzone_critica_pct")[["id_lote", "subzone_critica_pct"]]
top_anomaly = df_anom.nlargest(8, "anomaly_score")[["id_lote", "anomaly_score", "anomaly_flags"]]


app.layout = html.Div(
    style={"backgroundColor": C_DARK, "minHeight": "100vh", "fontFamily": "monospace", "color": C_TEXT},
    children=[
        html.Div(
            style={"backgroundColor": C_CARD, "borderBottom": f"1px solid {C_BORDER}", "padding": "18px 28px"},
            children=[
                html.H1("Tanager-1 Physiology and Management Dashboard", style={"margin": 0, "fontSize": "22px", "color": C_ACCENT}),
                html.P(
                    "Mato Grosso, Brazil | 66 lots | 426 bands | red-edge + NIR + SWIR physiology inference",
                    style={"margin": "6px 0 0 0", "color": C_MUTED},
                ),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "12px", "padding": "16px 28px", "flexWrap": "wrap"},
            children=[
                *[
                    html.Div(
                        style={**card_style, "flex": "1", "minWidth": "150px", "textAlign": "center", "marginBottom": 0},
                        children=[
                            html.P(label, style={"margin": 0, "fontSize": "11px", "color": C_MUTED}),
                            html.H3(value, style={"margin": "6px 0 0 0", "color": C_ACCENT, "fontSize": "22px"}),
                        ],
                    )
                    for label, value in [
                        ("Lots", "66"),
                        ("Spectral bands", "426"),
                        ("Biochemical variables", "6"),
                        ("QA layers", "uncertainty + anomalies"),
                        ("Subzones", "critical / transition / high"),
                        ("Focus", "competition-ready"),
                    ]
                ]
            ],
        ),
        dcc.Tabs(
            colors={"border": C_BORDER, "primary": C_ACCENT, "background": C_CARD},
            children=[
                dcc.Tab(
                    label="Overview",
                    children=[
                        html.Div(
                            style={"padding": "16px 28px"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                                    children=[
                                        html.Div(
                                            style={**card_style, "flex": "3", "minWidth": "420px"},
                                            children=[
                                                html.Label("Mapped variable", style={"fontSize": "12px", "color": C_MUTED}),
                                                dcc.Dropdown(
                                                    id="map-var",
                                                    options=[{"label": k, "value": v} for k, v in MAP_VARS.items()],
                                                    value="NDRE_mean",
                                                    style={"color": "#000"},
                                                ),
                                                dcc.Graph(id="map-fig", style={"height": "560px"}),
                                            ],
                                        ),
                                        html.Div(
                                            style={**card_style, "flex": "1", "minWidth": "320px"},
                                            children=[
                                                html.H4("Lot inspector", style={"marginTop": 0, "color": C_ACCENT}),
                                                html.Div(id="lot-panel"),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                                    children=[
                                        html.Div(
                                            style={**card_style, "flex": "1", "minWidth": "320px"},
                                            children=[
                                                html.H4("Highest chlorophyll", style={"marginTop": 0, "color": C_ACCENT}),
                                                dash_table.DataTable(
                                                    columns=[{"name": c, "id": c} for c in top_cab.columns],
                                                    data=top_cab.round(2).to_dict("records"),
                                                    style_table=table_style,
                                                    style_cell=cell_style,
                                                    style_header=header_style,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={**card_style, "flex": "1", "minWidth": "320px"},
                                            children=[
                                                html.H4("Largest critical share", style={"marginTop": 0, "color": C_ACCENT}),
                                                dash_table.DataTable(
                                                    columns=[{"name": c, "id": c} for c in top_critical.columns],
                                                    data=top_critical.round(2).to_dict("records"),
                                                    style_table=table_style,
                                                    style_cell=cell_style,
                                                    style_header=header_style,
                                                ),
                                            ],
                                        ),
                                        html.Div(
                                            style={**card_style, "flex": "1", "minWidth": "320px"},
                                            children=[
                                                html.H4("Priority anomalies", style={"marginTop": 0, "color": C_ACCENT}),
                                                dash_table.DataTable(
                                                    columns=[{"name": c, "id": c} for c in top_anomaly.columns],
                                                    data=top_anomaly.round(3).to_dict("records"),
                                                    style_table=table_style,
                                                    style_cell=cell_style,
                                                    style_header=header_style,
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Spectra",
                    children=[
                        html.Div(
                            style={"padding": "16px 28px"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                                    children=[
                                        html.Div(
                                            style={**card_style, "flex": "2", "minWidth": "420px"},
                                            children=[
                                                html.Label("Lots to compare", style={"fontSize": "12px", "color": C_MUTED}),
                                                dcc.Dropdown(
                                                    id="spec-lotes",
                                                    options=[{"label": lid, "value": lid} for lid in lotes],
                                                    value=["A23", "A56", "A65"],
                                                    multi=True,
                                                    style={"color": "#000"},
                                                ),
                                                dcc.Graph(id="spec-fig", style={"height": "540px"}),
                                            ],
                                        ),
                                        html.Div(
                                            style={**card_style, "flex": "1", "minWidth": "320px"},
                                            children=[
                                                html.H4("Physiology guide", style={"marginTop": 0, "color": C_ACCENT}),
                                                dcc.Markdown(
                                                    """
- `376-500 nm`: pigmentos accesorios y respuesta carotenoide.
- `531-570 nm`: ventana asociada a `PRI` y eficiencia fotoquímica.
- `680-750 nm`: red-edge, sensible a clorofila, nitrógeno y estructura foliar.
- `750-1300 nm`: NIR dominado por dispersión del mesófilo y arquitectura del dosel.
- `1300-2499 nm`: SWIR dominado por agua foliar y química estructural.
- Las ventanas `1340-1460` y `1790-1970 nm` se excluyen por absorción atmosférica del agua.
                                                    """,
                                                    style={"color": C_TEXT},
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Biochemistry",
                    children=[
                        html.Div(
                            style={"padding": "16px 28px"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                                    children=[
                                        html.Div(
                                            style={**card_style, "flex": "1", "minWidth": "340px"},
                                            children=[
                                                html.Label("Lot", style={"fontSize": "12px", "color": C_MUTED}),
                                                dcc.Dropdown(
                                                    id="bio-lote",
                                                    options=[{"label": lid, "value": lid} for lid in lotes],
                                                    value="A23",
                                                    style={"color": "#000"},
                                                ),
                                                dcc.Graph(id="bio-radar", style={"height": "430px"}),
                                            ],
                                        ),
                                        html.Div(
                                            style={**card_style, "flex": "2", "minWidth": "460px"},
                                            children=[
                                                html.Label("Biochemical variable", style={"fontSize": "12px", "color": C_MUTED}),
                                                dcc.Dropdown(
                                                    id="bio-var",
                                                    options=[
                                                        {"label": "Cab estimada", "value": "Cab_est_ugcm2"},
                                                        {"label": "N foliar relativo", "value": "N_foliar_rel"},
                                                        {"label": "H2O foliar relativo", "value": "H2O_foliar_rel"},
                                                        {"label": "Eficiencia fotosintetica", "value": "Efic_fotosint_rel"},
                                                        {"label": "Biomasa relativa", "value": "Biomasa_rel"},
                                                        {"label": "Estres carotenoide", "value": "Estres_car_rel"},
                                                    ],
                                                    value="Cab_est_ugcm2",
                                                    style={"color": "#000"},
                                                ),
                                                dcc.Graph(id="bio-bar", style={"height": "430px"}),
                                            ],
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style=card_style,
                                    children=[
                                        html.H4("Biochemical table", style={"marginTop": 0, "color": C_ACCENT}),
                                        dash_table.DataTable(
                                            columns=[{"name": c, "id": c} for c in df_bio.columns],
                                            data=df_bio.round(3).to_dict("records"),
                                            page_size=12,
                                            sort_action="native",
                                            filter_action="native",
                                            style_table=table_style,
                                            style_cell=cell_style,
                                            style_header=header_style,
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="PCA and VIP",
                    children=[
                        html.Div(
                            style={"padding": "16px 28px"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                                    children=[
                                        html.Div(
                                            style={**card_style, "flex": "1", "minWidth": "420px"},
                                            children=[
                                                html.Label("PCA color", style={"fontSize": "12px", "color": C_MUTED}),
                                                dcc.RadioItems(
                                                    id="pca-color",
                                                    options=[
                                                        {"label": "Cluster", "value": "cluster_name"},
                                                        {"label": "Cab", "value": "Cab_est_ugcm2"},
                                                        {"label": "NDRE", "value": "NDRE_mean"},
                                                        {"label": "Anomaly", "value": "anomaly_score"},
                                                    ],
                                                    value="cluster_name",
                                                    inline=True,
                                                    labelStyle={"marginRight": "16px"},
                                                ),
                                                dcc.Graph(id="pca-fig", style={"height": "450px"}),
                                            ],
                                        ),
                                        html.Div(
                                            style={**card_style, "flex": "1", "minWidth": "420px"},
                                            children=[
                                                html.Label("VIP variable", style={"fontSize": "12px", "color": C_MUTED}),
                                                dcc.Dropdown(
                                                    id="vip-var",
                                                    options=[{"label": c.replace("VIP_", ""), "value": c} for c in vip_columns],
                                                    value=vip_columns[0],
                                                    style={"color": "#000"},
                                                ),
                                                dcc.Graph(id="vip-fig", style={"height": "450px"}),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Quality and Zoning",
                    children=[
                        html.Div(
                            style={"padding": "16px 28px"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                                    children=[
                                        html.Div(style={**card_style, "flex": "1", "minWidth": "420px"}, children=[dcc.Graph(id="unc-fig", style={"height": "420px"})]),
                                        html.Div(style={**card_style, "flex": "1", "minWidth": "420px"}, children=[dcc.Graph(id="rededge-fig", style={"height": "420px"})]),
                                    ],
                                ),
                                html.Div(
                                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                                    children=[
                                        html.Div(style={**card_style, "flex": "1", "minWidth": "420px"}, children=[dcc.Graph(id="anom-fig", style={"height": "430px"})]),
                                        html.Div(style={**card_style, "flex": "1", "minWidth": "420px"}, children=[dcc.Graph(id="subzone-fig", style={"height": "430px"})]),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(
                    label="Interpretation",
                    children=[
                        html.Div(
                            style={"padding": "16px 28px"},
                            children=[
                                html.Div(
                                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                                    children=[
                                        html.Div(
                                            style={**card_style, "flex": "1", "minWidth": "320px"},
                                            children=[
                                                html.Label("Lot", style={"fontSize": "12px", "color": C_MUTED}),
                                                dcc.Dropdown(
                                                    id="explain-lote",
                                                    options=[{"label": lid, "value": lid} for lid in lotes],
                                                    value="A23",
                                                    style={"color": "#000"},
                                                ),
                                                html.Div(id="explain-panel", style={"marginTop": "14px"}),
                                            ],
                                        ),
                                        html.Div(
                                            style={**card_style, "flex": "2", "minWidth": "520px"},
                                            children=[
                                                html.H4("Technical scientific brief", style={"marginTop": 0, "color": C_ACCENT}),
                                                dcc.Markdown(brief_text),
                                                html.Hr(style={"borderColor": C_BORDER}),
                                                html.H4("Competition summary", style={"color": C_ACCENT}),
                                                dcc.Markdown(comp_text),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("map-fig", "figure"),
    Output("lot-panel", "children"),
    Input("map-var", "value"),
    Input("map-fig", "clickData"),
)
def update_map(var_col: str, click_data: dict | None):
    reverse_scale = var_col in {"uncertainty_score", "anomaly_score", "subzone_critica_pct"}
    fig = px.choropleth_mapbox(
        df_master,
        geojson=GEOJSON,
        locations="id_lote",
        featureidkey="properties.id_lote",
        color=var_col,
        color_continuous_scale="RdYlGn_r" if reverse_scale else "RdYlGn",
        mapbox_style="carto-darkmatter",
        center={"lat": -15.45, "lon": -55.02},
        zoom=10,
        opacity=0.78,
        hover_name="id_lote",
        hover_data={
            "cluster_name": True,
            "anomaly_class": True,
            "subzone_critica_pct": ":.1f",
            var_col: ":.3f",
        },
    )
    fig.update_layout(
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_CARD,
        font_color=C_TEXT,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
    )

    default_lot = "A23"
    lot_id = default_lot
    if click_data and click_data.get("points"):
        lot_id = click_data["points"][0].get("location", default_lot)
    row = df_master[df_master["id_lote"] == lot_id].iloc[0]

    cab_rank = quantile_rank(df_master["Cab_est_ugcm2"], float(row["Cab_est_ugcm2"]))
    h2o_rank = quantile_rank(df_master["H2O_foliar_rel"], float(row["H2O_foliar_rel"]))
    unc_rank = quantile_rank(df_master["uncertainty_score"], float(row["uncertainty_score"]))

    panel = html.Div(
        children=[
            html.H4(lot_id, style={"marginTop": 0, "color": C_ACCENT}),
            html.P(f"Cluster: {fmt(row['cluster_name'])}", style={"margin": "3px 0"}),
            html.P(f"Anomaly: {fmt(row['anomaly_class'])}", style={"margin": "3px 0"}),
            html.P(f"Dominant subzone: {fmt(row['dominant_subzone'])}", style={"margin": "3px 0"}),
            html.Hr(style={"borderColor": C_BORDER}),
            html.P(f"NDRE: {fmt(row['NDRE_mean'])}", style={"margin": "3px 0"}),
            html.P(f"REIP: {fmt(row['REIP_mean'])} nm", style={"margin": "3px 0"}),
            html.P(f"Cab: {fmt(row['Cab_est_ugcm2'])} ({cab_rank})", style={"margin": "3px 0"}),
            html.P(f"H2O relative: {fmt(row['H2O_foliar_rel'], 1)} ({h2o_rank})", style={"margin": "3px 0"}),
            html.P(f"Uncertainty: {fmt(row['uncertainty_score'], 5)} ({unc_rank})", style={"margin": "3px 0"}),
            html.P(f"Critical subzone: {fmt(row['subzone_critica_pct'], 1)} %", style={"margin": "3px 0"}),
            html.Hr(style={"borderColor": C_BORDER}),
            html.P("Interpretation", style={"margin": "6px 0", "color": C_ACCENT}),
            html.P(fmt(row["interpretation_short"]), style={"whiteSpace": "pre-wrap", "lineHeight": "1.5"}),
            html.P("Recommendation", style={"margin": "8px 0 0 0", "color": C_ACCENT}),
            html.P(fmt(row["management_recommendation"]), style={"whiteSpace": "pre-wrap", "lineHeight": "1.5"}),
        ]
    )
    return fig, panel


@app.callback(Output("spec-fig", "figure"), Input("spec-lotes", "value"))
def update_spectra(selected: list[str] | None):
    selected = (selected or [])[:5]
    fig = go.Figure()
    if not selected:
        return fig

    scene_mean = df_spec.mean(axis=0).values
    fig.add_trace(
        go.Scatter(
            x=wl_nm,
            y=scene_mean,
            mode="lines",
            name="Scene mean",
            line=dict(color="#b0bec5", width=2, dash="dash"),
        )
    )
    colors = px.colors.qualitative.Set2
    for i, lid in enumerate(selected):
        if lid not in df_spec.index:
            continue
        fig.add_trace(
            go.Scatter(
                x=wl_nm,
                y=df_spec.loc[lid].values,
                mode="lines",
                name=lid,
                line=dict(color=colors[i % len(colors)], width=2.6),
            )
        )

    for lo, hi in [(1340, 1460), (1790, 1970)]:
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#1d3557", opacity=0.18, line_width=0)
    for lo, hi, col, text in [
        (376, 500, "#457b9d", "Blue"),
        (500, 680, "#2a9d8f", "Visible"),
        (680, 750, "#e76f51", "Red-edge"),
        (750, 1300, "#adb5bd", "NIR"),
        (1300, 1800, "#f4a261", "SWIR-1"),
        (1800, 2499, "#9c6644", "SWIR-2"),
    ]:
        fig.add_vrect(x0=lo, x1=hi, fillcolor=col, opacity=0.05, line_width=0)
        fig.add_annotation(x=(lo + hi) / 2, y=1.02, xref="x", yref="paper", text=text, showarrow=False, font=dict(size=9, color=col))

    fig.update_layout(
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_DARK,
        font_color=C_TEXT,
        xaxis=dict(title="Wavelength (nm)", range=[376, 2499], gridcolor=C_BORDER),
        yaxis=dict(title="Surface reflectance", gridcolor=C_BORDER),
        legend=dict(bgcolor=C_CARD, bordercolor=C_BORDER),
        margin=dict(t=40, b=40),
        title="Spectral signatures and physiology-sensitive regions",
    )
    return fig


@app.callback(Output("bio-radar", "figure"), Input("bio-lote", "value"))
def update_radar(lid: str):
    row = df_bio[df_bio["id_lote"] == lid]
    if row.empty:
        return go.Figure()
    cols = [
        "Cab_est_ugcm2",
        "N_foliar_rel",
        "H2O_foliar_rel",
        "Efic_fotosint_rel",
        "Biomasa_rel",
        "Estres_car_rel",
    ]
    labels = ["Chlorophyll", "N", "Water", "Efficiency", "Biomass", "Carotenoid stress"]
    all_vals = df_bio[cols].values.astype(float)
    mn = np.nanmin(all_vals, axis=0)
    mx = np.nanmax(all_vals, axis=0)
    vals = []
    row0 = row.iloc[0]
    for i, c in enumerate(cols):
        vals.append((float(row0[c]) - mn[i]) / (mx[i] - mn[i]) * 100 if mx[i] > mn[i] else 0)
    vals.append(vals[0])
    labels_closed = labels + [labels[0]]

    mean_vals = []
    for i, c in enumerate(cols):
        mean_vals.append((float(np.nanmean(df_bio[c])) - mn[i]) / (mx[i] - mn[i]) * 100 if mx[i] > mn[i] else 0)
    mean_vals.append(mean_vals[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=mean_vals, theta=labels_closed, fill="toself", name="Scene mean", opacity=0.14, line=dict(color="#94a3b8", dash="dash")))
    fig.add_trace(go.Scatterpolar(r=vals, theta=labels_closed, fill="toself", name=lid, opacity=0.35, line=dict(color=C_ACCENT, width=3)))
    fig.update_layout(
        paper_bgcolor=C_CARD,
        font_color=C_TEXT,
        polar=dict(
            bgcolor=C_DARK,
            radialaxis=dict(range=[0, 100], gridcolor=C_BORDER, tickfont=dict(color=C_TEXT)),
            angularaxis=dict(gridcolor=C_BORDER, tickfont=dict(color=C_TEXT)),
        ),
        title=f"Biochemical fingerprint: {lid}",
        legend=dict(bgcolor=C_CARD),
    )
    return fig


@app.callback(Output("bio-bar", "figure"), Input("bio-var", "value"))
def update_bio_bar(col: str):
    top = df_bio.sort_values(col, ascending=False)
    fig = px.bar(top, x="id_lote", y=col, color=col, color_continuous_scale="Viridis")
    fig.update_layout(
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_DARK,
        font_color=C_TEXT,
        xaxis=dict(title="Lot", tickangle=-45, gridcolor=C_BORDER),
        yaxis=dict(title=col, gridcolor=C_BORDER),
        coloraxis_colorbar=dict(title=col),
        margin=dict(t=40, b=80),
        title=f"Lot ranking: {col}",
    )
    return fig


@app.callback(Output("pca-fig", "figure"), Input("pca-color", "value"))
def update_pca(color_col: str):
    if color_col == "cluster_name":
        fig = px.scatter(
            df_master,
            x="PC1",
            y="PC2",
            text="id_lote",
            color="cluster_name",
            color_discrete_map=ZONE_COLORS,
            hover_data=["Cab_est_ugcm2", "hetero_score", "anomaly_score"],
        )
    else:
        fig = px.scatter(
            df_master,
            x="PC1",
            y="PC2",
            text="id_lote",
            color=color_col,
            color_continuous_scale="RdYlGn",
            hover_data=["cluster_name", "hetero_score", "anomaly_score"],
        )
    fig.update_traces(textposition="top center", textfont_size=8, marker_size=10)
    fig.add_hline(y=0, line_dash="dash", line_color=C_BORDER)
    fig.add_vline(x=0, line_dash="dash", line_color=C_BORDER)
    fig.update_layout(
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_DARK,
        font_color=C_TEXT,
        xaxis=dict(title="PC1", gridcolor=C_BORDER),
        yaxis=dict(title="PC2", gridcolor=C_BORDER),
        legend=dict(bgcolor=C_CARD, bordercolor=C_BORDER),
        margin=dict(t=40, b=30),
        title="PCA lot space",
    )
    return fig


@app.callback(Output("vip-fig", "figure"), Input("vip-var", "value"))
def update_vip(col: str):
    wl = df_vip["wavelength_nm"].values
    vip = df_vip[col].values
    colors = np.where(vip >= 1.0, C_WARN, "#457b9d")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=wl, y=vip, marker_color=colors, width=5))
    fig.add_hline(y=1.0, line_dash="dash", line_color="#ffffff")
    for lo, hi in [(1340, 1460), (1790, 1970)]:
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#1d3557", opacity=0.18, line_width=0)
    for idx in np.argsort(vip)[-5:][::-1]:
        fig.add_annotation(x=wl[idx], y=vip[idx], text=f"{wl[idx]:.0f}", showarrow=True, arrowhead=1, arrowcolor=C_WARN)
    fig.update_layout(
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_DARK,
        font_color=C_TEXT,
        xaxis=dict(title="Wavelength (nm)", range=[376, 2499], gridcolor=C_BORDER),
        yaxis=dict(title="VIP score", gridcolor=C_BORDER),
        margin=dict(t=40, b=40),
        title=f"VIP profile: {col.replace('VIP_', '')}",
    )
    return fig


@app.callback(
    Output("unc-fig", "figure"),
    Output("rededge-fig", "figure"),
    Output("anom-fig", "figure"),
    Output("subzone-fig", "figure"),
    Input("map-var", "value"),
)
def update_quality_figs(_):
    fig_unc = px.scatter(
        df_master,
        x="unc_rededge_mean",
        y="unc_swir_mean",
        size="subzone_critica_pct",
        color="anomaly_score",
        hover_name="id_lote",
        color_continuous_scale="Turbo",
    )
    fig_unc.update_layout(
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_DARK,
        font_color=C_TEXT,
        xaxis=dict(title="Red-edge uncertainty", gridcolor=C_BORDER),
        yaxis=dict(title="SWIR uncertainty", gridcolor=C_BORDER),
        margin=dict(t=40, b=30),
        title="Uncertainty space",
    )

    fig_red = px.scatter(
        df_master,
        x="re_slope_max",
        y="Cab_est_ugcm2",
        size="H2O_foliar_rel",
        color="hetero_score",
        hover_name="id_lote",
        color_continuous_scale="Viridis",
    )
    fig_red.update_layout(
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_DARK,
        font_color=C_TEXT,
        xaxis=dict(title="Max red-edge slope", gridcolor=C_BORDER),
        yaxis=dict(title="Estimated chlorophyll", gridcolor=C_BORDER),
        margin=dict(t=40, b=30),
        title="Red-edge structure vs chlorophyll",
    )

    top_anom_local = df_master.sort_values("anomaly_score", ascending=False).head(12)
    fig_anom = px.bar(
        top_anom_local,
        x="id_lote",
        y="anomaly_score",
        color="anomaly_class",
        hover_data=["anomaly_flags", "subzone_critica_pct"],
        color_discrete_map={"ALTA": C_WARN, "MEDIA": "#f4a261", "NORMAL": C_OK},
    )
    fig_anom.update_layout(
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_DARK,
        font_color=C_TEXT,
        xaxis=dict(title="Lot", tickangle=-45, gridcolor=C_BORDER),
        yaxis=dict(title="Anomaly score", gridcolor=C_BORDER),
        margin=dict(t=40, b=80),
        title="Priority anomaly ranking",
    )

    stacked = df_sub.melt(
        id_vars="id_lote",
        value_vars=["subzone_critica_pct", "subzone_transicion_pct", "subzone_alta_pct"],
        var_name="subzone",
        value_name="pct",
    )
    sub_labels = {
        "subzone_critica_pct": "Critica",
        "subzone_transicion_pct": "Transicion",
        "subzone_alta_pct": "Alta",
    }
    stacked["subzone"] = stacked["subzone"].map(sub_labels)
    fig_sub = px.bar(
        stacked,
        x="id_lote",
        y="pct",
        color="subzone",
        barmode="stack",
        color_discrete_map={"Critica": C_WARN, "Transicion": "#f4a261", "Alta": C_OK},
    )
    fig_sub.update_layout(
        paper_bgcolor=C_CARD,
        plot_bgcolor=C_DARK,
        font_color=C_TEXT,
        xaxis=dict(title="Lot", tickangle=-45, gridcolor=C_BORDER),
        yaxis=dict(title="Subzone share (%)", gridcolor=C_BORDER),
        margin=dict(t=40, b=80),
        title="Intra-lot management zoning",
    )
    return fig_unc, fig_red, fig_anom, fig_sub


@app.callback(Output("explain-panel", "children"), Input("explain-lote", "value"))
def update_explain(lid: str):
    row = df_master[df_master["id_lote"] == lid]
    if row.empty:
        return html.P("Lot not found.")
    r = row.iloc[0]
    return html.Div(
        children=[
            html.H4(lid, style={"marginTop": 0, "color": C_ACCENT}),
            html.P(f"Diagnosis: {fmt(r['anomaly_class'])}", style={"margin": "4px 0"}),
            html.P(f"Flags: {fmt(r['anomaly_flags'])}", style={"margin": "4px 0"}),
            html.P(f"Critical subzone: {fmt(r['subzone_critica_pct'], 1)} %", style={"margin": "4px 0"}),
            html.P(f"Red-edge slope: {fmt(r['re_slope_max'], 4)}", style={"margin": "4px 0"}),
            html.P(f"REIP range: {fmt(r['REIP_range'], 2)} nm", style={"margin": "4px 0"}),
            html.P(f"Uncertainty: {fmt(r['uncertainty_score'], 5)}", style={"margin": "4px 0"}),
            html.Hr(style={"borderColor": C_BORDER}),
            html.P("Interpretation", style={"margin": "6px 0", "color": C_ACCENT}),
            html.P(fmt(r["interpretation_short"]), style={"whiteSpace": "pre-wrap", "lineHeight": "1.6"}),
            html.P("Recommendation", style={"margin": "6px 0", "color": C_ACCENT}),
            html.P(fmt(r["management_recommendation"]), style={"whiteSpace": "pre-wrap", "lineHeight": "1.6"}),
        ]
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
