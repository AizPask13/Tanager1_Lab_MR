# -*- coding: utf-8 -*-
"""
Tanager-1 Precision Agriculture Dashboard
Mato Grosso, Brazil — May 2025
Manuel Ramos
Deploy: Render.com | gunicorn app:server
"""

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, dash_table
from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────
DATA  = Path("data")
IMG   = "assets/img"

# ── Cargar datos ──────────────────────────────────────────────────────
with open(DATA / "lotes_dashboard.geojson", encoding="utf-8") as f:
    geojson = json.load(f)

df_idx    = pd.read_csv(DATA / "06_estadisticas_por_lote.csv")
df_bio    = pd.read_csv(DATA / "07_bioquimica_por_lote.csv")
df_spec   = pd.read_csv(DATA / "09_perfiles_todos_lotes.csv", index_col="id_lote")
df_pca    = pd.read_csv(DATA / "09_pca_scores.csv")
df_vip    = pd.read_csv(DATA / "10_vip_scores.csv")
df_het    = pd.read_csv(DATA / "12_heterogeneidad_lotes.csv")
df_reip   = pd.read_csv(DATA / "13_reip_por_lote.csv")
df_clust  = pd.read_csv(DATA / "14_clustering_lotes.csv").rename(columns={"Unnamed: 0": "id_lote"})

wl_nm = df_spec.columns.astype(float).values
lotes = df_spec.index.tolist()

# Master dataframe por lote
df_master = (df_idx[["id_lote","NDRE_mean","CIre_mean","PRI_mean","WBI_mean",
                       "NDVI_mean","REIP_mean","NDWI_mean"]]
             .merge(df_bio[["id_lote","Cab_est_ugcm2","N_foliar_rel","H2O_foliar_rel",
                              "Efic_fotosint_rel","Biomasa_rel","Estres_car_rel"]], on="id_lote", how="left")
             .merge(df_het[["id_lote","hetero_score","NDRE_cv","REIP_cv"]], on="id_lote", how="left")
             .merge(df_reip[["id_lote","REIP_mean","REIP_range"]], on="id_lote",
                    how="left", suffixes=("","_reip"))
             .merge(df_pca[["id_lote","PC1","PC2","PC3"]], on="id_lote", how="left")
             .merge(df_clust[["id_lote","cluster","cluster_name"]], on="id_lote", how="left"))

# Variables disponibles para el mapa
MAP_VARS = {
    "NDRE (Clorofila)":          "NDRE_mean",
    "CIre (Clorofila)":          "CIre_mean",
    "PRI (Efic. fotosintetica)": "PRI_mean",
    "WBI (Estres hidrico)":      "WBI_mean",
    "NDVI (Vigor)":              "NDVI_mean",
    "REIP (N foliar, nm)":       "REIP_mean",
    "Cab estimada (ug/cm2)":     "Cab_est_ugcm2",
    "N foliar (relativo)":       "N_foliar_rel",
    "H2O foliar (relativo)":     "H2O_foliar_rel",
    "Biomasa (relativo)":        "Biomasa_rel",
    "Heterogeneidad intra-lote": "hetero_score",
    "PC1 (agua SWIR-1)":         "PC1",
    "PC2 (N/lignina SWIR-2)":    "PC2",
}

ZONE_COLORS = {"Zona A": "#2ecc71", "Zona B": "#e74c3c",
               "Zona C": "#3498db", "Zona D": "#f39c12"}

# ── Paleta ─────────────────────────────────────────────────────────────
C_DARK   = "#0d1117"
C_CARD   = "#161b22"
C_BORDER = "#30363d"
C_TEXT   = "#e6edf3"
C_ACCENT = "#58a6ff"
C_GREEN  = "#3fb950"

# ── App ────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

card_style = {
    "backgroundColor": C_CARD,
    "border": f"1px solid {C_BORDER}",
    "borderRadius": "8px",
    "padding": "16px",
    "marginBottom": "16px",
}

app.layout = html.Div(style={"backgroundColor": C_DARK, "minHeight": "100vh",
                               "fontFamily": "monospace", "color": C_TEXT}, children=[

    # Header
    html.Div(style={"backgroundColor": C_CARD, "borderBottom": f"1px solid {C_BORDER}",
                    "padding": "16px 32px", "display": "flex",
                    "alignItems": "center", "gap": "16px"}, children=[
        html.Div([
            html.H1("Tanager-1 · Precision Agriculture Dashboard",
                    style={"margin": 0, "fontSize": "20px", "color": C_ACCENT}),
            html.P("Mato Grosso, Brazil · 2025-05-01 · 66 lotes · 426 bands · 30m · Manuel Ramos",
                   style={"margin": 0, "fontSize": "12px", "color": "#8b949e"}),
        ])
    ]),

    # KPI row
    html.Div(style={"display": "flex", "gap": "12px", "padding": "16px 32px",
                    "flexWrap": "wrap"}, children=[
        *[html.Div(style={**card_style, "flex": "1", "minWidth": "140px",
                          "textAlign": "center", "marginBottom": 0}, children=[
            html.P(label, style={"margin": 0, "fontSize": "11px", "color": "#8b949e"}),
            html.H3(value, style={"margin": 0, "color": C_ACCENT, "fontSize": "22px"}),
        ]) for label, value in [
            ("Lotes analizados", "66"),
            ("Bandas espectrales", "426"),
            ("Resolución", "30 m"),
            ("Variables bioquímicas", "6"),
            ("Zonas de manejo", "2"),
            ("R² PLSR (LOO-CV)", ">0.93"),
        ]]
    ]),

    # Tabs
    dcc.Tabs(id="tabs", value="tab-mapa", style={"padding": "0 32px"},
             colors={"border": C_BORDER, "primary": C_ACCENT, "background": C_CARD},
             children=[

        # ── TAB 1: Mapa interactivo ───────────────────────────────────
        dcc.Tab(label="Mapa interactivo", value="tab-mapa",
                style={"backgroundColor": C_CARD, "color": C_TEXT},
                selected_style={"backgroundColor": C_DARK, "color": C_ACCENT,
                                "borderTop": f"2px solid {C_ACCENT}"},
                children=[html.Div(style={"padding": "16px 32px"}, children=[
                    html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                        html.Div(style={**card_style, "flex": "3", "minWidth": "400px"}, children=[
                            html.Label("Variable:", style={"fontSize": "12px", "color": "#8b949e"}),
                            dcc.Dropdown(
                                id="map-var",
                                options=[{"label": k, "value": v} for k, v in MAP_VARS.items()],
                                value="NDRE_mean",
                                style={"backgroundColor": C_CARD, "color": "#000", "marginBottom": "8px"}
                            ),
                            dcc.Graph(id="mapa-coropletico", style={"height": "520px"}),
                        ]),
                        html.Div(style={**card_style, "flex": "1", "minWidth": "280px"}, children=[
                            html.H4("Lote seleccionado", style={"color": C_ACCENT, "marginTop": 0}),
                            html.Div(id="lote-info-panel",
                                     style={"fontSize": "13px", "lineHeight": "2"}),
                        ]),
                    ]),
                ])]),

        # ── TAB 2: Perfil espectral ───────────────────────────────────
        dcc.Tab(label="Perfil espectral", value="tab-espectro",
                style={"backgroundColor": C_CARD, "color": C_TEXT},
                selected_style={"backgroundColor": C_DARK, "color": C_ACCENT,
                                "borderTop": f"2px solid {C_ACCENT}"},
                children=[html.Div(style={"padding": "16px 32px"}, children=[
                    html.Div(style=card_style, children=[
                        html.Label("Seleccionar lotes (máx 5):",
                                   style={"fontSize": "12px", "color": "#8b949e"}),
                        dcc.Dropdown(
                            id="spec-lotes",
                            options=[{"label": l, "value": l} for l in sorted(lotes)],
                            value=["A23", "A39", "A65"],
                            multi=True,
                            style={"backgroundColor": C_CARD, "color": "#000"}
                        ),
                        dcc.Graph(id="spectral-plot", style={"height": "500px"}),
                    ]),
                    html.Div(style=card_style, children=[
                        html.H4("Imagen de análisis: diferencia espectral top vs bottom",
                                style={"color": C_ACCENT, "marginTop": 0}),
                        html.Img(src=f"/{IMG}/08_diferencia_espectral.png",
                                 style={"width": "100%", "borderRadius": "4px"}),
                    ]),
                ])]),

        # ── TAB 3: Bioquímica ─────────────────────────────────────────
        dcc.Tab(label="Bioquímica por lote", value="tab-bio",
                style={"backgroundColor": C_CARD, "color": C_TEXT},
                selected_style={"backgroundColor": C_DARK, "color": C_ACCENT,
                                "borderTop": f"2px solid {C_ACCENT}"},
                children=[html.Div(style={"padding": "16px 32px"}, children=[
                    html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                        html.Div(style={**card_style, "flex": "1", "minWidth": "320px"}, children=[
                            html.Label("Lote:", style={"fontSize": "12px", "color": "#8b949e"}),
                            dcc.Dropdown(
                                id="bio-lote",
                                options=[{"label": l, "value": l} for l in sorted(lotes)],
                                value="A23",
                                style={"backgroundColor": C_CARD, "color": "#000", "marginBottom": "8px"}
                            ),
                            dcc.Graph(id="bio-radar", style={"height": "420px"}),
                        ]),
                        html.Div(style={**card_style, "flex": "2", "minWidth": "400px"}, children=[
                            html.H4("Heatmap bioquímico — todos los lotes",
                                    style={"color": C_ACCENT, "marginTop": 0}),
                            html.Img(src=f"/{IMG}/07_heatmap_bioquimica.png",
                                     style={"width": "100%", "borderRadius": "4px"}),
                        ]),
                    ]),
                    html.Div(style=card_style, children=[
                        html.H4("Tabla completa bioquímica",
                                style={"color": C_ACCENT, "marginTop": 0}),
                        dash_table.DataTable(
                            id="bio-table",
                            columns=[{"name": c, "id": c} for c in
                                     ["id_lote","Cab_est_ugcm2","N_foliar_rel",
                                      "H2O_foliar_rel","Efic_fotosint_rel",
                                      "Biomasa_rel","Estres_car_rel"]],
                            data=df_bio.to_dict("records"),
                            sort_action="native", filter_action="native",
                            page_size=15,
                            style_table={"overflowX": "auto"},
                            style_cell={"backgroundColor": C_CARD, "color": C_TEXT,
                                        "border": f"1px solid {C_BORDER}", "fontSize": "12px"},
                            style_header={"backgroundColor": C_DARK, "color": C_ACCENT,
                                          "fontWeight": "bold"},
                        ),
                    ]),
                ])]),

        # ── TAB 4: PCA + Clustering ───────────────────────────────────
        dcc.Tab(label="PCA / Zonas de manejo", value="tab-pca",
                style={"backgroundColor": C_CARD, "color": C_TEXT},
                selected_style={"backgroundColor": C_DARK, "color": C_ACCENT,
                                "borderTop": f"2px solid {C_ACCENT}"},
                children=[html.Div(style={"padding": "16px 32px"}, children=[
                    html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                        html.Div(style={**card_style, "flex": "1", "minWidth": "360px"}, children=[
                            html.H4("Biplot PCA — PC1 vs PC2", style={"color": C_ACCENT, "marginTop": 0}),
                            html.Label("Color:", style={"fontSize": "12px", "color": "#8b949e"}),
                            dcc.RadioItems(
                                id="pca-color",
                                options=[{"label": "Zona cluster", "value": "cluster_name"},
                                         {"label": "NDRE", "value": "NDRE_mean"},
                                         {"label": "Cab (µg/cm²)", "value": "Cab_est_ugcm2"}],
                                value="cluster_name",
                                inline=True,
                                style={"fontSize": "12px", "marginBottom": "8px"},
                                labelStyle={"marginRight": "16px"}
                            ),
                            dcc.Graph(id="pca-biplot", style={"height": "420px"}),
                        ]),
                        html.Div(style={**card_style, "flex": "1", "minWidth": "360px"}, children=[
                            html.H4("Mapa zonas de manejo (clustering)",
                                    style={"color": C_ACCENT, "marginTop": 0}),
                            html.Img(src=f"/{IMG}/14_fingerprint_clustering.png",
                                     style={"width": "100%", "borderRadius": "4px"}),
                        ]),
                    ]),
                    html.Div(style=card_style, children=[
                        html.H4("PC Loadings — importancia espectral",
                                style={"color": C_ACCENT, "marginTop": 0}),
                        html.Img(src=f"/{IMG}/09_pca_espectral.png",
                                 style={"width": "100%", "borderRadius": "4px"}),
                    ]),
                ])]),

        # ── TAB 5: PLSR + VIP ─────────────────────────────────────────
        dcc.Tab(label="PLSR / VIP scores", value="tab-plsr",
                style={"backgroundColor": C_CARD, "color": C_TEXT},
                selected_style={"backgroundColor": C_DARK, "color": C_ACCENT,
                                "borderTop": f"2px solid {C_ACCENT}"},
                children=[html.Div(style={"padding": "16px 32px"}, children=[
                    html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                        html.Div(style={**card_style, "flex": "1", "minWidth": "360px"}, children=[
                            html.H4("VIP scores por variable",
                                    style={"color": C_ACCENT, "marginTop": 0}),
                            html.Label("Variable:", style={"fontSize": "12px", "color": "#8b949e"}),
                            dcc.Dropdown(
                                id="vip-var",
                                options=[{"label": c.replace("VIP_",""), "value": c}
                                         for c in df_vip.columns if c.startswith("VIP_")],
                                value=[c for c in df_vip.columns if c.startswith("VIP_")][0],
                                style={"backgroundColor": C_CARD, "color": "#000", "marginBottom": "8px"}
                            ),
                            dcc.Graph(id="vip-plot", style={"height": "420px"}),
                        ]),
                        html.Div(style={**card_style, "flex": "1", "minWidth": "360px"}, children=[
                            html.H4("PLSR — Observado vs Predicho (LOO-CV)",
                                    style={"color": C_ACCENT, "marginTop": 0}),
                            html.Img(src=f"/{IMG}/10_plsr_bioquimica.png",
                                     style={"width": "100%", "borderRadius": "4px"}),
                        ]),
                    ]),
                ])]),

        # ── TAB 6: Análisis espacial ──────────────────────────────────
        dcc.Tab(label="Análisis espacial", value="tab-spatial",
                style={"backgroundColor": C_CARD, "color": C_TEXT},
                selected_style={"backgroundColor": C_DARK, "color": C_ACCENT,
                                "borderTop": f"2px solid {C_ACCENT}"},
                children=[html.Div(style={"padding": "16px 32px"}, children=[
                    html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                        html.Div(style={**card_style, "flex": "1", "minWidth": "360px"}, children=[
                            html.H4("Gradiente N foliar — REIP continuo 30m",
                                    style={"color": C_ACCENT, "marginTop": 0}),
                            html.Img(src=f"/{IMG}/13_reip_gradiente_N.png",
                                     style={"width": "100%", "borderRadius": "4px"}),
                        ]),
                        html.Div(style={**card_style, "flex": "1", "minWidth": "360px"}, children=[
                            html.H4("Heterogeneidad intra-lote (CV NDRE 150m)",
                                    style={"color": C_ACCENT, "marginTop": 0}),
                            html.Img(src=f"/{IMG}/12_heterogeneidad_lotes.png",
                                     style={"width": "100%", "borderRadius": "4px"}),
                        ]),
                    ]),
                    html.Div(style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}, children=[
                        html.Div(style={**card_style, "flex": "1", "minWidth": "360px"}, children=[
                            html.H4("RGB + lotes + calidad",
                                    style={"color": C_ACCENT, "marginTop": 0}),
                            html.Img(src=f"/{IMG}/03_RGB_calidad_lotes.png",
                                     style={"width": "100%", "borderRadius": "4px"}),
                        ]),
                        html.Div(style={**card_style, "flex": "1", "minWidth": "360px"}, children=[
                            html.H4("Mapa PLSR píxel a píxel",
                                    style={"color": C_ACCENT, "marginTop": 0}),
                            html.Img(src=f"/{IMG}/10_mapa_plsr_pixeles.png",
                                     style={"width": "100%", "borderRadius": "4px"}),
                        ]),
                    ]),
                ])]),
    ]),
])

# ── Callbacks ──────────────────────────────────────────────────────────

@app.callback(
    Output("mapa-coropletico", "figure"),
    Output("lote-info-panel", "children"),
    Input("map-var", "value"),
    Input("mapa-coropletico", "clickData"),
)
def update_map(var_col, click_data):
    label = {v: k for k, v in MAP_VARS.items()}.get(var_col, var_col)
    df_plot = df_master.copy()

    fig = px.choropleth_mapbox(
        df_plot, geojson=geojson,
        locations="id_lote", featureidkey="properties.id_lote",
        color=var_col, color_continuous_scale="RdYlGn",
        mapbox_style="carto-darkmatter",
        center={"lat": -15.45, "lon": -55.02}, zoom=10,
        hover_name="id_lote",
        hover_data={"id_lote": False, var_col: ":.3f",
                    "cluster_name": True, "hetero_score": ":.1f"},
        labels={var_col: label, "cluster_name": "Zona", "hetero_score": "Hetero CV%"},
        opacity=0.75,
    )
    fig.update_layout(
        paper_bgcolor=C_CARD, plot_bgcolor=C_CARD,
        font_color=C_TEXT, margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title=label[:18], tickfont=dict(color=C_TEXT),
                                titlefont=dict(color=C_TEXT)),
    )

    # Info panel
    info = html.P("Haz clic en un lote para ver detalles.",
                  style={"color": "#8b949e"})
    if click_data:
        lid = click_data["points"][0].get("location")
        row = df_master[df_master["id_lote"] == lid]
        if not row.empty:
            r = row.iloc[0]
            info = html.Div([
                html.H4(lid, style={"color": C_ACCENT, "marginTop": 0}),
                html.P(f"Zona: {r.get('cluster_name','—')}", style={"margin": "2px 0"}),
                html.Hr(style={"borderColor": C_BORDER}),
                *[html.P(f"{lbl}: {r.get(col, np.nan):.3f}",
                         style={"margin": "2px 0", "fontSize": "13px"})
                  for lbl, col in [
                      ("NDRE", "NDRE_mean"), ("REIP (nm)", "REIP_mean"),
                      ("Cab (µg/cm²)", "Cab_est_ugcm2"), ("N foliar", "N_foliar_rel"),
                      ("H₂O foliar", "H2O_foliar_rel"), ("Biomasa", "Biomasa_rel"),
                      ("Heterogeneidad", "hetero_score"), ("PC1", "PC1"), ("PC2", "PC2"),
                  ]],
            ])
    return fig, info


@app.callback(
    Output("spectral-plot", "figure"),
    Input("spec-lotes", "value"),
)
def update_spectra(selected):
    if not selected:
        return go.Figure()
    selected = selected[:5]
    colors = px.colors.qualitative.Set2
    WATER = [(1340, 1460), (1790, 1970)]

    fig = go.Figure()
    for i, lid in enumerate(selected):
        if lid not in df_spec.index:
            continue
        y = df_spec.loc[lid].values
        fig.add_trace(go.Scatter(
            x=wl_nm, y=y, mode="lines", name=lid,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    for lo, hi in WATER:
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#1f6aa5", opacity=0.15,
                      line_width=0, annotation_text="H₂O", annotation_position="top left",
                      annotation_font_size=9, annotation_font_color="#8b949e")

    regions = [(376,500,"Blue"),(500,680,"Green"),(680,750,"Red-Edge"),
               (750,1300,"NIR"),(1300,1800,"SWIR-1"),(1800,2499,"SWIR-2")]
    region_colors = ["#4488CC","#44AA44","#FF7700","#888888","#CC8844","#994422"]
    for (xlo, xhi, lbl), col in zip(regions, region_colors):
        fig.add_vrect(x0=xlo, x1=xhi, fillcolor=col, opacity=0.04, line_width=0)
        fig.add_annotation(x=(xlo+xhi)/2, y=1.02, xref="x", yref="paper",
                           text=lbl, showarrow=False, font=dict(size=9, color=col))

    fig.update_layout(
        paper_bgcolor=C_CARD, plot_bgcolor=C_DARK, font_color=C_TEXT,
        xaxis_title="Longitud de onda (nm)", yaxis_title="Reflectancia superficial",
        xaxis=dict(range=[376, 2499], gridcolor=C_BORDER),
        yaxis=dict(gridcolor=C_BORDER),
        legend=dict(bgcolor=C_CARD, bordercolor=C_BORDER),
        title="Perfil espectral completo — 426 bandas (Tanager-1)",
        margin=dict(t=60, b=40),
    )
    return fig


@app.callback(
    Output("bio-radar", "figure"),
    Input("bio-lote", "value"),
)
def update_radar(lid):
    row = df_bio[df_bio["id_lote"] == lid]
    if row.empty:
        return go.Figure()
    r = row.iloc[0]
    cats = ["Clorofila", "N foliar", "H₂O foliar", "Efic. fotosin.", "Biomasa", "Estrés car."]
    cols = ["Cab_est_ugcm2","N_foliar_rel","H2O_foliar_rel",
            "Efic_fotosint_rel","Biomasa_rel","Estres_car_rel"]

    # Normalize to 0-100 scale
    all_vals = df_bio[cols].values
    mn = np.nanmin(all_vals, axis=0)
    mx = np.nanmax(all_vals, axis=0)
    vals = [(float(r[c]) - mn[i]) / (mx[i] - mn[i]) * 100
            if mx[i] > mn[i] else 0 for i, c in enumerate(cols)]
    vals_closed = vals + [vals[0]]
    cats_closed = cats + [cats[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed, fill="toself",
        name=lid, line_color=C_ACCENT, fillcolor=C_ACCENT,
        opacity=0.3,
    ))
    # Scene mean
    scene_mean = [(np.nanmean(df_bio[c]) - mn[i]) / (mx[i] - mn[i]) * 100
                  if mx[i] > mn[i] else 0 for i, c in enumerate(cols)]
    scene_closed = scene_mean + [scene_mean[0]]
    fig.add_trace(go.Scatterpolar(
        r=scene_closed, theta=cats_closed, fill="toself",
        name="Media escena", line_color="#8b949e", fillcolor="#8b949e",
        opacity=0.15, line_dash="dash",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=C_DARK,
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=C_BORDER,
                            tickfont=dict(color=C_TEXT, size=9)),
            angularaxis=dict(gridcolor=C_BORDER, tickfont=dict(color=C_TEXT)),
        ),
        paper_bgcolor=C_CARD, font_color=C_TEXT,
        legend=dict(bgcolor=C_CARD),
        title=f"Perfil bioquímico — {lid}",
        margin=dict(t=60, b=20),
    )
    return fig


@app.callback(
    Output("pca-biplot", "figure"),
    Input("pca-color", "value"),
)
def update_pca(color_col):
    df_plot = df_master.copy()
    is_cat = color_col == "cluster_name"

    if is_cat:
        fig = px.scatter(
            df_plot, x="PC1", y="PC2", text="id_lote",
            color="cluster_name",
            color_discrete_map=ZONE_COLORS,
            hover_data=["id_lote","NDRE_mean","Cab_est_ugcm2","hetero_score"],
            title="Biplot PCA — PC1 (agua SWIR-1) vs PC2 (N/lignina SWIR-2)",
        )
    else:
        fig = px.scatter(
            df_plot, x="PC1", y="PC2", text="id_lote",
            color=color_col, color_continuous_scale="RdYlGn",
            hover_data=["id_lote","cluster_name","hetero_score"],
            title="Biplot PCA — PC1 vs PC2",
        )
    fig.update_traces(textposition="top center", textfont_size=8, marker_size=10)
    fig.add_hline(y=0, line_dash="dash", line_color=C_BORDER)
    fig.add_vline(x=0, line_dash="dash", line_color=C_BORDER)
    fig.update_layout(
        paper_bgcolor=C_CARD, plot_bgcolor=C_DARK, font_color=C_TEXT,
        xaxis=dict(gridcolor=C_BORDER, title="PC1 — agua foliar (64.2%)"),
        yaxis=dict(gridcolor=C_BORDER, title="PC2 — N/lignina SWIR-2 (23.8%)"),
        legend=dict(bgcolor=C_CARD, bordercolor=C_BORDER),
        margin=dict(t=60, b=40),
    )
    return fig


@app.callback(
    Output("vip-plot", "figure"),
    Input("vip-var", "value"),
)
def update_vip(col):
    vip = df_vip[col].values
    wl  = df_vip["wavelength_nm"].values
    colors = np.where(vip >= 1.0, "#e74c3c", "#3498db")
    WATER = [(1340, 1460), (1790, 1970)]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=wl, y=vip, marker_color=colors, width=5, name="VIP"))
    fig.add_hline(y=1.0, line_dash="dash", line_color="white",
                  annotation_text="VIP=1.0 (umbral)", annotation_font_color=C_TEXT)
    for lo, hi in WATER:
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#1f6aa5", opacity=0.2, line_width=0)

    # Anotar top 5
    top5 = np.argsort(vip)[-5:][::-1]
    for idx in top5:
        fig.add_annotation(x=wl[idx], y=vip[idx], text=f"{wl[idx]:.0f}nm",
                           showarrow=True, arrowhead=2, arrowcolor="#e74c3c",
                           font=dict(size=9, color="#e74c3c"), yshift=5)

    fig.update_layout(
        paper_bgcolor=C_CARD, plot_bgcolor=C_DARK, font_color=C_TEXT,
        xaxis=dict(title="Longitud de onda (nm)", range=[376, 2499], gridcolor=C_BORDER),
        yaxis=dict(title="VIP score", gridcolor=C_BORDER),
        title=f"VIP scores — {col.replace('VIP_','')} | rojo=VIP>1 (importante)",
        margin=dict(t=60, b=40),
        showlegend=False,
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)
