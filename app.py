# -*- coding: utf-8 -*-
# Application Dash - Analyse du service de vélos en libre-service (Rouen)
# Ce tableau de bord intègre toutes les visualisations identifiées dans le notebook d’analyse.
# Les graphiques sont organisés par onglets thématiques et comprennent des contrôles interactifs.
# © Projet Data – 2025

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans  # nécessaire pour l'analyse de clusters des stations
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# --- 1. CHARGEMENT ET PRÉPARATION DES DONNÉES ---

# Chargement des données depuis les fichiers CSV (station status et station information)
df_status = pd.read_csv(
    'data/data.1.csv', 
    parse_dates=['timestamp'],
    date_parser=lambda x: pd.to_datetime(x, unit='s', utc=True).tz_convert('Europe/Paris')
)
df_status = df_status.rename(columns={'id': 'station_id'})
df_info = pd.read_json('https://gbfs.urbansharing.com/lovelolibreservice.fr/station_information.json')
# Si le JSON en ligne n'est pas accessible, on peut charger df_info depuis un fichier local:
# df_info = pd.read_json('data/station_information.json')

# Extraire la table des stations depuis le JSON (structure GBFS)
df_info = pd.json_normalize(df_info['data']['stations'])
df_info = df_info[['station_id', 'name', 'address', 'cross_street', 'lat', 'lon', 'capacity']]
df_status['station_id'] = df_status['station_id'].astype(str)
df_info['station_id'] = df_info['station_id'].astype(str)
# Fusion des informations station (df_info) avec les données de statut (df_status)
df = pd.merge(df_status, df_info, on='station_id', how='left')

# Calcul du taux d'occupation (%) pour chaque enregistrement: (vélos occupés / capacité * 100)
# vélos occupés = vélos disponibles (si tous les emplacements sont occupés par des vélos)
df['taux_occupation'] = (df['available'] / (df['available'] + df['free'])) * 100

# Création de colonnes temporelles utiles
df['heure'] = df['timestamp'].dt.hour                   # heure de la journée (0-23)
df['date'] = df['timestamp'].dt.date                    # date (sans l'heure)
df['weekend'] = df['timestamp'].dt.dayofweek >= 5       # booléen: True si samedi/dimanche
# Traduction du jour de la semaine en français
jours_fr = {
    'Monday': 'Lundi', 'Tuesday': 'Mardi', 'Wednesday': 'Mercredi',
    'Thursday': 'Jeudi', 'Friday': 'Vendredi', 'Saturday': 'Samedi', 'Sunday': 'Dimanche'
}
df['jour_semaine'] = df['timestamp'].dt.day_name().map(jours_fr)
# Définition des périodes de la journée
def get_periode(h):
    if 6 <= h <= 11:
        return "Matin"
    elif 12 <= h <= 17:
        return "Midi"
    elif 18 <= h <= 23:
        return "Soir"
    else:
        return "Nuit"
df['période'] = df['heure'].apply(get_periode)

# Mise en index temporel pour les résumés par période (pour resample mensuel et journalier)
df = df.set_index('timestamp').sort_index()
df['date'] = df.index.date
df['hour'] = df.index.hour

# Calcul de la tendance mensuelle (moyennes mensuelles de vélos disponibles et de places libres)
df_month = df[['available', 'free']].resample('MS').mean().reset_index()
df_month['month'] = df_month['timestamp'].dt.strftime('%Y-%m')  # étiquette mois-année pour affichage

# Calcul de la tendance journalière (moyennes par jour)
daily_avg = df[['available', 'taux_occupation']].resample('D').mean().reset_index()
daily_avg['date'] = daily_avg['timestamp'].dt.date  # extraire la date sans heure

# Calcul du profil horaire global (moyennes par heure sur l’ensemble de la période)
hourly = df.groupby('heure')[['available', 'taux_occupation']].mean().reset_index()

# Calcul du profil horaire séparé semaine/week-end (moyennes par heure, distinguant week-end)
weekend_comp = df.groupby(['weekend', 'heure'])[['available', 'taux_occupation']].mean().reset_index()
# Remplacer les booléens par des labels clairs
weekend_comp['Jour'] = weekend_comp['weekend'].map({False: 'Semaine', True: 'Week-end'})

# Préparation des données pour la heatmap jour vs période (moyennes du taux d’occupation)
pivot = df.groupby(['jour_semaine', 'période'])['taux_occupation'].mean().unstack(level='période')
pivot_long = pivot.reset_index().melt(id_vars='jour_semaine', var_name='période', value_name='taux_occupation')

# Calcul du Top 15 des stations (taux d’occupation moyen le plus élevé)
top15_occ = df.groupby('name')['taux_occupation'].mean().reset_index()
top15_occ = top15_occ.sort_values(by='taux_occupation', ascending=False).head(15)
# Calcul du Top 15 des stations (nombre de vélos disponibles moyen le plus élevé)
top15_velo = df.groupby('name')['available'].mean().reset_index()
top15_velo = top15_velo.sort_values(by='available', ascending=False).head(15)

# Préparation des données pour les camemberts par période (Top 10 stations par vélos disponibles moyens)
pie_data = {}
for periode in ['Matin', 'Midi', 'Soir', 'Nuit']:
    df_per = df[df['période'] == periode]
    per_top10 = df_per.groupby('name')['available'].mean().reset_index()
    per_top10 = per_top10.sort_values(by='available', ascending=False).head(10)
    pie_data[periode] = per_top10

# Préparation de la table des stations (moyennes par station) pour analyse par station et carte
stations = df.groupby(['station_id', 'name', 'lat', 'lon', 'capacity']).agg({
    'available': 'mean',
    'free': 'mean',
    'taux_occupation': 'mean'
}).reset_index()
stations.rename(columns={'available': 'available_mean', 'free': 'free_mean', 'taux_occupation': 'taux_occ_mean'}, inplace=True)

# Application d’un clustering (k-means) sur les stations en fonction de leur utilisation
# Nous utilisons 3 clusters pour catégoriser les stations (par exemple: faible, moyen, fort usage)
kmeans = KMeans(n_clusters=3, random_state=0)
stations['cluster'] = kmeans.fit_predict(stations[['taux_occ_mean', 'capacity']])
# Option: on peut trier les clusters par taux d'occupation moyen pour avoir un ordre logique (0 = faible, 2 = fort)
cluster_order = stations.groupby('cluster')['taux_occ_mean'].mean().sort_values().index.tolist()
# Remapper les numéros de cluster pour correspondre à l'ordre du taux d'occupation croissant (0,1,2)
cluster_mapping = {old: new for new, old in enumerate(cluster_order)}
stations['cluster'] = stations['cluster'].map(cluster_mapping)
# station daily
station_daily = df.groupby(['name', 'date'])[['available', 'taux_occupation']].mean().reset_index()


# --- 2. CRÉATION DE L'APPLICATION DASH ---

app = dash.Dash(__name__, title="Dashboard Vélos Libre-Service")
app.layout = html.Div([
    html.H1("Tableau de bord – Vélos en libre-service à Rouen", style={'textAlign': 'center'}),
    html.Hr(),
    dcc.Tabs([
        # Onglet 1: Évolution temporelle (mensuelle & journalière)
        dcc.Tab(label="Évolution temporelle", children=[
            html.Br(),
            html.H3("Tendance mensuelle des vélos disponibles et places libres"),
            dcc.Graph(
                id='graph-monthly',
                figure=px.line(
                    df_month, x='month', y=['available', 'free'],
                    labels={'month': 'Mois', 'value': 'Nombre moyen par station', 'variable': 'Statut'},
                    title="Évolution mensuelle – Vélos disponibles vs Places libres"
                )
            ),
            html.H3("Tendance journalière"),
            html.Label("Mesure à afficher :", style={'marginRight': '1em'}),
            dcc.RadioItems(
                id='toggle-daily-metric',
                options=[
                    {'label': 'Taux d’occupation (%)', 'value': 'taux'},
                    {'label': 'Vélos disponibles', 'value': 'velo'}
                ],
                value='taux',
                labelStyle={'display': 'inline-block', 'marginRight': '1em'}
            ),
            dcc.Graph(id='graph-daily'),
                    
            # NEW: Your custom plot
            dcc.Graph(id='graph-custom',
                      figure=px.line(station_daily, x='date', y='taux_occupation', color='name', title="Taux d'occupation journalier par station"))

        ]),
        # Onglet 2: Profil horaire (jour moyen & comparaison semaine/week-end)
        dcc.Tab(label="Profil horaire", children=[
            html.Br(),
            html.H3("Profil d'une journée type"),
            html.Label("Mesure à afficher :", style={'marginRight': '1em'}),
            dcc.RadioItems(
                id='toggle-hourly-metric',
                options=[
                    {'label': 'Taux d’occupation (%)', 'value': 'taux'},
                    {'label': 'Vélos disponibles', 'value': 'velo'}
                ],
                value='taux',
                labelStyle={'display': 'inline-block', 'marginRight': '1em'}
            ),
            dcc.Graph(id='graph-hourly'),
            html.H3("Comparaison Semaine vs Week-end"),
            dcc.Graph(
                id='graph-weekend',
                figure=px.line(
                    weekend_comp, x='heure', y='taux_occupation', color='Jour',
                    labels={'heure': 'Heure', 'taux_occupation': 'Taux d\'occupation (%)', 'Jour': 'Type de jour'},
                    title="Taux d'occupation – Semaine vs Week-end"
                )
            )
        ]),
        # Onglet 3: Périodes de la journée (boxplots & camemberts)
        dcc.Tab(label="Périodes de la journée", children=[
            html.Br(),
            html.H3("Variabilité par période de la journée"),
            html.Label("Mesure à afficher :", style={'marginRight': '1em'}),
            dcc.RadioItems(
                id='toggle-box-metric',
                options=[
                    {'label': 'Taux d’occupation (%)', 'value': 'taux'},
                    {'label': 'Vélos disponibles', 'value': 'velo'}
                ],
                value='taux',
                labelStyle={'display': 'inline-block', 'marginRight': '1em'}
            ),
            dcc.Graph(id='graph-box'),
            html.H3("Répartition des vélos disponibles par station et par période"),
            html.Label("Période :", style={'marginRight': '1em'}),
            dcc.Dropdown(
                id='select-periode',
                options=[{'label': p, 'value': p} for p in ['Matin', 'Midi', 'Soir', 'Nuit']],
                value='Matin', clearable=False, style={'width': '200px', 'display': 'inline-block'}
            ),
            dcc.Graph(id='graph-pie')
        ]),
        # Onglet 4: Jour de la semaine (heatmap & comparaison hebdomadaire)
        dcc.Tab(label="Jour de la semaine", children=[
            html.Br(),
            html.H3("Taux d’occupation moyen – Jour vs Période"),
            dcc.Graph(
                id='graph-heatmap',
                figure=px.density_heatmap(
                    pivot_long, x='période', y='jour_semaine', z='taux_occupation',
                    color_continuous_scale='YlOrRd', text_auto=True,
                    labels={'période': 'Période de la journée', 'jour_semaine': 'Jour de la semaine', 'taux_occupation': 'Taux d\'occupation (%)'},
                    title="Heatmap – Taux d'occupation moyen par Jour et Période"
                )
            ),
            html.H3("Écart semaine / week-end (rappel)"),
            html.P("Le graphique ci-dessous (identique à celui du profil horaire) met en évidence les différences de taux d’occupation entre les jours de semaine et le week-end."),
            dcc.Graph(
                id='graph-weekend-2',
                figure=px.line(
                    weekend_comp, x='heure', y='taux_occupation', color='Jour',
                    labels={'heure': 'Heure', 'taux_occupation': 'Taux d\'occupation (%)', 'Jour': 'Type de jour'},
                    title="Taux d'occupation – Semaine vs Week-end"
                )
            )
        ]),
        # Onglet 5: Analyse des stations (Top 15, scatter clusters)
        dcc.Tab(label="Analyse des stations", children=[
            html.Br(),
            html.H3("Top 15 des stations"),
            html.Label("Classement par :", style={'marginRight': '1em'}),
            dcc.RadioItems(
                id='toggle-top-metric',
                options=[
                    {'label': 'Taux d’occupation (%)', 'value': 'taux'},
                    {'label': 'Vélos disponibles', 'value': 'velo'}
                ],
                value='taux',
                labelStyle={'display': 'inline-block', 'marginRight': '1em'}
            ),
            dcc.Graph(id='graph-top15'),
            html.H3("Clustering des stations selon l’usage"),
            dcc.Graph(
                id='graph-scatter',
                figure=px.scatter(
                    stations, x='taux_occ_mean', y='available_mean',
                    color=stations['cluster'].astype(str),  # couleur par cluster (catégoriel)
                    hover_name='name',
                    size='capacity', size_max=15,
                    labels={
                        'taux_occ_mean': 'Taux d’occupation moyen (%)',
                        'available_mean': 'Vélos disponibles moyens',
                        'cluster': 'Cluster',
                        'capacity': 'Capacité station'
                    },
                    title="Stations – Taux d'occupation vs Vélos disponibles (couleur = cluster)"
                )
            )
        ]),
        # Onglet 6: Distribution globale
        dcc.Tab(label="Distribution globale", children=[
            html.Br(),
            html.H3("Distribution des indicateurs"),
            html.Label("Indicateur :", style={'marginRight': '1em'}),
            dcc.RadioItems(
                id='toggle-hist-metric',
                options=[
                    {'label': 'Taux d’occupation (%)', 'value': 'taux'},
                    {'label': 'Vélos disponibles', 'value': 'velo'}
                ],
                value='taux',
                labelStyle={'display': 'inline-block', 'marginRight': '1em'}
            ),
            dcc.Graph(id='graph-hist'),
            html.H3("Corrélation entre variables"),
            dcc.Graph(
                id='graph-corr',
                figure=px.imshow(
                    df[['available', 'free', 'capacity', 'taux_occupation']].corr(),
                    text_auto=True, color_continuous_scale='RdBu', zmin=-1, zmax=1,
                    labels={'x': 'Variable', 'y': 'Variable'},
                    title="Corrélation entre vélos dispo, places libres, capacité et taux d'occupation"
                )
            )
        ]),
        # Onglet 7: Carte des stations
        dcc.Tab(label="Carte des stations", children=[
            html.Br(),
            html.H3("Carte interactive des stations – Filtrée par date et heure"),

            html.Label("Date :"),
            dcc.DatePickerSingle(
                id='carte-date-picker',
                min_date_allowed=df['date'].min(),
                max_date_allowed=df['date'].max(),
                date=df['date'].min()
            ),
            html.Br(), html.Br(),

            html.Label("Heure :"),
            dcc.Dropdown(
                id='carte-hour-dropdown',
                options=[{'label': f'{h}h', 'value': h} for h in sorted(df['hour'].unique())],
                value=df['hour'].min(),
                clearable=False,
                style={'width': '150px'}
            ),
            html.Br(),

            dcc.Graph(id='carte-station-map')
        ]),
        # Onglet 8: Aide à la décision
        dcc.Tab(label="Aide à la décision", children=[
            html.Br(),
            html.H3("Stations avec le plus de vélos disponibles à un moment donné"),

            html.Label("Date :"),
            dcc.DatePickerSingle(
                id='decision-date',
                min_date_allowed=df['date'].min(),
                max_date_allowed=df['date'].max(),
                date=df['date'].min()
            ),
            html.Br(),

            html.Label("Heure :"),
            dcc.Dropdown(
                id='decision-hour',
                options=[{'label': f"{h}h", 'value': h} for h in sorted(df['hour'].unique())],
                value=df['hour'].min(),
                clearable=False,
                style={'width': '150px'}
            ),
            dcc.Graph(id='graph-top-disponibles'),

            html.H3("Comparer deux stations à une heure donnée"),
            html.Div([
                html.Label("Station A :"),
                dcc.Dropdown(
                    id='station-a',
                    options=[{'label': n, 'value': n} for n in sorted([str(n) for n in df['name'].dropna().unique()])
],
                    value=sorted([str(n) for n in df['name'].dropna().unique()])
[0],
                    style={'width': '45%', 'display': 'inline-block', 'margin-right': '5%'}
                ),
                html.Label("Station B :"),
                dcc.Dropdown(
                    id='station-b',
                    options=[{'label': n, 'value': n} for n in sorted([str(n) for n in df['name'].dropna().unique()])
],
                    value=sorted([str(n) for n in df['name'].dropna().unique()])
[1],
                    style={'width': '45%', 'display': 'inline-block'}
                ),
            ]),
            html.Br(),
            dcc.Graph(id='graph-compare-stations'),

            html.H3("Stations les plus fiables (variabilité faible)"),
            dcc.Graph(id='graph-fiabilite')
        ])

         
    ])
])

# --- 3. CALLBACKS POUR INTERACTIVITÉ ---

# Mise à jour du graphique journalièr (taux d'occupation vs vélos disponibles)
@app.callback(
    Output('graph-daily', 'figure'),
    Input('toggle-daily-metric', 'value')
)
def update_daily_graph(metric):
    if metric == 'taux':
        fig = px.line(
            daily_avg, x='date', y='taux_occupation',
            labels={'date': 'Date', 'taux_occupation': 'Taux d\'occupation (%)'},
            title="Évolution journalière du taux d'occupation"
        )
    else:
        fig = px.line(
            daily_avg, x='date', y='available',
            labels={'date': 'Date', 'available': 'Vélos disponibles (moyens)'},
            title="Évolution journalière du nombre de vélos disponibles"
        )
    fig.update_layout(xaxis_tickformat='%Y-%m-%d')
    return fig

# Mise à jour du graphique horaire (taux d'occupation vs vélos disponibles)
@app.callback(
    Output('graph-hourly', 'figure'),
    Input('toggle-hourly-metric', 'value')
)
def update_hourly_graph(metric):
    if metric == 'taux':
        fig = px.line(
            hourly, x='heure', y='taux_occupation',
            labels={'heure': 'Heure', 'taux_occupation': 'Taux d\'occupation (%)'},
            title="Évolution horaire du taux d'occupation (moyenne sur la période)"
        )
    else:
        fig = px.line(
            hourly, x='heure', y='available',
            labels={'heure': 'Heure', 'available': 'Vélos disponibles (moyenne)'},
            title="Évolution horaire du nombre de vélos disponibles (moyenne sur la période)"
        )
    fig.update_traces(mode='lines+markers')
    return fig

# Mise à jour du boxplot par période (taux d'occupation vs vélos disponibles)
@app.callback(
    Output('graph-box', 'figure'),
    Input('toggle-box-metric', 'value')
)
def update_box_graph(metric):
    if metric == 'taux':
        fig = px.box(
            df, x='période', y='taux_occupation', category_orders={"période": ["Nuit", "Matin", "Midi", "Soir"]},
            labels={'période': 'Période', 'taux_occupation': 'Taux d\'occupation (%)'},
            title="Distribution du taux d'occupation par période de la journée"
        )
    else:
        fig = px.box(
            df, x='période', y='available', category_orders={"période": ["Nuit", "Matin", "Midi", "Soir"]},
            labels={'période': 'Période', 'available': 'Vélos disponibles'},
            title="Distribution du nombre de vélos disponibles par période de la journée"
        )
    return fig

# Mise à jour du camembert selon la période sélectionnée
@app.callback(
    Output('graph-pie', 'figure'),
    Input('select-periode', 'value')
)
def update_pie_graph(periode):
    data = pie_data.get(periode, pd.DataFrame(columns=['name', 'available']))
    fig = px.pie(
        data, values='available', names='name',
        title=f"Répartition des vélos disponibles – {periode}"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

# Mise à jour du Top 15 (taux d'occupation vs vélos disponibles)
@app.callback(
    Output('graph-top15', 'figure'),
    Input('toggle-top-metric', 'value')
)
def update_top15_graph(metric):
    if metric == 'taux':
        fig = px.bar(
            top15_occ, x='taux_occupation', y='name', orientation='h',
            labels={'name': 'Station', 'taux_occupation': 'Taux d\'occupation moyen (%)'},
            title="Top 15 des stations les plus occupées"
        )
    else:
        fig = px.bar(
            top15_velo, x='available', y='name', orientation='h',
            labels={'name': 'Station', 'available': 'Vélos disponibles moyens'},
            title="Top 15 des stations avec le plus de vélos disponibles"
        )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})  # trier du plus grand au plus petit
    return fig

# Mise à jour de l'histogramme de distribution (taux d'occupation vs vélos disponibles)
@app.callback(
    Output('graph-hist', 'figure'),
    Input('toggle-hist-metric', 'value')
)
def update_hist_graph(metric):
    if metric == 'taux':
        fig = px.histogram(
            df, x='taux_occupation', nbins=30,
            labels={'taux_occupation': 'Taux d\'occupation (%)', 'count': 'Nombre d’enregistrements'},
            title="Distribution du taux d'occupation (sur l'ensemble des relevés)"
        )
    else:
        fig = px.histogram(
            df, x='available', nbins=30,
            labels={'available': 'Vélos disponibles', 'count': 'Nombre d’enregistrements'},
            title="Distribution du nombre de vélos disponibles (sur l'ensemble des relevés)"
        )
    return fig

# (Aucun callback nécessaire pour la heatmap, le scatter, la corrélation et la carte 
# car ils sont statiques une fois les données calculées)

@app.callback(
    Output('carte-station-map', 'figure'),
    Input('carte-date-picker', 'date'),
    Input('carte-hour-dropdown', 'value')
)
def update_carte_map(selected_date, selected_hour):
    d = pd.to_datetime(selected_date).date()
    filtered = df[(df['date'] == d) & (df['hour'] == selected_hour)]

    if filtered.empty:
        fig = px.scatter_mapbox(
            lat=[], lon=[], zoom=13, mapbox_style='open-street-map',
            title="Aucune donnée pour ce créneau"
        )
        return fig

    fig = px.scatter_mapbox(
        filtered, lat='lat', lon='lon', color='available',
        size='available', hover_name='name',
        hover_data={
        'available': True,
        'free': True,
        'capacity': True,
        'taux_occupation': ':.1f'
    },

        zoom=13, center={'lat': filtered['lat'].mean(), 'lon': filtered['lon'].mean()},
        mapbox_style='open-street-map',
        title=f"Disponibilité des stations – {d} à {selected_hour}h"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig
@app.callback(
    Output('graph-top-disponibles', 'figure'),
    Input('decision-date', 'date'),
    Input('decision-hour', 'value')
)
def update_top_stations(date_selected, hour_selected):
    d = pd.to_datetime(date_selected).date()
    filtered = df[(df['date'] == d) & (df['hour'] == hour_selected)]
    if filtered.empty:
        return px.bar(title="Aucune donnée disponible pour ce créneau")

    top = filtered.groupby('name')['available'].mean().sort_values(ascending=False).head(10).reset_index()
    fig = px.bar(top, x='available', y='name', orientation='h',
                 labels={'available': 'Vélos disponibles', 'name': 'Station'},
                 title=f"Top 10 stations – {d} à {hour_selected}h")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig
@app.callback(
    Output('graph-compare-stations', 'figure'),
    Input('station-a', 'value'),
    Input('station-b', 'value'),
    Input('decision-date', 'date'),
    Input('decision-hour', 'value')
)
def compare_stations(station_a, station_b, date_selected, hour_selected):
    d = pd.to_datetime(date_selected).date()
    filtered = df[(df['date'] == d) & (df['hour'] == hour_selected)]

    df_a = filtered[filtered['name'] == station_a]
    df_b = filtered[filtered['name'] == station_b]

    values = {
        'Station': [station_a, station_b],
        'Vélos disponibles': [
            df_a['available'].mean() if not df_a.empty else 0,
            df_b['available'].mean() if not df_b.empty else 0
        ]
    }
    comp_df = pd.DataFrame(values)
    fig = px.bar(comp_df, x='Station', y='Vélos disponibles',
                 title=f"Disponibilité comparée à {hour_selected}h le {d}")
    return fig
@app.callback(
    Output('graph-fiabilite', 'figure'),
    Input('decision-date', 'date')  # utilisé juste pour forcer un refresh
)
def show_fiabilite(_):
    std_df = df.groupby('name')['available'].std().reset_index()
    std_df = std_df.sort_values('available').head(10)
    fig = px.bar(std_df, x='available', y='name', orientation='h',
                 labels={'available': 'Écart-type (variabilité)', 'name': 'Station'},
                 title="Top 10 des stations les plus fiables (faible variabilité)")
    return fig

# --- Lancement du serveur Dash ---
if __name__ == '__main__':
 app.run(debug=False)
