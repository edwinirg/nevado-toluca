import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
# Cargar NDVI preprocesado y clusters
ndvi = pd.read_csv("/Users/samsu/Escritorio/Final script/salidas_dbscan/preprocessed_timeseries.csv", parse_dates=["date"])
clusters = pd.read_csv("/Users/samsu/Escritorio/Final script/salidas_dbscan/clusters_por_anio.csv")

subzonas = [c for c in ndvi.columns if c not in ("date","muestra_elegida")]
ndvi["year"] = ndvi["date"].dt.year

year = 2021
G = nx.Graph()
zones = clusters.query("year == @year")

for _, row in zones.iterrows():
    G.add_node(row["subzona"], cluster=row["cluster"])

# Conectar subzonas del mismo clúster
for c in zones["cluster"].unique():
    nodes = zones[zones["cluster"] == c]["subzona"].tolist()
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            G.add_edge(nodes[i], nodes[j])

pos = nx.spring_layout(G, seed=42)
colors = [zones.set_index("subzona").loc[node, "cluster"] for node in G.nodes]

nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.tab10, node_size=1200, font_size=10)
plt.title(f"Subzonas agrupadas por clúster - {year}")
plt.show()