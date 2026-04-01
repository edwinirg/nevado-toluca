import pandas as pd

# Ruta al archivo parquet
df = pd.read_parquet("/Users/samsu/Escritorio/Final script/Scripts/salidas_cluster_serie/salidas_dbscan/preprocessed_timeseries.parquet")

print(df)
