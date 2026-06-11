import sqlite3
import pandas as pd

conn = sqlite3.connect("./data/labour_market.db")
df = pd.read_sql("SELECT * FROM v_clustering_base", conn)
df.to_csv("./data/clustering_data.csv", index=False)

df_clusters = pd.read_sql("SELECT * FROM clusters", conn)
df_clusters.to_csv("./data/clusters.csv", index=False)
conn.close()