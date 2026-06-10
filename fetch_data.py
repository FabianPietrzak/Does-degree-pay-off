import sqlite3
import time
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os.path

DB_PATH = "data/labour_market.db"

# limit our analysis to the EU27 countries to keep the scope clean and comparable
EU27 = [
    "AT","BE","BG","CY","CZ","DE","DK","EE","EL","ES",
    "FI","FR","HR","HU","IE","IT","LT","LU","LV","MT",
    "NL","PL","PT","RO","SE","SI","SK"
]

YEARS = list(range(2010, 2024))

BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

def fetch_eurostat(dataset: str, params: dict) -> pd.DataFrame:
    """
    Downloads a dataset from Eurostat's JSON-stat API and parses it.
    Eurostat's JSON-stat 2.0 format is tricky. It doesn't always return 
    the same data structures (e.g., values can be lists or dicts), and it omits 
    index keys for single-value dimensions. This parser handles those edge cases.
    """
    url = f"{BASE_URL}/{dataset}"
    params["format"] = "JSON"
    params["lang"] = "EN"

    print(f"Downloading: {dataset}...")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    dims = data["id"]     # List of active dimension names (e.g., ['freq', 'unit', 'geo', 'time'])
    sizes = data["size"]   # Size of each dimension (used to calculate coordinate offsets)
    
    # SAFE EXTRACTION OF CATEGORY LABELS:
    # If a dimension size is 1 (e.g., sex='T'), Eurostat often skips the "index" key.
    # Also, Python dicts don't guarantee order, so i must sort by index value if present.
    dim_labels = {}
    for d in dims:
        category = data["dimension"][d]["category"]
        if "index" in category:
            idx = category["index"]
            if isinstance(idx, list):
                dim_labels[d] = idx
            elif isinstance(idx, dict):
                # Sort keys by their numeric index position to keep the exact API order
                dim_labels[d] = sorted(idx.keys(), key=lambda k: idx[k])
            else:
                dim_labels[d] = list(idx)
        else:
            # Fallback when 'index' is missing (usually when size is 1)
            dim_labels[d] = list(category["label"].keys())

    values = data["value"]
    rows = []

    # VALUE PARSING WORKAROUND:
    # Eurostat returns 'value' as a dictionary if the data is sparse (to save space),
    # or as a flat list if the data is dense.
    if isinstance(values, dict):
        iterator = ((int(k), v) for k, v in values.items())
    elif isinstance(values, list):
        iterator = enumerate(values)
    else:
        iterator = []

    for flat_idx, val in iterator:
        if val is None:
            continue  # Skip missing observations early
        
        # reconstruct multi-dimensional coordinates from a flat 1D index
        coords = {}
        remainder = flat_idx
        for dim, size in zip(reversed(dims), reversed(sizes)):
            coords[dim] = dim_labels[dim][remainder % size]
            remainder //= size
        coords["value"] = val
        rows.append(coords)

    df = pd.DataFrame(rows)
    
    # SAFEGUARD: If the query parameters returned nothing, raise a clear error
    if df.empty:
        raise ValueError(
            f"No data returned for dataset '{dataset}'. "
            f"Please verify if your API query parameter filters are correct."
        )
        
    return df


# 1. OVERQUALIFICATION RATE [lfsa_eoqgan]
def fetch_overqualification() -> pd.DataFrame:
    params = {
        "sex": "T",          # Total (Men + Women)
        "age": "Y25-64",     # Prime working age group
        "citizen": "TOTAL",  # All citizens (native + foreign-born combined)
    }
    df = fetch_eurostat("lfsa_eoqgan", params)
    df = df[df["geo"].isin(EU27)]
    df = df[df["time"].astype(int).isin(YEARS)]
    df = df.rename(columns={"value": "overqualification_rate", "time": "year"})
    df["year"] = df["year"].astype(int)
    return df[["geo", "year", "overqualification_rate"]]


# 2. LABOUR COST INDEX [lc_lci_lev]
def fetch_labour_cost() -> pd.DataFrame:
    params = {
        "unit": "EUR",           # Hourly wage in Euros (Eurostat uses EUR for this dataset)
        "lcstruct": "D1_D4_MD5", # Total labour costs
        "nace_r2": "B-S_X_O",    # Industry, construction and services (except public administration, defense, compulsory social security)
    }
    df = fetch_eurostat("lc_lci_lev", params)
    df = df[df["geo"].isin(EU27)]
    df = df[df["time"].astype(int).isin(YEARS)]
    df = df.rename(columns={"value": "labour_cost_eur_h", "time": "year"})
    df["year"] = df["year"].astype(int)
    return df[["geo", "year", "labour_cost_eur_h"]]


# 3. YOUTH UNEMPLOYMENT RATE (15-29) [yth_empl_020]
def fetch_youth_unemployment() -> pd.DataFrame:
    params = {
        "sex": "T",
        "age": "Y15-29",
        "unit": "PC",         # Percentage of active population
        "c_birth": "TOTAL",   # Total (native + foreign-born combined)
    }
    df = fetch_eurostat("yth_empl_020", params)
    df = df[df["geo"].isin(EU27)]
    df = df[df["time"].astype(int).isin(YEARS)]
    df = df.rename(columns={"value": "youth_unemployment_rate", "time": "year"})
    df["year"] = df["year"].astype(int)
    return df[["geo", "year", "youth_unemployment_rate"]]


# 4. TERTIARY EDUCATION RATE [edat_lfse_03]
def fetch_tertiary_education() -> pd.DataFrame:
    params = {
        "sex": "T",
        "age": "Y25-64",
        "unit": "PC",        # Percentage of total population
        "isced11": "ED5-8",  # Tertiary education (Short-cycle tertiary, Bachelor, Master, Doctorate)
    }
    df = fetch_eurostat("edat_lfse_03", params)
    df = df[df["geo"].isin(EU27)]
    df = df[df["time"].astype(int).isin(YEARS)]
    df = df.rename(columns={"value": "tertiary_edu_rate", "time": "year"})
    df["year"] = df["year"].astype(int)
    return df[["geo", "year", "tertiary_edu_rate"]]


# DATABASE EXPORT
def save_to_sqlite(dfs: dict[str, pd.DataFrame]):
    conn = sqlite3.connect(DB_PATH)
    for table_name, df in dfs.items():
        # Double check to prevent duplicates from breaking relational model
        df_clean = df.drop_duplicates(subset=["geo", "year"])
        df_clean.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Table '{table_name}' — {len(df_clean)} records saved")

    # Creating a base SQL view that joins all tables on country (geo) and year.
    # This acts as the clean flat dataset for Power BI and Python clustering.
    conn.execute("DROP VIEW IF EXISTS v_clustering_base")
    conn.execute("""
        CREATE VIEW v_clustering_base AS
        SELECT
            o.geo,
            o.year,
            o.overqualification_rate,
            lc.labour_cost_eur_h,
            yu.youth_unemployment_rate,
            te.tertiary_edu_rate
        FROM overqualification o
        LEFT JOIN labour_cost lc ON o.geo = lc.geo AND o.year = lc.year
        LEFT JOIN youth_unemployment yu ON o.geo = yu.geo AND o.year = yu.year
        LEFT JOIN tertiary_education te ON o.geo = te.geo AND o.year = te.year
        WHERE o.overqualification_rate IS NOT NULL
    """)
    conn.commit()
    conn.close()
    print(f"Database saved: {DB_PATH}")


# CLUSTERING ENGINE (K-Means)
def run_clustering():
    conn = sqlite3.connect(DB_PATH)
    # segmenting countries using the most recent complete year (e.g., 2022)
    df = pd.read_sql("SELECT * FROM v_clustering_base WHERE year = 2022", conn)
    conn.close()

    if df.empty:
        print("\nNo data found for the year 2022 in the view. Clustering aborted.")
        return

    features = ["overqualification_rate", "labour_cost_eur_h","youth_unemployment_rate", "tertiary_edu_rate"]
    #df.info()
    #print(df)
    X = df[features].values
    
    # Even if i filtered duplicates, some countries might have missing years.
    # We use median imputation to keep the dataset complete without throwing errors.
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Segmenting countries into groups (arbitrary starting k)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    df["cluster_id"] = kmeans.fit_predict(X_scaled)

    # Inverse transform the cluster centers to see what each group represents in real values
    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features
    )
    print("\nCluster Centers (Original Scale)")
    print(centers.round(1).to_string())

    # Save mapping of country/year to cluster ID to enrich our SQLite database
    conn = sqlite3.connect(DB_PATH)
    df[["geo", "year", "cluster_id"]].to_sql(
        "clusters", conn, if_exists="replace", index=False
    )
    conn.close()
    print("\nTable 'clusters' saved to SQLite")

    # Set style for a clean look
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    # i choose Labour Cost and Tertiary Education as our X and Y axes.
    # Hue and style are determined by the assigned Cluster ID.
    sns.scatterplot(
        data=df,
        x="labour_cost_eur_h",
        y="tertiary_edu_rate",
        hue="cluster_id",
        style="cluster_id",
        palette="deep",
        s=120,          # Marker size
        alpha=0.8
    )

    # text labels for each country next to their data points
    for i in range(df.shape[0]):
        plt.text(
            x=df["labour_cost_eur_h"].iloc[i] + 0.4,   # Offset on X axis
            y=df["tertiary_edu_rate"].iloc[i] + 0.3,   # Offset on Y axis
            s=df["geo"].iloc[i],                       # Country code 
            fontsize=9,
            weight='semibold',
            color='#333333',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.1')
        )

    plt.title("EU Countries Clustering - 2022 Data\nLabour Cost vs. Tertiary Education Rate", fontsize=14, pad=15)
    plt.xlabel("Labour Cost (EUR / hour)", fontsize=12)
    plt.ylabel("Tertiary Education Rate (%)", fontsize=12)
    plt.legend(title="Cluster ID", loc="upper left")
    

    plt.savefig("data/clustering_2022.png", dpi=300, bbox_inches="tight")
    print("\nPlot saved as 'clustering_2022.png'")
    
    plt.show()

if __name__ == "__main__":

    if os.path.isfile(DB_PATH) is not True:
        print("Fetching Data from Eurostat\n")

        datasets = {}
        fetchers = {
            "overqualification":   fetch_overqualification,
            "labour_cost":         fetch_labour_cost,
            "youth_unemployment":  fetch_youth_unemployment,
            "tertiary_education":  fetch_tertiary_education,
        }

        # Execute all download tasks sequentially
        for name, fn in fetchers.items():
            try:
                datasets[name] = fn()
                time.sleep(1)  # Respect Eurostat API limits and prevent rate limiting
            except Exception as e:
                print(f"Error in {name}: {e}")

        print("\nSaving to SQLite\n")
        # Only run database integration and clustering if all datasets are successfully fetched
        if len(datasets) == len(fetchers):
            save_to_sqlite(datasets)
            print("\nRunning K-Means Clustering (Year 2022)\n")
            run_clustering()
        else:
            print("Database save skipped because one or more datasets failed to download.")
    else:
        run_clustering()

    print("\nDONE")