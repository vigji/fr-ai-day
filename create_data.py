# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import faker

source_data = Path("/Users/vigji/Desktop/frAIday/Distpe25.csv")

df = pd.read_csv(source_data, sep=";").dropna(subset=["Naz"])
# Clean up typos
df["Mansioni"] = df["Mansioni"].str.strip().str.lower()
for old, new in [("asistente", "assistente"), ("fondaizoni", "fondazioni"), ("fondazione", "fondazioni")]:
    df["Mansioni"] = df["Mansioni"].str.replace(old, new)
df = df[~df["Mansioni"].isin(["basiliano"])]

df_domanda = df[["Num", "Naz", "Cantiere", "Mansioni"]]

df_offerta = df[["Cognome", "Nome", "Mansioni"]]

# df_cantieri = df[["Num", "Naz", "Cantiere"]]

# Generate random locations for each unique cantiere
unique_cantieri = df["Cantiere"].unique()
np.random.seed(42)  # For reproducibility

# Create dictionary of cantiere to random x,y coordinates
locations = {
    cantiere: (np.random.uniform(-10, 10), np.random.uniform(-10, 10))
    for cantiere in unique_cantieri
}

# Create locations dataframe
df_locations = pd.DataFrame({
    "Cantiere": list(locations.keys()),
    "location": list(locations.values())
})

# If you need to extract x,y into separate columns
df_locations["x"] = df_locations["location"].apply(lambda loc: loc[0])
df_locations["y"] = df_locations["location"].apply(lambda loc: loc[1])

# Next step


# %%
df.columns
# %%
