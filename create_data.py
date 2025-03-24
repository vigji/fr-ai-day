# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import faker

# Create a Faker instance for generating names
fake = faker.Faker()

source_data = Path("/Users/vigji/Desktop/frAIday/Distpe25.csv")

df = pd.read_csv(source_data, sep=";").dropna(subset=["Naz"])
# Clean up typos
df["Mansioni"] = df["Mansioni"].str.strip().str.lower()
for old, new in [("asistente", "assistente"), ("fondaizoni", "fondazioni"), ("fondazione", "fondazioni")]:
    df["Mansioni"] = df["Mansioni"].str.replace(old, new)
df = df[~df["Mansioni"].isin(["basiliano"])]

# df_cantieri = df[["Num", "Naz", "Cantiere"]]

# Generate random locations for each unique cantiere

def generate_random_location(min_val=-10, max_val=10):
    """Generate a random (x,y) coordinate tuple within specified range"""
    return (np.random.uniform(min_val, max_val), np.random.uniform(min_val, max_val))


unique_cantieri = df["Cantiere"].unique()
np.random.seed(42)  # For reproducibility

# Create dictionary of cantiere to random x,y coordinates
locations = {
    cantiere: generate_random_location()
    for cantiere in unique_cantieri
}

df = df.copy()
df["location"] = df["Cantiere"].map(locations)

for i, coord in enumerate(["x", "y"]):
    df[coord] = df["location"].apply(lambda loc: loc[i])


df_domanda = df[["Num", "Naz", "Cantiere", "Mansioni", "location"]].copy()

# Replace "Cantiere" with random geographic location name using faker
city_mapping = {cantiere: fake.city() for cantiere in unique_cantieri}
df_domanda["Cantiere"] = df_domanda["Cantiere"].map(city_mapping)

df_offerta = df[["Cognome", "Nome", "Mansioni", "location"]].copy()

# Replace names and surnames in df_offerta with random ones
df_offerta["Nome"] = [fake.first_name() for _ in range(len(df_offerta))]
df_offerta["Cognome"] = [fake.last_name() for _ in range(len(df_offerta))]

# Add some noise to the coordinates
noise = 1
df_offerta["x"] = df_offerta["x"] + np.random.uniform(-noise, noise, len(df_offerta))
df_offerta["y"] = df_offerta["y"] + np.random.uniform(-noise, noise, len(df_offerta))

dest_dir = Path(__file__).parent / "assets"
dest_dir.mkdir(exist_ok=True)

df_offerta.to_csv(dest_dir / "offerta.csv", index=False)
df_domanda.to_csv(dest_dir / "domanda.csv", index=False)

plt.figure(figsize=(10, 10))
plt.scatter(df_offerta["x"], df_offerta["y"], label="Offerta", s=5)
plt.scatter(df_domanda["x"], df_domanda["y"], label="Domanda", s=5)
# plt.legend()
plt.show()


# %%

# %%
