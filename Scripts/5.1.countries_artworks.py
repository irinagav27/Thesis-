import pandas as pd
import matplotlib.pyplot as plt
"""
This script analyses the inclusivity of the dataset of this study.
First, it connects the country to the continent 
Then it counts the amount of curatorial texts per continent
The result is a graph of the amount of curatorial texts per continent
"""
file_path = "Data/Original/Dataset_Curatorial_Text.xlsx"

df = pd.read_excel(file_path)

country_to_continent = {
    # Europe
    "United Kingdom": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Netherlands": "Europe",
    "Belgium": "Europe",
    "Austria": "Europe",
    "Switzerland": "Europe",
    "Sweden": "Europe",
    "Norway": "Europe",
    "Denmark": "Europe",
    "Poland": "Europe",
    "Greece": "Europe",
    "Portugal": "Europe",
    "Czech Republic": "Europe",
    "Ireland": "Europe",
    "Finland": "Europe",
    "Hungary": "Europe",
    "Romania": "Europe",
    
    # North America
    "United States": "North America",
    "USA": "North America",
    "Canada": "North America",
    "Mexico": "North America",
    
    # South America
    "Brazil": "South America",
    "Argentina": "South America",
    "Chile": "South America",
    "Colombia": "South America",
    "Peru": "South America",
    
    # Asia
    "China": "Asia",
    "Japan": "Asia",
    "India": "Asia",
    "Thailand": "Asia",
    "Vietnam": "Asia",
    "South Korea": "Asia",
    "Indonesia": "Asia",
    "Philippines": "Asia",
    "Malaysia": "Asia",
    "Singapore": "Asia",
    "Pakistan": "Asia",
    "Bangladesh": "Asia",
    
    # Africa
    "Egypt": "Africa",
    "South Africa": "Africa",
    "Nigeria": "Africa",
    "Kenya": "Africa",
    "Ethiopia": "Africa",
    "Morocco": "Africa",
    "Algeria": "Africa",
    
    # Oceania
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    
    # Middle East
    "Iran": "Middle East",
    "Iraq": "Middle East",
    "Saudi Arabia": "Middle East",
    "United Arab Emirates": "Middle East",
    "Israel": "Middle East",
    "Turkey": "Middle East",
}


country_column = "PlaceOfOrigination"

# Remove missing / NULL values
df_countries = df[
    df[country_column].notna() &
    (df[country_column].astype(str).str.upper() != "NULL")
].copy()

# Clean spaces
df_countries[country_column] = df_countries[country_column].astype(str).str.strip()


# Add continent column
df_countries["Continent"] = df_countries[country_column].map(country_to_continent)

# Remove rows where country couldn't be mapped to continent
df_continents = df_countries[df_countries["Continent"].notna()].copy()

continent_counts = df_continents["Continent"].value_counts()

print("\nArtworks by Continent:")
print(continent_counts)
print(f"\nTotal mapped: {continent_counts.sum()}")

plt.figure(figsize=(12, 6))

continent_counts.sort_values(ascending=True).plot(kind="barh", color="steelblue")

plt.title("Number of Artworks by Continent of Origin", fontsize=14, fontweight="bold")
plt.xlabel("Number of Artworks", fontsize=12)
plt.ylabel("Continent", fontsize=12)

plt.tight_layout()

plt.show()