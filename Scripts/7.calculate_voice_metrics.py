"""
This is the script that will calculate the curatorial voice for the adaptations.
It will calculate the amount of curatorial voice based on the following metrics:
- hedging
- attribution
- art terminology density

Lists expanded and verified against the full dataset of 699 curatorial texts.
All terms included, including broader vocabulary present in curatorial writing.
"""

import pandas as pd
import os
import re
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

HEDGING_MARKERS = [
    # Multi-word phrases (matched before single words)
    "can be seen as",
    "appears to",
    "seems to",
    "seem to",
    "appear to",
    "is thought to",
    "is believed to",
    "one interpretation",
    "it is possible",
    "may indicate",
    "can suggest",
    "may represent",
    "may have been",
    "may suggest",
    "may reflect",
    "it has been suggested",
    "it is often said",
    "it seems",
    "it appears",
    "almost certainly",
    "most likely",
    "is regarded as",
    "is seen as",
    "is thought",
    "is not certain",
    "is likely",
    "are likely",
    "in all likelihood",
    "tends to",
    "tend to",
    "to some extent",
    "to a certain degree",
    "in some ways",
    "in some respects",
    "in part",
    # Single-word hedges
    "may",
    "might",
    "could",
    "perhaps",
    "possibly",
    "probably",
    "likely",
    "appears",
    "seems",
    "suggests",
    "suggest",
    "suggested",
    "indicates",
    "potentially",
    "apparently",
    "evidently",
    "often",
    "sometimes",
    "frequently",
    "typically",
    "generally",
    "arguably",
    "presumably",
    "conceivably",
    "ostensibly",
    "purportedly",
    "reportedly",
    "roughly",
    "approximately",
    "almost",
    "nearly",
    "around",
    "about",
]

ATTRIBUTION_MARKERS = [
    # Multi-word phrases
    "according to",
    "as noted by",
    "as described by",
    "as observed by",
    "as argued by",
    "as the artist recalled",
    "as the artist noted",
    "as the artist wrote",
    "as the artist explained",
    "as written by",
    "as reported by",
    "as stated by",
    "as cited in",
    "as recorded in",
    "research indicates",
    "scholars suggest",
    "scholars note",
    "scholars believe",
    "scholars have noted",
    "scholars have argued",
    "scholars have suggested",
    "scholars have identified",
    "some scholars",
    "critics argue",
    "critics have argued",
    "historians contend",
    "documented in",
    "is documented",
    "are documented",
    "have been documented",
    "it is recorded",
    "it was recorded",
    "it has been recorded",
    "it is documented",
    "is well documented",
    "the artist stated",
    "the artist explained",
    "the artist described",
    "the artist depicted",
    "the artist noted",
    "the artist wrote",
    "the artist recalled",
    "the artist shared",
    "the artist said",
    "the artist believed",
    "the artist intended",
    "the artist reflected",
    "the artist observed",
    "art historians",
    "art historical",
    "recent scholarship",
    "museum records",
    "historical accounts",
    "archival sources",
    "contemporary accounts",
    "contemporary sources",
    "primary sources",
    "inventory records",
    "evidence suggests",
    "evidence indicates",
    "sources indicate",
    "sources suggest",
    "tradition holds",
    "tradition suggests",
    "it is traditionally",
    "traditionally attributed",
    "is attributed",
    "has been attributed",
    "attributed to",
    # Single-word
    "inscribed",
    "documented",
]

ART_TERMS = [
    # ── Techniques & processes ──────────────────────────────────────────────────
    "watercolor",
    "gouache",
    "oil paint",
    "acrylic",
    "woodblock",
    "woodcut",
    "engraving",
    "drypoint",
    "aquatint",
    "screenprint",
    "mezzotint",
    "monotype",
    "pochoir",
    "intaglio",
    "etching",
    "lithograph",
    "calligraphy",
    "brushwork",
    "impasto",
    "underpainting",
    "gilding",
    "casting",
    "firing",
    "carving",
    "modeling",
    "burnishing",
    "glazing",
    "gilded",
    "gilt",
    "chiaroscuro",
    "sfumato",
    "foreshortening",
    "cross-hatching",
    "grisaille",
    "tempera",
    "fresco",
    "encaustic",
    # ── Materials & media ───────────────────────────────────────────────────────
    "bronze",
    "marble",
    "terracotta",
    "stoneware",
    "earthenware",
    "porcelain",
    "ceramic",
    "enamel",
    "mosaic",
    "tapestry",
    "textile",
    "embroidery",
    "silk",
    "linen",
    "parchment",
    "vellum",
    "papyrus",
    "ivory",
    "lacquer",
    "jade",
    "gold leaf",
    "silver",
    "granite",
    "limestone",
    "sandstone",
    "basalt",
    "alabaster",
    "obsidian",
    "plaster",
    "pigment",
    "glaze",
    "canvas",
    "gesso",
    "varnish",
    "ink",
    "woven",
    # ── Formal & compositional terms ────────────────────────────────────────────
    "composition",
    "perspective",
    "palette",
    "motif",
    "figuration",
    "foreground",
    "background",
    "symmetry",
    "asymmetry",
    "negative space",
    "picture plane",
    "vanishing point",
    "ground line",
    "chroma",
    "hue",
    "texture",
    "contour",
    "form",
    "line",
    "tone",
    "value",
    "trompe l'oeil",
    "contrapposto",
    # ── Object types ────────────────────────────────────────────────────────────
    "portrait",
    "landscape",
    "sculpture",
    "painting",
    "drawing",
    "print",
    "photograph",
    "medium",
    "altarpiece",
    "triptych",
    "diptych",
    "polyptych",
    "predella",
    "manuscript",
    "relief",
    "bas-relief",
    "high relief",
    "bust",
    "figurine",
    "statuette",
    "vessel",
    "ewer",
    "amphora",
    "sarcophagus",
    "funerary",
    "reliquary",
    "codex",
    "scroll",
    "album",
    "folio",
    "collage",
    "assemblage",
    "installation",
    "still life",
    "genre painting",
    "history painting",
    "vanitas",
    "memento mori",
    "ex-voto",
    "illuminated",
    "icon",
    "narrative",
    # ── Styles, periods & movements ─────────────────────────────────────────────
    "renaissance",
    "baroque",
    "gothic",
    "romanesque",
    "byzantine",
    "neoclassical",
    "rococo",
    "mannerism",
    "mannerist",
    "impressionist",
    "expressionist",
    "surrealism",
    "cubism",
    "dadaism",
    "fauvism",
    "futurism",
    "minimalism",
    "symbolism",
    "naturalism",
    "realism",
    "abstraction",
    "abstract",
    "avant-garde",
    "modernist",
    "conceptual",
    "figurative",
    "allegorical",
    "devotional",
    "votive",
    "pictorial",
    "decorative",
    "ornamental",
    "dynasty",
    "period",
    # ── Institutional & art-historical terms ────────────────────────────────────
    "provenance",
    "iconography",
    "attribution",
    "dating",
    "conservation",
    "restoration",
    "patina",
    "commission",
    "patron",
    "patronage",
    "workshop",
    "studio",
    "atelier",
    "school of",
    "circle of",
    "follower of",
    "manner of",
    "style of",
    "catalogue",
    "monograph",
    "retrospective",
    "acquisition",
    "bequest",
    "donation",
    "collection",
    "exhibition",
    "loan",
    "oeuvre",
    "verso",
    "recto",
    "cartouche",
    "putto",
    "putti",
    "after",
]


def calculate_voice_metrics(text):
    """Calculate voice metrics for the texts."""

    if pd.isna(text) or str(text).strip() == "":
        return {
            'hedging': 0,
            'attribution': 0,
            'art_terms': 0
        }

    text = str(text)
    text_lower = text.lower()

    words = re.findall(r'\b[\w-]+\b', text_lower)
    word_count = len(words)

    if word_count == 0:
        return {
            'hedging': 0,
            'attribution': 0,
            'art_terms': 0
        }

    # Hedging density
    hedging_count = sum(
        text_lower.count(marker)
        for marker in HEDGING_MARKERS
    )
    hedging_density = (hedging_count / word_count) * 100

    # Attribution density
    attribution_count = sum(
        text_lower.count(marker)
        for marker in ATTRIBUTION_MARKERS
    )
    attribution_density = (attribution_count / word_count) * 100

    # Art terminology density
    art_term_count = sum(
        text_lower.count(term)
        for term in ART_TERMS
    )
    art_term_density = (art_term_count / word_count) * 100

    return {
        'hedging': hedging_density,
        'attribution': attribution_density,
        'art_terms': art_term_density
    }


print("=" * 70)
print("CALCULATING VOICE METRICS")
print("=" * 70)

print(f"\nList sizes:")
print(f"  Hedging markers:     {len(HEDGING_MARKERS)}")
print(f"  Attribution markers: {len(ATTRIBUTION_MARKERS)}")
print(f"  Art terms:           {len(ART_TERMS)}")

print("\nLoading dataset...")

df = pd.read_csv(
    'Data/Merging_Files/dataset_all_adaptations.csv',
    encoding='utf-8-sig'
)

print(f"Found {len(df)} texts")

text_columns = [
    'CuratorialText',
    'lexical_adapted',
    'persona1_adapted',
    'persona2_adapted',
    'persona3_adapted'
]

# Calculate the metrics
for col in text_columns:

    if col not in df.columns:
        print(f"Skipping {col} - column not found")
        continue

    print(f"\nProcessing {col}...")

    metrics_list = []

    for text in tqdm(df[col], desc=f"  {col}"):
        metrics = calculate_voice_metrics(text)
        metrics_list.append(metrics)

    df[f'{col}_hedging'] = [m['hedging'] for m in metrics_list]
    df[f'{col}_attribution'] = [m['attribution'] for m in metrics_list]
    df[f'{col}_art_terms'] = [m['art_terms'] for m in metrics_list]


output_dir = 'Data/Voice_Metrics'
os.makedirs(output_dir, exist_ok=True)

output_path = f'{output_dir}/dataset_with_voice_metrics.csv'

print("\nSaving dataset with voice metrics...")
df.to_csv(output_path, index=False, encoding='utf-8-sig')

# Summary of the run
print("\n" + "=" * 70)
print("SUMMARY: AVERAGE VOICE METRICS BY ADAPTATION")
print("=" * 70)

summary_data = []

for col in text_columns:
    if col not in df.columns:
        continue
    
    # Get the three metric columns
    hedging_col = f'{col}_hedging'
    attribution_col = f'{col}_attribution'
    art_terms_col = f'{col}_art_terms'
    
    # Calculate averages
    avg_hedging = df[hedging_col].mean()
    avg_attribution = df[attribution_col].mean()
    avg_art_terms = df[art_terms_col].mean()
    
    # Calculate overall curatorial voice density (average of all three)
    overall_voice = (avg_hedging + avg_attribution + avg_art_terms) / 3
    
    # Format column name nicely
    col_display = col.replace('_adapted', '').replace('_', ' ').title()
    if col == 'CuratorialText':
        col_display = 'Original'
    
    summary_data.append({
        'Text Type': col_display,
        'Hedging (%)': f"{avg_hedging:.4f}",
        'Attribution (%)': f"{avg_attribution:.4f}",
        'Art Terms (%)': f"{avg_art_terms:.4f}",
        'Overall Voice (%)': f"{overall_voice:.4f}"
    })

summary_df = pd.DataFrame(summary_data)

print("\n")
print(summary_df.to_string(index=False))

# Calculate % change from original
print("\n" + "=" * 70)
print("VOICE LOSS/PRESERVATION BY ADAPTATION")
print("=" * 70)

original_overall = None
for _, row in summary_df.iterrows():
    if row['Text Type'] == 'Original':
        original_overall = float(row['Overall Voice (%)'])
        break

if original_overall is not None:
    print("\nPercentage change compared to original:\n")
    for _, row in summary_df.iterrows():
        if row['Text Type'] != 'Original':
            adapted_overall = float(row['Overall Voice (%)'])
            percent_change = ((adapted_overall - original_overall) / original_overall) * 100
            print(f"  {row['Text Type']:12} {percent_change:+.2f}%  {'↓' if percent_change < 0 else '↑'}")

# Save summary to CSV as well
summary_path = f'{output_dir}/voice_metrics_summary.csv'
summary_df.to_csv(summary_path, index=False)
print(f"\n Summary saved to: {summary_path}")

print("\n" + "=" * 70)
print("VOICE METRICS COMPLETE")
print("=" * 70)
print(f" All metrics calculated")
print(f" Full dataset saved to: {output_path}")
print(f" Summary statistics saved to: {summary_path}")
print("=" * 70)

"""
Results of the voice metrics after the run:


Text Type Hedging (%) Attribution (%) Art Terms (%) Overall Voice (%)
 Original      0.7849          0.0481        4.7549            1.8626
  Lexical      0.8304          0.0161        4.0959            1.6475
 Persona1      0.9752          0.0092        3.3518            1.4454
 Persona2      0.8962          0.0467        4.5481            1.8304
 Persona3      0.9688          0.0104        3.5844            1.5212

======================================================================
VOICE LOSS/PRESERVATION BY ADAPTATION
======================================================================

Percentage change compared to original:

  Lexical      -11.55%  ↓
  Persona1     -22.40%  ↓
  Persona2     -1.73%  ↓
  Persona3     -18.33%  ↓

 Summary saved to: Data/Voice_Metrics/voice_metrics_summary.csv

======================================================================
VOICE METRICS COMPLETE
======================================================================
 All metrics calculated
 Full dataset saved to: Data/Voice_Metrics/dataset_with_voice_metrics.csv
 Summary statistics saved to: Data/Voice_Metrics/voice_metrics_summary.csv
======================================================================
(deepseek) PS C:\Users\PC\OneDrive\Desktop\Thesis> 
"""

