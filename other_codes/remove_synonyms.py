import pandas as pd
import yaml

# ========= INPUT / OUTPUT FILES =========
CSV_IN = "/Users/agrimgoyal/Desktop/DLP/data/processed_dataset.csv"      # output from clean_attributes.py
YAML_IN = "/Users/agrimgoyal/Desktop/DLP/data/attributes_pooled.yaml"         # output from clean_attributes.py

CSV_OUT = "/Users/agrimgoyal/Desktop/DLP/data/processed_dataset_normalized.csv"
YAML_OUT = "/Users/agrimgoyal/Desktop/DLP/data/attributes_normalized.yaml"


# ---- CANONICAL MAPPINGS ----
mapping = {
    "color": {
        "grey": "gray",
        "multi-color": "multicolor",
        "clear": "transparent",
        "golden": "gold"
    },
    "material": {
        "wooden": "wood",
        "cloth": "fabric",
        "cotton": "fabric",
        "canvas": "fabric",
        "nylon": "fabric",
        "velvet": "fabric",
        "suede": "fabric",
        "steel": "metal",
        "clay": "ceramic",
        "plastic and metal": "plastic"
    }
}

df = pd.read_csv(CSV_IN)
with open(YAML_IN, "r") as f:
    schema = yaml.safe_load(f)

attr_keys = list(schema.keys())

def normalize(attr, value):
    value = str(value).strip().lower()
    if value in mapping.get(attr, {}):
        return mapping[attr][value]
    return value

# normalize columns
for attr in attr_keys:
    if attr in df.columns:
        df[attr] = df[attr].apply(lambda v, a=attr: normalize(a, v))

# rebuild attributes string
df["attributes"] = df.apply(lambda row: ";".join(f"{k}:{row[k]}" for k in attr_keys), axis=1)

# rebuild YAML based on remaining values
new_schema = {k: sorted(df[k].unique()) for k in attr_keys}
for k in new_schema:
    if "unknown" not in new_schema[k]:
        new_schema[k].append("unknown")

df.to_csv(CSV_OUT, index=False)

with open(YAML_OUT, "w") as f:
    yaml.safe_dump(new_schema, f)

print("✔ Synonyms merged and attributes normalized.")
print(f"→ Saved CSV: {CSV_OUT}")
print(f"→ Saved YAML: {YAML_OUT}")