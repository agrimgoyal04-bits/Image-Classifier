import pandas as pd
import yaml
import os

# ========= FILE PATHS (update if needed) =========
CSV_IN = "/Users/agrimgoyal/Desktop/DLP/data/processed_dataset.csv"
YAML_IN = "/Users/agrimgoyal/Desktop/DLP/data/attributes.yaml"

CSV_OUT = "/Users/agrimgoyal/Desktop/DLP/data/processed_dataset_clean.csv"
YAML_OUT = "/Users/agrimgoyal/Desktop/DLP/data/attributes_updated.yaml"


# ========= LOAD DATA =========
df = pd.read_csv(CSV_IN)

with open(YAML_IN, "r") as f:
    schema = yaml.safe_load(f)


# Expected attribute keys (do NOT allow new keys)
attr_keys = list(schema.keys())


# ========= HELPERS =========
def parse_attr(attr_str: str):
    """Convert string like 'color:black;material:metal' to dict."""
    parsed = {}
    if isinstance(attr_str, str):
        s = attr_str.replace(",", ";")  # normalize delimiters
        for part in s.split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                parsed[k.strip()] = v.strip()
    return parsed


def normalize_value(v: str):
    """Lowercase, strip quotes, fix blanks."""
    if not isinstance(v, str):
        return "unknown"
    v = v.strip().lower().strip('"\'')
    return v if v else "unknown"


def clean_attrs(attr_str: str):
    parsed = parse_attr(attr_str)
    cleaned = {}

    for key in attr_keys:
        raw_val = parsed.get(key, "")
        val = normalize_value(raw_val)

        # If unseen but valid, add to YAML schema
        if val not in schema[key]:
            if val != "unknown":     # ignore new "unknown"
                schema[key].append(val)

        cleaned[key] = val

    return cleaned


# ========= PROCESS DATAFRAME =========
cleaned_rows = []

for i, row in df.iterrows():
    cleaned = clean_attrs(row.get("attributes", ""))
    attr_str = ";".join(f"{k}:{cleaned[k]}" for k in attr_keys)

    cleaned_rows.append((attr_str, cleaned))


# Write data back to DF
df_out = df.copy()
for (new_attr_str, cleaned_vals), idx in zip(cleaned_rows, df_out.index):
    df_out.at[idx, "attributes"] = new_attr_str
    for k in attr_keys:
        df_out.at[idx, k] = cleaned_vals[k]


# ===== KEEP ONLY REQUIRED COLUMNS =====
CORE_COLS = {
    "image_path", "class_label", "caption", "instance_id", "mapped_class", "split"
}

columns_to_keep = [c for c in list(CORE_COLS) + ["attributes"] + attr_keys if c in df_out.columns]
df_out = df_out[columns_to_keep]


# ========= SAVE OUTPUTS =========
df_out.to_csv(CSV_OUT, index=False)

with open(YAML_OUT, "w") as f:
    yaml.safe_dump(schema, f)

print("\n✔ Cleaning complete!")
print(f"→ Clean CSV saved as: {CSV_OUT}")
print(f"→ Updated YAML saved as: {YAML_OUT}")
