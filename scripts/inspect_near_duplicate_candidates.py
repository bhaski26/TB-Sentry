import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = ROOT / "results" / "data_audit" / "near_duplicate_candidates.csv"

with INPUT_CSV.open("r", newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

rows.sort(
    key=lambda row: (
        int(row["hamming_distance"]),
        row["comparison"],
        row["image_path_1"],
        row["image_path_2"],
    )
)

cross_split = [
    row for row in rows
    if row["comparison"] == "cross_split"
]

within_split = [
    row for row in rows
    if row["comparison"] == "within_split"
]

print(f"Total candidate pairs: {len(rows)}")
print(f"Cross-split pairs: {len(cross_split)}")
print(f"Within-split pairs: {len(within_split)}")

print("\n=== CLOSEST CROSS-SPLIT MATCHES ===")

for row in cross_split[:30]:
    print(
        f"Distance={row['hamming_distance']:>2} | "
        f"Label1={row['label_1']} | "
        f"Label2={row['label_2']}\n"
        f"  {row['image_path_1']}\n"
        f"  {row['image_path_2']}\n"
    )

print("\n=== CLOSEST WITHIN-SPLIT MATCHES ===")

for row in within_split[:20]:
    print(
        f"Distance={row['hamming_distance']:>2} | "
        f"Split={row['split_1']} | "
        f"Label1={row['label_1']} | "
        f"Label2={row['label_2']}\n"
        f"  {row['image_path_1']}\n"
        f"  {row['image_path_2']}\n"
    )