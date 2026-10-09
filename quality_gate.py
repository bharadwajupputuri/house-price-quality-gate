
import json
import sys

# Minimum acceptable R2 score
MINIMUM_R2 = 0.70

print("HOUSE PRICE ML QUALITY GATE")
print("---------------------------")

try:
    with open("metrics.json", "r") as file:
        metrics = json.load(file)

    r2_score = metrics["r2_score"]

except (FileNotFoundError, KeyError, json.JSONDecodeError) as error:
    print("ERROR: Could not read model metrics.")
    print(error)
    sys.exit(1)

print("Model R2 Score:", round(r2_score, 4))
print("Minimum Required R2 Score:", MINIMUM_R2)

if r2_score >= MINIMUM_R2:
    print("\nQUALITY GATE PASSED")
    print("The model meets the minimum performance requirement.")
    sys.exit(0)
else:
    print("\nQUALITY GATE FAILED")
    print("The model does not meet the minimum performance requirement.")
    sys.exit(1)
