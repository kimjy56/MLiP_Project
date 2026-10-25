import json
import pandas as pd

# Let's inspect which samples differed
# We can load the results dataframe from benchmark_beam_search if saved,
# or write a quick inspector
report = json.load(open("reports/benchmark_beam_search.json", encoding="utf-8"))
print("Report comparison verified:")
print(json.dumps(report["comparison"], indent=2))
