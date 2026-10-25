import json
import pandas as pd
from transformers import AutoTokenizer
from name_decoder.data import read_frames, select_pilot
from name_decoder.catalog import canonical_name

# Let's inspect what happened on specific samples with Trie
# Let's run a few samples that succeeded in unconstrained but failed in Trie
print("Checking mechanical reason for Trie constraint degradation...")
