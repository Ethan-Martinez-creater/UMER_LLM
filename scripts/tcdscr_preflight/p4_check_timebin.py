import torch
from pathlib import Path

REF = Path("/data/jyz/next/llm/data/pheme_240h_data/graph_final")
for eid in ["498317138554150912", "498272309535191041", "498284429333524480"]:
    g = torch.load(REF / f"{eid}.pt", map_location="cpu", weights_only=True)
    tb = g["time_bin"]
    print(eid,
          "max_time_steps=", int(g["max_time_steps"]),
          "time_bin_max=", int(tb.max()),
          "time_bin_min=", int(tb.min()),
          "n=", int(g["num_nodes"]))
