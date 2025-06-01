import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

df = pd.read_csv("tuning_results.csv")
heatmap_data = df.pivot(index='MC', columns='KC', values='GFLOPS')

plt.figure(figsize=(8, 6))
plt.imshow(heatmap_data, aspect='auto', cmap='viridis')
plt.colorbar(label='GFLOPS')

plt.xticks(ticks=range(len(heatmap_data.columns)), labels=heatmap_data.columns)
plt.yticks(ticks=range(len(heatmap_data.index)), labels=heatmap_data.index)
plt.xlabel('Tile size along x')
plt.ylabel('Tile size along y')
plt.title('GFLOPS')

plt.tight_layout()
plt.savefig("tile_size_gflops.png")
