import os
import pandas as pd
import matplotlib.pyplot as plt

# Folder containing the Excel files (change this to wherever they are)
data_dir = r"C:\Users\rehev\Documents\Physics\Project 2\Graph"

files = {
    "max_mass_vs_time_random.xlsx": "Run 1",
    "max_mass_vs_time-2.xlsx": "Run 2",
    "max_mass_vs_time3.xlsx": "Run 3",
    "max_mass_vs_time8.xlsx": "Run 8",
}

fig, ax = plt.subplots(figsize=(10, 6))

styles = {
    "Run 1": dict(linewidth=1.8),
    "Run 2": dict(linewidth=4, linestyle="--"),
    "Run 3": dict(linewidth=1.8),
    "Run 8": dict(linewidth=1.8),
}

for filename, label in files.items():
    df = pd.read_excel(os.path.join(data_dir, filename))
    ax.plot(df["t_yr"], df["m_max_Msun"], label=label, **styles[label])

ax.set_xlabel("Time (yr)")
ax.set_ylabel(r"Maximum mass ($M_\odot$)")
ax.set_title("Maximum Mass vs. Time")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(data_dir, "max_mass_vs_time_combined.png"), dpi=300)
plt.show()
