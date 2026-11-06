"""Plot all Fluent outlet profiles with original and diagnostic AIAC references.

Run from the Iterations folder: python plot_all_profiles_for_paper.py
The diagnostic reference was fitted to iteration 7 and is not validation data.
"""
from pathlib import Path
import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "aiac_fig24_naca0009_pdf_points.csv"
FITTED = HERE / "iteration7_local_dip_fit_diagnostic.csv"

def read_fluent(path):
    pairs = re.findall(r"^\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s*$",
                       path.read_text(encoding="utf-8"), re.M)
    a = np.asarray(pairs, dtype=float)
    return a[np.argsort(a[:, 0])]

profiles = {i: read_fluent(next(HERE.glob(f"iteration-{i}-*"))) for i in range(1, 8)}
paper = np.loadtxt(REFERENCE, delimiter=",", skiprows=1)
fitted = np.loadtxt(FITTED, delimiter=",", skiprows=1)
diagnostic = fitted[:, [0, 3]]
grid = np.linspace(2.85, 10.75, 240)

def score(profile, reference):
    a = PchipInterpolator(profile[:, 0], profile[:, 1])(grid)
    b = PchipInterpolator(reference[:, 0], reference[:, 1])(grid)
    return float(np.sqrt(np.mean((a-b)**2)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True, sharey=True)
colors = {1:"#aaaeb5", 2:"#8e939b", 3:"#777c85", 4:"#626771", 5:"#4d525c",
          6:"#009E73", 7:"#0072B2"}
for ax, ref, title, ref_label, ref_style in [
    (axes[0], paper, "A  Published reference",
     "Published NACA 0009 reference", dict(color="#111111", lw=2.7)),
    (axes[1], diagnostic, "B  Fitted curve (diagnostic)",
     "Fitted diagnostic curve", dict(color="#ad357e", lw=2.5, ls="--")),
]:
    ax.plot(ref[:, 0], ref[:, 1], label=ref_label, zorder=8, **ref_style)
    for i, arr in profiles.items():
        prominent = i in (6, 7)
        ax.plot(arr[:, 0], arr[:, 1], color=colors[i],
                lw=2.25 if prominent else 1.1, alpha=1 if prominent else .66,
                zorder=6 if prominent else 2,
                label=f"Iteration {i}")
    ax.set(title=title, xlim=(2.85,10.75), ylim=(44.5,54.5),
           xlabel="Outlet position, x (m)")
    ax.grid(color="#d6d9dc", lw=.65)
    ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Velocity magnitude (m/s)")
handles0, labels0 = axes[0].get_legend_handles_labels()
handles1, labels1 = axes[1].get_legend_handles_labels()
fig.legend([handles0[0],handles1[0]]+handles0[1:],
           [labels0[0],labels1[0]]+labels0[1:],
           loc="lower center", bbox_to_anchor=(.5,.09), ncol=5,
           frameon=False, fontsize=8.5)
fig.text(.5,.015,"Panel B was fitted to iteration 7 and is not an independent validation reference.",
         ha="center",fontsize=9,color="#555555")
fig.tight_layout(rect=(0,.19,1,1))
png = HERE / "outlet_velocity_all_iterations_comparison.png"
pdf = HERE / "outlet_velocity_all_iterations_comparison.pdf"
fig.savefig(png, dpi=300)
fig.savefig(pdf)
plt.close(fig)

with (HERE / "outlet_velocity_profile_rmse.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["iteration", "rmse_published_reference_m_per_s",
                     "rmse_iteration7_fitted_diagnostic_m_per_s"])
    for i, arr in profiles.items():
        writer.writerow([i, f"{score(arr,paper):.6f}", f"{score(arr,diagnostic):.6f}"])
print(png)
print(pdf)
