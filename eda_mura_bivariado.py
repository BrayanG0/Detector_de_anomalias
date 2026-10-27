import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="muted")

df = pd.read_csv("mura_manifest.csv")

# ------------------------------------------------------------------
# 1. Tabla de contingencia: conteo de negative/positive por bodypart
# ------------------------------------------------------------------
contingency = pd.crosstab(df["bodypart"], df["label"])
contingency_pct = pd.crosstab(df["bodypart"], df["label"], normalize="index") * 100

print("Conteo absoluto:")
print(contingency)
print("\nPorcentaje por fila (bodypart):")
print(contingency_pct.round(1))

# ------------------------------------------------------------------
# 2. Heatmap de contingencia (proporciones) - fácil de leer el %
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(contingency_pct, annot=True, fmt=".1f", cmap="coolwarm",
            vmin=0, vmax=100, cbar_kws={"label": "% dentro de la región"}, ax=ax)
ax.set_title("% de casos Normal vs. Anómalo por región anatómica")
plt.tight_layout()
plt.savefig("09_heatmap_bodypart_vs_label.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 3. Stacked bar plot - proporciones normalizadas, ordenado por %
#    de positivos para ver rápido qué región tiene más anomalías
# ------------------------------------------------------------------
order = contingency_pct.sort_values("positive", ascending=False).index
contingency_pct.loc[order].plot(
    kind="barh", stacked=True, figsize=(8, 5),
    color=["#4C72B0", "#DD8452"]
)
plt.title("Proporción Normal vs. Anómalo por región anatómica (ordenado)")
plt.xlabel("% de imágenes")
plt.ylabel("Región anatómica")
plt.legend(title="Diagnóstico", loc="lower right")
plt.tight_layout()
plt.savefig("10_stacked_bar_bodypart_label.png", dpi=150)
plt.close()

print("\nListo. Revisa 09_heatmap_bodypart_vs_label.png y 10_stacked_bar_bodypart_label.png")
