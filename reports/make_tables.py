import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent

def dump_table(df, title="% Table"):
    print(title)
    print("\\begin{tabular}{lrrrrrr}")
    print("\\toprule")
    print("Method & Sil. & CH & DB & ARI & NMI & Purity \\\")
    print("\\midrule")
    for idx, row in df.iterrows():
        print(f"{idx} & {row.silhouette_score:.4f} & {row.calinski_harabasz_score:.4f} & {row.davies_bouldin_score:.4f} & {row.adjusted_rand_score:.4f} & {row.nmi:.4f} & {row.cluster_purity:.4f} \\\")
    print("\\bottomrule\n\\end{tabular}\n")

easy = root.parent / "easy" / "results" / "metrics_comparison.csv"
med = root.parent / "medium" / "results" / "clustering_comparison.csv"
hard = root.parent / "hard" / "results" / "comprehensive_comparison.csv"

if easy.exists():
    df = pd.read_csv(easy, index_col=0)
    dump_table(df, "% Easy")

if med.exists():
    df = pd.read_csv(med, index_col=0)
    dump_table(df, "% Medium")

if hard.exists():
    df = pd.read_csv(hard, index_col=0)
    dump_table(df.head(6), "% Hard (subset)")
