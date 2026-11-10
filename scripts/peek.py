"""Tiny helper for local inspection: read a local lakehouse table into pandas."""
import glob, os, re, sys
import pandas as pd, pyarrow.parquet as pq

def rd(name, root="data"):
    out = []
    for f in glob.glob(os.path.join(root, "lakehouse", name, "**", "*.parquet"), recursive=True):
        df = pq.read_table(f).to_pandas()
        for k, v in re.findall(r"([a-z_]+)=([^/]+)(?=/)", f.replace(os.sep, "/")):
            df[k] = v
        out.append(df)
    return pd.concat(out, ignore_index=True)

if __name__ == "__main__":
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 60)
    print(rd(sys.argv[1]).head(int(sys.argv[2]) if len(sys.argv) > 2 else 10))
