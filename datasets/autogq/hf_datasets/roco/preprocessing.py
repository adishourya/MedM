import polars as pl
df = pl.read_parquet("./train_roco_vqa.parquet")
print(df.shape, df.columns)
