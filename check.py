import duckdb
con = duckdb.connect()
r = con.execute("SELECT COUNT(*) FROM 'data/faers_flat_deduped.parquet'").fetchone()
print('Righe:', r[0])
