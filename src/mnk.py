import pandas as pd
df = pd.read_csv("./../data/data_learning", header="None", engine="python" )
u = df.iloc[0:end, 0].values
y = df.iloc[0:end, 1].values
