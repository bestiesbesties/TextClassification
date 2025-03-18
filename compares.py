import os
import pandas as pd

v1 = pd.read_csv(os.path.join("eval","preloads","at_2025-03-01_17-25_data.csv"))
v2 = pd.read_csv(os.path.join("eval","preloads","at_2025-03-12_13-05_data.csv"))

v2["Base"] = v1["Correct"]

v2 = v2[["Resume_str", "Label", "Prediction", "Base", "Correct"]]
df = v2



result = df[((df["Base"] == True) & (df["Correct"] == False))]
len(result)
result.to_csv(os.path.join("eval", "preloads", "True_False.csv"))



result = df[((df["Base"] == False) & (df["Correct"] == True))]
len(result)
result.to_csv(os.path.join("eval", "preloads", "False_True.csv"))



result["Prediction"].value_counts()


import json
import os

preloads_v1 = json.load(open(os.path.join("eval","preloads","at_2025-03-01_17-25_preloads.json"),"r"))
a = preloads_v1["preloads"]["data"]["healthcare"]["keywords"]


preloads_v2 = json.load(open(os.path.join("eval","preloads","at_2025-03-12_13-05_preloads.json"),"r"))
b = preloads_v2["preloads"]["data"]["healthcare"]["keywords"]


x = set(a) - set(b)
y = set(b) - set(a)

len(x) + len(y)

p = set(list(x) + list(y))
len(p)