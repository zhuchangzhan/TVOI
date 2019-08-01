import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

"""
file = open("../../input/Sector8/interesting/101395750_raw.txt","r").read()

x, y = [],[]
for i in file.split("\n"):
    try:
        a,b = i.split()
        x.append(float(a))
        y.append(float(b))

    except:
        pass

plt.plot(x,y,".")
plt.show()


df = pd.read_excel("../../input/Machine_Learning_Labels/SPOC_Sector9.xlsx")

trim_df = df[df["Comments"].notnull()]

problematic = trim_df["TIC"].values



Interest = ['1008153', '11932649', '11974455', '18658256', '19438266', '19826549', '20319492', '20376554', '20414865', '20448010', '21841099', '22123033', '4847760', '5165416', '5229041', '5574945', '11970587', '11974826', '14343100', '19271382', '20500037', '5092088', '5452928', '5471443', '5528993']
"""
Interest = ["141494666","356069146"]

Interest = ["272551828"]

center = ",".join(["*%s*"%x for x in Interest])

text = "scp zzhan@pdo:/pdo/spoc-data/sector-10/light-curve/\{%s\} ."%(center)

print(text) 