import numpy as np
import pandas as pd

df = pd.read_csv("iris.csv", header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "label"],
                 encoding="utf-8")

# select Iris-setosa and Iris-versicolor
df = df[(df.label == "Iris-setosa") | (df.label == "Iris-versicolor")]
y = df.label.values
y = np.where(y == "Iris-setosa", 1, 0)
# extract sepal length and petal length
X = df[["sepal_length", "petal_length"]].values