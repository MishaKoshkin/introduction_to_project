import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

#1 Создание категориального признака из target
df['species'] = pd.Categorical(iris.target, categories=[0, 1, 2], ordered=False)

#2 Группировка по категориальному признаку
#3 Вычисление среднего, суммы и количества
result = df.groupby('species', observed=False).agg({
    'sepal length (cm)': ['mean', 'sum', 'count'],
    'sepal width (cm)': ['mean', 'sum', 'count'],
    'petal length (cm)': ['mean', 'sum', 'count'],
    'petal width (cm)': ['mean', 'sum', 'count']
})

#4 Вывод результата
print(result)

