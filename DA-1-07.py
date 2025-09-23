import pandas as pd
from sklearn.datasets import load_iris

def analyze_iris_data():
    """
    Анализирует датасет Iris, создавая категориальный столбец вида, группируя по видам
    и вычисляя агрегации (среднее, сумма, количество, минимум, максимум) для всех признаков.
    
    Сохраняет результат в CSV-файл.
    Возвращает pandas.DataFrame с результатами агрегации по видам.
    """
    try:
        # Загрузка датасета Iris
        iris = load_iris()
    except Exception as e:
        raise Exception(f"Не удалось загрузить датасет Iris: {str(e)}")
    
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    
    # Создание категориального признака
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = pd.Categorical(
        [species_map[target] for target in iris.target],
        categories=['setosa', 'versicolor', 'virginica'],
        ordered=False
    )
    
    # Группировка по видам и вычисление агрегаций
    result = df.groupby('species', observed=False).agg({
        'sepal length (cm)': ['mean', 'sum', 'count', 'min', 'max'],
        'sepal width (cm)': ['mean', 'sum', 'count', 'min', 'max'],
        'petal length (cm)': ['mean', 'sum', 'count', 'min', 'max'],
        'petal width (cm)': ['mean', 'sum', 'count', 'min', 'max']
    })
    
    # Сохранение результата в CSV
    result.to_csv('iris_analysis_result.csv')
    
    return result

if __name__ == "__main__":
    try:
        result = analyze_iris_data()
        print(result)
    except Exception as e:
        print(f"Ошибка: {str(e)}")
