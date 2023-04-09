# Standart Kütüphaneleri İçe Aktar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# sklearn'dan Veri Kümesini İçe Aktar
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Makine Öğrenmesi Kütüphanelerini İçe Aktar
from sklearn.model_selection import train_test_split

# Iris Verisini Yükle
iris = load_iris()

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
target_df = pd.DataFrame(data=iris.target, columns=['species'])


# Tür Dönüştürücü Fonksiyonu
def donusturucu(tur):
    if tur == 0:
        return 'setosa'
    elif tur == 1:
        return 'versicolor'
    else:
        return 'virginica'


target_df['species'] = target_df['species'].apply(donusturucu)

iris_df = pd.concat([iris_df, target_df], axis=1)

# Grafiklerle Veri Görselleştirme
plt.style.use('ggplot')
sns.pairplot(iris_df, hue='species')

sns_plot = sns.pairplot(iris_df, hue='species')
sns_plot.savefig("snspairplot_iris.png")

# "Sepal Length (cm)" Özelliğini Tahmin Etmek İçin Model Oluşturma
iris_df.drop('species', axis=1, inplace=True)

target_df = pd.DataFrame(columns=['species'], data=iris.target)

iris_df = pd.concat([iris_df, target_df], axis=1)

X = iris_df.drop(labels='sepal length (cm)', axis=1)
y = iris_df['sepal length (cm)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr.predict(X_test)

pred = lr.predict(X_test)

# Evaluating Model's Eğitilen modelin performans parametreleri
print('Ortalama Mutlak Hata:', mean_absolute_error(y_test, pred))
print('Ortalama Kare Hata:', mean_squared_error(y_test, pred))
print('Ortalama Kök Kare Hata:', np.sqrt(mean_squared_error(y_test, pred)))

dummy_test_data = {'sepal length (cm)': [4.9],
                   'sepal width (cm)': [3.1],
                   'petal length (cm)': [2.1],
                   'petal width (cm)': [0.6],
                   'species': 1}

test_df = pd.DataFrame(data=dummy_test_data)

X_test = test_df.drop('sepal length (cm)', axis=1)
y_test = test_df['sepal length (cm)']

lr.predict(X_test)

pred = lr.predict(X_test)

print('Tahmin Edilen Çanak Yaprak Uzunluğu (cm):', pred[0])
print('Gerçek Çanak Yaprak Uzunluğu (cm):', 4.9)
