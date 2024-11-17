import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Входные и выходные данные или как описывается в работе (экспереминтальные данные)
x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], dtype=np.float32)
y = np.array([2.86, 2.21, 2.96, 3.27, 3.58, 3.76, 3.93, 3.67, 3.90, 3.64, 4.09], dtype=np.float32)

# Архитектура модели
model = Sequential()
model.add(Dense(10, input_dim=1, activation='tanh'))   #Скрытый слой из 10 нейронов с функцией активации tanh
model.add(Dense(1, activation='linear'))               #Выходной слой с линейной функцией активации

# Компиляция модели
model.compile(optimizer=tf.keras.optimizers.Adam(),    # Используем оптимизатор Adam, 
              loss='mean_squared_error')               # Функция потерь MSE (МНК)

# Обучение модели
history = model.fit(x, y, epochs=2000, verbose=1)       # Обучаем на 1000 эпох

# Предсказание
predictions = model.predict(x)

# Отображение, графика
plt.plot(x, y, 'r+', label='Исходные данные')           
plt.plot(x, predictions, 'g-', label='Предсказания')    

# Дополнительные точки для предсказания
xx = np.array([0.185, 0.86], dtype=np.float32)
vv = model.predict(xx)

plt.plot(xx, vv, 'ob', markersize=5, label='Предсказания для новых данных')   # Синие круги - новые данных
plt.legend()
plt.show()
