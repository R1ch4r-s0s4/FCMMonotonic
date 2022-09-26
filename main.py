import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping

x = np.arange(-5, 5, 0.1)
y = np.power(x, 3)
y1 = -1.*np.power(x,3)

x = x.reshape((len(x), 1))
y = y.reshape((len(y), 1))

scale_x = MinMaxScaler()
x = scale_x.fit_transform(x)
scale_y = MinMaxScaler()
y = scale_y.fit_transform(y)
#plt.plot(x,y)
#plt.plot(x,y1)
#plt.show()

model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu', kernel_initializer='he_uniform'))   #Hidden1
model.add(Dense(10, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1))  # Output
#
model.compile(loss='mse', optimizer = 'adam')
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5,
                         verbose=1, mode='auto', restore_best_weights=True)
model.fit(x, y, verbose=2, batch_size=10, epochs=500)
#
# # Measure RMSE error.
pred = model.predict(x)
score = np.sqrt(metrics.mean_squared_error(pred, y))
print(f"Final Score (RMSE): {score}")
#
#

plt.plot(pred)
plt.plot(y)

# plt.plot(pred)
# plt.plot(y)
#
plt.show()