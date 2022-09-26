from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from numpy import  asarray


# xs = np.linspace(-200,200,5000).reshape(-1,1)
# ys = xs**3
xs = np.arange(-5, 5, 0.001)
ys = asarray([-i**3.0 for i in xs])
print(xs.min(), xs.max(), ys.min(), ys.max())

noise = np.random.normal(0,20)
print(noise)

xs = xs.reshape((len(xs), 1))
ys = ys.reshape((len(ys), 1))


# xs = MinMaxScaler().fit_transform(xs)
# ys = MinMaxScaler().fit_transform(ys)
scale_x = MinMaxScaler()
xs = scale_x.fit_transform(xs)
scale_y = MinMaxScaler()
ys = scale_y.fit_transform(ys)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=128,input_shape=(1,), activation='relu'))
model.add(tf.keras.layers.Dense(25, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
#model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1,))
model.compile(optimizer='adam',loss = 'mae')

model.fit(xs,ys,epochs=300,verbose=1, batch_size=32)
y_pred = model.predict(([[2]]))
print((y_pred))

y_pred = model.predict(xs, batch_size=16)
x_plot = scale_x.inverse_transform(xs)
y_plot = scale_y.inverse_transform(ys)
yhat_plot = scale_y.inverse_transform(y_pred)
plt.plot(x_plot.reshape(-1), yhat_plot, 'r')
plt.plot(x_plot.reshape(-1), y_plot, 'b:')
plt.show()