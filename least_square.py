import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')

x_data = np.array([0.0, 0.3, 0.5, 0.7, 0.9])
y_data = np.array([1.25, 1.95, 2.5, 2.65, 2.75])

def func(x, C0, C1):
    return C0 + C1 * np.tan(np.pi / 2 * x)

A = np.vstack([np.tan(np.pi / 2 * x_data), np.ones(len(x_data))]).T

alpha = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y_data)
C1, C0 = alpha

print(f"Fitted parameters: C0 = {C0}, C1 = {C1}")

x_fit = np.linspace(0, 0.99, 1000)  
y_fit = func(x_fit, C0, C1)

plt.figure(figsize=(10, 8))
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_fit, y_fit, label='Fitted function', color='red')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.ylim(0, 10) 
plt.legend()
plt.show()
