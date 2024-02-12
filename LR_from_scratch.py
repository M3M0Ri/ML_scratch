import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/content/data.csv")

data = data.drop(columns=['First Name', 'Last Name'])

def loss_function(m, b, points):
  total_error = 0
  for i in range(len(points)):
    x = points.iloc[i].Age
    y = points.iloc[i].Salary
    total_error += (y - (m * x+b)) ** 2
  total_error / float(len(points))

def gradinet_descent(m_now, b_now, points, L):
  m_graient = 0
  b_graient = 0

  n = len(points)
  for i in range(n):
    x = points.iloc[i].Age
    y = points.iloc[i].Salary

    m_graient += -(2/n) * x * (y- (m_now * x + b_now))
    b_graient += -(2/n) * (y- (m_now * x + b_now))

  m = m_now - m_graient * L
  b = b_now - m_graient * L
  return m, b

m = 0
b = 0
L = 0.0001
epochs = 1000
for i in range(epochs):
  m, b = gradinet_descent(m, b, data, L)

print(m, b)
plt.scatter(data.Age, data.Salary, color="black")
plt.plot(list(range(20, 80)), [m * x + b for x in range(20, 80)], color="red")
plt.show()

