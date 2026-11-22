"""Unit 1: derivatives, partial derivatives and the gradient.

We compare exact derivatives with numerical ones (the change in f
for a very small change in the input), then take a few steps
downhill on a function of two variables.
"""
import numpy as np


def f(w):                       # one variable: minimum at w = 3
    return (w - 3) ** 2


def g(v):                       # two variables: minimum at (1, -2)
    x, y = v
    return (x - 1) ** 2 + 2 * (y + 2) ** 2


def grad_g(v):                  # exact partial derivatives
    x, y = v
    return np.array([2 * (x - 1), 4 * (y + 2)])


h = 1e-6
print("f'(5) exact = 4, numerical =", round((f(5 + h) - f(5)) / h, 4))

v = np.array([4.0, 1.0])
num = [(g(v + h * e) - g(v)) / h for e in np.eye(2)]
print("gradient of g at (4, 1): exact", grad_g(v),
      "numerical", np.round(num, 3))

print("\n Learing rate 0.2")
for step in range(1, 6):        # gradient descent, learning rate 0.2
    v = v - 0.2 * grad_g(v)
    print(f"step {step}: v = {np.round(v, 3)}  g(v) = {g(v):.4f}")


print("\n Learning rate 0.05")
v = np.array([4.0, 1.0])
for step in range(1, 6):        # gradient descent, learning rate 0.2
    v = v - 0.05 * grad_g(v)
    print(f"step {step}: v = {np.round(v, 3)}  g(v) = {g(v):.4f}")
    
    
print("\n Learning rate 0.4")
v = np.array([4.0, 1.0])
for step in range(1, 6):        # gradient descent, learning rate 0.2
    v = v - 0.4 * grad_g(v)
    print(f"step {step}: v = {np.round(v, 3)}  g(v) = {g(v):.4f}")
    

print("\n Learning rate 0.6")
v = np.array([4.0, 1.0])
for step in range(1, 6):        # gradient descent, learning rate 0.2
    v = v - 0.6 * grad_g(v)
    print(f"step {step}: v = {np.round(v, 3)}  g(v) = {g(v):.4f}")
    
    
# The learning rate of 0.4 converges fastest becuase after step 5, its gradient
# is close to 0 (0.1088)

# The learning rate 0.6 diverges on the Y axis, and just after step 2, the gradient 
# increases to 69.1632