import numpy as np
from random import seed, random
import matplotlib.pyplot as plt
# Parametros:
I = 0
a = 0.03
b = -2
# Espacios nulos funcion:
def v_null1(a, b, I):
   
    x1 = (b + (b ** 2 - 4 * a * I) ** 0.5) / (2 )
    x2 = (b - (b ** 2 - 4 * a * I) ** 0.5) / (2 )

    # Return the calculated values
    return x1, x2

# Call the function and store the results in variables
v1, v2 = v_null1(a=a, b=b, I=I)

# Print the calculated values
print("v1 =", v1)
print("v2 =", v2)

def u_null1(b, v):
    return b * v

u1 = u_null1(b=b, v=v1)
print('valor de u1 =', abs(u_null1(b=b, v=v1)))
u2 = u_null1(b=b, v=v2)
print('valor de u2 =', abs(u2))
# Calculamos Tao y Delta 
Tao1 = -a
Delta1 = a * b
print('Tao1', Tao1)
print('Delta1', Delta1)
Tao2 = (2 * b) - a
Delta2 = a * b + 1
print('Tao2', Tao2)
print('Delta2', Delta2)
# Determinamos los valores de lambda
def valores_porp(D, Tao, I):
    # Perform calculations using the general formula
    lmd1 = (I + (Tao ** 2 - 4 * D) ** 0.5) / 2
    lmd2 = (I - (Tao ** 2 - 4 * D) ** 0.5) / 2

    # Return the calculated values
    return lmd1, lmd2

lmd1, lmd2 = valores_porp(D=Delta1, Tao=Tao1, I=I)

# Print the calculated values
print("lmd1 =", lmd1)
print("lmd2 =", lmd2)
# Determinamos los valores de lambda
def valores_porp(D, Tao, I):
    # Perform calculations using the general formula
    lmd1 = (I + (Tao ** 2 - 4 * D) ** 0.5) / 2
    lmd2 = (I - (Tao ** 2 - 4 * D) ** 0.5) / 2

    # Return the calculated values
    return lmd1, lmd2

lmd1, lmd2 = valores_porp(D=Delta1, Tao=Tao1, I=I)

# Print the calculated values
print("lmd1 =", lmd1)
print("lmd2 =", lmd2)
# Function to calculate f(v)
def calculate_f(v):
    return v**2 - b*v + I

# Generate values for v
v = np.linspace(-12, 10, 100)

# Calculate corresponding values for f(v)
f = calculate_f(v)

# Plot the parabola and the identity line
plt.figure(figsize=(8, 6))
plt.plot(v, f, label='f(v) = v^2 - bv + I', color='blue')
plt.plot(v, v, label='Identity line', color='red', linestyle='--')
plt.ylim(-10, 10)
plt.xlim(-2.5, 5)
plt.xlabel('v')
plt.ylabel('f(v)')
plt.legend()
plt.title('Parabola with Identity Line: f(v) = v^2 - bv + I')
plt.grid(True)

# Calculate the solutions using the quadratic formula
a1 = (-(-b) + np.sqrt(b**2 - 4*I)) / 2
a2 = (-(-b) - np.sqrt(b**2 - 4*I)) / 2

# Store the solutions in a variable
solutions = (a1, a2)

# Print the solutions
print("Solutions:", solutions)

# Define the dynamic system
def system(v, f):
    dv_dt = 2 * v - f
    df_dt = v + f
    return dv_dt, df_dt

# Generate grid of points
v = np.linspace(-12, 10, 20)
f = np.linspace(-10, 10, 20)
V, F = np.meshgrid(v, f)

# Calculate derivatives for each point in the grid
dV_dt, dF_dt = system(V, F)

# Plot the phase portrait
plt.quiver(V, F, dV_dt, dF_dt)

plt.show()

# Parametros:
I = 0
a = 0.03
b = 2

# Espacios nulos funcion:
def v_null1(a, b, I):
    # Perform calculations using the general formula
    x1 = (b + (b ** 2 - 4 * a * I) ** 0.5) / (2 )
    x2 = (b - (b ** 2 - 4 * a * I) ** 0.5) / (2 )

    # Return the calculated values
    return x1, x2

# Call the function and store the results in variables
v1, v2 = v_null1(a=a, b=b, I=I)

# Print the calculated values
print("v1 =", v1)
print("v2 =", v2)

def u_null1(b, v):
    return b * v

u1 = u_null1(b=b, v=v1)
print('valor de u1 =', abs(u_null1(b=b, v=v1)))
u2 = u_null1(b=b, v=v2)
print('valor de u2 =', abs(u2))

# Calculamos Tao y Delta para ambos casos
Tao1 = -a
Delta1 = a * b
print('Tao1', Tao1)
print('Delta1', Delta1)

Tao2 = (2 * b) - a
Delta2 = a * b + 1
print('Tao2', Tao2)
print('Delta2', Delta2)

# Determinamos los valores de lambda
def valores_porp(D, Tao, I):
    # Perform calculations using the general formula
    lmd1 = (I + (Tao ** 2 - 4 * D) ** 0.5) / 2
    lmd2 = (I - (Tao ** 2 - 4 * D) ** 0.5) / 2

    # Return the calculated values
    return lmd1, lmd2

lmd1, lmd2 = valores_porp(D=Delta1, Tao=Tao1, I=I)

# Print the calculated values
print("lmd1 =", lmd1)
print("lmd2 =", lmd2)

# Function to calculate f(v)
def calculate_f(v):
    return v**2 - b*v + I

# Generate values for v
v = np.linspace(-10, 10, 100)

# Calculate corresponding values for f(v)
f = calculate_f(v)

# Plot the parabola and the identity line
plt.figure(figsize=(8, 6))
plt.plot(v, f, label='f(v) = v^2 - bv + I', color='blue')
plt.plot(v, v, label='Identity line', color='red', linestyle='--')
plt.ylim(-10,10)
plt.xlim(-10, 10)
plt.xlabel('v')
plt.ylabel('f(v)')
plt.legend()
plt.title('Parabola with Identity Line: f(v) = v^2 - bv + I')
plt.grid(True)

# Calculate the solutions using the quadratic formula
a1 = (-(-b) + np.sqrt(b**2 - 4*I)) / 2
a2 = (-(-b) - np.sqrt(b**2 - 4*I)) / 2

# Store the solutions in a variable
solutions = (a1, a2)

# Print the solutions
print("Solutions:", solutions)

# Define the dynamic system
def system(v, f):
    dv_dt = 2 * v - f
    df_dt = v + f
    return dv_dt, df_dt

# Generate grid of points
v = np.linspace(-12, 10)
f = np.linspace(-10, 10)
V, F = np.meshgrid(v, f)

# Calculate derivatives for each point in the grid
dV_dt, dF_dt = system(V, F)

# Plot the phase portrait
plt.quiver(V, F, dV_dt, dF_dt)

plt.show()