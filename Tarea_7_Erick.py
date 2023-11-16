import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve
#Parametros que vienen en el libro: 
C= 1
I= 0
EL= -80
gL=8
gNa= 20
gK= 10
m_v=-20
km= 15
n_v=-25
kn= 5
T= 1
ENa=60
EK=-90

# Funciones de activación e inactivación de los canales de sodio y potasio
def m_inf(V):
    return 1 / (1 + np.exp(-(V - m_v) / km))
def n_inf(V):
    return 1 / (1 + np.exp(-(V - n_v) / kn))

# Ecuaciones diferenciales del sistema dinámico de sodio y potasio
def dX_dt(X, t):
    V, n = X
    dV_dt = (I - gL * (V - EL) - gNa * m_inf(V) * (V - ENa) - gK * n * (V - EK)) / C
    dn_dt = (n_inf(V) - n) / T
    return [dV_dt, dn_dt]

# Función para calcular los puntos fijos del sistema
def find_fixed_points(r):
    # Crear una malla para V y n
    V, n = np.meshgrid(r, r)
 
     # Calcular las derivadas en cada punto de la malla
    dV, dn = dX_dt([V, n], 0)
 
    # Encontrar los puntos donde ambas derivadas son cero (puntos fijos)
    mask = (np.abs(dV) < 1e-5) & (np.abs(dn) < 1e-5)
 
    # Extraer las coordenadas de los puntos fijos
    fixed_points = np.array(list(zip(V[mask], n[mask])))
 
    return fixed_points
    
V_range = np.linspace(-100, 50, 20)
n_range = np.linspace(0, 1, 20)
# Crear una malla para V y n
V, n = np.meshgrid(V_range, n_range)
# Calcular las derivadas en cada punto de la malla
dV, dn = dX_dt([V, n], 0)
# Normalizar las flechas del campo vectorial
M = (np.hypot(dV, dn))
M[M == 0] = 1.
dV /= M
dn /= M
# Calcular los puntos fijos del sistema y guardarlos en una variable
fixed_points = find_fixed_points(V_range)
# Graficar el campo vectorial, las nulclinas y los puntos fijos por separado
plt.figure()
plt.quiver(V, n, dV, dn, M)
if fixed_points.size > 0:
    plt.scatter(fixed_points[:,0], fixed_points[:,1], c='b', marker='x', label='Puntos')
plt.xlabel('V')
plt.ylabel('n')
plt.title('Campo vectorial')


# Valores de V para evaluar
V_vals = np.linspace(-100, 50, 100)

# Calcular las nulclinas
V_nullcline = (I - gL * (V_vals - EL) - gNa * m_inf(V_vals) * (V_vals - ENa)) / (gK * (V_vals - EK))
n_nullcline = n_inf(V_vals)

# Encontrar todas las intersecciones de las nulclinas
def intersection_function(x):
    return (I - gL * (x - EL) - gNa * m_inf(x) * (x - ENa)) / (gK * (x - EK)) - n_inf(x)

# Inicializar una lista para almacenar las intersecciones
intersections = []

# Buscar las intersecciones en el rango de valores de V
for V in V_vals:
    roots = fsolve(intersection_function, V)
    for root in roots:
        if np.isclose(intersection_function(root), 0):
            intersection = (root, n_inf(root))
            if not any(np.isclose(intersection, existing_intersection).all() for existing_intersection in intersections):
                intersections.append(intersection)

# Convertir la lista de intersecciones a un arreglo de NumPy
intersections = np.array(intersections)

# Graficar las nulclinas y las intersecciones
plt.figure(figsize=(8, 6))
plt.plot(V_vals, V_nullcline, 'g', label='V-nullcline')
plt.plot(V_vals, n_nullcline, 'b', label='n-nullcline')
plt.scatter(intersections[:, 0], intersections[:, 1], c='k', marker='o', label='Intersecciones')
plt.xlabel('V')
plt.ylabel('n')
plt.title('Nulclinas')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar los valores numéricos de las intersecciones
for i, intersection in enumerate(intersections):
    print(f'Intersección {i+1}: V = {intersection[0]:.2f}, n = {intersection[1]:.2f}')

# Parámetros cambiados: 
EL = -78 # Cambio
n_v = -45 #Cambio
# Funciones de activación e inactivación de los canales de sodio y potasio

# Ecuaciones diferenciales del sistema dinámico de sodio y potasio
def dX_dt(X, t):
    V, n = X
    dV_dt = (I - gL * (V - EL) - gNa * m_inf(V) * (V - ENa) - gK * n * (V - EK)) / C
    dn_dt = (n_inf(V) - n) / T
    return [dV_dt, dn_dt]

# Función para calcular los puntos fijos del sistema
def find_fixed_points(r):
    # Crear una malla para V y n
    V, n = np.meshgrid(r, r)
    
    # Calcular las derivadas en cada punto de la malla
    dV, dn = dX_dt([V, n], 0)
    
    # Encontrar los puntos donde ambas derivadas son cero (puntos fijos)
    mask = (np.abs(dV) < 1e-5) & (np.abs(dn) < 1e-5)
    
    # Extraer las coordenadas de los puntos fijos
    fixed_points = np.array(list(zip(V[mask], n[mask])))
    
    return fixed_points

# Rango de valores para V y n
V_range = np.linspace(-100, 50, 20)
n_range = np.linspace(0, 1, 20)

# Crear una malla para V y n
V, n = np.meshgrid(V_range, n_range)

# Calcular las derivadas en cada punto de la malla
dV, dn = dX_dt([V, n], 0)

# Normalizar las flechas del campo vectorial
M = (np.hypot(dV, dn))
M[M == 0] = 1.
dV /= M
dn /= M

# Calcular los puntos fijos del sistema y guardarlos en una variable
fixed_points = find_fixed_points(V_range)

# Graficar el campo vectorial, las nulclinas y los puntos fijos por separado

plt.figure()
plt.quiver(V, n, dV, dn, M)
if fixed_points.size > 0:
    plt.scatter(fixed_points[:,0], fixed_points[:,1], c='b', marker='x', label='Puntos fijos')
plt.xlabel('V')
plt.ylabel('n')
plt.title('Campo vectorial')


# Valores de V para evaluar
V_vals = np.linspace(-100, 50, 100)

# Calcular las nulclinas
V_nullcline = (I - gL * (V_vals - EL) - gNa * m_inf(V_vals) * (V_vals - ENa)) / (gK * (V_vals - EK))
n_nullcline = n_inf(V_vals)

# Encontrar las intersecciones de las nulclinas
def intersection_function(x):
    return (I - gL * (x - EL) - gNa * m_inf(x) * (x - ENa)) / (gK * (x - EK)) - n_inf(x)

intersections = fsolve(intersection_function, [-80])

# Graficar las nulclinas y las intersecciones
plt.figure(figsize=(8, 6))
plt.plot(V_vals, V_nullcline, 'g', label='V-nullcline')
plt.plot(V_vals, n_nullcline, 'b', label='n-nullcline')
plt.plot(intersections, n_inf(intersections), 'ko', label='Intersecciones')
plt.xlabel('V')
plt.ylabel('n')
plt.title('Nulclinas')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar los valores numéricos de las intersecciones
for i, intersection in enumerate(intersections):
    print(f'Intersección {i+1}: V = {intersection:.2f}, n = {n_inf(intersection):.2f}')