import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
c= 10 #capacitancia 
g = 19 #conductancia
gNa = 74 #conductancia de Na
E = -67#potencial de equilibrio de membrana
ENa = 60 #potencial de equilibrio de Na
V12 = 1.5 #voltaje medio de apertura del canal de Na
k = 16 #Delta voltaje para apertura del canal de Na
V = np.linspace(-60, 60, 100)
n = 1 / (1 + np.exp((V12 - V) / k))
T=100
h= 0.001
def FV(I):
    FV = I - g * (V - E) - gNa * (V - ENa) * n
    return FV

I0=0
plt.plot(V, FV(I0), label='F(V)')
plt.plot(V, np.zeros(V.shape), label='Zero Line',)
plt.xlabel('V')
plt.ylabel('F(V)')
plt.title('Gráfica de la función F(V)')
plt.legend()
plt.show()

def umbral_F(umbral, reset):
    for V0 in range(-70, 71, 5):
        V_sol0 = np.zeros(T)
        V_sol0[0] = V0

        for cn in range(T - 1):
            n = 1 / (1 + np.exp((V12 - V_sol0[cn]) / k))
            FVcn = I0 - g * (V_sol0[cn] - E) - gNa * (V_sol0[cn] - ENa) * n
            V_sol0[cn + 1] = V_sol0[cn] + h * FVcn
            if V_sol0[cn + 1] > umbral:
                V_sol0[cn + 1] = reset
# 
        plt.plot(range(1, T + 1), V_sol0, "k")
    plt.xlabel("T")
    plt.ylabel("V_sol")
    plt.show()


#Llamamos a la funcion
U0= 30
R0= -30
umbral_F(U0, R0)


#Ahora haremos el modelos cuadratico 
def G(I):
 G = I + V**2
 return G
# Graficar FV y línea en 0
plt.plot(V, G(I0), 'r')
plt.axhline(y=0, color='b', linestyle='--') 
plt.xlabel("V")
plt.ylabel("F(V0)")
plt.title("Gráfico de F(V0)")
plt.grid(True)
plt.show()


def graficar_voltaje(I, V_umbral, V_reset):
    for V0 in range(0, 70, 10):
        V_sol = np.zeros(T)
        V_sol[0] = V0
        for cn in range(T - 1):
            FVcn = I + V_sol[cn]**2
            # V_sol4[cn + 1] = FVcn # Actualizar el valor de V_sol22 en cada itera
            V_sol[cn + 1] = V_sol[cn] + h * FVcn
            if V_sol[cn + 1] > V_umbral:
                V_sol[cn + 1] = V_reset
        plt.plot(range(1, T + 1), V_sol, "b")
    plt.xlabel("T")
    plt.ylabel("V_sol")
    plt.show()

graficar_voltaje(I0,U0,R0)


#Cambio de umbral
graficar_voltaje(I0,40,30)

I16=16
plt.plot(V, FV(I16), label='F(V)')
plt.plot(V, np.zeros(V.shape), label='Zero Line')
plt.xlabel('V')
plt.ylabel('F(V)')
plt.title('Gráfica de la función F(V)')
plt.legend()
plt.show()


U16= 30
R16= -50
umbral_F(U16, R16)

# Graficar FV y línea en 0
plt.plot(V, G(I16), 'b')
plt.axhline(y=0, color='g', linestyle='-')
plt.xlim(-20,20)
plt.ylim(0,100)
plt.xlabel("V")
plt.ylabel("F(V0)")
plt.title("Gráfico de F(V0)")
plt.grid(True)
plt.show()


#Usamos la funcion: 
graficar_voltaje(I16,U16,R16)


#Cambiamos el umbral
graficar_voltaje(I16,50,40)

I50=50
plt.plot(V, FV(I50), label='F(V)')
plt.plot(V, np.zeros(V.shape), label='Zero Line')
plt.xlabel('V')
plt.ylabel('F(V)')
plt.title('Gráfica de la función F(V)')
plt.legend()
plt.show()

#Llamamos a la funcion
U50= 30
R50= -70
umbral_F(U50, R50)


#Modelo cuadratico: 
# Graficar FV y línea en 0
plt.plot(V, G(I50), 'b')
plt.axhline(y=0, color='g', linestyle='-')
plt.xlabel("V")
plt.ylabel("F(V0)")
plt.title("Gráfico de F(V0)")
plt.grid(True)
plt.show()


#Usamos la funcion: 
graficar_voltaje(I50,U50,R50)

#Cambiamos el umbral
graficar_voltaje(I50,50,40)


I10=10
plt.plot(V, FV(I10), label='F(V)')
plt.plot(V, np.zeros(V.shape), label='Zero Line')
plt.xlabel('V')
plt.ylabel('F(V)')
plt.title('Gráfica de la función F(V)')
plt.legend()
plt.show()


#Llamamos a la funcion
U10= 20
R10= -50
umbral_F(U10, R10)

# Graficar FV y línea en 0
plt.plot(V, G(I10), 'b')
plt.axhline(y=0, color='g', linestyle='-')
plt.xlim(-20,20)
plt.ylim(0,100)
plt.xlabel("V")
plt.ylabel("F(V0)")
plt.title("Gráfico de F(V0)")
plt.grid(True)
plt.show()



#Usamos la funcion: 
graficar_voltaje(I10,U10,R10)

#Cambiamos el umbral
graficar_voltaje(I10,50,40)


In=-100
plt.plot(V, FV(In), label='F(V)')
plt.plot(V, np.zeros(V.shape), label='Zero Line')
plt.xlabel('V')
plt.ylabel('F(V)')
plt.title('Gráfica de la función F(V)')
plt.legend()
plt.show()


#Llamamos a la funcion
Un= 30
Rn= -10
umbral_F(Un, Rn)


#Modelo cuadratico: 
# Graficar FV y línea en 0
plt.plot(V, G(In), 'b')
plt.axhline(y=0, color='g', linestyle='-')
plt.xlabel("V")
plt.ylabel("F(V0)")
plt.title("Gráfico de F(V0)")
plt.grid(True)
plt.show()


graficar_voltaje(In,Un,Rn)

#Cambiamos el umbral
graficar_voltaje(In,60,50)