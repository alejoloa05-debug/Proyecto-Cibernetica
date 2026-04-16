# Simulación Cibernética: Discusión de Pareja
Este proyecto es un modelo de sistemas dinámicos que utiliza Python para simular la interacción entre dos agentes en una discusión. El objetivo es analizar cómo el estrés, la reactividad y la calma influyen en la estabilidad de un sistema social.

# Características
Visualización en Tiempo Real: Gráficas animadas con matplotlib.
Análisis de Espacio de Fases: Observa la relación de tensión entre los agentes.
Escenarios Predefinidos: 6 configuraciones que demuestran desde la homeostasis hasta la escalada total.
Factores Externos: Simulación de estrés ambiental y presión social.

# Conceptos Cibernéticos Aplicados
Sistema Abierto: El modelo intercambia información con el entorno (clase FactoresExternos).
Retroalimentación (Feedback): * Positiva: Escalada del conflicto basada en la reactividad.
Negativa: Procesos homeostáticos para recuperar el equilibrio (calma).
Umbral de Estabilidad: El punto de bifurcación donde el sistema cambia de modo racional a reactivo.
Entropía: Introducción de "ruido" comunicativo aleatorio.

# Uso
Asegúrate de tener instalado numpy y matplotlib.
Bash
#Ejecutar con parámetros por defecto
python simulacion_animada.py

#Ejecutar un escenario específico (1-6)
python simulacion_animada.py --escenario 2

#Ver ayuda y todos los parámetros
python simulacion_animada.py --ayuda

# Resultados del Sistema
El sistema puede converger en tres estados finales (atractores):
Resolución: El sistema disipa la energía y vuelve al equilibrio.
Escalada: El sistema colapsa por exceso de retroalimentación positiva.
Ciclo: El sistema queda atrapado en un bucle infinito de repetición.

Universidad Sergio Arboleda.
Participantes: Alejandro Loaiza, Sofia Salinas, Jhosep Rodriguez, Angely Burgos, Esteban Castillo, Miguel Lopez
