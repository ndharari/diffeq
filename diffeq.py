# %%
# Bulk Imports and setup
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
from sympy.utilities.lambdify import lambdify, implemented_function
x, y, lamda = sm.symbols('x, y, lamda')

def move_sympyplot_to_axes(p, ax):
    backend = p.backend(p)
    backend.ax = ax
    backend._process_series(backend.parent._series, ax, backend.parent)
    backend.ax.spines['right'].set_color('none')
    backend.ax.spines['bottom'].set_position('zero')
    backend.ax.spines['top'].set_color('none')
    plt.close(backend.fig)


# %%
# Defining the equations via sympy
dot_x = x -y   # f(x,y)
dot_y = -2*x+y  # g(x,y)
# %%
dot_x
# %%
dot_y
# %%
# Setting equations to zero and computing Fixed Points
dot_x_Equal, dot_y_Equal = sm.Eq(dot_x, 0), sm.Eq(dot_y, 0)
equilibria = sm.solve((dot_x_Equal, dot_y_Equal), x, y, dict=True)
equilibria
# %%
## A matrix from dot(X)=AX (if system is linear)
# We contruct the vector <f(x,y), g(x,y)>.
A_vector = sm.Matrix([dot_x, dot_y])
# Jacobian for the A_vector. In a linear system this gives the A matrix from dot(X)=AX
A_matrix = A_vector.jacobian(([x, y]))
A_matrix
# %%
# Calculates Characteristic Polynomial
charpoly = A_matrix.charpoly()
charpoly 
# %%
# Calculates eigenvals for each eq
# Note, each eigenval is presented as {eiganval: multiplicity}
eigenval_list = [[eq, A_matrix.subs([(x, eq[x]), (y, eq[y])]).eigenvals()]
              for eq in equilibria]
eigenval_list
# %%
# Ploting  parameters
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(1,1,1)
# ax.set_title("Sistema diferencial")

# Setting Plot Boundaries (needed of sympy)
x_min, y_min, y_max, x_max = -2, -2, 2, 2
# For plotting the trajectory
time, dt = 50, 0.1
x_0, y_0 = 0.5, 0.6 

#################################
# Plot fixed points
for point in equilibria:
    ax.plot(point[x],point[y],"red", marker = "o", markersize = 10.0)
    

# To plot we translate from sympy to numpy
x_movement = lambdify((x, y), dot_x) 
y_movement = lambdify((x, y), dot_y)

# define a grid and compute direction at each point
x_line = np.linspace(x_min, x_max, 20)
y_line = np.linspace(y_min, y_max, 20)

# Plot a singular trayectory
x_list= [x_0]
y_list = [y_0]
for i in range(time):
    x_list.append(x_list[i] + (x_movement(x_list[i], y_list[i])) * dt)
    y_list.append(y_list[i] + (y_movement(x_list[i], y_list[i])) * dt)

ax.plot(x_list, y_list, color="black")

# Quiverplot
# create a grid and set the dir vecs
X, Y = np.meshgrid(x_line, y_line)               
x_dir, y_dir = x_movement(X,Y), y_movement(X,Y)  
# prepare the colour
M = (np.hypot(x_dir, y_dir))                                                   
plt.quiver(X, Y, 
           x_dir, 
           y_dir,
           M,
           pivot='mid')

# Plot Nullclines
p1 = sm.plot_implicit(dot_x_Equal, (x, x_min, x_max), (y, y_min, y_max), show= False)
p2 = sm.plot_implicit(dot_y_Equal, (x, x_min, x_max), (y, y_min, y_max), show= False)

move_sympyplot_to_axes(p1, ax)
move_sympyplot_to_axes(p2, ax)

plt.show()
