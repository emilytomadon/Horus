import os
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from scipy import spatial

def fun(x,y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)

n_points = 5
def get_xy_points():

    # Generate x, y coordinates for a grid between -1.0 and 1.0
    x_coords, y_coords = np.meshgrid(
        np.linspace(-1.0, 1.0, n_points),
        np.linspace(-1.0, 1.0, n_points),
    )

    # Add random offsets to the coordinates
    np.random.seed(42)
    x_coords += np.random.uniform(low=-0.2, high=0.2, size=(n_points, n_points))
    y_coords += np.random.uniform(low=-0.2, high=0.2, size=(n_points, n_points))

    # Flatten the coordinates into 1D arrays
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    xy_points = np.array(list(zip(x_coords, y_coords)))
    return xy_points


# ax.plot_surface(X, Y, Z, cmap="autumn_r")#, lw=0.5, rstride=1, cstride=1)

def common_graph():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x = np.mgrid[-1:1:50j, -1:1:50j]
    X, Y = np.mgrid[-1:1:50j, -1:1:50j]
    Z = np.sin(np.pi*X)*np.sin(np.pi*Y)
    plt.xlabel(r'$x$',size = 14,math_fontfamily='cm')
    plt.ylabel(r'$y$',size = 14,math_fontfamily='cm')
    ax.set_zlabel("Optimal Threshold",size = 11)
    ax.view_init(30,-45)


    return ax, X, Y, Z

def falcon_graph():
    ax, X, Y, Z = common_graph()
    ax.plot_surface(X, Y, Z, cmap="autumn_r")#, lw=0.5, rstride=1, cstride=1)
    if SHOW: plt.show()
    else: 
        folder = os.path.join(os.path.abspath(os.getcwd()), "Bachelorarbeit", "figures")
        plt.savefig(os.path.join(folder,"falcon.png"), bbox_inches='tight', dpi=300)

def fsn_graph():
    ax, X, Y, Z = common_graph()
    xy_points = get_xy_points()
    Zs = np.array([])
    for x,y in xy_points:
        Zs = np.append(Zs,fun(x,y))
    new_Z = np.copy(Z)
    for i in range(X.shape[0]):    
        for j in range(X.shape[1]):
            new_Z[i][j]= Zs[spatial.KDTree(xy_points).query((X[i][j],Y[i][j]))[1]]
    ax.plot_surface(X, Y, new_Z,cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
    if SHOW: plt.show()
    else: 
        folder = os.path.join(os.path.abspath(os.getcwd()), "Bachelorarbeit", "figures")
        plt.savefig(os.path.join(folder,"fsn.png"), bbox_inches='tight', dpi=300)

def voronoi():
    from scipy.spatial import Voronoi, voronoi_plot_2d
    vor = Voronoi(get_xy_points())

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    voronoi_plot_2d(vor, ax=ax)
    plt.xlabel(r'$x$',size = 14, math_fontfamily='cm')
    plt.ylabel(r'$y$',size = 14,math_fontfamily='cm')

    if SHOW: plt.show()
    else: 
        folder = os.path.join(os.path.abspath(os.getcwd()), "Bachelorarbeit", "figures")
        plt.savefig(os.path.join(folder,"voronoi.png"), bbox_inches='tight', dpi=300)

SHOW = True
voronoi()
fsn_graph()
falcon_graph()