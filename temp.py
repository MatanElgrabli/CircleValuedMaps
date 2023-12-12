# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 23:54:28 2023

@author: matan
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

#matplotlib.use('Agg')

# Create a parametric representation of a torus
theta = np.linspace(0, 2 * np.pi, 20)
phi = np.linspace(0, 2 * np.pi, 20)
theta, phi = np.meshgrid(theta, phi)
R = 1  # Major radius of the torus
r = 0.5  # Minor radius of the torus


def chart_inverse(theta, phi,r=0.5,R=1):
    # Calculate torus points
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    
    return x, y, z

x, y, z =chart_inverse(theta,phi)


# %%
def push_forward(theta, phi, u):
    e_theta = (-r*np.sin(theta)*np.cos(phi), -r*np.sin(theta)*np.sin(phi), r*np.cos(theta))
    e_phi = (-(R+r*np.cos(theta))*np.sin(phi), (R+r*np.cos(theta))*np.cos(phi), np.zeros_like(theta))
    
    U = np.real(u)*e_theta[0]+np.imag(u)*e_phi[0]
    V = np.real(u)*e_theta[1]+np.imag(u)*e_phi[1]
    W = np.real(u)*e_theta[2]+np.imag(u)*e_phi[2]


    return U, V, W
    

def plot_vector_field_3d(theta,phi, u, points=[]):
    # Calculate the vectors at each point on the torus
    U, V, W = push_forward(theta, phi, u)
    x, y, z = chart_inverse(theta,phi)
    
    x_eps, y_eps, z_eps =chart_inverse(theta,phi,r=0.5)
    
    # Create the 3D torus plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the torus
    ax.plot_surface(x, y, z, cmap='viridis', alpha=1, antialiased=True, zorder=2)
    
 
    # Create the 3D vector field plot
    vector_field_plot = ax.quiver(x_eps, y_eps, z_eps, U, V, W, length=0.15,  linewidth=1, alpha = 1.0, 
                                  normalize=True, color='black', pivot='middle', zorder=1)
        
    
    
    for p in points:
        p_x, p_y, p_z = chart_inverse(np.imag(p),np.real(p))
        ax.scatter(p_x, p_y, p_z, c='r', marker='o', s=1000)
    
    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('3D Vector Field on a Torus')
    
    # Set axis limits
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-1, 1)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
def plot_vector_field_2d(theta, phi, u, points=[]):
    
    
    # Extract real and imaginary parts
    dphi = np.real(u)
    dtheta = np.imag(u)
    
    # Create the 3D torus plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    
    for p in points:
        ax.scatter(np.real(p), np.imag(p), c='red', marker='o', label='Points')
    
    # Create the vector field plot
    vector_field_plot = ax.quiver( phi, theta,  dphi,dtheta,scale=40, color='blue')
    
    # Customize arrow properties
    vector_field_plot.set_edgecolor('k')  # Add black edges to arrows
    vector_field_plot.set_linewidth(0.5)  # Adjust arrow linewidth
    
    # Set axis limits and labels
    ax.set_xlim(-0.1, 2*np.pi+0.1)
    ax.set_ylim(-0.1, 2*np.pi+0.1)
    ax.set_xlabel(r'$\phi$-axis')
    ax.set_ylabel(r'$\theta$-axis')
    
    # Show the plot
    plt.show()

    
n = 1 # Real part
m = 1  # Imaginary part
u = lambda theta, phi: np.exp(1j*(n*theta+m*phi))
plot_vector_field_3d(theta, phi,u(theta,phi))
plot_vector_field_2d(theta, phi, u(theta,phi))

# %%

a, b = np.pi*1.5+np.pi*0.5j, np.pi*0.5+np.pi*1.5j
def f(theta,phi):
    z= phi+1j*theta
    u=(z-a)/np.abs(z-a)*np.abs(z-b)/(z-b)
    return u

plot_vector_field_2d(theta, phi, f(theta,phi))

# %%
import numba
from numba import jit
xi = np.angle(f(theta,phi))

@numba.jit("f8[:,:](f8[:,:], i8)", nopython=True, nogil=True)
def harmonic_extension(xi, n_iter):
    length=len(xi[0])
    for n in range(n_iter):
        for i in range(1,length-1):
            for j in range(1,length-1):
                xi[i][j]=1/4*(xi[j+1][i]+xi[j-1][i]+xi[j][i+1]+xi[j][i-1])
    return xi


psi = np.zeros_like(xi)
psi[0,:] = xi[0,:]
psi[-1,:] = xi[-1,:]
psi[:,0] = xi[:,0]
psi[:,-1] = xi[:,-1]

psi = harmonic_extension(psi, 100000)
g = f(theta, phi)*np.exp(-psi*1j)

plot_vector_field_2d(theta, phi, g, [a,b])

n=-1
m=-1
g = g*np.exp(1j*(n*theta+m*phi))
plot_vector_field_2d(theta, phi, g, [a,b])
# plot_vector_field_3d(theta, phi, g)
