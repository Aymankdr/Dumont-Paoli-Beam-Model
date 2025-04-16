import numpy as np
import os, shutil
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import qpsolvers
import quadprog
import time
from scipy.optimize import minimize
import copy
import sys
import platform
# Importing other files
sys.path.insert(1,'C:/Users/AYMAN/Desktop/scripts_python/PhD_beam/personal_version')
import BeamPropertiesClass as bp
import ObstaclePropertiesClass as op
import BeamModelClass as bm
import TimeIntegrationClass as tim
#import parameters_dumont_paoli as pm


if __name__ == '__main__':
    # ----------- Constants ------------

    E = 2*10**11 # Constant Young Modulus
    diam_max = 0.01   # Cylindrical beam - Outer circle
    diam_min = 0.009  # Cylindrical beam - Inner circle
    I = (np.pi/64)*(diam_max**4-diam_min**4) # Inertia Moment
    S = np.pi*((diam_max/2)**2-(diam_min/2)**2) # Section Area
    rho = 8000   # Density
    alpha = 1e-2 # Damping Coefficient

    # ----------- Space discretised beam model ------------
    L  = 1.501  # Total Length
    Nbel = 60   # Number of Elements
    mass_matrix_type = 'Regular'
    mass_matrix_type = 'Singular'

    # ------------------- Obstacles ----------------------
    contact_type = 'Ponctual'
    contact_type = 'Planar'
    gap, bending = 0.05, 0.03
    inf_obs, sup_obs = -gap, gap
    inf_obs, sup_obs = -gap, (gap, 2*gap)
    inf_obs, sup_obs = -gap, [4*bending/L**2, -4*bending/L, gap]
    inf_obs, sup_obs = [-4*bending/L**2, 4*bending/L, -gap], [4*bending/L**2, -4*bending/L, gap]

    # ----------- Time discretised beam model ------------
    dt    = 0.01
    tmax  = 3.0
    amp   = 0.5
    omega = 10.0
    beta  = 1/4
    e     = 0
    solve_method = 'Penalty'
    solve_method = 'Lagrangian'
    epsilonp = 5e+0

    # ------------ Initial Conditions --------------------
    u0 = np.poly1d([0]) # lambda x: 0
    v0 = np.poly1d([0]) # lambda x: 0

    # -------------------- Classes -------------------
    BP = bp.BeamProperties(E,I,rho,S,alpha) # BeamProperties class element
    OP = op.ObstacleProperties(inf_obs, sup_obs, 2*Nbel, L, contact_type) # ObstacleProperties class element
    BM = bm.DumontPaoliBeamModel(L, Nbel, mass_matrix_type, BP)
    TIM = tim.TimeIntegrationModel(BM, OP, u0, v0, dt, tmax, amp, omega, beta, e, solve_method, epsilonp)

    # --------------------Tests------------------
    Qn = 1.5*np.ones(2 * Nbel)
    x = np.linspace(0,L,Nbel+1)

    P = TIM.compute_FirstTermMatrix()

    r = TIM.penalty_force(np.array(Nbel//2*[2,0,-3,0])); print(r)

    """
    print(OP.contact_test(Qn))
    print(BP.Density([1]))
    print(BM.E)
    print(r)
    print(TIM.Uddt)
    """

    tn, Vn, U, Et, Rt, Ut, Vt, Uddt = 0, np.zeros(Nbel), [np.zeros(2*Nbel)], [], [], [], [], []
    for i in range(1700):
        TIM.iterate()

    TIM.plot_animation(2)