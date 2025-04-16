import numpy as np

class BeamProperties:
    def __init__(self,E0,I0,rho0,S0,alpha0):
        # Storing reference constants
        self.E0     = E0
        self.I0     = I0
        self.rho0   = rho0
        self.S0     = S0
        self.alpha0 = alpha0

    # ----- Defining Space distribution --------

    def YoungModulus(self,x):
        return self.E0 * np.ones(len(x))

    def InertiaMoment(self,x):
        return self.I0 * np.ones(len(x))

    def Density(self,x):
        return self.rho0 * np.ones(len(x))

    def SectionArea(self,x):
        return self.S0 * np.ones(len(x))

    def DampingCoefficient(self,x):
        return self.alpha0 * np.ones(len(x))





