import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as alg


class DumontPaoliBeamModel():
    def __init__(self, L, Nbel, mass_matrix_type, BP):
        # ----- Defining Mesh Parameters --------------
        self.L     = L                         # Total Length
        self.Nbel  = Nbel                      # Number of Elements
        self.Ndof  = self.Nbel*2               # Number of Degrees of Freedom
        self.Nb_perturbed_nodes  = self.Ndof   # Number of Perturbed Nodes
        self.Nb_observed_nodes_U = self.Ndof   # Number of Observed Displacement Nodes
        self.Nb_observed_nodes_V = self.Nb_observed_nodes_U//2 # Number of Observed Velocity Nodes
        self.h     = self.L/self.Nbel          # Length of one Element (mesh-size)
        # ------- Elements Properties --------------
        self.nodes           = np.linspace(0,L,self.Nbel+1)
        self.element_centers = 0.5 * (self.nodes[1:] + self.nodes[:-1])
        self.E               = BP.YoungModulus(self.element_centers)
        self.I               = BP.InertiaMoment(self.element_centers)
        self.rho             = BP.Density(self.element_centers)
        self.S               = BP.SectionArea(self.element_centers)
        self.alpha           = BP.DampingCoefficient(self.element_centers)
        # -------- Type of Model ----------------------
        self.mass_matrix_type = mass_matrix_type

        # --- Computing Elementary Matrices -----
        self.Ke = self.compute_Ke() # Elementary stiffness matrix
        if mass_matrix_type == 'Singular':
            self.Ce = self.compute_Ce() # Elementary mass matrix (P1-Lagrange)
            self.Be = self.compute_Be() # Elementary transfert matrix (from Hermite to Lagrange)
        else:
            self.Me_reg = self.compute_Me_reg() # Elementary mass matrix

        # ----- Computing Global Matrices -------
        self.K  = self.compute_K() # Stiffness matrix (with all nodes)
        self.A  = self.compute_A() # Damping matrix (with all nodes)
        if mass_matrix_type == 'Singular':
            self.C = self.compute_C()
            self.B = self.compute_B()
            # This version of 'M' depends on 'B' and 'C'
            self.M = self.compute_M()
        elif mass_matrix_type == 'Regular':
            self.M = self.compute_M()

        # --- Computing the shift function ------
        self.x_shift = self.compute_x_shift()


    def compute_Ke(self):
        h = self.h
        return (1/(h**3))*np.array([[12, 6*h, -12, 6*h],
                                    [6*h, 4*h**2, -6*h, 2*h**2],
                                    [-12, -6*h, 12, -6*h],
                                    [6*h, 2*h**2, -6*h, 4*h**2]])

    def compute_Me_reg(self):
        h = self.h
        return h/420*np.array([[156, 22*h, 54, -13*h],
                                [22*h, 4*h**2, 13*h, -3*h**2],
                                [54, 13*h, 156, -22*h],
                                [-13*h, -3*h**2, -22*h, 4*h**2]])

    def compute_Ce(self):
        return (self.h/6)*np.array([[2, 1],
                                    [1, 2]])

    def compute_Be(self):
        h = self.h
        return (h/60)*np.array([[9, 21],
                                [2*h, 3*h],
                                [21, 9],
                                [-3*h, -2*h]]).T

    def compute_K(self):
        # Global stiffness matrix with all DOFs initially
        K_global = sp.lil_matrix((self.Ndof + 2, self.Ndof + 2))
        # Assembly loop (element by element)
        for e in range(self.Nbel):
            idx = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
            K_global[np.ix_(idx, idx)] += self.E[e] * self.I[e] * self.Ke
        # Extract the reduced stiffness matrix for free DOFs (DOFs 0 and 1 are zero)
        K = K_global[2:,:][:,2:].tocsr()
        return K

    def compute_A(self):
        # Global damping matrix with all DOFs initially
        A_global = sp.lil_matrix((self.Ndof + 2, self.Ndof + 2))
        # Assembly loop (element by element)
        for e in range(self.Nbel):
            idx = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
            A_global[np.ix_(idx, idx)] += self.alpha[e] * self.E[e] * self.I[e] * self.Ke
        # Extract the reduced damping matrix for free DOFs (DOFs 0 and 1 are zero)
        A = A_global[2:,:][:,2:].tocsr()
        return A

    def compute_C(self):
        # Global stiffness matrix with all DOFs initially
        C_global = sp.lil_matrix((self.Nbel + 1, self.Nbel + 1))
        # Assembly loop (element by element)
        for e in range(self.Nbel):
            idx = [e, e + 1]
            C_global[np.ix_(idx, idx)] += self.rho[e] * self.S[e] * self.Ce
        # Extract the reduced stiffness matrix for free DOFs (DOFs 0 and 1 are zero)
        C = C_global[1:,:][:,1:].tocsc()
        return C

    def compute_B(self):
        # Global stiffness matrix with all DOFs initially
        B_global = sp.lil_matrix((self.Nbel + 1, self.Ndof + 2))
        # Assembly loop (element by element)
        for e in range(self.Nbel):
            id1 = [e, e + 1]
            id2 = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
            B_global[np.ix_(id1, id2)] += self.rho[e] * self.S[e] * self.Be
        # Extract the reduced stiffness matrix for free DOFs (DOFs 0 and 1 are zero)
        B = B_global[1:,:][:,2:].tocsr()
        return B

    def compute_M(self):
        if self.mass_matrix_type == 'Singular':
            return (self.B).T @ alg.inv(self.C).tocsr() @ (self.B)
        elif self.mass_matrix_type == 'Regular':
            # Global stiffness matrix with all DOFs initially
            M_global = sp.lil_matrix((self.Ndof + 2, self.Ndof + 2))
            # Assembly loop (element by element)
            for e in range(self.Nbel):
                idx = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
                M_global[np.ix_(idx, idx)] += self.rho[e] * self.S[e] * self.Me_reg
            # Extract the reduced stiffness matrix for free DOFs (DOFs 0 and 1 are zero)
            M_reg = M_global[2:,:][:,2:].tocsr()
            return M_reg

    # -------- Shifting function ------------

    def compute_x_shift_e(self,i): # the element being [x_{i-1}, x_{i}]
        l = self.h
        L = self.L
        N = self.Nbel
        # Calculating the integral on the element of [N_1(x), N_2(x), N_3(x), N_4(x)] * (1 - x/L):
        return np.array([l * (2.0 * ((0.5 * N - 0.5 * i + 0.5) / N) + 0.2 * l / L) / 2,
                        l * (2.0 * ((0.125 * N * l - 0.125 * i * l + 0.125 * l) / N)
                        - 0.025 * l**2 / L + (2.0/3.0) * ((-0.125 * L * N * l +
                        0.125 * L * i * l - 0.125 * L * l + 0.0625 * N * l**2) / (L * N))) / 2,
                        l * (2.0 * ((0.5 * N - 0.5 * i + 0.5) / N) - 0.2 * l / L) / 2,
                        l * (2.0 * ((-0.125 * N * l + 0.125 * i * l - 0.125 * l) / N)
                        - 0.025 * l**2 / L + (2.0/3.0) * ((0.125 * L * N * l -
                        0.125 * L * i * l + 0.125 * L * l + 0.0625 * N * l**2) / (L * N))) / 2
                        ])
        """
        return np.array([h/2 * (1 + h/(5*L)) ,
                        h * (h/4 - h**2/(40*L)) + (-h/8 + h**2/(16*L))/3 ,
                        h/2 * (1 - h/(5*L)) ,
                        -h * (h/4 + h**2/(40*L)) + (h/8 + h**2/(16*L))/3])
        """

    def compute_x_shift(self):
        # Global shift vector with all DOFs initially
        x_shift_global = np.zeros(self.Ndof + 2)
        # Assembly loop (element by element)
        for e in range(self.Nbel):
            idx = [2*e, 2*e + 1, 2*e + 2, 2*e + 3]
            x_shift_global[idx] += self.compute_x_shift_e(e + 1)
        # Extract the reduced stiffness matrix for free DOFs (DOFs 0 and 1 are zero)
        x_shift = x_shift_global[2:]
        return x_shift
