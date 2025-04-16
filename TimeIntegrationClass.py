import numpy as np
import qpsolvers
import scipy.sparse as sp
import scipy.sparse.linalg as alg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

class TimeIntegrationModel:
    def __init__(self, BM, OP, u0, v0, dt, tmax, amp, omega, beta, e, solve_method, epsilonp=0, storage_option = 'all'):
        # Beam Model and Obstacle Properties classes
        self.BM    = BM
        self.OP    = OP
        # Initial Conditions' Space Polynomials
        self.u0    = u0
        self.v0    = v0
        # Time Scheme constants
        self.dt    = dt
        self.tmax  = tmax
        self.amp   = amp
        self.omega = omega
        self.beta  = beta
        self.e     = e
        self.solve_method = solve_method
        self.epsilonp = epsilonp
        # Mesh size
        self.Ndof = BM.Ndof
        self.Nbel = BM.Nbel
        # Obstacles
        self.lower_bound = OP.lower_bound
        self.upper_bound = OP.upper_bound
        self.lb, self.ub = OP.lb, OP.ub
        # Space Domain
        self.x  = OP.x
        # Excitation force
        self.F0 = amp * omega**2 * self.BM.x_shift
        # Original Matrices
        self.M  = BM.M
        self.A  = BM.A
        self.K  = BM.K
        if BM.mass_matrix_type == 'Singular':
            self.B, self.C = BM.B, BM.C
        # Matrices multiplying previous terms in Gn
        self.Pn   = 2*self.M - (self.dt**2)*(1-2*self.beta)*self.K
        self.Pnm1 = self.M + (dt**2)*beta*self.K - (dt/2)*self.A
        self.Pnp1 = self.compute_FirstTermMatrix() # 'Unp1' is the unknown of the system
        # Iterative scheme storage option for 'U', 'Ud' and 'Udd' ('last_node' or 'all')
        self.storage_option = storage_option
        # Lists of strorage
        self.U, self.V, self.Ud, self.Udd = self.init_problem_unknowns()
        self.Et, self.Rt, self.Ut, self.Udt, self.Uddt = [], [], [], [], []
        self.t = [0, self.dt]

    def compute_FirstTermMatrix(self):
        # The Newmark-beta scheme can be rearranged into : P U^{n+1} = G^n
        dt = self.dt
        P  = self.M +(dt**2)*self.beta*self.K + (dt/2)*self.A
        return P

    def penalty_force(self, Unp1):
        #Initialisation
        epsilonp  = self.epsilonp
        _, where_upper_contact, where_lower_contact = self.OP.contact_test(Unp1)
        Unp1_lower_contact = np.where(where_lower_contact, Unp1, 0)
        Unp1_upper_contact = np.where(where_upper_contact, Unp1, 0)
        lower_force = (Unp1_lower_contact - self.lower_bound*where_lower_contact) # < 0
        upper_force = (Unp1_upper_contact - self.upper_bound*where_upper_contact) # > 0
        return - (1/epsilonp) * (lower_force + upper_force)

    def compute_F(self, t):
        # F_i(t) = amp * omega^2 * sin (omega * t) * < 1 - x , \phi_i >_{L^2}
        return np.sin(self.omega * t) * self.F0

    def compute_Gn(self, Unm1, Un, Fn):
        # Newmark-beta scheme
        return (self.Pn) @ Un - (self.Pnm1) @ Unm1 + (self.dt**2)*Fn

    def compute_Qn_and_Rn(self, Un, Gn):
        # 'Un' has the role of an initial value for system solvers
        Pnp1 = self.Pnp1
        if self.solve_method == 'Lagrangian':
            Qn = qpsolvers.solve_qp(self.Pnp1.todense(), -Gn, lb=self.lb, ub=self.ub,
                solver='quadprog', initvals=Un) #.reshape(self.Ndof,1)
            Rn = (self.Pnp1) @ Qn - Gn
        elif self.solve_method == 'Penalty':
            Rn = self.penalty_force(Un)
            Qn = alg.cg(self.Pnp1, Gn + Rn, x0 = Un)[0]
            # Rn = (self.Pnp1) @ Qn - Gn (with residual)
        return Qn, Rn

    def init_problem_unknowns(self):
        # Sizing
        U0, Ud0 = np.zeros(self.Ndof), np.zeros(self.Ndof)
        # Filling
        U0[::2], Ud0[::2] = self.u0(self.x), self.v0(self.x)
        U0[1::2], Ud0[1::2] = self.u0.deriv()(self.x), self.v0.deriv()(self.x)
        # Initial H-Velocity
        if self.BM.mass_matrix_type == 'Singular':
            V0 = self.B.T @ alg.cg(self.C, self.B @ Ud0)[0]
        else:
            V0 = np.copy(Ud0)
        # Projecting on the mass matrix kernel complementary to find W-Acceleration:
        #      <<<   M^2 Udd(0) = M (F(0) - A Ud(0) - K U(0))   >>>
        self.M2 = self.M @ self.M
        rhs0 = self.M @ (self.F0 - self.A @ Ud0 - self.K @ U0 )
        Udd0 = alg.cg(self.M2, rhs0)[0]
        # The next iteration
        Ud1 = Ud0 + self.dt * Udd0 # non centered finite difference
        if self.BM.mass_matrix_type == 'Singular':
            V1 = self.B.T @ alg.cg(self.C, self.B @ Ud1)[0]
        else:
            V1 = np.copy(Ud1)
        U1   = U0 + self.dt * Ud0
        rhs1 = self.M @ (self.compute_F(self.dt) - self.A @ Ud1 - self.K @ U1 )
        Udd1 = alg.cg(self.M2, rhs1)[0]
        # Output
        U   = [U0, U1]
        Ud  = [Ud0, Ud1]
        V   = [V0, V1]
        Udd = [Udd0, Udd1]
        return U, V, Ud, Udd


    def iterate(self):
        """
        -/ Et, Rt, Ut, Vt are the time-lists of stored variables
        -/ tn, Vn are variables depending on the iteration 'n'
        -/ U stores mainly 'Unm1' and 'Un'
        """
        # ----- Extracting previous instances --------------
        # try:         except: Unm1 = U[-1].copy()
        # Starting at t1, there is only one vector U0 so there is no U[-2]
        Unm1   = self.U[-2].copy()
        Un     = self.U[-1].copy()
        Udnm1  = self.Ud[-2].copy()
        Udn    = self.Ud[-1].copy()
        Uddnm1 = self.Udd[-2].copy()
        Uddn   = self.Udd[-1].copy()
        # ------------ Excitation Force ---------------------
        tnp1 = self.t[-1] + self.dt
        Fnp1 = self.compute_F(tnp1)
        # ------- Computing next iteration --------------------
        Gn = self.compute_Gn(Unm1, Un, Fnp1)
        Unp1, Rn = self.compute_Qn_and_Rn(Un, Gn)
        # --- Computing W-acceleration and (W/H)-velocity -------
        Udnp1  = (3*Unp1 - 4*Un + Unm1)/(2*self.dt)
        Uddnp1 = (Unp1 -2*Un + Unm1)/(self.dt**2)
        # ---------- Mechanical Energy --------------------
        En = (0.125/self.dt**2) * (Unp1-Unm1) @ self.M @ (Unp1-Unm1) + 0.5*(Un @ self.K @ Un)
        # ---------- Filling the time lists ----------------
        self.Rt.append(Rn[-1])
        self.Ut.append(Unp1[-2])
        self.Udt.append(Udnp1[-2])
        self.Uddt.append(Uddnp1[-2])
        self.Et.append(En)
        self.t.append(tnp1)
        # ----------- Swapping/Storing vectors ------------
        if self.storage_option == 'last_node':
            self.U   = [Un, Unp1]
        else:
            # For the animation of 'U(t)':
            self.U.append(Unp1)
        # --------------- Storing vectors ----------------
        self.Ud  = [Udn, Udnp1]
        self.Udd = [Uddn, Uddnp1]

    def plot_animation(self, frame_packet_size):
        assert self.storage_option == 'all'
        Uplot = []
        dt_ref = 1e-3
        L = self.BM.L
        for u in self.U[::frame_packet_size]:# self.U[::int((dt_ref/self.dt + 1)/2)]:
            Uplot.append(u[::2])
        t = self.t #self.t[::int((dt_ref/self.dt + 1)/2)]

        fig, ax = plt.subplots(figsize=(12,5))
        ax.set_xlim([-0.1, L*1.1])
        if self.OP.contact_type=='Planar':
            ax.set_ylim([1.4*self.OP.inf_obs, 1.4*self.OP.sup_obs])
        elif self.OP.contact_type=='Ponctual':
            ax.set_ylim([self.OP.inf_obs*1.1, self.OP.sup_obs*1.1])

        # ax.set_ylim([-5,5])

        # Plot static permanent lines within the y-axis limits
        if self.OP.contact_type == 'Planar':
            static_line1, = ax.plot([0, L], [self.OP.sup_obs, self.OP.sup_obs], '--', color='red', linewidth=3)
            static_line2, = ax.plot([0, L], [self.OP.inf_obs, self.OP.inf_obs], '--', color='red', linewidth=3)
        elif self.OP.contact_type == 'Ponctual':
            static_line1 = ax.scatter([L], [self.OP.sup_obs], s=50, color='red', marker='o')
            static_line2 = ax.scatter([L], [self.OP.inf_obs], s=50, color='red', marker='o')
        else:
            static_line1, = ax.plot([], [])
            static_line2, = ax.plot([], [])

        line, = ax.plot([], [], linestyle='-',marker='o', linewidth=3)

        # Add a text object to display the time
        time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='blue', fontsize=12, verticalalignment='top')

        def init():
            line.set_data([], [])
            time_text.set_text('')  # Initialize the time text
            return line, static_line1, static_line2, time_text

        def animate(i):
            y_data = Uplot[i]
            x_data = np.linspace(0,L,self.Nbel)
            line.set_data(x_data, y_data)
            current_time = t[i]  # Get the current time from the array `t`
            time_text.set_text(f't = {current_time:.2f} seconds')  # Update time text
            return line, static_line1, static_line2, time_text

        ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Uplot), interval=1, blit=True)
        plt.xlabel('Length [m]')
        plt.ylabel('Vertical displacement [m]')
        plt.title('Beam displacement as a function of time')
        # Uncomment the following lines to save the animation
        # ani.save('animation.gif', writer='pillow', fps=30)
        plt.show()



"""
        if self.BM.mass_matrix_type == 'Singular':
            # <<< V^{n+1} = V^n + dt * C^{-1} B (Udd^{n-1} + Udd^n) >>>
            C_delVn = (self.dt/2) * self.B @ (Uddnm1 + Uddn) # Formula from Matthieu Schorsch's code
            delVn = alg.cg(self.C, C_delVn, x0 = Vn)[0]
            Vnp1 = Vn + delVn
"""




