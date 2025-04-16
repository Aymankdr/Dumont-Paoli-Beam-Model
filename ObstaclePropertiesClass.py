import numpy as np

class ObstacleProperties:
    def __init__(self, inf_obs, sup_obs, Ndof, L, contact_type='Planar'):
        # Store constants correctly
        self.sup_obs = sup_obs #max(sup_obs, inf_obs) # only for scalar value
        self.inf_obs = inf_obs #min(sup_obs, inf_obs) # only for scalar value
        self.contact_type = contact_type
        self.nodal_adaptation = False

        # Discretisation properties
        self.Ndof = Ndof
        self.x    = np.linspace(0,L,Ndof//2 + 1)[1:]

        # For the optimisation problem
        self.lb = np.ones(Ndof) * (- np.inf)
        self.lb[::2] = self.g1(self.x)
        self.ub = np.ones(Ndof) * (np.inf)
        self.ub[::2] = self.g2(self.x)

        # For calculating penalty
        self.lower_bound = np.zeros(Ndof)
        self.lower_bound[::2] = self.g1(self.x)
        self.upper_bound = np.zeros(Ndof)
        self.upper_bound[::2] = self.g2(self.x)

        # Determine impacted nodes based on obstacle type
        if self.contact_type == 'Planar':
            has_contact = np.zeros(Ndof)
            # Displacement DOFs (every even-indexed DOF: 0,2,4,...)
            has_contact[::2] = 1
        elif self.contact_type == 'Ponctual':
            has_contact = np.zeros(Ndof)
            has_contact[-2] = 1  # Only last displacement node
        else:
            has_contact = np.zeros(Ndof)

        self.has_contact = has_contact.astype(bool)

        # Setting obstacle bounds where there is contact; elsewhere infinite
        self.sup_gp = np.where(self.has_contact[::2], self.upper_bound[::2], np.inf)
        self.inf_gp = np.where(self.has_contact[::2], self.lower_bound[::2], -np.inf)

    # ----- Define obstacle distributions -------
    def g1(self, x):
        if (type(self.inf_obs) is float) or (type(self.inf_obs) is int):
            return self.inf_obs * np.ones_like(x)
        elif (type(self.inf_obs) is tuple):
            return np.linspace(self.inf_obs[0], self.inf_obs[1], len(x))
        elif (type(self.inf_obs) is list):
            p_inf = np.poly1d(self.inf_obs)
            return p_inf(x)

    def g2(self, x):
        if (type(self.sup_obs) is float) or (type(self.sup_obs) is int):
            return self.sup_obs * np.ones_like(x)
        elif (type(self.sup_obs) is tuple):
            return np.linspace(self.sup_obs[0], self.sup_obs[1], len(x))
        elif (type(self.sup_obs) is list):
            p_sup = np.poly1d(self.sup_obs)
            return p_sup(x)

    # ----------- Contact Test Method -------------
    def contact_test(self, Qn):
        """
        Returns a boolean indicating if there's contact,
        and a binary array indicating which nodes have contact.

        Qn : The displacement-rotation vector (Ndof x 1)
        """

        # Reshape Qn to match node-based bounds
        disp_dofs = Qn[::2]  # Extract only displacement DOFs

        # Element-wise contact test
        upper_contact_nodes = (disp_dofs >= self.upper_bound[::2])
        lower_contact_nodes = (disp_dofs <= self.lower_bound[::2])
        contact_nodes = lower_contact_nodes | upper_contact_nodes

        # Map back to full DOF array
        where_upper_contact = np.zeros(self.Ndof, dtype=bool)
        where_lower_contact = np.zeros(self.Ndof, dtype=bool)
        where_upper_contact[::2] = upper_contact_nodes  # only displacement DOFs have obstacle
        where_lower_contact[::2] = lower_contact_nodes  # only displacement DOFs have obstacle

        established_contact = contact_nodes.any()

        return established_contact, where_upper_contact.astype(int), where_lower_contact.astype(int)