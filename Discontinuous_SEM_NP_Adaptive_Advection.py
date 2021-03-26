### Discontinuous Galerkin SEM - Adapted for 1D from the single domain approximation.
import numpy as np
from matplotlib import pyplot as plt

class NodalDiscontinuousGalerkin():
    def __init__(self, N, K, xk):
        # Initialising Global Variables
        self.xk = xk
        self.xk_orig = xk
        self.K = K
        self.c = 1.0
        self.split_elems = []


        ### ----------- Changing N ----------- ###
        # (Vary with N)
        #            N_dict[KEY = xk[k] -> N
        self.N = N
        self.Nmax = 17
        self.size = 0

        # Initialise N_dict
        self.N_dict = {}
        self.initialise_N()

        # Initialise Nodes_and_Weights_dict
        #           Nodes_and_Weights[KEY = N] -> [LIST OF NODES[0] & WEIGHTS[1]] #
        self.Nodes_and_Weights_dict = {}
        self.bcw_dict = {}
        self.initialise_NaW()

        # Initialise the Dhat
        self.Dhat_dict = {}
        self.initialise_Dhat()
        # print("ENDING NEW INITIALISE PROCESS")
        ### ----------- Changing N ----------- ###

        [self.LGx, self.LGw] = self.LegendreGaussNodesAndWeights(N)
        self.wb = self.barycentricWeights(self.LGx)

        # Initialising elements and their lengths
        # Initialise the Jacobian for each element
        self.elementInit(K, xk)

        # Initial Conditions
        self.initial_conditions()

        # Initialise interpolated values
        self.fluxInit(K)
        return

    # Initialisation
    def initialise_N(self, el_list = None):
        if el_list != None:
            self.N_dict[el_list[0]] = self.N_dict[self.xk[el_list[1]][0]]
        else:
            for i in self.xk:
                self.N_dict[i[0]] = self.N
        return
    def initialise_NaW(self):
        for n in range(3, self.Nmax):
            nodes, weights = self.LegendreGaussNodesAndWeights(n)
            bcw = self.barycentricWeights(nodes)
            self.Nodes_and_Weights_dict[n] = [nodes, weights]
            self.bcw_dict[n] = [bcw]
    def initialise_Dhat(self):
        for n in range(3, self.Nmax):
            nodesandweights = self.Nodes_and_Weights_dict[n]
            LGx = nodesandweights[0]
            LGw = nodesandweights[1]
            Dij = self.polynomialDerivativeMatrix(LGx)
            Dhij = np.zeros_like(Dij, dtype='float')
            for j in range(0, n):
                for i in range(0, n):
                    Dhij[i,j] = -Dij[j,i] * (LGw[j]/LGw[i])
            self.Dhat_dict[n] = Dhij
    def elementInit(self, K, xk, el_key=None):
        # Initialising elements and their lengths
        self.xk = xk
        self.delta_x = np.ones(K)
        for k in range(0, K):
            self.delta_x[k] = xk[k][1] - xk[k][0]
        # Initialise the Jacobian for each element
        self.J = self.delta_x / 2
        self.Ji = 1 / self.J

        # Initialises new N_dict before soln split
        self.initialise_N(el_key)
        return
    def initial_conditions(self):
        for k in range(self.K):
            self.size += self.N_dict[self.xk[k][0]]

        self.xij = np.zeros((self.size), dtype='float')
        self.xi = np.zeros((self.size), dtype='float')
        m = 0
        sigma = 0.2
        for k in range(self.K):
            N = self.N_dict[self.xk[k][0]]
            LGx = self.Nodes_and_Weights_dict[N][0]
            for j in range(N):
                self.xi[j + m] = self.xk[k][0] + ((LGx[j] + 1.0) / 2.0) * self.delta_x[k]
                self.xij[j + m] = np.exp(-np.log(2) * np.power((self.xi[j + m] + 0.5), 2) / sigma ** 2)
            m += N
        self.init_cond = self.xij
    def fluxInit(self, ks):
        self.Fluxes = np.empty(shape=(ks), dtype='object')
        ind = 0
        for k in range(0, ks):
            N = self.N_dict[self.xk[k][0]]
            self.Fluxes[k] = self.fluxes(self.xij[ind:ind + N], k)

            ind += N
        return

    # h-refinement
    def element_split(self, el):
        self.xi_old = self.xi
        xk_new = []
        for k in range(self.K):
            if k < el:
                xk_new.append(self.xk[k])
            elif k == el:
                # Split
                a1 = self.xk[k][0]
                a2 = self.xk[k][1]
                a_mid = (a1+a2)/2

                xk_new.append(np.array([a1,a_mid]))
                xk_new.append(np.array([a_mid,a2]))
            else:
                xk_new.append(self.xk[k])
        self.elementInit(self.K+1, xk_new, [a_mid, el])
        self.solution_split(el)
    def solution_split(self, el):
        # Finding xis and xijs to be interpolated
        N = self.N_dict[self.xk[el][0]]
        K = self.K
        interp_val = np.zeros(N)
        xis_old_interp = np.zeros(N)

        # Get the first index of the 'split' element
        ind = 0
        for k in range(el):
            ind += self.N_dict[self.xk[k][0]]

        N = self.N_dict[self.xk[el][0]]
        ctr = 0
        for i in range(ind, ind+N):
            xis_old_interp[ctr] = self.xi_old[i]
            interp_val[ctr] = self.xij[i]
            ctr += 1

        # Recaluclating xi

        m = 0
        sigma = 0.2

        self.size = 0
        for k in range(K+1):
            self.size += self.N_dict[self.xk[k][0]]
        self.xi = np.zeros(self.size, dtype='float')
        for k in range(K+1):
            N = self.N_dict[self.xk[k][0]]
            LGx = self.Nodes_and_Weights_dict[N][0]
            for j in range(N):
                self.xi[j + m] = self.xk[k][0] + ((LGx[j] + 1.0) / 2.0) * self.delta_x[k]
            m += N

        N = self.N_dict[self.xk[el][0]]
        xis_new_interp = np.zeros(2*N)
        ctr = 0

        for i in range(ind, ind+2*N):
            xis_new_interp[ctr] = self.xi[i]
            ctr += 1

        T = self.polynomialInterpolationMatrix(xis_old_interp, self.bcw_dict[N][0], xis_new_interp)
        f = self.interpolateToNewPoints(T, interp_val)

        # Assembling new xij
        xij_new = np.zeros_like(self.xi, dtype='float')
        for i in range(len(xij_new)):
            if i < ind:
                xij_new[i] = self.xij[i]
            elif  ind <= i <ind+2*N:
                xij_new[i] = f[i-ind]
            else:
                xij_new[i] = self.xij[i-N]
        self.split_elems.append(self.xk[el][0])
        self.K += 1
        self.xij = xij_new
        self.fluxInit(self.K)

    # p-refinement
    def P_refinement(self, el):
        # Finding xis and xijs to be interpolated
        self.xi_old = self.xi
        N = self.N_dict[self.xk[el][0]]
        K = self.K
        interp_val = np.zeros(N)
        xis_old_interp = np.zeros(N)

        # Get the first index of the 'split' element
        ind = 0
        for k in range(el):
            ind += self.N_dict[self.xk[k][0]]

        N = self.N_dict[self.xk[el][0]]
        ctr = 0
        for i in range(ind, ind+N):
            xis_old_interp[ctr] = self.xi_old[i]
            interp_val[ctr] = self.xij[i]
            ctr += 1

        # Recaluclating xi

        m = 0
        sigma = 0.2
        self.size = 0
        for k in range(K):
            self.size += self.N_dict[self.xk[k][0]]
        self.size += 1
        self.xi = np.zeros(self.size, dtype='float')
        self.N_dict[self.xk[el][0]] += 1

        for k in range(self.K):
            N = self.N_dict[self.xk[k][0]]
            LGx = self.Nodes_and_Weights_dict[N][0]
            for j in range(N):
                self.xi[j + m] = self.xk[k][0] + ((LGx[j] + 1.0) / 2.0) * self.delta_x[k]
            m += N

        N = self.N_dict[self.xk[el][0]]
        xis_new_interp = np.zeros(N)
        ctr = 0
        for i in range(ind, ind+N):
            xis_new_interp[ctr] = self.xi[i]
            ctr += 1
        T = self.polynomialInterpolationMatrix(xis_old_interp, self.bcw_dict[N][0], xis_new_interp)
        f = self.interpolateToNewPoints(T, interp_val)

        # Assembling new xij
        xij_new = np.zeros_like(self.xi, dtype='float')

        N = self.N_dict[self.xk[el][0]]

        for i in range(len(xij_new)):
            if i < ind:
                xij_new[i] = self.xij[i]
            elif  ind <= i <ind+N:
                xij_new[i] = f[i-ind]
            else:
                xij_new[i] = self.xij[i-1]

        self.xij = xij_new
        self.fluxInit(self.K)


    # Calculations
    def DGTimeDerivative(self, t, k, xij):
        self.Fluxes[k] = self.fluxes(xij, k)
        if k == 0 and self.c > 0:
            xiL = self.Fluxes[-1][1]
            xiR = self.Fluxes[k][1]
            xijtd = -self.c * self.DGDerivative(xiL, xiR, xij, k)

        else:
            xiL = self.Fluxes[k-1][1]
            xiR = self.Fluxes[k][1]
            xijtd = - self.c * self.DGDerivative(xiL, xiR, xij, k)
        return xijtd
    def DGDerivative(self, xiL, xiR, xij, k):
        # Initialising Nodes/Weights
        N = self.N_dict[self.xk[k][0]]
        LGw = self.Nodes_and_Weights_dict[N][1]
        Dhij = self.Dhat_dict[N]
        xijd = np.matmul(Dhij, xij) * self.Ji[k]

        for j in range(0, len(xij)):
            xijd[j] = xijd[j] + ((xiR * self.lj1[j] - xiL * self.ljn1[j])/LGw[j]) * self.Ji[k]

        return xijd
    def fluxes(self, xij, k):
        N = self.N_dict[self.xk[k][0]]
        LGx = self.Nodes_and_Weights_dict[N][0]
        wb = self.bcw_dict[N][0]

        self.ljn1 = self.lagrangeInterpolatingPolynomials(-1, LGx, wb)
        self.lj1 = self.lagrangeInterpolatingPolynomials(1, LGx, wb)

        xiL = self.InterpolateToBoundary(xij, self.ljn1)
        xiR = self.InterpolateToBoundary(xij, self.lj1)

        return np.array([xiL, xiR])
    def g(self, t):
        sigma = 0.2
        ans = np.exp(-np.log(2)*np.power((0.0-0.5),2)/sigma**2)
        return ans
    def polynomialDerivativeMatrix(self, xj):
        wj = self.barycentricWeights(xj)
        D = np.zeros((len(xj), len(xj)))


        for i in range(0, len(xj)):
            D[i,i] = 0
            for j in range(0, len(xj)):
                if j != i:
                    D[i,j] = (wj[j]/wj[i]) * 1/(xj[i] - xj[j])
                    D[i,i] = D[i,i] - D[i,j]

        return D
    def result(self):
        return [self.xi, self.xij]


    # Interpolation
    def InterpolateToBoundary(self, xij, lj):
        interpolatedValue = 0
        for j in range(0, len(xij)):
            interpolatedValue = interpolatedValue + lj[j] * xij[j]
        return interpolatedValue
    def barycentricWeights(self, xj):
        w = np.ones_like(xj)

        # Length of x = N
        for j in range(1, len(xj)):
            for k in range(0, j):
                w[k] = w[k] * (xj[k] - xj[j])
                w[j] = w[j] * (xj[j] - xj[k])

        for j in range(0, len(w)):
            w[j] = 1.0 / w[j]
        return w
    def lagrangeInterpolatingPolynomials(self, x, xj, wj):
        lj = np.zeros_like(xj)

        xMatchesNode = False
        for j in range(0, len(xj)):
            lj[j] = 0
            if (self.AlmostEqual(x, xj[j])):
                lj[j] = 1
                xMatchesNode = True

        if xMatchesNode:
            return lj

        s = 0
        for j in range(0, len(xj)):
            t = wj[j]/(x-xj[j])
            lj[j] = t
            s = s + t

        for j in range(0, len(xj)):
            lj[j] = lj[j] / s

        return lj
    def polynomialInterpolationMatrix(self, xj, wj, xij):
        Tkj = np.zeros((len(xij), len(xj)), dtype='float')
        for k in range(0, len(xij)):
            rowHasMatch = False
            for j in range(0, len(xj)):
                if (self.AlmostEqual(xij[k], xj[j])):
                    rowHasMatch = True
                    Tkj[k, j] = 1

            if (rowHasMatch == False):
                s = 0
                for j in range(0, len(xj)):
                    t = wj[j] / (xij[k] - xj[j])
                    Tkj[k, j] = t
                    s = s + t

                for j in range(0, len(xj)):
                    Tkj[k, j] = Tkj[k, j] / s
        return (Tkj)
    def interpolateToNewPoints(self, Tij, fj):
        fInterp = np.ones(Tij.shape[0])
        for i in range(0, Tij.shape[0]):
            t = 0
            for j in range(0, Tij.shape[1]):
                t = t + Tij[i, j] * fj[j]
            fInterp[i] = t
        return fInterp
    def AlmostEqual(self, a, b):
        eps = np.finfo(float).eps
        if (a == 0 or b == 0):
            if (abs(a - b) <= 2 * eps):
                return True
            else:
                return False
        else:
            if (abs(a - b) <= abs(a) * eps and abs(a - b) <= abs(b) * eps):
                return True
            else:
                return False


    # Legendre Polynomials
    def legendre_function(self, N, x):
        if (N == 0):
            LN = 1
            LdashN = 0
        elif (N == 1):
            LN = x
            LdashN = 1
        elif N == 2:
            LN2 = 1
            LN1 = x
            LdashN2 = 0
            LdashN1 = 1
            LN = (((2 * N - 1) / N) * x * LN1) - (((N - 1) / N) * LN2)
            LdashN = LdashN2 + (((2 * N) - 1) * LN1)
        else:
            LN2 = 1
            LN1 = x
            LdashN2 = 0
            LdashN1 = 1

            for k in range(2, N):
                LN = (((2 * k - 1) / k) * x * LN1) - (((k - 1) / k) * LN2)
                LdashN = LdashN2 + (((2 * k) - 1) * LN1)

                LN2 = LN1
                LN1 = LN

                LdashN2 = LdashN1
                LdashN1 = LdashN

        result = [LN, LdashN]

        return result
    def LegendreGaussNodesAndWeights(self, N):
        x = np.zeros(N, dtype='float')
        w = np.zeros(N, dtype='float')

        if (N == 0):
            x[0] = 0.0
            w[0] = 2.0
        elif (N == 1):
            x[0] = -np.sqrt(1.0/3.0)
            w[0] = 1.0
            x[1] = -x[0]
            w[1] = w[0]
        else:

            h = 0.01
            i = 0

            for j in np.arange(-1, 1, h):
                root = self.bisection(N+1, self.legendre_function, j, j + h)
                if (root != 999):
                    x[i] = root
                    i += 1

            for i in range(0, N):
                Result = self.legendre_function((N+1), x[i])
                w[i] = (2 / ((1.0-(pow(x[i], 2)))*pow(Result[1],2)))

        return x, w
    def bisection(self, N, func, a, b, eps=np.finfo(float).eps, max_iter=1000000):
        if ((func(N, a)[0] * func(N, b)[0]) <= 0):
            iterator = 0
            while (abs(a - b) >= eps and iterator <= max_iter):

                c = (a + b) / 2

                if ((func(N, a)[0] * func(N, c)[0]) > 0):
                    a = c
                elif ((func(N, a)[0] * func(N, c)[0]) < 0):
                    b = c
                elif ((func(N, c)[0] == 0)):
                    return c

                iterator += 1
            return c
        else:
            return 999


    def plot(self, t, T="N/A", errors=False):
        points = 5
        height = 1.2
        for num, val in enumerate(self.xk):
            x = np.ones(points) * val[0]
            y = np.arange(0, height, (height / points))
            plt.plot(x, y, linestyle='dashed', color='k')

            if num == len(mesh) - 1:
                x = np.ones(points) * val[1]
                y = np.arange(0, height, (height / points))
                plt.plot(x, y, linestyle='dashed', color='k')
                plt.title(
                    "Element splitting for DG SEM - Advection Equation (K={})\n(t={} for T={})".format(
                        self.K, np.round(t,2), np.round(T,2)))
        plt.scatter(self.xi, self.xij)
        xReal = np.arange(-1, 1, 2 / 1000)
        sigma = 0.2
        fReal0 = np.exp(-np.log(2) * np.power((xReal + 0.5 - t), 2) / sigma ** 2)
        fReal1 = np.exp(-np.log(2) * np.power((xReal + 2.5 - t), 2) / sigma ** 2)
        j = 0.4
        plt.plot(xReal, fReal0, color=(0.4+j, 0.1, 0.1), label='T (Real) = {}'.format(0.5+t))
        plt.plot(xReal, fReal1, color=(0.4+j, 0.1, 0.1), label='T (Real) = {}'.format(2.5+t))

        if errors == True:
            k_list_alt = [(self.xk[i][0]+self.xk[i][1])/2 for i in self.k_list]
            plt.plot(k_list_alt, np.abs(self.errors))
        plt.show()


    def coefficients(self, n, k):
        total = 0
        N = self.N_dict[self.xk[k][0]]
        ind = 0
        for i in range(k):
            ind += self.N_dict[self.xk[i][0]]

        LGn, LGw = self.Nodes_and_Weights_dict[N]
        for i in range(N):
            x = LGn[i]
            LN = self.legendre_function(n, x)[0]
            total += self.xij[ind + i] * LN * LGw[i]
        an = total * ((2 * n + 1) / 2)
        return an
    def error_indicator(self, plot=False, printing=False, tol = 1.0):
        self.k_list = []
        self.sigmas = []
        self.errors = []
        for k in range(self.K):
            self.k_list.append(k)
            an = []
            nl = []
            N = self.N_dict[self.xk[k][0]]
            for n in range(N):
                an.append(abs(self.coefficients(n, k)))
                nl.append(n)
            an = np.array(an)
            nl = np.array(nl)

            LG = LinearRegression()
            coeffs = LG.estimate_coef(nl[-5:], np.log(an[-5:]))

            self.sigmas.append(coeffs[1])

            C = np.exp(coeffs[0])
            sigma = np.abs(coeffs[1])
            error = (np.sqrt((C ** 2) / (2 * sigma)) * np.exp(-sigma * (N + 1)))
            if printing:
                print("k: {}   Sigma: {}  error: {}  threshold: {}" .format(k, sigma, error, tol * self.L2norm_solution(k)))
            self.errors.append(error)

        return
    def L2norm_solution(self, k):
        L2Norm = 0.0
        total = 0.0
        N = self.N_dict[self.xk[k][0]]
        LGw = self.Nodes_and_Weights_dict[N][1]
        ind = 0
        for i in range(k):
            ind += self.N_dict[self.xk[i][0]]

        for n in range(N):
            total += np.power(self.xij[ind + n],2) * LGw[n]
        L2Norm += np.power(total,0.5)
        return L2Norm


def splitting(t, T, htol, ptol, dg, printing=False, hL2_lim=0, pL2_lim=0, Kmax=40):
    if printing:
        dg.error_indicator(printing=True, tol=htol)
    else:
        dg.error_indicator(tol=htol)

    p_refinement = []
    for k in range(dg.K):
        L2 = dg.L2norm_solution(k)
        # print("k={} - Error {} | {} Threshold | Sigma {}".format(k, dg.errors[k], tol2 * L2, np.abs(dg.sigmas[k])))
        if dg.errors[k] >= ptol * L2 and np.abs(dg.sigmas[k]) > 1.0 and dg.N_dict[dg.xk[k][0]] <= DG.Nmax and L2 > pL2_lim :
            print("P-REFINEMENT: {}".format(k))
            dg.plot(t, T)
            p_refinement.append(k)

    while len(p_refinement) != 0:
        dg.P_refinement(p_refinement[0])
        p_refinement.pop(0)  # log(n)
        print("Polynomial orders: {}".format(dg.N_dict))

    dg.error_indicator(tol=htol)
    splitting = []
    for k in range(dg.K):
        if dg.K+len(splitting) >= Kmax:
            break
        L2 = dg.L2norm_solution(k)
        if dg.errors[k] >= htol * L2 and np.abs(dg.sigmas[k]) < 1.0 and L2 > hL2_lim:
            print("SPLITTING: {}".format(k))
            splitting.append(k)
    while len(splitting) != 0:
        dg.element_split(splitting[0])
        splitting.pop(0)
        for num, val in enumerate(splitting):
            splitting[num] = val + 1

    dg.error_indicator()

def DGStepByRK3(tn, dt, dg):
    am = np.array([0.0, -5/9, -153/128], dtype='float')
    bm = np.array([0.0, 1/3, 3/4], dtype='float')
    gm = np.array([1/3, 15/16, 8/15], dtype='float')
    xijdt = np.zeros(dg.size, dtype='float')
    Gj = np.zeros_like(xijdt)

    for m in range(0, 3):
        t = tn + bm[m] * dt
        ind = 0
        for k in range(dg.K):
            N = dg.N_dict[dg.xk[k][0]]
            dg.Fluxes[k] = dg.fluxes(dg.xij[ind:ind+N], k) * dg.Ji[k]
            xijdt[ind:ind+N] = dg.DGTimeDerivative(t, k, dg.xij[ind:ind+N])
            ind += N

        for j in range(0, dg.size):
            Gj[j] = am[m] * Gj[j] + xijdt[j]
            dg.xij[j] = dg.xij[j] + gm[m] * dt * Gj[j]
    return dg

def LegendreCollocationIntegrator(Nt, T, dg, split=False, tol=1.0, tol2=1.0, pL2_lim=0, hL2_lim=0, Kmax = 40, Pmax=14):
    dt = T/Nt
    tn = 0.0
    n = 0.0
    # split_flag = True
    dg.Nmax = Pmax
    while tn < T:
        dg = DGStepByRK3(tn, dt, dg)
        #
        # if np.floor(n%(Nt/1028)) and split_flag == True:
        #     # print("Splitting at: {}".format(tn))
        #     for k in range(DG.K):
        #         DG.solution_split(k, pflag=True)
        #     # DG.element_split(0)
        #     dg.plot(tn + dt, T, errors=False)
        #     split_flag = False


        if np.floor(n%4 == 0) and split == True:
            if DG.K <= Kmax:
                splitting(tn+dt, T, tol, tol2, dg, printing=False, pL2_lim=pL2_lim, hL2_lim=hL2_lim, Kmax=40)

        if n%np.floor(Nt/8) == 0:
            print("time: {}".format(np.round(n/Nt,8)))
            # print(dt, " <= ", 2.5 / (len(dg.xk)*dg.N))
            # dg.error_indicator(plot=True, tol=tol)
            dg.plot(tn+dt, T, errors=False)

        if n==Nt-1 and split == True:
            dg.plot(tn+dt, T, errors=False)

        Nt = (T)*5*dg.N*(len(dg.xk))
        dt = T / Nt
        tn = tn + dt
        n += 1

    return dg.xij, dg.xi





if __name__ == "__main__":

    # Importing/ generating a 1D mesh
    from matplotlib import pyplot as plt
    from Linear_regression import LinearRegression
    from Kopriva_Continuous_SEM.MeshGenerator import Mesh
    mesh_obj = Mesh()

    mesh = mesh_obj.mesh_gen(1, -1, -0.5)
    mesh.extend(mesh_obj.mesh_gen(1, -0.5, 0.0))
    mesh.extend(mesh_obj.mesh_gen(1, 0.0, 0.5))
    mesh.extend(mesh_obj.mesh_gen(1, 0.5, 1.0))

    # INITIALISING SOLVER
    N = 8 # Started tests on 12
    K = len(mesh)
    DG = NodalDiscontinuousGalerkin(N, K, mesh)

    from tqdm import tqdm
    T = 0.0
    tol = 0.0005
    tol2 = 0.0005
    pL2_lim = 0.08
    hL2_lim = 0.08

    for i in np.arange(0.8, 0.9, 0.4):
        DG = NodalDiscontinuousGalerkin(N, K, mesh)
        DG.plot(0.0, i)
        xijout, X = LegendreCollocationIntegrator((T+i)*(5*N*K), T+i, DG, split=True, tol=tol, tol2 = tol2, pL2_lim=pL2_lim, hL2_lim=hL2_lim, Kmax=25, Pmax=14)
        print(DG.N_dict)






