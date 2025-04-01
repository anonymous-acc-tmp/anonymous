import torch as th

def compute_ess(w, dim=-1):
    ess = (w.sum(dim=dim))**2 / th.sum(w**2, dim=dim)
    return ess

def compute_ess_from_log_w(log_w, dim=-1):
    return compute_ess(normalize_weights(log_w, dim=dim), dim=dim)

def normalize_weights(log_weights, dim=-1):
    return th.exp(normalize_log_weights(log_weights, dim=dim))

def normalize_log_weights(log_weights, dim):
    log_weights = log_weights - log_weights.max(dim=dim, keepdims=True)[0]
    log_weights = log_weights - th.logsumexp(log_weights, dim=dim, keepdims=True)
    return log_weights


import numpy as np
from numba import jit
from numpy import random


@jit(nopython=True) 
def inverse_cdf(su, W):
    """Inverse CDF algorithm for a finite distribution.
        Parameters
        ----------
        su: (M,) ndarray
            M sorted uniform variates (i.e. M ordered points in [0,1]).
        W: (N,) ndarray
            a vector of N normalized weights (>=0 and sum to one)
        Returns
        -------
        A: (M,) ndarray
            a vector of M indices in range 0, ..., N-1
    """
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, dtype=np.int64)
    for n in range(M):
        while su[n] > s:
            if j == M-1:
                break  # avoiding numerical issue 
            j += 1
            s += W[j]
        A[n] = j
    return A

def uniform_spacings(N):
    """Generate ordered uniform variates in O(N) time.
    Parameters
    ----------
    N: int (>0)
        the expected number of uniform variates
    Returns
    -------
    (N,) float ndarray
        the N ordered variates (ascending order)
    Note
    ----
    This is equivalent to::
        from numpy import random
        u = sort(random.rand(N))
    but the line above has complexity O(N*log(N)), whereas the algorithm
    used here has complexity O(N).
    """
    z = np.cumsum(-np.log(random.rand(N + 1)))
    return z[:-1] / z[-1]


def multinomial(W, M):
    """Multinomial resampling.
    Popular resampling scheme, which amounts to sample N independently from
    the multinomial distribution that generates n with probability W^n.
    This resampling scheme is *not* recommended for various reasons; basically
    schemes like stratified / systematic / SSP tends to introduce less noise,
    and may be faster too (in particular systematic).
    """
    return inverse_cdf(uniform_spacings(M), W)


def stratified(W, M):
    """Stratified resampling.
    """
    su = (random.rand(M) + np.arange(M)) / M
    return inverse_cdf(su, W)


def systematic(W, M):
    """Systematic resampling.
    """
    su = (random.rand(1) + np.arange(M)) / M
    return inverse_cdf(su, W)


def residual(W, M):
    """Residual resampling.
    """
    N = W.shape[0]
    A = np.empty(M, dtype=np.int64)
    MW = M * W
    intpart = np.floor(MW).astype(np.int64)
    sip = np.sum(intpart)
    res = MW - intpart
    sres = M - sip
    A[:sip] = np.arange(N).repeat(intpart)
    # each particle n is repeated intpart[n] times
    if sres > 0:
        A[sip:] = multinomial(res / sres, M=sres)
    return A

@jit(nopython=True)
def ssp(W, M):
    """SSP resampling.

    SSP stands for Srinivasan Sampling Process. This resampling scheme is
    discussed in Gerber et al (2019). Basically, it has similar properties as
    systematic resampling (number of off-springs is either k or k + 1, with
    k <= N W^n < k +1), and in addition is consistent. See that paper for more
    details.

    Reference
    =========
    Gerber M., Chopin N. and Whiteley N. (2019). Negative association, ordering
    and convergence of resampling methods. Ann. Statist. 47 (2019), no. 4, 2236–2260.
    """
    N = W.shape[0]
    MW = M * W
    nr_children = np.floor(MW).astype(np.int64)
    xi = MW - nr_children
    u = random.rand(N - 1)
    i, j = 0, 1
    for k in range(N - 1):
        delta_i = min(xi[j], 1.0 - xi[i])  # increase i, decr j
        delta_j = min(xi[i], 1.0 - xi[j])  # the opposite
        sum_delta = delta_i + delta_j
        # prob we increase xi[i], decrease xi[j]
        pj = delta_i / sum_delta if sum_delta > 0.0 else 0.0
        # sum_delta = 0. => xi[i] = xi[j] = 0.
        if u[k] < pj:  # swap i, j, so that we always inc i
            j, i = i, j
            delta_i = delta_j
        if xi[j] < 1.0 - xi[i]:
            xi[i] += delta_i
            j = k + 2
        else:
            xi[j] -= delta_i
            nr_children[i] += 1
            i = k + 2
    # due to round-off error accumulation, we may be missing one particle
    if np.sum(nr_children) == M - 1:
        last_ij = i if j == k + 2 else j
        if xi[last_ij] > 0.99:
            nr_children[last_ij] += 1
    if np.sum(nr_children) != M:
        # file a bug report with the vector of weights that causes this
        raise ValueError("ssp resampling: wrong size for output")
    return np.arange(N).repeat(nr_children)


Resample_dict = dict(systematic=systematic,
                     stratified=stratified,
                     residual=residual,
                     multinomial=multinomial,
                     ssp=ssp)

def resampling_function(resample_strategy="systematic", ess_threshold=None, verbose=False):
    """resampling_function returns a resampling function that may be used in
    smc_FK
    ess_threshold: in [0, 1]
    """
    resample_fn = Resample_dict[resample_strategy]

    def resample(log_w): # log_w.shape = (batch_size, num_particles)
        assert log_w.dim() == 2, "Dimension of log_w should be 2"

        log_normalized_weights = normalize_log_weights(log_w, dim=-1)
        normalized_weights = th.exp(log_normalized_weights)
        
        P = log_w.shape[-1]
        ess = compute_ess(normalized_weights, dim=-1)

        resample_indices = th.zeros_like(log_w, device=log_w.device, dtype=th.int)
        is_resampled = th.zeros(log_w.shape[0], device=log_w.device, dtype=th.bool)
        
        for i, ess_batch in enumerate(ess):
            if ess_threshold is None or ess_batch < P*ess_threshold:
                # Resampling
                if verbose: print("resample")
                resample_indices[i] = th.from_numpy(resample_fn(W=np.array(normalized_weights[i].cpu()), M=P))
                is_resampled[i] = True
                log_w[i] = - th.log(th.Tensor([P]).to(log_w.device))
            else:
                # No Resampling
                resample_indices[i] = th.arange(P)
                is_resampled[i] = False
                log_w[i] = log_normalized_weights[i]

        return resample_indices, is_resampled, log_w

    return resample


from scipy.optimize import minimize

def adaptive_tempering(log_w, log_prob_diffusion, log_twist_func, log_prob_proposal, log_twist_func_prev, min_scale, ess_threshold=1., num_iterations=500): # log potential in our setting is -r(x)/\alpha; beware of the minus sign
    P = log_twist_func.shape[-1]
    scale_factor = th.zeros_like(min_scale, device=min_scale.device, dtype=th.float)

    def _ess(scale, i):
        tmp_log_w = log_w[i] + log_prob_diffusion[i] + scale*log_twist_func[i] - log_prob_proposal[i] - log_twist_func_prev[i]
        W = normalize_weights(tmp_log_w, dim=-1)
        ess = compute_ess(W)
        return ess

    def _esslw(scale, i):
        ess = _ess(scale, i)
        return (ess - ess_threshold*P)**2



    for i in range(log_twist_func.shape[0]):
        scale = th.tensor([min_scale[i]], requires_grad=True, device=log_twist_func.device) 
        # scale = minimize(_esslw, 0.01, args=(i), bounds=[(min_scale[i], 1.)], tol=1e-6).x[0]
        optimizer = th.optim.Adam([scale], lr=1e-4)
        # Define the closure function required by optimization
        def closure():
            optimizer.zero_grad() 
            loss = _esslw(scale, i)
            loss.backward()
            return loss
        for _ in range(num_iterations):
            prev_scale = scale.clone()
            optimizer.step(closure)
            scale.data = scale.clamp(min_scale[i], 1.)
            if th.abs(scale - prev_scale) < 1e-6:
                break

        scale_factor[i] = scale
        # print(_ess(0, i))
        # print(_ess(0.01, i))
        # print(_ess(0.1, i))
        # print(_ess(scale, i))
        # print(_ess(1, i))

    return scale_factor