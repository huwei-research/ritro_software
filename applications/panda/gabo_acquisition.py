"""Author AIRM kernel with LogEI over the full SPD matrix-log chart.

Finite random multistart proposals are initialization, not domain constraints.
Acquisition optimization uses unconstrained BFGS; failed restarts are recorded.
This is the separate recovery candidate: failed GP fits restart from the
same state with maxls=50. Existing baseline imports remain unchanged.
"""
import warnings
import copy
from botorch.exceptions.errors import ModelFittingError
import numpy as np
from scipy.optimize import minimize
import gabo_support as gp


CANDIDATES = [{}, {'initial_beta': .6}, {'initial_beta': 2.},
              {'noise_floor': 1e-6}, {'noise_floor': 1e-4}]


class Acquisition:
    def __init__(self, seed, parameters):
        self.rng = np.random.default_rng(seed)
        gp.torch.manual_seed(seed)
        self.parameters = parameters
        self.state = None
        self.records = []
        self.fit_records = []

    def _fit(self, mll):
        initial = copy.deepcopy(mll.state_dict())
        rng_state = gp.torch.get_rng_state().clone()
        attempts = []
        for index, options in enumerate(({'maxiter': 50}, {'maxiter': 50, 'maxls': 50})):
            if index:
                mll.load_state_dict(initial)
                gp.torch.set_rng_state(rng_state)
            row = dict(options=options, status='running')
            attempts.append(row)
            try:
                gp.fit_gpytorch_mll(mll, optimizer_kwargs={'options': options}, max_attempts=1)
                row['status'] = 'ok'
                self.fit_records.append(dict(observations=int(mll.model.train_inputs[0].shape[0]), attempts=attempts))
                return
            except ModelFittingError as exc:
                row.update(status='error', error=str(exc))
                if index:
                    self.fit_records.append(dict(observations=int(mll.model.train_inputs[0].shape[0]), attempts=attempts))
                    raise

    def propose(self, points, values):
        Z = np.array([gp.logs(np.asarray(x).reshape(3,3)) for x in points])
        X = gp.mandel(gp.matrix(gp.torch.tensor(Z)))
        y = np.array(values); scale = max(y.std(),1e-10)
        Y = gp.torch.tensor((y-y.mean())/scale).unsqueeze(-1)
        cov = gp.gpytorch.kernels.ScaleKernel(
            gp.author.SpdAffineInvariantGaussianKernel(beta_min=.5),
            outputscale_prior=gp.gpytorch.priors.GammaPrior(2.,.15))
        if 'initial_beta' in self.parameters:
            cov.base_kernel.beta = self.parameters['initial_beta']
        likelihood = gp.gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=gp.gpytorch.priors.GammaPrior(1.1,.05),
            noise_constraint=gp.gpytorch.constraints.GreaterThan(self.parameters.get('noise_floor',1e-8)))
        model = gp.SingleTaskGP(X,Y,covar_module=cov,likelihood=likelihood,outcome_transform=None)
        if self.state is not None: model.load_state_dict(self.state)
        mll = gp.gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood,model)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            self._fit(mll)
            acq = gp.LogExpectedImprovement(model,best_f=Y.min(),maximize=False)
            # Include global finite proposals and Gaussian proposals with full
            # support around the incumbent in the unconstrained log chart.
            starts = gp.sample(self.rng,50)
            incumbent = Z[int(np.argmin(values))]
            starts = np.vstack([starts,incumbent+self.rng.normal(size=(49,6)),incumbent])
            with gp.torch.no_grad():
                scores = acq(gp.mandel(gp.matrix(gp.torch.tensor(starts))).unsqueeze(-2)).numpy()
            valid = np.flatnonzero(np.isfinite(scores))
            if not len(valid): raise FloatingPointError('No finite acquisition start')
            candidates = [(float(scores[j]),starts[j]) for j in valid[np.argsort(scores[valid])[-5:]]]
            def objective(z):
                v = gp.torch.tensor(z,requires_grad=True)
                value = -acq(gp.mandel(gp.matrix(v)).reshape(1,1,6)).sum()
                grad = gp.torch.autograd.grad(value,v)[0]
                f,g = float(value.detach()),grad.detach().numpy()
                if not np.isfinite(f) or not np.all(np.isfinite(g)):
                    raise FloatingPointError('Nonfinite acquisition value or gradient')
                return f,g
            statuses = []
            for _,z in list(candidates):
                try:
                    result = minimize(objective,z,jac=True,method='BFGS',options={'maxiter':100,'gtol':1e-6})
                    statuses.append(dict(success=bool(result.success),message=str(result.message)))
                    if np.isfinite(result.fun): candidates.append((-float(result.fun),result.x))
                except (RuntimeError,ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    statuses.append(dict(success=False,message=str(exc)))
            z = max(candidates,key=lambda item:item[0])[1]
            point = gp.matrix(gp.torch.tensor(z)).detach().numpy()
            if not np.all(np.isfinite(point)) or np.min(np.linalg.eigvalsh(point))<=0:
                raise FloatingPointError('Invalid SPD acquisition point')
            self.records.append(dict(beta=float(cov.base_kernel.beta.detach().squeeze()),
                noise=float(likelihood.noise.detach().squeeze()),acquisition_statuses=statuses,
                warnings=[str(w.message) for w in ws],eigenvalues=np.linalg.eigvalsh(point).tolist()))
        self.state = model.state_dict()
        return point.ravel()
