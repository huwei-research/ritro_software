"""Declared MuJoCo reconstruction of Jaquier et al. Table 4 (f1)."""
import numpy as np
from panda_impedance import Panda

TARGET = np.array([.66, -.01, .69])
FORCE = np.array([0., 20., -20.])
CONFIG = dict(horizon=4., dt=.002, target=TARGET.tolist(), force=FORCE.tolist(),
              weights=dict(position=1., determinant=1e-13, condition=1e-4),
              aggregation='time mean over [0,4); constant reference and force',
              late_interval=[3.,4.], domain='full SPD(3), no clipping',
              position_threshold_m=.02)


class Objective:
    def __init__(self, dt=.002):
        self.sim = Panda(dt=dt, horizon=4.)
        self.records = []
        self.reference = (np.tile(TARGET, (self.sim.steps, 1)),
                          np.zeros((self.sim.steps, 3)))

    def evaluate(self, x, trace=False):
        K = np.asarray(x).reshape(3,3)
        np.testing.assert_allclose(K, K.T, rtol=1e-10, atol=1e-9)
        w = np.linalg.eigvalsh(K)
        if not np.all(np.isfinite(w)) or w.min() <= 0:
            raise ValueError('A finite SPD matrix is required')
        r = self.sim.rollout(K, FORCE, stiffness=True, trace=True, reference=self.reference)
        e = np.linalg.norm(r['position']-TARGET, axis=1)
        det_term = 1e-13*float(np.prod(w))
        cond_term = 1e-4*float(w[-1]/w[0])
        position_term = float(np.mean(e**2))
        value = position_term+det_term+cond_term
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite objective')
        late = e[r['time']>=3.]
        metrics = dict(value=value, position_term=position_term, determinant_term=det_term,
                       condition_term=cond_term, rms_m=float(np.sqrt(position_term)),
                       late_rms_m=float(np.sqrt(np.mean(late**2))),
                       terminal_error_m=float(e[-1]), peak_error_m=float(e.max()),
                       torque_integral_nm2s=float(self.sim.dt*np.sum(r['torque']**2)),
                       saturation=r['saturated_fraction'], joint_limit_violation=r['joint_limit_violation'],
                       eigenvalues=w.tolist(), rollouts=1)
        metrics['task_success'] = bool(metrics['late_rms_m']<=.02 and metrics['joint_limit_violation']<=1e-6)
        self.records.append(metrics)
        return (value, metrics, r) if trace else value

    def __call__(self, x):
        return self.evaluate(x)
