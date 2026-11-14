"""Full Panda rigid-body impedance rollout, adapted from Jaquier et al. CoRL19.

Menagerie inertial/kinematic model; free-space only (meshes removed).
The default variable is a fixed-spectrum SO(3) stiffness eigenframe;
stiffness=True accepts a full SPD(3) stiffness matrix. Both tasks are
adaptations of the published experiment.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from ritro._geometry.report_problems import Product
from ritro.manifolds import Stiefel

ASSET = Path(__file__).parent/'assets/panda_original.xml'
Q0 = np.array([0., .3, 0., -1., 0., 1.5, 0.])
TARGET = np.array([.5, -.4, .75])
FORCE = np.array([0., 20., -20.])
SPECTRUM = np.array([100., 300., 900.])

class Panda:
    def __init__(self, dt=.004, horizon=4., mass_scale=1.):
        root=ET.parse(ASSET).getroot()
        if not np.isfinite(mass_scale) or mass_scale<=0:raise ValueError('Positive mass scale required')
        if mass_scale!=1.:
            for inertial in root.findall('.//inertial'):
                for key in ['mass','diaginertia','fullinertia']:
                    if key in inertial.attrib:
                        inertial.set(key,' '.join(str(float(v)*mass_scale) for v in inertial.get(key).split()))
        # All masses and inertias are explicit; visual/collision meshes have no
        # role in this free-space experiment. Preserve the finger constraints.
        for parent in root.iter():
            for node in list(parent):
                if node.tag in ('geom','mesh','keyframe','actuator'):
                    parent.remove(node)
        root.find('option').set('timestep',str(dt))
        hand=root.find(".//body[@name='hand']")
        ET.SubElement(hand,'site',name='tcp',pos='0 0 0.1034')
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.data=mujoco.MjData(self.model)
        self.site=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'tcp')
        self.dt=dt;self.steps=round(horizon/dt);self.calls=0

    def rollout(self,z,force=FORCE,trace=False,stiffness=False,reference=None):
        self.calls+=1
        m,d=self.model,self.data
        mujoco.mj_resetData(m,d);d.qpos[:7]=Q0;d.qpos[7:]=.02
        mujoco.mj_forward(m,d)
        R=np.asarray(z).reshape(3,3)
        # Also defines a smooth ambient extension for Ext-RFD near SO(3).
        if stiffness:
            K=(R+R.T)/2
            eigenvalues,Q=np.linalg.eigh(K)
            if np.min(eigenvalues)<=0:raise ValueError('Stiffness must be positive definite')
            D=2*(Q*np.sqrt(eigenvalues))@Q.T
        else:
            K=(R*SPECTRUM)@R.T;D=(R*(2*np.sqrt(SPECTRUM)))@R.T
        J=np.zeros((3,m.nv));Jr=np.zeros_like(J)
        values=[];positions=[];torques=[];joints=[];limit_violation=0.;clipped=0
        limits=np.array([87.,87.,87.,87.,12.,12.,12.])
        forces=np.asarray(force)
        if forces.shape not in [(3,),(self.steps,3)]:raise ValueError('Force must be constant or a time series')
        if reference is not None:
            desired,desired_velocity=map(np.asarray,reference)
            if desired.shape!=(self.steps,3) or desired_velocity.shape!=(self.steps,3):raise ValueError('Reference shape mismatch')
        for i in range(self.steps):
            mujoco.mj_forward(m,d)
            mujoco.mj_jacSite(m,d,J,Jr,self.site)
            p=d.site_xpos[self.site].copy();v=J@d.qvel
            target=TARGET if reference is None else desired[i]
            target_velocity=np.zeros(3) if reference is None else desired_velocity[i]
            task=J[:,:7].T@(K@(target-p)-D@(v-target_velocity))
            # Fixed joint damping stabilizes redundant motion. Gravity and
            # Coriolis compensation are internal to the feedback controller.
            command=task+d.qfrc_bias[:7]-2*d.qvel[:7]
            tau=np.clip(command,-limits,limits)
            clipped+=int(np.any(tau!=command))
            d.qfrc_applied[:]=J.T@(forces if forces.ndim==1 else forces[i])
            d.qfrc_applied[:7]+=tau
            d.qfrc_applied[7:]+=-100*(d.qpos[7:]-.02)-2*d.qvel[7:]
            error=float(np.sum((target-p)**2));effort=float(tau@tau)
            values.append([error,effort]);positions.append(p);torques.append(tau);joints.append(d.qpos[:7].copy())
            limit_violation=max(limit_violation,float(np.max(np.maximum(m.jnt_range[:7,0]-d.qpos[:7],d.qpos[:7]-m.jnt_range[:7,1]))))
            mujoco.mj_step(m,d)
            if not np.all(np.isfinite(d.qpos)):raise FloatingPointError('Nonfinite robot state')
            if abs(d.time-(i+1)*self.dt)>1e-8:raise FloatingPointError('Simulator time reset')
            if any(w.number for w in d.warning):raise FloatingPointError('MuJoCo warning during rollout')
        values=np.array(values)
        # Smooth part of paper f2, averaged over the entire fixed rollout.
        result=dict(cost=float(np.mean(values[:,0]+1e-5*values[:,1])),
                    rms=float(np.sqrt(np.mean(values[:,0]))),
                    terminal_error=float(np.linalg.norm((TARGET if reference is None else desired[-1])-positions[-1])),
                    mean_squared_torque=float(np.mean(values[:,1])),
                    saturated_fraction=clipped/self.steps,joint_limit_violation=limit_violation)
        if trace:result.update(time=np.arange(self.steps)*self.dt,position=np.array(positions),torque=np.array(torques),q=np.array(joints))
        return result

def problem(name,seed):
    M=Product([Stiefel(3,3)]);rng=np.random.default_rng(seed)
    R=np.linalg.qr(rng.normal(size=(3,3)))[0];R[:,-1]*=np.linalg.det(R)
    sim=Panda()
    def f(z):return sim.rollout(z)['cost']
    def gn(z):return None  # No costly derivative observer in the application pilot.
    return M,R.ravel(),f,gn,dict(x0=R.ravel(),q0=Q0,target=TARGET,force=FORCE,spectrum=SPECTRUM)
