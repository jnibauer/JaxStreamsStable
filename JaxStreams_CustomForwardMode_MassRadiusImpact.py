from functools import partial
from astropy.constants import G
import astropy.coordinates as coord
import astropy.units as u
# gala
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import dimensionless, galactic, UnitSystem

import jax
import jax.numpy as jnp

from jax.config import config
config.update("jax_enable_x64", True)
import jax.random as random 
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from diffrax import diffeqsolve, ODETerm, Dopri5,SaveAt,PIDController,DiscreteTerminatingEvent, DirectAdjoint, RecursiveCheckpointAdjoint, ConstantStepSize, Euler, StepTo
import diffrax
import equinox as eqx
usys = UnitSystem(u.kpc, u.Myr, u.Msun, u.radian)
## ***Updated subhalo prescription included***


## Note, I was usit tsit5 for regular orbitintegrator run and then Heun for prestep calculation. Did this at flatiron
## Then i switched to leapfrog midpoint and found exact converngence autograd deriv and my estimate
class Potential:
    def __init__(self, units, params):
        if units is None:
            units = dimensionless
        self.units = UnitSystem(units)
        
        if self.units == dimensionless:
            self._G = 1
        else:
            self._G = G.decompose(self.units).value
        
        for name, param in params.items():
            if hasattr(param, 'unit'):
                param = param.decompose(self.units).value
            setattr(self, name, param)
    
    @partial(jax.jit, static_argnums=(0,))
    def gradient(self, xyz, t):
        grad_func = jax.grad(self.potential)
        return grad_func(xyz, t)
    
    @partial(jax.jit, static_argnums=(0,))
    def density(self, xyz, t):
        lap = jnp.trace(jax.hessian(self.potential)(xyz, t))
        return lap / (4 * jnp.pi * self._G)
    
    @partial(jax.jit, static_argnums=(0,))
    def acceleration(self, xyz, t):
        return -self.gradient(xyz, t)
    
    @partial(jax.jit, static_argnums=(0,))
    def local_circular_velocity(self,xyz,t):
        r = jnp.sqrt(jnp.sum(xyz**2))
        r_hat = xyz / r
        grad_phi = self.gradient(xyz,t)
        dphi_dr = jnp.sum(grad_phi*r_hat)
        return jnp.sqrt( r*dphi_dr )
   
    @partial(jax.jit,static_argnums=(0,))
    def jacobian_force_mw(self, xyz, t):
        jacobian_force_mw = jax.jacfwd(self.gradient)
        return jacobian_force_mw(xyz, t)
    
    @partial(jax.jit,static_argnums=(0,))
    def d2phidr2_mw(self, x, t):
        """
        Computes the second derivative of the Milky Way potential at a position x (in the simulation frame)
        Args:
          x: 3d position (x, y, z) in [kpc]
        Returns:
          Second derivative of force (per unit mass) in [1/Myr^2]
        Examples
        --------
        >>> d2phidr2_mw(x=jnp.array([8.0, 0.0, 0.0]))
        """
        rad = jnp.linalg.norm(x)
        r_hat = x/rad
        dphi_dr_func = lambda x: jnp.sum(self.gradient(x,t)*r_hat)
        return jnp.sum(jax.grad(dphi_dr_func)(x)*r_hat)
        
        ##return jnp.matmul(jnp.transpose(x), jnp.matmul(self.jacobian_force_mw(x, t), x)) / rad**2


    @partial(jax.jit,static_argnums=(0,))
    def omega(self, x,v):
        """
        Computes the magnitude of the angular momentum in the simulation frame
        Args:
          x: 3d position (x, y, z) in [kpc]
          v: 3d velocity (v_x, v_y, v_z) in [kpc/Myr]
        Returns:
          Magnitude of angular momentum in [rad/Myr]
        Examples
        --------
        >>> omega(x=jnp.array([8.0, 0.0, 0.0]), v=jnp.array([8.0, 0.0, 0.0]))
        """
        rad = jnp.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
        omega_vec = jnp.cross(x, v) / (rad**2)
        return jnp.linalg.norm(omega_vec)

    @partial(jax.jit,static_argnums=(0,))
    def tidalr_mw(self, x, v, Msat, t):
        """
        Computes the tidal radius of a cluster in the potential
        Args:
          x: 3d position (x, y, z) in [kpc]
          v: 3d velocity (v_x, v_y, v_z) in [kpc/Myr]
          Msat: Cluster mass in [Msol]
        Returns:
          Tidal radius of the cluster in [kpc]
        Examples
        --------
        >>> tidalr_mw(x=jnp.array([8.0, 0.0, 0.0]), v=jnp.array([8.0, 0.0, 0.0]), Msat=1e4)
        """
        return (self._G * Msat / ( self.omega(x, v) ** 2 - self.d2phidr2_mw(x, t)) ) ** (1.0 / 3.0)
    
    @partial(jax.jit,static_argnums=(0,))
    def lagrange_pts(self,x,v,Msat, t):
        r_tidal = self.tidalr_mw(x,v,Msat, t)
        r_hat = x/jnp.linalg.norm(x)
        L_close = x - r_hat*r_tidal
        L_far = x + r_hat*r_tidal
        return L_close, L_far  
    
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration(self,t,xv,args):
        x, v = xv[:3], xv[3:]
        acceleration = -self.gradient(x,t)
        return jnp.hstack([v,acceleration])
    
    @partial(jax.jit,static_argnums=((0,5)))
    def orbit_integrator_run_dense(self,w0,t0,t1,ts, steps):
        term = ODETerm(self.velocity_acceleration)
        solver = diffrax.Dopri8(scan_kind='bounded')

        saveat = SaveAt(t0=False, t1=False,ts=None,dense=True,steps=False)##jax.lax.switch(0, branches, None)
        
        #if steps == True:
        #    saveat = SaveAt(t0=False, t1=False,ts=None,dense=True,steps=False) # ts=ts,
        #if steps == False:
        #    saveat = SaveAt(t0=False, t1=True,ts=ts,dense=False) # ts=ts,
        rtol: float = 1e-7 #1e-7
        atol: float = 1e-7 #1e-7
        def true_func(dt):
            return 0.5#0.05
        def false_func(dt):
            return -0.5#-0.05
        dt0= None#jax.lax.cond(t0<t1,true_func,false_func,None)
        
        #if t0<t1:
        #    dt0=0.5
        #if t0>t1:
        #    dt0=-0.5
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=0.05,dtmax=None,force_dtmin=True,)#ConstantStepSize()#PIDController(rtol=rtol, atol=atol, dtmin=0.005,force_dtmin=True)#ConstantStepSize()# ## #dtmin 0.01
        max_steps: int = 800#1_500#1_000#2_000
        t0 = t0#0.0
        t1 = t1#4000.
        dense = True# WAS TRUE MADE FALSE.. NEED TRUE FOR GRADIENTS!!!! MAXSTEPS!
        #y0= w_init

        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=None,
            max_steps=(max_steps if dense else None),
            adjoint=DirectAdjoint()
        )
        #if steps == True:
        #    return solution
        #if steps == False:
        #    return solution.ys
        return solution
    
    
    @partial(jax.jit,static_argnums=((0,5)))
    def orbit_integrator_run_notdense(self,w0,t0,t1,ts, steps):
        term = ODETerm(self.velocity_acceleration)
        solver = diffrax.Dopri8()#(scan_kind="bounded") #Dopri5, Dopri8
        saveat = SaveAt(t0=False, t1=True,ts=ts,dense=False)
        
        #if steps == True:
        #    saveat = SaveAt(t0=False, t1=False,ts=None,dense=True,steps=False) # ts=ts,
        #if steps == False:
        #    saveat = SaveAt(t0=False, t1=True,ts=ts,dense=False) # ts=ts,
        rtol: float = 1e-6#1e-6  
        atol: float = 1e-6  #.5e-6  
        def true_func(dt):
            return .5#.5
        def false_func(dt):
            return -.5#-.5
        dt0= jax.lax.cond(t0<t1,true_func,false_func,None) #None
        
       
        stepsize_controller = ConstantStepSize()#PIDController(rtol=rtol, atol=atol, dtmin=.5,force_dtmin=True,
                              #              )# dtmin = 0.05#ConstantStepSize()#PIDController(rtol=rtol, atol=atol, dtmin=0.05,force_dtmin=True)#ConstantStepSize()#PIDController(rtol=rtol, atol=atol, dtmin=0.005,force_dtmin=True)#ConstantStepSize()# ## #dtmin 0.01
        max_steps: int = 10_000#16**4#8100#4000
        t0 = t0#0.0
        t1 = t1#4000.
        dense = True# WAS TRUE MADE FALSE.. NEED TRUE FOR GRADIENTS!!!! MAXSTEPS!
        #y0= w_init

        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=None,
            max_steps=(max_steps if dense else None),
            adjoint=DirectAdjoint()
        )
        #if steps == True:
        #    return solution
        #if steps == False:
        #    return solution.ys
        return solution.ys

    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_generic(self,t,xv,args):
        x0, v0 = xv[:3], xv[3:6]
        x1, v1 = xv[6:9], xv[9:12]

        acceleration0 = -self.gradientPotentialBase(x0,t)
        acceleration1 = -self.gradientPotentialPerturbation(x0,t)
        d2H_dq2 = -jax.jacrev(self.gradientPotentialBase)(x0,t)

        d_qdot_d_eps = v1 
        d_pdot_d_eps = acceleration1 + jnp.matmul(d2H_dq2,x1)
        return jnp.hstack([v0,acceleration0, d_qdot_d_eps, d_pdot_d_eps])

    @partial(jax.jit,static_argnums=((0,)))
    def velocity_acceleration_perturbation_savedstate(self,t,xv,args):
        statefunc = args
        base_loc = statefunc(t)
        x0, v0 = base_loc[:3], base_loc[3:]
        x1, v1 = xv[:3], xv[3:]
    

        acceleration1 = -self.gradientPotentialPerturbation(x0,t)
        d2H_dq2 = -jax.jacrev(self.gradientPotentialBase)(x0,t)

        d_qdot_d_eps = v1 
        d_pdot_d_eps = acceleration1 + jnp.matmul(d2H_dq2,x1)
        return jnp.hstack([d_qdot_d_eps, d_pdot_d_eps])

    @partial(jax.jit,static_argnums=((0,)))
    def orbit_integrator_run_and_deriv_dense(self,w0,t0,t1, state):
        term = ODETerm(self.velocity_acceleration_perturbation_savedstate)
        solver = diffrax.Heun(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=False, ts=None, dense=True,steps=False)
        rtol: float = 1e-12 #1e-7 or 1e-11 or 1e-13
        atol: float = 1e-13 #1e-11
        dt0=None
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=.05,dtmax=10.,force_dtmin=True) # dtmin = 0.75, dtmax=100
        max_steps: int = 150_000#50_000#16**4#16**4 #was 16**5
        t0 = t0
        t1 = t1
        dense = False
        terminating_event =  None#DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=state,
            max_steps=max_steps,
            adjoint=DirectAdjoint()
        )
       
        return solution

    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_from_noPreComp_mass_radius(self,t,xv,args):
        """
        use when integrating particle spray model, or if you have distinct leading trailing states!
        """
        # Unperturbed, base flow
        x0, v0 = xv[0][:3], xv[0][3:6] #length 6 array
        # Epsilon Derivs
        x1, v1 = xv[1][:,:3], xv[1][:,3:6] #nSHS x 3

        # Structural derivs
        dx1_dtheta, dv1_dtheta = xv[1][:,6:9], xv[1][:,9:]

        acceleration0 = -self.gradientPotentialBase(x0,t)
        
        # acceleration due to perturbation
        acceleration1 = -self.gradientPotentialPerturbation_per_SH(x0,t) # nSH x 3
        # tidal tensor
        d2H_dq2 = -jax.jacrev(self.gradientPotentialBase)(x0,t) 
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # this is dpdot/deps. Injected acceleration + matmul(tidal tensor, q)
        d_pdot_d_eps = acceleration1 + jnp.einsum('ij,kj->ki',d2H_dq2,x1,optimize='optimal') #nSH x 3 #jnp.matmul(d2H_dq2,x1) #jnp.einsum('ijk,ij->ik',d2H_dq2,x1,precision=jax.lax.Precision('highest'))   #n_star x 3
        
        # Now handle radius deviations
        acceleration1_r = -self.gradientPotentialStructural_per_SH(x0,t) # nSH x 3
        d_qalpha1dot_dtheta = dv1_dtheta
        d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ij,kj->ki',d2H_dq2,dx1_dtheta,optimize='optimal')#jnp.matmul(d2H_dq2,dx1_dtheta)
        
        # Package the output
        return [jnp.hstack([v0,acceleration0]), 
                jnp.hstack([d_qdot_d_eps,d_pdot_d_eps, d_qalpha1dot_dtheta, d_palpha1dot_dtheta])]
        #jnp.hstack([v0,acceleration0,d_qdot_d_eps,d_pdot_d_eps,d_qalpha1dot_dtheta,d_palpha1dot_dtheta])#[jnp.hstack([v0,acceleration0]),
               # jnp.hstack([d_qdot_d_eps,d_pdot_d_eps]),
                #jnp.hstack([d_qalpha1dot_dtheta, d_palpha1dot_dtheta])
                # ]
    

    @partial(jax.jit,static_argnums=((0,5)))
    def orbit_integrator_run_and_deriv_noPreComp_mass_radius(self,w0,t0,t1, ts, dense_arg):
        term = ODETerm(self.velocity_acceleration_perturbation_from_noPreComp_mass_radius)
        solver = diffrax.Dopri8(scan_kind="bounded")#diffrax.Dopri5(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=ts, dense=dense_arg,steps=False)
        rtol: float = 1e-9#1e-10
        atol: float = 1e-9#1e-10 
        dt0=None
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=.05,dtmax=None,force_dtmin=True) # dtmin = 0.05, dtmax=100
        max_steps: int = 4_000#4_000#50_000#16**4#16**4 #was 16**5
        t0 = t0
        t1 = t1
        dense = False
        terminating_event =  None#DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=None,
            max_steps=max_steps,
            adjoint=DirectAdjoint()
        )
       
        return solution

    
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_from_PreComp_mass_radius_SingleStar(self,t,xv,args):
        """
       
        """
        ind, stream_func = args
        w0_at_t = interp_func(t,ind,stream_func)
        
        
        #rr, narr = eqx.partition(spl, eqx.is_array)
        #arr = jax.tree_util.tree_map(lambda x: x[particle_ind], arr)
        #interp = eqx.combine(arr, narr)
        #w0_at_t = interp.evaluate(t)

        x0, v0 = w0_at_t[:3], w0_at_t[3:] 
        
        # Epsilon Derivs
        x1, v1 = xv[:,:3], xv[:,3:6] #nSHS x 3

        # Structural derivs
        dx1_dtheta, dv1_dtheta = xv[:,6:9], xv[:,9:]

        #######acceleration0 = -self.gradientPotentialBase(x0,t)
        
        # acceleration due to perturbation
        acceleration1 = -self.gradientPotentialPerturbation_per_SH(x0,t) # nSH x 3
        # tidal tensor
        d2H_dq2 = -jax.jacrev(self.gradientPotentialBase)(x0,t) 
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # this is dpdot/deps. Injected acceleration + matmul(tidal tensor, q)
        d_pdot_d_eps = acceleration1 + jnp.einsum('ij,kj->ki',d2H_dq2,x1,optimize='optimal') #nSH x 3 #jnp.matmul(d2H_dq2,x1) #jnp.einsum('ijk,ij->ik',d2H_dq2,x1,precision=jax.lax.Precision('highest'))   #n_star x 3
        
        # Now handle radius deviations
        acceleration1_r = -self.gradientPotentialStructural_per_SH(x0,t) # nSH x 3
        d_qalpha1dot_dtheta = dv1_dtheta
        d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ij,kj->ki',d2H_dq2,dx1_dtheta, optimize='optimal')#jnp.matmul(d2H_dq2,dx1_dtheta)
        
        # Package the output
        return jnp.hstack([d_qdot_d_eps,d_pdot_d_eps, d_qalpha1dot_dtheta, d_palpha1dot_dtheta])
        #jnp.hstack([v0,acceleration0,d_qdot_d_eps,d_pdot_d_eps,d_qalpha1dot_dtheta,d_palpha1dot_dtheta])#[jnp.hstack([v0,acceleration0]),
               # jnp.hstack([d_qdot_d_eps,d_pdot_d_eps]),
                #jnp.hstack([d_qalpha1dot_dtheta, d_palpha1dot_dtheta])
                
                # ]
    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_StreamSpline_PID_mass_radius_SingleStar(self,w0,t0,t1, ts, state):
        #def event_fun(state, **kwargs):
        #    return state.tnext > 0.0#>=0.02
        #spl, ind = state
        #arr, narr = eqx.partition(spl, eqx.is_array)
        #arr = jax.tree_util.tree_map(lambda x: x[ind], arr)
        #interp = eqx.combine(arr, narr)

        #new_state = (interp,ind)


        term = ODETerm(self.velocity_acceleration_perturbation_from_PreComp_mass_radius_SingleStar)
        solver = diffrax.Dopri5(scan_kind="bounded")#Heun(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=ts, dense=False,steps=False)###CHANGE DENSE TO FALSE AGAIN!
        rtol: float = 1e-15#1e-14 | 1e-121e-13#1e-11and dtmin 0.1
        atol: float = 1e-15#1e-14 | 1e-121e-13#1e-11 and dtmin 0.1
        dt0=None
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=.2,dtmax=None,force_dtmin=True)#.6 for timer runs
        max_steps: int = 5_000#50_000#10_000#80_000
        t0 = t0
        t1 = t1
        dense = False
        terminating_event =  None#DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=state,
            max_steps=max_steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint()#DirectAdjoint()
        )
       
        return solution

    ### INTEGRATE LEAD / TRAIL TOGETHER!!  #####
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_from_PreComp_mass_radius_LeadTrail_Single(self,t,xv,args):
        """
       
        """
        ind, stream_func_lead, stream_func_trail = args

        w0_at_t_lead = interp_func(t,ind,stream_func_lead)
        w0_at_t_trail = interp_func(t,ind,stream_func_trail)
        
        
        #rr, narr = eqx.partition(spl, eqx.is_array)
        #arr = jax.tree_util.tree_map(lambda x: x[particle_ind], arr)
        #interp = eqx.combine(arr, narr)
        #w0_at_t = interp.evaluate(t)

        x0, v0 = jnp.vstack([w0_at_t_lead[:3],w0_at_t_trail[:3]]), jnp.vstack([w0_at_t_lead[3:],w0_at_t_trail[3:]])

        #x0, v0 = w0_at_t[:3], w0_at_t[3:] 
        
        # Epsilon Derivs
        x1, v1 = xv[:,:,:3], xv[:,:,3:6] # 2 x nSHS x 3

        # Structural derivs
        dx1_dtheta, dv1_dtheta = xv[:,:,6:9], xv[:,:,9:] # 2 x nSHS x 3

        acceleration0 = -jax.vmap(self.gradientPotentialBase,in_axes=((0,None)))(x0,t)
        
        # acceleration due to perturbation
        acceleration1 = -jax.vmap(self.gradientPotentialPerturbation_per_SH,in_axes=((0,None)))(x0,t) # 2 x nSH x 3
        # tidal tensor
        d2H_dq2 = -jax.vmap(jax.jacrev(self.gradientPotentialBase),in_axes=((0,None)))(x0,t) 
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # this is dpdot/deps. Injected acceleration + matmul(tidal tensor, q)
        d_pdot_d_eps = acceleration1 + jnp.einsum('ijk,ilk->ilj',d2H_dq2,x1,optimize='optimal')
        ##d_pdot_d_eps = acceleration1 + jnp.einsum('ij,kj->ki',d2H_dq2,x1,optimize='optimal') #nSH x 3 #jnp.matmul(d2H_dq2,x1) #jnp.einsum('ijk,ij->ik',d2H_dq2,x1,precision=jax.lax.Precision('highest'))   #n_star x 3
        
        # Now handle radius deviations
        acceleration1_r = -jax.vmap(self.gradientPotentialStructural_per_SH,in_axes=((0,None)))(x0,t) # nSH x 3
        d_qalpha1dot_dtheta = dv1_dtheta
        #d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ij,kj->ki',d2H_dq2,dx1_dtheta, optimize='optimal')#jnp.matmul(d2H_dq2,dx1_dtheta)
        d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ijk,ilk->ilj',d2H_dq2,dx1_dtheta,optimize='optimal')#jnp.einsum('ijk,ik->ij',d2H_dq2,dx1_dtheta,precision=jax.lax.Precision('highest'))

        
        # Package the output
        return jnp.dstack([d_qdot_d_eps,d_pdot_d_eps, d_qalpha1dot_dtheta, d_palpha1dot_dtheta])# n_star x nSH x 12 #jnp.hstack([d_qdot_d_eps,d_pdot_d_eps, d_qalpha1dot_dtheta, d_palpha1dot_dtheta])
        #jnp.hstack([v0,acceleration0,d_qdot_d_eps,d_pdot_d_eps,d_qalpha1dot_dtheta,d_palpha1dot_dtheta])#[jnp.hstack([v0,acceleration0]),
               # jnp.hstack([d_qdot_d_eps,d_pdot_d_eps]),
                #jnp.hstack([d_qalpha1dot_dtheta, d_palpha1dot_dtheta])
                
                # ]
    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_StreamSpline_PID_mass_radius_LeadTrail_Single(self,w0,t0,t1, ts, state):
        #def event_fun(state, **kwargs):
        #    return state.tnext > 0.0#>=0.02
        #spl, ind = state
        #arr, narr = eqx.partition(spl, eqx.is_array)
        #arr = jax.tree_util.tree_map(lambda x: x[ind], arr)
        #interp = eqx.combine(arr, narr)

        #new_state = (interp,ind)


        term = ODETerm(self.velocity_acceleration_perturbation_from_PreComp_mass_radius_LeadTrail_Single)
        solver = diffrax.Dopri5(scan_kind="bounded")#Heun(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=ts, dense=False,steps=False)
        rtol: float = 1e-12# 
        atol: float = 1e-12# 
        dt0=None
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=.6,dtmax=None,force_dtmin=True)#.25
        max_steps: int = 5_000#50_000#10_000#80_000
        t0 = t0
        t1 = t1
        dense = False
        terminating_event =  None#DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=state,
            max_steps=max_steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint()#DirectAdjoint()
        )
       
        return solution


    #######
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_from_PreComp_mass_radius_impact_velocity_Single(self,t,xv,args):
        """
        Vary mass, radius, impact parameter, and impact velocity
        """
        ind, stream_func = args

        w0_at_t = interp_func(t,ind,stream_func)
        
        
        #rr, narr = eqx.partition(spl, eqx.is_array)
        #arr = jax.tree_util.tree_map(lambda x: x[particle_ind], arr)
        #interp = eqx.combine(arr, narr)
        #w0_at_t = interp.evaluate(t)

        x0, v0 = w0_at_t[:3], w0_at_t[3:] #jnp.vstack([w0_at_t_lead[:3],w0_at_t_trail[:3]]), jnp.vstack([w0_at_t_lead[3:],w0_at_t_trail[3:]])

        #x0, v0 = w0_at_t[:3], w0_at_t[3:] 
        
        # Epsilon Derivs
        x1, v1 = xv[0][:,:3], xv[0][:,3:6] # nSHS x 3

        # Structural derivs
        dx1_dtheta, dv1_dtheta = xv[1][:,:3], xv[1][:,3:6] # nSHS x 3

        # Impact derivs
        dx1_db, dv1_db = xv[2][:,:3], xv[2][:,3:6] # nSHS x 3

        # Velocity derivs
        dx1_dv, dv1_dv = xv[3][:,:3], xv[3][:,3:6] # nSHS x 3


        #######acceleration0 = -jax.vmap(self.gradientPotentialBase,in_axes=((0,None)))(x0,t)
        
        # acceleration due to perturbation
        acceleration1 = -self.gradientPotentialPerturbation_per_SH(x0,t) # nSH x 3
        # tidal tensor
        d2H_dq2 = -jax.jacrev(self.gradientPotentialBase)(x0,t) 
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # this is dpdot/deps. Injected acceleration + matmul(tidal tensor, q)
        d_pdot_d_eps = acceleration1 + jnp.einsum('ij,kj->ki',d2H_dq2,x1,optimize='optimal')
        ##d_pdot_d_eps = acceleration1 + jnp.einsum('ij,kj->ki',d2H_dq2,x1,optimize='optimal') #nSH x 3 #jnp.matmul(d2H_dq2,x1) #jnp.einsum('ijk,ij->ik',d2H_dq2,x1,precision=jax.lax.Precision('highest'))   #n_star x 3
        
        # Now handle radius deviations
        acceleration1_r = -self.gradientPotentialStructural_per_SH(x0,t) # nSH x 3
        d_qalpha1dot_dtheta = dv1_dtheta
        #d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ij,kj->ki',d2H_dq2,dx1_dtheta, optimize='optimal')#jnp.matmul(d2H_dq2,dx1_dtheta)
        d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ij,kj->ki',d2H_dq2,dx1_dtheta,optimize='optimal')#jnp.einsum('ijk,ik->ij',d2H_dq2,dx1_dtheta,precision=jax.lax.Precision('highest'))

        

        # Now handle impact param deviations
        acceleration1_b = -self.gradientPotentialImpact_per_SH(x0,t) # nSH x 3
        d_qalpha1dot_db = dv1_db 
        d_palpha1dot_db = acceleration1_b + jnp.einsum('ij,kj->ki',d2H_dq2,dx1_db,optimize='optimal')

        # Now handle impact velocity deviations
        acceleration1_v = -self.gradientPotentialVelocity_per_SH(x0,t)
        d_qalpha1dot_dv = dv1_dv 
        d_palpha1dot_dv = acceleration1_v + jnp.einsum('ij,kj->ki',d2H_dq2,dx1_dv,optimize='optimal')


        # Package the output
        return jnp.stack([jnp.hstack([d_qdot_d_eps,d_pdot_d_eps]),
                jnp.hstack([d_qalpha1dot_dtheta, d_palpha1dot_dtheta]),
                jnp.hstack([d_qalpha1dot_db, d_palpha1dot_db]), 
                jnp.hstack([d_qalpha1dot_dv, d_palpha1dot_dv])]) # 4 x Nsh x 6
    
        #jnp.dstack([d_qdot_d_eps,d_pdot_d_eps, d_qalpha1dot_dtheta, d_palpha1dot_dtheta])# n_star x nSH x 12 #jnp.hstack([d_qdot_d_eps,d_pdot_d_eps, d_qalpha1dot_dtheta, d_palpha1dot_dtheta])
        #jnp.hstack([v0,acceleration0,d_qdot_d_eps,d_pdot_d_eps,d_qalpha1dot_dtheta,d_palpha1dot_dtheta])#[jnp.hstack([v0,acceleration0]),
               # jnp.hstack([d_qdot_d_eps,d_pdot_d_eps]),
                #jnp.hstack([d_qalpha1dot_dtheta, d_palpha1dot_dtheta])
    
    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_StreamSpline_PID_mass_radius_impact_velocity_Single(self,w0,t0,t1, ts, state):
        #def event_fun(state, **kwargs):
        #    return state.tnext > 0.0#>=0.02
        #spl, ind = state
        #arr, narr = eqx.partition(spl, eqx.is_array)
        #arr = jax.tree_util.tree_map(lambda x: x[ind], arr)
        #interp = eqx.combine(arr, narr)

        #new_state = (interp,ind)


        term = ODETerm(self.velocity_acceleration_perturbation_from_PreComp_mass_radius_impact_velocity_Single)
        solver = diffrax.Dopri5(scan_kind="bounded")#Heun(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=ts, dense=False,steps=False)
        rtol: float = 1e-12# 
        atol: float = 1e-12# 
        dt0=None
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=.6,dtmax=None,force_dtmin=True)#.25
        max_steps: int = 5_000#50_000#10_000#80_000
        t0 = t0
        t1 = t1
        dense = False
        terminating_event =  None#DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=state,
            max_steps=max_steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint()#DirectAdjoint()
        )
       
        return solution
    
    ######## Second order derivatives ########################
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_from_PreComp_mass_radius_secondOrder_Single(self,t,xv,args):
        """
        Vary mass, radius, impact parameter, and impact velocity
        """
        ind, stream_func = args

        w0_at_t = interp_func(t,ind,stream_func)


        x0, v0 = w0_at_t[:3], w0_at_t[3:] 

        # Epsilon Derivs
        x1, v1 = xv[0][:,:3], xv[0][:,3:6] # nSHS x 3

        # Structural derivs
        dx1_dtheta, dv1_dtheta = xv[1][:,:3], xv[1][:,3:6] # nSHS x 3


        # Second order epsilon
        x2, v2 = xv[2][:,:3], xv[2][:,3:6]


        # acceleration due to perturbation
        acceleration1 = -self.gradientPotentialPerturbation_per_SH(x0,t) # nSH x 3
        # tidal tensor
        d2H_dq2 = -jax.jacrev(self.gradientPotentialBase)(x0,t) 
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # this is dpdot/deps. Injected acceleration + matmul(tidal tensor, q)
        d_pdot_d_eps = acceleration1 + jnp.einsum('ij,kj->ki',d2H_dq2,x1,optimize='optimal')

        # Now handle radius deviations
        acceleration1_r = -self.gradientPotentialStructural_per_SH(x0,t) # nSH x 3
        d_qalpha1dot_dtheta = dv1_dtheta
        d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ij,kj->ki',d2H_dq2,dx1_dtheta,optimize='optimal')#jnp.einsum('ijk,ik->ij',d2H_dq2,dx1_dtheta,precision=jax.lax.Precision('highest'))

        # Now handle second order correction (eps^2)
        tridal_tensor = jax.jacfwd(jax.jacfwd(self.gradientPotentialBase))
        tridal_tensor_eval = tridal_tensor(x0,t) # 3 x 3 x 3

        tidal_tensor_SH_eval = jax.jacfwd(self.gradientPotentialPerturbation_per_SH)(x0,t) # nSH x 3 x 3


        inner_contraction = jnp.einsum('ijk,lk->lij',tridal_tensor_eval,x1,optimize='optimal') # nSH x 3 x 3
        tridal_dot_dqdeps = jnp.sum(inner_contraction*x1[:,:,None],axis=2) # nSH x 3
        tidal_dot_d2qdeps2 = jnp.einsum('ij,kj->ki',-d2H_dq2,x2,optimize='optimal') # nSH x 3
        first_bracket = -(tridal_dot_dqdeps + tidal_dot_d2qdeps2) # nSH x 3

        second_bracket = -2*jnp.einsum('ijk,ik->ij',tidal_tensor_SH_eval,x1,optimize='optimal') # nSH x 3

        d2_pdot_deps2 = first_bracket + second_bracket


        output = jnp.stack([
            jnp.hstack([d_qdot_d_eps,d_pdot_d_eps]),
            jnp.hstack([d_qalpha1dot_dtheta, d_palpha1dot_dtheta]),
            jnp.hstack([v2, d2_pdot_deps2])
        ])

        return output

    
    
    
    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_StreamSpline_PID_mass_radius_SecondOrder_Single(self,w0,t0,t1, ts, state):
    

        term = ODETerm(self.velocity_acceleration_perturbation_from_PreComp_mass_radius_secondOrder_Single)
        solver = diffrax.Dopri5(scan_kind="bounded")#Heun(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=ts, dense=False,steps=False)
        rtol: float = 1e-12# 
        atol: float = 1e-12# 
        dt0=None
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=.6,dtmax=None,force_dtmin=True)#.25
        max_steps: int = 5_000#50_000#10_000#80_000
        t0 = t0
        t1 = t1
        dense = False
        terminating_event =  None#DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=state,
            max_steps=max_steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint()#DirectAdjoint()
        )
       
        return solution
    
    
    #####################
    ## Mass radius impact and velocity perturbations, NO PRECOMP 
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_no_PreComp_mass_radius_impact_velocity(self,t,xv,args):
        """
        Vary mass, radius, impact parameter, and impact velocity
        """
        #ind, stream_func_lead, stream_func_trail = args

        #w0_at_t_lead = interp_func(t,ind,stream_func_lead)
        #w0_at_t_trail = interp_func(t,ind,stream_func_trail)
        
        
        #rr, narr = eqx.partition(spl, eqx.is_array)
        #arr = jax.tree_util.tree_map(lambda x: x[particle_ind], arr)
        #interp = eqx.combine(arr, narr)
        #w0_at_t = interp.evaluate(t)

        x0, v0 = xv[0][:3], xv[0][3:6] # 3
        
        # Epsilon Derivs
        x1, v1 = xv[1][:,:3], xv[1][:,3:6] # 2 x nSHS x 3

        # Structural derivs
        dx1_dtheta, dv1_dtheta = xv[2][:,:3], xv[2][:,3:6] # nSHS x 3

        # Impact derivs
        dx1_db, dv1_db = xv[3][:,:3], xv[3][:,3:6] # nSHS x 3

        # Velocity derivs
        dx1_dv, dv1_dv = xv[4][:,:3], xv[4][:,3:6] # nSHS x 3


        acceleration0 = -self.gradientPotentialBase(x0,t) #jax.vmap(self.gradientPotentialBase,in_axes=((0,None)))(x0,t)
        
        # acceleration due to perturbation
        acceleration1 = -self.gradientPotentialPerturbation_per_SH(x0,t) # nSH x 3# -jax.vmap(self.gradientPotentialPerturbation_per_SH,in_axes=((0,None)))(x0,t) # 2 x nSH x 3
        # tidal tensor
        d2H_dq2 = -jax.jacrev(self.gradientPotentialBase)(x0,t) #-jax.vmap(jax.jacrev(self.gradientPotentialBase),in_axes=((0,None)))(x0,t) 
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # this is dpdot/deps. Injected acceleration + matmul(tidal tensor, q)
        d_pdot_d_eps = acceleration1 + jnp.einsum('ij,kj->ki',d2H_dq2,x1,optimize='optimal')#jnp.einsum('ijk,ilk->ilj',d2H_dq2,x1,optimize='optimal')
        ##d_pdot_d_eps = acceleration1 + jnp.einsum('ij,kj->ki',d2H_dq2,x1,optimize='optimal') #nSH x 3 #jnp.matmul(d2H_dq2,x1) #jnp.einsum('ijk,ij->ik',d2H_dq2,x1,precision=jax.lax.Precision('highest'))   #n_star x 3
        
        # Now handle radius deviations
        acceleration1_r = -self.gradientPotentialStructural_per_SH(x0,t)#-jax.vmap(self.gradientPotentialStructural_per_SH,in_axes=((0,None)))(x0,t) # nSH x 3
        d_qalpha1dot_dtheta = dv1_dtheta
        #d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ij,kj->ki',d2H_dq2,dx1_dtheta, optimize='optimal')#jnp.matmul(d2H_dq2,dx1_dtheta)
        d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ij,kj->ki',d2H_dq2,dx1_dtheta,optimize='optimal')#jnp.einsum('ijk,ilk->ilj',d2H_dq2,dx1_dtheta,optimize='optimal')#jnp.einsum('ijk,ik->ij',d2H_dq2,dx1_dtheta,precision=jax.lax.Precision('highest'))

        

        # Now handle impact param deviations
        acceleration1_b =  self.gradientPotentialImpact_per_SH(x0,t) # nSH x 3# jax.vmap(self.gradientPotentialImpact_per_SH, in_axes=((0,None)))(x0,t) # nSH x 3
        d_qalpha1dot_db = dv1_db 
        d_palpha1dot_db = acceleration1_b +  jnp.einsum('ij,kj->ki',d2H_dq2,dx1_db,optimize='optimal')#jnp.einsum('ijk,ilk->ilj',d2H_dq2,dx1_db,optimize='optimal')

        # Now handle impact velocity deviations
        acceleration1_v = self.gradientPotentialVelocity_per_SH(x0,t)#jax.vmap(self.gradientPotentialVelocity_per_SH, in_axes=((0,None)))(x0,t)
        d_qalpha1dot_dv = dv1_dv 
        d_palpha1dot_dv = acceleration1_v +  jnp.einsum('ij,kj->ki',d2H_dq2,dx1_dv,optimize='optimal')#jnp.einsum('ijk,ilk->ilj',d2H_dq2,dx1_dv,optimize='optimal')


        # Package the output
        return [jnp.hstack([v0, acceleration0]),
                jnp.hstack([d_qdot_d_eps,d_pdot_d_eps]),
                jnp.hstack([d_qalpha1dot_dtheta, d_palpha1dot_dtheta]),
                jnp.hstack([d_qalpha1dot_db, d_palpha1dot_db]), 
                jnp.hstack([d_qalpha1dot_dv, d_palpha1dot_dv])]
    
        #jnp.dstack([d_qdot_d_eps,d_pdot_d_eps, d_qalpha1dot_dtheta, d_palpha1dot_dtheta])# n_star x nSH x 12 #jnp.hstack([d_qdot_d_eps,d_pdot_d_eps, d_qalpha1dot_dtheta, d_palpha1dot_dtheta])
        #jnp.hstack([v0,acceleration0,d_qdot_d_eps,d_pdot_d_eps,d_qalpha1dot_dtheta,d_palpha1dot_dtheta])#[jnp.hstack([v0,acceleration0]),
               # jnp.hstack([d_qdot_d_eps,d_pdot_d_eps]),
                #jnp.hstack([d_qalpha1dot_dtheta, d_palpha1dot_dtheta])
    
    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_mass_radius_impact_velocity_no_precomp(self,w0,t0,t1, ts):
        #def event_fun(state, **kwargs):
        #    return state.tnext > 0.0#>=0.02
        #spl, ind = state
        #arr, narr = eqx.partition(spl, eqx.is_array)
        #arr = jax.tree_util.tree_map(lambda x: x[ind], arr)
        #interp = eqx.combine(arr, narr)

        #new_state = (interp,ind)


        term = ODETerm(self.velocity_acceleration_perturbation_no_PreComp_mass_radius_impact_velocity)
        solver = diffrax.Dopri8(scan_kind="bounded")#Heun(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=ts, dense=False,steps=False)
        rtol: float = 1e-11# 
        atol: float = 1e-11# 
        dt0=None
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=.6,dtmax=None,force_dtmin=True)#.25
        max_steps: int = 5_000#50_000#10_000#80_000
        t0 = t0
        t1 = t1
        dense = False
        terminating_event =  None#DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=None,
            max_steps=max_steps,
            adjoint=diffrax.RecursiveCheckpointAdjoint()#DirectAdjoint()
        )
       
        return solution



    ########
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_from_streamState_mass_radius(self,t,xv,args):
        """
        use when integrating particle spray model, or if you have distinct leading trailing states!
        xv has length n_star x 12. First six is d{q,p}/deps, last six is d^2{q,p}/deps*dr
        xv[:,:6]
        """
        # lead and trailing arm unperturbed spline functions
        lead_spl, trail_spl, _ = args
        # phase-space position of lead/trail arms
        
        lead_w0_at_t, trail_w0_at_t = jax.vmap(lambda s: s.evaluate(t))(lead_spl), jax.vmap(lambda s: s.evaluate(t))(trail_spl)
        
        
        w0_at_t = jnp.vstack([lead_w0_at_t,trail_w0_at_t])  #n_star x 6
        # q, p for particles in unperturbed model at time t
        x0, v0 = w0_at_t[:,:3], w0_at_t[:,3:] # n_star x 3, n_star x 3
        
        # numerically integrated velocity and acceleration (i.e., pos and vel) up to current time t
        x1,v1 = xv[:,:,:3], xv[:,:,3:6] # n_star x nSHs x 3, n_star x nSHs x 3
        dx1_dtheta, dv1_dtheta = xv[:,:,6:9], xv[:,:,9:] # n_star x nSHs x 3, n_star x nSHs x 3
        
        # acceleration due to perturbation
        acceleration1 = -jax.vmap(self.gradientPotentialPerturbation_per_SH,in_axes=(0,None))(x0,t) #n_star x nSH x  3
        # tidal tensor
        d2H_dq2 = -jax.vmap(jax.jacrev(self.gradientPotentialBase),in_axes=((0,None)))(x0,t) #n_star x 3 x3
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # this is dpdot/deps. Injected acceleration + matmul(tidal tensor, q)
        d_pdot_d_eps = acceleration1 + jnp.einsum('ijk,ilk->ilj',d2H_dq2,x1,optimize='optimal')#jnp.einsum('ijk,ik->ij',d2H_dq2,x1,precision=jax.lax.Precision('highest'))   #n_star x 3 
        
        # Some of the splines will evaluate to nans. Replace these with zero: they do not exist in the simulation yet.
        d_qdot_d_eps = jnp.where(jnp.isnan(x0[:,None,:])+jnp.isnan(v0[:,None,:]),0.0,d_qdot_d_eps)#jnp.where(jnp.isnan(d_qdot_d_eps),0.0,d_qdot_d_eps)
        d_pdot_d_eps =  jnp.where(jnp.isnan(x0[:,None,:])+jnp.isnan(v0[:,None,:]),0.0,d_pdot_d_eps)#jnp.where(jnp.isnan(d_pdot_d_eps),0.0,d_pdot_d_eps)
        
        # Now deal with structural parameter (radius)
        acceleration1_r = -jax.vmap(self.gradientPotentialStructural_per_SH,in_axes=(0,None))(x0,t) #n_star x nSH x  3
        d_qalpha1dot_dtheta = dv1_dtheta
        d_palpha1dot_dtheta = acceleration1_r + jnp.einsum('ijk,ilk->ilj',d2H_dq2,dx1_dtheta,optimize='optimal')#jnp.einsum('ijk,ik->ij',d2H_dq2,dx1_dtheta,precision=jax.lax.Precision('highest'))

        # Some of the splines will evaluate to nans. Replace these with zero: they do not exist in the simulation yet.
        d_qalpha1dot_dtheta  = jnp.where(jnp.isnan(x0[:,None,:])+jnp.isnan(v0[:,None,:]),0.0,d_qalpha1dot_dtheta)
        d_palpha1dot_dtheta = jnp.where(jnp.isnan(x0[:,None,:])+jnp.isnan(v0[:,None,:]),0.0,d_palpha1dot_dtheta)

        # Package the output
        return jnp.dstack([d_qdot_d_eps,d_pdot_d_eps, d_qalpha1dot_dtheta, d_palpha1dot_dtheta])# n_star x nSH x 12
    
    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_StreamSpline_PID_mass_radius(self,w0,t0,t1, ts, state):
        def event_fun(state, **kwargs):
            return state.tnext > 0.0#>=0.02
        term = ODETerm(self.velocity_acceleration_perturbation_from_streamState_mass_radius)
        solver = diffrax.Bosh3()#diffrax.Ralston()#scan_kind="bounded")#diffrax.Dopri5(scan_kind="bounded")#diffrax.Heun(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=ts, dense=False,steps=False)
        rtol: float = 1e-13#1e-6#1e-4and dtmin 0.1
        atol: float = 1e-13#1e-5
        dt0=None#0.5
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=.05,dtmax=None,force_dtmin=True, jump_ts=state[2])  
        max_steps: int = 50_000#50_000#10_000#80_000
        t0 = t0
        t1 = t1
        dense = False
        terminating_event =  None#DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=state,
            max_steps=max_steps,
            adjoint=diffrax.DirectAdjoint()#DirectAdjoint()

        )
       
        return solution


    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_from_streamState(self,t,xv,args):
        """
        use when integrating particle spray model, or if you have distinct leading trailing states!
        """
        # lead and trailing arm unperturbed spline functions
        lead_spl, trail_spl, _ = args
        # phase-space position of lead/trail arms
        
        lead_w0_at_t, trail_w0_at_t = jax.vmap(lambda s: s.evaluate(t))(lead_spl), jax.vmap(lambda s: s.evaluate(t))(trail_spl)
        w0_at_t = jnp.vstack([lead_w0_at_t,trail_w0_at_t])  #n_star x 6
        # q, p for particles in unperturbed model at time t
        x0, v0 = w0_at_t[:,:3], w0_at_t[:,3:] # n_star x 3, n_star x 3
        
        # numerically integrated velocity and acceleration (i.e., pos and vel) up to current time t
        x1,v1 = xv[:,:3], xv[:,3:] 
        
        # acceleration due to perturbation
        acceleration1 = -jax.vmap(self.gradientPotentialPerturbation,in_axes=(0,None))(x0,t) #n_star x 3
        # tidal tensor
        d2H_dq2 = -jax.vmap(jax.jacrev(self.gradientPotentialBase),in_axes=((0,None)))(x0,t) #n_star x 3 x3
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # this is dpdot/deps. Injected acceleration + matmul(tidal tensor, q)
        d_pdot_d_eps = acceleration1 + jnp.einsum('ijk,ik->ij',d2H_dq2,x1,precision=jax.lax.Precision('highest'))   #n_star x 3 CHNAGED IJK SIGNATURE>>> CHECK IT!
        
        # Some of the splines will evaluate to nans. Replace these with zero: they do not exist in the simulation yet.
        d_qdot_d_eps = jnp.where(jnp.isnan(x0)+jnp.isnan(v0),0.0,d_qdot_d_eps)#jnp.where(jnp.isnan(d_qdot_d_eps),0.0,d_qdot_d_eps)
        d_pdot_d_eps =  jnp.where(jnp.isnan(x0)+jnp.isnan(v0),0.0,d_pdot_d_eps)#jnp.where(jnp.isnan(d_pdot_d_eps),0.0,d_pdot_d_eps)
        
        # Package the output
        return jnp.hstack([d_qdot_d_eps,d_pdot_d_eps])# n_star x 6



    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_StreamSpline_PID(self,w0,t0,t1, ts, state):
        def event_fun(state, **kwargs):
            return state.tnext > 0.0#>=0.02
        term = ODETerm(self.velocity_acceleration_perturbation_from_streamState)
        solver = diffrax.Dopri5(scan_kind="bounded")#Heun(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=ts, dense=False,steps=False)
        rtol: float = 1e-11#1e-7 and dtmin 0.1
        atol: float = 1e-11#1e-7 and dtmin 0.1
        dt0=None
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=.05,dtmax=None,force_dtmin=True, jump_ts=state[2]) # dtmin = 0.1
        max_steps: int = 50_000#50_000#10_000#80_000
        t0 = t0
        t1 = t1
        dense = False
        terminating_event =  None#DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=state,
            max_steps=max_steps,
            adjoint=diffrax.DirectAdjoint()#DirectAdjoint()
        )
       
        return solution


    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_from_streamState_Single_lead_trail(self,t,xv,args):
        """
        use when integrating particle spray model, or if you have distinct leading trailing states!
        """
         # lead and trailing arm unperturbed spline functions
        lead_spl, trail_spl, lead_trail_bool, particle_num = args
        # phase-space position of lead/trail arms

        def lead_true_func(particle_num):
            return jax.vmap(lambda s: s.evaluate(t))(lead_spl)[particle_num,:]
        def trail_true_func(particle_num):
            return jax.vmap(lambda s: s.evaluate(t))(trail_spl)[particle_num,:]

        w0_at_t = jax.lax.cond(lead_trail_bool,lead_true_func,trail_true_func,particle_num)


        
        ##lead_w0_at_t, trail_w0_at_t = jax.vmap(lambda s: s.evaluate(t))(lead_spl)[particle_num,:], jax.vmap(lambda s: s.evaluate(t))(trail_spl)[particle_num,:]
        #w0_at_t = jnp.hstack([lead_w0_at_t,trail_w0_at_t])  #n_star x 6
        # q, p for particles in unperturbed model at time t
        #xv_lead, xv_trail = xv[0], xv[1]

        x0, v0 = w0_at_t[:3], w0_at_t[3:] # n_star x 3, n_star x 3
        
        # numerically integrated velocity and acceleration (i.e., pos and vel) up to current time t
        x1,v1 = xv[:3], xv[3:] 
        
        # acceleration due to perturbation
        acceleration1 = -self.gradientPotentialPerturbation(x0,t) 
        # tidal tensor
        d2H_dq2 = -jax.jacrev(self.gradientPotentialBase)(x0,t) 
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # this is dpdot/deps. Injected acceleration + matmul(tidal tensor, q)
        d_pdot_d_eps = acceleration1 + jnp.matmul(d2H_dq2,x1,precision=jax.lax.Precision('highest')) 
        
        # Package the output
        return jnp.hstack([d_qdot_d_eps,d_pdot_d_eps]) 


    
    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_StreamSpline_PID_vmapped(self,w0,t0,t1, lead_trail_bool, particle_num, spl_lead, spl_trail,):
        def event_fun(state, **kwargs):
            return state.tnext > 0.0#>=0.02
        term = ODETerm(self.velocity_acceleration_perturbation_from_streamState_Single_lead_trail)
        solver = diffrax.Heun(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=None, dense=False,steps=False)
        rtol: float = 1e-10 #1e-7 or 1e-11 or 1e-13
        atol: float = 1e-12 #1e-11
        dt0=None
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=.05,dtmax=.5,force_dtmin=True) # dtmin = 0.75, dtmax=100
        max_steps: int = 2000#50_000#16**4#16**4 #was 16**5
        t0 = t0
        t1 = t1
        dense = False
        terminating_event =  None#DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=(spl_lead, spl_trail,lead_trail_bool, particle_num),
            max_steps=max_steps,
            adjoint=DirectAdjoint()
        )
       
        return solution

    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_from_noPreComp(self,t,xv,args):
        """
        use when integrating particle spray model, or if you have distinct leading trailing states!
        """
        x0, v0 = xv[:3], xv[3:6]
        x1, v1 = xv[6:9], xv[9:]

        acceleration0 = -self.gradientPotentialBase(x0,t)
        
        # acceleration due to perturbation
        acceleration1 = -self.gradientPotentialPerturbation(x0,t)
        # tidal tensor
        d2H_dq2 = -jax.jacrev(self.gradientPotentialBase)(x0,t) 
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # this is dpdot/deps. Injected acceleration + matmul(tidal tensor, q)
        d_pdot_d_eps = acceleration1 + jnp.matmul(d2H_dq2,x1)#jnp.einsum('ijk,ij->ik',d2H_dq2,x1,precision=jax.lax.Precision('highest'))   #n_star x 3
        
        
        # Package the output
        return jnp.hstack([v0,acceleration0,d_qdot_d_eps,d_pdot_d_eps])# n_star x 6
    

    @partial(jax.jit,static_argnums=((0,5)))
    def orbit_integrator_run_and_deriv_noPreComp(self,w0,t0,t1, ts, dense_arg):
        term = ODETerm(self.velocity_acceleration_perturbation_from_noPreComp)
        solver = diffrax.Dopri8(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=ts, dense=dense_arg,steps=False)
        rtol: float = 1e-7 #1e-13
        atol: float = 1e-7 #1e-13
        dt0=None
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=.1,dtmax=None,force_dtmin=True) # dtmin = 0.05, dtmax=100
        max_steps: int = 4_000#50_000#16**4#16**4 #was 16**5
        t0 = t0
        t1 = t1
        dense = False
        terminating_event =  None#DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=None,
            max_steps=max_steps,
            adjoint=DirectAdjoint()
        )
       
        return solution



    
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_ballParticles(self,t,xv,args):
        """
        use when integrating ball of particles!
        """
        ##particle_timesteps, particle_xvs = args.ts, args.ys
        ##time_index = jnp.argmax(jnp.isclose(particle_timesteps,t,atol=1e-30))#jnp.argmin(jnp.abs(particle_timesteps - t))
        ##x0,v0 = particle_xvs[time_index,:3], particle_xvs[time_index,3:]
        ##x1,v1 = xv[:3], xv[3:]
        
        ###orbit_func = args
        
        w0_at_t = jax.vmap(lambda s: s.evaluate(t))(args)
        x0, v0 = w0_at_t[:,:3], w0_at_t[:,3:] # n_star x 3, n_star x 3
        
        x1,v1 = xv[:,:3], xv[:,3:]
        
        acceleration1 = -jax.vmap(self.gradientPotentialPerturbation,in_axes=(0,None))(x0,t) #n_star x 3
        d2H_dq2 = -jax.vmap(jax.jacrev(self.gradientPotentialBase),in_axes=((0,None)))(x0,t) #n_star x 3 x3 ####-jax.jacrev(self.gradientPotentialBase)(x0,t) 
        d_qdot_d_eps = v1 
        d_pdot_d_eps = acceleration1 + jnp.einsum('ijk,ij->ik',d2H_dq2,x1,precision=jax.lax.Precision('highest'))     ##jnp.matmul(d2H_dq2,x1,precision=jax.lax.Precision('highest')) # #n_star x 3
        #jnp.where(d_qdot_d_eps==jnp.nan,0.0,None)
        ########d_qdot_d_eps = jnp.where(jnp.isnan(d_qdot_d_eps),0.0,d_qdot_d_eps)
        ########d_pdot_d_eps =  jnp.where(jnp.isnan(d_pdot_d_eps),0.0,d_pdot_d_eps)
        
        return jnp.hstack([d_qdot_d_eps,d_pdot_d_eps])#jnp.hstack([v1,acceleration]) # n_star x 6
    
    
    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_prestep(self,w0,states,steps):
        def event_fun(state, **kwargs):
            return state.tnext == 1#>=0.02
        term = ODETerm(self.velocity_acceleration_perturbation)
        solver = diffrax.LeapfrogMidpoint()####diffrax.Heun(scan_kind="bounded")
        saveat = SaveAt(t0=True, t1=True, ts=None, dense=False,steps=False)
        rtol: float = 1e-7 #1e-7
        atol: float = 1e-7 #1e-7
        dt0=None
        stepsize_controller = StepTo(ts=steps) #dtmin 0.01
        max_steps: int = 50_000#16**4#16**4 #was 16**5
        t0 = jnp.min(steps)#-4000.#states.ts*((states.ts<jnp.inf).astype(int)).min()#particle_timesteps.min()#0.0
        t1 = jnp.max(steps)#0.0#states.ts[states.ts<jnp.inf].max()#0.#particle_timesteps.max()#max_timesteps#4000.
        dense = False# WAS TRUE MADE FALSE.. NEED TRUE FOR GRADIENTS!!!! MAXSTEPS!
        #y0= w_init
        terminating_event =  None#DiscreteTerminatingEvent(event_fun)
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=None,#terminating_event,
            args=states,
            max_steps=max_steps,
            adjoint=DirectAdjoint()
        )
       
        return solution
    
    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_prestep_StreamSpline(self,w0,t0,t1,states, StepTo):
        def event_fun(state, **kwargs):
            return state.tnext > 0.0#>=0.02
        term = ODETerm(self.velocity_acceleration_perturbation_ballParticles)#(self.velocity_acceleration_perturbation)
        solver = diffrax.Tsit5(scan_kind="bounded")#diffrax.Midpoint()#diffrax.Heun(scan_kind="bounded")#diffrax.LeapfrogMidpoint()#diffrax.Dopri8(scan_kind="bounded")#diffrax.LeapfrogMidpoint()#diffrax.Dopri8(scan_kind="bounded")#diffrax.Heun(scan_kind="bounded")#Dopri5(scan_kind="bounded")#diffrax.LeapfrogMidpoint()####diffrax.Heun(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=None, dense=False,steps=False)
        rtol: float = 1e-6 #1e-7
        atol: float = 1e-6 #1e-7
        dt0=None#0.1#1.0#None#0.5#None#.5#None
        stepsize_controller = diffrax.StepTo(ts=StepTo)#PIDController(rtol=rtol, atol=atol, dtmin=0.05,force_dtmin=True,dtmax=1.0)#ConstantStepSize()#PIDController(rtol=rtol, atol=atol, dtmin=0.01,force_dtmin=True,dtmax=1.0)#StepTo(ts=steps) #dtmin 0.01
        max_steps: int = 40_100#50_000#16**4#16**4 #was 16**5
        t0 = t0
        t1 = t1
        dense = False# WAS TRUE MADE FALSE.. NEED TRUE FOR GRADIENTS!!!! MAXSTEPS!
        #y0= w_init
        terminating_event =  DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=states,
            max_steps=max_steps,
            adjoint=DirectAdjoint()
        )
       
        return solution
    
    ################################ ENSEMBLE SPLINE, SINGLE EVALUATION############
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_from_streamState_sinlge(self,t,xv,args):
        """
        use when integrating particle spray model, or if you have distinct leading trailing states!
        """
        # lead and trailing arm unperturbed spline functions
        spl_func, particle_num = args
        # phase-space position of lead/trail arms
        
        w0_at_t = jax.vmap(lambda s: s.evaluate(t))(spl_func)[particle_num,:] #, jax.vmap(lambda s: s.evaluate(t))(trail_spl)[particle_num,:]
        #w0_at_t = jnp.hstack([lead_w0_at_t,trail_w0_at_t])  #6
        # q, p for particles in unperturbed model at time t
        x0, v0 = w0_at_t[:3], w0_at_t[3:] 
        
        # numerically integrated velocity and acceleration (i.e., pos and vel) up to current time t
        x1,v1 = xv[:3], xv[3:] 
        
        # acceleration due to perturbation
        acceleration1 = -self.gradientPotentialPerturbation(x0,t) 
        # tidal tensor
        d2H_dq2 = -jax.jacrev(self.gradientPotentialBase)(x0,t) 
        # For the Hamiltonian considered, dp/deps is just the integrated acceleration. Relabling...
        d_qdot_d_eps = v1 
        # this is dpdot/deps. Injected acceleration + matmul(tidal tensor, q)
        d_pdot_d_eps = acceleration1 + jnp.matmul(d2H_dq2,x1,precision=jax.lax.Precision('highest')) 
        
        # Package the output
        return jnp.hstack([d_qdot_d_eps,d_pdot_d_eps]) 

    
    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_fromState_iterator(self,w0,t0,t1,states):
        def event_fun(state, **kwargs):
            return state.tnext > 0.0#>=0.02
        term = ODETerm(self.velocity_acceleration_perturbation_from_streamState_sinlge)
        solver = diffrax.Heun(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=None, dense=False,steps=False)
        rtol: float = 1e-9 #1e-7 or 1e-11 or 1e-13
        atol: float = 1e-12 #1e-12
        dt0=None
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=.2,dtmax=None,force_dtmin=True) # dtmin = 0.75, dtmax=100
        max_steps: int = 50_100#50_000#16**4#16**4 #was 16**5
        t0 = t0
        t1 = t1
        dense = False
        terminating_event =  DiscreteTerminatingEvent(event_fun) #None
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=states,
            max_steps=max_steps,
            adjoint=DirectAdjoint()
        )
       
        return solution
    
    
    ############################### BATCHED PRESTEP ###############################
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_batched(self,t,xv,args):
        particleid, particle_timesteps, particle_xvs = args[0], args[1].ts, args[1].ys
        particle_timesteps, particle_xvs = particle_timesteps[particleid,:], particle_xvs[particleid,:,:]
        time_index = jnp.argmax(jnp.isclose(particle_timesteps,t,atol=1e-30))#jnp.argmin(jnp.abs(particle_timesteps - t))
        x0,v0 = particle_xvs[time_index,:3], particle_xvs[time_index,3:]
        x1,v1 = xv[:3], xv[3:]
        
        acceleration1 = -self.gradientPotentialPerturbation(x0,t)
        d2H_dq2 = -jax.jacrev(self.gradientPotentialBase)(x0,t)
        d_qdot_d_eps = v1
        d_pdot_d_eps = acceleration1 + jnp.matmul(d2H_dq2,x1,precision=jax.lax.Precision('highest'))
        

        return jnp.hstack([d_qdot_d_eps,d_pdot_d_eps])#jnp.hstack([v1,acceleration])
    
    
    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_batched_prestep(self,w0, particleid, states,steps):
        def event_fun(state, **kwargs):
            return state.tnext > 0.0
        term = ODETerm(self.velocity_acceleration_perturbation_batched)
        solver = diffrax.LeapfrogMidpoint()####diffrax.Heun(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=None, dense=False,steps=False)
        rtol: float = 1e-7 #1e-7
        atol: float = 1e-7 #1e-7
        dt0=None
        stepsize_controller = StepTo(ts=steps) #dtmin 0.01
        max_steps: int = 50_000#16**4#16**4 #was 16**5
        t0 = jnp.min(steps)#-4000.#states.ts*((states.ts<jnp.inf).astype(int)).min()#particle_timesteps.min()#0.0
        t1 = jnp.max(steps)#0.0#states.ts[states.ts<jnp.inf].max()#0.#particle_timesteps.max()#max_timesteps#4000.
        dense = False# WAS TRUE MADE FALSE.. NEED TRUE FOR GRADIENTS!!!! MAXSTEPS!
        #y0= w_init
        terminating_event = DiscreteTerminatingEvent(event_fun)
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,#terminating_event,
            args=(particleid,states),
            max_steps=max_steps,
            adjoint=DirectAdjoint()
        )
       
        return solution
    
    ############################### SPLINES #######################################
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_splines(self,t,xv,args):
        #particle_timesteps, particle_xvs = args
        #time_index = jnp.argmin(jnp.abs(particle_timesteps - t))
        x0,v0 = jnp.array([self.spl_x(t), self.spl_y(t), self.spl_z(t)]), jnp.array([self.spl_vx(t), self.spl_vy(t), self.spl_vz(t)])#particle_xvs[time_index,:3], particle_xvs[time_index,3:]
        x1,v1 = xv[:3], xv[3:]
        
        acceleration1 = -self.gradientPotentialPerturbation(x0,t)
        d2H_dq2 = -jax.jacfwd(self.gradientPotentialBase)(x0,t)
        d_qdot_d_eps = v1
        d_pdot_d_eps = acceleration1 + jnp.matmul(d2H_dq2,x1)
        

        return jnp.hstack([d_qdot_d_eps,d_pdot_d_eps])#jnp.hstack([v1,acceleration])
    
    @partial(jax.jit,static_argnums=((0,5)))
    def orbit_integrator_run_splines(self,w0,t0,t1,ts, steps):
        term = ODETerm(self.velocity_acceleration_perturbation_splines)
        solver = Dopri5(scan_kind="bounded")
        if steps == True:
            saveat = SaveAt(t0=True, t1=False,ts=None,dense=False,steps=True) # ts=ts,
        if steps == False:
            saveat = SaveAt(t0=False, t1=True,ts=ts,dense=False) # ts=ts,
        rtol: float = 1e-7 #1e-7
        atol: float = 1e-7 #1e-7
        def true_func(dt):
            return 0.5
        def false_func(dt):
            return -0.5
        dt0= jax.lax.cond(t0<t1,true_func,false_func,None)
        
        #if t0<t1:
        #    dt0=0.5
        #if t0>t1:
        #    dt0=-0.5
        stepsize_controller = ConstantStepSize()#PIDController(rtol=rtol, atol=atol, dtmin=0.01,force_dtmin=True)#ConstantStepSize()# ## #dtmin 0.01
        max_steps: int = 8100#4000
        t0 = t0#0.0
        t1 = t1#4000.
        dense = True# WAS TRUE MADE FALSE.. NEED TRUE FOR GRADIENTS!!!! MAXSTEPS!
        #y0= w_init

        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=None,
            max_steps=(max_steps if dense else None),
            adjoint=DirectAdjoint()
        )
        if steps == True:
            return solution
        if steps == False:
            return solution.ys
    
    
    ###############################################################################
    ####### DENSE ###########
    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_Dense(self,t,xv,args):
        #particle_timesteps, particle_xvs = args
        #time_index = jnp.argmin(jnp.abs(particle_timesteps - t))
        x0v0 = args.evaluate(t)#jnp.array([self.spl_x(t), self.spl_y(t), self.spl_z(t)]), jnp.array([self.spl_vx(t), self.spl_vy(t), self.spl_vz(t)])#particle_xvs[time_index,:3], particle_xvs[time_index,3:]
        x0,v0 = x0v0[:3],x0v0[3:]
        x1,v1 = xv[:3], xv[3:]
        
        acceleration1 = -self.gradientPotentialPerturbation(x0,t)
        d2H_dq2 = -jax.jacfwd(self.gradientPotentialBase)(x0,t)
        d_qdot_d_eps = v1
        d_pdot_d_eps = acceleration1 + jnp.matmul(d2H_dq2,x1)
        

        return jnp.hstack([d_qdot_d_eps,d_pdot_d_eps])#jnp.hstack([v1,acceleration])
    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_prestepDense(self,w0,states):
        def event_fun(state, **kwargs):
            return state.tnext == 1#>=0.02
        term = ODETerm(self.velocity_acceleration_perturbation_Dense)
        solver = diffrax.Tsit5()#Dopri5(scan_kind="bounded")#diffrax.LeapfrogMidpoint()#(scan_kind="bounded")#Dopri5(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=None, dense=False,steps=False)
        rtol: float = 1e-8 #1e-7
        atol: float = 1e-8 #1e-7
        dt0=None
        stepsize_controller = PIDController(rtol=rtol, atol=atol, dtmin=0.01,force_dtmin=True,dtmax=0.5)#ConstantStepSize()#StepTo(ts=particle_timesteps) #dtmin 0.01
        max_steps: int = 16**4#16**4 #was 16**5
        t0 = -4000.#states.ts*((states.ts<jnp.inf).astype(int)).min()#particle_timesteps.min()#0.0
        t1 = 0.0#states.ts[states.ts<jnp.inf].max()#0.#particle_timesteps.max()#max_timesteps#4000.
        dense = True# WAS TRUE MADE FALSE.. NEED TRUE FOR GRADIENTS!!!! MAXSTEPS!
        #y0= w_init
        terminating_event =  DiscreteTerminatingEvent(event_fun)
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=None,#terminating_event,
            args=states,
            max_steps=16**4,
            adjoint=DirectAdjoint()
        )
       
        return solution
    
    
    
    
    
    ###########
    

    @partial(jax.jit,static_argnums=(0,))
    def velocity_acceleration_perturbation_allTogether(self,t,xv,args):
        x0,v0 = xv[0][:3], xv[0][3:]
        x1,v1 = xv[1][:3], xv[1][3:]
        
        acc_unperturbed = -self.gradientPotentialBase(x0,t)
        
        acceleration1 = -self.gradientPotentialPerturbation(x0,t)
        d2H_dq2 = -jax.jacfwd(self.gradientPotentialBase)(x0,t)
        d_qdot_d_eps = v1
        d_pdot_d_eps = acceleration1 + jnp.matmul(d2H_dq2,x1)
        

        return jnp.vstack([jnp.hstack([v0,acc_unperturbed]),jnp.hstack([d_qdot_d_eps,d_pdot_d_eps])])
    

    @partial(jax.jit,static_argnums=(0,))
    def orbit_integrator_run_and_deriv(self,w0,t0,t1):
        def event_fun(state, **kwargs):
            return state.tnext > 0.0#>=0.02
        term = ODETerm(self.velocity_acceleration_perturbation_allTogether)
        solver = Dopri5(scan_kind="bounded")
        saveat = SaveAt(t0=False, t1=True, ts=None, dense=False,steps=False)
        rtol: float = 1e-7 #1e-7
        atol: float = 1e-7 #1e-7
        dt0=0.5
        stepsize_controller = ConstantStepSize()#StepTo(ts=particle_timesteps) #dtmin 0.01
        max_steps: int = 8100#16**4 #was 16**5
        t0 = t0#0.0
        t1 = t1#max_timesteps#4000.
        dense = False# WAS TRUE MADE FALSE.. NEED TRUE FOR GRADIENTS!!!! MAXSTEPS!
        #y0= w_init
        terminating_event =  None#DiscreteTerminatingEvent(event_fun)
        solution = diffeqsolve(
            terms=term,
            solver=solver,
            t0=t0,
            t1=t1,
            y0=w0,
            dt0=dt0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            discrete_terminating_event=terminating_event,
            args=None,
            max_steps=(max_steps if dense else None),
            adjoint=DirectAdjoint()
        )
       
        return solution
    
    @partial(jax.jit,static_argnums=(0,))
    def release_model(self, x, v, Msat,i, t, seed_num):
        """
        Simplification of particle spray: just release particles in gaussian blob at each lagrange point.
        User sets the spatial and velocity dispersion for the "leaking" of particles
        TODO: change random key handling... need to do all of the sampling up front...
        """
        key_master = jax.random.PRNGKey(seed_num)
        random_ints = jax.random.randint(key=key_master,shape=(5,),minval=0,maxval=1000)

        keya = jax.random.PRNGKey(i*random_ints[0])#jax.random.PRNGKey(i*13)
        keyb = jax.random.PRNGKey(i*random_ints[1])#jax.random.PRNGKey(i*23)
        
        keyc = jax.random.PRNGKey(i*random_ints[2])#jax.random.PRNGKey(i*27)
        keyd = jax.random.PRNGKey(i*random_ints[3])#jax.random.PRNGKey(i*3)
        keye = jax.random.PRNGKey(i*random_ints[4])#jax.random.PRNGKey(i*17)
        
        L_close, L_far = self.lagrange_pts(x,v,Msat, t) # each is an xyz array
        
        omega_val = self.omega(x,v)
        
        
        r = jnp.linalg.norm(x)
        r_hat = x/r
        r_tidal = self.tidalr_mw(x,v,Msat, t)
        rel_v = omega_val*r_tidal #relative velocity
        
        #circlar_velocity
        dphi_dr = jnp.sum(self.gradient(x, t)*r_hat)
        v_circ = rel_v##jnp.sqrt( r*dphi_dr )
        
        L_vec = jnp.cross(x,v)
        z_hat = L_vec / jnp.linalg.norm(L_vec)
        
        phi_vec = v - jnp.sum(v*r_hat)*r_hat
        phi_hat = phi_vec/jnp.linalg.norm(phi_vec)
        vt_sat = jnp.sum(v*phi_hat)
        
        
        kr_bar = 2.0
        kvphi_bar = 0.3
        ####################kvt_bar = 0.3 ## FROM GALA
        
        kz_bar = 0.0
        kvz_bar = 0.0
        
        sigma_kr = 0.5
        sigma_kvphi = 0.5
        sigma_kz = 0.5
        sigma_kvz = 0.5
        ##############sigma_kvt = 0.5 ##FROM GALA
        
        kr_samp =  kr_bar + jax.random.normal(keya,shape=(1,))*sigma_kr
        kvphi_samp = kr_samp*(kvphi_bar  + jax.random.normal(keyb,shape=(1,))*sigma_kvphi)
        kz_samp = kz_bar + jax.random.normal(keyc,shape=(1,))*sigma_kz
        kvz_samp = kvz_bar + jax.random.normal(keyd,shape=(1,))*sigma_kvz
        ########kvt_samp = kvt_bar + jax.random.normal(keye,shape=(1,))*sigma_kvt
        
        ## Trailing arm
        pos_trail = x + kr_samp*r_hat*(r_tidal) #nudge out
        pos_trail  = pos_trail + z_hat*kz_samp*(r_tidal/1.0)#r #nudge above/below orbital plane
        v_trail = v + (0.0 + kvphi_samp*v_circ*(1.0))*phi_hat#v + (0.0 + kvphi_samp*v_circ*(-r_tidal/r))*phi_hat #nudge velocity along tangential direction
        v_trail = v_trail + (kvz_samp*v_circ*(1.0))*z_hat#v_trail + (kvz_samp*v_circ*(-r_tidal/r))*z_hat #nudge velocity along vertical direction
        
        ## Leading arm
        pos_lead = x + kr_samp*r_hat*(-r_tidal) #nudge in
        pos_lead  = pos_lead + z_hat*kz_samp*(-r_tidal/1.0)#r #nudge above/below orbital plane
        v_lead = v + (0.0 + kvphi_samp*v_circ*(-1.0))*phi_hat#v + (0.0 + kvphi_samp*v_circ*(r_tidal/r))*phi_hat #nudge velocity along tangential direction
        v_lead = v_lead + (kvz_samp*v_circ*(-1.0))*z_hat#v_lead + (kvz_samp*v_circ*(r_tidal/r))*z_hat #nudge velocity against vertical direction
        
    
        
        
        return pos_lead, pos_trail, v_lead, v_trail
    
    @partial(jax.jit,static_argnums=(0,))
    def gen_stream_ics(self, ts, prog_w0, Msat, seed_num):
        ws_jax = self.orbit_integrator_run_notdense(prog_w0,jnp.min(ts),jnp.max(ts),ts,False)#self.orbit_integrator_run(prog_w0,jnp.min(ts),jnp.max(ts),ts,False)
        
        def scan_fun(carry, t):
            i, pos_close, pos_far, vel_close, vel_far = carry
            pos_close_new, pos_far_new, vel_close_new, vel_far_new = self.release_model(ws_jax[i,:3], ws_jax[i,3:], Msat,i, t, seed_num)
            return [i+1, pos_close_new, pos_far_new, vel_close_new, vel_far_new], [pos_close_new, pos_far_new, vel_close_new, vel_far_new]#[i+1, pos_close_new, pos_far_new, vel_close_new, vel_far_new]
            
            
        #init_carry = [0, 0, 0, 0, 0]
        init_carry = [0, jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.]), jnp.array([0.0,0.0,0.])] 
        final_state, all_states = jax.lax.scan(scan_fun, init_carry, ts)#[:-1])#ts[1:])
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = all_states
        return pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr
    
            
    #####@partial(jax.jit,static_argnums=(0,))
    def gen_stream_scan(self, ts, prog_w0, Msat, seed_num, steps):
        """
        Generate stellar stream by scanning over the release model/integration. Better for CPU usage.
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat, seed_num)
        ########@jax.jit
        def scan_fun(carry, particle_idx):
            i, pos_close_curr, pos_far_curr, vel_close_curr, vel_far_curr = carry
            curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
            curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
            ########w0_lead_trail = jnp.vstack([curr_particle_w0_close,curr_particle_w0_far])
            
            minval, maxval =  ts[i],ts[-1]
            ####integrate_different_ics = lambda ics:  self.orbit_integrator_run(ics,minval,maxval,None, steps)#####[0]
            #####w_particle_close, w_particle_far = jax.vmap(integrate_different_ics,in_axes=(0,))(w0_lead_trail) #vmap over leading and trailing arm
            
            w_particle_close = self.orbit_integrator_run_dense(curr_particle_w0_close,minval,maxval,None,steps)###[0]
            w_particle_far = self.orbit_integrator_run_dense(curr_particle_w0_far,minval,maxval,None,steps)###[0]
            
            
            
            return [i+1, pos_close_arr[i+1,:], pos_far_arr[i+1,:], vel_close_arr[i+1,:], vel_far_arr[i+1,:]], [w_particle_close, w_particle_far]
        init_carry = [0, pos_close_arr[0,:], pos_far_arr[0,:], vel_close_arr[0,:], vel_far_arr[0,:]]
        particle_ids = jnp.arange(len(pos_close_arr))
        final_state, all_states = jax.lax.scan(scan_fun, init_carry, particle_ids)
        lead_arm, trail_arm = all_states
        return lead_arm, trail_arm
    
    @partial(jax.jit,static_argnums=((0,5)))
    def gen_stream_vmapped(self, ts, prog_w0, Msat, seed_num, steps):
        """
        Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat, seed_num)
        @jax.jit
        def single_particle_integrate(particle_number,pos_close_curr,pos_far_curr,vel_close_curr,vel_far_curr):
            curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
            curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
            t_release = ts[particle_number]
            t_final = ts[-1] + .01
            
            w_particle_close = self.orbit_integrator_run_dense(curr_particle_w0_close,t_release,t_final,None,steps)###[0]
            w_particle_far = self.orbit_integrator_run_dense(curr_particle_w0_far,t_release,t_final,None,steps)###[0]
            
            return w_particle_close, w_particle_far
        particle_ids = jnp.arange(len(pos_close_arr))
        
        return jax.vmap(single_particle_integrate,in_axes=(0,0,0,0,0,))(particle_ids,pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr)
    
    
    @partial(jax.jit,static_argnums=((0,5)))
    def gen_stream_vmapped_notdense(self, ts, prog_w0, Msat, seed_num, steps):
        """
        Generate stellar stream by vmapping over the release model/integration. Better for GPU usage.
        """
        pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr = self.gen_stream_ics(ts, prog_w0, Msat, seed_num)
        @jax.jit
        def single_particle_integrate(particle_number,pos_close_curr,pos_far_curr,vel_close_curr,vel_far_curr):
            curr_particle_w0_close = jnp.hstack([pos_close_curr,vel_close_curr])
            curr_particle_w0_far = jnp.hstack([pos_far_curr,vel_far_curr])
            t_release = ts[particle_number]
            t_final = ts[-1] #################+ .01
            
            w_particle_close = self.orbit_integrator_run_notdense(curr_particle_w0_close,t_release,t_final,None,steps)###[0]
            w_particle_far = self.orbit_integrator_run_notdense(curr_particle_w0_far,t_release,t_final,None,steps)###[0]
            
            return w_particle_close, w_particle_far
        particle_ids = jnp.arange(len(pos_close_arr))
        
        return jax.vmap(single_particle_integrate,in_axes=(0,0,0,0,0,))(particle_ids,pos_close_arr, pos_far_arr, vel_close_arr, vel_far_arr)
    
@jax.jit
def interp_func(t,ind,stream_func):
    arr, narr = eqx.partition(stream_func, eqx.is_array)
    arr = jax.tree_util.tree_map(lambda x: x[ind], arr)
    interp = eqx.combine(arr, narr)
    w0_at_t = interp.evaluate(t)
    return w0_at_t

class Hamiltonian():
    def __init__(self, unperturbed_pot, perturbed_pot, units=None):
        super().__init__(units, {'unperturbed_pot':unperturbed_pot, 'perturbed_pot':perturbed_pot})
    
    @partial(jax.jit,static_argnums=(0,))
    def H0(self, q,p,t):
        return 0.5*(jnp.sum(p**2)) + self.unperturbed_pot(q,t)
    
    @partial(jax.jit,static_argnums=(0,))
    def H1(self, q,p,t):
        return self.perturbed_pot(q,t)
    
    



class LMCPotential(Potential):
    def __init__(self, LMC_internal, LMC_orbit, units=None):
        """
        LMC_internal: dictionary of LMC internal (i.e., structural) parameters
        LMC_orbit: {x,y,z,t}
        """
        super().__init__(units, {'LMC_internal': LMC_internal, 'LMC_orbit': LMC_orbit})
        self.spl_x = InterpolatedUnivariateSpline(self.LMC_orbit['t'], self.LMC_orbit['x'],k=3)
        self.spl_y = InterpolatedUnivariateSpline(self.LMC_orbit['t'], self.LMC_orbit['y'],k=3)
        self.spl_z = InterpolatedUnivariateSpline(self.LMC_orbit['t'], self.LMC_orbit['z'],k=3)

    
    @partial(jax.jit, static_argnums=(0,))
    def potential(self, xyz, t):
        LMC_pos = jnp.array([ self.spl_x(t), self.spl_y(t), self.spl_z(t) ])
        xyz_adjust = xyz - LMC_pos
        
        
        potential_lmc = NFWPotential(m=self.LMC_internal['m_NFW'], r_s=self.LMC_internal['r_s_NFW'],units=usys)
        #pot_bar = BarPotential(m=self.LMC_internal['bar_m'], a=self.LMC_internal['bar_a'],
        #                         b=self.LMC_internal['bar_b'], c=self.LMC_internal['bar_c'],Omega=self.LMC_internal['bar_Omega'],units=usys)
        #potential_list = [pot_NFW,pot_bar]
        #potential_lmc = Potential_Combine(potential_list=potential_list,units=usys)
        return potential_lmc.potential(xyz_adjust, t)

        
    
@jax.jit
def get_splines(x_eval,x,y):
    return InterpolatedUnivariateSpline(x,y,k=3)(x_eval)#jaxsplines(x,y)(x_eval)#InterpolatedUnivariateSpline(x,y,k=3)(x_eval)

@jax.jit
def map_splines(x_eval,x,y_fit):
    return jax.vmap(get_splines,in_axes=((None,None,1,)))(x_eval,x,y_fit)

    
@jax.jit
def single_subhalo_potential(dct,xyz,t):
    """
    Potential for a single subhalo
    TODO: custom unit specification/subhalo potential specficiation. 
    Currently supports units kpc, Myr, Msun, rad.
    """
    pot_single = Isochrone(m=dct['m'],a=dct['a'],units=usys)
    return pot_single.potential(xyz,t)

class SubHaloPopulation(Potential):
    def __init__(self, m, a, txyz_subhalo_arr, t_orbit, units=None):
        """
        m has length n_subhalo
        a has length n_subhalo
        txyz_subhalo_arr has shape t_orbit x n_subhalo x 3
        t_orbit is the array of times the subhalos are integrated over
        """
        super().__init__(units, {'m': m, 'a': a, 'txyz_subhalo_arr': txyz_subhalo_arr, 't_orbit': t_orbit})
        self.dct = {'m': self.m, 'a': self.a,}    
        
        
    @partial(jax.jit, static_argnums=(0,))
    def potential(self, xyz, t):
            
        x_at_t_eval = map_splines(t, self.t_orbit,self.txyz_subhalo_arr[:,:,0]) # expect n_subhalo x-positions
        y_at_t_eval = map_splines(t, self.t_orbit,self.txyz_subhalo_arr[:,:,1]) # expect n_subhalo y-positions
        z_at_t_eval = map_splines(t, self.t_orbit,self.txyz_subhalo_arr[:,:,2]) # expect n_subhalo z-positions


        subhalo_locations = jnp.vstack([x_at_t_eval,y_at_t_eval,z_at_t_eval]).T # n_subhalo x 3: the position of all subhalos at time t
        
        delta_position = xyz - subhalo_locations # n_subhalo x 3
        # sum over potential due to all subhalos in the field by vmapping over m, a, and delta_position
        pot_total = jnp.sum( jax.vmap(single_subhalo_potential,in_axes=( ({'m':0,'a':0,},0,None) ) )(self.dct,delta_position,t) ) 
        return pot_total


class StationarySubhaloPotential_Progenitor(Potential):
    def __init__(self, m, a, subhalo_locs, subhalo_timeparams, units=None):
        super().__init__(units, {'m': m, 'a': a, 'subhalo_locs': subhalo_locs, 'subhalo_timeparams':subhalo_timeparams})
        self.subhalo_dct = {'m': self.m, 'a': self.a, 'subhalo_locs': self.subhalo_locs, 
        'subhalo_timeparams':self.subhalo_timeparams }        
        self.gradient = self.subhalo_gradient
        
    @partial(jax.jit,static_argnums=(0,))
    def single_static_subhalo(self, subhalo_loc, t, subhalo_mass, subhalo_a):
        """
        Potential due to A SINGLE SUBHALO!
        subhalo_loc is the location of the subhalo
        t is the evaluation time
        subhalo_mass and subhalo_a are subhalo parameters for the SINGLE subhalo.
        This function is vmappable.
        Example usage:
        progenitor.single_static_subhalo( subhalo_locs[0], 1.0, m[0], a[0])
        """
        # prog location in simulation frame
        prog_loc = 0.0


        # relative location between evaluation point and subhalo 
        relative_pos = - subhalo_loc

        # evaluate subhalo potential at relative_pos
        pot_sub = NFWPotential(m=subhalo_mass,r_s=subhalo_a,units=usys)
        return pot_sub.potential(relative_pos,t)
    
    
    @partial(jax.jit,static_argnums=(0,))
    def amplitude_acceleration(self,t, dct):
        """
        Model for ampltiudes A_{x,y,z}(t) such that acc = {A_x*a_x, A_y*a_y, A_z*a_z}
        Suppose A_j(t) = amp_j*exp(-(t/sig_j_t)**2)*jnp.sin(t*(2*pi/tau_j) + delta_j)
        """
        t = t-dct['t_impact']
        A_x = dct['amp_x']*jnp.exp(-(t/dct['sigx_t'])**2 )*jnp.sin(t*(2*jnp.pi/dct['tau_x']) + dct['delta_x'])
        A_y = dct['amp_y']*jnp.exp(-(t/dct['sigy_t'])**2 )*jnp.sin(t*(2*jnp.pi/dct['tau_y']) + dct['delta_y'])
        A_z = dct['amp_z']*jnp.exp(-(t/dct['sigz_t'])**2 )*jnp.sin(t*(2*jnp.pi/dct['tau_z']) + dct['delta_z'])
        return jnp.vstack([A_x, A_y, A_z]).T
    

    @partial(jax.jit,static_argnums=(0,))
    def subhalo_gradient(self, xyz, t):
        accelerations = self.acceleration_intrinsic(t)##-jax.grad(self.potential)(self.subhalo_locs,t) #Outputs an array N_subhalos x 3
        amplitudes = self.amplitude_acceleration(t,self.subhalo_dct['subhalo_timeparams'])
        return -jnp.sum( amplitudes*accelerations, axis = 0) #ADDED MINUS SIGN SO ITS GRAD_PHI, NOT ACCELERATION!' 

    @partial(jax.jit,static_argnums=(0,))
    def potential_per_subhalo(self, xyz,t):
        """potential due to each subhalo, when evaluated at xyz. Outshape has length N_subhalos"""
        return jax.vmap(self.single_static_subhalo, in_axes=(0,None,0,0))( xyz, t, self.m, self.a )

                                                                            
    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t):
        pot_total = jnp.sum(self.potential_per_subhalo(self.subhalo_locs,t))#jnp.sum( jax.vmap(self.single_static_subhalo,in_axes=( (None, None, {'m':0,'a':0,'subhalo_locs':0,
                    #                                                        'subhalo_timeparams':0},) ) )(xyz,t,self.subhalo_dct) )                                                       
        return pot_total

    @partial(jax.jit,static_argnums=(0,))
    def acceleration_intrinsic(self, t):
        pot_total_func = lambda subhalo_locs, t: jnp.sum(self.potential_per_subhalo(subhalo_locs,t))
        return -jax.grad(pot_total_func)(self.subhalo_locs, t)





class StationarySubhaloPotential(Potential):
    def __init__(self, m, a, subhalo_locs, subhalo_timeparams, prog_xyz_arr, prog_time_arr, units=None):
        super().__init__(units, {'m': m, 'a': a, 'subhalo_locs': subhalo_locs, 'subhalo_timeparams':subhalo_timeparams, 
        'prog_xyz_arr':prog_xyz_arr, 'prog_time_arr':prog_time_arr})
        self.subhalo_dct = {'m': self.m, 'a': self.a, 'subhalo_locs': self.subhalo_locs, 
        'subhalo_timeparams':self.subhalo_timeparams }
        self.prog_splines = self.get_prog_splines()        
        self.gradient = self.subhalo_gradient

    def get_prog_splines(self):
        spl_x = InterpolatedUnivariateSpline(self.prog_time_arr,self.prog_xyz_arr[:,0], k=3)#jaxsplines(self.prog_time_arr,self.prog_xyz_arr[:,0])
        spl_y = InterpolatedUnivariateSpline(self.prog_time_arr,self.prog_xyz_arr[:,1], k=3)#jaxsplines(self.prog_time_arr,self.prog_xyz_arr[:,1])
        spl_z = InterpolatedUnivariateSpline(self.prog_time_arr,self.prog_xyz_arr[:,2], k=3)#jaxsplines(self.prog_time_arr,self.prog_xyz_arr[:,2])
        return [spl_x, spl_y, spl_z]
    
    @partial(jax.jit,static_argnums=(0,))
    def spl_prog_xyz(self, t):
        spl_x, spl_y, spl_z = self.prog_splines
        return jnp.array([spl_x(t), spl_y(t), spl_z(t)])

    @partial(jax.jit,static_argnums=(0,))
    def single_static_subhalo(self, xyz, t, dct):
        """
        Single static subhalo. This function is vmappable.
        Satellite frame is the "primed" frame
        """
        # prog location in simulation frame
        prog_loc = self.spl_prog_xyz(t)

        # evaluation point in satellite frame (the primed frame)
        xyz_prime = xyz - prog_loc

        # relative location between evaluation point and subhalo 
        relative_pos = xyz_prime - dct['subhalo_locs']

        # evaluate subhalo potential at relative_pos
        pot_sub = NFWPotential(m=dct['m'],r_s=dct['a'],units=usys)
        return pot_sub.potential(relative_pos,t)

    @partial(jax.jit,static_argnums=(0,))
    def amplitude_acceleration(self,t, dct):
        """
        Model for ampltiudes A_{x,y,z}(t) such that acc = {A_x*a_x, A_y*a_y, A_z*a_z}
        Suppose A_j(t) = amp_j*exp(-(t/sig_j_t)**2)*jnp.sin(t*(2*pi/tau_j) + delta_j)
        """
        t = t-dct['t_impact']
        A_x = dct['amp_x']*jnp.exp(-(t/dct['sigx_t'])**2 )*jnp.sin(t*(2*jnp.pi/dct['tau_x']) + dct['delta_x'])
        A_y = dct['amp_y']*jnp.exp(-(t/dct['sigy_t'])**2 )*jnp.sin(t*(2*jnp.pi/dct['tau_y']) + dct['delta_y'])
        A_z = dct['amp_z']*jnp.exp(-(t/dct['sigz_t'])**2 )*jnp.sin(t*(2*jnp.pi/dct['tau_z']) + dct['delta_z'])
        return jnp.vstack([A_x, A_y, A_z]).T
    
  
    @partial(jax.jit,static_argnums=(0,))
    def subhalo_gradient(self, xyz, t):
        accelerations = -jax.grad(self.potential)(xyz,t)#-jax.jacrev(self.potential_per_subhalo)(xyz,t)
        amplitudes = self.amplitude_acceleration(t,self.subhalo_dct['subhalo_timeparams'])
        return -jnp.sum( amplitudes*accelerations, axis = 0 ) 

    @partial(jax.jit,static_argnums=(0,))
    def potential_per_subhalo(self, xyz,t):
        """potential for each subhalo. Outshape has length N_subhalos"""
        return jax.vmap(self.single_static_subhalo,in_axes=( (None, None, {'m':0,'a':0,'subhalo_locs':0,
                                                                            'subhalo_timeparams':0},) ) )(xyz,t,self.subhalo_dct) 

    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t):
        pot_total = jnp.sum(self.potential_per_subhalo(xyz,t))                                                    
        return pot_total
    
    

class Subhalo_Dynamic_Potential(Potential):
    def __init__(self, m, a, subhalo_w0, ext_pot, t0, tf, num_spline, units=None):
        super().__init__(units, {'m': m, 'a': a, 'subhalo_w0': subhalo_w0, 'ext_pot': ext_pot, 't0':t0, 'tf':tf,
        'num_spline': num_spline})
        self.spline_times = jnp.linspace(self.t0,self.tf,self.num_spline)
        self.subhalo_locs = self.subhalo_integrate()
        #self.subhalo_wcurr = self.subhalo_w0
        #self.t_curr = 0.0
        #self.subhalo_xyzcurr = self.subhalo_w0[:,:3]
        #self.single_subhalo_potential_func = jax.vmap(NFWPotential, in_axes=((0,0,)))(self.m, self.a)

    @partial(jax.jit,static_argnums=(0,))
    def single_subhalo_potential(self, xyz, m, a, t):
        return NFWPotential(m=m, r_s=a,units=usys).potential(xyz,t)

    @partial(jax.jit,static_argnums=(0,))
    def spline_eval(self,times_fit,xyz_fit,t_eval):
        return InterpolatedUnivariateSpline(times_fit,xyz_fit,k=3)(t_eval)

    @partial(jax.jit,static_argnums=(0,))
    def vmapped_spline_eval(self,times, locs, t):
        return jax.vmap(self.spline_eval,in_axes=((None,0,None)))(times, locs,t)

    @partial(jax.jit,static_argnums=(0,))
    def spline_eval_ensemble(self, t):
        x_of_t = self.vmapped_spline_eval(self.spline_times, self.subhalo_locs[:,:-1,0], t)
        y_of_t = self.vmapped_spline_eval(self.spline_times, self.subhalo_locs[:,:-1,1], t)
        z_of_t = self.vmapped_spline_eval(self.spline_times, self.subhalo_locs[:,:-1,2], t)
        return jnp.vstack([x_of_t, y_of_t, z_of_t]).T



    @partial(jax.jit,static_argnums=(0,))
    def subhalo_integrate(self):
        new_subhalo_locs = jax.vmap(self.ext_pot.orbit_integrator_run,in_axes=((0,None,None,None)))(self.subhalo_w0,self.t0,self.tf,self.spline_times)
        return new_subhalo_locs

    
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        new_subhalo_locs = self.spline_eval_ensemble(t) # Evolve subhalos to  time t
        #print(self.subhalo_wcurr.shape)
        ####subhalo_xyz = jax.lax.dynamic_slice(self.subhalo_wcurr,(0,0),(len(self.subhalo_w0),3))
        #print(self.subhalo_xyzcurr.shape)
        relaive_position = xyz - new_subhalo_locs#self.subhalo_xyzcurr#subhalo_xyz#self.subhalo_wcurr#[:,:3]
        #print(relaive_position.shape)
        pot_all_subhalos_func = jax.vmap(self.single_subhalo_potential,in_axes=((0,0,0,None)))
        pot_values = pot_all_subhalos_func(relaive_position,self.m,self.a,t)
        return jnp.sum(pot_values,axis=0)


class SubhaloLinePotential(Potential):
    def __init__(self, m, a, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=None):
        super().__init__(units, {'m': m, 'a': a, 'subhalo_x0': subhalo_x0, 'subhalo_v': subhalo_v, 'subhalo_t0':subhalo_t0, 't_window':t_window})
    

    @partial(jax.jit,static_argnums=(0,))
    def single_subhalo_potential(self, xyz, m, a, t):
        return PlummerPotential(m=m, r_s=a,units=usys).potential(xyz,t) ##Was NFWPotential

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position, m,a,t)#pot_all_subhalos_func(relative_position,self.m,self.a,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, t)#jax.lax.cond(pred, true_func, false_func)
        return jnp.sum(pot_per_subhalo)

    @partial(jax.jit,static_argnums=(0,))
    def potential_per_SH(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position, m,a,t)#pot_all_subhalos_func(relative_position,self.m,self.a,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, t)#jax.lax.cond(pred, true_func, false_func)
        return pot_per_subhalo



class SubhaloLinePotential_Radius(Potential):
    def __init__(self, m, a, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=None):
        super().__init__(units, {'m': m, 'a': a, 'subhalo_x0': subhalo_x0, 'subhalo_v': subhalo_v, 'subhalo_t0':subhalo_t0, 't_window':t_window})
    """
    Strategy is to use d/dtheta( dphi/dx )  = d/dx( dphi / dtheta ). Assuming theta is 1d, dphi/dtheta is a new potential that we will use to obtain
    the radius corrections to the EOM.
    For many structural parameters (multivariate) need to take a jacobian, but the same principle will apply.
    """

    @partial(jax.jit,static_argnums=(0,))
    def single_subhalo_potential(self, xyz, m, a, t):
        func = lambda m, r_s: PlummerPotential(m=m, r_s=r_s,units=usys).potential(xyz,t)
        return jax.grad(func,argnums=(1))(m, a) # returns gradient of potential with respect to scale radius. Output is still a 1d potential evaluation (scalar)

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position, m,a,t)#pot_all_subhalos_func(relative_position,self.m,self.a,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, t)#jax.lax.cond(pred, true_func, false_func)
        return jnp.sum(pot_per_subhalo)

    @partial(jax.jit,static_argnums=(0,))
    def potential_per_SH(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position, m,a,t)#pot_all_subhalos_func(relative_position,self.m,self.a,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, t)#jax.lax.cond(pred, true_func, false_func)
        return pot_per_subhalo

    

class SubhaloLinePotential_Impact(Potential):
    def __init__(self, m, a, subhalo_x0, subhalo_v, subhalo_t0, t_window, bImpact_hat, units=None):
        super().__init__(units, {'m': m, 'a': a, 'subhalo_x0': subhalo_x0, 'subhalo_v': subhalo_v, 'subhalo_t0':subhalo_t0, 
        't_window':t_window,'bImpact_hat':bImpact_hat})
    """
    Strategy is to use d/dtheta( dphi/dx )  = d/dx( dphi / dtheta ). Assuming theta is 1d, dphi/dtheta is a new potential that we will use to obtain
    the radius corrections to the EOM.
    For many structural parameters (multivariate) need to take a jacobian, but the same principle will apply.

    Here we only vary the impact parameter, so d/db( dphi / dX) = d/dX (dphi/db)
    """

    @partial(jax.jit,static_argnums=(0,))
    def single_subhalo_potential(self, xyz, m, a, bImpact_hat, t):
        grad_pot = NFWPotential(m=m, r_s=a,units=usys).gradient(xyz,t)
        return -jnp.matmul(grad_pot,bImpact_hat)

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a,bImpact_hat, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position, m,a,bImpact_hat, t)#pot_all_subhalos_func(relative_position,self.m,self.a,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, bImpact_hat, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, self.bImpact_hat,t)#jax.lax.cond(pred, true_func, false_func)
        return jnp.sum(pot_per_subhalo)

    @partial(jax.jit,static_argnums=(0,))
    def potential_per_SH(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, bImpact_hat, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position, m,a, bImpact_hat,t)#pot_all_subhalos_func(relative_position,self.m,self.a,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a,bImpact_hat,  t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, self.bImpact_hat,t)#jax.lax.cond(pred, true_func, false_func)
        return pot_per_subhalo


class SubhaloLinePotential_Velocity(Potential):
    def __init__(self, m, a, subhalo_x0, subhalo_v, subhalo_t0, t_window, units=None):
        super().__init__(units, {'m': m, 'a': a, 'subhalo_x0': subhalo_x0, 'subhalo_v': subhalo_v, 'subhalo_t0':subhalo_t0, 
        't_window':t_window})
        self.v_hat = subhalo_v / jnp.sqrt(jnp.sum(subhalo_v**2,axis=1))[:,None]
    """
    Strategy is to use d/dtheta( dphi/dx )  = d/dx( dphi / dtheta ). Assuming theta is 1d, dphi/dtheta is a new potential that we will use to obtain
    the radius corrections to the EOM.
    For many structural parameters (multivariate) need to take a jacobian, but the same principle will apply.

    Here we only vary the impact parameter, so d/db( dphi / dX) = d/dX (dphi/db)
    """

    @partial(jax.jit,static_argnums=(0,))
    def single_subhalo_potential(self, xyz, m, a, v_hat, t):
        grad_pot = NFWPotential(m=m, r_s=a,units=usys).gradient(xyz,t)
        return -jnp.matmul(grad_pot,v_hat)

    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a,v_hat, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position, m,a,v_hat, t)*(t-subhalo_t0)#pot_all_subhalos_func(relative_position,self.m,self.a,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, v_hat, t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, self.v_hat,t)#jax.lax.cond(pred, true_func, false_func)
        return jnp.sum(pot_per_subhalo)

    @partial(jax.jit,static_argnums=(0,))
    def potential_per_SH(self,xyz, t):
        """
        xyz is where we want to evalaute the potential due to the ensemble of subhalos
        t is evaluation time.
        """
        def true_func(subhalo_x0, subhalo_v, subhalo_t0, m, a, v_hat, t):
            relative_position = xyz - (subhalo_x0 + subhalo_v*(t - subhalo_t0))
            pot_values = self.single_subhalo_potential(relative_position, m,a, v_hat,t)*(t-subhalo_t0)#pot_all_subhalos_func(relative_position,self.m,self.a,t)
            return pot_values
        
        def false_func(subhalo_x0, subhalo_v, subhalo_t0, m, a,v_hat,  t):
            return jnp.array(0.0)

        pred = jnp.abs(t - self.subhalo_t0) < self.t_window # True if in window, false otherwise

        vmapped_cond = jax.vmap(jax.lax.cond,in_axes=((0,None,None,0,0,0,0,0,0,None)))
        pot_per_subhalo = vmapped_cond(pred,true_func,false_func, self.subhalo_x0, self.subhalo_v, self.subhalo_t0, self.m, self.a, self.v_hat,t)#jax.lax.cond(pred, true_func, false_func)
        return pot_per_subhalo




    
class SubhaloIntegratorPotential(Potential):
    def __init__(self, m, a, subhalo_w0, ext_pot, units=None):
        super().__init__(units, {'m': m, 'a': a, 'subhalo_w0': subhalo_w0, 'ext_pot': ext_pot})
        self.dct = {'m': self.m, 'a': self.a,}    
    
    @partial(jax.jit,static_argnums=(0,))
    def multiple_particle_integrator(self,w0,t):
        return jax.vmap(self.ext_pot.orbit_integrator_run,in_axes=((0,None,None,None)))(w0,0,t,None)[:,0,:]


    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t):
        subhalo_locations = self.multiple_particle_integrator(self.subhalo_w0,t)[:,:3]
        delta_position = xyz - subhalo_locations
        pot_total = jnp.sum( jax.vmap(single_subhalo_potential,in_axes=( ({'m':0,'a':0,},0,None) ) )(self.dct,delta_position,t) ) 
        return pot_total




    
class Isochrone(Potential):
    
    def __init__(self, m, a, units=None):
        super().__init__(units, {'m': m, 'a': a})
    
    @partial(jax.jit, static_argnums=(0,))
    def potential(self, xyz, t):
        r = jnp.linalg.norm(xyz, axis=0)
        return - self._G * self.m / (self.a + jnp.sqrt(r**2 + self.a**2))
    

    
class Isochrone_centered(Potential):
    
    def __init__(self, m, a, spline_eval_func, splines, t_min, t_max, m_ext, a_ext, units=None):
        super().__init__(units, {'m': m, 'a': a, 'spline_eval_func': spline_eval_func, 'splines': splines, 't_min': t_min, 't_max': t_max,
                                'm_ext': m_ext, 'a_ext': a_ext})
    
    @partial(jax.jit, static_argnums=(0,))
    def potential(self, xyz, t):
        is_cond_met = (t > self.t_min) & (t < self.t_max) # True if yes, False if no
        pot_ext = Isochrone(m=self.m_ext, a=self.a_ext, units=self.units) 
        
        def true_func(xyz_t):
            xyz_, t = xyz_t[:3], xyz_t[-1]
            xyz = xyz_ - self.spline_eval_func(t,self.splines)
            r = jnp.linalg.norm(xyz, axis=0)
            return - self._G * self.m / (self.a + jnp.sqrt(r**2 + self.a**2))  + pot_ext.potential(xyz_,t)#+ self.pot_ext.potential(xyz_,t)
        def false_func(xyz_t):
            xyz, t = xyz_t[:3], xyz_t[-1]
            return pot_ext.potential(xyz,t)#0.#self.pot_ext.potential(xyz,t)
        xyz_t = jnp.hstack([xyz,t])
        return jax.lax.cond(pred=is_cond_met, true_fun=true_func, false_fun=false_func,operand=xyz_t)
    
class MiyamotoNagaiDisk(Potential):
    def __init__(self, m, a, b, units=None):
        super().__init__(units, {'m': m, 'a': a, 'b': b,})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        R2 = xyz[0]**2 + xyz[1]**2
        return -self._G*self.m / jnp.sqrt(R2 + jnp.square(jnp.sqrt(xyz[2]**2 + self.b**2) + self.a))
    
class NFWPotential_holder(Potential):
    """
    Flattening in potential, not density
    Form from http://gala.adrian.pw/en/v0.1.2/api/gala.potential.FlattenedNFWPotential.html
    """
    def __init__(self, v_c, r_s, q, units=None):
        super().__init__(units, {'v_c': v_c, 'r_s': r_s, 'q': q})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        m = jnp.sqrt(xyz[0]**2 + xyz[1]**2 + (xyz[2]/self.q)**2)
        return -((self.v_c**2)/jnp.sqrt(jnp.log(2.0)-0.5) )*jnp.log(1.0 + m/self.r_s)/(m/self.r_s)
    
class NFWPotential(Potential):
    """
    standard def see spherical model @ https://github.com/adrn/gala/blob/main/gala/potential/potential/builtin/builtin_potentials.c
    """
    def __init__(self, m, r_s, units=None):
        super().__init__(units, {'m': m, 'r_s': r_s})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        """
        CHANGED TO PLUMMER FOR TESTING PURPOSES!!
        """
        v_h2 = -self._G*self.m/self.r_s
        m = jnp.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2 + .01)/self.r_s ##added softening! used to be 0.01
        return v_h2*jnp.log(1.0+ m) / m
        

class PlummerPotential(Potential):
    def __init__(self, m, r_s, units=None):
        super().__init__(units, {'m': m, 'r_s': r_s})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        r_squared = xyz[0]**2 + xyz[1]**2 + xyz[2]**2
        return -self._G*self.m / jnp.sqrt(r_squared + self.r_s**2)


        
    
class BarPotential(Potential):
    """
    Rotating bar potentil, with hard-coded rotation.
    Eq 8a in https://articles.adsabs.harvard.edu/pdf/1992ApJ...397...44L
    Rz according to https://en.wikipedia.org/wiki/Rotation_matrix
    """
    def __init__(self, m, a, b, c, Omega, units=None):
        super().__init__(units, {'m': m, 'a': a, 'b': b, 'c': c, 'Omega': Omega})
    @partial(jax.jit,static_argnums=(0,))
    def potential(self,xyz,t):
        ## First take the simulation frame coordinates and rotate them by Omega*t
        ang = -self.Omega*t
        Rot_mat = jnp.array([[jnp.cos(ang), -jnp.sin(ang), 0], [jnp.sin(ang), jnp.cos(ang), 0.], [0.0, 0.0, 1.0] ])
        Rot_inv = jnp.linalg.inv(Rot_mat)
        xyz_corot = jnp.matmul(Rot_mat,xyz)
        
        T_plus = jnp.sqrt( (self.a + xyz_corot[0])**2 + xyz_corot[1]**2 + ( self.b + jnp.sqrt(self.c**2 + xyz_corot[2]**2) )**2 )
        T_minus = jnp.sqrt( (self.a - xyz_corot[0])**2 + xyz_corot[1]**2 + ( self.b + jnp.sqrt(self.c**2 + xyz_corot[2]**2) )**2 )
        
        pot_corot_frame = (self._G*self.m/(2.0*self.a))*jnp.log( (xyz_corot[0] - self.a + T_minus)/(xyz_corot[0] + self.a + T_plus) )
        return pot_corot_frame
    
    
class Potential_Combine(Potential):
    def __init__(self, potential_list, units=None):
        super().__init__(units, {'potential_list': potential_list })
        self.gradient = self.gradient_func

    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t,):
        output = []
        for i in range(len(self.potential_list)):
            output.append(self.potential_list[i].potential(xyz,t))
        return jnp.sum(jnp.array(output))

    @partial(jax.jit,static_argnums=(0,))
    def gradient_func(self, xyz, t,):
        output = []
        for i in range(len(self.potential_list)):
            output.append(self.potential_list[i].gradient(xyz,t))
        return jnp.sum( jnp.array(output), axis = 0)

class PotentialBase_and_PotentialPerturbation(Potential):
    def __init__(self, potential_base, potential_perturbation, units=None):
        super().__init__(units,{'potential_base':potential_base, 'potential_perturbation':potential_perturbation})
        self.gradient = None
        self.gradientPotentialBase = potential_base.gradient
        self.gradientPotentialPerturbation = potential_perturbation.gradient
        
    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t,):
        raise NotImplementedError

class PotentialBase_and_PotentialPerturbation_and_PotentialStructural(Potential):
    def __init__(self, potential_base, potential_perturbation, potential_structural, units=None):
        super().__init__(units,{'potential_base':potential_base, 'potential_perturbation':potential_perturbation, 'potential_structural':potential_structural})
        self.gradient = None
        self.gradientPotentialBase = potential_base.gradient
        self.gradientPotentialPerturbation = potential_perturbation.gradient
        self.gradientPotentialStructural = potential_structural.gradient

        self.gradientPotentialPerturbation_per_SH = jax.jacfwd(potential_perturbation.potential_per_SH)
        self.gradientPotentialStructural_per_SH = jax.jacfwd(potential_structural.potential_per_SH)
        
    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t,):
        raise NotImplementedError
    

class PotentialBase_and_PotentialPerturbation_and_PotentialStructural_and_PotentialImpact_and_PotentialVelocity(Potential):
    def __init__(self, potential_base, potential_perturbation, potential_structural, potential_impact, potential_velocity, units=None):
        super().__init__(units,{'potential_base':potential_base, 'potential_perturbation':potential_perturbation, 
                                'potential_structural':potential_structural,
                                'potential_impact': potential_impact,
                                'potential_velocity': potential_velocity})
        self.gradient = None
        self.gradientPotentialBase = potential_base.gradient
        self.gradientPotentialPerturbation = potential_perturbation.gradient
        self.gradientPotentialStructural = potential_structural.gradient
        self.gradientPotentialImpact = potential_impact.gradient
        self.gradientPotentialVelocity = potential_velocity.gradient

        self.gradientPotentialPerturbation_per_SH = jax.jacfwd(potential_perturbation.potential_per_SH)
        self.gradientPotentialStructural_per_SH = jax.jacfwd(potential_structural.potential_per_SH)
        self.gradientPotentialImpact_per_SH = jax.jacfwd(potential_impact.potential_per_SH)
        self.gradientPotentialVelocity_per_SH = jax.jacfwd(potential_velocity.potential_per_SH)

        
        
    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t,):
        raise NotImplementedError
        




class PotentialBase_and_PotentialPerturbation_flow(Potential):
    def __init__(self, potential_base, potential_perturbation, times,positions,units=None):
        super().__init__(units,{'potential_base':potential_base, 'potential_perturbation':potential_perturbation,'times':times,
                               'positions':positions})
        self.gradient = None
        self.gradientPotentialBase = potential_base.gradient
        self.gradientPotentialPerturbation = potential_perturbation.gradient
        self.spl_x = InterpolatedUnivariateSpline(self.times, self.positions[:,0],k=3)
        self.spl_y = InterpolatedUnivariateSpline(self.times, self.positions[:,1],k=3)
        self.spl_z = InterpolatedUnivariateSpline(self.times, self.positions[:,2],k=3)
        
        self.spl_vx = InterpolatedUnivariateSpline(self.times, self.positions[:,3],k=3)
        self.spl_vy = InterpolatedUnivariateSpline(self.times, self.positions[:,4],k=3)
        self.spl_vz = InterpolatedUnivariateSpline(self.times, self.positions[:,5],k=3)
        
    @partial(jax.jit,static_argnums=(0,))
    def potential(self, xyz, t,):
        raise NotImplementedError
    
    
def leapfrog_step(func, y0, t0, dt, a0):
    ndim = y0.shape[0] // 2
    tf = t0 + dt
    
    x0 = y0[:ndim]
    v0 = y0[ndim:]
    
    v1_2 = v0 + a0 * dt / 2.
    xf = x0 + v1_2 * dt
    af = - func(xf, tf)
    
    vf = v1_2 + af * dt / 2
    
    return tf, jnp.concatenate((xf, vf)), af

@partial(jax.jit, static_argnames=['potential_gradient', 'args'])
def leapfrog_run(w0, ts, potential_gradient, args=()):
    func_ = lambda y, t: potential_gradient(y, t, *args)
    
    def scan_fun(carry, t):
        i, y0, t0, dt, a0 = carry
        tf, yf, af = leapfrog_step(func_, y0, t0, dt, a0)
        dt_new = ts[i+1] - ts[i]
        is_cond_met = jnp.abs(dt_new) > 0.  ### !!! ADDED jnp.abs AFTER derivs worked. Note for future debugging efforts!
        
        def true_func(dt_new):
            return ts[-1]-ts[-2] #dt_base !!!ASSUMING dt = 0.5 Myr by default!!!!
        def false_func(dt_new):
            return 0.0
        dt_new = jax.lax.cond(pred=is_cond_met, true_fun=true_func, false_fun=false_func,operand=dt_new)
        
        ###tf = tf + dt_new
        return [i + 1, yf, tf, dt_new, af], yf
    
    ndim = w0.shape[0] // 2
    a0 = -func_(w0[:ndim], ts[0]) ##### SHOULD THIS BE NEGATIVE??? TODO
    dt = ts[1]-ts[0] ## I ADDED THIS
    init_carry = [0, w0, ts[0], dt, a0]
    _, ws = jax.lax.scan(scan_fun, init_carry, ts[1:])
    res_ws = jnp.concatenate((w0[None], ws))
    
    return res_ws

    


