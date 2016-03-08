from liquid_style.hmc_temp import simulate_dynamics, metropolis_hastings_accept, hamiltonian, hmc_updates
from plato.core import create_shared_variable, add_update
from plato.interfaces.helpers import get_theano_rng

__author__ = 'peter'

"""
Hamiltonian Monte Carlo

Mainly adapted from: http://deeplearning.net/tutorial/code/hmc/hmc.py

"""


class HamiltonianMonteCarlo(object):

    def __init__(self, initial_state, energy_fcn,
            initial_stepsize=0.01,
            target_acceptance_rate=.9,
            n_steps=20,
            stepsize_dec=0.98,
            stepsize_min=0.001,
            stepsize_max=0.25,
            stepsize_inc=1.02,  # used in geometric avg. 1.0 would be not moving at all
            avg_acceptance_slowness=0.9,
            rng=None):
        self.energy_fcn = energy_fcn
        self.stepsize = create_shared_variable(initial_stepsize, name = 'hmc_stepsize')
        self.n_steps = n_steps
        self.avg_acceptance_rate = create_shared_variable(target_acceptance_rate, name = 'avg_acceptance_rate')
        self.target_acceptance_rate = target_acceptance_rate
        self.state = create_shared_variable(initial_state)
        self.stepsize_dec = stepsize_dec
        self.stepsize_min = stepsize_min
        self.stepsize_max = stepsize_max
        self.stepsize_inx = stepsize_inc
        self.avg_acceptance_slowness = avg_acceptance_slowness
        self.rng = get_theano_rng(rng)

    def get_new_state(self):

        initial_vel = self.rng.normal(size = self.state.ishape)
        final_pos, final_vel = simulate_dynamics(
            initial_pos = self.state,
            initial_vel = initial_vel,
            stepsize = self.step_size,
            n_steps = self.n_steps,
            energy_fn = self.energy_fcn
        )

        accept = metropolis_hastings_accept(
            energy_prev=hamiltonian(self.initial_state, initial_vel, self.energy_fn),
            energy_next=hamiltonian(final_pos, final_vel, self.energy_fn),
            s_rng=self.rng
            )

        simulate_updates = hmc_updates(
            positions=self.state,
            stepsize=self.stepsize,
            avg_acceptance_rate=self.avg_acceptance_rate,
            final_pos=final_pos,
            accept=accept,
            stepsize_min=self.stepsize_min,
            stepsize_max=self.stepsize_max,
            stepsize_inc=self.stepsize_inc,
            stepsize_dec=self.stepsize_dec,
            target_acceptance_rate=self.target_acceptance_rate,
            avg_acceptance_slowness=self.avg_acceptance_slowness)

        for shared_var, new_val in simulate_updates.iteritems():
            add_update(shared_var, new_val)

        return simulate_updates[self.state]
