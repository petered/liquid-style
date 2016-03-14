from general.numpy_helpers import get_rng
from liquid_style.hmc_temp import simulate_dynamics, metropolis_hastings_accept, hamiltonian, hmc_updates
from plato.core import create_shared_variable, add_update, symbolic, symbolic_stateless, symbolic_updater
from plato.interfaces.helpers import get_theano_rng
import theano
import theano.tensor as tt
__author__ = 'peter'

"""
Hamiltonian Monte Carlo

Mainly adapted from: http://deeplearning.net/tutorial/code/hmc/hmc.py

"""


# class HamiltonianMonteCarlo(object):
#
#     def __init__(self, initial_state, energy_fcn,
#             initial_stepsize=0.01,
#             target_acceptance_rate=.9,
#             n_steps=20,
#             stepsize_dec=0.98,
#             stepsize_min=0.001,
#             stepsize_max=0.25,
#             stepsize_inc=1.02,  # used in geometric avg. 1.0 would be not moving at all
#             avg_acceptance_slowness=0.9,
#             rng=None):
#         self.energy_fcn = energy_fcn
#         self.stepsize = create_shared_variable(initial_stepsize, name = 'hmc_stepsize')
#         self.n_steps = n_steps
#         self.avg_acceptance_rate = create_shared_variable(target_acceptance_rate, name = 'avg_acceptance_rate')
#         self.target_acceptance_rate = target_acceptance_rate
#         self.state = create_shared_variable(initial_state)
#         self.stepsize_dec = stepsize_dec
#         self.stepsize_min = stepsize_min
#         self.stepsize_max = stepsize_max
#         self.stepsize_inc = stepsize_inc
#         self.avg_acceptance_slowness = avg_acceptance_slowness
#         self.rng = get_theano_rng(rng)
#
#     @symbolic
#     def update_state(self):
#
#         initial_vel = self.rng.normal(size = self.state.ishape)
#         final_pos, final_vel = simulate_dynamics(
#             initial_pos = self.state,
#             initial_vel = initial_vel,
#             stepsize = self.stepsize,
#             n_steps = self.n_steps,
#             energy_fn = self.energy_fcn
#             )
#
#         accept = metropolis_hastings_accept(
#             energy_prev=hamiltonian(self.state, initial_vel, self.energy_fcn),
#             energy_next=hamiltonian(final_pos, final_vel, self.energy_fcn),
#             s_rng=self.rng
#             )
#
#         [(positions, new_positions), (stepsize, new_stepsize), (avg_acceptance_rate, new_acceptance_rate)] = hmc_updates(
#             positions=self.state,
#             stepsize=self.stepsize,
#             avg_acceptance_rate=self.avg_acceptance_rate,
#             final_pos=final_pos,
#             accept=accept,
#             stepsize_min=self.stepsize_min,
#             stepsize_max=self.stepsize_max,
#             stepsize_inc=self.stepsize_inc,
#             stepsize_dec=self.stepsize_dec,
#             target_acceptance_rate=self.target_acceptance_rate,
#             avg_acceptance_slowness=self.avg_acceptance_slowness)
#
#         add_update(positions, new_positions)
#         add_update(stepsize, new_stepsize)
#         add_update(avg_acceptance_rate, new_acceptance_rate)
#
#         return new_positions




class HMCStepSizeUpdater(object):
    """
    Updates the step size for the HMC search according to recent rejection rates.
    """

    def __init__(self,
            stepsize_dec=0.98,
            stepsize_min=0.001,
            stepsize_max=0.25,
            stepsize_inc=1.02,  # used in geometric avg. 1.0 would be not moving at all
            avg_acceptance_slowness=0.9,
            target_acceptance_rate=.9,
            ):
        self.stepsize_dec = stepsize_dec
        self.stepsize_min = stepsize_min
        self.stepsize_max = stepsize_max
        self.stepsize_inc = stepsize_inc
        self.avg_acceptance_slowness = avg_acceptance_slowness
        self.target_acceptance_rate = target_acceptance_rate
        self.avg_acceptance_rate = create_shared_variable(target_acceptance_rate, name = 'avg_acceptance_rate')

    @symbolic_updater
    def __call__(self, stepsize, accept_rate):
        """
        stepsize: A  scalar shared-variable indicating the step size to take in HMC search
        accept_rate: A scalar indicating the mean acceptance rate of the last batch (Should be between 0 and 1)

        This function updates the value of the step-size.
        """
        _new_stepsize = tt.switch(self.avg_acceptance_rate > self.target_acceptance_rate,
                              stepsize * self.stepsize_inc, stepsize * self.stepsize_dec)
        # maintain stepsize in [stepsize_min, stepsize_max]
        new_stepsize = tt.clip(_new_stepsize, self.stepsize_min, self.stepsize_max)

        # end-snippet-7 start-snippet-6
        ## ACCEPT RATE UPDATES ##
        # perform exponential moving average
        mean_dtype = theano.scalar.upcast(accept_rate.dtype, self.avg_acceptance_rate.dtype)
        new_acceptance_rate = self.avg_acceptance_slowness*self.avg_acceptance_rate + (1.0-self.avg_acceptance_slowness)*accept_rate

        add_update(self.avg_acceptance_rate, new_acceptance_rate.astype(theano.config.floatX))
        add_update(stepsize, new_stepsize)
        # # end-snippet-6 start-snippet-8
        # return [(positions, new_positions),
        #         (stepsize, new_stepsize),
        #         (avg_acceptance_rate, new_acceptance_rate)]



class HamiltonianMonteCarlo(object):

    def __init__(self,
            initial_state,
            energy_fcn,
            initial_stepsize=0.01,
            n_steps = 20,
            step_size_updater = HMCStepSizeUpdater(),
            alpha = 0,
            rng=None):
        """
        initial_state: A shared (n_samples, ...) tensor
        energy_fcn: A function that takes in the state tensor and returns a vector of energies (per-sample)
        initial_stepsize: Initial stepsize for the leapfrog algorithm
        n_steps: Number of leapfrog steps to take in a single simulation
        step_size_updater: An HMCStepSizeUpdater object that defines the rules for step-size adaptation.
        alpha: For partial momentum refreshment (in range (-1, 1), 0 means regular HMC with full momentum refreshment)
        """
        self.energy_fcn = energy_fcn
        self.stepsize = create_shared_variable(initial_stepsize, name = 'hmc_stepsize')
        self.n_steps = n_steps
        self.position = create_shared_variable(initial_state)
        self.step_size_updater = step_size_updater
        self.alpha = alpha
        self.rng = get_theano_rng(rng)
        self.np_rng = get_rng(rng)

    @symbolic
    def update_state(self):

        if self.alpha == 0:
            velocity = self.rng.normal(size = self.position.ishape)  # This is resetting every time, right?
        else:  # Partial Momentum refreshment
            velocity_shared = create_shared_variable(self.np_rng.normal(size = self.position.ishape))
            velocity = self.alpha*velocity_shared + tt.sqrt(1.-self.alpha**2)*self.rng.normal(size = self.position.ishape)
            add_update(velocity_shared, velocity)

        final_position, final_vel = simulate_dynamics(
            initial_pos = self.position,
            initial_vel = velocity,
            stepsize = self.stepsize,
            n_steps = self.n_steps,
            energy_fn = self.energy_fcn
            )
        accept = metropolis_hastings_accept(
            energy_prev=hamiltonian(self.position, velocity, self.energy_fcn),
            energy_next=hamiltonian(final_position, final_vel, self.energy_fcn),
            s_rng=self.rng
            )
        new_position = tt.switch(accept.dimshuffle(0, *(('x',) * (final_position.ndim - 1))), final_position, self.position)
        add_update(self.position, new_position)
        self.step_size_updater(stepsize = self.stepsize, accept_rate = accept.mean())
        return new_position




def hmc_updates(positions, stepsize, avg_acceptance_rate, final_pos, accept,
                target_acceptance_rate, stepsize_inc, stepsize_dec,
                stepsize_min, stepsize_max, avg_acceptance_slowness):
    """This function is executed after `n_steps` of HMC sampling
    (`hmc_move` function). It creates the updates dictionary used by
    the `simulate` function. It takes care of updating: the position
    (if the move is accepted), the stepsize (to track a given target
    acceptance rate) and the average acceptance rate (computed as a
    moving average).

    Parameters
    ----------
    positions: shared variable, theano matrix
        Shared theano matrix whose rows contain the old position
    stepsize: shared variable, theano scalar
        Shared theano scalar containing current step size
    avg_acceptance_rate: shared variable, theano scalar
        Shared theano scalar containing the current average acceptance rate
    final_pos: shared variable, theano matrix
        Shared theano matrix whose rows contain the new position
    accept: theano scalar
        Boolean-type variable representing whether or not the proposed HMC move
        should be accepted or not.
    target_acceptance_rate: float
        The stepsize is modified in order to track this target acceptance rate.
    stepsize_inc: float
        Amount by which to increment stepsize when acceptance rate is too high.
    stepsize_dec: float
        Amount by which to decrement stepsize when acceptance rate is too low.
    stepsize_min: float
        Lower-bound on `stepsize`.
    stepsize_min: float
        Upper-bound on `stepsize`.
    avg_acceptance_slowness: float
        Average acceptance rate is computed as an exponential moving average.
        (1-avg_acceptance_slowness) is the weight given to the newest
        observation.

    Returns
    -------
    rval1: dictionary-like
        A dictionary of updates to be used by the `HMC_Sampler.simulate`
        function.  The updates target the position, stepsize and average
        acceptance rate.

    """

    ## POSITION UPDATES ##
    # broadcast `accept` scalar to tensor with the same dimensions as
    # final_pos.
    accept_matrix = accept.dimshuffle(0, *(('x',) * (final_pos.ndim - 1)))
    # if accept is True, update to `final_pos` else stay put
    new_positions = TT.switch(accept_matrix, final_pos, positions)
    # end-snippet-5 start-snippet-7
    ## STEPSIZE UPDATES ##
    # if acceptance rate is too low, our sampler is too "noisy" and we reduce
    # the stepsize. If it is too high, our sampler is too conservative, we can
    # get away with a larger stepsize (resulting in better mixing).
    _new_stepsize = TT.switch(avg_acceptance_rate > target_acceptance_rate,
                              stepsize * stepsize_inc, stepsize * stepsize_dec)
    # maintain stepsize in [stepsize_min, stepsize_max]
    new_stepsize = TT.clip(_new_stepsize, stepsize_min, stepsize_max)

    # end-snippet-7 start-snippet-6
    ## ACCEPT RATE UPDATES ##
    # perform exponential moving average
    mean_dtype = theano.scalar.upcast(accept.dtype, avg_acceptance_rate.dtype)
    new_acceptance_rate = TT.add(
        avg_acceptance_slowness * avg_acceptance_rate,
        (1.0 - avg_acceptance_slowness) * accept.mean(dtype=mean_dtype))
    # end-snippet-6 start-snippet-8
    return [(positions, new_positions),
            (stepsize, new_stepsize),
            (avg_acceptance_rate, new_acceptance_rate)]
    # end-snippet-8



#
# class HMCPartial(object):
#
#     def __init__(self):