__author__ = 'peter'

"""
Hamiltonian Monte Carlo

Mainly adapted from: http://deeplearning.net/tutorial/code/hmc/hmc.py

"""


class HamiltonianMonteCarlo(object):

    def __init__(self, initial_state, energy_fcn, initial_step_size, step)
        self.energy_fcn = energy_fcn
        self.step_size = initial_step_size


    def get_new_state(self):

        pass




def kinetic_energy(vel):
    """
    :param vel: A (n_samples, n_dims) array of velocity vectors
    :return: The Kinetic Energy Term - (a vector of energies for each sample)
    """
    return 0.5 * (vel ** 2).sum(axis=1)


def hamiltonian(pos, vel, energy_fn):
    """

    :param pos: A (n_samples, n_dims) array of position vectors
    :param vel: A (n_samples, n_dims) array of velocity vectors
    :param energy_fn: The energy function
    :return: The Hamtonian (a vector of energies for each sample)
    """
    return energy_fn(pos) + kinetic_energy(vel)


def metropolis_hastings_accept(energy_prev, energy_next, s_rng):
    """
    Performs a Metropolis-Hastings accept-reject move.

    Parameters
    ----------
    energy_prev: theano vector
        Symbolic theano tensor which contains the energy associated with the
        configuration at time-step t.
    energy_next: theano vector
        Symbolic theano tensor which contains the energy associated with the
        proposed configuration at time-step t+1.
    s_rng: theano.tensor.shared_randomstreams.RandomStreams
        Theano shared random stream object used to generate the random number
        used in proposal.

    Returns
    -------
    return: boolean
        True if move is accepted, False otherwise
    """
    ediff = energy_prev - energy_next
    return (TT.exp(ediff) - s_rng.uniform(size=energy_prev.shape)) >= 0
