from abc import ABC, abstractmethod

import numpy as np
import jittor as jt

def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "lossaware":
        return LossSecondMomentResampler(diffusion)
    elif name == "fixstep":
        return FixSampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p: jt.Var = w / jt.sum(w) 
        _p = p.numpy()
        indices_np = np.random.choice(len(_p), size=(batch_size,), p=_p)
        indices = jt.Var(indices_np).long()
        weights_np = 1 / (len(p) * p[indices_np])
        weights = jt.Var(weights_np).float()
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = jt.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

class FixSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion

        ###############################################################
        ### You can custome your own sampling weight of steps here. ###
        ###############################################################
        # self._weights = n_p.concatenate([n_p.ones([diffusion.num_timesteps//2]), n_p.zeros([diffusion.num_timesteps//2]) + 0.5])
        self._weights = jt.concat([jt.ones([diffusion.num_timesteps//2]), jt.zeros(diffusion.num_timesteps//2) + 0.5])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [0]
        max_bs = max(batch_sizes)
        timestep_batches = [jt.zeros(max_bs) for bs in batch_sizes]
        loss_batches = [jt.zeros(max_bs) for bs in batch_sizes]
        timesteps = [
            x for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = jt.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=jt.float64
        )
        self._loss_counts = jt.zeros([diffusion.num_timesteps], dtype=jt.int)

    def weights(self):
        if not self._warmed_up():
            return jt.ones([self.diffusion.num_timesteps], dtype=jt.float64)
        weights = jt.sqrt(jt.mean(self._loss_history ** 2, axis=-1))
        weights /= jt.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all().item()
