import logging
import torch


logger = logging.getLogger(__name__)


class StatisticsAccumulator:
    def __init__(self, batch=False, atomistic=False):
        """
        Use the incremental Welford algorithm described in [1]_ to accumulate
        the mean and standard deviation over a set of samples.

        Args:
            batch: If set to true, assumes sample is batch and uses leading
                   dimension as batch size
            atomistic: If set to true, average over atom dimension

        References:
        -----------
        .. [1] https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        """
        # Initialize state variables
        self.count = 0  # Sample count
        self.mean = 0  # Incremental average
        self.M2 = 0  # Sum of squares of differences
        self.batch = batch
        self.atomistic = atomistic

    def add_sample(self, sample_value):
        """
        Add a sample to the accumulator and update running estimators.
        Differentiates between different types of samples.

        Args:
            sample_value (torch.Tensor): data sample
        """

        # Check different cases
        if not self.batch and not self.atomistic:
            self._add_sample(sample_value)
        elif not self.batch and self.atomistic:
            n_atoms = sample_value.size(0)
            for i in range(n_atoms):
                self._add_sample(sample_value[i, :])
        elif self.batch and not self.atomistic:
            n_batch = sample_value.size(0)
            for i in range(n_batch):
                self._add_sample(sample_value[i, :])
        else:
            n_batch = sample_value.shape[0]
            n_atoms = sample_value.shape[1]
            for i in range(n_batch):
                for j in range(n_atoms):
                    self._add_sample(sample_value[i, j, :])

    def _add_sample(self, sample_value):
        # Update count
        self.count += 1
        delta_old = sample_value - self.mean
        # Difference to old mean
        self.mean += delta_old / self.count
        # Update mean estimate
        delta_new = sample_value - self.mean
        # Update sum of differences
        self.M2 += delta_old * delta_new

    def get_statistics(self):
        """
        Compute statistics of all data collected by the accumulator.

        Returns:
            torch.Tensor: Mean of data
            torch.Tensor: Standard deviation of data
        """
        # Compute standard deviation from M2
        mean = self.mean
        stddev = torch.sqrt(self.M2 / self.count)

        return mean, stddev

    def get_mean(self):
        return self.mean

    def get_stddev(self):
        return torch.sqrt(self.M2 / self.count)
