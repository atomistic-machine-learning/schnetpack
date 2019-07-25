"""
Collection of utilities for computing autocorrelation functions and molecular spectra
from HDF5 files generated during molecular dynamics. For a good overview on how to compute
spectra from molecular dynamics simulations and details on the techniques used, we recommend [#spectra1]_ .

References
----------
.. [#spectra1] Martin, Brehm, Fligg, Vöhringer, Kirchner:
               Computing vibrational spectra from ab initio molecular dynamics.
               Phys. Chem. Chem. Phys., 15 (18), 6608--6622. 2013.
"""
import numpy as np
from ase import units
from schnetpack.md.utils import MDUnits
from schnetpack import Properties
import logging


def cosine_sq_window(n_points):
    """
    Squared cosine window function for spectra.

    Args:
        n_points (int): Number of points in spectrum

    Returns:
        numpy.array: Squared cosine window function.
    """
    points = np.arange(n_points)
    window = np.cos(np.pi * points / (n_points - 1) / 2) ** 2
    return window


def fft_autocorrelation(data, n_lags):
    """
    Routine for fast computation of autocorrelation using FFT and Wiener--Kinchie theorem.

    Args:
        data (numpy.array): Array containing data for which autocorrelation should be computed.
        n_lags (int): Number of time lags used for extracting the autocorrelation.

    Returns:
        numpy.array: Autocorrelation function of the input array
    """
    # Normalize data to get autocorrelation[0] = 1
    data = (data - np.mean(data)) / np.std(data)
    n_points = data.shape[0]
    fft_forward = np.fft.fft(data, n=2 * n_points)
    fft_autocorr = fft_forward * np.conjugate(fft_forward)
    fft_backward = np.fft.ifft(fft_autocorr)[:n_points] / n_points
    autocorrelation = np.real(fft_backward[: n_lags + 1])
    return autocorrelation


class VibrationalSpectrum:
    """
    Base class for computing vibrational spectra from HDF5 datasets using autocorrelation functions and
    fast fourier transforms.

    Args:
        data (schnetpack.md.utils.HDF5Loader): Loaded dataset.
        resolution (int): Resolution used when computing the spectrum. Indicates how many time lags are considered
                          in the autocorrelation function is used.
        window (function, optional): Window function used for computing the spectrum.
    """

    def __init__(self, data, resolution=4096, window=cosine_sq_window):
        self.data = data
        self.timestep = data.time_step / MDUnits.fs2atu  # Convert to fs
        self.resolution = resolution
        self.window = window

        spectral_range = 0.5 / self.timestep / (units._c / 1e13)
        spectral_resolution = spectral_range / (4 * resolution)
        logging.info(
            "Spectral resolutions: {:12.3f} [cm^-1]".format(spectral_resolution)
        )
        logging.info("Spectral range:       {:12.3f} [cm^-1]".format(spectral_range))

        self.res = spectral_resolution
        self.frequencies = []
        self.intensities = []

    def compute_spectrum(self, molecule_idx=0):
        """
        Main routine for computing spectra. First the relavant data is read,
        then autocorrelations are computed and processed. Based on the
        processed autocorrelations, spectra are computed and, if requested,
        subjected to additional postprocessing.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
                                Uses the same conventions as schnetpack.md.System.
        """
        # Get appropriate data
        relevant_data = self._get_data(molecule_idx)

        # Compute autocorrelation function
        autocorrelation = self._compute_autocorrelations(relevant_data)
        # Process the autocorrelation function (e.g. weighting, Raman, ...)
        autocorrelation = self._process_autocorrelation(autocorrelation)

        self.frequencies = []
        self.intensities = []
        # Compute spectrum
        for autocorr in autocorrelation:
            frequencies, intensities = self._compute_spectrum(autocorr)
            self.frequencies.append(frequencies)
            self.intensities.append(intensities)

        self._process_spectrum()

    def _compute_spectrum(self, autocorrelation):
        """
        Compute the spectrum from the autocorrelation function.

        Args:
            autocorrelation (numpy.array): Autorcorrelation function.

        Returns:
            (numpy.array,numpy.array):
                frequencies:
                    Vibrational frequencies in inverse centimeters.
                intensities:
                    Intensities of the vibrational bands.
        """
        data = autocorrelation[: self.resolution]

        # Various tricks for nicer spectra
        # 1) Apply window function
        n_unpadded = data.shape[0]
        if self.window is not None:
            data *= self.window(n_unpadded)
        # 2) Zero padding
        data_padded = np.zeros(4 * n_unpadded)
        data_padded[:n_unpadded] = data
        # 3) Mirror around origin
        data_mirrored = np.hstack((np.flipud(data_padded), data_padded))
        # Compute the spectrum
        n_fourier = 8 * n_unpadded
        intensities = np.abs(
            self.timestep * np.fft.fft(data_mirrored, n=n_fourier)[: n_fourier // 2]
        )
        frequencies = np.arange(n_fourier // 2) / (n_fourier * self.timestep)
        # Conversion to inverse cm
        frequencies /= units._c / 1e13
        return frequencies, intensities

    @staticmethod
    def _compute_autocorrelations(data):
        """
        Compute the autocorrelation function of the data. A separate autocorrelation is computred
        for every array dimension except the first axis.

        Args:
            data (numpy.array): Function array.

        Returns:
            numpy.array: Autocorrelation of the inputs.
        """
        n_data = data.shape[0]
        data_dim = data.shape[1:]
        n_lags = n_data - 2

        # Flatten data for easier iteration
        reshaped_data = data.reshape((n_data, -1))
        n_fields = reshaped_data.shape[1]

        # Compute all individual autocorrelations
        autocorrelations = np.zeros((n_fields, n_lags + 1))
        for field in range(n_fields):
            autocorrelations[field, ...] = fft_autocorrelation(
                reshaped_data[..., field], n_lags
            )

        # Reconstruct shape of original property
        autocorrelations = autocorrelations.reshape((*data_dim, -1))
        return autocorrelations

    def _get_data(self, molecule_idx):
        """
        Placeholder for extracting teh required data from the HDF5 dataset.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
                                Uses the same conventions as schnetpack.md.System.
        """
        raise NotImplementedError

    def _process_autocorrelation(self, autocorrelation):
        """
        Placeholder for postprocessing the autocorrelation functions (e.g. weighting).

        Args:
            autocorrelation (numpy.array): Autorcorrelation function.
        """
        raise NotImplementedError

    def _process_spectrum(self):
        """
        Placeholder function if postprocessing should be applied to the spectrum (e.g. quantum coorections).
        """
        pass

    def get_spectrum(self):
        """
        Returns all computed spectra in the form of a list of tuples of frequencies and intensities.

        Returns:
            list: List of tuples of frequencies and intensities of all computed spectra.
        """
        spectrum = list(zip(self.frequencies, self.intensities))
        if len(spectrum) == 1:
            return spectrum[0]
        else:
            return spectrum


class PowerSpectrum(VibrationalSpectrum):
    """
    Compute power spectra from a molecular dynamics HDF5 dataset.

    Args:
        data (schnetpack.md.utils.HDF5Loader): Loaded dataset.
        resolution (int, optional): Resolution used when computing the spectrum. Indicates how many time lags
                                    are considered in the autocorrelation function is used.
    """

    def __init__(self, data, resolution=4096):
        super(PowerSpectrum, self).__init__(data, resolution=resolution)

    def _get_data(self, molecule_idx):
        """
        Extract the molecular velocities for computing the spectrum.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
                                Uses the same conventions as schnetpack.md.System.

        Returns:
            numpy.array: Array holding molecular velocities.
        """
        relevant_data = self.data.get_velocities(molecule_idx)
        return relevant_data

    def _process_autocorrelation(self, autocorrelation):
        """
        Sum over number of atoms and the three Cartesian components.

        Args:
            autocorrelation (numpy.array): Autorcorrelation function.

        Returns:
            numpy.array: Updated autocorrelation.
        """
        vdos_autocorrelation = np.sum(autocorrelation, axis=1)
        vdos_autocorrelation = np.mean(vdos_autocorrelation, axis=0)
        return [vdos_autocorrelation]


class IRSpectrum(VibrationalSpectrum):
    """
    Compute infrared spectra from a molecular dynamics HDF5 dataset. This class requires the dipole moments
    to be present in the HDF5 dataset.

    Args:
        data (schnetpack.md.utils.HDF5Loader): Loaded dataset.
        resolution (int, optional): Resolution used when computing the spectrum. Indicates how many time lags
                                    are considered in the autocorrelation function is used.
        dipole_moment_handle (str, optional): Indentifier used for extracting dipole data.
    """

    def __init__(
        self, data, resolution=4096, dipole_moment_handle=Properties.dipole_moment
    ):
        super(IRSpectrum, self).__init__(data, resolution=resolution)
        self.dipole_moment_handle = dipole_moment_handle

    def _get_data(self, molecule_idx):
        """
        Extract the molecular dipole moments and compute their time derivative via central
        difference.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
                                Uses the same conventions as schnetpack.md.System.

        Returns:
            numpy.array: Array holding the dipole moment derivatives.
        """
        relevant_data = self.data.get_property(self.dipole_moment_handle, molecule_idx)
        # Compute numerical derivative via central differences
        relevant_data = (relevant_data[2:, ...] - relevant_data[:-2, ...]) / (
            2 * self.timestep
        )
        return relevant_data

    def _process_autocorrelation(self, autocorrelation):
        """
        Sum over the three Cartesian components.

        Args:
            autocorrelation (numpy.array): Dipole moment flux autorcorrelation functions.

        Returns:
            numpy.array: Updated autocorrelation.
        """
        dipole_autocorrelation = np.sum(autocorrelation, axis=0)
        return [dipole_autocorrelation]
