import numpy as np
from ase import units
from schnetpack.md.utils import MDUnits
import logging


def cosine_sq_window(n_points):
    """
    Squared cosine window function for spectra
    """
    points = np.arange(n_points)
    window = np.cos(np.pi * points / (n_points - 1) / 2) ** 2
    return window


def fft_autocorrelation(data, n_lags):
    """
    Computation of autocorrelation using FFT and Wiener--Kinchie theorem.
    """
    # Normalize data to get autocorrelation[0] = 1
    data = (data - np.mean(data)) / np.std(data)
    n_points = data.shape[0]
    fft_forward = np.fft.fft(data, n=2 * n_points)
    fft_autocorr = fft_forward * np.conjugate(fft_forward)
    fft_backward = np.fft.ifft(fft_autocorr)[:n_points] / n_points
    autocorrelation = np.real(fft_backward[: n_lags + 1])
    return autocorrelation


def lorentz_convolution(data, start, stop, n_points, width):
    offset = np.linspace(start, stop, n_points)
    spectrum = 1 / (1 + ((data[:, None] - offset[None, :]) / width) ** 2)
    return offset, np.sum(spectrum, axis=0)


class VibrationalSpectrum:
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
        # Get appropriate data
        relevant_data = self._get_data(molecule_idx)

        # Compute autocorrelation function
        autocorrelation = self._compute_autocorrelations(relevant_data)
        # Proccess the autocorrelation function (e.g. weighting, Raman, ...)
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
        Compute the spectrum. Timestep needs to be given in fs
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
        """Compute general autocorrelations"""
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
        raise NotImplementedError

    def _process_autocorrelation(self, autocorrelation):
        raise NotImplementedError

    def _process_spectrum(self):
        pass

    def get_spectrum(self):
        spectrum = list(zip(self.frequencies, self.intensities))
        if len(spectrum) == 1:
            return spectrum[0]
        else:
            return spectrum


class PowerSpectrum(VibrationalSpectrum):
    def __init__(self, data, resolution=4096):
        super(PowerSpectrum, self).__init__(data, resolution=resolution)

    def _get_data(self, molecule_idx):
        relevant_data = self.data.get_velocities(molecule_idx)
        return relevant_data

    def _process_autocorrelation(self, autocorrelation):
        vdos_autocorrelation = np.sum(autocorrelation, axis=1)
        vdos_autocorrelation = np.mean(vdos_autocorrelation, axis=0)
        return [vdos_autocorrelation]


class IRSpectrum(VibrationalSpectrum):
    def __init__(self, data, resolution=4096):
        super(IRSpectrum, self).__init__(data, resolution=resolution)

    def _get_data(self, molecule_idx):
        relevant_data = self.data.get_property(molecule_idx, "dipole_moment")
        # Compute numerical derivative via central differences
        relevant_data = (relevant_data[2:, ...] - relevant_data[:-2, ...]) / (
            2 * self.timestep
        )
        return relevant_data

    def _process_autocorrelation(self, autocorrelation):
        dipole_autocorrelation = np.sum(autocorrelation, axis=0)
        return [dipole_autocorrelation]
