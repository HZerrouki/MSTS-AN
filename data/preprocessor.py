"""
EEG Preprocessing Pipeline for MSTS-AN.

This module implements the preprocessing pipeline described in the paper:
1. Bandpass Filtering (0.5-45 Hz)
2. ICA Artifact Removal
3. Wavelet Decomposition (Daubechies 4)
4. Segmentation into 4-second epochs
"""

import numpy as np
import mne
import pywt
from scipy import signal
from typing import Tuple, List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """
    EEG Preprocessing Pipeline.

    Implements the four-step preprocessing from the paper:
    1. FIR Bandpass Filtering
    2. ICA Artifact Removal
    3. Multi-scale Wavelet Decomposition
    4. Segmentation

    Args:
        sampling_rate: EEG sampling rate in Hz (default: 256)
        filter_low: Low cutoff frequency in Hz (default: 0.5)
        filter_high: High cutoff frequency in Hz (default: 45)
        filter_order: FIR filter order (default: 512)
        use_ica: Whether to apply ICA artifact removal (default: True)
        wavelet: Wavelet type for decomposition (default: 'db4')
        decomposition_level: Number of wavelet decomposition levels (default: 4)
        segment_length: Segment length in seconds (default: 4)
    """

    def __init__(
        self,
        sampling_rate: int = 256,
        filter_low: float = 0.5,
        filter_high: float = 45.0,
        filter_order: int = 512,
        use_ica: bool = True,
        wavelet: str = 'db4',
        decomposition_level: int = 4,
        segment_length: int = 4
    ):
        self.sampling_rate = sampling_rate
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_order = filter_order
        self.use_ica = use_ica
        self.wavelet = wavelet
        self.decomposition_level = decomposition_level
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sampling_rate)

        # Define EEG frequency bands
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30)
        }

    def bandpass_filter(
        self,
        data: np.ndarray
    ) -> np.ndarray:
        """
        Apply FIR bandpass filter to remove baseline drift and high-frequency artifacts.

        Args:
            data: Raw EEG data of shape (n_channels, n_samples)

        Returns:
            Filtered EEG data of shape (n_channels, n_samples)
        """
        nyquist = self.sampling_rate / 2
        low = self.filter_low / nyquist
        high = self.filter_high / nyquist

        # Design FIR filter with Hamming window
        fir_coeff = signal.firwin(
            self.filter_order,
            [low, high],
            pass_zero=False,
            window='hamming'
        )

        # Apply filter with zero-phase (forward and backward)
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered_data[i] = signal.filtfilt(fir_coeff, 1.0, data[i])

        return filtered_data

    def remove_artifacts_ica(
        self,
        data: np.ndarray,
        channel_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Remove ocular and muscular artifacts using Extended Infomax ICA.

        Args:
            data: Filtered EEG data of shape (n_channels, n_samples)
            channel_names: List of channel names (default: standard 10-20 names)

        Returns:
            Cleaned EEG data with artifacts removed
        """
        if not self.use_ica:
            return data

        n_channels = data.shape[0]

        # Create MNE info structure
        if channel_names is None:
            channel_names = self._get_default_channel_names(n_channels)

        info = mne.create_info(
            ch_names=channel_names,
            sfreq=self.sampling_rate,
            ch_types='eeg'
        )

        # Create Raw object
        raw = mne.io.RawArray(data, info, verbose=False)

        # Apply ICA
        ica = mne.preprocessing.ICA(
            n_components=min(n_channels, 30),
            method='infomax',
            fit_params=dict(extended=True),
            random_state=42,
            verbose=False
        )

        # Fit ICA
        ica.fit(raw, verbose=False)

        # Automatically detect artifact components using ICLabel-like approach
        # Identify EOG and muscle components based on topography and frequency characteristics
        ica.exclude = self._detect_artifact_components(ica, raw)

        # Apply ICA to remove artifact components
        ica.apply(raw, verbose=False)

        return raw.get_data()

    def _detect_artifact_components(
        self,
        ica: mne.preprocessing.ICA,
        raw: mne.io.Raw
    ) -> List[int]:
        """
        Detect artifact components based on topography and spectral characteristics.

        Heuristic approach to identify:
        - Eye blink components (frontal, low frequency)
        - Eye movement components (lateral frontal)
        - Muscle components (high frequency, peripheral)

        Args:
            ica: Fitted ICA object
            raw: Raw EEG data

        Returns:
            List of component indices to exclude
        """
        exclude = []

        # Get ICA components
        components = ica.get_components()

        for idx in range(components.shape[1]):
            component = components[:, idx]

            # Calculate spectral characteristics
            freqs, psd = signal.welch(component, fs=self.sampling_rate, nperseg=256)

            # Check for muscle artifacts (high frequency power)
            high_freq_power = np.mean(psd[freqs > 30])
            total_power = np.mean(psd)

            if high_freq_power / (total_power + 1e-10) > 0.5:
                exclude.append(idx)
                continue

            # Check for eye blink (frontal dominance, low frequency)
            frontal_channels = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4']
            frontal_indices = []
            for ch in frontal_channels:
                if ch in raw.ch_names:
                    frontal_indices.append(raw.ch_names.index(ch))

            if frontal_indices:
                frontal_power = np.mean([component[i]**2 for i in frontal_indices])
                total_spatial_power = np.mean(component**2)

                if frontal_power / (total_spatial_power + 1e-10) > 0.6:
                    low_freq_power = np.mean(psd[freqs < 4])
                    if low_freq_power / (total_power + 1e-10) > 0.5:
                        exclude.append(idx)

        return exclude

    def wavelet_decomposition(
        self,
        data: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Perform multi-scale wavelet decomposition using Daubechies 4.

        Decomposes signal into standard EEG frequency bands:
        - Delta (0.5-4 Hz)
        - Theta (4-8 Hz)
        - Alpha (8-13 Hz)
        - Beta (13-30 Hz)

        Args:
            data: Cleaned EEG data of shape (n_channels, n_samples)

        Returns:
            Dictionary mapping band names to wavelet coefficients
        """
        band_coeffs = {}

        for band_name, (low_freq, high_freq) in self.bands.items():
            band_data = np.zeros_like(data)

            for ch in range(data.shape[0]):
                # Perform discrete wavelet transform
                coeffs = pywt.wavedec(
                    data[ch],
                    self.wavelet,
                    level=self.decomposition_level
                )

                # Select appropriate decomposition levels for each band
                # Level 1: ~64-128 Hz, Level 2: ~32-64 Hz, Level 3: ~16-32 Hz, Level 4: ~8-16 Hz
                # Level 5: ~4-8 Hz, Level 6: ~2-4 Hz, Level 7: ~1-2 Hz, Level 8: ~0.5-1 Hz

                if band_name == 'delta':
                    # Approximation coefficients (low frequencies)
                    selected_coeffs = [coeffs[-1]]
                elif band_name == 'theta':
                    # Detail level corresponding to theta band
                    if len(coeffs) >= 5:
                        selected_coeffs = [coeffs[-2]]
                    else:
                        selected_coeffs = [coeffs[-1]]
                elif band_name == 'alpha':
                    # Detail level corresponding to alpha band
                    if len(coeffs) >= 4:
                        selected_coeffs = [coeffs[-3]]
                    else:
                        selected_coeffs = [coeffs[-2]]
                elif band_name == 'beta':
                    # Detail level corresponding to beta band
                    if len(coeffs) >= 3:
                        selected_coeffs = [coeffs[-4] if len(coeffs) >= 4 else coeffs[-3]]
                    else:
                        selected_coeffs = [coeffs[-2]]
                else:
                    selected_coeffs = coeffs

                # Reconstruct band-limited signal
                # Zero out other coefficients
                modified_coeffs = [np.zeros_like(c) for c in coeffs]
                for i, sc in enumerate(selected_coeffs):
                    if i < len(modified_coeffs):
                        modified_coeffs[-(i+1)] = sc

                band_data[ch] = pywt.waverec(modified_coeffs, self.wavelet)[:data.shape[1]]

            band_coeffs[band_name] = band_data

        return band_coeffs

    def segment_data(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[List[np.ndarray], Optional[List[int]]]:
        """
        Segment continuous EEG into non-overlapping epochs.

        Args:
            data: EEG data of shape (n_channels, n_samples)
            labels: Optional labels for each segment

        Returns:
            Tuple of (list of segments, list of labels)
        """
        n_samples = data.shape[1]
        n_segments = n_samples // self.segment_samples

        segments = []
        segment_labels = [] if labels is not None else None

        for i in range(n_segments):
            start = i * self.segment_samples
            end = start + self.segment_samples
            segment = data[:, start:end]
            segments.append(segment)

            if labels is not None:
                segment_labels.append(labels)

        return segments, segment_labels

    def preprocess(
        self,
        raw_data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        channel_names: Optional[List[str]] = None
    ) -> Dict[str, List[np.ndarray]]:
        """
        Complete preprocessing pipeline.

        Args:
            raw_data: Raw EEG data of shape (n_channels, n_samples) or (n_channels, n_samples, n_trials)
            labels: Optional labels
            channel_names: List of channel names

        Returns:
            Dictionary with preprocessed data for each frequency band
        """
        # Step 1: Bandpass filtering
        filtered_data = self.bandpass_filter(raw_data)

        # Step 2: ICA artifact removal
        cleaned_data = self.remove_artifacts_ica(filtered_data, channel_names)

        # Step 3: Wavelet decomposition
        band_data = self.wavelet_decomposition(cleaned_data)

        # Step 4: Segmentation
        result = {}
        for band_name, band_signal in band_data.items():
            segments, seg_labels = self.segment_data(band_signal, labels)
            result[band_name] = {
                'data': segments,
                'labels': seg_labels
            }

        return result

    def preprocess_batch(
        self,
        data_list: List[np.ndarray],
        labels_list: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Dict[str, List]]:
        """
        Preprocess a batch of EEG recordings.

        Args:
            data_list: List of EEG recordings
            labels_list: Optional list of labels

        Returns:
            Dictionary with batched preprocessed data
        """
        all_results = {
            'delta': {'data': [], 'labels': []},
            'theta': {'data': [], 'labels': []},
            'alpha': {'data': [], 'labels': []},
            'beta': {'data': [], 'labels': []}
        }

        for idx, data in enumerate(data_list):
            labels = labels_list[idx] if labels_list else None
            result = self.preprocess(data, labels)

            for band in all_results.keys():
                all_results[band]['data'].extend(result[band]['data'])
                if labels is not None:
                    all_results[band]['labels'].extend(result[band]['labels'])

        return all_results

    def _get_default_channel_names(self, n_channels: int) -> List[str]:
        """Generate default channel names based on 10-20 system."""
        # Standard 19-channel 10-20 system
        standard_channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'T3', 'C3', 'Cz', 'C4', 'T4',
            'T5', 'P3', 'Pz', 'P4', 'T6',
            'O1', 'O2'
        ]

        if n_channels <= len(standard_channels):
            return standard_channels[:n_channels]
        else:
            # Generate additional channel names
            return standard_channels + [f'Ch{i+1}' for i in range(n_channels - len(standard_channels))]


if __name__ == "__main__":
    # Example usage
    preprocessor = EEGPreprocessor()

    # Generate synthetic data for testing
    n_channels = 19
    n_samples = 10240  # 40 seconds at 256 Hz
    synthetic_data = np.random.randn(n_channels, n_samples)

    # Preprocess
    result = preprocessor.preprocess(synthetic_data)

    print("Preprocessing complete!")
    for band, data in result.items():
        print(f"{band}: {len(data['data'])} segments")
