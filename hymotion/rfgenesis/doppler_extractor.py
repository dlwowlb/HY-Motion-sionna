"""
Doppler Spectrum Extractor for RF-Genesis Integration
 
This module extracts Doppler spectrum data from 3D mesh sequences,
simulating mmWave radar returns from moving human bodies.
 
The Doppler shift is calculated based on the radial velocity of
each body part relative to the radar position.
"""
 
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
 
try:
    from scipy.fft import fft, fftfreq, fft2, fftshift
    from scipy.signal import windows
    from scipy.ndimage import gaussian_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
 
 
@dataclass
class DopplerResult:
    """Container for Doppler extraction results."""
 
    # Time-Doppler map (spectrogram)
    spectrogram: np.ndarray  # (num_frames, num_doppler_bins)
 
    # Range-Doppler map
    range_doppler: np.ndarray  # (num_range_bins, num_doppler_bins)
 
    # Micro-Doppler signature
    micro_doppler: np.ndarray  # (num_frames, num_doppler_bins)
 
    # Velocity axis
    velocity_axis: np.ndarray  # (num_doppler_bins,) in m/s
 
    # Time axis
    time_axis: np.ndarray  # (num_frames,) in seconds
 
    # Range axis
    range_axis: np.ndarray  # (num_range_bins,) in meters
 
    # Frame rate
    fps: float
 
 
class DopplerExtractor:
    """
    Extracts Doppler spectrum from mesh vertex motion.
 
    This simulates the Doppler returns that would be observed by
    a mmWave radar observing a moving human body.
    """
 
    def __init__(
        self,
        radar_config: Any,  # RadarConfig
        doppler_config: Any,  # DopplerConfig
        fps: float = 30.0,
    ):
        """
        Initialize the Doppler extractor.
 
        Args:
            radar_config: Radar hardware configuration
            doppler_config: Doppler processing configuration
            fps: Frame rate of the motion data
        """
        self.radar_config = radar_config
        self.doppler_config = doppler_config
        self.fps = fps
 
        # Speed of light
        self.c = 3e8
 
        # Wavelength
        self.wavelength = self.c / radar_config.start_frequency
 
    def extract(
        self,
        vertices: np.ndarray,
        tx_position: List[float],
        rx_position: List[float],
        return_raw: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Extract Doppler spectrum from mesh vertex sequence.
 
        Args:
            vertices: Mesh vertices (num_frames, num_vertices, 3)
            tx_position: Transmitter position [x, y, z]
            rx_position: Receiver position [x, y, z]
            return_raw: Whether to return raw intermediate data
 
        Returns:
            Dictionary containing:
                - spectrogram: Time-Doppler map (num_frames, num_doppler_bins)
                - range_doppler: Range-Doppler map (num_range_bins, num_doppler_bins)
                - micro_doppler: Micro-Doppler signature
                - velocity_axis: Velocity axis in m/s
                - time_axis: Time axis in seconds
                - range_axis: Range axis in meters
        """
        num_frames, num_vertices, _ = vertices.shape
 
        tx_pos = np.array(tx_position)
        rx_pos = np.array(rx_position)
 
        # Calculate velocity for each vertex at each frame
        velocities = self._compute_velocities(vertices)
 
        # Calculate radial velocity (Doppler-relevant component)
        radial_velocities = self._compute_radial_velocities(
            vertices, velocities, tx_pos, rx_pos
        )
 
        # Calculate range for each vertex
        ranges = self._compute_ranges(vertices, tx_pos)
 
        # Generate Doppler spectrogram
        spectrogram, velocity_axis = self._generate_spectrogram(radial_velocities)
 
        # Generate Range-Doppler map
        range_doppler, range_axis = self._generate_range_doppler(
            ranges, radial_velocities
        )
 
        # Extract micro-Doppler signature
        micro_doppler = self._extract_micro_doppler(spectrogram)
 
        # Time axis
        time_axis = np.arange(num_frames) / self.fps
 
        result = {
            "spectrogram": spectrogram,
            "range_doppler": range_doppler,
            "micro_doppler": micro_doppler,
            "velocity_axis": velocity_axis,
            "time_axis": time_axis,
            "range_axis": range_axis,
            "fps": self.fps,
        }
 
        if return_raw:
            result["raw_radial_velocities"] = radial_velocities
            result["raw_ranges"] = ranges
 
        return result
 
    def _compute_velocities(self, vertices: np.ndarray) -> np.ndarray:
        """
        Compute velocity vectors from vertex positions.
 
        Args:
            vertices: (num_frames, num_vertices, 3)
 
        Returns:
            velocities: (num_frames, num_vertices, 3)
        """
        velocities = np.zeros_like(vertices)
 
        # Central difference for interior frames
        velocities[1:-1] = (vertices[2:] - vertices[:-2]) * self.fps / 2
 
        # Forward difference for first frame
        velocities[0] = (vertices[1] - vertices[0]) * self.fps
 
        # Backward difference for last frame
        velocities[-1] = (vertices[-1] - vertices[-2]) * self.fps
 
        return velocities
 
    def _compute_radial_velocities(
        self,
        vertices: np.ndarray,
        velocities: np.ndarray,
        tx_pos: np.ndarray,
        rx_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Compute radial velocity (component towards/away from radar).
 
        For monostatic radar (tx_pos == rx_pos), this is simply
        the dot product of velocity with the direction to radar.
 
        For bistatic radar, we need to consider both paths.
 
        Args:
            vertices: (num_frames, num_vertices, 3)
            velocities: (num_frames, num_vertices, 3)
            tx_pos: Transmitter position
            rx_pos: Receiver position
 
        Returns:
            radial_velocities: (num_frames, num_vertices)
        """
        num_frames, num_vertices, _ = vertices.shape
        radial_velocities = np.zeros((num_frames, num_vertices))
 
        for t in range(num_frames):
            for v in range(num_vertices):
                point = vertices[t, v]
                vel = velocities[t, v]
 
                # Direction from point to TX
                dir_to_tx = tx_pos - point
                dist_to_tx = np.linalg.norm(dir_to_tx)
                if dist_to_tx > 0:
                    dir_to_tx /= dist_to_tx
 
                # Direction from point to RX
                dir_to_rx = rx_pos - point
                dist_to_rx = np.linalg.norm(dir_to_rx)
                if dist_to_rx > 0:
                    dir_to_rx /= dist_to_rx
 
                # Bistatic radial velocity (average of both directions)
                radial_vel = (np.dot(vel, dir_to_tx) + np.dot(vel, dir_to_rx)) / 2
 
                radial_velocities[t, v] = radial_vel
 
        return radial_velocities
 
    def _compute_ranges(
        self,
        vertices: np.ndarray,
        tx_pos: np.ndarray,
    ) -> np.ndarray:
        """
        Compute range from each vertex to the radar.
 
        Args:
            vertices: (num_frames, num_vertices, 3)
            tx_pos: Transmitter position
 
        Returns:
            ranges: (num_frames, num_vertices)
        """
        # Broadcasting: vertices is (T, V, 3), tx_pos is (3,)
        diff = vertices - tx_pos  # (T, V, 3)
        ranges = np.linalg.norm(diff, axis=2)  # (T, V)
        return ranges
 
    def _generate_spectrogram(
        self,
        radial_velocities: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time-Doppler spectrogram from radial velocities.
 
        This creates a histogram of radial velocities at each time frame,
        weighted by an assumed radar cross-section.
 
        Args:
            radial_velocities: (num_frames, num_vertices)
 
        Returns:
            spectrogram: (num_frames, num_doppler_bins)
            velocity_axis: (num_doppler_bins,)
        """
        num_frames, num_vertices = radial_velocities.shape
        num_doppler_bins = self.doppler_config.doppler_fft_size
 
        # Velocity range
        v_min = self.doppler_config.min_velocity
        v_max = self.doppler_config.max_velocity
        velocity_axis = np.linspace(v_min, v_max, num_doppler_bins)
        bin_width = (v_max - v_min) / num_doppler_bins
 
        spectrogram = np.zeros((num_frames, num_doppler_bins))
 
        for t in range(num_frames):
            frame_velocities = radial_velocities[t]
 
            # Histogram of velocities
            for v in frame_velocities:
                if v_min <= v <= v_max:
                    bin_idx = int((v - v_min) / bin_width)
                    bin_idx = np.clip(bin_idx, 0, num_doppler_bins - 1)
                    spectrogram[t, bin_idx] += 1
 
        # Normalize
        spectrogram = spectrogram / (num_vertices + 1e-8)
 
        # Apply smoothing
        if SCIPY_AVAILABLE:
            spectrogram = gaussian_filter(spectrogram, sigma=1)
 
        return spectrogram, velocity_axis
 
    def _generate_range_doppler(
        self,
        ranges: np.ndarray,
        radial_velocities: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate range-Doppler map (averaged over time).
 
        Args:
            ranges: (num_frames, num_vertices)
            radial_velocities: (num_frames, num_vertices)
 
        Returns:
            range_doppler: (num_range_bins, num_doppler_bins)
            range_axis: (num_range_bins,)
        """
        num_range_bins = self.doppler_config.output_range_bins
        num_doppler_bins = self.doppler_config.doppler_fft_size
 
        # Range and velocity limits
        r_min, r_max = 0.0, self.radar_config.max_range
        v_min = self.doppler_config.min_velocity
        v_max = self.doppler_config.max_velocity
 
        range_axis = np.linspace(r_min, r_max, num_range_bins)
        velocity_axis = np.linspace(v_min, v_max, num_doppler_bins)
 
        range_doppler = np.zeros((num_range_bins, num_doppler_bins))
 
        range_bin_width = (r_max - r_min) / num_range_bins
        vel_bin_width = (v_max - v_min) / num_doppler_bins
 
        # Flatten and process all frames
        all_ranges = ranges.flatten()
        all_velocities = radial_velocities.flatten()
 
        for r, v in zip(all_ranges, all_velocities):
            if r_min <= r <= r_max and v_min <= v <= v_max:
                r_idx = int((r - r_min) / range_bin_width)
                v_idx = int((v - v_min) / vel_bin_width)
 
                r_idx = np.clip(r_idx, 0, num_range_bins - 1)
                v_idx = np.clip(v_idx, 0, num_doppler_bins - 1)
 
                range_doppler[r_idx, v_idx] += 1
 
        # Normalize
        range_doppler = range_doppler / (len(all_ranges) + 1e-8)
 
        # Apply smoothing
        if SCIPY_AVAILABLE:
            range_doppler = gaussian_filter(range_doppler, sigma=1)
 
        return range_doppler, range_axis
 
    def _extract_micro_doppler(
        self,
        spectrogram: np.ndarray,
    ) -> np.ndarray:
        """
        Extract micro-Doppler signature from spectrogram.
 
        Micro-Doppler captures the fine motion details like
        arm swings and leg movements.
 
        Args:
            spectrogram: (num_frames, num_doppler_bins)
 
        Returns:
            micro_doppler: (num_frames, num_doppler_bins)
        """
        # High-pass filter to remove bulk motion
        if not SCIPY_AVAILABLE:
            return spectrogram.copy()
 
        # Simple high-pass: subtract moving average (bulk motion)
        window_size = min(5, spectrogram.shape[0] // 2)
        if window_size < 1:
            return spectrogram.copy()
 
        # Compute bulk motion (low frequency component)
        bulk_motion = np.zeros_like(spectrogram)
        for i in range(spectrogram.shape[0]):
            start = max(0, i - window_size)
            end = min(spectrogram.shape[0], i + window_size + 1)
            bulk_motion[i] = spectrogram[start:end].mean(axis=0)
 
        # Micro-Doppler is the residual
        micro_doppler = spectrogram - bulk_motion
 
        # Keep only positive values
        micro_doppler = np.maximum(micro_doppler, 0)
 
        return micro_doppler
 
 
def extract_doppler_from_mesh_sequence(
    vertices: np.ndarray,
    tx_position: List[float] = [0, 0, 1.5],
    rx_position: List[float] = [0, 0, 1.5],
    frequency: float = 77e9,
    fps: float = 30.0,
) -> Dict[str, np.ndarray]:
    """
    Convenience function to extract Doppler data from mesh sequence.
 
    Args:
        vertices: Mesh vertices (num_frames, num_vertices, 3)
        tx_position: Transmitter position [x, y, z]
        rx_position: Receiver position [x, y, z]
        frequency: Radar frequency in Hz
        fps: Frame rate of the motion data
 
    Returns:
        Dictionary containing Doppler data
    """
    from .rf_simulator import RadarConfig, DopplerConfig
 
    radar_config = RadarConfig(start_frequency=frequency)
    doppler_config = DopplerConfig()
 
    extractor = DopplerExtractor(radar_config, doppler_config, fps=fps)
 
    return extractor.extract(vertices, tx_position, rx_position)