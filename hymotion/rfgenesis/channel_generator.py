"""
Channel Impulse Response (CIR) and Channel State Information (CSI) Generator

This module generates realistic CIR and CSI data from 3D mesh sequences,
simulating the wireless channel characteristics affected by human body motion.

CIR: Time-domain representation of the channel
CSI: Frequency-domain representation (Fourier transform of CIR)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

try:
    from scipy.fft import fft, ifft, fftfreq
    from scipy.signal import windows
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ChannelGenerator:
    """
    Base class for channel generation.

    Implements common ray tracing functionality for both CIR and CSI generation.
    """

    def __init__(self, config: Any):  # RFConfig
        """
        Initialize the channel generator.

        Args:
            config: RF simulation configuration
        """
        self.config = config
        self.c = 3e8  # Speed of light
        self.wavelength = self.c / config.frequency

    def trace_rays(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        tx_pos: np.ndarray,
        rx_pos: np.ndarray,
        frame_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Trace rays from TX to RX, considering reflections off the human body.

        This is a simplified ray tracing implementation that considers:
        1. Direct path (Line of Sight)
        2. Single-bounce reflections off the human body
        3. Ground reflections

        Args:
            vertices: Mesh vertices (num_frames, num_vertices, 3) or (num_vertices, 3)
            faces: Mesh faces (num_faces, 3)
            tx_pos: Transmitter position
            rx_pos: Receiver position
            frame_idx: Frame index to use

        Returns:
            List of ray dictionaries with:
                - path_length: Total path length (meters)
                - delay: Propagation delay (seconds)
                - amplitude: Complex amplitude
                - doppler: Doppler shift (Hz)
                - path_type: "los", "reflection", "ground"
        """
        rays = []

        # Get frame vertices
        if vertices.ndim == 3:
            verts = vertices[frame_idx]
        else:
            verts = vertices

        # 1. Direct path (LOS)
        los_ray = self._compute_los_ray(tx_pos, rx_pos, verts)
        if los_ray is not None:
            rays.append(los_ray)

        # 2. Reflections off human body
        body_rays = self._compute_body_reflections(
            verts, faces, tx_pos, rx_pos
        )
        rays.extend(body_rays)

        # 3. Ground reflection
        ground_ray = self._compute_ground_reflection(tx_pos, rx_pos)
        if ground_ray is not None:
            rays.append(ground_ray)

        return rays

    def _compute_los_ray(
        self,
        tx_pos: np.ndarray,
        rx_pos: np.ndarray,
        body_verts: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Compute Line-of-Sight ray if not blocked by body."""
        path_vector = rx_pos - tx_pos
        path_length = np.linalg.norm(path_vector)

        # Check if LOS is blocked by body (simplified)
        body_center = body_verts.mean(axis=0)
        body_radius = np.linalg.norm(body_verts - body_center, axis=1).max()

        # Distance from body center to LOS line
        t = np.dot(body_center - tx_pos, path_vector) / (path_length ** 2 + 1e-8)
        t = np.clip(t, 0, 1)
        closest_point = tx_pos + t * path_vector
        dist_to_body = np.linalg.norm(body_center - closest_point)

        # If LOS passes through body, attenuate
        if dist_to_body < body_radius:
            # Partial obstruction
            obstruction_factor = dist_to_body / body_radius
            attenuation = 0.1 + 0.9 * obstruction_factor  # 10% to 100%
        else:
            attenuation = 1.0

        # Free-space path loss
        fspl = (self.wavelength / (4 * np.pi * path_length)) ** 2

        # Complex amplitude (phase from path length)
        phase = 2 * np.pi * path_length / self.wavelength
        amplitude = np.sqrt(fspl) * attenuation * np.exp(-1j * phase)

        return {
            "path_length": path_length,
            "delay": path_length / self.c,
            "amplitude": amplitude,
            "doppler": 0.0,  # Static for single frame
            "path_type": "los",
        }

    def _compute_body_reflections(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        tx_pos: np.ndarray,
        rx_pos: np.ndarray,
        num_reflections: int = 50,
    ) -> List[Dict[str, Any]]:
        """Compute reflections off the human body surface."""
        rays = []

        # Sample reflection points on body surface
        num_vertices = len(vertices)
        sample_indices = np.random.choice(
            num_vertices,
            min(num_reflections, num_vertices),
            replace=False,
        )

        for idx in sample_indices:
            reflection_point = vertices[idx]

            # Path: TX -> reflection point -> RX
            d1 = np.linalg.norm(reflection_point - tx_pos)
            d2 = np.linalg.norm(rx_pos - reflection_point)
            total_path = d1 + d2

            # Simplified reflection coefficient for human body
            # (depends on incidence angle and material properties)
            reflection_coef = self.config.human_reflectivity * 0.5

            # Free-space path loss for total path
            fspl = (self.wavelength / (4 * np.pi * total_path)) ** 2

            # Phase
            phase = 2 * np.pi * total_path / self.wavelength

            # Complex amplitude
            amplitude = np.sqrt(fspl) * reflection_coef * np.exp(-1j * phase)

            rays.append({
                "path_length": total_path,
                "delay": total_path / self.c,
                "amplitude": amplitude,
                "doppler": 0.0,
                "path_type": "body_reflection",
                "reflection_point": reflection_point.tolist(),
            })

        return rays

    def _compute_ground_reflection(
        self,
        tx_pos: np.ndarray,
        rx_pos: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """Compute ground reflection (two-ray model)."""
        # Mirror TX position below ground
        tx_mirror = tx_pos.copy()
        tx_mirror[2] = -tx_pos[2]

        # Path length via ground
        ground_path = np.linalg.norm(rx_pos - tx_mirror)

        # Ground reflection coefficient (approximate)
        reflection_coef = -self.config.floor_reflectivity  # Phase reversal

        # Free-space path loss
        fspl = (self.wavelength / (4 * np.pi * ground_path)) ** 2

        # Phase
        phase = 2 * np.pi * ground_path / self.wavelength

        # Complex amplitude
        amplitude = np.sqrt(fspl) * reflection_coef * np.exp(-1j * phase)

        return {
            "path_length": ground_path,
            "delay": ground_path / self.c,
            "amplitude": amplitude,
            "doppler": 0.0,
            "path_type": "ground",
        }


class CIRGenerator(ChannelGenerator):
    """
    Channel Impulse Response (CIR) Generator.

    Generates time-domain channel impulse response from mesh motion,
    capturing multipath propagation effects.
    """

    def __init__(
        self,
        config: Any,
        num_taps: int = 256,
        tap_spacing: float = 1e-9,  # 1 ns
    ):
        """
        Initialize CIR generator.

        Args:
            config: RF configuration
            num_taps: Number of delay taps
            tap_spacing: Time spacing between taps (seconds)
        """
        super().__init__(config)
        self.num_taps = num_taps
        self.tap_spacing = tap_spacing
        self.max_delay = num_taps * tap_spacing

    def generate(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        tx_position: List[float],
        rx_position: List[float],
    ) -> Dict[str, np.ndarray]:
        """
        Generate CIR from mesh sequence.

        Args:
            vertices: Mesh vertices (num_frames, num_vertices, 3)
            faces: Mesh faces (num_faces, 3)
            tx_position: Transmitter position
            rx_position: Receiver position

        Returns:
            Dictionary containing:
                - cir: Complex CIR (num_frames, num_taps)
                - cir_magnitude: |CIR| (num_frames, num_taps)
                - cir_phase: Phase of CIR (num_frames, num_taps)
                - delay_axis: Delay axis (num_taps,) in seconds
                - power_delay_profile: Average PDP (num_taps,)
        """
        tx_pos = np.array(tx_position)
        rx_pos = np.array(rx_position)

        num_frames = vertices.shape[0]
        cir = np.zeros((num_frames, self.num_taps), dtype=np.complex128)

        # Generate CIR for each frame
        for frame_idx in range(num_frames):
            # Trace rays for this frame
            rays = self.trace_rays(vertices, faces, tx_pos, rx_pos, frame_idx)

            # Add Doppler shift if we have previous frame
            if frame_idx > 0:
                rays = self._add_doppler(
                    rays, vertices, frame_idx, tx_pos, rx_pos
                )

            # Convert rays to CIR
            frame_cir = self._rays_to_cir(rays)
            cir[frame_idx] = frame_cir

        # Compute derived quantities
        cir_magnitude = np.abs(cir)
        cir_phase = np.angle(cir)
        delay_axis = np.arange(self.num_taps) * self.tap_spacing

        # Power delay profile (average over frames)
        power_delay_profile = np.mean(cir_magnitude ** 2, axis=0)

        return {
            "cir": cir,
            "cir_magnitude": cir_magnitude,
            "cir_phase": cir_phase,
            "delay_axis": delay_axis,
            "power_delay_profile": power_delay_profile,
            "num_frames": num_frames,
            "num_taps": self.num_taps,
            "tap_spacing": self.tap_spacing,
        }

    def _add_doppler(
        self,
        rays: List[Dict],
        vertices: np.ndarray,
        frame_idx: int,
        tx_pos: np.ndarray,
        rx_pos: np.ndarray,
    ) -> List[Dict]:
        """Add Doppler shift to rays based on motion between frames."""
        fps = 30.0  # Assumed frame rate

        for ray in rays:
            if ray["path_type"] == "body_reflection" and "reflection_point" in ray:
                # Get reflection point
                ref_point = np.array(ray["reflection_point"])

                # Find closest vertex
                verts_curr = vertices[frame_idx]
                dists = np.linalg.norm(verts_curr - ref_point, axis=1)
                closest_idx = np.argmin(dists)

                # Compute velocity
                if frame_idx > 0:
                    velocity = (vertices[frame_idx, closest_idx] -
                               vertices[frame_idx - 1, closest_idx]) * fps
                else:
                    velocity = np.zeros(3)

                # Radial velocity component
                dir_to_tx = (tx_pos - ref_point) / (np.linalg.norm(tx_pos - ref_point) + 1e-8)
                dir_to_rx = (rx_pos - ref_point) / (np.linalg.norm(rx_pos - ref_point) + 1e-8)

                radial_vel = (np.dot(velocity, dir_to_tx) + np.dot(velocity, dir_to_rx)) / 2

                # Doppler shift
                doppler = 2 * radial_vel / self.wavelength
                ray["doppler"] = doppler

        return rays

    def _rays_to_cir(self, rays: List[Dict]) -> np.ndarray:
        """Convert ray list to CIR vector."""
        cir = np.zeros(self.num_taps, dtype=np.complex128)

        for ray in rays:
            delay = ray["delay"]
            amplitude = ray["amplitude"]

            # Find tap index
            tap_idx = int(delay / self.tap_spacing)

            if 0 <= tap_idx < self.num_taps:
                # Add contribution to tap
                # (could use interpolation for sub-tap delays)
                cir[tap_idx] += amplitude

        return cir


class CSIGenerator(ChannelGenerator):
    """
    Channel State Information (CSI) Generator.

    Generates frequency-domain channel response (CSI) from mesh motion.
    CSI is widely used in WiFi sensing applications.
    """

    def __init__(
        self,
        config: Any,
        num_subcarriers: int = 64,
        bandwidth: float = 20e6,  # 20 MHz (WiFi-like)
    ):
        """
        Initialize CSI generator.

        Args:
            config: RF configuration
            num_subcarriers: Number of OFDM subcarriers
            bandwidth: Total bandwidth (Hz)
        """
        super().__init__(config)
        self.num_subcarriers = num_subcarriers
        self.bandwidth = bandwidth
        self.subcarrier_spacing = bandwidth / num_subcarriers

    def generate(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        tx_position: List[float],
        rx_position: List[float],
    ) -> Dict[str, np.ndarray]:
        """
        Generate CSI from mesh sequence.

        Args:
            vertices: Mesh vertices (num_frames, num_vertices, 3)
            faces: Mesh faces (num_faces, 3)
            tx_position: Transmitter position
            rx_position: Receiver position

        Returns:
            Dictionary containing:
                - csi: Complex CSI (num_frames, num_subcarriers)
                - csi_magnitude: |CSI| (num_frames, num_subcarriers)
                - csi_phase: Phase of CSI (num_frames, num_subcarriers)
                - frequency_axis: Frequency axis (num_subcarriers,) in Hz
                - csi_amplitude_variance: Per-subcarrier variance
        """
        tx_pos = np.array(tx_position)
        rx_pos = np.array(rx_position)

        num_frames = vertices.shape[0]
        csi = np.zeros((num_frames, self.num_subcarriers), dtype=np.complex128)

        # Frequency axis
        center_freq = self.config.frequency
        freq_axis = center_freq + np.linspace(
            -self.bandwidth / 2,
            self.bandwidth / 2,
            self.num_subcarriers,
        )

        # Generate CSI for each frame
        for frame_idx in range(num_frames):
            # Trace rays
            rays = self.trace_rays(vertices, faces, tx_pos, rx_pos, frame_idx)

            # Convert to CSI
            frame_csi = self._rays_to_csi(rays, freq_axis)
            csi[frame_idx] = frame_csi

        # Compute derived quantities
        csi_magnitude = np.abs(csi)
        csi_phase = np.angle(csi)

        # Unwrap phase for continuity
        csi_phase_unwrapped = np.unwrap(csi_phase, axis=0)

        # Per-subcarrier variance (useful for sensing)
        csi_amplitude_variance = np.var(csi_magnitude, axis=0)

        return {
            "csi": csi,
            "csi_magnitude": csi_magnitude,
            "csi_phase": csi_phase,
            "csi_phase_unwrapped": csi_phase_unwrapped,
            "frequency_axis": freq_axis,
            "csi_amplitude_variance": csi_amplitude_variance,
            "num_frames": num_frames,
            "num_subcarriers": self.num_subcarriers,
            "bandwidth": self.bandwidth,
            "center_frequency": center_freq,
        }

    def _rays_to_csi(
        self,
        rays: List[Dict],
        freq_axis: np.ndarray,
    ) -> np.ndarray:
        """Convert ray list to CSI vector."""
        csi = np.zeros(len(freq_axis), dtype=np.complex128)

        for ray in rays:
            delay = ray["delay"]
            amplitude = ray["amplitude"]
            doppler = ray.get("doppler", 0.0)

            # Frequency response: sum of complex exponentials
            # H(f) = sum_i a_i * exp(-j * 2 * pi * f * tau_i)
            for i, f in enumerate(freq_axis):
                # Phase from delay
                phase = -2 * np.pi * f * delay

                # Doppler contribution (small effect for single frame)
                # In practice, Doppler affects phase over multiple frames

                csi[i] += np.abs(amplitude) * np.exp(1j * phase)

        return csi

    def generate_mimo_csi(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        tx_positions: List[List[float]],
        rx_positions: List[List[float]],
    ) -> Dict[str, np.ndarray]:
        """
        Generate MIMO CSI matrix from mesh sequence.

        Args:
            vertices: Mesh vertices (num_frames, num_vertices, 3)
            faces: Mesh faces (num_faces, 3)
            tx_positions: List of TX antenna positions
            rx_positions: List of RX antenna positions

        Returns:
            Dictionary containing MIMO CSI:
                - csi: (num_frames, num_rx, num_tx, num_subcarriers)
        """
        num_tx = len(tx_positions)
        num_rx = len(rx_positions)
        num_frames = vertices.shape[0]

        csi_mimo = np.zeros(
            (num_frames, num_rx, num_tx, self.num_subcarriers),
            dtype=np.complex128,
        )

        for tx_idx, tx_pos in enumerate(tx_positions):
            for rx_idx, rx_pos in enumerate(rx_positions):
                # Generate CSI for this TX-RX pair
                csi_data = self.generate(
                    vertices, faces, tx_pos, rx_pos
                )
                csi_mimo[:, rx_idx, tx_idx, :] = csi_data["csi"]

        return {
            "csi_mimo": csi_mimo,
            "num_tx": num_tx,
            "num_rx": num_rx,
            "num_frames": num_frames,
            "num_subcarriers": self.num_subcarriers,
        }
