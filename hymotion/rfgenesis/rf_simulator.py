"""
RF-Genesis Ray Tracing Simulator for HY-Motion
 
This module provides RF-Genesis style ray tracing simulation for generating
Doppler, CIR, and CSI data from HY-Motion generated 3D human meshes.
 
Based on RF-Genesis: https://github.com/Asixa/RF-Genesis
Adapted for HY-Motion integration with custom ray tracing implementation.
"""
 
import os
import json
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
 
import numpy as np
import torch
 
# Optional imports
try:
    import scipy.signal as signal
    from scipy.fft import fft, fftfreq, fft2, fftshift
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[Warning] scipy not available. Install with: pip install scipy")
 
 
@dataclass
class RadarConfig:
    """Configuration for radar hardware (TI AWR1843 style)."""
 
    # Radar parameters
    num_tx: int = 3  # Number of transmit antennas
    num_rx: int = 4  # Number of receive antennas
    num_chirps: int = 128  # Number of chirps per frame
    num_samples: int = 256  # ADC samples per chirp
 
    # Chirp parameters
    start_frequency: float = 77e9  # Start frequency (Hz)
    frequency_slope: float = 70e12  # Frequency slope (Hz/s)
    idle_time: float = 7e-6  # Idle time between chirps (s)
    adc_start_time: float = 6e-6  # ADC start time (s)
    ramp_end_time: float = 60e-6  # Ramp end time (s)
 
    # Derived parameters (computed)
    @property
    def bandwidth(self) -> float:
        """Chirp bandwidth in Hz."""
        return self.frequency_slope * self.ramp_end_time
 
    @property
    def range_resolution(self) -> float:
        """Range resolution in meters."""
        c = 3e8  # Speed of light
        return c / (2 * self.bandwidth)
 
    @property
    def max_range(self) -> float:
        """Maximum unambiguous range in meters."""
        c = 3e8
        fs = self.num_samples / self.ramp_end_time  # Sampling rate
        return fs * c / (2 * self.frequency_slope)
 
    @property
    def velocity_resolution(self) -> float:
        """Velocity resolution in m/s."""
        c = 3e8
        wavelength = c / self.start_frequency
        frame_time = self.num_chirps * (self.idle_time + self.ramp_end_time)
        return wavelength / (2 * frame_time)
 
    @property
    def max_velocity(self) -> float:
        """Maximum unambiguous velocity in m/s."""
        c = 3e8
        wavelength = c / self.start_frequency
        chirp_time = self.idle_time + self.ramp_end_time
        return wavelength / (4 * chirp_time)
 
 
@dataclass
class DopplerConfig:
    """Configuration for Doppler spectrum extraction."""
 
    # FFT parameters
    range_fft_size: int = 256
    doppler_fft_size: int = 128
    angle_fft_size: int = 64
 
    # Processing parameters
    window_type: str = "hanning"  # Window function for FFT
    cfar_guard_cells: int = 4
    cfar_training_cells: int = 16
    cfar_threshold_factor: float = 5.0
 
    # Output parameters
    output_range_bins: int = 128
    output_doppler_bins: int = 64
    min_velocity: float = -10.0  # m/s
    max_velocity: float = 10.0  # m/s
 
 
@dataclass
class RFConfig:
    """Main configuration for RF simulation."""
 
    # Carrier frequency
    frequency: float = 77e9  # Default: 77 GHz mmWave
 
    # Radar configuration
    radar: RadarConfig = field(default_factory=RadarConfig)
 
    # Doppler configuration
    doppler: DopplerConfig = field(default_factory=DopplerConfig)
 
    # Environment (can use EnvironmentConfig or manual settings)
    environment: Optional[Any] = None  # EnvironmentConfig instance
    room_size: Tuple[float, float, float] = (10.0, 10.0, 3.0)  # meters (fallback)
 
    # Transmitter/Receiver positions
    tx_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.5])
    rx_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.5])
 
    # Material properties
    human_reflectivity: float = 0.7  # Reflection coefficient for human body
    wall_reflectivity: float = 0.5
    floor_reflectivity: float = 0.3
 
    # Ray tracing parameters
    max_bounces: int = 3
    num_rays: int = 10000
 
    # Output options
    output_doppler: bool = True
    output_cir: bool = True
    output_csi: bool = True
    output_point_cloud: bool = True
 
    @classmethod
    def from_environment_prompt(
        cls,
        prompt: str,
        frequency: float = 77e9,
        **kwargs,
    ) -> "RFConfig":
        """
        Create RFConfig from RFLoRA-style environment prompt.
 
        Args:
            prompt: Natural language environment description
                    e.g., "a living room with a sofa, TV, and two windows"
            frequency: Carrier frequency in Hz
            **kwargs: Additional RFConfig parameters
 
        Returns:
            RFConfig instance with environment configured
 
        Example:
            >>> config = RFConfig.from_environment_prompt(
            ...     "a kitchen with a refrigerator, table, and microwave",
            ...     frequency=60e9
            ... )
        """
        from .environment import EnvironmentConfig
 
        env = EnvironmentConfig.from_prompt(prompt)
 
        return cls(
            frequency=frequency,
            environment=env,
            room_size=env.room_size,
            tx_position=env.radar_position,
            rx_position=env.radar_position,
            **kwargs,
        )
 
    @classmethod
    def from_environment_preset(
        cls,
        preset_name: str,
        frequency: float = 77e9,
        **kwargs,
    ) -> "RFConfig":
        """
        Create RFConfig from environment preset.
 
        Available presets: living_room, office, bedroom, corridor,
                          kitchen, bathroom, empty_room, outdoor_open
 
        Args:
            preset_name: Name of the preset
            frequency: Carrier frequency in Hz
            **kwargs: Additional RFConfig parameters
 
        Returns:
            RFConfig instance with preset environment
        """
        from .environment import EnvironmentConfig
 
        env = EnvironmentConfig.preset(preset_name)
 
        return cls(
            frequency=frequency,
            environment=env,
            room_size=env.room_size,
            tx_position=env.radar_position,
            rx_position=env.radar_position,
            **kwargs,
        )
 
 
class RFGenesisSimulator:
    """
    Main RF simulation class integrating RF-Genesis concepts with HY-Motion.
 
    This simulator generates realistic RF sensing data from 3D human motion:
    - Doppler spectrum from body movement
    - CIR (Channel Impulse Response)
    - CSI (Channel State Information)
    - Radar point cloud
    """
 
    def __init__(self, config: Optional[RFConfig] = None):
        """
        Initialize the RF simulator.
 
        Args:
            config: RF simulation configuration
        """
        self.config = config or RFConfig()
        self._temp_dir = tempfile.mkdtemp(prefix="rfgenesis_")
 
        # Wavelength
        self.wavelength = 3e8 / self.config.frequency
 
        # Initialize sub-modules
        from .doppler_extractor import DopplerExtractor
        from .channel_generator import CIRGenerator, CSIGenerator
 
        self.doppler_extractor = DopplerExtractor(
            self.config.radar,
            self.config.doppler,
        )
        self.cir_generator = CIRGenerator(self.config)
        self.csi_generator = CSIGenerator(self.config)
 
    def simulate_from_motion(
        self,
        runtime: Any,  # T2MRuntime
        text_prompt: str,
        duration: float = 3.0,
        seed: int = 42,
        cfg_scale: float = 5.0,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate motion and simulate RF data.
 
        Args:
            runtime: HY-Motion T2MRuntime instance
            text_prompt: Text description of the motion
            duration: Duration in seconds
            seed: Random seed
            cfg_scale: Classifier-free guidance scale
            output_dir: Output directory
 
        Returns:
            Dictionary containing:
                - doppler: Doppler spectrum data
                - cir: Channel Impulse Response
                - csi: Channel State Information
                - point_cloud: Radar point cloud
                - metadata: Simulation metadata
        """
        output_dir = output_dir or tempfile.mkdtemp(prefix="rfgenesis_output_")
        os.makedirs(output_dir, exist_ok=True)
 
        print(f"[RFGenesisSimulator] Generating motion for: '{text_prompt}'")
 
        # Step 1: Generate motion
        motion_data = self._generate_motion(runtime, text_prompt, duration, seed, cfg_scale)
 
        # Step 2: Extract mesh sequence
        mesh_sequence = self._extract_mesh_sequence(motion_data, runtime)
 
        # Step 3: Simulate RF data
        results = self.simulate_from_mesh_sequence(mesh_sequence, output_dir)
 
        # Add metadata
        results["metadata"] = {
            "text_prompt": text_prompt,
            "duration": duration,
            "seed": seed,
            "frequency": self.config.frequency,
            "num_frames": mesh_sequence["vertices"].shape[0],
        }
 
        # Save results
        self._save_results(results, output_dir)
 
        print(f"[RFGenesisSimulator] Results saved to: {output_dir}")
        return results
 
    def simulate_from_mesh_sequence(
        self,
        mesh_sequence: Dict[str, np.ndarray],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Simulate RF data from pre-computed mesh sequence.
 
        Args:
            mesh_sequence: Dictionary containing:
                - vertices: (num_frames, num_vertices, 3)
                - faces: (num_faces, 3)
            output_dir: Output directory
 
        Returns:
            Dictionary with RF simulation results
        """
        vertices = mesh_sequence["vertices"]  # (T, V, 3)
        faces = mesh_sequence["faces"]  # (F, 3)
 
        num_frames = vertices.shape[0]
        print(f"[RFGenesisSimulator] Processing {num_frames} frames...")
 
        results = {}
 
        # 1. Extract Doppler data
        if self.config.output_doppler:
            print("  Extracting Doppler spectrum...")
            doppler_data = self.doppler_extractor.extract(
                vertices,
                self.config.tx_position,
                self.config.rx_position,
            )
            results["doppler"] = doppler_data
 
        # 2. Generate CIR
        if self.config.output_cir:
            print("  Generating CIR...")
            cir_data = self.cir_generator.generate(
                vertices,
                faces,
                self.config.tx_position,
                self.config.rx_position,
            )
            results["cir"] = cir_data
 
        # 3. Generate CSI
        if self.config.output_csi:
            print("  Generating CSI...")
            csi_data = self.csi_generator.generate(
                vertices,
                faces,
                self.config.tx_position,
                self.config.rx_position,
            )
            results["csi"] = csi_data
 
        # 4. Generate radar point cloud
        if self.config.output_point_cloud:
            print("  Generating radar point cloud...")
            point_cloud = self._generate_point_cloud(vertices, faces)
            results["point_cloud"] = point_cloud
 
        return results
 
    def _generate_motion(
        self,
        runtime: Any,
        text_prompt: str,
        duration: float,
        seed: int,
        cfg_scale: float,
    ) -> Dict[str, Any]:
        """Generate motion using HY-Motion."""
        # Use prompt engineering if available
        if hasattr(runtime, 'prompt_rewriter') and runtime.prompt_rewriter is not None:
            try:
                predicted_duration, rewritten_text = runtime.rewrite_text_and_infer_time(text_prompt)
                duration = predicted_duration
                text_prompt = rewritten_text
            except Exception as e:
                print(f"  [Warning] Prompt rewriting failed: {e}")
 
        # Generate motion
        html_content, fbx_files, model_output = runtime.generate_motion(
            text=text_prompt,
            seeds_csv=str(seed),
            duration=duration,
            cfg_scale=cfg_scale,
            output_format="dict",
        )
 
        return model_output
 
    def _extract_mesh_sequence(
        self,
        motion_data: Dict[str, Any],
        runtime: Any,
    ) -> Dict[str, np.ndarray]:
        """Extract mesh vertices and faces from motion data."""
        # Get body model from runtime
        if hasattr(runtime, 'pipelines') and runtime.pipelines:
            body_model = runtime.pipelines[0].body_model
        else:
            from ..pipeline.body_model import WoodenMesh
            body_model = WoodenMesh()
 
        # Get rotation and translation data
        rot6d = motion_data["rot6d"]  # (B, L, J, 6)
        transl = motion_data["transl"]  # (B, L, 3)
 
        # Use first sample if batch
        if rot6d.ndim == 4:
            rot6d = rot6d[0]
            transl = transl[0]
 
        # Convert to torch if numpy
        if isinstance(rot6d, np.ndarray):
            rot6d = torch.from_numpy(rot6d).float()
        if isinstance(transl, np.ndarray):
            transl = torch.from_numpy(transl).float()
 
        # Get device
        device = torch.device('cpu')
        try:
            device = next(body_model.buffers()).device
        except StopIteration:
            try:
                device = next(body_model.parameters()).device
            except StopIteration:
                pass
 
        rot6d = rot6d.to(device)
        transl = transl.to(device)
 
        # Forward pass
        with torch.no_grad():
            output = body_model.forward({
                "rot6d": rot6d,
                "trans": transl,
            })
 
        vertices = output["vertices"].cpu().numpy()
        faces = body_model.faces
 
        return {
            "vertices": vertices,
            "faces": faces,
            "keypoints3d": motion_data.get("keypoints3d", None),
        }
 
    def _generate_point_cloud(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Generate radar point cloud from mesh sequence.
 
        Simulates radar detections on the human body surface.
        """
        num_frames = vertices.shape[0]
        tx_pos = np.array(self.config.tx_position)
        rx_pos = np.array(self.config.rx_position)
 
        all_points = []
        all_velocities = []
        all_intensities = []
        all_frame_indices = []
 
        for frame_idx in range(num_frames):
            frame_verts = vertices[frame_idx]  # (V, 3)
 
            # Calculate velocity if not first frame
            if frame_idx > 0:
                velocity = (vertices[frame_idx] - vertices[frame_idx - 1]) * 30  # Assuming 30 fps
            else:
                velocity = np.zeros_like(frame_verts)
 
            # Sample points from mesh surface
            num_points = min(100, len(frame_verts))
            indices = np.random.choice(len(frame_verts), num_points, replace=False)
 
            for idx in indices:
                point = frame_verts[idx]
                vel = velocity[idx]
 
                # Calculate range
                range_val = np.linalg.norm(point - tx_pos)
 
                # Calculate radial velocity (Doppler)
                direction_to_radar = (tx_pos - point) / (range_val + 1e-8)
                radial_velocity = np.dot(vel, direction_to_radar)
 
                # Calculate intensity (simplified RCS model)
                intensity = self.config.human_reflectivity / (range_val ** 4 + 1e-8)
 
                all_points.append(point)
                all_velocities.append(radial_velocity)
                all_intensities.append(intensity)
                all_frame_indices.append(frame_idx)
 
        return {
            "points": np.array(all_points),  # (N, 3)
            "velocities": np.array(all_velocities),  # (N,)
            "intensities": np.array(all_intensities),  # (N,)
            "frame_indices": np.array(all_frame_indices),  # (N,)
        }
 
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Save simulation results to files."""
        # Save Doppler data
        if "doppler" in results:
            np.savez(
                os.path.join(output_dir, "doppler_data.npz"),
                **{k: v for k, v in results["doppler"].items() if isinstance(v, np.ndarray)}
            )
 
        # Save CIR data
        if "cir" in results:
            np.savez(
                os.path.join(output_dir, "cir_data.npz"),
                **{k: v for k, v in results["cir"].items() if isinstance(v, np.ndarray)}
            )
 
        # Save CSI data
        if "csi" in results:
            np.savez(
                os.path.join(output_dir, "csi_data.npz"),
                **{k: v for k, v in results["csi"].items() if isinstance(v, np.ndarray)}
            )
 
        # Save point cloud
        if "point_cloud" in results:
            np.savez(
                os.path.join(output_dir, "point_cloud.npz"),
                **results["point_cloud"]
            )
 
        # Save metadata
        if "metadata" in results:
            with open(os.path.join(output_dir, "metadata.json"), "w") as f:
                json.dump(results["metadata"], f, indent=2)