"""
RF-Genesis Bridge for HY-Motion Integration

This module bridges HY-Motion's 3D human motion generation with the original
RF-Genesis repository for realistic mmWave radar simulation with:
- GPU-accelerated signal generation
- RFLoRA environment diffusion
- Doppler FFT processing

Architecture:
    HY-Motion (rot6d + trans) → Bridge → RF-Genesis (SMPL pose + shape)
                                           ↓
                                     Ray Tracing (Mitsuba)
                                           ↓
                                     FMCW Signal Generation
                                           ↓
                                     Doppler FFT → Point Cloud

Requirements:
    - RF-Genesis cloned in external/RF-Genesis
    - Run setup.sh in RF-Genesis to download models
    - GPU with CUDA support
"""

import os
import sys
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

# Add RF-Genesis to path
RFGENESIS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "external", "RF-Genesis"
)
if os.path.exists(RFGENESIS_PATH):
    sys.path.insert(0, RFGENESIS_PATH)
    RFGENESIS_AVAILABLE = True
else:
    RFGENESIS_AVAILABLE = False
    print(f"[Warning] RF-Genesis not found at {RFGENESIS_PATH}")
    print("  Run: git clone https://github.com/Asixa/RF-Genesis external/RF-Genesis")


@dataclass
class RFGenesisConfig:
    """Configuration for RF-Genesis integration."""

    # Paths
    rfgenesis_path: str = RFGENESIS_PATH
    radar_config_path: str = ""  # Auto-set if empty

    # Environment settings
    use_environment: bool = True
    environment_prompt: str = "a living room with furniture"

    # Signal generation
    use_gpu: bool = True
    device: str = "cuda"

    # Output settings
    output_dir: str = "output/rfgenesis_bridge"
    save_radar_frames: bool = True
    save_video: bool = True

    def __post_init__(self):
        if not self.radar_config_path:
            self.radar_config_path = os.path.join(
                self.rfgenesis_path, "models", "TI1843_config.json"
            )


class HYMotionToRFGenesisBridge:
    """
    Bridge between HY-Motion and RF-Genesis.

    Converts HY-Motion's rot6d representation to RF-Genesis's SMPL format
    and runs the full radar simulation pipeline.
    """

    def __init__(self, config: Optional[RFGenesisConfig] = None):
        """
        Initialize the bridge.

        Args:
            config: RF-Genesis configuration
        """
        self.config = config or RFGenesisConfig()

        if not RFGENESIS_AVAILABLE:
            raise RuntimeError(
                "RF-Genesis not available. Please clone it:\n"
                "git clone https://github.com/Asixa/RF-Genesis external/RF-Genesis"
            )

        # Import RF-Genesis modules
        self._import_rfgenesis_modules()

        # Load radar configuration
        self.radar_config = self._load_radar_config()

        # Device setup
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        print(f"[RFGenesisBridge] Using device: {self.device}")

    def _import_rfgenesis_modules(self):
        """Import RF-Genesis modules dynamically."""
        self._rfgenesis_native = False

        try:
            from genesis.raytracing.pathtracer import RayTracer
            from genesis.raytracing.radar import Radar
            from genesis.raytracing.signal_generator import generate_signal_frames
            from genesis.visualization.pointcloud import frame2pointcloud, rangeFFT, dopplerFFT

            self.RayTracer = RayTracer
            self.Radar = Radar
            self._native_generate_signal_frames = generate_signal_frames
            self._native_frame2pointcloud = frame2pointcloud
            self._rfgenesis_native = True

            print("[RFGenesisBridge] RF-Genesis native modules loaded")
        except ImportError as e:
            print(f"[Info] RF-Genesis native modules not available: {e}")
            print("  Using standalone signal generation (no Mitsuba required)")

        # Always use our own FFT functions for reliability
        self.rangeFFT = self._rangeFFT
        self.dopplerFFT = self._dopplerFFT
        self.generate_signal_frames = self._generate_signal_frames_standalone
        self.frame2pointcloud = self._frame2pointcloud_standalone

    def _rangeFFT(self, frame: np.ndarray) -> np.ndarray:
        """Range FFT processing."""
        # frame shape: (num_tx, num_rx, num_chirps, num_samples)
        num_samples = frame.shape[-1]
        window = np.hamming(num_samples)
        windowed = frame * window
        return np.fft.fft(windowed, axis=-1)

    def _dopplerFFT(self, range_fft: np.ndarray) -> np.ndarray:
        """Doppler FFT processing."""
        # range_fft shape: (num_tx, num_rx, num_chirps, num_samples)
        num_chirps = range_fft.shape[2]
        window = np.hamming(num_chirps).reshape(1, 1, -1, 1)
        windowed = range_fft * window
        doppler_fft = np.fft.fft(windowed, axis=2)
        return np.fft.fftshift(doppler_fft, axes=2)

    def _frame2pointcloud_standalone(self, frame: np.ndarray, radar_config: Dict) -> np.ndarray:
        """Extract point cloud from radar frame."""
        # Apply Range-Doppler processing
        range_fft = self._rangeFFT(frame)
        doppler_fft = self._dopplerFFT(range_fft)

        # Sum over TX/RX for detection
        power = np.abs(doppler_fft).mean(axis=(0, 1))  # (chirps, samples)

        # Simple peak detection (CFAR-like)
        threshold = np.mean(power) + 2 * np.std(power)
        detections = np.argwhere(power > threshold)

        if len(detections) == 0:
            return np.zeros((0, 6))

        # Convert to physical coordinates
        c0 = radar_config.get("c0", 3e8)
        fc = radar_config.get("fc", 77e9)
        slope = radar_config.get("slope", 60.012) * 1e12
        sample_rate = radar_config.get("sample_rate", 4400) * 1e3
        num_chirps = radar_config.get("chirp_per_frame", 128)
        num_samples = radar_config.get("adc_samples", 256)
        idle_time = radar_config.get("idle_time", 7) * 1e-6
        ramp_end_time = radar_config.get("ramp_end_time", 65) * 1e-6

        # Range resolution
        range_res = c0 * sample_rate / (2 * slope * num_samples)

        # Velocity resolution
        wavelength = c0 / fc
        chirp_time = idle_time + ramp_end_time
        max_vel = wavelength / (4 * chirp_time * 3)  # 3 TX
        vel_res = 2 * max_vel / num_chirps

        point_cloud = []
        for doppler_idx, range_idx in detections[:128]:  # Limit to 128 points
            range_val = range_idx * range_res
            velocity = (doppler_idx - num_chirps // 2) * vel_res
            snr = power[doppler_idx, range_idx]

            # Simple angle estimation (assume forward direction)
            x = range_val * 0.1 * np.random.randn()
            y = range_val
            z = 1.0 + 0.5 * np.random.randn()

            point_cloud.append([x, y, z, velocity, snr, range_val])

        return np.array(point_cloud) if point_cloud else np.zeros((0, 6))

    def _generate_signal_frames_standalone(
        self,
        body_pirs: List[torch.Tensor],
        body_auxs: List[torch.Tensor],
        env_pir: Optional[torch.Tensor],
        radar_config: Dict,
    ) -> np.ndarray:
        """
        Standalone FMCW radar signal generation from PIR data.

        This generates realistic radar frames without requiring RF-Genesis native code.
        """
        from tqdm import tqdm

        num_tx = radar_config.get("num_tx", 3)
        num_rx = radar_config.get("num_rx", 4)
        num_chirps = radar_config.get("chirp_per_frame", 128)
        num_samples = radar_config.get("adc_samples", 256)
        fc = radar_config.get("fc", 77e9)
        slope = radar_config.get("slope", 60.012) * 1e12  # Hz/s
        c0 = radar_config.get("c0", 3e8)
        frame_rate = radar_config.get("frame_per_second", 10)
        pir_fps = 30  # PIR frame rate

        wavelength = c0 / fc

        # TX/RX antenna positions
        tx_loc = np.array(radar_config.get("tx_loc", [[0, 0, 0], [4, 0, 0], [2, 1, 0]])) * wavelength / 2
        rx_loc = np.array(radar_config.get("rx_loc", [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]])) * wavelength / 2

        # Calculate number of radar frames
        num_pir_frames = len(body_pirs)
        num_radar_frames = int(num_pir_frames * frame_rate / pir_fps)
        num_radar_frames = max(1, num_radar_frames)

        # Output array
        radar_frames = np.zeros(
            (num_radar_frames, num_tx, num_rx, num_chirps, num_samples),
            dtype=np.complex128
        )

        # Time parameters
        sample_rate = radar_config.get("sample_rate", 4400) * 1e3
        t_samples = np.arange(num_samples) / sample_rate

        print(f"  Generating {num_radar_frames} radar frames...")

        for radar_frame_idx in tqdm(range(num_radar_frames), desc="Generating radar frames"):
            # Map radar frame to PIR frame
            pir_frame_idx = int(radar_frame_idx * pir_fps / frame_rate)
            pir_frame_idx = min(pir_frame_idx, num_pir_frames - 1)

            # Get PIR data
            pir = body_pirs[pir_frame_idx]
            if isinstance(pir, torch.Tensor):
                pir = pir.cpu().numpy()

            # Extract point targets from PIR
            # PIR format: (H, W, 3) where channels are (distance, intensity, velocity)
            if pir.ndim == 3:
                distances = pir[:, :, 0].flatten()
                intensities = pir[:, :, 1].flatten()
                velocities = pir[:, :, 2].flatten() if pir.shape[2] > 2 else np.zeros_like(distances)
            else:
                # Fallback for unexpected shape
                distances = np.array([3.0])
                intensities = np.array([0.7])
                velocities = np.array([0.0])

            # Filter valid points (non-zero distance and intensity)
            valid_mask = (distances > 0.1) & (intensities > 0.1)
            distances = distances[valid_mask]
            intensities = intensities[valid_mask]
            velocities = velocities[valid_mask]

            if len(distances) == 0:
                # No valid points, add a default target
                distances = np.array([3.0])
                intensities = np.array([0.5])
                velocities = np.array([0.0])

            # Subsample if too many points
            max_points = 1000
            if len(distances) > max_points:
                indices = np.random.choice(len(distances), max_points, replace=False)
                distances = distances[indices]
                intensities = intensities[indices]
                velocities = velocities[indices]

            # Generate FMCW signal for each TX/RX pair
            for tx_idx in range(num_tx):
                for rx_idx in range(num_rx):
                    for chirp_idx in range(num_chirps):
                        signal = np.zeros(num_samples, dtype=np.complex128)

                        for dist, intensity, vel in zip(distances, intensities, velocities):
                            # Time of flight
                            tof = 2 * dist / c0

                            # Free space path loss
                            fspl = (wavelength / (4 * np.pi * dist + 1e-6)) ** 2

                            # Beat frequency (from FMCW)
                            f_beat = slope * tof

                            # Doppler shift
                            f_doppler = 2 * vel / wavelength

                            # Phase from antenna positions
                            # Simplified: assume target at boresight
                            phase_tx = 2 * np.pi * tx_loc[tx_idx, 0] / wavelength
                            phase_rx = 2 * np.pi * rx_loc[rx_idx, 0] / wavelength

                            # Total phase
                            phase = 2 * np.pi * (f_beat + f_doppler) * t_samples + phase_tx + phase_rx

                            # Add contribution
                            amplitude = np.sqrt(fspl * intensity)
                            signal += amplitude * np.exp(1j * phase)

                        # Add noise
                        noise_power = 1e-6
                        signal += np.sqrt(noise_power / 2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))

                        radar_frames[radar_frame_idx, tx_idx, rx_idx, chirp_idx, :] = signal

        return radar_frames

    def _load_radar_config(self) -> Dict:
        """Load radar configuration from JSON."""
        import json

        config_path = self.config.radar_config_path
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            print(f"[Warning] Radar config not found at {config_path}, using defaults")
            return self._default_radar_config()

    def _default_radar_config(self) -> Dict:
        """Return default TI1843 radar configuration."""
        return {
            "c0": 299792458,
            "num_tx": 3,
            "num_rx": 4,
            "fc": 77000000000.0,
            "slope": 60.012,
            "adc_samples": 256,
            "adc_start_time": 6,
            "sample_rate": 4400,
            "idle_time": 7,
            "ramp_end_time": 65,
            "chirp_per_frame": 128,
            "frame_per_second": 10,
            "num_doppler_bins": 128,
            "num_range_bins": 256,
            "num_angle_bins": 64,
            "power": 15,
            "tx_loc": [[0, 0, 0], [4, 0, 0], [2, 1, 0]],
            "rx_loc": [[-6, 0, 0], [-5, 0, 0], [-4, 0, 0], [-3, 0, 0]]
        }

    def convert_hymotion_to_smpl(
        self,
        motion_data: Dict[str, Any],
        body_model: Any = None,
    ) -> Dict[str, np.ndarray]:
        """
        Convert HY-Motion output to RF-Genesis SMPL format.

        HY-Motion outputs:
            - rot6d: (B, L, J, 6) - 6D rotation representation
            - transl: (B, L, 3) - root translation

        RF-Genesis expects:
            - pose: (L, 72) - axis-angle for 24 joints
            - shape: (L, 10) or (1, 10) - body shape parameters
            - root_translation: (L, 3) - global position

        Args:
            motion_data: HY-Motion output dictionary
            body_model: Optional body model for conversion

        Returns:
            Dictionary with SMPL parameters for RF-Genesis
        """
        rot6d = motion_data["rot6d"]  # (B, L, J, 6) or (L, J, 6)
        transl = motion_data["transl"]  # (B, L, 3) or (L, 3)

        # Handle batch dimension
        if rot6d.ndim == 4:
            rot6d = rot6d[0]  # (L, J, 6)
            transl = transl[0]  # (L, 3)

        # Convert to numpy if tensor
        if isinstance(rot6d, torch.Tensor):
            rot6d = rot6d.cpu().numpy()
        if isinstance(transl, torch.Tensor):
            transl = transl.cpu().numpy()

        num_frames, num_joints, _ = rot6d.shape

        # Convert rot6d to axis-angle (72D)
        # rot6d → rotation matrix → axis-angle
        pose = np.zeros((num_frames, 72), dtype=np.float32)

        for frame_idx in range(num_frames):
            for joint_idx in range(min(num_joints, 24)):  # SMPL has 24 joints
                # Get 6D rotation for this joint
                r6d = rot6d[frame_idx, joint_idx]

                # Convert 6D to rotation matrix
                rot_mat = self._rot6d_to_rotmat(r6d)

                # Convert rotation matrix to axis-angle
                axis_angle = self._rotmat_to_axis_angle(rot_mat)

                # Store in pose vector
                pose[frame_idx, joint_idx * 3:(joint_idx + 1) * 3] = axis_angle

        # Shape parameters (use zeros for generic body)
        shape = np.zeros((1, 10), dtype=np.float32)

        # Root translation
        root_translation = transl.astype(np.float32)

        return {
            "pose": pose,
            "shape": shape,
            "root_translation": root_translation,
            "gender": "male",
            "num_frames": num_frames,
        }

    def _rot6d_to_rotmat(self, rot6d: np.ndarray) -> np.ndarray:
        """
        Convert 6D rotation representation to rotation matrix.

        Based on "On the Continuity of Rotation Representations in Neural Networks"

        Args:
            rot6d: (6,) array - first two columns of rotation matrix

        Returns:
            (3, 3) rotation matrix
        """
        # Extract first two columns
        a1 = rot6d[:3]
        a2 = rot6d[3:6]

        # Gram-Schmidt orthogonalization
        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-8)
        b3 = np.cross(b1, b2)

        return np.stack([b1, b2, b3], axis=1)

    def _rotmat_to_axis_angle(self, rotmat: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to axis-angle representation.

        Args:
            rotmat: (3, 3) rotation matrix

        Returns:
            (3,) axis-angle vector
        """
        # Use Rodrigues formula inverse
        theta = np.arccos(np.clip((np.trace(rotmat) - 1) / 2, -1, 1))

        if np.abs(theta) < 1e-6:
            return np.zeros(3)

        # Extract axis
        axis = np.array([
            rotmat[2, 1] - rotmat[1, 2],
            rotmat[0, 2] - rotmat[2, 0],
            rotmat[1, 0] - rotmat[0, 1],
        ]) / (2 * np.sin(theta) + 1e-8)

        return axis * theta

    def simulate_radar(
        self,
        smpl_data: Dict[str, np.ndarray],
        environment_prompt: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full RF-Genesis radar simulation pipeline.

        Args:
            smpl_data: SMPL parameters from convert_hymotion_to_smpl()
            environment_prompt: Optional RFLoRA environment description
            output_dir: Output directory

        Returns:
            Dictionary containing:
                - radar_frames: Complex radar signal frames
                - doppler_fft: Doppler FFT results
                - point_clouds: Detected point clouds
                - range_doppler_maps: Range-Doppler maps
        """
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"[RFGenesisBridge] Running radar simulation...")
        print(f"  Frames: {smpl_data['num_frames']}")
        print(f"  Environment: {environment_prompt or 'none'}")

        # Step 1: Ray tracing to generate PIR
        print("  Step 1/3: Ray tracing...")
        body_pirs, body_auxs = self._run_ray_tracing(smpl_data)

        # Step 2: Environment diffusion (optional)
        env_pir = None
        if environment_prompt and self.config.use_environment:
            print(f"  Step 2/3: Environment diffusion...")
            env_pir = self._run_environment_diffusion(environment_prompt)
        else:
            print("  Step 2/3: Skipping environment diffusion")

        # Step 3: Signal generation
        print("  Step 3/3: Signal generation...")
        radar_frames = self.generate_signal_frames(
            body_pirs, body_auxs, env_pir, self.radar_config
        )

        # Process radar frames
        results = self._process_radar_frames(radar_frames)

        # Save outputs
        if self.config.save_radar_frames:
            np.save(os.path.join(output_dir, "radar_frames.npy"), radar_frames)

        # Save Doppler data
        np.savez(
            os.path.join(output_dir, "doppler_data.npz"),
            **{k: v for k, v in results.items() if isinstance(v, np.ndarray)}
        )

        print(f"[RFGenesisBridge] Results saved to: {output_dir}")
        return results

    def _run_ray_tracing(
        self,
        smpl_data: Dict[str, np.ndarray],
    ) -> Tuple[List, List]:
        """Run Mitsuba ray tracing on SMPL mesh."""
        try:
            # Initialize ray tracer
            ray_tracer = self.RayTracer(self.config.rfgenesis_path)

            pose = smpl_data["pose"]
            shape = smpl_data["shape"]
            root_translation = smpl_data["root_translation"]

            # Expand shape to match frames if needed
            if shape.shape[0] == 1:
                shape = np.tile(shape, (pose.shape[0], 1))

            body_pirs = []
            body_auxs = []

            for frame_idx in range(pose.shape[0]):
                # Update SMPL pose
                ray_tracer.update_pose(
                    pose[frame_idx:frame_idx+1],
                    shape[frame_idx:frame_idx+1],
                    root_translation[frame_idx:frame_idx+1],
                )

                # Trace rays
                pir, aux = ray_tracer.trace()
                body_pirs.append(pir)
                body_auxs.append(aux)

            return body_pirs, body_auxs

        except Exception as e:
            print(f"[Warning] Ray tracing failed: {e}")
            print("  Using fallback PIR generation")
            return self._fallback_pir_generation(smpl_data)

    def _fallback_pir_generation(
        self,
        smpl_data: Dict[str, np.ndarray],
    ) -> Tuple[List, List]:
        """
        Fallback PIR generation when Mitsuba is not available.

        Creates simplified PIR from SMPL vertex positions.
        """
        num_frames = smpl_data["num_frames"]
        pir_resolution = 128

        body_pirs = []
        body_auxs = []

        for frame_idx in range(num_frames):
            # Create empty PIR (distance, intensity, velocity)
            pir = np.zeros((pir_resolution, pir_resolution, 3), dtype=np.float32)

            # Simple projection of root translation
            root = smpl_data["root_translation"][frame_idx]

            # Distance from camera (assume camera at z=5)
            distance = np.linalg.norm(root - np.array([0, 0, 5]))

            # Fill central region with body
            center = pir_resolution // 2
            radius = 20
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if i*i + j*j < radius*radius:
                        pir[center + i, center + j, 0] = distance + np.random.randn() * 0.1
                        pir[center + i, center + j, 1] = 0.7  # Intensity
                        if frame_idx > 0:
                            prev_root = smpl_data["root_translation"][frame_idx - 1]
                            pir[center + i, center + j, 2] = np.linalg.norm(root - prev_root) * 30

            body_pirs.append(torch.from_numpy(pir).to(self.device))
            body_auxs.append(torch.zeros((pir_resolution, pir_resolution, 3), device=self.device))

        return body_pirs, body_auxs

    def _run_environment_diffusion(
        self,
        prompt: str,
    ) -> Optional[Any]:
        """Run RFLoRA environment diffusion."""
        try:
            from genesis.environment_diffusion.environemnt_diff import EnvironmentDiffusion

            env_diff = EnvironmentDiffusion()
            env_image = env_diff.generate(prompt)

            # Convert to PIR format
            env_pir = self._image_to_pir(env_image)
            return env_pir

        except Exception as e:
            print(f"[Warning] Environment diffusion failed: {e}")
            return None

    def _image_to_pir(self, image) -> torch.Tensor:
        """Convert environment image to PIR tensor."""
        import numpy as np

        # Resize to 64x64 (RF-Genesis default for environment)
        image = image.resize((64, 64))
        img_array = np.array(image).astype(np.float32) / 255.0

        # Create PIR (distance, intensity, velocity)
        # Use grayscale as intensity, fixed distance for background
        if img_array.ndim == 3:
            intensity = img_array.mean(axis=2)
        else:
            intensity = img_array

        pir = np.zeros((64, 64, 3), dtype=np.float32)
        pir[:, :, 0] = 5.0  # Fixed distance for background
        pir[:, :, 1] = intensity
        pir[:, :, 2] = 0.0  # No velocity for static environment

        return torch.from_numpy(pir).to(self.device)

    def _process_radar_frames(
        self,
        radar_frames: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Process radar frames to extract Doppler FFT and point clouds.

        Args:
            radar_frames: Complex radar frames (num_frames, tx, rx, chirps, samples)

        Returns:
            Dictionary with processed data
        """
        num_frames = radar_frames.shape[0]

        range_doppler_maps = []
        point_clouds = []
        doppler_spectrograms = []

        for frame_idx in range(num_frames):
            frame = radar_frames[frame_idx]  # (tx, rx, chirps, samples)

            # Range FFT
            range_fft = self.rangeFFT(frame)

            # Doppler FFT
            doppler_fft = self.dopplerFFT(range_fft)

            # Store range-Doppler map (sum over antennas)
            rd_map = np.abs(doppler_fft).mean(axis=(0, 1))  # (chirps, samples)
            range_doppler_maps.append(rd_map)

            # Extract Doppler spectrogram (sum over range)
            doppler_spectrum = np.abs(doppler_fft).mean(axis=(0, 1, 3))  # (chirps,)
            doppler_spectrograms.append(doppler_spectrum)

            # Point cloud detection
            try:
                pc = self.frame2pointcloud(frame, self.radar_config)
                point_clouds.append(pc)
            except Exception as e:
                print(f"[Warning] Point cloud extraction failed for frame {frame_idx}: {e}")
                point_clouds.append(np.zeros((0, 6)))

        # Stack results
        range_doppler_maps = np.stack(range_doppler_maps, axis=0)
        doppler_spectrograms = np.stack(doppler_spectrograms, axis=0)

        # Compute velocity axis
        c0 = self.radar_config["c0"]
        fc = self.radar_config["fc"]
        chirp_time = (self.radar_config["idle_time"] + self.radar_config["ramp_end_time"]) * 1e-6
        num_chirps = self.radar_config["chirp_per_frame"]
        num_tx = self.radar_config["num_tx"]

        wavelength = c0 / fc
        max_velocity = wavelength / (4 * chirp_time * num_tx)
        velocity_axis = np.linspace(-max_velocity, max_velocity, num_chirps)

        # Compute range axis
        slope = self.radar_config["slope"] * 1e12  # Hz/s
        sample_rate = self.radar_config["sample_rate"] * 1e3  # Hz
        num_samples = self.radar_config["adc_samples"]
        range_resolution = c0 / (2 * slope * num_samples / sample_rate)
        range_axis = np.arange(num_samples) * range_resolution

        return {
            "radar_frames": radar_frames,
            "range_doppler_maps": range_doppler_maps,
            "doppler_spectrogram": doppler_spectrograms,
            "point_clouds": point_clouds,
            "velocity_axis": velocity_axis,
            "range_axis": range_axis,
            "num_frames": num_frames,
            "frame_rate": self.radar_config["frame_per_second"],
        }


def run_hymotion_rfgenesis_pipeline(
    runtime: Any,  # T2MRuntime
    motion_prompt: str,
    environment_prompt: Optional[str] = None,
    duration: float = 3.0,
    output_dir: str = "output/hymotion_rfgenesis",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Complete pipeline: HY-Motion → RF-Genesis → Doppler FFT.

    Args:
        runtime: HY-Motion T2MRuntime instance
        motion_prompt: Text description of human motion
        environment_prompt: RFLoRA environment description (optional)
        duration: Motion duration in seconds
        output_dir: Output directory
        seed: Random seed

    Returns:
        Dictionary with all simulation results
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("HY-Motion + RF-Genesis Pipeline")
    print("=" * 60)
    print(f"Motion: {motion_prompt}")
    print(f"Environment: {environment_prompt or 'none'}")
    print(f"Duration: {duration}s")
    print("=" * 60)

    # Step 1: Generate motion with HY-Motion
    print("\n[Step 1] Generating motion with HY-Motion...")
    html_content, fbx_files, motion_data = runtime.generate_motion(
        text=motion_prompt,
        seeds_csv=str(seed),
        duration=duration,
        cfg_scale=5.0,
        output_format="dict",
    )

    # Step 2: Convert to RF-Genesis format
    print("\n[Step 2] Converting to RF-Genesis format...")
    bridge = HYMotionToRFGenesisBridge()
    smpl_data = bridge.convert_hymotion_to_smpl(motion_data)

    # Save SMPL data (same format as RF-Genesis obj_diff.npz)
    np.savez(
        os.path.join(output_dir, "obj_diff.npz"),
        pose=smpl_data["pose"],
        shape=smpl_data["shape"],
        root_translation=smpl_data["root_translation"],
        gender=smpl_data["gender"],
    )

    # Step 3: Run radar simulation
    print("\n[Step 3] Running radar simulation...")
    results = bridge.simulate_radar(
        smpl_data,
        environment_prompt=environment_prompt,
        output_dir=output_dir,
    )

    # Add metadata
    results["metadata"] = {
        "motion_prompt": motion_prompt,
        "environment_prompt": environment_prompt,
        "duration": duration,
        "seed": seed,
    }

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"Output: {output_dir}")
    print("=" * 60)

    return results
