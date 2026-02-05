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
        """Import RF-Genesis modules - REQUIRED (no fallback)."""
        try:
            from genesis.raytracing.pathtracer import RayTracer
            from genesis.raytracing.radar import Radar
            from genesis.raytracing.signal_generator import generate_signal_frames
            from genesis.visualization.pointcloud import frame2pointcloud, rangeFFT, dopplerFFT

            self.RayTracer = RayTracer
            self.Radar = Radar
            self.generate_signal_frames = generate_signal_frames
            self.frame2pointcloud = frame2pointcloud
            self.rangeFFT = rangeFFT
            self.dopplerFFT = dopplerFFT

            print("[RFGenesisBridge] RF-Genesis modules loaded (Mitsuba ray tracing enabled)")

        except ImportError as e:
            raise ImportError(
                f"RF-Genesis modules are REQUIRED but not available: {e}\n\n"
                "Please set up RF-Genesis:\n"
                "  1. git clone https://github.com/Asixa/RF-Genesis external/RF-Genesis\n"
                "  2. cd external/RF-Genesis && sh setup.sh\n"
                "  3. pip install mitsuba==3.5.2\n\n"
                "This will enable:\n"
                "  - Mitsuba ray tracing for accurate PIR generation\n"
                "  - RFLoRA environment diffusion\n"
                "  - GPU-accelerated signal generation"
            )

        # Import RFLoRA environment diffusion - REQUIRED
        try:
            from genesis.environment_diffusion.environemnt_diff import EnvironmentDiffusion
            self.EnvironmentDiffusion = EnvironmentDiffusion
            print("[RFGenesisBridge] RFLoRA environment diffusion enabled")
        except ImportError as e:
            raise ImportError(
                f"RFLoRA environment diffusion is REQUIRED but not available: {e}\n\n"
                "Please install dependencies:\n"
                "  pip install diffusers transformers accelerate\n"
                "  pip install peft  # For LoRA support"
            )

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
        environment_prompt: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full RF-Genesis radar simulation pipeline with Mitsuba + RFLoRA.

        This is the REQUIRED pipeline that uses:
        - Mitsuba ray tracing for accurate PIR generation
        - RFLoRA environment diffusion for realistic scenes
        - GPU-accelerated signal generation

        Args:
            smpl_data: SMPL parameters from convert_hymotion_to_smpl()
            environment_prompt: RFLoRA environment description (REQUIRED)
                               e.g., "a living room with a sofa and TV"
            output_dir: Output directory

        Returns:
            Dictionary containing:
                - radar_frames: Complex radar signal frames
                - doppler_fft: Doppler FFT results
                - point_clouds: Detected point clouds
                - range_doppler_maps: Range-Doppler maps
        """
        if not environment_prompt:
            raise ValueError(
                "environment_prompt is REQUIRED for RF-Genesis simulation.\n"
                "Please provide an environment description, e.g.:\n"
                '  environment_prompt="a living room with a sofa, TV, and coffee table"'
            )

        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"[RFGenesisBridge] Running full RF-Genesis pipeline...")
        print(f"  Frames: {smpl_data['num_frames']}")
        print(f"  Environment: {environment_prompt}")

        # Step 1: Mitsuba Ray tracing to generate PIR (REQUIRED)
        print("\n  Step 1/3: Mitsuba Ray Tracing...")
        body_pirs, body_auxs = self._run_ray_tracing(smpl_data)

        # Step 2: RFLoRA Environment diffusion (REQUIRED)
        print("\n  Step 2/3: RFLoRA Environment Diffusion...")
        env_pir = self._run_environment_diffusion(environment_prompt)

        # Step 3: Signal generation
        print("\n  Step 3/3: FMCW Signal Generation...")
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
        """
        Run Mitsuba ray tracing on SMPL mesh - REQUIRED.

        This uses RF-Genesis's pathtracer with Mitsuba for accurate
        PIR (Perspective Intensity Representation) generation.
        """
        from tqdm import tqdm

        # Initialize ray tracer (RF-Genesis RayTracer)
        ray_tracer = self.RayTracer()

        pose = smpl_data["pose"]
        shape = smpl_data["shape"]
        root_translation = smpl_data["root_translation"]
        num_frames = smpl_data["num_frames"]

        # Expand shape to match frames if needed
        if shape.shape[0] == 1:
            shape = np.tile(shape, (num_frames, 1))

        body_pirs = []
        body_auxs = []

        print(f"  Running Mitsuba ray tracing for {num_frames} frames...")

        for frame_idx in tqdm(range(num_frames), desc="Ray tracing"):
            # Update SMPL pose in the scene
            ray_tracer.update_pose(
                pose[frame_idx:frame_idx+1],
                shape[frame_idx:frame_idx+1],
                root_translation[frame_idx:frame_idx+1],
            )

            # Trace rays and get PIR
            pir, aux = ray_tracer.trace()
            body_pirs.append(pir)
            body_auxs.append(aux)

        return body_pirs, body_auxs

    def _run_environment_diffusion(
        self,
        prompt: str,
    ) -> torch.Tensor:
        """
        Run RFLoRA environment diffusion - REQUIRED.

        Uses Stable Diffusion with RFLoRA fine-tuned weights to generate
        realistic indoor environment images for radar simulation.

        Args:
            prompt: Environment description (e.g., "a living room with furniture")

        Returns:
            Environment PIR tensor
        """
        print(f"  Generating environment with RFLoRA: '{prompt}'")

        # Use pre-loaded EnvironmentDiffusion class
        env_diff = self.EnvironmentDiffusion()
        env_image = env_diff.generate(prompt)

        # Convert to PIR format
        env_pir = self._image_to_pir(env_image)

        print(f"  Environment PIR generated: {env_pir.shape}")
        return env_pir

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
    environment_prompt: str,
    duration: float = 3.0,
    output_dir: str = "output/hymotion_rfgenesis",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Complete pipeline: HY-Motion → RF-Genesis → Doppler FFT.

    This pipeline REQUIRES:
    - Mitsuba ray tracing for accurate PIR generation
    - RFLoRA environment diffusion for realistic scenes

    Args:
        runtime: HY-Motion T2MRuntime instance
        motion_prompt: Text description of human motion
        environment_prompt: RFLoRA environment description (REQUIRED)
                           e.g., "a living room with a sofa and TV"
        duration: Motion duration in seconds
        output_dir: Output directory
        seed: Random seed

    Returns:
        Dictionary with all simulation results

    Raises:
        ValueError: If environment_prompt is empty
        ImportError: If RF-Genesis modules are not available
    """
    if not environment_prompt:
        raise ValueError(
            "environment_prompt is REQUIRED for RF-Genesis simulation.\n"
            "Please provide an environment description, e.g.:\n"
            '  environment_prompt="a living room with a sofa, TV, and coffee table"'
        )

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("HY-Motion + RF-Genesis Pipeline")
    print("=" * 60)
    print(f"Motion: {motion_prompt}")
    print(f"Environment: {environment_prompt}")
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
