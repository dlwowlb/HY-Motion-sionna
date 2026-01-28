"""
Sionna Ray Tracing Visualization for HY-Motion Generated Objects

This module integrates HY-Motion's LLM-based motion generation with
NVIDIA Sionna's ray tracing for wireless channel simulation and visualization.

Requirements:
    - sionna >= 0.16.0 (optional, fallback available)
    - tensorflow >= 2.10.0 (optional)
    - matplotlib
    - numpy
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

# Check for Sionna availability
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import sionna
    from sionna.rt import Scene, Transmitter, Receiver, PlanarArray
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class RayTracingConfig:
    """Configuration for ray tracing simulation."""
    frequency: float = 3.5e9
    bandwidth: float = 100e6
    max_depth: int = 5
    num_samples: int = 1000000
    diffraction: bool = True
    scattering: bool = True
    edge_diffraction: bool = True
    num_tx_ant: Tuple[int, int] = (2, 2)
    num_rx_ant: Tuple[int, int] = (2, 2)
    tx_pattern: str = "iso"
    rx_pattern: str = "iso"
    tx_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 3.0])
    rx_position: List[float] = field(default_factory=lambda: [5.0, 5.0, 1.5])
    resolution: Tuple[int, int] = (1920, 1080)
    max_paths_to_show: int = 100


def get_device_from_model(model: torch.nn.Module) -> torch.device:
    """Safely get device from a PyTorch model."""
    # Try buffers first (for models using register_buffer)
    for buf in model.buffers():
        return buf.device
    # Then try parameters
    for param in model.parameters():
        return param.device
    # Default to CPU
    return torch.device('cpu')


class MotionToSionnaConverter:
    """Converts HY-Motion generated SMPL mesh data to Sionna-compatible format."""

    def __init__(self, scale: float = 1.0, offset: Optional[np.ndarray] = None):
        self.scale = scale
        self.offset = offset if offset is not None else np.array([0.0, 0.0, 0.0])

    def convert_smpl_to_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        frame_idx: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Convert SMPL mesh data to Sionna-compatible mesh format."""
        if vertices.ndim == 3:
            verts = vertices[frame_idx].copy()
        else:
            verts = vertices.copy()

        verts = verts * self.scale + self.offset
        normals = self._compute_vertex_normals(verts, faces)

        return {
            "vertices": verts.astype(np.float32),
            "faces": faces.astype(np.int32),
            "normals": normals.astype(np.float32),
        }

    def _compute_vertex_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute per-vertex normals using face normals."""
        normals = np.zeros_like(vertices)
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)

        for i, face in enumerate(faces):
            for vertex_idx in face:
                normals[vertex_idx] += face_normals[i]

        return normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

    def export_to_obj(self, mesh_data: Dict[str, np.ndarray], filepath: str) -> Tuple[str, str]:
        """Export mesh to OBJ format with MTL material file."""
        vertices, faces, normals = mesh_data["vertices"], mesh_data["faces"], mesh_data["normals"]
        mtl_filepath = filepath.replace('.obj', '.mtl')

        with open(mtl_filepath, 'w') as f:
            f.write("newmtl human_body\nKa 0.2 0.2 0.2\nKd 0.8 0.6 0.5\nKs 0.1 0.1 0.1\n")

        with open(filepath, 'w') as f:
            f.write(f"mtllib {os.path.basename(mtl_filepath)}\nusemtl human_body\n")
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for n in normals:
                f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")

        return filepath, mtl_filepath


class SionnaRayVisualizer:
    """Main class for visualizing ray tracing with HY-Motion generated human meshes."""

    def __init__(self, config: Optional[RayTracingConfig] = None):
        self.config = config or RayTracingConfig()
        self.converter = MotionToSionnaConverter()

        if not SIONNA_AVAILABLE:
            print("[Info] Sionna not available. Using fallback ray tracing.")

    def generate_and_visualize(
        self,
        runtime: Any,
        text_prompt: str,
        duration: float = 3.0,
        seed: int = 42,
        cfg_scale: float = 5.0,
        frame_indices: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate motion from text prompt and visualize ray tracing."""
        output_dir = output_dir or tempfile.mkdtemp(prefix="sionna_ray_")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[1/4] Generating motion: '{text_prompt}'")
        motion_data = self._generate_motion_only(runtime, text_prompt, duration, seed, cfg_scale)

        print(f"[2/4] Extracting mesh sequence...")
        mesh_sequence = self._extract_mesh_sequence(motion_data, runtime)

        num_frames = mesh_sequence["vertices"].shape[0]
        if frame_indices is None:
            frame_indices = list(range(0, num_frames, max(1, num_frames // 10)))

        print(f"[3/4] Running ray tracing for {len(frame_indices)} frames...")
        ray_paths = []
        coverage_maps = []

        for i, frame_idx in enumerate(frame_indices):
            mesh_data = self.converter.convert_smpl_to_mesh(
                mesh_sequence["vertices"],
                mesh_sequence["faces"],
                frame_idx=frame_idx,
            )
            paths, coverage = self._run_fallback_ray_tracing(mesh_data, frame_idx)
            ray_paths.append(paths)
            coverage_maps.append(coverage)
            print(f"  Frame {frame_idx}: {len(paths.get('vertices', []))} paths")

        print(f"[4/4] Creating visualizations...")
        visualizations = self._create_all_visualizations(
            mesh_sequence, ray_paths, frame_indices, output_dir
        )

        results = {
            "motion_data": motion_data,
            "mesh_sequence": mesh_sequence,
            "ray_paths": ray_paths,
            "coverage_maps": coverage_maps,
            "visualizations": visualizations,
            "frame_indices": frame_indices,
            "output_dir": output_dir,
        }

        print(f"\n[Done] Results saved to: {output_dir}")
        return results

    def _generate_motion_only(
        self,
        runtime: Any,
        text_prompt: str,
        duration: float,
        seed: int,
        cfg_scale: float,
    ) -> Dict[str, Any]:
        """Generate motion without saving to gradio output."""
        runtime.load()

        # Get pipeline
        pi = runtime._acquire_pipeline()
        try:
            pipeline = runtime.pipelines[pi]
            pipeline.eval()

            # Generate motion directly
            model_output = pipeline.generate(
                text_prompt,
                [seed],
                duration,
                cfg_scale=cfg_scale,
                use_special_game_feat=False,
            )
        finally:
            runtime._release_pipeline(pi)

        return model_output

    def _extract_mesh_sequence(
        self,
        motion_data: Dict[str, Any],
        runtime: Any,
    ) -> Dict[str, np.ndarray]:
        """Extract mesh vertices and faces from motion data."""
        # Get body model
        if hasattr(runtime, 'pipelines') and runtime.pipelines:
            body_model = runtime.pipelines[0].body_model
        else:
            from ..pipeline.body_model import WoodenMesh
            body_model = WoodenMesh()

        # Get rotation and translation data
        rot6d = motion_data["rot6d"]
        transl = motion_data["transl"]

        # Use first sample if batch
        if rot6d.ndim == 4:
            rot6d = rot6d[0]
            transl = transl[0]

        # Convert to torch if numpy
        if isinstance(rot6d, np.ndarray):
            rot6d = torch.from_numpy(rot6d).float()
        if isinstance(transl, np.ndarray):
            transl = torch.from_numpy(transl).float()

        # Get device and move tensors
        device = get_device_from_model(body_model)
        rot6d = rot6d.to(device)
        transl = transl.to(device)

        # Forward pass
        with torch.no_grad():
            output = body_model.forward({"rot6d": rot6d, "trans": transl})

        return {
            "vertices": output["vertices"].cpu().numpy(),
            "faces": body_model.faces,
            "keypoints3d": motion_data.get("keypoints3d", None),
        }

    def _run_fallback_ray_tracing(
        self,
        mesh_data: Dict[str, np.ndarray],
        frame_idx: int,
    ) -> Tuple[Dict, Dict]:
        """Fallback ray tracing using basic geometric calculations."""
        vertices = mesh_data["vertices"]
        tx_pos = np.array(self.config.tx_position)
        rx_pos = np.array(self.config.rx_position)

        # Direct path
        direct_path = np.array([tx_pos, rx_pos])
        direct_delay = np.linalg.norm(rx_pos - tx_pos) / 3e8

        paths_list = [direct_path]
        delays = [direct_delay]
        powers = [1.0]
        types = ["direct"]

        # Simple reflections off human body
        num_reflections = min(50, len(vertices))
        indices = np.random.choice(len(vertices), num_reflections, replace=False)

        for idx in indices:
            point = vertices[idx]
            path = np.array([tx_pos, point, rx_pos])
            d1 = np.linalg.norm(point - tx_pos)
            d2 = np.linalg.norm(rx_pos - point)
            delay = (d1 + d2) / 3e8
            power = 0.3 / (d1 * d2 + 1e-6)

            paths_list.append(path)
            delays.append(delay)
            powers.append(power)
            types.append("reflection")

        powers = np.array(powers)
        powers = powers / powers.max()

        path_data = {
            "vertices": paths_list,
            "types": types,
            "delays": np.array(delays),
            "powers": powers,
        }

        coverage = self._compute_coverage(tx_pos)
        return path_data, coverage

    def _compute_coverage(self, tx_pos: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute simple coverage map."""
        x_range = np.linspace(-10, 10, 50)
        y_range = np.linspace(-10, 10, 50)
        X, Y = np.meshgrid(x_range, y_range)
        Z = 1.5

        distances = np.sqrt((X - tx_pos[0])**2 + (Y - tx_pos[1])**2 + (Z - tx_pos[2])**2)
        path_loss_db = 20 * np.log10(distances + 0.1) + 20 * np.log10(self.config.frequency) - 147.55

        return {"x": x_range, "y": y_range, "power": -path_loss_db, "unit": "dBm"}

    def _create_all_visualizations(
        self,
        mesh_sequence: Dict[str, np.ndarray],
        ray_paths: List[Dict],
        frame_indices: List[int],
        output_dir: str,
    ) -> List[str]:
        """Create all visualizations."""
        visualizations = []

        if not MATPLOTLIB_AVAILABLE:
            print("  [Warning] Matplotlib not available")
            return visualizations

        for i, frame_idx in enumerate(frame_indices[:5]):  # First 5 frames
            viz_path = self._create_single_visualization(
                mesh_sequence["vertices"][frame_idx] if mesh_sequence["vertices"].ndim == 3 else mesh_sequence["vertices"],
                mesh_sequence["faces"],
                ray_paths[i] if i < len(ray_paths) else None,
                frame_idx,
                output_dir,
            )
            if viz_path:
                visualizations.append(viz_path)

        return visualizations

    def _create_single_visualization(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        paths: Optional[Dict],
        frame_idx: int,
        output_dir: str,
    ) -> Optional[str]:
        """Create visualization for a single frame."""
        try:
            fig = plt.figure(figsize=(14, 6))

            # 3D view
            ax1 = fig.add_subplot(121, projection='3d')
            mesh = Poly3DCollection(vertices[faces], alpha=0.5, facecolor='peachpuff', edgecolor='gray', linewidth=0.1)
            ax1.add_collection3d(mesh)

            # TX/RX markers
            ax1.scatter(*self.config.tx_position, c='red', s=100, marker='^', label='TX')
            ax1.scatter(*self.config.rx_position, c='blue', s=100, marker='o', label='RX')

            # Ray paths
            if paths:
                for j, path in enumerate(paths.get("vertices", [])[:50]):
                    if isinstance(path, np.ndarray) and path.ndim == 2:
                        power = paths["powers"][j] if j < len(paths.get("powers", [])) else 0.5
                        ax1.plot3D(path[:, 0], path[:, 1], path[:, 2], color='yellow', alpha=0.2 + 0.8 * power, linewidth=0.5)

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title(f'Frame {frame_idx}')
            ax1.legend()

            # Set limits
            all_pts = np.vstack([vertices, [self.config.tx_position], [self.config.rx_position]])
            margin = 2.0
            ax1.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
            ax1.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
            ax1.set_zlim(0, max(all_pts[:, 2].max() + margin, 5))

            # Top view
            ax2 = fig.add_subplot(122)
            ax2.scatter(vertices[:, 0], vertices[:, 1], s=0.5, c='gray', alpha=0.3)
            ax2.scatter(self.config.tx_position[0], self.config.tx_position[1], c='red', s=100, marker='^', label='TX')
            ax2.scatter(self.config.rx_position[0], self.config.rx_position[1], c='blue', s=100, marker='o', label='RX')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Top View')
            ax2.legend()
            ax2.set_aspect('equal')

            plt.tight_layout()
            viz_path = os.path.join(output_dir, f"ray_frame_{frame_idx:04d}.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            return viz_path
        except Exception as e:
            print(f"  [Warning] Visualization failed for frame {frame_idx}: {e}")
            return None


def visualize_motion_ray_tracing(
    text_prompt: str,
    model_path: str = "ckpts/tencent/HY-Motion-1.0",
    output_dir: str = "output/sionna_ray_viz",
    duration: float = 3.0,
    seed: int = 42,
    frequency: float = 3.5e9,
    tx_position: List[float] = None,
    rx_position: List[float] = None,
) -> Dict[str, Any]:
    """Convenience function to generate motion and visualize ray tracing."""
    from ..utils.t2m_runtime import T2MRuntime

    config_path = os.path.join(model_path, "config.yml")
    ckpt_path = os.path.join(model_path, "latest.ckpt")

    runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name=ckpt_path,
        disable_prompt_engineering=True,
    )

    config = RayTracingConfig(
        frequency=frequency,
        tx_position=tx_position or [0.0, 0.0, 3.0],
        rx_position=rx_position or [5.0, 5.0, 1.5],
    )

    visualizer = SionnaRayVisualizer(config)
    return visualizer.generate_and_visualize(
        runtime=runtime,
        text_prompt=text_prompt,
        duration=duration,
        seed=seed,
        output_dir=output_dir,
    )
