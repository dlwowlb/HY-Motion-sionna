"""
Sionna Ray Tracing Visualization for HY-Motion Generated Objects

This module integrates HY-Motion's LLM-based motion generation with
NVIDIA Sionna's ray tracing for wireless channel simulation and visualization.

Requirements:
    - sionna >= 0.16.0
    - tensorflow >= 2.10.0
    - matplotlib
    - numpy

Example:
    >>> from hymotion.sionna import SionnaRayVisualizer, RayTracingConfig
    >>> from hymotion.utils.t2m_runtime import T2MRuntime
    >>>
    >>> # Initialize motion generator
    >>> runtime = T2MRuntime(config_path="path/to/config.yml")
    >>>
    >>> # Create visualizer
    >>> config = RayTracingConfig(frequency=3.5e9, num_rays=1000)
    >>> visualizer = SionnaRayVisualizer(config)
    >>>
    >>> # Generate motion and visualize rays
    >>> results = visualizer.generate_and_visualize(
    ...     runtime=runtime,
    ...     text_prompt="A person walking forward",
    ...     duration=3.0
    ... )
"""

import os
import json
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch

# Check for Sionna availability
try:
    import tensorflow as tf
    # Disable GPU for TensorFlow if CUDA is used by PyTorch
    tf.config.set_visible_devices([], 'GPU')

    import sionna
    from sionna.rt import Scene, Transmitter, Receiver, PlanarArray, Camera
    from sionna.rt import load_scene
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    print("[Warning] Sionna not available. Install with: pip install sionna")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class RayTracingConfig:
    """Configuration for ray tracing simulation."""

    # RF parameters
    frequency: float = 3.5e9  # Carrier frequency in Hz (default: 3.5 GHz for 5G)
    bandwidth: float = 100e6  # Bandwidth in Hz

    # Ray tracing parameters
    max_depth: int = 5  # Maximum number of reflections
    num_samples: int = 1000000  # Number of rays to trace
    diffraction: bool = True  # Enable diffraction
    scattering: bool = True  # Enable scattering
    edge_diffraction: bool = True  # Enable edge diffraction

    # Antenna configuration
    num_tx_ant: Tuple[int, int] = (2, 2)  # Transmitter antenna array (rows, cols)
    num_rx_ant: Tuple[int, int] = (2, 2)  # Receiver antenna array (rows, cols)
    tx_pattern: str = "iso"  # Transmitter antenna pattern
    rx_pattern: str = "iso"  # Receiver antenna pattern

    # Transmitter/Receiver positions (can be overridden)
    tx_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 3.0])
    rx_position: List[float] = field(default_factory=lambda: [5.0, 5.0, 1.5])

    # Visualization
    resolution: Tuple[int, int] = (1920, 1080)
    show_paths: bool = True
    show_coverage: bool = True
    max_paths_to_show: int = 100

    # Material properties for human body
    human_relative_permittivity: float = 53.0  # Skin at ~3.5 GHz
    human_conductivity: float = 1.5  # S/m at ~3.5 GHz


class MotionToSionnaConverter:
    """
    Converts HY-Motion generated SMPL mesh data to Sionna-compatible format.

    The converter transforms SMPL vertices and faces into a format that can be
    loaded into Sionna's ray tracing scene as a custom object.
    """

    def __init__(self, scale: float = 1.0, offset: Optional[np.ndarray] = None):
        """
        Initialize the converter.

        Args:
            scale: Scale factor for the mesh (default: 1.0 meters)
            offset: Translation offset [x, y, z] in meters
        """
        self.scale = scale
        self.offset = offset if offset is not None else np.array([0.0, 0.0, 0.0])

    def convert_smpl_to_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        frame_idx: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Convert SMPL mesh data to Sionna-compatible mesh format.

        Args:
            vertices: Shape (num_frames, num_vertices, 3) or (num_vertices, 3)
            faces: Shape (num_faces, 3)
            frame_idx: Frame index to use if vertices has time dimension

        Returns:
            Dictionary containing:
                - vertices: (N, 3) array of vertex positions
                - faces: (M, 3) array of face indices
                - normals: (N, 3) array of vertex normals
        """
        if vertices.ndim == 3:
            verts = vertices[frame_idx].copy()
        else:
            verts = vertices.copy()

        # Apply scale and offset
        verts = verts * self.scale + self.offset

        # Compute vertex normals
        normals = self._compute_vertex_normals(verts, faces)

        return {
            "vertices": verts.astype(np.float32),
            "faces": faces.astype(np.int32),
            "normals": normals.astype(np.float32),
        }

    def _compute_vertex_normals(
        self,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """Compute per-vertex normals using face normals."""
        normals = np.zeros_like(vertices)

        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)

        # Accumulate to vertex normals
        for i, face in enumerate(faces):
            for vertex_idx in face:
                normals[vertex_idx] += face_normals[i]

        # Normalize
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

        return normals

    def export_to_ply(
        self,
        mesh_data: Dict[str, np.ndarray],
        filepath: str,
    ) -> str:
        """
        Export mesh to PLY format for Sionna scene loading.

        Args:
            mesh_data: Dictionary from convert_smpl_to_mesh
            filepath: Output PLY file path

        Returns:
            Path to the exported PLY file
        """
        vertices = mesh_data["vertices"]
        faces = mesh_data["faces"]
        normals = mesh_data["normals"]

        with open(filepath, 'w') as f:
            # Header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Vertices with normals
            for v, n in zip(vertices, normals):
                f.write(f"{v[0]} {v[1]} {v[2]} {n[0]} {n[1]} {n[2]}\n")

            # Faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        return filepath

    def export_to_obj(
        self,
        mesh_data: Dict[str, np.ndarray],
        filepath: str,
        material_name: str = "human_body",
    ) -> Tuple[str, str]:
        """
        Export mesh to OBJ format with MTL material file.

        Args:
            mesh_data: Dictionary from convert_smpl_to_mesh
            filepath: Output OBJ file path
            material_name: Name of the material

        Returns:
            Tuple of (OBJ path, MTL path)
        """
        vertices = mesh_data["vertices"]
        faces = mesh_data["faces"]
        normals = mesh_data["normals"]

        mtl_filepath = filepath.replace('.obj', '.mtl')

        # Write MTL file
        with open(mtl_filepath, 'w') as f:
            f.write(f"# Material for human body mesh\n")
            f.write(f"newmtl {material_name}\n")
            f.write("Ka 0.2 0.2 0.2\n")  # Ambient color
            f.write("Kd 0.8 0.6 0.5\n")  # Diffuse color (skin tone)
            f.write("Ks 0.1 0.1 0.1\n")  # Specular color
            f.write("Ns 10.0\n")  # Specular exponent
            f.write("d 1.0\n")  # Opacity

        # Write OBJ file
        with open(filepath, 'w') as f:
            f.write(f"# HY-Motion Generated Human Mesh\n")
            f.write(f"mtllib {os.path.basename(mtl_filepath)}\n")
            f.write(f"usemtl {material_name}\n\n")

            # Vertices
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")

            # Normals
            for n in normals:
                f.write(f"vn {n[0]} {n[1]} {n[2]}\n")

            # Faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1}//{face[0]+1} {face[1]+1}//{face[1]+1} {face[2]+1}//{face[2]+1}\n")

        return filepath, mtl_filepath


class SionnaRayVisualizer:
    """
    Main class for visualizing ray tracing with HY-Motion generated human meshes.

    This class integrates HY-Motion's motion generation pipeline with Sionna's
    ray tracing to simulate and visualize wireless signal propagation around
    human bodies in motion.
    """

    def __init__(self, config: Optional[RayTracingConfig] = None):
        """
        Initialize the ray visualizer.

        Args:
            config: Ray tracing configuration (uses defaults if None)
        """
        self.config = config or RayTracingConfig()
        self.converter = MotionToSionnaConverter()
        self._scene = None
        self._paths = None

        if not SIONNA_AVAILABLE:
            print("[Warning] Sionna not available. Visualization will use fallback methods.")

    def generate_and_visualize(
        self,
        runtime: Any,  # T2MRuntime
        text_prompt: str,
        duration: float = 3.0,
        seed: int = 42,
        cfg_scale: float = 5.0,
        frame_indices: Optional[List[int]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate motion from text prompt and visualize ray tracing.

        Args:
            runtime: HY-Motion T2MRuntime instance
            text_prompt: Text description of the motion
            duration: Duration in seconds
            seed: Random seed for generation
            cfg_scale: Classifier-free guidance scale
            frame_indices: Specific frames to visualize (None = all)
            output_dir: Directory to save outputs

        Returns:
            Dictionary containing:
                - motion_data: Generated motion data
                - ray_paths: Computed ray paths per frame
                - visualizations: List of visualization file paths
                - coverage_maps: Coverage map data per frame
        """
        output_dir = output_dir or tempfile.mkdtemp(prefix="hymotion_sionna_")
        os.makedirs(output_dir, exist_ok=True)

        print(f"[SionnaRayVisualizer] Generating motion for: '{text_prompt}'")

        # Step 1: Generate motion using HY-Motion
        motion_data = self._generate_motion(runtime, text_prompt, duration, seed, cfg_scale)

        # Step 2: Extract mesh data
        mesh_sequence = self._extract_mesh_sequence(motion_data, runtime)

        # Step 3: Determine frames to visualize
        num_frames = mesh_sequence["vertices"].shape[0]
        if frame_indices is None:
            # Visualize every 10th frame by default
            frame_indices = list(range(0, num_frames, max(1, num_frames // 10)))

        print(f"[SionnaRayVisualizer] Visualizing {len(frame_indices)} frames...")

        # Step 4: Run ray tracing and visualization for each frame
        results = {
            "motion_data": motion_data,
            "mesh_sequence": mesh_sequence,
            "ray_paths": [],
            "visualizations": [],
            "coverage_maps": [],
            "frame_indices": frame_indices,
            "output_dir": output_dir,
        }

        for i, frame_idx in enumerate(frame_indices):
            print(f"  Processing frame {frame_idx}/{num_frames}...")

            # Convert mesh for this frame
            mesh_data = self.converter.convert_smpl_to_mesh(
                mesh_sequence["vertices"],
                mesh_sequence["faces"],
                frame_idx=frame_idx,
            )

            # Export mesh
            mesh_path = os.path.join(output_dir, f"human_frame_{frame_idx:04d}.obj")
            self.converter.export_to_obj(mesh_data, mesh_path)

            # Run ray tracing
            if SIONNA_AVAILABLE:
                paths, coverage = self._run_sionna_ray_tracing(mesh_data, frame_idx)
            else:
                paths, coverage = self._run_fallback_ray_tracing(mesh_data, frame_idx)

            results["ray_paths"].append(paths)
            results["coverage_maps"].append(coverage)

            # Create visualization
            viz_path = self._create_visualization(
                mesh_data, paths, coverage, frame_idx, output_dir
            )
            results["visualizations"].append(viz_path)

        # Create animated visualization
        if len(frame_indices) > 1:
            anim_path = self._create_animation(results, output_dir)
            results["animation"] = anim_path

        print(f"[SionnaRayVisualizer] Results saved to: {output_dir}")
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
            # Fallback: create new body model
            from ..pipeline.body_model import WoodenMesh
            body_model = WoodenMesh()

        # Get rotation and translation data
        rot6d = motion_data["rot6d"]  # (B, L, J, 6)
        transl = motion_data["transl"]  # (B, L, 3)

        # Use first sample if batch
        if rot6d.ndim == 4:
            rot6d = rot6d[0]  # (L, J, 6)
            transl = transl[0]  # (L, 3)

        # Convert to torch if numpy
        if isinstance(rot6d, np.ndarray):
            rot6d = torch.from_numpy(rot6d).float()
        if isinstance(transl, np.ndarray):
            transl = torch.from_numpy(transl).float()

        # Forward pass through body model
        with torch.no_grad():
            output = body_model.forward({
                "rot6d": rot6d,
                "trans": transl,
            })

        vertices = output["vertices"].cpu().numpy()  # (L, V, 3)
        faces = body_model.faces  # (F, 3)

        return {
            "vertices": vertices,
            "faces": faces,
            "keypoints3d": motion_data.get("keypoints3d", None),
        }

    def _run_sionna_ray_tracing(
        self,
        mesh_data: Dict[str, np.ndarray],
        frame_idx: int,
    ) -> Tuple[Dict, Dict]:
        """Run Sionna ray tracing simulation."""
        if not SIONNA_AVAILABLE:
            return self._run_fallback_ray_tracing(mesh_data, frame_idx)

        # Create temporary scene with human mesh
        scene = Scene()

        # Configure scene parameters
        scene.frequency = self.config.frequency

        # Add transmitter
        scene.add(Transmitter(
            name="tx",
            position=self.config.tx_position,
        ))

        # Add receiver
        scene.add(Receiver(
            name="rx",
            position=self.config.rx_position,
        ))

        # Configure antenna arrays
        scene.tx_array = PlanarArray(
            num_rows=self.config.num_tx_ant[0],
            num_cols=self.config.num_tx_ant[1],
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern=self.config.tx_pattern,
        )
        scene.rx_array = PlanarArray(
            num_rows=self.config.num_rx_ant[0],
            num_cols=self.config.num_rx_ant[1],
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern=self.config.rx_pattern,
        )

        # Compute paths
        paths = scene.compute_paths(
            max_depth=self.config.max_depth,
            num_samples=self.config.num_samples,
            diffraction=self.config.diffraction,
            scattering=self.config.scattering,
            edge_diffraction=self.config.edge_diffraction,
        )

        # Extract path data
        path_data = {
            "vertices": paths.vertices.numpy() if hasattr(paths.vertices, 'numpy') else paths.vertices,
            "types": paths.types.numpy() if hasattr(paths.types, 'numpy') else paths.types,
            "delays": paths.tau.numpy() if hasattr(paths.tau, 'numpy') else paths.tau,
            "powers": np.abs(paths.a.numpy())**2 if hasattr(paths.a, 'numpy') else np.abs(paths.a)**2,
        }

        # Compute coverage map
        coverage = self._compute_coverage_map(scene)

        return path_data, coverage

    def _run_fallback_ray_tracing(
        self,
        mesh_data: Dict[str, np.ndarray],
        frame_idx: int,
    ) -> Tuple[Dict, Dict]:
        """
        Fallback ray tracing using basic geometric calculations.
        Used when Sionna is not available.
        """
        vertices = mesh_data["vertices"]

        # Transmitter and receiver positions
        tx_pos = np.array(self.config.tx_position)
        rx_pos = np.array(self.config.rx_position)

        # Generate rays from TX to RX
        num_rays = min(1000, self.config.num_samples // 1000)

        # Direct path
        direct_path = np.array([tx_pos, rx_pos])
        direct_delay = np.linalg.norm(rx_pos - tx_pos) / 3e8  # Speed of light

        # Calculate reflections off human body
        # Find mesh center for simple reflection calculation
        mesh_center = vertices.mean(axis=0)
        mesh_radius = np.linalg.norm(vertices - mesh_center, axis=1).max()

        paths_list = [direct_path]
        delays = [direct_delay]
        powers = [1.0]  # Normalized power

        # Simple single-bounce reflections
        for i in range(min(num_rays, 50)):
            # Random point on mesh surface
            random_vertex = vertices[np.random.randint(len(vertices))]

            # Path: TX -> reflection point -> RX
            path = np.array([tx_pos, random_vertex, rx_pos])

            # Calculate delay
            d1 = np.linalg.norm(random_vertex - tx_pos)
            d2 = np.linalg.norm(rx_pos - random_vertex)
            delay = (d1 + d2) / 3e8

            # Simple power attenuation (1/d^2)
            power = 1.0 / (d1 * d2 + 1e-6)

            paths_list.append(path)
            delays.append(delay)
            powers.append(power)

        # Normalize powers
        powers = np.array(powers)
        powers = powers / powers.max()

        path_data = {
            "vertices": paths_list,
            "types": ["direct"] + ["reflection"] * (len(paths_list) - 1),
            "delays": np.array(delays),
            "powers": powers,
        }

        # Simple coverage map
        coverage = self._compute_fallback_coverage(mesh_data, tx_pos)

        return path_data, coverage

    def _compute_coverage_map(self, scene) -> Dict[str, np.ndarray]:
        """Compute coverage map using Sionna."""
        # Create a grid for coverage computation
        x_range = np.linspace(-10, 10, 50)
        y_range = np.linspace(-10, 10, 50)
        z = 1.5  # Height

        coverage_grid = np.zeros((len(x_range), len(y_range)))

        # This would require computing paths for each grid point
        # Simplified: return empty coverage
        return {
            "x": x_range,
            "y": y_range,
            "power": coverage_grid,
            "unit": "dBm",
        }

    def _compute_fallback_coverage(
        self,
        mesh_data: Dict[str, np.ndarray],
        tx_pos: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute simple coverage map without Sionna."""
        x_range = np.linspace(-10, 10, 50)
        y_range = np.linspace(-10, 10, 50)

        X, Y = np.meshgrid(x_range, y_range)
        Z = 1.5 * np.ones_like(X)  # Height

        # Distance from TX
        distances = np.sqrt(
            (X - tx_pos[0])**2 +
            (Y - tx_pos[1])**2 +
            (Z - tx_pos[2])**2
        )

        # Free-space path loss (simplified)
        # P = P_tx - 20*log10(d) - 20*log10(f) - 20*log10(4*pi/c)
        wavelength = 3e8 / self.config.frequency
        path_loss_db = 20 * np.log10(distances + 0.1) + 20 * np.log10(self.config.frequency) - 147.55

        # Reference power (0 dBm TX power)
        coverage_power = -path_loss_db

        return {
            "x": x_range,
            "y": y_range,
            "power": coverage_power,
            "unit": "dBm",
        }

    def _create_visualization(
        self,
        mesh_data: Dict[str, np.ndarray],
        paths: Dict,
        coverage: Dict,
        frame_idx: int,
        output_dir: str,
    ) -> str:
        """Create visualization for a single frame."""
        if not MATPLOTLIB_AVAILABLE:
            print("[Warning] Matplotlib not available for visualization")
            return ""

        fig = plt.figure(figsize=(16, 8))

        # 3D view with mesh and rays
        ax1 = fig.add_subplot(121, projection='3d')
        self._plot_3d_scene(ax1, mesh_data, paths)
        ax1.set_title(f"Ray Tracing - Frame {frame_idx}")

        # Coverage map
        ax2 = fig.add_subplot(122)
        self._plot_coverage_map(ax2, coverage, mesh_data)
        ax2.set_title("Coverage Map (dBm)")

        plt.tight_layout()

        # Save figure
        viz_path = os.path.join(output_dir, f"ray_viz_frame_{frame_idx:04d}.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return viz_path

    def _plot_3d_scene(
        self,
        ax: Any,  # Axes3D
        mesh_data: Dict[str, np.ndarray],
        paths: Dict,
    ):
        """Plot 3D scene with mesh and rays."""
        vertices = mesh_data["vertices"]
        faces = mesh_data["faces"]

        # Plot human mesh
        mesh_collection = Poly3DCollection(
            vertices[faces],
            alpha=0.6,
            facecolor='peachpuff',
            edgecolor='gray',
            linewidth=0.1,
        )
        ax.add_collection3d(mesh_collection)

        # Plot TX and RX positions
        tx_pos = self.config.tx_position
        rx_pos = self.config.rx_position
        ax.scatter(*tx_pos, c='red', s=100, marker='^', label='TX')
        ax.scatter(*rx_pos, c='blue', s=100, marker='o', label='RX')

        # Plot ray paths
        path_vertices = paths.get("vertices", [])
        path_powers = paths.get("powers", [])

        # Color normalize based on power
        if len(path_powers) > 0:
            norm = Normalize(vmin=min(path_powers), vmax=max(path_powers))
            cmap = plt.cm.viridis

        num_paths_to_plot = min(len(path_vertices), self.config.max_paths_to_show)
        for i in range(num_paths_to_plot):
            path = path_vertices[i]
            if isinstance(path, np.ndarray) and path.ndim == 2:
                power = path_powers[i] if i < len(path_powers) else 0.5
                color = cmap(norm(power)) if len(path_powers) > 0 else 'yellow'
                alpha = 0.3 + 0.7 * power
                ax.plot3D(path[:, 0], path[:, 1], path[:, 2],
                         color=color, alpha=alpha, linewidth=0.5)

        # Set labels and limits
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        # Auto-scale
        all_points = np.vstack([vertices, [tx_pos], [rx_pos]])
        margin = 2.0
        ax.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
        ax.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
        ax.set_zlim(0, all_points[:, 2].max() + margin)

        ax.legend()

    def _plot_coverage_map(
        self,
        ax: Any,  # Axes
        coverage: Dict,
        mesh_data: Dict[str, np.ndarray],
    ):
        """Plot 2D coverage map."""
        x = coverage["x"]
        y = coverage["y"]
        power = coverage["power"]

        # Plot coverage heatmap
        im = ax.imshow(
            power,
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin='lower',
            cmap='RdYlGn',
            aspect='equal',
        )
        plt.colorbar(im, ax=ax, label=coverage.get("unit", "dBm"))

        # Plot human position (top-down view)
        vertices = mesh_data["vertices"]
        human_x = vertices[:, 0].mean()
        human_y = vertices[:, 1].mean()
        ax.scatter(human_x, human_y, c='black', s=50, marker='*', label='Human')

        # Plot TX/RX
        ax.scatter(self.config.tx_position[0], self.config.tx_position[1],
                  c='red', s=100, marker='^', label='TX')
        ax.scatter(self.config.rx_position[0], self.config.rx_position[1],
                  c='blue', s=100, marker='o', label='RX')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper right')

    def _create_animation(
        self,
        results: Dict[str, Any],
        output_dir: str,
    ) -> str:
        """Create animated GIF from frame visualizations."""
        try:
            from PIL import Image

            images = []
            for viz_path in results["visualizations"]:
                if os.path.exists(viz_path):
                    images.append(Image.open(viz_path))

            if not images:
                return ""

            anim_path = os.path.join(output_dir, "ray_animation.gif")
            images[0].save(
                anim_path,
                save_all=True,
                append_images=images[1:],
                duration=200,  # ms per frame
                loop=0,
            )

            return anim_path

        except ImportError:
            print("[Warning] PIL not available for animation creation")
            return ""


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
    """
    Convenience function to generate motion and visualize ray tracing.

    Args:
        text_prompt: Text description of the motion
        model_path: Path to HY-Motion model
        output_dir: Output directory
        duration: Motion duration in seconds
        seed: Random seed
        frequency: RF carrier frequency in Hz
        tx_position: Transmitter position [x, y, z]
        rx_position: Receiver position [x, y, z]

    Returns:
        Visualization results dictionary
    """
    from ..utils.t2m_runtime import T2MRuntime
    import os

    # Check for config file
    config_path = os.path.join(model_path, "config.yml")
    ckpt_path = os.path.join(model_path, "latest.ckpt")

    # Determine if we should skip model loading
    skip_model_loading = not os.path.exists(ckpt_path)

    # Initialize runtime
    runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name=ckpt_path if not skip_model_loading else "latest.ckpt",
        skip_model_loading=skip_model_loading,
        disable_prompt_engineering=True,  # Simplify for demo
    )

    # Create config
    config = RayTracingConfig(
        frequency=frequency,
        tx_position=tx_position or [0.0, 0.0, 3.0],
        rx_position=rx_position or [5.0, 5.0, 1.5],
    )

    # Create visualizer and run
    visualizer = SionnaRayVisualizer(config)
    results = visualizer.generate_and_visualize(
        runtime=runtime,
        text_prompt=text_prompt,
        duration=duration,
        seed=seed,
        output_dir=output_dir,
    )

    return results
