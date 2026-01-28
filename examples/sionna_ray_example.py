#!/usr/bin/env python3
"""
HY-Motion + Sionna Ray Tracing Example

This example demonstrates how to:
1. Generate human motion from text using HY-Motion's LLM-based pipeline
2. Create a ray tracing scene with the generated human mesh
3. Run Sionna ray tracing simulation
4. Visualize the results with interactive 3D visualization

Usage:
    # Basic usage with default settings
    python examples/sionna_ray_example.py

    # Custom prompt and settings
    python examples/sionna_ray_example.py \\
        --prompt "A person walking forward slowly" \\
        --duration 5.0 \\
        --frequency 28e9 \\
        --output_dir output/ray_viz

    # With specific model path
    python examples/sionna_ray_example.py \\
        --model_path ckpts/tencent/HY-Motion-1.0 \\
        --prompt "A person waving their hand"

Requirements:
    - HY-Motion model checkpoint
    - Optional: sionna (pip install sionna) for full ray tracing
    - matplotlib for static visualizations
    - PIL/Pillow for GIF generation
"""

import argparse
import os
import sys
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description="HY-Motion + Sionna Ray Tracing Visualization Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate motion and visualize rays for indoor scene
  python examples/sionna_ray_example.py --scene indoor --prompt "person walking"

  # Outdoor scene with 5G mmWave frequency
  python examples/sionna_ray_example.py --scene outdoor --frequency 28e9

  # Custom TX/RX positions
  python examples/sionna_ray_example.py --tx 0,0,3 --rx 5,5,1.5
        """,
    )

    # Motion generation parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="ckpts/tencent/HY-Motion-1.0",
        help="Path to HY-Motion model checkpoint directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A person walking forward naturally",
        help="Text prompt describing the motion",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Motion duration in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for motion generation",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale (default: 5.0)",
    )

    # Ray tracing parameters
    parser.add_argument(
        "--frequency",
        type=float,
        default=3.5e9,
        help="Carrier frequency in Hz (default: 3.5e9 for 5G sub-6GHz)",
    )
    parser.add_argument(
        "--tx",
        type=str,
        default="0,0,3",
        help="Transmitter position as x,y,z (default: 0,0,3)",
    )
    parser.add_argument(
        "--rx",
        type=str,
        default="5,5,1.5",
        help="Receiver position as x,y,z (default: 5,5,1.5)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum ray reflection depth (default: 5)",
    )
    parser.add_argument(
        "--num_rays",
        type=int,
        default=100000,
        help="Number of rays to trace (default: 100000)",
    )

    # Scene parameters
    parser.add_argument(
        "--scene",
        type=str,
        default="indoor",
        choices=["indoor", "outdoor", "custom"],
        help="Scene type (default: indoor)",
    )
    parser.add_argument(
        "--room_size",
        type=str,
        default="10,10,3",
        help="Room dimensions as width,depth,height for indoor scene",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/sionna_ray_viz",
        help="Output directory (default: output/sionna_ray_viz)",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=10,
        help="Number of frames to visualize (default: 10)",
    )
    parser.add_argument(
        "--create_gif",
        action="store_true",
        help="Create animated GIF of the visualization",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Create interactive HTML visualization (default: True)",
    )

    # Debug options
    parser.add_argument(
        "--skip_model",
        action="store_true",
        help="Skip model loading (use random motion for testing)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Parse positions
    tx_position = [float(x) for x in args.tx.split(",")]
    rx_position = [float(x) for x in args.rx.split(",")]
    room_size = [float(x) for x in args.room_size.split(",")]

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("HY-Motion + Sionna Ray Tracing Visualization")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Duration: {args.duration}s")
    print(f"Frequency: {args.frequency/1e9:.2f} GHz")
    print(f"TX Position: {tx_position}")
    print(f"RX Position: {rx_position}")
    print(f"Scene Type: {args.scene}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Import modules
    from hymotion.sionna.ray_visualization import (
        SionnaRayVisualizer,
        RayTracingConfig,
        MotionToSionnaConverter,
    )
    from hymotion.sionna.scene_builder import (
        SionnaSceneBuilder,
        create_indoor_scene,
        create_outdoor_scene,
    )
    from hymotion.sionna.interactive_viz import (
        InteractiveRayVisualizer,
        create_html_visualization,
    )

    # Step 1: Initialize motion generator (if not skipping)
    runtime = None
    if not args.skip_model:
        print("\n[Step 1] Initializing HY-Motion runtime...")
        try:
            from hymotion.utils.t2m_runtime import T2MRuntime

            config_path = os.path.join(args.model_path, "config.yml")
            ckpt_path = os.path.join(args.model_path, "latest.ckpt")

            if not os.path.exists(config_path):
                print(f"  Warning: Config not found at {config_path}")
                print("  Will use random motion for demonstration")
                args.skip_model = True
            else:
                skip_loading = not os.path.exists(ckpt_path)
                if skip_loading:
                    print(f"  Warning: Checkpoint not found at {ckpt_path}")
                    print("  Using randomly initialized weights")

                runtime = T2MRuntime(
                    config_path=config_path,
                    ckpt_name=ckpt_path if not skip_loading else "latest.ckpt",
                    skip_model_loading=skip_loading,
                    disable_prompt_engineering=True,
                )
                print("  Runtime initialized successfully!")
        except Exception as e:
            print(f"  Error initializing runtime: {e}")
            print("  Falling back to random motion")
            args.skip_model = True

    # Step 2: Generate or create motion data
    print("\n[Step 2] Generating motion...")
    if args.skip_model or runtime is None:
        # Generate random motion for testing
        print("  Using random motion (model not loaded)")
        num_frames = int(args.duration * 30)  # 30 fps

        # Create random mesh data
        from hymotion.pipeline.body_model import WoodenMesh

        try:
            body_model = WoodenMesh()
            num_verts = body_model.num_verts
            faces = body_model.faces
        except Exception:
            # Fallback to simple mesh
            num_verts = 1000
            faces = np.random.randint(0, num_verts, (2000, 3))

        # Random walking motion
        vertices = np.zeros((num_frames, num_verts, 3))
        for f in range(num_frames):
            t = f / num_frames
            # Simple forward motion
            offset = np.array([t * 2, 0, 0])  # Walk forward
            vertices[f] = np.random.randn(num_verts, 3) * 0.1 + [0, 0, 1] + offset

        mesh_data = {
            "vertices": vertices,
            "faces": faces,
        }
        print(f"  Generated {num_frames} frames of random motion")
    else:
        # Use HY-Motion to generate actual motion
        print(f"  Generating motion: '{args.prompt}'")

        config = RayTracingConfig(
            frequency=args.frequency,
            tx_position=tx_position,
            rx_position=rx_position,
            max_depth=args.max_depth,
            num_samples=args.num_rays,
        )

        visualizer = SionnaRayVisualizer(config)

        # Generate motion
        results = visualizer.generate_and_visualize(
            runtime=runtime,
            text_prompt=args.prompt,
            duration=args.duration,
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            output_dir=args.output_dir,
        )

        mesh_data = results["mesh_sequence"]
        print(f"  Generated {mesh_data['vertices'].shape[0]} frames")

        # Results already contain ray tracing data
        if "ray_paths" in results:
            ray_paths = results["ray_paths"]
        else:
            ray_paths = None

    # Step 3: Build scene
    print("\n[Step 3] Building ray tracing scene...")
    if args.scene == "indoor":
        scene_builder = create_indoor_scene(
            room_width=room_size[0],
            room_depth=room_size[1],
            room_height=room_size[2],
            frequency=args.frequency,
            tx_position=tx_position,
            rx_position=rx_position,
        )
        print(f"  Created indoor scene: {room_size[0]}x{room_size[1]}x{room_size[2]}m")
    elif args.scene == "outdoor":
        scene_builder = create_outdoor_scene(
            frequency=args.frequency,
            tx_position=tx_position,
            rx_position=rx_position,
        )
        print("  Created outdoor scene with buildings")
    else:
        scene_builder = SionnaSceneBuilder(frequency=args.frequency)
        scene_builder.add_ground_plane()
        print("  Created custom scene")

    # Add human mesh to scene
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]

    if isinstance(vertices, torch.Tensor):
        vertices = vertices.numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.numpy()

    # Use middle frame for static scene
    mid_frame = vertices.shape[0] // 2 if vertices.ndim == 3 else 0
    if vertices.ndim == 3:
        frame_verts = vertices[mid_frame]
    else:
        frame_verts = vertices

    scene_builder.add_human_mesh(
        name="human_motion",
        vertices=frame_verts,
        faces=faces,
        position=[0.0, 0.0, 0.0],
    )
    print("  Added human mesh to scene")

    # Export scene
    scene_path = scene_builder.export_scene(args.output_dir)
    print(f"  Scene exported to: {scene_path}")

    # Step 4: Run ray tracing (if not already done)
    print("\n[Step 4] Running ray tracing simulation...")

    # Determine frames to process
    total_frames = vertices.shape[0] if vertices.ndim == 3 else 1
    frame_indices = np.linspace(0, total_frames - 1, args.num_frames, dtype=int).tolist()

    if args.skip_model or not runtime:
        # Run simple ray tracing
        config = RayTracingConfig(
            frequency=args.frequency,
            tx_position=tx_position,
            rx_position=rx_position,
            max_depth=args.max_depth,
            num_samples=args.num_rays,
        )

        converter = MotionToSionnaConverter()
        ray_visualizer = SionnaRayVisualizer(config)

        ray_paths = []
        for idx in frame_indices:
            if vertices.ndim == 3:
                frame_mesh = converter.convert_smpl_to_mesh(vertices, faces, frame_idx=idx)
            else:
                frame_mesh = converter.convert_smpl_to_mesh(vertices, faces, frame_idx=0)

            paths, coverage = ray_visualizer._run_fallback_ray_tracing(frame_mesh, idx)
            ray_paths.append(paths)
            print(f"  Frame {idx}: {len(paths.get('vertices', []))} ray paths computed")

    # Step 5: Create visualizations
    print("\n[Step 5] Creating visualizations...")

    if args.interactive:
        # Create interactive HTML visualization
        print("  Creating interactive HTML visualization...")

        scene_config = {
            "tx_position": tx_position,
            "rx_position": rx_position,
            "frequency": args.frequency,
        }

        # Prepare mesh data for visualization
        if vertices.ndim == 3:
            viz_vertices = vertices[frame_indices].tolist()
        else:
            viz_vertices = [vertices.tolist()]

        html_path = os.path.join(args.output_dir, "ray_visualization.html")

        viz = InteractiveRayVisualizer()
        viz.create_visualization(
            mesh_data={"vertices": viz_vertices, "faces": faces.tolist() if isinstance(faces, np.ndarray) else faces},
            ray_data=ray_paths,
            scene_config=scene_config,
            prompt=args.prompt,
            output_path=html_path,
        )
        print(f"  Interactive visualization: {html_path}")

    # Create static visualizations
    print("  Creating static visualizations...")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        for i, frame_idx in enumerate(frame_indices[:5]):  # First 5 frames
            fig = plt.figure(figsize=(12, 6))

            # 3D view
            ax1 = fig.add_subplot(121, projection='3d')

            if vertices.ndim == 3:
                frame_verts = vertices[frame_idx]
            else:
                frame_verts = vertices

            # Plot mesh
            mesh = Poly3DCollection(
                frame_verts[faces],
                alpha=0.5,
                facecolor='peachpuff',
                edgecolor='gray',
                linewidth=0.1,
            )
            ax1.add_collection3d(mesh)

            # Plot TX/RX
            ax1.scatter(*tx_position, c='red', s=100, marker='^', label='TX')
            ax1.scatter(*rx_position, c='blue', s=100, marker='o', label='RX')

            # Plot rays
            if ray_paths and i < len(ray_paths):
                paths = ray_paths[i].get("vertices", [])
                powers = ray_paths[i].get("powers", [])

                for j, path in enumerate(paths[:50]):  # First 50 paths
                    if isinstance(path, np.ndarray) and path.ndim == 2:
                        power = powers[j] if j < len(powers) else 0.5
                        alpha = 0.2 + 0.8 * power
                        ax1.plot3D(path[:, 0], path[:, 1], path[:, 2],
                                  color='yellow', alpha=alpha, linewidth=0.5)

            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_zlabel('Z (m)')
            ax1.set_title(f'Ray Tracing - Frame {frame_idx}')
            ax1.legend()

            # Set limits
            all_points = np.vstack([frame_verts, [tx_position], [rx_position]])
            margin = 2.0
            ax1.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
            ax1.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
            ax1.set_zlim(0, max(all_points[:, 2].max() + margin, 5))

            # Top view
            ax2 = fig.add_subplot(122)
            ax2.scatter(frame_verts[:, 0], frame_verts[:, 1], s=1, c='gray', alpha=0.3)
            ax2.scatter(tx_position[0], tx_position[1], c='red', s=100, marker='^', label='TX')
            ax2.scatter(rx_position[0], rx_position[1], c='blue', s=100, marker='o', label='RX')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            ax2.set_title('Top View')
            ax2.legend()
            ax2.set_aspect('equal')

            plt.tight_layout()
            fig_path = os.path.join(args.output_dir, f"ray_viz_frame_{frame_idx:04d}.png")
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"    Saved: {fig_path}")

    except ImportError as e:
        print(f"  Warning: Could not create static visualizations: {e}")

    # Create GIF if requested
    if args.create_gif:
        print("  Creating animated GIF...")
        try:
            from PIL import Image

            images = []
            for frame_idx in frame_indices[:10]:
                img_path = os.path.join(args.output_dir, f"ray_viz_frame_{frame_idx:04d}.png")
                if os.path.exists(img_path):
                    images.append(Image.open(img_path))

            if images:
                gif_path = os.path.join(args.output_dir, "ray_animation.gif")
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=200,
                    loop=0,
                )
                print(f"    Animation: {gif_path}")
        except ImportError:
            print("    Warning: PIL not available for GIF creation")

    # Summary
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  - {f} ({size:.1f} KB)")

    if args.interactive:
        print(f"\nOpen {os.path.join(args.output_dir, 'ray_visualization.html')} in a browser")
        print("to view the interactive 3D visualization.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
