#!/usr/bin/env python3
"""
RF-Genesis Integration Example for HY-Motion

This example demonstrates how to:
1. Generate 3D human motion from text prompts using HY-Motion
2. Simulate RF sensing data using RF-Genesis style ray tracing
3. Extract Doppler, CIR, and CSI data from the motion

Requirements:
    - HY-Motion model weights (ckpts/tencent/HY-Motion-1.0)
    - scipy for signal processing
    - matplotlib for visualization

Usage:
    python examples/rfgenesis_example.py --prompt "A person walking forward"
    python examples/rfgenesis_example.py --prompt "A person waving hands" --frequency 60e9
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="RF-Genesis Integration Example")
    parser.add_argument(
        "--prompt", type=str, default="A person walking forward slowly",
        help="Text prompt describing the motion"
    )
    parser.add_argument(
        "--duration", type=float, default=3.0,
        help="Motion duration in seconds"
    )
    parser.add_argument(
        "--frequency", type=float, default=77e9,
        help="Radar/RF frequency in Hz (default: 77 GHz mmWave)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/rfgenesis_example",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--model_path", type=str, default="ckpts/tencent/HY-Motion-1.0",
        help="Path to HY-Motion model"
    )
    parser.add_argument(
        "--demo_mode", action="store_true",
        help="Run in demo mode with synthetic motion data"
    )
    return parser.parse_args()


def create_synthetic_motion(num_frames: int = 90, num_vertices: int = 6890):
    """
    Create synthetic walking motion for demo purposes.

    This generates a simple approximation of human walking motion
    without requiring the full HY-Motion model.
    """
    print("[Demo Mode] Creating synthetic walking motion...")

    # Create a basic human-like point cloud that moves forward
    vertices = np.zeros((num_frames, num_vertices, 3))

    # Initialize base shape (simplified human silhouette)
    # Head
    head_verts = 500
    head_center = np.array([0, 0, 1.7])
    vertices[0, :head_verts] = head_center + np.random.randn(head_verts, 3) * 0.1

    # Torso
    torso_verts = 2000
    torso_center = np.array([0, 0, 1.2])
    vertices[0, head_verts:head_verts+torso_verts] = torso_center + np.random.randn(torso_verts, 3) * np.array([0.15, 0.1, 0.3])

    # Arms
    arm_verts = 1000
    for arm_idx, arm_x in enumerate([-0.3, 0.3]):
        start = head_verts + torso_verts + arm_idx * arm_verts
        arm_center = np.array([arm_x, 0, 1.2])
        vertices[0, start:start+arm_verts] = arm_center + np.random.randn(arm_verts, 3) * np.array([0.05, 0.05, 0.25])

    # Legs
    leg_verts = 1195  # Remaining vertices
    for leg_idx, leg_x in enumerate([-0.1, 0.1]):
        start = head_verts + torso_verts + 2 * arm_verts + leg_idx * (leg_verts // 2)
        end = start + leg_verts // 2
        leg_center = np.array([leg_x, 0, 0.5])
        vertices[0, start:end] = leg_center + np.random.randn(end - start, 3) * np.array([0.05, 0.05, 0.4])

    # Fill remaining with random body points
    remaining = num_vertices - (head_verts + torso_verts + 2 * arm_verts + leg_verts)
    if remaining > 0:
        vertices[0, -remaining:] = np.array([0, 0, 1.0]) + np.random.randn(remaining, 3) * 0.3

    # Animate: walking forward with arm swing and leg movement
    walking_speed = 1.0  # m/s
    fps = 30.0
    arm_swing_freq = 1.5  # Hz
    leg_swing_freq = 1.5  # Hz

    for t in range(1, num_frames):
        time = t / fps

        # Copy previous frame
        vertices[t] = vertices[0].copy()

        # Forward translation
        forward_motion = walking_speed * time
        vertices[t, :, 1] += forward_motion

        # Arm swing (sinusoidal)
        arm_swing = 0.2 * np.sin(2 * np.pi * arm_swing_freq * time)

        # Left arm swings opposite to right arm
        left_arm_start = head_verts + torso_verts
        left_arm_end = left_arm_start + arm_verts
        vertices[t, left_arm_start:left_arm_end, 1] += arm_swing

        right_arm_start = left_arm_end
        right_arm_end = right_arm_start + arm_verts
        vertices[t, right_arm_start:right_arm_end, 1] -= arm_swing

        # Leg swing
        leg_swing = 0.15 * np.sin(2 * np.pi * leg_swing_freq * time)

        left_leg_start = head_verts + torso_verts + 2 * arm_verts
        left_leg_end = left_leg_start + leg_verts // 2
        vertices[t, left_leg_start:left_leg_end, 1] += leg_swing

        right_leg_start = left_leg_end
        right_leg_end = right_leg_start + leg_verts // 2
        vertices[t, right_leg_start:right_leg_end, 1] -= leg_swing

        # Vertical bobbing
        bob = 0.02 * np.sin(2 * np.pi * 2 * leg_swing_freq * time)
        vertices[t, :, 2] += bob

    # Create simple faces (triangular mesh)
    # For simplicity, use Delaunay-like triangulation on subsampled points
    num_faces = 10000
    faces = np.random.randint(0, num_vertices, size=(num_faces, 3))

    return {
        "vertices": vertices,
        "faces": faces,
    }


def visualize_doppler(doppler_data: dict, output_dir: str):
    """Visualize Doppler spectrum data."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Time-Doppler spectrogram
    ax = axes[0, 0]
    spectrogram = doppler_data["spectrogram"]
    time_axis = doppler_data["time_axis"]
    velocity_axis = doppler_data["velocity_axis"]

    im = ax.imshow(
        spectrogram.T,
        aspect="auto",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], velocity_axis[0], velocity_axis[-1]],
        cmap="jet",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Time-Doppler Spectrogram")
    plt.colorbar(im, ax=ax, label="Intensity")

    # Micro-Doppler signature
    ax = axes[0, 1]
    micro_doppler = doppler_data["micro_doppler"]
    im = ax.imshow(
        micro_doppler.T,
        aspect="auto",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], velocity_axis[0], velocity_axis[-1]],
        cmap="hot",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Micro-Doppler Signature")
    plt.colorbar(im, ax=ax, label="Intensity")

    # Range-Doppler map
    ax = axes[1, 0]
    range_doppler = doppler_data["range_doppler"]
    range_axis = doppler_data["range_axis"]
    im = ax.imshow(
        range_doppler.T,
        aspect="auto",
        origin="lower",
        extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Range-Doppler Map")
    plt.colorbar(im, ax=ax, label="Intensity")

    # Average Doppler profile over time
    ax = axes[1, 1]
    avg_doppler = spectrogram.mean(axis=0)
    ax.plot(velocity_axis, avg_doppler, "b-", linewidth=2)
    ax.fill_between(velocity_axis, 0, avg_doppler, alpha=0.3)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Average Intensity")
    ax.set_title("Average Doppler Profile")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "doppler_analysis.png"), dpi=150)
    plt.close()
    print(f"  Saved: doppler_analysis.png")


def visualize_cir(cir_data: dict, output_dir: str):
    """Visualize Channel Impulse Response data."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    delay_axis = cir_data["delay_axis"] * 1e9  # Convert to nanoseconds
    cir_magnitude = cir_data["cir_magnitude"]
    cir_phase = cir_data["cir_phase"]

    # Time-varying CIR magnitude
    ax = axes[0, 0]
    im = ax.imshow(
        cir_magnitude,
        aspect="auto",
        origin="lower",
        extent=[delay_axis[0], delay_axis[-1], 0, cir_magnitude.shape[0]],
        cmap="plasma",
    )
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Frame Index")
    ax.set_title("Time-varying CIR Magnitude")
    plt.colorbar(im, ax=ax, label="Magnitude")

    # Power Delay Profile
    ax = axes[0, 1]
    pdp = cir_data["power_delay_profile"]
    pdp_db = 10 * np.log10(pdp + 1e-12)
    ax.plot(delay_axis, pdp_db, "b-", linewidth=2)
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Power (dB)")
    ax.set_title("Power Delay Profile (Average)")
    ax.grid(True, alpha=0.3)

    # CIR at specific frames
    ax = axes[1, 0]
    num_frames = cir_magnitude.shape[0]
    frames_to_plot = [0, num_frames // 4, num_frames // 2, 3 * num_frames // 4, num_frames - 1]
    for frame_idx in frames_to_plot:
        ax.plot(delay_axis, cir_magnitude[frame_idx], label=f"Frame {frame_idx}", alpha=0.7)
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Magnitude")
    ax.set_title("CIR at Different Frames")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Phase evolution at first tap
    ax = axes[1, 1]
    first_tap_phase = cir_phase[:, np.argmax(cir_magnitude.mean(axis=0))]
    ax.plot(first_tap_phase, "g-", linewidth=1.5)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Phase Evolution (Strongest Tap)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cir_analysis.png"), dpi=150)
    plt.close()
    print(f"  Saved: cir_analysis.png")


def visualize_csi(csi_data: dict, output_dir: str):
    """Visualize Channel State Information data."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    freq_axis = (csi_data["frequency_axis"] - csi_data["center_frequency"]) / 1e6  # MHz offset
    csi_magnitude = csi_data["csi_magnitude"]
    csi_phase = csi_data["csi_phase_unwrapped"]

    # Time-varying CSI magnitude
    ax = axes[0, 0]
    im = ax.imshow(
        csi_magnitude,
        aspect="auto",
        origin="lower",
        extent=[freq_axis[0], freq_axis[-1], 0, csi_magnitude.shape[0]],
        cmap="viridis",
    )
    ax.set_xlabel("Frequency Offset (MHz)")
    ax.set_ylabel("Frame Index")
    ax.set_title("Time-varying CSI Magnitude")
    plt.colorbar(im, ax=ax, label="Magnitude")

    # Time-varying CSI phase
    ax = axes[0, 1]
    im = ax.imshow(
        csi_phase,
        aspect="auto",
        origin="lower",
        extent=[freq_axis[0], freq_axis[-1], 0, csi_phase.shape[0]],
        cmap="twilight",
    )
    ax.set_xlabel("Frequency Offset (MHz)")
    ax.set_ylabel("Frame Index")
    ax.set_title("Time-varying CSI Phase (Unwrapped)")
    plt.colorbar(im, ax=ax, label="Phase (rad)")

    # CSI magnitude variance per subcarrier
    ax = axes[1, 0]
    variance = csi_data["csi_amplitude_variance"]
    ax.bar(freq_axis, variance, width=(freq_axis[1] - freq_axis[0]) * 0.8, color="steelblue")
    ax.set_xlabel("Frequency Offset (MHz)")
    ax.set_ylabel("Variance")
    ax.set_title("CSI Magnitude Variance per Subcarrier")
    ax.grid(True, alpha=0.3, axis="y")

    # Average frequency response
    ax = axes[1, 1]
    avg_magnitude = csi_magnitude.mean(axis=0)
    avg_magnitude_db = 20 * np.log10(avg_magnitude + 1e-12)
    ax.plot(freq_axis, avg_magnitude_db, "b-", linewidth=2)
    ax.set_xlabel("Frequency Offset (MHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Average Frequency Response")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "csi_analysis.png"), dpi=150)
    plt.close()
    print(f"  Saved: csi_analysis.png")


def visualize_point_cloud(point_cloud: dict, output_dir: str):
    """Visualize radar point cloud."""
    fig = plt.figure(figsize=(14, 5))

    points = point_cloud["points"]
    velocities = point_cloud["velocities"]
    frame_indices = point_cloud["frame_indices"]

    # 3D scatter plot
    ax = fig.add_subplot(131, projection="3d")
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=velocities, cmap="coolwarm", s=1, alpha=0.5
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Radar Point Cloud (color = velocity)")
    plt.colorbar(scatter, ax=ax, label="Radial Velocity (m/s)", shrink=0.5)

    # XY projection
    ax = fig.add_subplot(132)
    scatter = ax.scatter(
        points[:, 0], points[:, 1],
        c=frame_indices, cmap="viridis", s=1, alpha=0.3
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Point Cloud (XY plane, color = frame)")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, label="Frame Index")

    # Velocity histogram
    ax = fig.add_subplot(133)
    ax.hist(velocities, bins=50, color="steelblue", edgecolor="white", alpha=0.7)
    ax.set_xlabel("Radial Velocity (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("Velocity Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "point_cloud_analysis.png"), dpi=150)
    plt.close()
    print(f"  Saved: point_cloud_analysis.png")


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("RF-Genesis Integration Example for HY-Motion")
    print("=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Duration: {args.duration}s")
    print(f"Frequency: {args.frequency / 1e9:.1f} GHz")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Import RF-Genesis modules
    from hymotion.rfgenesis import RFGenesisSimulator, RFConfig, RadarConfig, DopplerConfig

    # Configure RF simulation
    radar_config = RadarConfig(start_frequency=args.frequency)
    doppler_config = DopplerConfig()

    rf_config = RFConfig(
        frequency=args.frequency,
        radar=radar_config,
        doppler=doppler_config,
        tx_position=[0.0, -3.0, 1.5],  # Radar position
        rx_position=[0.0, -3.0, 1.5],  # Same position (monostatic)
    )

    # Create simulator
    simulator = RFGenesisSimulator(rf_config)

    if args.demo_mode:
        # Use synthetic motion data
        print("\n[Demo Mode] Using synthetic motion data...")
        num_frames = int(args.duration * 30)  # 30 fps
        mesh_sequence = create_synthetic_motion(num_frames=num_frames)

        # Simulate RF data
        results = simulator.simulate_from_mesh_sequence(
            mesh_sequence,
            output_dir=args.output_dir,
        )
        results["metadata"] = {
            "text_prompt": args.prompt,
            "duration": args.duration,
            "seed": args.seed,
            "frequency": args.frequency,
            "num_frames": num_frames,
            "mode": "demo",
        }
    else:
        # Use HY-Motion to generate motion
        print("\n[Full Mode] Loading HY-Motion model...")
        try:
            from hymotion.utils.t2m_runtime import T2MRuntime

            config_path = os.path.join(args.model_path, "config.yml")
            if not os.path.exists(config_path):
                print(f"[Warning] Config not found at {config_path}")
                print("[Warning] Falling back to demo mode...")
                args.demo_mode = True
                return main()

            runtime = T2MRuntime(
                config_path=config_path,
                disable_prompt_engineering=True,
            )

            # Generate motion and simulate RF
            results = simulator.simulate_from_motion(
                runtime=runtime,
                text_prompt=args.prompt,
                duration=args.duration,
                seed=args.seed,
                output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"[Error] Failed to load HY-Motion: {e}")
            print("[Warning] Falling back to demo mode...")
            args.demo_mode = True
            return main()

    # Visualize results
    print("\nGenerating visualizations...")

    if "doppler" in results:
        visualize_doppler(results["doppler"], args.output_dir)

    if "cir" in results:
        visualize_cir(results["cir"], args.output_dir)

    if "csi" in results:
        visualize_csi(results["csi"], args.output_dir)

    if "point_cloud" in results:
        visualize_point_cloud(results["point_cloud"], args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if "doppler" in results:
        doppler = results["doppler"]
        print(f"Doppler Spectrogram: {doppler['spectrogram'].shape}")
        print(f"  Time range: {doppler['time_axis'][0]:.2f} - {doppler['time_axis'][-1]:.2f} s")
        print(f"  Velocity range: {doppler['velocity_axis'][0]:.1f} - {doppler['velocity_axis'][-1]:.1f} m/s")

    if "cir" in results:
        cir = results["cir"]
        print(f"CIR: {cir['cir'].shape}")
        print(f"  Delay taps: {cir['num_taps']}")
        print(f"  Tap spacing: {cir['tap_spacing']*1e9:.1f} ns")

    if "csi" in results:
        csi = results["csi"]
        print(f"CSI: {csi['csi'].shape}")
        print(f"  Subcarriers: {csi['num_subcarriers']}")
        print(f"  Bandwidth: {csi['bandwidth']/1e6:.1f} MHz")

    if "point_cloud" in results:
        pc = results["point_cloud"]
        print(f"Point Cloud: {pc['points'].shape[0]} points")
        print(f"  Velocity range: {pc['velocities'].min():.2f} - {pc['velocities'].max():.2f} m/s")

    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
