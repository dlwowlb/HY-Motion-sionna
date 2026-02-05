#!/usr/bin/env python3
"""
HY-Motion + RF-Genesis Integrated Pipeline Example

This example demonstrates the complete pipeline:
1. HY-Motion: Generate 3D human motion from text prompt
2. RF-Genesis: Simulate mmWave radar signals with RFLoRA environments (REQUIRED)
3. Doppler FFT: Extract Doppler spectrum and point clouds

IMPORTANT: This pipeline REQUIRES:
    - Mitsuba ray tracing for accurate PIR generation
    - RFLoRA environment diffusion for realistic scenes
    - environment_prompt is REQUIRED (use --environment or --env_preset)

Requirements:
    - HY-Motion model weights
    - RF-Genesis (git clone https://github.com/Asixa/RF-Genesis external/RF-Genesis)
    - RF-Genesis setup (cd external/RF-Genesis && sh setup.sh)
    - pip install mitsuba==3.5.2 diffusers transformers accelerate peft
    - GPU with CUDA support

Usage:
    # Full pipeline with custom environment
    python examples/hymotion_rfgenesis_pipeline.py \
        --motion "A person walking forward" \
        --environment "a living room with a sofa, TV, and coffee table"

    # Using environment preset
    python examples/hymotion_rfgenesis_pipeline.py \
        --motion "A person waving hands" \
        --env_preset living_room

    # Demo mode (synthetic motion, no HY-Motion model needed)
    python examples/hymotion_rfgenesis_pipeline.py \
        --motion "A person walking" \
        --env_preset office \
        --demo_mode
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="HY-Motion + RF-Genesis Integrated Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With RFLoRA environment prompt
  python %(prog)s --motion "A person walking" --environment "a living room"

  # With environment preset
  python %(prog)s --motion "A person walking" --env_preset living_room

  # Demo mode (synthetic motion, environment still required)
  python %(prog)s --motion "walking" --env_preset office --demo_mode

  # List available environment presets
  python %(prog)s --list_env_presets
"""
    )

    # Motion settings
    parser.add_argument(
        "--motion", "-m", type=str, default="A person walking forward slowly",
        help="Motion prompt for HY-Motion"
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=3.0,
        help="Motion duration in seconds"
    )

    # Environment settings (RFLoRA) - REQUIRED
    parser.add_argument(
        "--environment", "-e", type=str, default=None,
        help="RFLoRA environment prompt (REQUIRED, e.g., 'a living room with furniture')"
    )
    parser.add_argument(
        "--env_preset", type=str, default=None,
        choices=["living_room", "office", "bedroom", "corridor", "kitchen"],
        help="Use predefined environment preset (alternative to --environment)"
    )
    parser.add_argument(
        "--list_env_presets", action="store_true",
        help="List available environment presets"
    )

    # Output settings
    parser.add_argument(
        "--output_dir", "-o", type=str, default="output/hymotion_rfgenesis",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    # Model settings
    parser.add_argument(
        "--model_path", type=str, default="ckpts/tencent/HY-Motion-1.0",
        help="Path to HY-Motion model"
    )
    parser.add_argument(
        "--demo_mode", action="store_true",
        help="Demo mode with synthetic motion (no HY-Motion model needed)"
    )

    # Visualization
    parser.add_argument(
        "--no_visualize", action="store_true",
        help="Skip visualization"
    )

    return parser.parse_args()


# Environment presets
ENV_PRESETS = {
    "living_room": "a living room with a sofa, a TV on the wall, a coffee table, two armchairs, and a large window",
    "office": "an office with a desk, computer monitor, office chair, bookshelf, and potted plant",
    "bedroom": "a bedroom with a bed, wardrobe, bedside table, lamp, and window with curtains",
    "corridor": "a long corridor with doors on both sides, ceiling lights, and a plant at the end",
    "kitchen": "a kitchen with a refrigerator, stove, sink, dining table, and cabinets",
}


def create_demo_smpl_data(num_frames: int = 90) -> dict:
    """Create synthetic SMPL data for demo mode."""
    print("[Demo Mode] Creating synthetic SMPL motion data...")

    # Walking motion parameters
    pose = np.zeros((num_frames, 72), dtype=np.float32)
    shape = np.zeros((1, 10), dtype=np.float32)
    root_translation = np.zeros((num_frames, 3), dtype=np.float32)

    # Simulate walking forward
    for t in range(num_frames):
        time = t / 30.0  # Assuming 30 FPS

        # Forward motion
        root_translation[t, 1] = time * 1.0  # 1 m/s forward
        root_translation[t, 2] = 0.0  # Ground level

        # Leg swing (simplified)
        leg_angle = 0.3 * np.sin(2 * np.pi * 1.5 * time)  # 1.5 Hz swing
        pose[t, 3] = leg_angle  # Right hip
        pose[t, 6] = -leg_angle  # Left hip

        # Arm swing (opposite to legs)
        arm_angle = 0.2 * np.sin(2 * np.pi * 1.5 * time + np.pi)
        pose[t, 48] = arm_angle  # Right shoulder
        pose[t, 51] = -arm_angle  # Left shoulder

    return {
        "pose": pose,
        "shape": shape,
        "root_translation": root_translation,
        "gender": "male",
        "num_frames": num_frames,
    }


def visualize_results(results: dict, output_dir: str):
    """Create visualization of Doppler FFT results."""

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Doppler Spectrogram (Time-Doppler)
    ax = axes[0, 0]
    spectrogram = results["doppler_spectrogram"]
    velocity_axis = results["velocity_axis"]
    num_frames = results["num_frames"]
    frame_rate = results["frame_rate"]

    time_axis = np.arange(num_frames) / frame_rate

    im = ax.imshow(
        spectrogram.T,
        aspect="auto",
        origin="lower",
        extent=[time_axis[0], time_axis[-1], velocity_axis[0], velocity_axis[-1]],
        cmap="jet",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Doppler Spectrogram (RF-Genesis)")
    plt.colorbar(im, ax=ax, label="Power")

    # 2. Range-Doppler Map (average)
    ax = axes[0, 1]
    rd_map = results["range_doppler_maps"].mean(axis=0)
    range_axis = results["range_axis"]

    # Limit range for display
    max_range_idx = min(128, len(range_axis))
    rd_map_display = rd_map[:, :max_range_idx]
    range_axis_display = range_axis[:max_range_idx]

    im = ax.imshow(
        10 * np.log10(rd_map_display + 1e-10),
        aspect="auto",
        origin="lower",
        extent=[range_axis_display[0], range_axis_display[-1],
                velocity_axis[0], velocity_axis[-1]],
        cmap="viridis",
        vmin=-40, vmax=0,
    )
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Range-Doppler Map (dB)")
    plt.colorbar(im, ax=ax, label="Power (dB)")

    # 3. Point Cloud (all frames)
    ax = axes[1, 0]
    all_points = []
    all_velocities = []
    for frame_idx, pc in enumerate(results["point_clouds"]):
        if pc is not None and len(pc) > 0:
            all_points.append(pc[:, :3])  # x, y, z
            all_velocities.append(pc[:, 3])  # velocity

    if all_points:
        points = np.vstack(all_points)
        velocities = np.concatenate(all_velocities)

        scatter = ax.scatter(
            points[:, 0], points[:, 1],
            c=velocities, cmap="coolwarm",
            s=2, alpha=0.5
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Radar Point Cloud (XY plane)")
        plt.colorbar(scatter, ax=ax, label="Velocity (m/s)")
    else:
        ax.text(0.5, 0.5, "No point cloud data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Radar Point Cloud")

    # 4. Average Doppler profile
    ax = axes[1, 1]
    avg_doppler = spectrogram.mean(axis=0)
    ax.plot(velocity_axis, avg_doppler, "b-", linewidth=2)
    ax.fill_between(velocity_axis, 0, avg_doppler, alpha=0.3)
    ax.set_xlabel("Velocity (m/s)")
    ax.set_ylabel("Average Power")
    ax.set_title("Average Doppler Profile")
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rfgenesis_doppler_fft.png"), dpi=150)
    plt.close()
    print(f"  Saved: rfgenesis_doppler_fft.png")


def main():
    args = parse_args()

    # Handle --list_env_presets
    if args.list_env_presets:
        print("Available Environment Presets (RFLoRA):")
        print("=" * 60)
        for name, prompt in ENV_PRESETS.items():
            print(f"\n  {name}:")
            print(f"    {prompt}")
        print("\nUsage: --env_preset living_room")
        print("   or: --environment 'your custom environment description'")
        return

    # Determine environment prompt (REQUIRED)
    env_prompt = None
    if args.env_preset:
        env_prompt = ENV_PRESETS.get(args.env_preset)
        print(f"Using environment preset: {args.env_preset}")
    elif args.environment:
        env_prompt = args.environment

    # Environment is REQUIRED
    if not env_prompt:
        print("ERROR: Environment is REQUIRED for RF-Genesis simulation.")
        print("\nPlease specify one of:")
        print("  --environment 'a living room with furniture'")
        print("  --env_preset living_room")
        print("\nRun with --list_env_presets to see available presets.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("HY-Motion + RF-Genesis Integrated Pipeline")
    print("=" * 70)
    print(f"Motion Prompt:      {args.motion}")
    print(f"Duration:           {args.duration}s")
    print(f"Environment:        {env_prompt}")
    print(f"Output Directory:   {args.output_dir}")
    print(f"Mode:               {'Demo' if args.demo_mode else 'Full'}")
    print("=" * 70)

    # Import bridge module
    from hymotion.rfgenesis.rfgenesis_bridge import (
        HYMotionToRFGenesisBridge,
        RFGenesisConfig,
    )

    # Initialize bridge (environment is always enabled)
    config = RFGenesisConfig(
        use_environment=True,
        environment_prompt=env_prompt,
        output_dir=args.output_dir,
    )

    try:
        bridge = HYMotionToRFGenesisBridge(config)
    except Exception as e:
        print(f"\n[Error] Failed to initialize RF-Genesis bridge: {e}")
        print("\nMake sure RF-Genesis is set up:")
        print("  1. git clone https://github.com/Asixa/RF-Genesis external/RF-Genesis")
        print("  2. cd external/RF-Genesis && sh setup.sh")
        return

    if args.demo_mode:
        # Demo mode: use synthetic SMPL data
        print("\n[Step 1] Creating synthetic motion data (demo mode)...")
        num_frames = int(args.duration * 30)
        smpl_data = create_demo_smpl_data(num_frames=num_frames)

    else:
        # Full mode: use HY-Motion
        print("\n[Step 1] Generating motion with HY-Motion...")
        try:
            from hymotion.utils.t2m_runtime import T2MRuntime

            config_path = os.path.join(args.model_path, "config.yml")
            if not os.path.exists(config_path):
                print(f"[Warning] HY-Motion config not found at {config_path}")
                print("[Warning] Falling back to demo mode...")
                num_frames = int(args.duration * 30)
                smpl_data = create_demo_smpl_data(num_frames=num_frames)
            else:
                runtime = T2MRuntime(
                    config_path=config_path,
                    disable_prompt_engineering=True,
                )

                # Generate motion
                html_content, fbx_files, motion_data = runtime.generate_motion(
                    text=args.motion,
                    seeds_csv=str(args.seed),
                    duration=args.duration,
                    cfg_scale=5.0,
                    output_format="dict",
                )

                # Convert to SMPL format
                print("\n[Step 2] Converting to RF-Genesis SMPL format...")
                smpl_data = bridge.convert_hymotion_to_smpl(motion_data)

        except Exception as e:
            print(f"[Error] HY-Motion failed: {e}")
            print("[Warning] Falling back to demo mode...")
            num_frames = int(args.duration * 30)
            smpl_data = create_demo_smpl_data(num_frames=num_frames)

    # Save SMPL data
    np.savez(
        os.path.join(args.output_dir, "obj_diff.npz"),
        pose=smpl_data["pose"],
        shape=smpl_data["shape"],
        root_translation=smpl_data["root_translation"],
        gender=smpl_data["gender"],
    )
    print(f"  Saved: obj_diff.npz ({smpl_data['num_frames']} frames)")

    # Run radar simulation
    print("\n[Step 3] Running RF-Genesis radar simulation...")
    try:
        results = bridge.simulate_radar(
            smpl_data,
            environment_prompt=env_prompt,
            output_dir=args.output_dir,
        )

        # Visualize results
        if not args.no_visualize:
            print("\n[Step 4] Generating visualizations...")
            visualize_results(results, args.output_dir)

        # Print summary
        print("\n" + "=" * 70)
        print("Results Summary")
        print("=" * 70)
        print(f"Radar Frames:       {results['num_frames']}")
        print(f"Frame Rate:         {results['frame_rate']} Hz")
        print(f"Velocity Range:     {results['velocity_axis'][0]:.2f} to {results['velocity_axis'][-1]:.2f} m/s")
        print(f"Range Resolution:   {results['range_axis'][1] - results['range_axis'][0]:.3f} m")
        print(f"Doppler Shape:      {results['doppler_spectrogram'].shape}")
        print(f"Range-Doppler Shape: {results['range_doppler_maps'].shape}")

        if results['point_clouds']:
            total_points = sum(len(pc) for pc in results['point_clouds'] if pc is not None and len(pc) > 0)
            print(f"Total Point Cloud:  {total_points} points")

        print(f"\nOutput saved to: {args.output_dir}")
        print("=" * 70)

    except Exception as e:
        print(f"\n[Error] Radar simulation failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nThis may be due to missing RF-Genesis dependencies.")
        print("Try running: cd external/RF-Genesis && sh setup.sh")


if __name__ == "__main__":
    main()
