#!/usr/bin/env python3
"""
HY-Motion + Sionna Ray Tracing Example

Usage:
    python examples/sionna_ray_example.py --prompt "A person walking forward"
    python examples/sionna_ray_example.py --prompt "걷는 사람" --frequency 28e9
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="HY-Motion + Sionna Ray Tracing")
    parser.add_argument("--model_path", type=str, default="ckpts/tencent/HY-Motion-1.0")
    parser.add_argument("--prompt", type=str, default="A person walking forward")
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--frequency", type=float, default=3.5e9)
    parser.add_argument("--tx", type=str, default="0,0,3")
    parser.add_argument("--rx", type=str, default="5,5,1.5")
    parser.add_argument("--output_dir", type=str, default="output/sionna_ray_viz")
    parser.add_argument("--num_frames", type=int, default=10)
    args = parser.parse_args()

    tx_position = [float(x) for x in args.tx.split(",")]
    rx_position = [float(x) for x in args.rx.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("HY-Motion + Sionna Ray Tracing")
    print("=" * 50)
    print(f"Prompt: {args.prompt}")
    print(f"Duration: {args.duration}s")
    print(f"Frequency: {args.frequency/1e9:.2f} GHz")
    print(f"TX: {tx_position}, RX: {rx_position}")
    print(f"Output: {args.output_dir}")
    print("=" * 50)

    from hymotion.sionna.ray_visualization import SionnaRayVisualizer, RayTracingConfig
    from hymotion.utils.t2m_runtime import T2MRuntime

    # Initialize runtime
    config_path = os.path.join(args.model_path, "config.yml")
    ckpt_path = os.path.join(args.model_path, "latest.ckpt")

    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return 1

    runtime = T2MRuntime(
        config_path=config_path,
        ckpt_name=ckpt_path,
        disable_prompt_engineering=True,
    )

    # Configure ray tracing
    config = RayTracingConfig(
        frequency=args.frequency,
        tx_position=tx_position,
        rx_position=rx_position,
    )

    # Run visualization
    visualizer = SionnaRayVisualizer(config)
    results = visualizer.generate_and_visualize(
        runtime=runtime,
        text_prompt=args.prompt,
        duration=args.duration,
        seed=args.seed,
        cfg_scale=args.cfg_scale,
        output_dir=args.output_dir,
    )

    # Print results
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"  - {f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
