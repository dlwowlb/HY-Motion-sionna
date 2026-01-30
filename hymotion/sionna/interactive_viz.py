"""
Interactive Web-based Visualization for Ray Tracing Results
 
This module provides interactive 3D visualization using Three.js,
similar to HY-Motion's existing visualization but with added ray tracing
path visualization capabilities.
"""
 
import json
import os
from typing import Dict, List, Optional, Any, Union
 
import numpy as np
 
 
class InteractiveRayVisualizer:
    """
    Creates interactive HTML visualizations for ray tracing results
    with human motion from HY-Motion.
    """
 
    # HTML template with embedded Three.js for ray visualization
    HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HY-Motion + Sionna Ray Tracing Visualization</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
            background: #1a1a2e;
        }
        #container { width: 100vw; height: 100vh; }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            padding: 15px;
            background: rgba(0,0,0,0.8);
            color: #fff;
            border-radius: 8px;
            font-size: 14px;
            max-width: 300px;
            z-index: 100;
        }
        #info h2 { margin-bottom: 10px; color: #00d4ff; }
        #info p { margin: 5px 0; opacity: 0.9; }
        #info .label { color: #888; }
        #controls {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 10px;
            padding: 15px;
            background: rgba(0,0,0,0.8);
            border-radius: 8px;
            z-index: 100;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.3s;
        }
        .btn-primary { background: #00d4ff; color: #000; }
        .btn-primary:hover { background: #00a8cc; }
        .btn-secondary { background: #444; color: #fff; }
        .btn-secondary:hover { background: #555; }
        #frame-slider {
            width: 200px;
            accent-color: #00d4ff;
        }
        #frame-display {
            color: #fff;
            min-width: 80px;
            text-align: center;
        }
        #legend {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 15px;
            background: rgba(0,0,0,0.8);
            color: #fff;
            border-radius: 8px;
            font-size: 12px;
            z-index: 100;
        }
        #legend h3 { margin-bottom: 10px; color: #00d4ff; }
        .legend-item { display: flex; align-items: center; margin: 5px 0; }
        .legend-color {
            width: 20px;
            height: 4px;
            margin-right: 10px;
            border-radius: 2px;
        }
        #stats {
            position: absolute;
            bottom: 80px;
            right: 10px;
            padding: 10px;
            background: rgba(0,0,0,0.8);
            color: #fff;
            border-radius: 8px;
            font-size: 12px;
            z-index: 100;
        }
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h2>Ray Tracing Visualization</h2>
        <p><span class="label">Prompt:</span> {{ prompt }}</p>
        <p><span class="label">Frequency:</span> {{ frequency }} GHz</p>
        <p><span class="label">TX Position:</span> {{ tx_position }}</p>
        <p><span class="label">RX Position:</span> {{ rx_position }}</p>
    </div>
    <div id="legend">
        <h3>Ray Types</h3>
        <div class="legend-item">
            <div class="legend-color" style="background: #00ff00;"></div>
            <span>Direct Path (LoS)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ffff00;"></div>
            <span>Single Reflection</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ff8800;"></div>
            <span>Multiple Reflections</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #ff0000;"></div>
            <span>Diffraction</span>
        </div>
    </div>
    <div id="stats">
        <div>Paths: <span id="path-count">0</span></div>
        <div>Power: <span id="total-power">0</span> dBm</div>
    </div>
    <div id="controls">
        <button class="btn-primary" id="play-btn">Play</button>
        <button class="btn-secondary" id="reset-btn">Reset</button>
        <input type="range" id="frame-slider" min="0" max="100" value="0">
        <span id="frame-display">Frame: 0</span>
        <button class="btn-secondary" id="toggle-rays">Toggle Rays</button>
        <button class="btn-secondary" id="toggle-mesh">Toggle Mesh</button>
    </div>
 
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.161.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.161.0/examples/jsm/"
        }
    }
    </script>
 
    <script type="module">
        import * as THREE from 'three';
        import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
 
        // Embedded data
        const meshData = {{ mesh_data_json }};
        const rayData = {{ ray_data_json }};
        const sceneConfig = {{ scene_config_json }};
 
        // Scene setup
        const container = document.getElementById('container');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a2e);
        scene.fog = new THREE.Fog(0x1a1a2e, 50, 200);
 
        // Camera
        const camera = new THREE.PerspectiveCamera(
            60,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        camera.position.set(15, 10, 15);
        camera.lookAt(0, 1, 0);
 
        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        container.appendChild(renderer.domElement);
 
        // Controls
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, 1, 0);
 
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        scene.add(ambientLight);
 
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
        directionalLight.position.set(10, 20, 10);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        scene.add(directionalLight);
 
        // Ground plane with grid
        const gridHelper = new THREE.GridHelper(50, 50, 0x444444, 0x333333);
        scene.add(gridHelper);
 
        const groundGeometry = new THREE.PlaneGeometry(50, 50);
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0x222233,
            roughness: 0.8,
        });
        const ground = new THREE.Mesh(groundGeometry, groundMaterial);
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -0.01;
        ground.receiveShadow = true;
        scene.add(ground);
 
        // Human mesh
        let humanMesh = null;
        let humanGeometry = null;
        const humanMaterial = new THREE.MeshStandardMaterial({
            color: 0xffccaa,
            roughness: 0.6,
            metalness: 0.1,
            side: THREE.DoubleSide,
        });
 
        // Ray lines group
        const rayGroup = new THREE.Group();
        scene.add(rayGroup);
        let raysVisible = true;
 
        // TX and RX markers
        const txGeometry = new THREE.ConeGeometry(0.2, 0.5, 8);
        const txMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
        const txMarker = new THREE.Mesh(txGeometry, txMaterial);
        txMarker.position.set(...sceneConfig.tx_position);
        txMarker.rotation.x = Math.PI;
        scene.add(txMarker);
 
        // TX antenna cone
        const txCone = new THREE.Mesh(
            new THREE.ConeGeometry(2, 4, 16, 1, true),
            new THREE.MeshBasicMaterial({
                color: 0xff0000,
                transparent: true,
                opacity: 0.1,
                side: THREE.DoubleSide,
            })
        );
        txCone.position.copy(txMarker.position);
        txCone.position.y -= 2;
        scene.add(txCone);
 
        const rxGeometry = new THREE.SphereGeometry(0.2, 16, 16);
        const rxMaterial = new THREE.MeshBasicMaterial({ color: 0x0088ff });
        const rxMarker = new THREE.Mesh(rxGeometry, rxMaterial);
        rxMarker.position.set(...sceneConfig.rx_position);
        scene.add(rxMarker);
 
        // Animation state
        let currentFrame = 0;
        let isPlaying = false;
        let meshVisible = true;
        const numFrames = meshData.length;
        const frameSlider = document.getElementById('frame-slider');
        frameSlider.max = numFrames - 1;
 
        // Create human mesh for a frame
        function updateHumanMesh(frameIndex) {
            if (!meshData[frameIndex]) return;
 
            const frameData = meshData[frameIndex];
            const vertices = new Float32Array(frameData.vertices.flat());
 
            if (!humanGeometry) {
                humanGeometry = new THREE.BufferGeometry();
                humanGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
 
                if (frameData.faces) {
                    const indices = new Uint32Array(frameData.faces.flat());
                    humanGeometry.setIndex(new THREE.BufferAttribute(indices, 1));
                }
 
                humanGeometry.computeVertexNormals();
                humanMesh = new THREE.Mesh(humanGeometry, humanMaterial);
                humanMesh.castShadow = true;
                humanMesh.receiveShadow = true;
                scene.add(humanMesh);
            } else {
                humanGeometry.attributes.position.array.set(vertices);
                humanGeometry.attributes.position.needsUpdate = true;
                humanGeometry.computeVertexNormals();
            }
 
            humanMesh.visible = meshVisible;
        }
 
        // Update ray paths for a frame
        function updateRays(frameIndex) {
            // Clear existing rays
            while (rayGroup.children.length > 0) {
                rayGroup.remove(rayGroup.children[0]);
            }
 
            const frameRays = rayData[frameIndex];
            if (!frameRays || !frameRays.paths) return;
 
            const paths = frameRays.paths;
            const powers = frameRays.powers || [];
            const types = frameRays.types || [];
 
            let pathCount = 0;
            let totalPower = 0;
 
            paths.forEach((path, i) => {
                if (!path || path.length < 2) return;
 
                const power = powers[i] || 0.5;
                const type = types[i] || 'reflection';
 
                // Color based on path type
                let color;
                if (type === 'direct' || type === 'los') {
                    color = new THREE.Color(0x00ff00);
                } else if (type === 'reflection' && path.length <= 3) {
                    color = new THREE.Color(0xffff00);
                } else if (type === 'diffraction') {
                    color = new THREE.Color(0xff0000);
                } else {
                    color = new THREE.Color(0xff8800);
                }
 
                // Adjust opacity based on power
                const opacity = Math.min(1.0, 0.2 + power * 0.8);
 
                const points = path.map(p => new THREE.Vector3(p[0], p[2], p[1])); // Y-up
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({
                    color: color,
                    transparent: true,
                    opacity: opacity,
                    linewidth: 1,
                });
                const line = new THREE.Line(geometry, material);
                rayGroup.add(line);
 
                pathCount++;
                totalPower += power;
            });
 
            rayGroup.visible = raysVisible;
 
            // Update stats
            document.getElementById('path-count').textContent = pathCount;
            document.getElementById('total-power').textContent =
                (10 * Math.log10(totalPower + 1e-10)).toFixed(1);
        }
 
        // Animation loop
        function animate() {
            requestAnimationFrame(animate);
 
            if (isPlaying) {
                currentFrame = (currentFrame + 1) % numFrames;
                frameSlider.value = currentFrame;
                document.getElementById('frame-display').textContent = `Frame: ${currentFrame}`;
                updateHumanMesh(currentFrame);
                updateRays(currentFrame);
            }
 
            controls.update();
            renderer.render(scene, camera);
        }
 
        // Event handlers
        document.getElementById('play-btn').addEventListener('click', () => {
            isPlaying = !isPlaying;
            document.getElementById('play-btn').textContent = isPlaying ? 'Pause' : 'Play';
        });
 
        document.getElementById('reset-btn').addEventListener('click', () => {
            currentFrame = 0;
            frameSlider.value = 0;
            document.getElementById('frame-display').textContent = 'Frame: 0';
            updateHumanMesh(0);
            updateRays(0);
        });
 
        frameSlider.addEventListener('input', (e) => {
            currentFrame = parseInt(e.target.value);
            document.getElementById('frame-display').textContent = `Frame: ${currentFrame}`;
            updateHumanMesh(currentFrame);
            updateRays(currentFrame);
        });
 
        document.getElementById('toggle-rays').addEventListener('click', () => {
            raysVisible = !raysVisible;
            rayGroup.visible = raysVisible;
        });
 
        document.getElementById('toggle-mesh').addEventListener('click', () => {
            meshVisible = !meshVisible;
            if (humanMesh) humanMesh.visible = meshVisible;
        });
 
        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
 
        // Initialize
        updateHumanMesh(0);
        updateRays(0);
        animate();
    </script>
</body>
</html>
'''
 
    def __init__(self):
        """Initialize the interactive visualizer."""
        pass
 
    def create_visualization(
        self,
        mesh_data: Dict[str, Any],
        ray_data: List[Dict[str, Any]],
        scene_config: Dict[str, Any],
        prompt: str = "",
        output_path: str = "ray_viz.html",
    ) -> str:
        """
        Create an interactive HTML visualization.
 
        Args:
            mesh_data: Mesh sequence data with vertices and faces
            ray_data: Ray tracing results per frame
            scene_config: Scene configuration (tx/rx positions, frequency, etc.)
            prompt: Original text prompt
            output_path: Output HTML file path
 
        Returns:
            Path to the generated HTML file
        """
        # Prepare mesh data for JavaScript
        mesh_frames = self._prepare_mesh_data(mesh_data)
 
        # Prepare ray data for JavaScript
        ray_frames = self._prepare_ray_data(ray_data)
 
        # Prepare scene config
        config = {
            "tx_position": scene_config.get("tx_position", [0, 0, 3]),
            "rx_position": scene_config.get("rx_position", [5, 5, 1.5]),
            "frequency": scene_config.get("frequency", 3.5e9),
        }
 
        # Format values for display
        freq_ghz = f"{config['frequency'] / 1e9:.2f}"
        tx_pos_str = f"[{config['tx_position'][0]:.1f}, {config['tx_position'][1]:.1f}, {config['tx_position'][2]:.1f}]"
        rx_pos_str = f"[{config['rx_position'][0]:.1f}, {config['rx_position'][1]:.1f}, {config['rx_position'][2]:.1f}]"
 
        # Generate HTML
        html = self.HTML_TEMPLATE
        html = html.replace("{{ mesh_data_json }}", json.dumps(mesh_frames))
        html = html.replace("{{ ray_data_json }}", json.dumps(ray_frames))
        html = html.replace("{{ scene_config_json }}", json.dumps(config))
        html = html.replace("{{ prompt }}", prompt or "N/A")
        html = html.replace("{{ frequency }}", freq_ghz)
        html = html.replace("{{ tx_position }}", tx_pos_str)
        html = html.replace("{{ rx_position }}", rx_pos_str)
 
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
 
        return output_path
 
    def _prepare_mesh_data(
        self,
        mesh_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Convert mesh data to JavaScript-friendly format."""
        vertices = mesh_data.get("vertices", [])
        faces = mesh_data.get("faces", [])
 
        if isinstance(vertices, np.ndarray):
            vertices = vertices.tolist()
        if isinstance(faces, np.ndarray):
            faces = faces.tolist()
 
        # Handle different input formats
        if len(vertices) == 0:
            return []
 
        # Check if vertices has time dimension
        if isinstance(vertices[0], list) and isinstance(vertices[0][0], list):
            # Shape: (num_frames, num_vertices, 3)
            frames = []
            for frame_verts in vertices:
                frames.append({
                    "vertices": frame_verts,
                    "faces": faces,
                })
            return frames
        else:
            # Single frame: (num_vertices, 3)
            return [{"vertices": vertices, "faces": faces}]
 
    def _prepare_ray_data(
        self,
        ray_data: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert ray data to JavaScript-friendly format."""
        frames = []
 
        for frame_rays in ray_data:
            if frame_rays is None:
                frames.append({"paths": [], "powers": [], "types": []})
                continue
 
            paths = frame_rays.get("vertices", [])
            powers = frame_rays.get("powers", [])
            types = frame_rays.get("types", [])
 
            # Convert numpy arrays to lists
            if isinstance(paths, np.ndarray):
                paths = paths.tolist()
            if isinstance(powers, np.ndarray):
                powers = powers.tolist()
 
            # Handle nested arrays
            processed_paths = []
            for path in paths:
                if isinstance(path, np.ndarray):
                    processed_paths.append(path.tolist())
                else:
                    processed_paths.append(path)
 
            frames.append({
                "paths": processed_paths,
                "powers": powers if isinstance(powers, list) else list(powers),
                "types": list(types),
            })
 
        return frames
 
    def create_multi_view(
        self,
        mesh_data: Dict[str, Any],
        ray_data: List[Dict[str, Any]],
        coverage_maps: List[Dict[str, Any]],
        scene_config: Dict[str, Any],
        prompt: str = "",
        output_path: str = "ray_viz_multiview.html",
    ) -> str:
        """
        Create a multi-view visualization with 3D scene and 2D coverage map.
 
        Args:
            mesh_data: Mesh sequence data
            ray_data: Ray tracing results
            coverage_maps: Coverage map data per frame
            scene_config: Scene configuration
            prompt: Original text prompt
            output_path: Output HTML file path
 
        Returns:
            Path to the generated HTML file
        """
        # For now, use the basic visualization
        # A more complex multi-view implementation would go here
        return self.create_visualization(
            mesh_data=mesh_data,
            ray_data=ray_data,
            scene_config=scene_config,
            prompt=prompt,
            output_path=output_path,
        )
 
 
def create_html_visualization(
    vertices: np.ndarray,
    faces: np.ndarray,
    ray_paths: List[Dict],
    tx_position: List[float],
    rx_position: List[float],
    frequency: float = 3.5e9,
    prompt: str = "",
    output_path: str = "visualization.html",
) -> str:
    """
    Convenience function to create HTML visualization.
 
    Args:
        vertices: Mesh vertices (num_frames, num_verts, 3) or (num_verts, 3)
        faces: Mesh faces (num_faces, 3)
        ray_paths: List of ray path dictionaries per frame
        tx_position: Transmitter position
        rx_position: Receiver position
        frequency: Carrier frequency in Hz
        prompt: Text prompt
        output_path: Output file path
 
    Returns:
        Path to generated HTML file
    """
    visualizer = InteractiveRayVisualizer()
 
    mesh_data = {
        "vertices": vertices.tolist() if isinstance(vertices, np.ndarray) else vertices,
        "faces": faces.tolist() if isinstance(faces, np.ndarray) else faces,
    }
 
    scene_config = {
        "tx_position": tx_position,
        "rx_position": rx_position,
        "frequency": frequency,
    }
 
    return visualizer.create_visualization(
        mesh_data=mesh_data,
        ray_data=ray_paths,
        scene_config=scene_config,
        prompt=prompt,
        output_path=output_path,
    )