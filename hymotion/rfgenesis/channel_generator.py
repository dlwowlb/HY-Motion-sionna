"""
RF-Genesis Integration for HY-Motion
 
This module integrates RF-Genesis ray tracing simulation with HY-Motion's
3D human motion generation for generating realistic RF sensing data.
 
Features:
    - Doppler spectrum extraction from human motion
    - CIR (Channel Impulse Response) generation
    - CSI (Channel State Information) simulation
    - mmWave radar point cloud generation
 
Example:
    >>> from hymotion.rfgenesis import RFGenesisSimulator, RFConfig
    >>> from hymotion.utils.t2m_runtime import T2MRuntime
    >>>
    >>> runtime = T2MRuntime(config_path="path/to/config.yml")
    >>> rf_sim = RFGenesisSimulator(RFConfig(frequency=77e9))  # 77GHz mmWave
    >>>
    >>> results = rf_sim.simulate_from_motion(
    ...     runtime=runtime,
    ...     text_prompt="A person walking forward",
    ...     duration=3.0
    ... )
    >>> doppler_data = results["doppler"]
    >>> cir_data = results["cir"]
"""
 
from .rf_simulator import (
    RFGenesisSimulator,
    RFConfig,
    DopplerConfig,
    RadarConfig,
)
 
from .doppler_extractor import (
    DopplerExtractor,
    extract_doppler_from_mesh_sequence,
)
 
from .channel_generator import (
    ChannelGenerator,
    CIRGenerator,
    CSIGenerator,
)
 
from .environment import (
    EnvironmentConfig,
    SceneObject,
    RF_MATERIALS,
    OBJECT_TEMPLATES,
    ENVIRONMENT_PRESETS,
    list_presets,
    list_materials,
    list_objects,
)
 
# RF-Genesis Bridge (requires external/RF-Genesis)
try:
    from .rfgenesis_bridge import (
        HYMotionToRFGenesisBridge,
        RFGenesisConfig,
        run_hymotion_rfgenesis_pipeline,
    )
    RFGENESIS_BRIDGE_AVAILABLE = True
except ImportError:
    RFGENESIS_BRIDGE_AVAILABLE = False
    HYMotionToRFGenesisBridge = None
    RFGenesisConfig = None
    run_hymotion_rfgenesis_pipeline = None
 
__all__ = [
    # Simulator (standalone)
    "RFGenesisSimulator",
    "RFConfig",
    "DopplerConfig",
    "RadarConfig",
    # Doppler
    "DopplerExtractor",
    "extract_doppler_from_mesh_sequence",
    # Channel
    "ChannelGenerator",
    "CIRGenerator",
    "CSIGenerator",
    # Environment (RFLoRA style)
    "EnvironmentConfig",
    "SceneObject",
    "RF_MATERIALS",
    "OBJECT_TEMPLATES",
    "ENVIRONMENT_PRESETS",
    "list_presets",
    "list_materials",
    "list_objects",
    # RF-Genesis Bridge (requires external/RF-Genesis clone)
    "HYMotionToRFGenesisBridge",
    "RFGenesisConfig",
    "run_hymotion_rfgenesis_pipeline",
    "RFGENESIS_BRIDGE_AVAILABLE",
]