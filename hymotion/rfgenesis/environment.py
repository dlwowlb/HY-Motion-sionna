"""
Environment Configuration for RF-Genesis Integration

This module provides RFLoRA-style environment configuration using natural
language prompts to describe indoor/outdoor scenes for RF simulation.

Supports:
    - Natural language environment descriptions (RFLoRA style)
    - Predefined environment presets (living room, office, corridor, etc.)
    - Custom object placement with RF material properties
    - Multi-room and outdoor configurations

Example:
    >>> from hymotion.rfgenesis import EnvironmentConfig, SceneBuilder
    >>>
    >>> # RFLoRA style prompt
    >>> env = EnvironmentConfig.from_prompt(
    ...     "a living room with a sofa, a TV, a coffee table, and two windows"
    ... )
    >>>
    >>> # Or use presets
    >>> env = EnvironmentConfig.preset("office")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import re
import json


# RF Material Properties Database (at 77 GHz mmWave)
RF_MATERIALS = {
    "concrete": {
        "permittivity": 5.31,
        "conductivity": 0.0326,
        "reflectivity": 0.6,
        "description": "Standard concrete wall/floor",
    },
    "glass": {
        "permittivity": 6.27,
        "conductivity": 0.0043,
        "reflectivity": 0.3,
        "description": "Window glass",
    },
    "wood": {
        "permittivity": 1.99,
        "conductivity": 0.0047,
        "reflectivity": 0.4,
        "description": "Wooden furniture",
    },
    "metal": {
        "permittivity": 1.0,
        "conductivity": 1e7,
        "reflectivity": 0.95,
        "description": "Metal surfaces (strong reflector)",
    },
    "plasterboard": {
        "permittivity": 2.94,
        "conductivity": 0.0116,
        "reflectivity": 0.5,
        "description": "Drywall/plasterboard",
    },
    "carpet": {
        "permittivity": 1.5,
        "conductivity": 0.001,
        "reflectivity": 0.1,
        "description": "Carpet flooring (absorber)",
    },
    "fabric": {
        "permittivity": 1.8,
        "conductivity": 0.002,
        "reflectivity": 0.15,
        "description": "Fabric (sofa, curtains)",
    },
    "human_body": {
        "permittivity": 53.0,
        "conductivity": 1.5,
        "reflectivity": 0.7,
        "description": "Human body tissue at 77 GHz",
    },
    "plastic": {
        "permittivity": 2.5,
        "conductivity": 0.001,
        "reflectivity": 0.2,
        "description": "Plastic surfaces",
    },
    "ceramic": {
        "permittivity": 6.0,
        "conductivity": 0.01,
        "reflectivity": 0.4,
        "description": "Ceramic tiles",
    },
}


# Object templates with default properties
OBJECT_TEMPLATES = {
    # Furniture
    "sofa": {"size": [2.0, 0.9, 0.8], "material": "fabric", "height": 0.0},
    "couch": {"size": [2.0, 0.9, 0.8], "material": "fabric", "height": 0.0},
    "chair": {"size": [0.5, 0.5, 0.9], "material": "wood", "height": 0.0},
    "table": {"size": [1.2, 0.8, 0.75], "material": "wood", "height": 0.0},
    "coffee_table": {"size": [1.0, 0.6, 0.45], "material": "wood", "height": 0.0},
    "desk": {"size": [1.4, 0.7, 0.75], "material": "wood", "height": 0.0},
    "bed": {"size": [2.0, 1.5, 0.6], "material": "fabric", "height": 0.0},
    "wardrobe": {"size": [1.5, 0.6, 2.0], "material": "wood", "height": 0.0},
    "bookshelf": {"size": [1.0, 0.3, 1.8], "material": "wood", "height": 0.0},
    "cabinet": {"size": [0.8, 0.5, 1.0], "material": "wood", "height": 0.0},

    # Electronics
    "tv": {"size": [1.2, 0.1, 0.7], "material": "plastic", "height": 1.0},
    "television": {"size": [1.2, 0.1, 0.7], "material": "plastic", "height": 1.0},
    "computer": {"size": [0.5, 0.4, 0.4], "material": "metal", "height": 0.75},
    "monitor": {"size": [0.6, 0.1, 0.4], "material": "plastic", "height": 0.75},
    "refrigerator": {"size": [0.7, 0.7, 1.8], "material": "metal", "height": 0.0},
    "microwave": {"size": [0.5, 0.4, 0.3], "material": "metal", "height": 1.0},

    # Other objects
    "lamp": {"size": [0.3, 0.3, 1.5], "material": "metal", "height": 0.0},
    "plant": {"size": [0.4, 0.4, 0.8], "material": "fabric", "height": 0.0},  # leaves absorb
    "mirror": {"size": [0.8, 0.05, 1.2], "material": "glass", "height": 0.8},
    "door": {"size": [0.9, 0.05, 2.1], "material": "wood", "height": 0.0},
    "window": {"size": [1.2, 0.05, 1.4], "material": "glass", "height": 0.8},
    "curtain": {"size": [1.5, 0.1, 2.0], "material": "fabric", "height": 0.5},

    # Kitchen
    "stove": {"size": [0.6, 0.6, 0.9], "material": "metal", "height": 0.0},
    "sink": {"size": [0.6, 0.5, 0.2], "material": "ceramic", "height": 0.85},

    # Bathroom
    "toilet": {"size": [0.4, 0.6, 0.8], "material": "ceramic", "height": 0.0},
    "bathtub": {"size": [1.7, 0.8, 0.6], "material": "ceramic", "height": 0.0},
}


# Environment presets
ENVIRONMENT_PRESETS = {
    "living_room": {
        "room_size": [6.0, 5.0, 2.8],
        "objects": [
            {"type": "sofa", "position": [3.0, 1.0, 0.0], "rotation": 0},
            {"type": "tv", "position": [3.0, 4.5, 1.0], "rotation": 180},
            {"type": "coffee_table", "position": [3.0, 2.5, 0.0], "rotation": 0},
            {"type": "lamp", "position": [1.0, 1.0, 0.0], "rotation": 0},
            {"type": "window", "position": [0.0, 2.5, 0.8], "rotation": 90},
            {"type": "plant", "position": [5.5, 4.5, 0.0], "rotation": 0},
        ],
        "wall_material": "plasterboard",
        "floor_material": "carpet",
        "description": "Standard living room with sofa, TV, and coffee table",
    },
    "office": {
        "room_size": [5.0, 4.0, 2.7],
        "objects": [
            {"type": "desk", "position": [2.5, 3.0, 0.0], "rotation": 0},
            {"type": "chair", "position": [2.5, 2.0, 0.0], "rotation": 0},
            {"type": "monitor", "position": [2.5, 3.2, 0.75], "rotation": 0},
            {"type": "bookshelf", "position": [0.5, 2.0, 0.0], "rotation": 90},
            {"type": "window", "position": [2.5, 4.0, 0.8], "rotation": 0},
            {"type": "cabinet", "position": [4.5, 1.0, 0.0], "rotation": -90},
        ],
        "wall_material": "plasterboard",
        "floor_material": "carpet",
        "description": "Office space with desk, chair, and bookshelf",
    },
    "bedroom": {
        "room_size": [4.5, 4.0, 2.7],
        "objects": [
            {"type": "bed", "position": [2.25, 3.0, 0.0], "rotation": 0},
            {"type": "wardrobe", "position": [0.75, 1.0, 0.0], "rotation": 90},
            {"type": "lamp", "position": [0.5, 3.5, 0.0], "rotation": 0},
            {"type": "window", "position": [2.25, 4.0, 0.8], "rotation": 0},
            {"type": "mirror", "position": [4.0, 2.0, 0.8], "rotation": -90},
        ],
        "wall_material": "plasterboard",
        "floor_material": "carpet",
        "description": "Bedroom with bed, wardrobe, and side lamp",
    },
    "corridor": {
        "room_size": [8.0, 2.0, 2.7],
        "objects": [
            {"type": "door", "position": [0.0, 1.0, 0.0], "rotation": 90},
            {"type": "door", "position": [8.0, 1.0, 0.0], "rotation": -90},
            {"type": "lamp", "position": [4.0, 1.0, 2.5], "rotation": 0},
        ],
        "wall_material": "plasterboard",
        "floor_material": "ceramic",
        "description": "Long corridor connecting rooms",
    },
    "kitchen": {
        "room_size": [4.0, 3.5, 2.7],
        "objects": [
            {"type": "refrigerator", "position": [0.5, 3.0, 0.0], "rotation": 0},
            {"type": "stove", "position": [2.0, 3.2, 0.0], "rotation": 0},
            {"type": "sink", "position": [3.5, 3.2, 0.85], "rotation": 0},
            {"type": "table", "position": [2.0, 1.5, 0.0], "rotation": 0},
            {"type": "chair", "position": [1.5, 1.0, 0.0], "rotation": 0},
            {"type": "chair", "position": [2.5, 1.0, 0.0], "rotation": 0},
            {"type": "microwave", "position": [1.0, 3.2, 1.0], "rotation": 0},
            {"type": "window", "position": [2.0, 3.5, 0.8], "rotation": 0},
        ],
        "wall_material": "ceramic",
        "floor_material": "ceramic",
        "description": "Kitchen with appliances and dining area",
    },
    "bathroom": {
        "room_size": [3.0, 2.5, 2.5],
        "objects": [
            {"type": "toilet", "position": [0.5, 2.0, 0.0], "rotation": 0},
            {"type": "sink", "position": [1.5, 2.3, 0.85], "rotation": 0},
            {"type": "bathtub", "position": [2.5, 1.0, 0.0], "rotation": -90},
            {"type": "mirror", "position": [1.5, 2.5, 0.8], "rotation": 0},
            {"type": "window", "position": [3.0, 1.5, 1.2], "rotation": -90},
        ],
        "wall_material": "ceramic",
        "floor_material": "ceramic",
        "description": "Bathroom with standard fixtures",
    },
    "empty_room": {
        "room_size": [5.0, 5.0, 3.0],
        "objects": [],
        "wall_material": "concrete",
        "floor_material": "concrete",
        "description": "Empty room for testing",
    },
    "outdoor_open": {
        "room_size": [20.0, 20.0, 10.0],
        "objects": [],
        "wall_material": None,  # No walls
        "floor_material": "concrete",
        "is_outdoor": True,
        "description": "Open outdoor area",
    },
}


@dataclass
class SceneObject:
    """Represents an object in the scene."""

    object_type: str
    position: List[float]  # [x, y, z]
    rotation: float = 0.0  # Degrees around Z axis
    size: Optional[List[float]] = None  # Override default size
    material: Optional[str] = None  # Override default material

    def get_size(self) -> List[float]:
        """Get object size (from override or template)."""
        if self.size is not None:
            return self.size
        template = OBJECT_TEMPLATES.get(self.object_type, {})
        return template.get("size", [1.0, 1.0, 1.0])

    def get_material(self) -> str:
        """Get object material (from override or template)."""
        if self.material is not None:
            return self.material
        template = OBJECT_TEMPLATES.get(self.object_type, {})
        return template.get("material", "wood")

    def get_material_properties(self) -> Dict[str, float]:
        """Get RF material properties."""
        material_name = self.get_material()
        return RF_MATERIALS.get(material_name, RF_MATERIALS["wood"])

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        size = np.array(self.get_size())
        pos = np.array(self.position)

        # Simple rotation handling (Z-axis only)
        angle = np.radians(self.rotation)
        c, s = np.cos(angle), np.sin(angle)

        # Rotated size (approximate AABB)
        rotated_size = np.array([
            abs(c * size[0]) + abs(s * size[1]),
            abs(s * size[0]) + abs(c * size[1]),
            size[2],
        ])

        half_size = rotated_size / 2
        min_corner = pos - half_size
        max_corner = pos + half_size

        return min_corner, max_corner


@dataclass
class EnvironmentConfig:
    """
    Configuration for RF simulation environment.

    Supports RFLoRA-style natural language prompts or manual configuration.
    """

    room_size: Tuple[float, float, float] = (5.0, 5.0, 3.0)  # (x, y, z) meters
    objects: List[SceneObject] = field(default_factory=list)
    wall_material: str = "plasterboard"
    floor_material: str = "concrete"
    ceiling_material: str = "plasterboard"
    is_outdoor: bool = False
    description: str = ""

    # Radar/TX/RX positions
    radar_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.5])
    human_start_position: List[float] = field(default_factory=lambda: [2.5, 2.5, 0.0])

    @classmethod
    def from_prompt(cls, prompt: str, room_size: Tuple[float, float, float] = None) -> "EnvironmentConfig":
        """
        Create environment configuration from RFLoRA-style natural language prompt.

        Args:
            prompt: Natural language description like
                    "a living room with a sofa, a TV, and two windows"
            room_size: Optional room size override

        Returns:
            EnvironmentConfig instance
        """
        prompt_lower = prompt.lower()

        # Detect room type from prompt
        room_type = cls._detect_room_type(prompt_lower)

        # Start with preset if available
        if room_type in ENVIRONMENT_PRESETS:
            preset = ENVIRONMENT_PRESETS[room_type]
            base_room_size = room_size or tuple(preset["room_size"])
            wall_mat = preset.get("wall_material", "plasterboard")
            floor_mat = preset.get("floor_material", "concrete")
            is_outdoor = preset.get("is_outdoor", False)
        else:
            base_room_size = room_size or (5.0, 5.0, 3.0)
            wall_mat = "plasterboard"
            floor_mat = "concrete"
            is_outdoor = "outdoor" in prompt_lower

        # Parse objects from prompt
        objects = cls._parse_objects_from_prompt(prompt_lower, base_room_size)

        return cls(
            room_size=base_room_size,
            objects=objects,
            wall_material=wall_mat if not is_outdoor else None,
            floor_material=floor_mat,
            is_outdoor=is_outdoor,
            description=prompt,
        )

    @classmethod
    def preset(cls, preset_name: str) -> "EnvironmentConfig":
        """
        Create environment from predefined preset.

        Available presets:
            - living_room
            - office
            - bedroom
            - corridor
            - kitchen
            - bathroom
            - empty_room
            - outdoor_open

        Args:
            preset_name: Name of the preset

        Returns:
            EnvironmentConfig instance
        """
        if preset_name not in ENVIRONMENT_PRESETS:
            available = list(ENVIRONMENT_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        preset = ENVIRONMENT_PRESETS[preset_name]

        objects = []
        for obj_data in preset.get("objects", []):
            obj = SceneObject(
                object_type=obj_data["type"],
                position=obj_data["position"],
                rotation=obj_data.get("rotation", 0),
            )
            objects.append(obj)

        return cls(
            room_size=tuple(preset["room_size"]),
            objects=objects,
            wall_material=preset.get("wall_material", "plasterboard"),
            floor_material=preset.get("floor_material", "concrete"),
            is_outdoor=preset.get("is_outdoor", False),
            description=preset.get("description", ""),
        )

    @staticmethod
    def _detect_room_type(prompt: str) -> str:
        """Detect room type from prompt."""
        room_keywords = {
            "living_room": ["living room", "living-room", "lounge"],
            "office": ["office", "workspace", "study"],
            "bedroom": ["bedroom", "bed room", "sleeping"],
            "corridor": ["corridor", "hallway", "hall"],
            "kitchen": ["kitchen", "cooking"],
            "bathroom": ["bathroom", "bath room", "restroom", "toilet"],
            "outdoor_open": ["outdoor", "outside", "open area", "parking"],
        }

        for room_type, keywords in room_keywords.items():
            for keyword in keywords:
                if keyword in prompt:
                    return room_type

        return "empty_room"

    @staticmethod
    def _parse_objects_from_prompt(prompt: str, room_size: Tuple[float, float, float]) -> List[SceneObject]:
        """Parse objects from natural language prompt."""
        objects = []

        # Find object mentions
        for obj_type in OBJECT_TEMPLATES.keys():
            # Check for object with optional count (e.g., "two chairs", "a sofa")
            patterns = [
                rf"(\d+|a|an|one|two|three|four|five)\s+{obj_type}s?",
                rf"{obj_type}s?",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, prompt)
                if matches:
                    # Determine count
                    count = 1
                    if matches[0] in ["two", "2"]:
                        count = 2
                    elif matches[0] in ["three", "3"]:
                        count = 3
                    elif matches[0] in ["four", "4"]:
                        count = 4
                    elif matches[0] in ["five", "5"]:
                        count = 5

                    # Place objects with random positions
                    for i in range(count):
                        # Random position within room (with margin)
                        margin = 0.5
                        x = margin + np.random.random() * (room_size[0] - 2 * margin)
                        y = margin + np.random.random() * (room_size[1] - 2 * margin)

                        template = OBJECT_TEMPLATES.get(obj_type, {})
                        z = template.get("height", 0.0)

                        obj = SceneObject(
                            object_type=obj_type,
                            position=[x, y, z],
                            rotation=np.random.choice([0, 90, 180, 270]),
                        )
                        objects.append(obj)
                    break

        return objects

    def get_reflective_surfaces(self) -> List[Dict[str, Any]]:
        """
        Get all reflective surfaces in the environment for ray tracing.

        Returns:
            List of surface dictionaries with position, normal, size, and material
        """
        surfaces = []

        # Floor
        surfaces.append({
            "type": "floor",
            "position": [self.room_size[0] / 2, self.room_size[1] / 2, 0],
            "normal": [0, 0, 1],
            "size": [self.room_size[0], self.room_size[1]],
            "material": self.floor_material,
            "properties": RF_MATERIALS.get(self.floor_material, RF_MATERIALS["concrete"]),
        })

        # Ceiling
        if not self.is_outdoor:
            surfaces.append({
                "type": "ceiling",
                "position": [self.room_size[0] / 2, self.room_size[1] / 2, self.room_size[2]],
                "normal": [0, 0, -1],
                "size": [self.room_size[0], self.room_size[1]],
                "material": self.ceiling_material,
                "properties": RF_MATERIALS.get(self.ceiling_material, RF_MATERIALS["plasterboard"]),
            })

            # Walls
            wall_configs = [
                {"pos": [0, self.room_size[1] / 2, self.room_size[2] / 2],
                 "normal": [1, 0, 0], "size": [self.room_size[1], self.room_size[2]]},
                {"pos": [self.room_size[0], self.room_size[1] / 2, self.room_size[2] / 2],
                 "normal": [-1, 0, 0], "size": [self.room_size[1], self.room_size[2]]},
                {"pos": [self.room_size[0] / 2, 0, self.room_size[2] / 2],
                 "normal": [0, 1, 0], "size": [self.room_size[0], self.room_size[2]]},
                {"pos": [self.room_size[0] / 2, self.room_size[1], self.room_size[2] / 2],
                 "normal": [0, -1, 0], "size": [self.room_size[0], self.room_size[2]]},
            ]

            for i, wall in enumerate(wall_configs):
                surfaces.append({
                    "type": f"wall_{i}",
                    "position": wall["pos"],
                    "normal": wall["normal"],
                    "size": wall["size"],
                    "material": self.wall_material,
                    "properties": RF_MATERIALS.get(self.wall_material, RF_MATERIALS["plasterboard"]),
                })

        # Objects
        for obj in self.objects:
            min_corner, max_corner = obj.get_bounding_box()
            center = (min_corner + max_corner) / 2
            size = max_corner - min_corner

            surfaces.append({
                "type": f"object_{obj.object_type}",
                "position": center.tolist(),
                "size": size.tolist(),
                "material": obj.get_material(),
                "properties": obj.get_material_properties(),
                "is_object": True,
            })

        return surfaces

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "room_size": list(self.room_size),
            "objects": [
                {
                    "type": obj.object_type,
                    "position": obj.position,
                    "rotation": obj.rotation,
                    "size": obj.size,
                    "material": obj.material,
                }
                for obj in self.objects
            ],
            "wall_material": self.wall_material,
            "floor_material": self.floor_material,
            "ceiling_material": self.ceiling_material,
            "is_outdoor": self.is_outdoor,
            "description": self.description,
            "radar_position": self.radar_position,
            "human_start_position": self.human_start_position,
        }

    def save(self, filepath: str):
        """Save environment configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "EnvironmentConfig":
        """Load environment configuration from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        objects = []
        for obj_data in data.get("objects", []):
            obj = SceneObject(
                object_type=obj_data["type"],
                position=obj_data["position"],
                rotation=obj_data.get("rotation", 0),
                size=obj_data.get("size"),
                material=obj_data.get("material"),
            )
            objects.append(obj)

        config = cls(
            room_size=tuple(data["room_size"]),
            objects=objects,
            wall_material=data.get("wall_material", "plasterboard"),
            floor_material=data.get("floor_material", "concrete"),
            ceiling_material=data.get("ceiling_material", "plasterboard"),
            is_outdoor=data.get("is_outdoor", False),
            description=data.get("description", ""),
        )

        if "radar_position" in data:
            config.radar_position = data["radar_position"]
        if "human_start_position" in data:
            config.human_start_position = data["human_start_position"]

        return config

    def __repr__(self) -> str:
        return (
            f"EnvironmentConfig(\n"
            f"  room_size={self.room_size},\n"
            f"  objects={len(self.objects)} items,\n"
            f"  wall={self.wall_material}, floor={self.floor_material},\n"
            f"  outdoor={self.is_outdoor}\n"
            f")"
        )


def list_presets() -> List[str]:
    """List available environment presets."""
    return list(ENVIRONMENT_PRESETS.keys())


def list_materials() -> Dict[str, str]:
    """List available RF materials with descriptions."""
    return {name: props["description"] for name, props in RF_MATERIALS.items()}


def list_objects() -> List[str]:
    """List available object types."""
    return list(OBJECT_TEMPLATES.keys())
