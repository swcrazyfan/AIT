import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Type, Optional

from video_tool.steps.base import BaseStep
from video_tool.core.models import StepConfig

class StepRegistry:
    """Auto-discovers and manages all available steps"""

    def __init__(self):
        self.steps: Dict[str, Dict[str, Type[BaseStep]]] = {}
        self._discovered = False

    def discover_steps(self):
        """Auto-discover all steps in the steps directory"""
        if self._discovered:
            return

        steps_dir = Path(__file__).parent.parent / "steps"

        # Iterate through category directories
        for category_dir in steps_dir.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith("_"):
                continue

            category = category_dir.name
            self.steps[category] = {}

            # Import all Python files in the category
            for module_file in category_dir.glob("*.py"):
                if module_file.name.startswith("_"):
                    continue

                try:
                    # Import the module
                    module_name = f"video_tool.steps.{category}.{module_file.stem}"
                    module = importlib.import_module(module_name)

                    # Find all BaseStep subclasses
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and
                            issubclass(obj, BaseStep) and
                            obj != BaseStep and
                            hasattr(obj, 'name') and
                            obj.name):  # Ensure it has a name

                            self.steps[category][obj.name] = obj

                except Exception as e:
                    print(f"Error loading step module {module_name}: {e}")

        self._discovered = True

    def get_step_class(self, category: str, name: str) -> Optional[Type[BaseStep]]:
        """Get a specific step class"""
        self.discover_steps()
        return self.steps.get(category, {}).get(name)

    def create_step(self, category: str, name: str, config: StepConfig) -> Optional[BaseStep]:
        """Create a step instance"""
        step_class = self.get_step_class(category, name)
        if step_class:
            return step_class(config)
        return None

    def list_steps(self) -> Dict[str, List[Dict[str, str]]]:
        """List all available steps with their info"""
        self.discover_steps()

        result = {}
        for category, steps_in_category in self.steps.items():
            result[category] = []
            for name, step_class in steps_in_category.items():
                # Create temporary instance to get info
                # Ensure a default StepConfig is passed if the step expects one for info
                try:
                    temp_step = step_class(StepConfig())
                    result[category].append({
                        "name": name,
                        "version": temp_step.version,
                        "description": temp_step.description
                    })
                except Exception as e:
                    # Handle cases where step instantiation for info might fail
                    # Or if a step's __init__ is more complex than just StepConfig
                    print(f"Could not get info for step {category}.{name}: {e}")
                    result[category].append({
                        "name": name,
                        "version": "N/A",
                        "description": "Error retrieving description"
                    })
        return result

    def get_categories(self) -> List[str]:
        """Get all step categories"""
        self.discover_steps()
        return list(self.steps.keys())