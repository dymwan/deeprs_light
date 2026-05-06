"""
Registry mechanism for managing datasets, transforms, backends, and other
components via string-based lookup.
"""

from typing import Dict, Type, List, Optional, Callable, Union


class Registry:
    """
    A registry for managing named components.

    Usage:
        DATASETS = Registry("datasets")

        @DATASETS.register()
        class MyDataset(DeepRSCocoDataset):
            ...

        @DATASETS.register("custom")
        class CustomDataset(DeepRSCocoDataset):
            ...

        cls = DATASETS.get("my_dataset")
    """

    def __init__(self, name: str):
        """
        Args:
            name: Registry name, used for logging and debugging.
        """
        self._name = name
        self._registry: Dict[str, Type] = {}

    def register(
        self, name: Union[str, Callable, None] = None
    ) -> Callable:
        """
        Decorator or manual registration method.

        As a decorator:
            @registry.register("name")
            @registry.register()

        As a manual call:
            registry.register("name", cls)

        Args:
            name: Registration name. If None, uses the decorated class's
                  lowercase name.

        Returns:
            A decorator function, or the registered class directly if
            called manually with a class as the first argument.
        """
        # Handle manual registration: register("name", cls)
        if isinstance(name, type):
            raise TypeError(
                f"Use @{self._name}.register('name') or"
                f" @{self._name}.register() as a decorator,"
                f" or call {self._name}.register('name', cls) manually."
            )

        def _decorator(cls: Type) -> Type:
            register_name = name
            if register_name is None:
                register_name = cls.__name__.lower()
            if register_name in self._registry:
                raise KeyError(
                    f"'{register_name}' is already registered in"
                    f" '{self._name}'. Existing: {list(self._registry.keys())}"
                )
            self._registry[register_name] = cls
            return cls

        # If called as @register() with a class directly (no name argument)
        if isinstance(name, type):
            cls = name
            name = None
            return _decorator(cls)

        return _decorator

    def get(self, name: str) -> Type:
        """
        Get a registered class by name.

        Args:
            name: The registered name.

        Returns:
            The registered class.

        Raises:
            KeyError: If the name is not found, with a hint of available names.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in registry '{self._name}'."
                f" Available: {available}"
            )
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        """Support 'name' in registry syntax."""
        return name in self._registry

    def list(self) -> List[str]:
        """List all registered names."""
        return list(self._registry.keys())

    def __repr__(self) -> str:
        return (
            f"Registry(name='{self._name}',"
            f" registered={len(self._registry)})"
        )


# Pre-created global registries
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
CACHE_BACKENDS = Registry("cache_backends")
