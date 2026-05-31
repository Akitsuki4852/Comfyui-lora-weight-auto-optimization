try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception:
    import importlib
    import pathlib
    import sys

    package_root = pathlib.Path(__file__).resolve().parent
    if str(package_root) not in sys.path:
        sys.path.append(str(package_root))

    imported = importlib.import_module("nodes")
    NODE_CLASS_MAPPINGS = imported.NODE_CLASS_MAPPINGS
    NODE_DISPLAY_NAME_MAPPINGS = imported.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
