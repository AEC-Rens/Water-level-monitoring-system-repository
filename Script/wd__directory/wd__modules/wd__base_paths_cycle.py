# wd__modules/wd__base_paths_cycle.py

def get_base_paths(image_path: str, output_path: str) -> dict:
    """
    Return a dictionary with standardized keys for image and output paths.
    This centralizes path definitions for easier maintenance.
    """
    return {
        "image_path": image_path,
        "output_path": output_path
    }
