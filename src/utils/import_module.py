import importlib.util
import sys
import os

def safe_import_module(module_name, file_path):
    try:
        abs_path = os.path.abspath(file_path)
        
        module_dir = os.path.dirname(abs_path)
        if module_dir not in sys.path:
            sys.path.append(module_dir)
            
        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None:
            raise ImportError(f"Could not find module at {abs_path}")
            
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
        
    except Exception as e:
        print(f"Failed to import {module_name} from {file_path}")
        print(f"Error: {str(e)}")
        print(f"Current sys.path: {sys.path}")
        raise