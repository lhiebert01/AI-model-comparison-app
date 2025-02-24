import os
from datetime import datetime

def create_project_structure(start_path=".", output_file="project-structure.txt"):
    """Create a detailed project structure map."""
    
    with open(output_file, "w", encoding='utf-8') as f:
        # Write header
        f.write(f"Project Structure Generated on: {datetime.now()}\n")
        f.write(f"Root Directory: {os.path.abspath(start_path)}\n\n")
        
        # Track all files in root
        root_files = []
        
        # Walk through directory
        for root, dirs, files in os.walk(start_path):
            # Skip venv and node_modules directories
            if 'venv' in dirs:
                dirs.remove('venv')
            if 'node_modules' in dirs:
                dirs.remove('node_modules')
            
            # Get relative path
            level = root.replace(start_path, '').count(os.sep)
            indent = '│   ' * level
            folder = os.path.basename(root)
            
            # Store root files for later
            if level == 0:
                root_files = files
                f.write("Directory Structure:\n")
            else:
                f.write(f"{indent}├── {folder}/\n")
            
            # Add files
            sub_indent = '│   ' * (level + 1)
            for file in files:
                if level != 0:  # Skip root files for now
                    f.write(f"{sub_indent}├── {file}\n")
        
        # Write root files at the top
        f.write("\nFiles in Root Directory:\n")
        for file in sorted(root_files):
            f.write(f"├── {file}\n")

if __name__ == "__main__":
    create_project_structure()
    print("Project structure has been written to project-structure.txt")