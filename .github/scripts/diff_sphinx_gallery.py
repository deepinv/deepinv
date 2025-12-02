import subprocess
import sys
import re
import os
import ast
import glob
from pathlib import Path


def get_changed_files(base_branch, current_branch):
    """Runs git diff to get the names of all changed files."""
    diff_range = f'{base_branch}..{current_branch}'

    try:
        # We need to capture the output of 'git diff --name-only'
        result = subprocess.run(
            ['git', 'diff', '--name-only', diff_range],
            capture_output=True,
            text=True,
            check=True
        )
        # Return a set for faster lookup later
        return set(result.stdout.strip().split('\n'))
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Git command not found. Ensure Git is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)


def get_example_dependencies(example_file_path: Path, core_lib_path: Path, core_lib_name: str) -> set[Path]:
    """
    Statically analyzes a Python file to find all top-level imports from the core library.
    Returns a set of Paths pointing to the actual dependent files in the core library.
    """
    dependencies = set()
    try:
        # Read file content and parse it into an Abstract Syntax Tree (AST)
        tree = ast.parse(example_file_path.read_text(encoding='utf-8'))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # Handles: import your_project_name.module_a
                for alias in node.names:
                    if alias.name.startswith(core_lib_name):
                        module_path = Path(*alias.name.split('.'))
                        # Check if it's a file (.py) or a directory (__init__.py)
                        if (core_lib_path / module_path).is_dir():
                            dependencies.add(core_lib_path / module_path / "__init__.py")
                        else:
                            dependencies.add(core_lib_path / f"{module_path}.py")

            elif isinstance(node, ast.ImportFrom) and node.module:
                # Handles: from your_project_name.module_a import function
                if node.module.startswith(core_lib_name):
                    module_path = Path(*node.module.split('.'))
                    # Assume it's a file unless it resolves to a directory
                    if (core_lib_path / module_path).is_dir():
                        dependencies.add(core_lib_path / module_path / "__init__.py")
                    else:
                        dependencies.add(core_lib_path / f"{module_path}.py")

    except Exception as e:
        # Log error but don't fail CI; default to running the example if parsing fails
        print(f"Warning: Could not parse {example_file_path}. Error: {e}", file=sys.stderr)
        return set()  # Return empty set if parsing failed

    # Convert paths to strings relative to the project root
    return {str(d) for d in dependencies if d.exists()}


def main():
    # Expects BASE_BRANCH and CURRENT_BRANCH as arguments from the shell script
    if len(sys.argv) < 3:
        # Fallback: If run without arguments, assume a local full build is desired
        print(".*")
        return

    base_branch = sys.argv[1]
    current_branch = sys.argv[2]

    # Use the name of the core library (e.g., 'your_project_name')
    CORE_LIB_NAME = 'deepinv'.strip(os.sep)

    # Path objects for easier manipulation
    CORE_LIB_ROOT = Path('deepinv')
    EXAMPLE_ROOT = Path('examples')

    changed_files = get_changed_files(base_branch, current_branch)

    if not changed_files or changed_files == {''}:
        # No files changed, generate a pattern that matches nothing
        print("^$")
        return

    # Set to store the final list of example file paths that need to be run
    examples_to_run = set()

    # 1. Identify all existing example files
    all_example_files = glob.glob(str(EXAMPLE_ROOT / 'demo_*.py'))

    # 2. Iterate over all examples to check for direct change or dependency change
    for example_file_path in all_example_files:
        example_path = Path(example_file_path)

        # A. Check for DIRECT change (the example itself was modified)
        if example_file_path in changed_files:
            examples_to_run.add(example_path)
            continue

        # B. Check for DEPENDENCY change
        # Get the set of core files this example imports
        dependencies = get_example_dependencies(example_path, CORE_LIB_ROOT, CORE_LIB_NAME)

        # Check if any of the example's dependencies are in the changed_files set
        if dependencies.intersection(changed_files):
            examples_to_run.add(example_path)

    if examples_to_run:
        # Generate a regex pattern that matches *only* the affected example filenames

        # 1. Extract just the example filename (e.g., 'plot_basic.py')
        example_names = [f.name for f in examples_to_run]

        # 2. Escape special characters and join with '|' (OR operator)
        pattern = "|".join([re.escape(name) for name in example_names])

        # 3. Create the final regex that matches the full path ending with one of the names
        final_pattern = f".*({pattern})$"
        print(final_pattern)
    else:
        # Only documentation or non-example files changed that are not imported
        print("^$")


if __name__ == '__main__':
    main()