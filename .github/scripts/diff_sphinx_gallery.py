import subprocess
import sys
import re

CORE_LIB_PATH = 'deepinv/'
EXAMPLE_PATH = 'examples/'

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
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Git command not found. Ensure Git is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)


def main():
    # Expects BASE_BRANCH and CURRENT_BRANCH as arguments from the shell script
    if len(sys.argv) < 3:
        # Fallback: If run without arguments, assume a local full build is desired
        print(".*")
        return

    base_branch = sys.argv[1]
    current_branch = sys.argv[2]

    changed_files = get_changed_files(base_branch, current_branch)

    if not changed_files or changed_files == ['']:
        # No files changed, generate a pattern that matches nothing (empty string at start/end)
        print("^$")
        return

    # 1. Check for changes in the core library code
    # If any core dependency changes, we must run all examples for safety.
    core_changes = [f for f in changed_files if f.startswith(CORE_LIB_PATH) and f.endswith('.py')]
    if core_changes:
        # Run all examples
        print(".*")
        return

    # 2. Check for changes in example files (only run the modified ones)
    example_files = [
        f for f in changed_files
        # Only consider files that are in the example directory and typically start with 'demo_'
        if f.startswith(EXAMPLE_PATH) and re.search(r'demo_.*\.py$', f)
    ]

    if example_files:
        # Generate a regex pattern that matches *only* the changed example filenames

        # 1. Extract just the example filename
        # The filename_pattern configuration matches the full path, so we construct
        # a regex to match the path ending with one of the changed filenames.
        example_names = [f.split('/')[-1] for f in example_files]

        # 2. Escape special characters and join with '|' (OR operator)
        pattern = "|".join([re.escape(name) for name in example_names])

        # 3. Create the final regex that matches the full path ending with one of the names
        # Example output: ".*(plot_a\.py|plot_b\.py)$"
        final_pattern = f".*({pattern})$"
        print(final_pattern)
    else:
        # Only documentation or non-example files changed - run nothing
        print("^$")


if __name__ == '__main__':
    main()