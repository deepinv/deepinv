import re
import glob
from pathlib import Path
import subprocess
import sys
import ast


def safe_parse_python_source(source):
    """Parse Python source into an AST safely."""
    try:
        return ast.parse(source)
    except Exception:
        return None


def read_file_from_git(branch, file_path):
    """Returns file contents from a specific branch using `git show`."""
    try:
        result = subprocess.run(
            ["git", "show", f"{branch}:{file_path}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None  # File may not exist in base branch


def read_current_file(file_path):
    """Read file from filesystem."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def get_top_level_symbol_nodes(tree):
    """
    Return a dict mapping symbol name -> AST node
    for top-level classes and functions.
    """
    if tree is None:
        return {}

    symbols = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols[node.name] = node
    return symbols


def node_source(source_code, node):
    """
    Extract the exact source text of an AST node.
    Requires Python 3.8+ (ast gives lineno + end_lineno).
    """
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return ""
    lines = source_code.splitlines()
    snippet = lines[node.lineno - 1 : node.end_lineno]
    return "\n".join(snippet)


def get_changed_files(base_branch, current_branch):
    """Runs git diff to get the names of all changed files."""
    diff_range = f"{base_branch}..{current_branch}"

    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", diff_range],
            capture_output=True,
            text=True,
            check=True,
        )
        return set(result.stdout.strip().split("\n"))
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Git is not installed.", file=sys.stderr)
        sys.exit(1)


def get_changed_symbols(base_branch, changed_files):
    """
    Returns:
    {
        "file.py": ["SymbolA", "SymbolB", ...],  # modified symbols only
    }
    """

    python_files = [f for f in changed_files if f.endswith(".py")]

    changed_symbols = {}

    for f in python_files:
        base_src = read_file_from_git(base_branch, f)
        curr_src = read_current_file(f)

        if base_src is None or curr_src is None:
            continue  # skip new or unreadable files

        base_tree = safe_parse_python_source(base_src)
        curr_tree = safe_parse_python_source(curr_src)

        base_nodes = get_top_level_symbol_nodes(base_tree)
        curr_nodes = get_top_level_symbol_nodes(curr_tree)

        # Only consider symbols present in both versions
        common_symbols = set(base_nodes.keys()) & set(curr_nodes.keys())

        modified = []

        for name in common_symbols:
            base_code = node_source(base_src, base_nodes[name])
            curr_code = node_source(curr_src, curr_nodes[name])

            if base_code != curr_code:
                modified.append(name)

        if modified:
            changed_symbols[f] = modified

    return changed_symbols


def symbols_used_in_file(file_path):
    """
    Returns a set of symbol names (Name and Attribute) used in the given file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
    except Exception:
        return set()

    used = set()

    class Visitor(ast.NodeVisitor):
        def visit_Name(self, node):
            used.add(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            used.add(node.attr)
            self.generic_visit(node)

    Visitor().visit(tree)
    return used


def modified_symbols_used_in_file(target_file, changed_symbols_dict):
    """
    changed_symbols_dict comes from get_changed_symbols()
    {
        "file1.py": ["funcA", "ClassB"],
        "file2.py": ["foo"],
    }

    Returns:
    {
        "file1.py": ["funcA"],   # used
        "file2.py": [],          # not used
    }
    """

    used = symbols_used_in_file(target_file)
    result = {}

    for file, modified_syms in changed_symbols_dict.items():
        found = [sym for sym in modified_syms if sym in used]
        result[file] = found

    return result


def main():
    # Expects BASE_BRANCH and CURRENT_BRANCH as arguments from the shell script
    if len(sys.argv) < 3:
        # Fallback: If run without arguments, assume a local full build is desired
        print(".*")
        return

    base_branch = sys.argv[1]
    current_branch = sys.argv[2]

    # find changed files and symbols
    changed_files = get_changed_files(base_branch, current_branch)
    changed_symbols = get_changed_symbols(base_branch, changed_files)

    # move to Path objects for easier comparison
    # filter only .py files
    changed_files = [f for f in changed_files if f.endswith(".py")]
    # remove all _demo.py files from changed_files
    changed_files = [f for f in changed_files if not f.startswith(str("examples"))]
    changed_files = set(Path(f) for f in changed_files)

    if not changed_files or changed_files == {""}:
        # No files changed, generate a pattern that matches nothing
        print("^$")
        return

    # Set to store the final list of example file paths that need to be run
    examples_to_run = set()

    # 1. Identify all existing example files
    all_example_files = glob.glob(str("examples/*/demo_*.py"))

    # 2. Iterate over all examples to check for direct change or dependency change
    for example_file_path in all_example_files:
        example_path = Path(example_file_path)

        # A. Check for DIRECT change (the example itself was modified)
        if example_path in changed_files:
            examples_to_run.add(example_path)
            continue

        # B. Check for DEPENDENCY change
        usage = modified_symbols_used_in_file(example_path, changed_symbols)
        for changed_file, used_syms in usage.items():
            if used_syms:
                examples_to_run.add(example_path)
                # for debug purposes
                # print(f"Example {example_path} uses modified symbols {used_syms} from {changed_file}")
                break

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


if __name__ == "__main__":
    main()
