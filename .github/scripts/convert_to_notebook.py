from sphinx_gallery.py_source_parser import split_code_and_text_blocks
from sphinx_gallery.notebook import jupyter_notebook, save_notebook
from sphinx_gallery import gen_gallery
from pathlib import Path
import copy
import nbformat
import ast


def _extract_mathjax_macros_from_conf(conf_path: Path):
    """Parse docs/source/conf.py to extract MathJax macros without executing it.

    Returns a dict mapping macro name -> tuple(replacement, n_args) or (replacement, 0).
    """
    try:
        conf_src = conf_path.read_text(encoding="utf-8")
        tree = ast.parse(conf_src, filename=str(conf_path))
        macros = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "mathjax3_config":
                        cfg = ast.literal_eval(node.value)
                        tex_cfg = cfg.get("tex", {}) if isinstance(cfg, dict) else {}
                        raw_macros = (
                            tex_cfg.get("macros", {})
                            if isinstance(tex_cfg, dict)
                            else {}
                        )
                        for name, val in raw_macros.items():
                            if isinstance(val, str):
                                macros[name] = (val, 0)
                            elif (
                                isinstance(val, (list, tuple))
                                and len(val) == 2
                                and isinstance(val[0], str)
                            ):
                                try:
                                    n_args = int(val[1])
                                except Exception:
                                    n_args = 0
                                macros[name] = (val[0], n_args)
                        return macros
        return {}
    except Exception:
        # Be tolerant: if anything goes wrong, just return empty to avoid breaking conversion
        return {}


def _build_macros_markdown_cell(macros: dict):
    """Create a Markdown cell content declaring macros with LaTeX newcommand.

    We put the definitions inside a display math block so MathJax processes them,
    and they remain available for subsequent cells.
    """
    if not macros:
        return None
    lines = [
        "<!-- MathJax macro definitions inserted automatically -->\n",
        "$$\n",
    ]
    for name, (body, n_args) in macros.items():
        # Ensure leading backslash in macro name and preserve body as-is
        if n_args and n_args > 0:
            lines.append(f"\\newcommand{{\\{name}}}[{n_args}]{{{body}}}\n")
        else:
            lines.append(f"\\newcommand{{\\{name}}}{{{body}}}\n")
    lines.append("$$\n")

    cell = nbformat.v4.new_markdown_cell(source="".join(lines))
    cell.metadata["language"] = "markdown"
    return cell


def convert_script_to_notebook(src_file: Path, output_file: Path, gallery_conf):
    """
    Convert a single Python script to a Jupyter notebook and save it under target_root,
    preserving relative path.
    """
    # Parse the Python file
    file_conf, blocks = split_code_and_text_blocks(str(src_file))

    # Convert to notebook
    example_nb = jupyter_notebook(blocks, gallery_conf, str(src_file.parent))

    # Prepend an installation cell for deepinv.
    # We only add it if the first cell does not already contain a pip install for deepinv.
    try:
        first_source = (
            "".join(example_nb["cells"][0].get("source", []))
            if example_nb.get("cells")
            else ""
        )
    except Exception:
        first_source = ""
    install_cmd = "%pip install deepinv"
    if "pip install" not in first_source or "deepinv" not in first_source:
        install_cell = nbformat.v4.new_code_cell(
            source="# Install deepinv (skip if already installed)\n" + install_cmd
        )
        # Sphinx-gallery does not set metadata.language by default; add for consistency.
        install_cell.metadata["language"] = "python"
        example_nb["cells"].insert(0, install_cell)

    # Insert MathJax macros markdown cell right after the install cell (or at top if none).
    try:
        repo_root = Path(__file__).resolve().parents[2]
        conf_path = repo_root / "docs" / "source" / "conf.py"
        macros = _extract_mathjax_macros_from_conf(conf_path)
        # Avoid duplicate insertion if a cell already contains newcommand definitions
        has_macros_cell = any(
            isinstance(c.get("source"), (str, list))
            and (
                "\\newcommand{"
                in (
                    c.get("source")
                    if isinstance(c.get("source"), str)
                    else "".join(c.get("source"))
                )
            )
            for c in example_nb.get("cells", [])
        )
        macros_cell = _build_macros_markdown_cell(macros)
        if macros_cell is not None and not has_macros_cell:
            # Determine insertion index: 1 if we added install cell, else 0
            insert_idx = (
                1
                if example_nb.get("cells")
                and "pip install"
                in (
                    "".join(example_nb["cells"][0].get("source", []))
                    if isinstance(example_nb["cells"][0].get("source", []), list)
                    else example_nb["cells"][0].get("source", "")
                )
                else 0
            )
            example_nb["cells"].insert(insert_idx, macros_cell)
    except Exception:
        # Do not fail conversion if anything goes wrong
        pass

    # Ensure the parent directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save notebook
    save_notebook(example_nb, output_file)
    print(f"Notebook saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert all Python scripts to notebooks."
    )
    parser.add_argument(
        "--input",
        default="examples/basics/demo_quickstart.py",
        help="Path to the Python script to convert",
    )
    parser.add_argument(
        "--output",
        default="examples/_notebooks/basics/demo_quickstart.ipynb",
        help="Path to save the converted notebook",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    target_path = Path(args.output)

    # Use default gallery configuration
    gallery_conf = copy.deepcopy(gen_gallery.DEFAULT_GALLERY_CONF)
    convert_script_to_notebook(input_path, target_path, gallery_conf)
