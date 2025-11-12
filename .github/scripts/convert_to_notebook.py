from sphinx_gallery.py_source_parser import split_code_and_text_blocks
from sphinx_gallery.notebook import jupyter_notebook, save_notebook
from sphinx_gallery import gen_gallery
from pathlib import Path
import copy
import nbformat
import ast
import re


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
            lines.append(f"\\renewcommand{{\\{name}}}[{n_args}]{{{body}}}\n")
        else:
            lines.append(f"\\renewcommand{{\\{name}}}{{{body}}}\n")
    lines.append("$$\n")

    cell = nbformat.v4.new_markdown_cell(source="".join(lines))
    cell.metadata["language"] = "markdown"
    return cell


def _extract_html_baseurl(conf_path: Path) -> str:
    """Extract html_baseurl from Sphinx conf safely via AST; fallback to project URL."""
    try:
        conf_src = conf_path.read_text(encoding="utf-8")
        tree = ast.parse(conf_src, filename=str(conf_path))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "html_baseurl":
                        url = ast.literal_eval(node.value)
                        if isinstance(url, str) and url:
                            return url.rstrip("/") + "/"
        return "https://deepinv.github.io/deepinv/"
    except Exception:
        return "https://deepinv.github.io/deepinv/"


def _convert_sphinx_roles_to_links(text: str, baseurl: str) -> str:
    """Convert Sphinx roles like :func:`deepinv.a.b` to Markdown links pointing to built docs.

    Supports roles: func, class, meth, mod, attr, data, obj. Also handles
    explicit text form: :func:`Text <deepinv.a.b>`.

    URL mapping based on this repo's docs:
    - Objects (func/class/meth/attr/data/obj):  {base}/api/stubs/{full.dotted.name}.html
    - Modules (mod):                            {base}/api/{module.dotted.name}.html
    """
    role_pattern = re.compile(r":(func|class|meth|mod|attr|data|obj):`([^`]+)`")

    def target_to_url(role: str, target: str):
        m = re.match(r"^\s*(.*?)\s*<\s*([^>]+)\s*>\s*$", target)
        if m:
            disp = m.group(1).strip()
            full = m.group(2).strip()
        else:
            disp = target.strip()
            full = target.strip()

        if role == "mod":
            url = f"{baseurl}api/{full}.html"
            display = disp or f"`{full}`"
            return display, url
        else:
            if not full or "." not in full:
                return None
            url = f"{baseurl}api/stubs/{full}.html"
            display = disp or f"`{full}`"
            return display, url

    def repl(m: re.Match):
        role = m.group(1)
        target = m.group(2)
        try:
            res = target_to_url(role, target)
        except Exception:
            res = None
        if not res:
            mm = re.match(r"^\s*(.*?)\s*<\s*([^>]+)\s*>\s*$", target)
            return mm.group(1) if mm and mm.group(1) else target
        display, url = res
        display_md = display
        if (
            display_md
            and not display_md.startswith("`")
            and "." in display_md
            and " " not in display_md
        ):
            display_md = f"`{display_md}`"
        return f"[{display_md}]({url})"

    return role_pattern.sub(repl, text)


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
                "\\renewcommand{"
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

    # Convert Sphinx roles in markdown cells to links
    try:
        repo_root = Path(__file__).resolve().parents[2]
        conf_path = repo_root / "docs" / "source" / "conf.py"
        baseurl = _extract_html_baseurl(conf_path)
        for cell in example_nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                src = cell.get("source", "")
                src_text = "".join(src) if isinstance(src, list) else src
                new_text = _convert_sphinx_roles_to_links(src_text, baseurl)
                cell["source"] = new_text
                cell.setdefault("metadata", {})
                cell["metadata"]["language"] = "markdown"
    except Exception:
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
