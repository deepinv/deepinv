from sphinx_gallery.py_source_parser import split_code_and_text_blocks
from sphinx_gallery.notebook import jupyter_notebook, save_notebook
from sphinx_gallery import gen_gallery
from pathlib import Path
import copy
import nbformat
import ast
import re
from typing import Dict, List, Tuple

###############################################################################
# Citation handling
###############################################################################

_CITATION_ROLE_PATTERN = re.compile(r":(?:footcite|cite)(?::[pt])?:`([^`]+)`")


def _load_bib_database(bib_path: Path) -> Dict[str, dict]:
    """Load a BibTeX database and return a mapping key -> entry dict.

    Tries to use ``bibtexparser`` if available; if not, falls back to a very
    lightweight (and imperfect) parser that splits on lines starting with '@'.
    The fallback supports simple key/value extraction for fields used here.
    """
    if not bib_path.exists():
        return {}
    src = bib_path.read_text(encoding="utf-8")
    entries = {}
    raw_items = re.split(r"\n@", src)
    for i, raw in enumerate(raw_items):
        if i == 0 and not raw.startswith("@"):
            # Skip header before first entry
            if not raw.strip().startswith("@"):
                continue
        if not raw.startswith("@"):
            raw = "@" + raw
        m = re.match(r"@\w+\{([^,]+),", raw)
        if not m:
            continue
        key = m.group(1).strip()
        body = raw[m.end() :]
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[\{\"]([^\{\"}]*)[\}\"]", body):
            fname = fm.group(1).lower()
            fval = fm.group(2).strip()
            fields[fname] = fval
        entries[key] = fields
    return entries


def _format_inline_citation(entry: dict) -> Tuple[str, str]:
    """Return (display_text, url) for inline citation.

    display_text follows "AuthorLast et al., YEAR" rules:
    - 1 author: Lastname YEAR
    - 2 authors: Lastname1 & Lastname2 YEAR
    - >=3 authors: Lastname1 et al., YEAR

    URL preference order: doi -> url -> arxiv(eprint) -> empty string.
    """
    authors_raw = entry.get("author") or entry.get("Authors") or ""
    year = entry.get("year", "n.d.")
    # Split authors on ' and '
    authors = [a.strip() for a in re.split(r"\band\b", authors_raw) if a.strip()]

    def last_name(author: str) -> str:
        # Handle "Last, First" vs "First Last" forms
        if "," in author:
            return author.split(",", 1)[0].strip()
        parts = author.split()
        return parts[-1] if parts else author

    display: str
    if not authors:
        display = year
    elif len(authors) == 1:
        display = f"{last_name(authors[0])} {year}"
    elif len(authors) == 2:
        display = f"{last_name(authors[0])} & {last_name(authors[1])} {year}"
    else:
        display = f"{last_name(authors[0])} et al., {year}"

    # URL resolution
    doi = entry.get("doi") or entry.get("DOI")
    url = entry.get("url") or entry.get("URL")
    eprint = entry.get("eprint")
    archive_prefix = entry.get("archiveprefix", "").lower()
    if doi:
        link = f"https://doi.org/{doi}"
    elif url:
        link = url
    elif eprint and (archive_prefix == "arxiv" or "arxiv" in (url or "")):
        link = f"https://arxiv.org/abs/{eprint}"
    else:
        link = ""
    return display, link


def _replace_citation_roles(
    text: str, bib_db: Dict[str, dict], cited_keys: List[str]
) -> str:
    """Replace Sphinx citation roles with inline Markdown author-year hyperlinks.

    Collects citation keys into cited_keys preserving order of first appearance.
    Supports multiple keys separated by commas or spaces inside a single role.
    """

    def repl(m: re.Match) -> str:
        raw_keys = m.group(1)
        # Split keys on comma/space
        keys = [k.strip() for k in re.split(r"[,\s]+", raw_keys) if k.strip()]
        rendered = []
        for key in keys:
            if key not in cited_keys:
                cited_keys.append(key)
            entry = bib_db.get(key)
            if not entry:
                rendered.append(key)
                continue
            display, link = _format_inline_citation(entry)
            if link:
                rendered.append(f"[{display}]({link})")
            else:
                rendered.append(display)
        # Join multiple citations with '; '
        return "(" + "; ".join(rendered) + ")"

    return _CITATION_ROLE_PATTERN.sub(repl, text)


def _build_references_cell(cited_keys: List[str], bib_db: Dict[str, dict]):
    """Create a Markdown cell listing full references for cited keys."""
    if not cited_keys:
        return None
    lines = ["## References\n\n"]
    for key in cited_keys:
        entry = bib_db.get(key)
        if not entry:
            lines.append(f"- {key}\n")
            continue
        authors_raw = entry.get("author", "").replace("\n", " ")
        title = entry.get("title", "").rstrip(".")
        year = entry.get("year", "n.d.")
        journal = (
            entry.get("journal")
            or entry.get("booktitle")
            or entry.get("publisher")
            or ""
        )
        doi = entry.get("doi") or entry.get("DOI")
        url = entry.get("url") or entry.get("URL")
        eprint = entry.get("eprint")
        archive_prefix = entry.get("archiveprefix", "").lower()
        # Simple author formatting: keep as-is but compress spaces
        authors_fmt = re.sub(r"\s+", " ", authors_raw).strip()
        ref = f"- {authors_fmt} ({year}). *{title}*."
        if journal:
            ref += f" {journal}."
        if doi:
            ref += f" [doi:{doi}](https://doi.org/{doi})"
        elif url:
            ref += f" [link]({url})"
        elif eprint and archive_prefix == "arxiv":
            ref += f" [arXiv:{eprint}](https://arxiv.org/abs/{eprint})"
        ref += "\n"
        lines.append(ref)
    cell = nbformat.v4.new_markdown_cell(source="".join(lines))
    cell.metadata["language"] = "markdown"
    return cell


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


def _scan_sphinx_labels(html_root: Path) -> dict:
    """Scan built HTML files to map label id -> relative HTML path.

    This is a lightweight fallback avoiding parsing ``objects.inv``.
    """
    label_map = {}
    if not html_root.exists():
        return label_map
    for html_file in html_root.rglob("*.html"):
        try:
            content = html_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = html_file.relative_to(html_root).as_posix()
        # Capture id attributes. We avoid duplicates, first occurrence wins.
        for m in re.finditer(r'id="([A-Za-z0-9_\-]+)"', content):
            label = m.group(1)
            label_map.setdefault(label, rel)
    return label_map


def _convert_backticked_label_refs(text: str, baseurl: str, label_map: dict) -> str:
    """Convert inline-code style references like `` `Some title <label>` `` to links.

    This compensates for upstream conversions that strip Sphinx roles and leave
    the inner ``Title <label>`` wrapped in backticks.
    """
    # Title: allow anything except backtick; Label: conservative id chars.
    pattern = re.compile(r"`([^`<>]+?)\s*<\s*([A-Za-z0-9_\-]+)\s*>`")

    def repl(m: re.Match):
        title = m.group(1).strip()
        label = m.group(2).strip()
        rel = label_map.get(label)
        if not rel:
            return title  # drop code ticks if unresolved
        url = (
            f"{baseurl}{rel}#{label}"
            if not rel.startswith("http")
            else f"{rel}#{label}"
        )
        return f"[{title}]({url})"

    return pattern.sub(repl, text)


def _convert_sg_example_refs(text: str, baseurl: str) -> str:
    """Convert sphinx-gallery example refs to Markdown links to built docs pages.

    Supported inputs:
    - Bare token (optionally backticked): `sphx_glr_auto_examples_{section}_{slug}.py`
    - With explicit title (optionally backticked): `Title <sphx_glr_auto_examples_{section}_{slug}.py>`

    Target URL:
    {base}/auto_examples/{section}/{slug}.html#sphx-glr-auto-examples-{section}-{slug-dashed}-py

    For bare tokens, preserves backticks by keeping them inside the link text.
    For titled form, uses the provided title as display text (without code ticks).
    """
    # 1) Titled form (with or without surrounding backticks): Title <sphx_glr_auto_examples_section_slug.py>
    titled_pattern = re.compile(
        r"`?([^`<>]+?)\s*<\s*sphx_glr_auto_examples_([a-z0-9\-]+)_([a-z0-9_\-]+)\.py\s*>`?",
        re.IGNORECASE,
    )

    def repl_titled(m: re.Match):
        title = m.group(1).strip()
        section = m.group(2)
        slug = m.group(3)
        anchor_slug = slug.replace("_", "-")
        url = (
            f"{baseurl}auto_examples/{section}/{slug}.html"
            f"#sphx-glr-auto-examples-{section}-{anchor_slug}-py"
        )
        # Use the explicit title as-is (no backticks)
        return f"[{title}]({url})"

    text = titled_pattern.sub(repl_titled, text)

    # 2) Bare token (with or without surrounding backticks): sphx_glr_auto_examples_section_slug.py
    bare_pattern = re.compile(
        r"`?sphx_glr_auto_examples_([a-z0-9\-]+)_([a-z0-9_\-]+)\.py`?",
        re.IGNORECASE,
    )

    def repl_bare(m: re.Match):
        section = m.group(1)
        slug = m.group(2)
        display = f"`{section}/{slug}.py`"  # keep code-style for bare tokens
        anchor_slug = slug.replace("_", "-")
        url = (
            f"{baseurl}auto_examples/{section}/{slug}.html"
            f"#sphx-glr-auto-examples-{section}-{anchor_slug}-py"
        )
        return f"[{display}]({url})"

    return bare_pattern.sub(repl_bare, text)


def _convert_sphinx_admonitions(text: str) -> str:
    """Convert simple Sphinx admonitions like '.. tip::' blocks into Markdown blockquotes.

    Example:
        .. hint:: Optional title
           body line 1
           body line 2

    becomes
        > Hint: Optional title
        >
        > body line 1
        > body line 2

    Handles common admonitions and also tolerates a non-standard plural like '.. tips::'.
    """
    lines = text.splitlines()
    out = []
    i = 0
    dir_re = re.compile(r"^\s*\.\.\s+([A-Za-z]+)s?::\s*(.*)$")
    known = {
        "note": "Note",
        "tip": "Tip",
        "hint": "Hint",
        "warning": "Warning",
        "important": "Important",
        "caution": "Caution",
        "attention": "Attention",
        "danger": "Danger",
        "error": "Error",
    }
    while i < len(lines):
        m = dir_re.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        role = m.group(1).lower()
        title = m.group(2).strip()
        label = known.get(role, role.capitalize())
        header = f"> **{label}**"
        if title:
            header += f": {title}"
        out.append(header)
        out.append(">")
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() == "":
                out.append(">")
                i += 1
                continue
            if line.startswith(" ") or line.startswith("\t"):
                out.append("> " + line.lstrip())
                i += 1
            else:
                break
        out.append("")
    return "\n".join(out)


def _remove_footbibliography_blocks(text: str) -> str:
    """Remove any References section markers meant for Sphinx footbibliography.

    Specifically strips lines like ':References:' and '.. footbibliography::'.
    Also collapses consecutive blank lines resulting from the removal.
    """
    lines = text.splitlines()
    out = []
    pat_ref = re.compile(r"^\s*:References:\s*$", re.IGNORECASE)
    pat_footb = re.compile(r"^\s*\.\.\s*footbibliography::\s*$", re.IGNORECASE)
    for ln in lines:
        if pat_ref.match(ln) or pat_footb.match(ln):
            continue
        out.append(ln)
    # Collapse multiple blank lines
    collapsed = []
    prev_blank = False
    for ln in out:
        is_blank = ln.strip() == ""
        if is_blank and prev_blank:
            continue
        collapsed.append(ln)
        prev_blank = is_blank
    return "\n".join(collapsed).strip()


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

    # Convert Sphinx roles, gallery refs, admonitions, and add inline citations
    try:
        repo_root = Path(__file__).resolve().parents[2]
        conf_path = repo_root / "docs" / "source" / "conf.py"
        baseurl = _extract_html_baseurl(conf_path)
        # Build label map once (scan built docs).
        html_root = repo_root / "docs" / "build" / "html"
        label_map = _scan_sphinx_labels(html_root)
        bib_path = repo_root / "docs" / "source" / "refs.bib"
        bib_db = _load_bib_database(bib_path)
        cited_keys: List[str] = []
        new_cells = []
        for cell in example_nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                src = cell.get("source", "")
                src_text = "".join(src) if isinstance(src, list) else src
                # 0) Remove any Sphinx footbibliography markers
                src_text = _remove_footbibliography_blocks(src_text)
                if not src_text:
                    # Skip empty cell after removal
                    continue
                # 1) Citations first to capture original roles before other rewrites
                new_text = _replace_citation_roles(src_text, bib_db, cited_keys)
                # 2) Convert Sphinx object roles
                new_text = _convert_sphinx_roles_to_links(new_text, baseurl)
                new_text = _convert_backticked_label_refs(new_text, baseurl, label_map)
                new_text = _convert_sg_example_refs(new_text, baseurl)
                new_text = _convert_sphinx_admonitions(new_text)
                cell["source"] = new_text
                cell.setdefault("metadata", {})
                cell["metadata"]["language"] = "markdown"
                new_cells.append(cell)
            else:
                new_cells.append(cell)
        example_nb["cells"] = new_cells
        # Append references cell if citations collected
        ref_cell = _build_references_cell(cited_keys, bib_db)
        if ref_cell is not None:
            example_nb["cells"].append(ref_cell)
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
