import os
from sphinx.application import Sphinx
from huggingface_hub import snapshot_download

# global variable storing benchmark mappings to be used in class templates
benchmark_mapping = {}

_HERE = os.path.dirname(__file__)
# Committed template for the main benchmarks landing page (``source/benchmarks.rst``).
# It is excluded from the Sphinx build via ``exclude_patterns``; the filled copy is
# written under ``auto_benchmarks/`` at build time.
MAIN_BENCHMARK_TEMPLATE = os.path.join(_HERE, "benchmarks.rst")
# Template for an individual benchmark page.
SINGLE_BENCHMARK_TEMPLATE = os.path.join(_HERE, "_templates", "benchmark_template.rst")


def generate_benchmarks(app):
    r"""
    Generate RST files for benchmarks from csv results files.

    This function downloads benchmarks from the Hugging Face repository https://huggingface.co/datasets/deepinv/benchmarks
    and generates one rst file for each results.csv file found in the repository.
    It additionally generates a main benchmarks.rst file listing all benchmarks.

    :param Sphinx app: The Sphinx application object.
    """
    # Define the root directory of the benchmarks repository
    benchmarks_root = os.path.join(os.path.dirname(__file__), "deepinv-benchmarks")

    # Load the dataset from Hugging Face and save it to disk
    snapshot_download(
        repo_id="deepinv/benchmarks", repo_type="dataset", local_dir=benchmarks_root
    )

    # Recursively find all .csv files
    csv_files = []
    for root, dirs, files in os.walk(benchmarks_root):
        for file in files:
            if file.endswith("csv"):
                csv_files.append((root, os.path.join(root, file)))

    if not csv_files:
        raise FileNotFoundError("No .csv files found in the benchmarks repository.")

    # Define the output directory for RST files
    source_dir = os.path.dirname(__file__)
    output_dir = os.path.join(source_dir, "auto_benchmarks")

    benchmark_info = []
    # process each csv file
    for folder, csv_file in csv_files:
        dataset, physics, noise, benchmark_name = benchmark_rst_from_csv(
            csv_file, output_dir
        )
        benchmark_info.append((benchmark_name, dataset, physics, noise))

        # We save mappings for classes used in dataset, physics, and noise
        # to later be included in the docstrings of these classes (see source/_templates/myclass_template.rst)
        if dataset not in benchmark_mapping:
            benchmark_mapping[dataset] = [benchmark_name]
        else:
            benchmark_mapping[dataset].append(benchmark_name)

        if physics not in benchmark_mapping:
            benchmark_mapping[physics] = [benchmark_name]
        else:
            benchmark_mapping[physics].append(benchmark_name)

        if noise not in benchmark_mapping:
            benchmark_mapping[noise] = [benchmark_name]
        else:
            benchmark_mapping[noise].append(benchmark_name)

    generate_main_benchmark_rst(benchmark_info, source_dir)


def benchmark_rst_from_csv(csv_path, output_dir):
    r"""
    Generate an .rst file from a csv benchmark results file.

    :param str, Path csv_path: Path to the results.csv file.
    :param str, Path output_dir: Directory where the rst file will be saved.
    :return: Tuple containing dataset, physics, and noise model names.
    :rtype: tuple[str, str, str, str]
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    dataset = (
        df["dataset_name"][0].split("[")[0]
        if "dataset_name" in df.columns
        else "Unknown"
    )
    physics = (
        df["p_dataset_physics"][0] if "p_dataset_physics" in df.columns else "Unknown"
    )
    noise = df["p_dataset_noise"][0] if "p_dataset_noise" in df.columns else "Unknown"

    benchmark_name = str(df["objective_name"][0])
    benchmark_link = benchmark_name.replace(" ", "_").replace("-", "_").lower()

    # Dataset parameter bullet list (e.g. sigma, img_size), one blank line apart.
    dataset_cols = [
        col
        for col in df.columns
        if col.startswith("p_dataset")
        and col != "p_dataset_physics"
        and col != "p_dataset_noise"
    ]
    dataset_params = "\n\n".join(
        f"- *{col.replace('p_dataset_', '')}*: {df.at[0, col]}"
        for col in dataset_cols
        if col.replace("p_dataset_", "") != "debug"
    )

    # filter duplicated rows by repeated solver_name entries, keeping only the first one
    df = df.drop_duplicates(subset=["solver_name"], keep="last")

    # Add link to 'model' column if exists
    if "solver_name" in df.columns and "file_solver" in df.columns:
        url = (
            "https://github.com/deepinv/benchmarks/blob/main/deepinv_bench/benchmarks/"
            + benchmark_link
            + "/"
        )
        for i, row in df.iterrows():
            model_name = row["solver_name"]
            file_link = url + row["file_solver"]
            link_cell = f"`{model_name} <{file_link}>`_"
            df.at[i, "solver_name"] = link_cell  # Update the DataFrame directly

    # extract metrics
    metric_cols = [
        col
        for col in df.columns
        if col.startswith("objective_") and col != "objective_name"
    ]
    std_cols = [col for col in metric_cols if col.endswith("_std")]
    mean_cols = [col for col in metric_cols if col not in std_cols]

    # Pre-sort rows by PSNR (descending)
    psnr_cols = [col for col in mean_cols if "psnr" in col.lower()]
    if psnr_cols:
        df = df.sort_values(by=psnr_cols[0], ascending=False, na_position="last")

    kept_cells = ["solver_name"] + mean_cols + std_cols

    # Filter the dataframe to keep only these columns
    df = df[df.columns.intersection(kept_cells)]

    # perform the mean +- std formatting
    for col in mean_cols:
        std_col = col + "_std"
        if std_col in df.columns:  # Check if it exists in the filtered DF
            df[col] = df.apply(
                lambda row: f"{row[col]:.2f} |plusminus| {row[std_col]:.2f}", axis=1
            )
            # Safe to drop now
            df = df.drop(columns=[std_col])

    # remove "objective_" prefix from metric columns
    df = df.rename(columns={col: col.replace("objective_", "") for col in df.columns})

    # Build the list-table body: header row followed by one row per solver.
    table_lines = ["   * - " + "\n     - ".join(map(str, df.columns.tolist()))]
    for _, row in df.iterrows():
        row_cells = [
            f"{val:.2f}" if isinstance(val, float) else str(val) for val in row
        ]
        table_lines.append("   * - " + "\n     - ".join(row_cells))
    table_rows = "\n".join(table_lines)

    # Fill the single-benchmark template with the values computed above.
    with open(SINGLE_BENCHMARK_TEMPLATE) as f:
        content = f.read()
    replacements = {
        "%%LABEL%%": benchmark_link,
        "%%TITLE%%": benchmark_name,
        "%%UNDERLINE%%": "=" * len(benchmark_name),
        "%%DATASET%%": dataset,
        "%%PHYSICS%%": physics,
        "%%NOISE%%": noise,
        "%%DATASET_PARAMS%%": dataset_params,
        "%%TABLE_ROWS%%": table_rows,
    }
    for key, value in replacements.items():
        content = content.replace(key, value)

    os.makedirs(output_dir, exist_ok=True)
    rst_path = os.path.join(output_dir, f"{benchmark_link}.rst")
    with open(rst_path, "w") as f:
        f.write(content)

    return dataset, physics, noise, benchmark_link


def generate_physics_group_pages(benchmark_info, auto_benchmarks_dir):
    r"""
    Generate one intermediate RST page per physics group.

    Each page holds a toctree of the individual benchmark pages belonging to
    that physics, and lives under ``auto_benchmarks/``.

    :param list[tuple] benchmark_info: List of (benchmark_name, dataset, physics, noise) tuples.
    :param str, Path auto_benchmarks_dir: Directory where the physics pages are written.
    :return: List of (physics, physics_slug) tuples, in display order.
    :rtype: list[tuple[str, str]]
    """
    from collections import defaultdict

    physics_groups = defaultdict(list)
    for name, _, physics, _ in benchmark_info:
        physics_groups[physics].append(name)

    physics_pages = []  # list of (physics, physics_slug)
    for physics in sorted(physics_groups.keys()):
        physics_slug = physics.lower() + "_benchmarks"
        physics_pages.append((physics, physics_slug))

        content = f".. _{physics_slug}:\n\n{physics}\n{'=' * len(physics)}\n\n"
        content += ".. toctree::\n   :maxdepth: 1\n\n"
        for name in physics_groups[physics]:
            content += f"   {name}\n"

        physics_rst_path = os.path.join(auto_benchmarks_dir, f"{physics_slug}.rst")
        with open(physics_rst_path, "w") as f:
            f.write(content)

    return physics_pages


def generate_main_benchmark_rst(benchmark_info, output_dir):
    r"""
    Generate the main benchmarks landing page from the committed template.

    The static layout of the page (intro, contribution note, usage
    instructions, and the table/toctree scaffolding) lives in the committed
    ``source/benchmarks.rst`` template and can be edited without touching this
    module. This function only fills in the two dynamic parts -- the rows of the
    "List of benchmarks" table and the hidden toctree of per-physics group pages
    -- and writes the result to a build-only copy under ``auto_benchmarks/``
    (the committed template itself is excluded from the Sphinx build).
    Benchmarks are grouped by physics name, each group getting its own
    intermediate page under ``auto_benchmarks/``.

    :param list[tuple] benchmark_info: List of tuples containing (benchmark_name, dataset, physics, noise).
    :param str, Path output_dir: Source directory containing the ``auto_benchmarks/`` output folder.
    """
    auto_benchmarks_dir = os.path.join(output_dir, "auto_benchmarks")
    os.makedirs(auto_benchmarks_dir, exist_ok=True)

    # Per-physics intermediate pages, plus the ordered list used for the toctree.
    physics_pages = generate_physics_group_pages(benchmark_info, auto_benchmarks_dir)

    # Rows of the "List of benchmarks" table.
    benchmark_rows = "\n".join(
        f"    * - :ref:`{name}`\n"
        f"      - :sclass:`deepinv.datasets.{dataset}`\n"
        f"      - :sclass:`deepinv.physics.{physics}`\n"
        f"      - :sclass:`deepinv.physics.{noise}`"
        for name, dataset, physics, noise in sorted(benchmark_info)
    )

    # Hidden toctree entries, one per physics group page. Paths are absolute
    # (from the source root) since the filled page lives under auto_benchmarks/.
    physics_toctree = "\n".join(
        f"   /auto_benchmarks/{physics_slug}" for _, physics_slug in physics_pages
    )

    with open(MAIN_BENCHMARK_TEMPLATE) as f:
        benchmarks_content = f.read()
    benchmarks_content = benchmarks_content.replace(
        "%%BENCHMARK_ROWS%%", benchmark_rows
    ).replace("%%PHYSICS_TOCTREE%%", physics_toctree)

    benchmarks_rst_path = os.path.join(auto_benchmarks_dir, "benchmarks.rst")
    with open(benchmarks_rst_path, "w") as f:
        f.write(benchmarks_content)


def add_benchmark_to_class(app, what, name, obj, options, lines):
    r"""
    Event handler for the 'autodoc-process-docstring' Sphinx event.

    This function adds a section to the docstring of classes that lists
    the benchmarks in which they are used.

    :param Sphinx app: The Sphinx application object.
    :param str what: The type of the object being documented (e.g., "class").
    :param str name: The fully qualified name of the object being documented.
    :param object obj: The object being documented.
    :param dict options: The options given to the directive.
    :param list[str] lines: The lines of the docstring to be modified.
    """
    if what != "class":
        return

    mapping = benchmark_mapping
    if not mapping:
        return

    short = name.split(".")[-1]

    if short not in mapping:
        return

    lines.extend(
        [
            "",
            "|sep|",
            "",
            ":Used in benchmarks:",
            "",
        ]
    )
    for benchmark in mapping[short]:
        lines.append(f"- :ref:`{benchmark.replace('-', '_').replace(' ', '_')}`")

    lines.append("")


def setup(app: Sphinx):
    r"""
    Sphinx extension setup function.

    :param Sphinx app: The Sphinx application object.
    :return: A dictionary with the extension version.
    """
    app.connect("builder-inited", generate_benchmarks)
    app.connect("autodoc-process-docstring", add_benchmark_to_class)
    return {"version": "1.0"}
