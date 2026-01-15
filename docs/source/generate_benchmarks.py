import os
from sphinx.application import Sphinx
from huggingface_hub import snapshot_download
from pathlib import Path

# global variable storing benchmark mappings to be used in class templates
benchmark_mapping = {}


def generate_benchmarks(app):
    r"""
    Generate RST files for benchmarks from parquet results files.

    This function downloads benchmarks from the Hugging Face repository https://huggingface.co/datasets/deepinv/benchmarks
    and generates one rst file for each results.parquet file found in the repository.
    It additionally generates a main benchmarks.rst file listing all benchmarks.

    :param Sphinx app: The Sphinx application object.
    """
    # Define the root directory of the benchmarks repository
    benchmarks_root = os.path.join(os.path.dirname(__file__), "deepinv-benchmarks")

    # Load the dataset from Hugging Face and save it to disk
    snapshot_download(
        repo_id="deepinv/benchmarks", repo_type="dataset", local_dir=benchmarks_root
    )

    # Recursively find all results.parquet files
    parquet_files = []
    for root, dirs, files in os.walk(benchmarks_root):
        for file in files:
            if file == "results.parquet":
                parquet_files.append((root, os.path.join(root, file)))

    if not parquet_files:
        raise FileNotFoundError(
            "No results.parquet files found in the benchmarks repository."
        )

    # Define the output directory for RST files
    source_dir = os.path.dirname(__file__)
    output_dir = os.path.join(source_dir, "auto_benchmarks")

    benchmark_info = []
    # process each parquet file
    for folder, parquet_file in parquet_files:
        dataset, physics, noise, benchmark_name = generate_rst_from_parquet(
            parquet_file, output_dir
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

    generate_main_rst(benchmark_info, source_dir)


def process_parquet_file(parquet_path):
    r"""
    Process a parquet benchmark results file and generate RST lines.

    :param str, Path parquet_path: Path to the results.parquet file.
    :return: Tuple containing the RST lines, dataset name, physics name, and noise model name.
    :rtype: tuple[list[str], str, str, str]
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)

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

    lines = [
        ".. |plusminus| unicode:: U+00B1 .. plus-minus sign",
        f"""
.. _{benchmark_link}:

{benchmark_name}
{'=' * len(benchmark_name)}



.. dropdown:: Benchmark information 
    :chevron: down-up

    """,
    ]

    lines.append(f"    - *Dataset*: :sclass:`deepinv.datasets.{dataset}` \n")
    lines.append(f"    - *Physics*: :sclass:`deepinv.physics.{physics}` \n")
    lines.append(f"    - *Noise model*: :sclass:`deepinv.physics.{noise}` \n")

    # search for entries starting as "objective_" except for "objective_name"
    dataset_cols = [
        col
        for col in df.columns
        if col.startswith("p_dataset")
        and col != "p_dataset_physics"
        and col != "p_dataset_noise"
    ]

    for col in dataset_cols:
        value = df.at[0, col]
        col_name = col.replace("p_dataset_", "")
        if col_name != "debug":
            lines.append(f"    - *{col_name}*: {value} \n")

    lines.append("\n")

    # Table directive
    lines.append(".. list-table::")
    lines.append("   :class: sortable-table")
    lines.append("   :header-rows: 1")
    lines.append("")

    # Add link to 'model' column if exists
    if "solver_name" in df.columns and "file_solver" in df.columns:
        url = (
            "https://github.com/deepinv/benchmarks/tree/main/"
            + str(Path(parquet_path).parent.name)
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

    # Header row
    header_cells = df.columns.tolist()
    lines.append("   * - " + "\n     - ".join(map(str, header_cells)))

    # Data rows
    for _, row in df.iterrows():
        row_cells = []
        for val in row:
            if isinstance(val, float):
                row_cells.append(f"{val:.2f}")
            else:
                row_cells.append(str(val))
        lines.append("   * - " + "\n     - ".join(row_cells))

    return lines, dataset, physics, noise, benchmark_link


def generate_rst_from_parquet(parquet_path, output_dir):
    r"""
    Generate an .rst file from a parquet benchmark results file.

    :param str, Path parquet_path: Path to the results.parquet file.
    :param str, Path output_dir: Directory where the rst file will be saved.
    :return: Tuple containing dataset, physics, and noise model names.
    :rtype: tuple[str, str, str]
    """
    # Write rst
    lines, dataset, physics, noise, benchmark_link = process_parquet_file(parquet_path)
    os.makedirs(output_dir, exist_ok=True)
    rst_path = os.path.join(output_dir, f"{benchmark_link}.rst")
    with open(rst_path, "w") as f:
        f.write("\n".join(lines))

    return dataset, physics, noise, benchmark_link


def generate_main_rst(benchmark_info, output_dir):
    r"""
    Generate the main benchmarks.rst file listing all benchmarks.

    :param list[tuple] benchmark_info: List of tuples containing (benchmark_name, dataset, physics, noise).
    :param str, Path output_dir: Directory where the benchmarks.rst file will be saved.
    """
    benchmarks_rst_path = os.path.join(output_dir, "benchmarks.rst")
    benchmarks_content = """Benchmarks
=================

This section provides benchmark results for various datasets and physics models.


You can try your model on one of the benchmarks, making sure that it receives `(y, physics)` as input and outputs the reconstructed image as `x`,
and then running:

.. code-block:: python

    from deepinv.auto_benchmarks import run_benchmark
    model = dinv.models.RAM() # replace with your model
    results = run_benchmark("benchmark_name", model)


.. list-table::
    :class: sortable-table
    :header-rows: 1

    * - Benchmark
      - Dataset
      - Physics
      - Noise Model
"""
    for name, dataset, physics, noise in benchmark_info:
        benchmarks_content += f"    * - :ref:`{name}`\n      - :sclass:`deepinv.datasets.{dataset}`\n      - :sclass:`deepinv.physics.{physics}`\n      - :sclass:`deepinv.physics.{noise}`\n"

    benchmarks_content += "\n.. toctree::\n   :maxdepth: 2\n   :hidden:\n\n"
    for name, _, _, _ in benchmark_info:
        benchmarks_content += f"   auto_benchmarks/{name}\n"

    with open(benchmarks_rst_path, "w") as f:
        f.write(benchmarks_content)


def add_benchmark_section(app, what, name, obj, options, lines):
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
    app.connect("autodoc-process-docstring", add_benchmark_section)
    return {"version": "1.0"}
