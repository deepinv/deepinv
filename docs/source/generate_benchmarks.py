import os
from sphinx.application import Sphinx
import subprocess
import shutil

benchmark_mapping = {}


def process_parquet_file(parquet_path, benchmark_name):
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

    lines = [
        ".. |plusminus| unicode:: U+00B1 .. plus-minus sign",
        f"""
.. _{benchmark_name.replace('-', '_').replace(' ', '_')}:

{benchmark_name.replace('-', ' ').replace('_', ' ')}
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
    if "solver_name" in df.columns and "file" in df.columns:
        for i, row in df.iterrows():
            model_name = row["solver_name"]
            # file_link = row["file"]
            link_cell = f"`{model_name}"  # f"`{model_name} <{file_link}>`_"
            df.at[i, "solver_name"] = link_cell  # Update the DataFrame directly

    # extract metrics
    metric_cols = [
        col
        for col in df.columns
        if col.startswith("objective_") and col != "objective_name"
    ]
    std_cols = [col for col in metric_cols if col.endswith("_std")]
    mean_cols = [col for col in metric_cols if col not in std_cols]

    kept_cells = ["solver_name", "objective_runtime"] + mean_cols + std_cols

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

    return lines, dataset, physics, noise


def generate_rst_from_parquet(parquet_path, output_dir, benchmark_name):
    # Write rst
    lines, dataset, physics, noise = process_parquet_file(parquet_path, benchmark_name)
    os.makedirs(output_dir, exist_ok=True)
    rst_path = os.path.join(output_dir, f"{benchmark_name}.rst")
    with open(rst_path, "w") as f:
        f.write("\n".join(lines))

    return dataset, physics, noise


def generate_main_rst(benchmark_info, output_dir):
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
        benchmarks_content += f"    * - :ref:`{name.replace('-', '_').replace(' ', '_')}`\n      - :sclass:`deepinv.datasets.{dataset}`\n      - :sclass:`deepinv.physics.{physics}`\n      - :sclass:`deepinv.physics.{noise}`\n"

    benchmarks_content += "\n.. toctree::\n   :maxdepth: 2\n   :hidden:\n\n"
    for name, _, _, _ in benchmark_info:
        benchmarks_content += f"   auto_benchmarks/{name}\n"

    with open(benchmarks_rst_path, "w") as f:
        f.write(benchmarks_content)


def on_rm_error(func, path, exc_info):
    """
    Error handler for shutil.rmtree.
    On Windows, some files in .git are read-only. This changes
    permissions and retries the removal.
    """
    import stat

    os.chmod(path, stat.S_IWRITE)
    func(path)


def on_builder_inited(app):
    # Define the root directory of the benchmarks repository
    benchmarks_root = os.path.join(os.path.dirname(__file__), "deepinv-benchmarks")

    # Clone the repository if it doesn't exist
    if not os.path.exists(benchmarks_root):
        print("Cloning deepinv/benchmarks repository...")
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/deepinv/benchmarks",
                    benchmarks_root,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print("Repository cloned successfully.")

            git_dir = os.path.join(benchmarks_root, ".git")
            if os.path.exists(git_dir):
                print("Stripping Git metadata...")
                # Use the error handler to ensure Windows compatibility
                shutil.rmtree(git_dir, onerror=on_rm_error)
                print("Successfully converted to a standard directory.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e.stderr}")
            raise

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
    for folder, parquet_file in parquet_files:
        parent_folder_path = os.path.dirname(folder)
        benchmark_name = os.path.basename(parent_folder_path)
        try:
            dataset, physics, noise = generate_rst_from_parquet(
                parquet_file, output_dir, benchmark_name
            )
            benchmark_info.append((benchmark_name, dataset, physics, noise))

            # Add mapping for dataset and physics
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

        except Exception as e:
            print(f"Failed to process {parquet_file}: {str(e)}")

    generate_main_rst(benchmark_info, source_dir)


def add_benchmark_section(app, what, name, obj, options, lines):
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
    app.connect("builder-inited", on_builder_inited)
    app.connect("autodoc-process-docstring", add_benchmark_section)
    return {"version": "1.0"}
