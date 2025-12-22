import os
import pandas as pd
from sphinx.application import Sphinx
import subprocess

benchmark_mapping = {}


def generate_rst_from_parquet(parquet_path, output_dir, benchmark_name):
    df = pd.read_parquet(parquet_path)
    dataset = df.attrs.get("dataset", "Unknown")
    physics = df.attrs.get("physics", "Unknown")
    noise = df.attrs.get("noise", "Unknown")
    os.makedirs(output_dir, exist_ok=True)
    rst_path = os.path.join(output_dir, f"{benchmark_name}.rst")

    lines = [
        f"""
.. _{benchmark_name.replace('-', '_').replace(' ', '_')}:

{benchmark_name.replace('-', ' ').replace('_', ' ')}
{'=' * len(benchmark_name)}

Benchmark results for {benchmark_name.replace('-', ' ').replace('_', ' ')}.

**Benchmark information**:
"""
    ]

    df = pd.read_parquet(parquet_path)

    if df.attrs:
        for key, value in df.attrs.items():
            if key == "physics":
                value = f":sclass:`deepinv.physics.{value}`\n"
            elif key == "dataset":
                value = f":sclass:`deepinv.datasets.{value}`\n"
            elif key == "noise":
                value = f":sclass:`deepinv.physics.{value}`\n"

            lines.append(f"{key.title()}: {value}")
        lines.append("")

    # Table directive
    lines.append(".. list-table::")
    lines.append("   :class: sortable-table")
    lines.append("   :header-rows: 1")
    lines.append("")

    # Add link to 'model' column if exists
    if "model" in df.columns and "file" in df.columns:
        for i, row in df.iterrows():
            model_name = row["model"]
            file_link = row["file"]
            link_cell = f"`{model_name} <{file_link}>`_"
            df.at[i, "model"] = link_cell  # Update the DataFrame directly

        # remove 'file' column from display
        df = df.drop(columns=["file"])

    # Header row
    header_cells = df.columns.tolist()
    lines.append("   * - " + "\n     - ".join(map(str, header_cells)))

    # Data rows
    for _, row in df.iterrows():
        row_cells = []
        for val in row:
            if isinstance(val, float):
                row_cells.append(f"{val:.4g}")
            else:
                row_cells.append(str(val))
        lines.append("   * - " + "\n     - ".join(row_cells))

    # Write rst
    with open(rst_path, "w") as f:
        f.write("\n".join(lines))

    return benchmark_name, dataset, physics, noise


def generate_benchmarks_rst(benchmark_info, output_dir):
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
        benchmark_name = os.path.basename(folder)
        try:
            name, dataset, physics, noise = generate_rst_from_parquet(
                parquet_file, output_dir, benchmark_name
            )
            benchmark_info.append((name, dataset, physics, noise))

            # Add mapping for dataset and physics
            if dataset not in benchmark_mapping:
                benchmark_mapping[dataset] = [name]
            else:
                benchmark_mapping[dataset].append(name)
            if physics not in benchmark_mapping:
                benchmark_mapping[physics] = [name]
            else:
                benchmark_mapping[physics].append(name)
            if noise not in benchmark_mapping:
                benchmark_mapping[noise] = [name]
            else:
                benchmark_mapping[noise].append(name)

        except Exception as e:
            print(f"Failed to process {parquet_file}: {str(e)}")

    generate_benchmarks_rst(benchmark_info, source_dir)


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
