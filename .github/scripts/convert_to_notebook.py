from sphinx_gallery.py_source_parser import split_code_and_text_blocks
from sphinx_gallery.notebook import jupyter_notebook, save_notebook
from sphinx_gallery import gen_gallery
from pathlib import Path
import copy


def convert_script_to_notebook(src_file: Path, output_file: Path, gallery_conf):
    """
    Convert a single Python script to a Jupyter notebook and save it under target_root,
    preserving relative path.
    """
    # Parse the Python file
    file_conf, blocks = split_code_and_text_blocks(str(src_file))

    # Convert to notebook
    example_nb = jupyter_notebook(blocks, gallery_conf, str(src_file.parent))

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
