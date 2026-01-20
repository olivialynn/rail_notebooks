"""Render notebooks to reStructuredText format using nbconvert."""

import glob
import os
import subprocess
import shutil
import sys
from pathlib import Path


def _parse_args():
    """Parse command line arguments for rendering notebooks.

    Returns
    -------
    tuple
        A tuple containing the notebook group (str) and a verbose flag (bool).
    """
    if len(sys.argv) < 2:
        raise ValueError(
            "Not enough arguments given. "
            "Usage: python render_notebooks.py <notebook_group> [--verbose | -v]"
        )

    verbose = False
    if "--verbose" in sys.argv or "-v" in sys.argv:
        verbose = True
        if "--verbose" in sys.argv:
            sys.argv.remove("--verbose")
        if "-v" in sys.argv:
            sys.argv.remove("-v")

    group_name = sys.argv[1]
    if group_name not in [
        "core",
        "creation",
        "estimation",
        "evaluation",
        "goldenspike",
    ]:
        raise ValueError(
            "Invalid notebook group given. "
            "Try 'core', 'creation', 'estimation', 'evaluation', or 'goldenspike'."
        )

    return group_name, verbose


def _clear_rendered_output(output_dir, group_name, verbose=False):
    """Clear all files in the rendered output directory.

    Parameters
    ----------
    output_dir : Path
        The directory containing rendered notebook files.
    group_name : str
        The group name of the notebooks being rendered.
    verbose : bool, optional
        If True, prints a message after clearing the directory.
    """
    # We'll see eg, 04_CMNN.rst and also 04_CMNN_files/; clear both.
    rendered_notebooks = output_dir / f"{group_name}_examples"
    shutil.rmtree(rendered_notebooks, ignore_errors=False)

    if verbose:
        print(f"Cleared rendered output directory: {rendered_notebooks}")


def render_notebook_group():
    """Render all notebooks in a specified group (core, creation, estimation,
    evaluation, goldenspike) to reStructuredText format using nbconvert.

    Usage
    -----
    python render_notebooks.py <notebook_group> [--verbose | -v]

    Parameters
    ----------
    notebook_group : str
        The group of notebooks to render. Must be one of 'core', 'creation',
        'estimation', 'evaluation', or 'goldenspike'.
    --verbose, -v : bool, optional
        If specified, enables verbose output.

    Raises
    ------
    ValueError
        If an invalid notebook group is provided or if any notebook fails to render.
    """
    status = {}
    group_name, verbose = _parse_args()
    if verbose:
        print(f"Rendering notebooks in group: {group_name}...")

    # Set paths for raw and rendered notebooks.
    raw_notebook_dir = Path("..", "rail", "examples", f"{group_name}_examples")
    rendered_root_dir = Path("docs", "rendered")
    rendered_group_dir = rendered_root_dir / f"{group_name}_examples"

    # Clear the rendered output directory.
    _clear_rendered_output(rendered_root_dir, group_name, verbose)

    # Render each notebook in the specified group.
    if not rendered_group_dir.exists():
        rendered_group_dir.mkdir(parents=True, exist_ok=True)
    for notebook in glob.glob(f"{raw_notebook_dir}/*.ipynb")[:2]:
        notebook_name = os.path.splitext(os.path.basename(notebook))[0]
        out_name = f"{notebook_name}.rst"

        if verbose:
            print(
                f"\nRENDERING NOTEBOOK {notebook_name}:"
                f"\n\tnotebook file: {notebook}"
                f"\n\toutput file: {out_name}"
            )

        result = subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "rst",
                "--output",
                str(out_name),
                "--output-dir",
                str(rendered_group_dir),
                "--execute",
                str(notebook),
                "--log-level",
                "ERROR",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if verbose and result.returncode:
            tail = "\n".join(result.stderr.splitlines()[-10:])
            print(tail)

        status[notebook] = result.returncode

    failed_notebooks = []
    if verbose:
        print("NOTEBOOK RENDERING STATUS:")
        print(f"{'R. Code':>7}  Notebook")
        print("-" * 7 + "  " + "-" * 60)
    for nb_name, nb_status in status.items():
        if verbose:
            print(f"{nb_status:>7}  {nb_name}")
        if nb_status != 0:
            failed_notebooks.append(nb_name)

    if failed_notebooks:
        print("\nFAILED NOTEBOOKS:")
        output = "\n".join(f"- {nb}" for nb in failed_notebooks)
        raise ValueError(f"The following notebooks failed to render:\n{output}")


if __name__ == "__main__":
    render_notebook_group()
