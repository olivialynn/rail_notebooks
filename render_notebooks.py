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


def _resolve_repo_paths():
    """Resolve important repository paths relative to the script location.

    Returns
    -------
    tuple[Path, Path]
        Tuple containing the render script root and rail repository root.

    Raises
    ------
    FileNotFoundError
        If the rail repository cannot be located.
    """
    script_root = Path(__file__).resolve().parent

    # GitHub runners clone rail inside this repository; local development keeps it adjacent.
    candidates = [
        script_root / "rail",
        script_root.parent / "rail",
    ]

    for candidate in candidates:
        if (candidate / "examples").exists():
            return script_root, candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not locate the rail repository. Checked: " + searched
    )


def _clear_rendered_output(rendered_dir, verbose=False):
    """Clear previously rendered outputs if the directory exists."""
    if rendered_dir.exists():
        shutil.rmtree(rendered_dir, ignore_errors=False)
        if verbose:
            print(f"Cleared rendered output directory: {rendered_dir}")


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

    script_root, rail_root = _resolve_repo_paths()

    # Set paths for raw and rendered notebooks.
    raw_notebook_dir = rail_root / "examples" / f"{group_name}_examples"
    rendered_group_dir = script_root / "docs" / "rendered" / f"{group_name}_examples"

    # Clear the rendered output directory.
    _clear_rendered_output(rendered_group_dir, verbose)

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
