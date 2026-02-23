"""Render notebooks to reStructuredText format using nbconvert."""

import glob
import random
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
        A tuple containing the notebook group (str), a verbose flag (bool), and
        a debug flag (bool).
    """
    # Check for verbose flags.
    verbose = False
    if "--verbose" in sys.argv or "-v" in sys.argv:
        verbose = True
        if "--verbose" in sys.argv:
            sys.argv.remove("--verbose")
        if "-v" in sys.argv:
            sys.argv.remove("-v")

    # Check for debug flags.
    debug = False
    if "--debug" in sys.argv or "-d" in sys.argv:
        debug = True
        if "--debug" in sys.argv:
            sys.argv.remove("--debug")
        if "-d" in sys.argv:
            sys.argv.remove("-d")

    # Get the notebook group.
    if len(sys.argv) < 3:
        raise ValueError(
            "Not enough arguments given. Usage: python render_notebooks.py "
            "<notebook_group> <super_group> [--verbose | -v] [--debug | -d]"
        )
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

    # get the super group
    super_group = sys.argv[2]

    return group_name, super_group, verbose, debug


def _resolve_repo_paths():
    """Resolve important repository paths relative to the repository root.

    Returns
    -------
    tuple[Path, Path]
        Tuple containing the repository root and the rail repository root.

    Raises
    ------
    FileNotFoundError
        If the rail repository cannot be located.
    """
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    # GitHub runners clone rail inside this repository; local development keeps it adjacent.
    candidates = [
        repo_root / "rail",
        repo_root.parent / "rail",
    ]

    for candidate in candidates:
        if (candidate / "examples").exists():
            return repo_root, candidate

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


def _render_notebook_group(
    raw_notebook_dir, rendered_group_dir, verbose=False, debug=False
):
    """Iterate through notebooks in a specified group directory and render them
    to reStructuredText format using nbconvert.

    Parameters
    ----------
    raw_notebook_dir : Path
        Path to the directory containing raw notebooks.
    rendered_group_dir : Path
        Path to the directory where rendered notebooks will be saved.
    verbose : bool, optional
        If True, enables verbose output.
    debug : bool, optional
        If True, enables debug mode (skips actual rendering).

    Returns
    -------
    dict
        A dictionary mapping notebook names to their rendering status codes."""
    status = {}

    if not rendered_group_dir.exists():
        rendered_group_dir.mkdir(parents=True, exist_ok=True)

    # Iterate through notebooks in the specified group directory of raw notebooks.
    for notebook in glob.glob(f"{raw_notebook_dir}/*.ipynb"):
        notebook_name = os.path.splitext(os.path.basename(notebook))[0]
        out_name = f"{notebook_name}.rst"

        if verbose:
            print(f"RENDERING: {notebook_name}")

        # Render the notebook using nbconvert.
        if debug:
            # Mock result and make empty output file for debugging purposes.
            class Result:
                def __init__(self):
                    self.returncode = random.random() > 0.8 and 1 or 0
                    self.stderr = (
                        "Mock render error message." if self.returncode else ""
                    )

            result = Result()
            if result.returncode == 0:
                with open(
                    rendered_group_dir / out_name, "w", encoding="utf-8"
                ) as handle:
                    handle.write("")
        else:
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

    return status


def _report_render_status(status, verbose=False):
    """Report the rendering status of notebooks and raise an error if any failed.

    Parameters
    ----------
    status : dict
        A dictionary mapping notebook names to their rendering status codes.
    verbose : bool, optional
        If True, enables verbose output.

    Raises
    ------
    ValueError
        If any notebook failed to render.
    """
    failed_notebooks = []

    if verbose:
        print("\nNOTEBOOK RENDERING STATUS:")
        print(f"{'R. Code':>7}  Notebook")
        print("-" * 7 + "  " + "-" * 60)

    for nb_name, nb_status in status.items():
        if verbose:
            print(f"{nb_status:>7}  {nb_name}")

        if nb_status != 0:
            failed_notebooks.append(nb_name)

    if failed_notebooks:  # Print regardless of verbosity
        print(f"\n{len(failed_notebooks)} FAILED NOTEBOOK(S):")
        for index, nb in enumerate(failed_notebooks, start=1):
            print(f"{index}. {nb}")
        print()

        raise ValueError(
            f"{len(failed_notebooks)} out of {len(status)} notebooks failed to render."
        )


def run_render_notebook_group():
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
    super_group: str
        The folder that this group of notebooks is found in. Should be one of
        'interactive_examples' or 'pipeline_examples'.
    --verbose, -v : bool, optional
        If specified, enables verbose output.

    Raises
    ------
    ValueError
        If an invalid notebook group is provided or if any notebook failed to render.
    """
    group_name, super_group, verbose, debug = _parse_args()
    if verbose:
        print(f"Rendering notebooks in group: {group_name}...")

    repo_root, rail_root = _resolve_repo_paths()
    raw_notebook_dir = rail_root / super_group / f"{group_name}_examples"
    rendered_group_dir = (
        repo_root / "docs" / super_group / "rendered" / f"{group_name}_examples"
    )

    _clear_rendered_output(rendered_group_dir, verbose)

    status = _render_notebook_group(
        raw_notebook_dir,
        rendered_group_dir,
        verbose,
        debug,
    )

    _report_render_status(status, verbose)


if __name__ == "__main__":
    run_render_notebook_group()
