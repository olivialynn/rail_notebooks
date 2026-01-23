"""Parses the output files in logs/ and updates README.md accordingly.

Note that this is pretty dependent on the exact format of the output logs, and
there being a specific marker in README.md to indicate where to insert the
updated tables.

If you feel inspired and have the cycles, please feel free to improve this
script! It is fairly brittle at the moment.
"""

from pathlib import Path
from datetime import date


def _resolve_repo_paths():
    """Resolve and return important repository paths.

    Returns
    -------
    dict
        A dictionary containing resolved paths.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    logs_path = repo_root / "logs"
    readme_path = repo_root / "README.md"

    return {
        "repo_root": repo_root,
        "logs_path": logs_path,
        "readme_path": readme_path,
    }

def _parse_output_files(output_files):
    """Parse output files to extract relevant data for tables.

    Parameters
    ----------
    output_files : list[Path]
        A list of paths to the output files.

    Returns
    -------
    dict
        A dictionary containing parsed data for table generation.
    """
    parsed_data = {}

    for output_file in output_files:
        with output_file.open("r") as f:
            raw_content = f.read()

            raw_table = raw_content.split("NOTEBOOK RENDERING STATUS:")[1]
            if "FAILED NOTEBOOK(S):" in raw_table:
                # The format is "X FAILED NOTEBOOK(S):" , where X is a number.
                # So, we want to cut that whole section off.
                raw_table = raw_table.split("FAILED NOTEBOOK(S):")[0]
                raw_table = raw_table.rsplit("\n", 1)[0]  # Remove the last line which has just a number

            table_contents = []
            for line in raw_table.splitlines(): 
                # Skip header lines.
                if line.strip() == "":
                    continue
                if line.strip().startswith("R. Code"):
                    continue
                if line.strip().startswith("----"):
                    continue

                # Expect a line like:
                #  1  /Users/.../rail/examples/estimation_examples/03_BPZ_lite_Custom_SEDs.ipynb
                return_code, notebook_path = line.strip().split("  ", 1)
                notebook_name = Path(notebook_path).name

                # Append to table contents.
                table_contents.append((notebook_name, return_code))

            table_name = output_file.stem.split("_")[0]  # e.g., "core" from "core_logs.out"
            parsed_data[table_name] = sorted(table_contents, key=lambda x: x[0])
            print(f"\nPARSED DATA for TABLE {table_name}:\n", "\n".join(str(item) for item in parsed_data[table_name]), "\n")
    return parsed_data

def _write_new_tables(parsed_tables):
    """Generate markdown tables from output files.

    Parameters
    ----------
    parsed_tables : dict
        A dictionary containing parsed data from output files.

    Returns
    -------
    str
        The generated markdown tables as a string.
    """
    new_tables = ""
    for expected_name in ["core", "creation", "estimation", "evaluation", "goldenspike"]:
        new_tables += f"### {expected_name.capitalize()} Notebooks\n\n"
        if expected_name not in parsed_tables:
            new_tables += "*Error when parsing output files for this section*\n\n"
        else:
            table_contents = parsed_tables[expected_name]

            new_tables += "|   | Notebook |\n"
            new_tables += "|---|----------|\n"
            for notebook_name, return_code in table_contents:
                icon = ":white_check_mark:" if return_code == "0" else ":x:"
                new_tables += f"| {icon} | {notebook_name} |\n"
            new_tables += "\n"
            
    return new_tables

def update_readme():
    """Update the README.md file based on the output files in logs/."""
    paths = _resolve_repo_paths()
    logs_path = paths["logs_path"]
    readme_path = paths["readme_path"] 
    output_files = list(logs_path.glob("*.out"))

    with readme_path.open("r") as f:
        readme_contents = f.readlines()
    
    new_contents = []
    found_marker = False
    for line in readme_contents:
        if line.strip() == "<!--auto update below-->":
            found_marker = True

            parsed_tables = _parse_output_files(output_files)
            new_tables_as_markdown = _write_new_tables(parsed_tables)

            new_contents.append(line)
            new_contents.append("\n")
            
            last_updated_date_string = date.today().strftime("%B %d, %Y")
            new_contents.append(f"**Tables last updated:** {last_updated_date_string}\n\n")
            
            new_contents.append(new_tables_as_markdown)
            new_contents.append("\n")
            break
        else:
            new_contents.append(line)

    if not found_marker:
        raise RuntimeError("Could not find marker in README.md to insert updated tables.")

    with readme_path.open("w") as f:
        f.writelines(new_contents)
    

if __name__ == "__main__":
    update_readme()
