import glob, os, sys

def render_notebook_group():
    status = {}

    group_name = sys.argv[1]
    if group_name not in ['core', 'creation', 'estimation', 'evaluation', 'goldenspike']:
        raise ValueError("Invalid notebook group given. Try 'core', 'creation', 'estimation', 'evaluation', or 'goldenspike'.")

    for nb_file in glob.glob(f'rail/examples/{group_name}_examples/*.ipynb'):
        
        subdir = os.path.dirname(nb_file).split('/')[-1]
        basename = os.path.splitext(os.path.basename(nb_file))[0]
        outfile = os.path.join('..', '..', 'docs', 'rendered', f"{subdir}/{basename}.rst")
        relpath = os.path.join('docs', 'rendered', subdir)

        try:
            print(relpath)
            os.makedirs(relpath)
        except FileExistsError:
            pass

        comline = f"jupyter nbconvert --to rst --output {outfile} --execute {nb_file}"
        render = os.system(comline)
        status[nb_file] = render

    failed_notebooks = []
    for nb_name, nb_status in status.items():
        print(f"{nb_name} {nb_status}")
        if nb_status != 0:
            failed_notebooks.append(nb_name)

    if failed_notebooks:
        raise ValueError(f"The following notebooks failed {str(failed_notebooks)}")

if __name__ == "__main__":
      render_notebook_group()
