import glob, os

def render_nb():
    status = {}

    #for nb_file in glob.glob("rail/examples/*_examples/*.ipynb"):
    #for nb_file in ['rail/examples/core_examples/FluxtoMag_and_Deredden_example.ipynb']:
    #for nb_file in ['docs/notebooks/core_examples/FluxtoMag_and_Deredden_example.ipynb']:
    for nb_file in ['docs/notebooks/core_examples/intro_notebook.ipynb', 'docs/notebooks/core_examples/FluxtoMag_and_Deredden_example.ipynb']:
        print(nb_file)
        subdir = os.path.dirname(nb_file).split('/')[-1]
        basename = os.path.splitext(os.path.basename(nb_file))[0]
        outfile = os.path.join('..', '..', 'rendered', f"{subdir}/{basename}.html")
        relpath = os.path.join('docs', 'rendered', subdir)
        print(f'\nsubdir: {subdir}\nbasename: {basename}\noutfile: {outfile}\nrelpath: {relpath}\n')

        try:
            print(relpath)
            os.makedirs(relpath)
        except FileExistsError:
            pass

        comline = f"jupyter nbconvert --to html --output {outfile} --execute {nb_file}"
        render = os.system(comline)
        status[nb_file] = render

    failed_notebooks = []
    for key, val in status.items():
        print(f"{key} {val}")
        if val != 0:
            failed_notebooks.append(key)

    if failed_notebooks:
        raise ValueError(f"The following notebooks failed {str(failed_notebooks)}")

if __name__ == "__main__":
      render_nb()
