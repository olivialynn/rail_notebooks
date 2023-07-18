def render_nb():
    command = "jupyter nbconvert"
    options = "--to html"

    status = {}

    for nb_file in inputs:
        subdir = os.path.dirname(nb_file).split('/')[-1]
        basename = os.path.splitext(os.path.basename(nb_file))[0]
        outfile = os.path.join('rendered', f"{subdir}/{basename}.html")
        relpath = os.path.join('rendered, f'{subdir}')

        try:
            print(relpath)
            os.makedirs(relpath)
        except FileExistsError:
            pass

        comline = f"{command} {options} --output {outfile} --execute {nb_file}"
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
