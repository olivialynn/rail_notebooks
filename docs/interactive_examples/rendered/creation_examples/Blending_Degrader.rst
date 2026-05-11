Blending Degrader demo
----------------------

Author: Shuang Liang

Last run successfully: Feb 9, 2026

This notebook demonstrate the use of
``rail.creation.degradation.unrec_bl_model``, which uses Friends of
Friends to finds sources close to each other and merge them into
unrecognized blends

**Note:** If you’re interested in running this in pipeline mode, see
`06_Blending_Degrader.ipynb <https://github.com/LSSTDESC/rail/blob/main/pipeline_examples/creation_examples/06_Blending_Degrader.ipynb>`__
in the ``pipeline_examples/creation_examples/`` folder.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from rail.interactive.creation.degraders import unrec_bl_model


.. parsed-literal::

    Install FSPS with the following commands:
    pip uninstall fsps
    git clone --recursive https://github.com/dfm/python-fsps.git
    cd python-fsps
    python -m pip install .
    export SPS_HOME=$(pwd)/src/fsps/libfsps
    
    LEPHAREDIR is being set to the default cache directory which is being created at:
    /home/runner/.cache/lephare/data
    More than 1Gb may be written there.
    LEPHAREWORK is being set to the default cache directory:
    /home/runner/.cache/lephare/work


.. parsed-literal::

    
    A module that was compiled using NumPy 1.x cannot be run in
    NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
    versions of NumPy, modules must be compiled with NumPy 2.0.
    Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
    
    If you are a user of the module, the easiest solution will be to
    downgrade to 'numpy<2' or try to upgrade the affected module.
    We expect that some modules will need time to support NumPy 2.
    
    Traceback (most recent call last):  File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/runpy.py", line 196, in _run_module_as_main
        return _run_code(code, main_globals, None,
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/runpy.py", line 86, in _run_code
        exec(code, run_globals)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel_launcher.py", line 18, in <module>
        app.launch_new_instance()
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/traitlets/config/application.py", line 1082, in launch_instance
        app.start()
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/kernelapp.py", line 758, in start
        self.io_loop.start()
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/tornado/platform/asyncio.py", line 211, in start
        self.asyncio_loop.run_forever()
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/asyncio/base_events.py", line 603, in run_forever
        self._run_once()
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/asyncio/base_events.py", line 1909, in _run_once
        handle._run()
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/asyncio/events.py", line 80, in _run
        self._context.run(self._callback, *self._args)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/utils.py", line 71, in preserve_context
        return await f(*args, **kwargs)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/kernelbase.py", line 621, in shell_main
        await self.dispatch_shell(msg, subshell_id=subshell_id)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/kernelbase.py", line 478, in dispatch_shell
        await result
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/ipkernel.py", line 372, in execute_request
        await super().execute_request(stream, ident, parent)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/kernelbase.py", line 834, in execute_request
        reply_content = await reply_content
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/ipkernel.py", line 464, in do_execute
        res = shell.run_cell(
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/ipykernel/zmqshell.py", line 663, in run_cell
        return super().run_cell(*args, **kwargs)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3077, in run_cell
        result = self._run_cell(
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3132, in _run_cell
        result = runner(coro)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/IPython/core/async_helpers.py", line 128, in _pseudo_sync_runner
        coro.send(None)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3336, in run_cell_async
        has_raised = await self.run_ast_nodes(code_ast.body, cell_name,
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3519, in run_ast_nodes
        if await self.run_code(code, result, async_=asy):
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/IPython/core/interactiveshell.py", line 3579, in run_code
        exec(code_obj, self.user_global_ns, self.user_ns)
      File "/tmp/ipykernel_4060/561744479.py", line 4, in <module>
        from rail.interactive.creation.degraders import unrec_bl_model
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/interactive/__init__.py", line 3, in <module>
        from . import calib, creation, estimation, evaluation, tools
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/interactive/calib/__init__.py", line 3, in <module>
        from rail.utils.interactive.initialize_utils import _initialize_interactive_module
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/utils/interactive/initialize_utils.py", line 17, in <module>
        from rail.utils.interactive.base_utils import (
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/utils/interactive/base_utils.py", line 10, in <module>
        rail.stages.import_and_attach_all(silent=True)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/stages/__init__.py", line 74, in import_and_attach_all
        RailEnv.import_all_packages(silent=silent)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/core/introspection.py", line 541, in import_all_packages
        _imported_module = importlib.import_module(pkg)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/importlib/__init__.py", line 126, in import_module
        return _bootstrap._gcd_import(name[level:], package, level)
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/som/__init__.py", line 1, in <module>
        from rail.creation.degraders.specz_som import *
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/rail/creation/degraders/specz_som.py", line 15, in <module>
        from somoclu import Somoclu
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/somoclu/__init__.py", line 11, in <module>
        from .train import Somoclu
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/somoclu/train.py", line 25, in <module>
        from .somoclu_wrap import train as wrap_train
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/somoclu/somoclu_wrap.py", line 11, in <module>
        import _somoclu_wrap


::


    ---------------------------------------------------------------------------

    ImportError                               Traceback (most recent call last)

    File /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/numpy/core/_multiarray_umath.py:44, in __getattr__(attr_name)
         39     # Also print the message (with traceback).  This is because old versions
         40     # of NumPy unfortunately set up the import to replace (and hide) the
         41     # error.  The traceback shouldn't be needed, but e.g. pytest plugins
         42     # seem to swallow it and we should be failing anyway...
         43     sys.stderr.write(msg + tb_msg)
    ---> 44     raise ImportError(msg)
         46 ret = getattr(_multiarray_umath, attr_name, None)
         47 if ret is None:


    ImportError: 
    A module that was compiled using NumPy 1.x cannot be run in
    NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
    versions of NumPy, modules must be compiled with NumPy 2.0.
    Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
    
    If you are a user of the module, the easiest solution will be to
    downgrade to 'numpy<2' or try to upgrade the affected module.
    We expect that some modules will need time to support NumPy 2.
    



.. parsed-literal::

    Warning: the binary library cannot be imported. You cannot train maps, but you can load and analyze ones that you have already saved.
    The problem occurs because either compilation failed when you installed Somoclu or a path is missing from the dependencies when you are trying to import it. Please refer to the documentation to see your options.


Create a random catalog with ugrizy+YJHF bands as the the true input
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    data = np.random.normal(24, 3, size=(1000, 13))
    data[:, 0] = np.random.uniform(low=0, high=0.03, size=1000)
    data[:, 1] = np.random.uniform(low=0, high=0.03, size=1000)
    data[:, 2] = np.random.uniform(low=0, high=2, size=1000)
    
    data_truth = pd.DataFrame(
        data=data,  # values
        columns=["ra", "dec", "z_true", "u", "g", "r", "i", "z", "y", "Y", "J", "H", "F"],
    )

.. code:: ipython3

    plt.scatter(data_truth["ra"], data_truth["dec"], s=5)
    plt.xlabel("Ra [Deg]", fontsize=14)
    plt.ylabel("Dec [Deg]", fontsize=14)
    plt.show()



.. image:: Blending_Degrader_files/Blending_Degrader_4_0.png


The blending model
~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    ## model configuration; linking length is in arcsecs
    lsst_zp_dict = {"u": 12.65, "g": 14.69, "r": 14.56, "i": 14.38, "z": 13.99, "y": 13.02}
    
    outputs = unrec_bl_model.unrec_bl_model(
        sample=data_truth,
        ra_label="ra",
        dec_label="dec",
        linking_lengths=1.0,
        bands="ugrizy",
        zp_dict=lsst_zp_dict,
        ref_band="i",
        redshift_col="z_true",
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, UnrecBlModel


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.pq, UnrecBlModel
    Inserting handle into data store.  compInd: inprogress_compInd.pq, UnrecBlModel


.. code:: ipython3

    samples_w_bl = outputs["output"]
    component_ind = outputs["compInd"]

.. code:: ipython3

    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    
    ax.scatter(
        data_truth["ra"],
        data_truth["dec"],
        s=10,
        facecolors="none",
        edgecolors="b",
        label="Original",
    )
    ax.scatter(samples_w_bl["ra"], samples_w_bl["dec"], s=5, c="r", label="w. Unrec-BL")
    
    ax.legend(loc=2, fontsize=12)
    ax.set_xlabel("Ra [Deg]", fontsize=14)
    ax.set_ylabel("Dec [Deg]", fontsize=14)
    
    plt.show()



.. image:: Blending_Degrader_files/Blending_Degrader_8_0.png


.. code:: ipython3

    b = "i"
    plt.hist(data_truth[b], bins=np.linspace(10, 30, 20), label="Original")
    plt.hist(samples_w_bl[b], bins=np.linspace(10, 30, 20), fill=False, label="w. Unrec-BL")
    
    plt.xlabel(rf"Magnitude ${b}$", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()



.. image:: Blending_Degrader_files/Blending_Degrader_9_0.png


.. code:: ipython3

    plt.hist(data_truth["z_true"], bins=20, label="True Redshift")
    plt.hist(samples_w_bl["z_weighted"], bins=20, fill=False, label="Weighted Mean")
    
    plt.xlabel(rf"Rdshift", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()



.. image:: Blending_Degrader_files/Blending_Degrader_10_0.png


Study one BL case
~~~~~~~~~~~~~~~~~

.. code:: ipython3

    ## find a source with more than 1 truth component
    
    group_size = 1
    while group_size == 1:
    
        rand_ind = np.random.randint(len(samples_w_bl))
        this_bl = samples_w_bl.iloc[rand_ind]
        group_id = this_bl["group_id"]
    
        mask = component_ind["group_id"] == group_id
        FoF_group = component_ind[mask]
        group_size = len(FoF_group)
    
    truth_comp = data_truth.iloc[FoF_group.index]
    
    print("Truth RA / Blended RA:")
    print(truth_comp["ra"].to_numpy(), "/", this_bl["ra"])
    print("")
    
    print("Truth DEC / Blended DEC:")
    print(truth_comp["dec"].to_numpy(), "/", this_bl["dec"])
    print("")
    
    for b in "ugrizy":
        print(f"Truth mag {b} / Blended mag {b}:")
        print(truth_comp[b].to_numpy(), "/", this_bl[b])
        print("")


.. parsed-literal::

    Truth RA / Blended RA:
    [0.00934513 0.00931235 0.00916263] / 0.009273368265771914
    
    Truth DEC / Blended DEC:
    [0.02595876 0.02616115 0.02626161] / 0.02612717586397927
    
    Truth mag u / Blended mag u:
    [20.61388194 25.42148157 26.02928296] / 20.593702601237318
    
    Truth mag g / Blended mag g:
    [28.96476483 22.29577314 28.17494977] / 22.288631356223238
    
    Truth mag r / Blended mag r:
    [31.33449704 24.48519243 15.00327312] / 15.00309784355504
    
    Truth mag i / Blended mag i:
    [20.02697914 18.88520431 26.18788197] / 18.558909240699855
    
    Truth mag z / Blended mag z:
    [19.58125887 25.56301474 20.18332971] / 19.085721710751233
    
    Truth mag y / Blended mag y:
    [24.10997295 23.80208794 29.71743219] / 23.189916102542796
    


.. code:: ipython3

    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    
    ax.scatter(this_bl["ra"] * 3600, this_bl["dec"] * 3600, s=1e4, c="r")
    ax.scatter(
        truth_comp["ra"] * 3600,
        truth_comp["dec"] * 3600,
        s=1e4,
        facecolors="none",
        edgecolors="b",
    )
    
    ax.scatter([], [], s=1e2, facecolors="none", edgecolors="b", label="Truth Components")
    ax.scatter([], [], s=1e2, c="r", label="Merged Source")
    
    fig_size = 1  ## in arcsecs
    ax.set_xlim(this_bl["ra"] * 3600 - fig_size, this_bl["ra"] * 3600 + fig_size)
    ax.set_ylim(this_bl["dec"] * 3600 - fig_size, this_bl["dec"] * 3600 + fig_size)
    
    ax.legend(fontsize=12)
    ax.set_xlabel("Ra [arcsecs]", fontsize=14)
    ax.set_ylabel("Dec [arcsecs]", fontsize=14)
    
    plt.show()



.. image:: Blending_Degrader_files/Blending_Degrader_13_0.png

