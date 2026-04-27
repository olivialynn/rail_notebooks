FlexZBoost PDF Representation Comparison
========================================

**Author:** Drew Oldag

**Last Run Successfully:** Feb 9, 2026

This notebook does a quick comparison of storage requirements for
Flexcode output using two different storage techniques. We’ll compare
``qp.interp`` (x,y interpolated) output against the native
parameterization of ``qp_flexzboost``.

**Note:** If you’re interested in running this in pipeline mode, see
`01_FlexZBoost_PDF_Representation_Comparison.ipynb <https://github.com/LSSTDESC/rail/blob/main/pipeline_examples/estimation_examples/01_FlexZBoost_PDF_Representation_Comparison.ipynb>`__
in the ``pipeline_examples/estimation_examples/`` folder.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    import qp
    import rail.interactive as ri
    import tables_io
    from rail.utils.path_utils import find_rail_file


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
      File "/opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/traitlets/config/application.py", line 1075, in launch_instance
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
      File "/tmp/ipykernel_3996/2305495096.py", line 4, in <module>
        import rail.interactive as ri
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


Create references to the training and test data.

.. code:: ipython3

    trainFile = find_rail_file("examples_data/testdata/test_dc2_training_9816.hdf5")
    testFile = find_rail_file("examples_data/testdata/test_dc2_validation_9816.hdf5")
    training_data = tables_io.read(trainFile)
    test_data = tables_io.read(testFile)

Define the configurations for the ML model to be trained by Flexcode.
Specifically we’ll use Xgboost with a set of 35 cosine basis functions.

.. code:: ipython3

    fz_dict = dict(
        zmin=0.0,
        zmax=3.0,
        nzbins=301,
        trainfrac=0.75,
        bumpmin=0.02,
        bumpmax=0.35,
        nbump=20,
        sharpmin=0.7,
        sharpmax=2.1,
        nsharp=15,
        max_basis=35,
        basis_system="cosine",
        hdf5_groupname="photometry",
        regression_params={"max_depth": 8, "objective": "reg:squarederror"},
    )

.. code:: ipython3

    model = ri.estimation.algos.flexzboost.flex_z_boost_informer(
        training_data=training_data, **fz_dict
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, FlexZBoostInformer
    stacking some data...
    read in training data
    fit the model...


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:47] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:47] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:47] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:47] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:47] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:48] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:48] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:48] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:48] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:48] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:48] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:48] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:48] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:49] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:49] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:49] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:49] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:49] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:49] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:49] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:50] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:50] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:50] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:50] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:50] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:50] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:51] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:51] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:51] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:51] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:51] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:51] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:51] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:52] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:31:52] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    finding best bump thresh...


.. parsed-literal::

    finding best sharpen parameter...


.. parsed-literal::

    Retraining with full training set...


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:39] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:39] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:39] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:39] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:39] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:40] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:40] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:40] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:40] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:40] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:40] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:40] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:40] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:41] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:41] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:41] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:41] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:42] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:42] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:42] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:42] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:42] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:42] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:42] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:42] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:43] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:43] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:43] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:43] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:43] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:43] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:43] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:43] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:44] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/xgboost/training.py:200: UserWarning: [12:32:44] WARNING: /__w/xgboost/xgboost/src/learner.cc:782: 
    Parameters: { "silent" } are not used.
    
      bst.update(dtrain, iteration=i, fobj=obj)


.. parsed-literal::

    Best bump = 0.08947368421052632, best sharpen = 1.2
    Inserting handle into data store.  model: inprogress_model.pkl, FlexZBoostInformer


Now we configure the RAIL stage that will evaluate test data using the
saved model. Note that we specify ``qp_representation='flexzboost'``
here to instruct ``rail_flexzboost`` to store the model weights using
``qp_flexzboost``.

Now we actually evaluate the test data, 20,449 example galaxies, using
the trained model.

.. code:: ipython3

    %%time
    fzresults_qp_flexzboost = ri.estimation.algos.flexzboost.flex_z_boost_estimator(
        input_data=test_data, model=model["model"]
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, FlexZBoostEstimator
    Inserting handle into data store.  model: <flexcode.core.FlexCodeModel object at 0x7fcdfcd67a60>, FlexZBoostEstimator
    Process 0 running estimator on chunk 0 - 20,449
    Process 0 estimating PZ PDF for rows 0 - 20,449


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, FlexZBoostEstimator
    CPU times: user 11 s, sys: 80.1 ms, total: 11 s
    Wall time: 11.2 s


Example calculating median and mode. Note that we’re using the
``%%timeit`` magic command to get an estimate of the time required for
calculating ``median``, but we’re using ``%%time`` to estimate the
``mode``. This is because ``qp`` will cache the output of the ``pdf``
function for a given grid. If we used ``%%timeit``, then the resulting
estimate would average the run time of one non-cached calculation and
N-1 cached calculations.

.. code:: ipython3

    zgrid = np.linspace(0, 3.0, 301)

.. code:: ipython3

    %%time
    fz_medians_qp_flexzboost = fzresults_qp_flexzboost["output"].median()


.. parsed-literal::

    CPU times: user 865 ms, sys: 3.01 ms, total: 868 ms
    Wall time: 863 ms


.. code:: ipython3

    %%time
    fz_modes_qp_flexzboost = fzresults_qp_flexzboost["output"].mode(grid=zgrid)


.. parsed-literal::

    CPU times: user 208 ms, sys: 55.7 ms, total: 264 ms
    Wall time: 263 ms


Plotting median values.

.. code:: ipython3

    fz_medians_qp_flexzboost = fzresults_qp_flexzboost["output"].median()
    
    plt.hist(fz_medians_qp_flexzboost, bins=np.linspace(-0.005, 3.005, 101))
    plt.xlabel("redshift")
    plt.ylabel("Number")




.. parsed-literal::

    Text(0, 0.5, 'Number')




.. image:: FlexZBoost_PDF_Representation_Comparison_files/FlexZBoost_PDF_Representation_Comparison_15_1.png


Example convertion to a ``qp.hist`` histogram representation.

.. code:: ipython3

    %%timeit
    bins = np.linspace(0, 3, 301)
    fzresults_qp_flexzboost["output"].convert_to(qp.hist_gen, bins=bins)


.. parsed-literal::

    311 ms ± 5.61 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


Now we’ll repeat the experiment using ``qp.interp`` storage. Again,
we’ll define the RAIL stage to evaluate the test data using the saved
model, but instruct ``rail_flexzboost`` to store the output as x,y
interpolated values using ``qp.interp``.

Finally we evaluate the test data again using the trained model, and
then print out the size of the file that was saved using the x,y
interpolated technique.

.. code:: ipython3

    fzresults_qp_interp = ri.estimation.algos.flexzboost.flex_z_boost_estimator(
        input_data=test_data,
        model=model["model"],
        qp_representation="interp",
        calculated_point_estimates=[],
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, FlexZBoostEstimator
    Inserting handle into data store.  model: <flexcode.core.FlexCodeModel object at 0x7fcdfcd67a60>, FlexZBoostEstimator
    Process 0 running estimator on chunk 0 - 20,449
    Process 0 estimating PZ PDF for rows 0 - 20,449


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, FlexZBoostEstimator


Example calculating median and mode. Note that we’re using the
``%%timeit`` magic command to get an estimate of the time required for
calculating ``median``, but we’re using ``%%time`` to estimate the
``mode``. This is because ``qp`` will cache the output of the ``pdf``
function for a given grid. If we used ``%%timeit``, then the resulting
estimate would average the run time of one non-cached calculation and
N-1 cached calculations.

.. code:: ipython3

    zgrid = np.linspace(0, 3.0, 301)

.. code:: ipython3

    %%timeit
    fz_medians_qp_interp = fzresults_qp_interp["output"].median()


.. parsed-literal::

    850 ms ± 4.63 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


.. code:: ipython3

    %%time
    fz_modes_qp_interp = fzresults_qp_interp["output"].mode(grid=zgrid)


.. parsed-literal::

    CPU times: user 202 ms, sys: 60.4 ms, total: 263 ms
    Wall time: 262 ms


Plotting median values.

.. code:: ipython3

    fz_medians_qp_interp = fzresults_qp_interp["output"].median()
    plt.hist(fz_medians_qp_interp, bins=np.linspace(-0.005, 3.005, 101))
    plt.xlabel("redshift")
    plt.ylabel("Number")




.. parsed-literal::

    Text(0, 0.5, 'Number')




.. image:: FlexZBoost_PDF_Representation_Comparison_files/FlexZBoost_PDF_Representation_Comparison_26_1.png


Example convertion to a ``qp.hist`` histogram representation.

.. code:: ipython3

    %%timeit
    bins = np.linspace(0, 3, 301)
    fzresults_qp_interp["output"].convert_to(qp.hist_gen, bins=bins)


.. parsed-literal::

    311 ms ± 5.59 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

