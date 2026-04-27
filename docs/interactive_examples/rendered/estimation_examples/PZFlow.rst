PZFlow Informer and Estimator Demo

Author: Tianqing Zhang

**Note:** If you’re interested in running this in pipeline mode, see
`09_PZFlow.ipynb <https://github.com/LSSTDESC/rail/blob/main/pipeline_examples/estimation_examples/09_PZFlow.ipynb>`__
in the ``pipeline_examples/estimation_examples/`` folder.

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
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
    
    LEPHAREDIR is being set to the default cache directory:
    /home/runner/.cache/lephare/data
    More than 1Gb may be written there.
    LEPHAREWORK is being set to the default cache directory:
    /home/runner/.cache/lephare/work
    Default work cache is already linked. 
    This is linked to the run directory:
    /home/runner/.cache/lephare/runs/20260427T123143


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
      File "/tmp/ipykernel_7472/1106421819.py", line 5, in <module>
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


.. code:: ipython3

    trainFile = find_rail_file("examples_data/testdata/test_dc2_training_9816.hdf5")
    testFile = find_rail_file("examples_data/testdata/test_dc2_validation_9816.hdf5")
    training_data = tables_io.read(trainFile)
    test_data = tables_io.read(testFile)

.. code:: ipython3

    pzflow_dict = dict(hdf5_groupname="photometry")

.. code:: ipython3

    # epoch = 200 gives a reasonable converged loss
    pzflow_model = ri.estimation.algos.pzflow_nf.pz_flow_informer(
        training_data=training_data, num_training_epochs=30, **pzflow_dict
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, PZFlowInformer


.. parsed-literal::

    Training 50 epochs 
    Loss:


.. parsed-literal::

    (0) 37.3273


.. parsed-literal::

    (1) 9.1679


.. parsed-literal::

    (3) 4.7060


.. parsed-literal::

    (5) 3.0395


.. parsed-literal::

    (7) 1.8210


.. parsed-literal::

    (9) 1.2419


.. parsed-literal::

    (11) 0.7195


.. parsed-literal::

    (13) 0.4739


.. parsed-literal::

    (15) 0.3899


.. parsed-literal::

    (17) 0.1622


.. parsed-literal::

    (19) 0.0236


.. parsed-literal::

    (21) 0.5141


.. parsed-literal::

    (23) 0.1235


.. parsed-literal::

    (25) -0.2421


.. parsed-literal::

    (27) -0.3631


.. parsed-literal::

    (29) -0.4974


.. parsed-literal::

    (31) -0.4768


.. parsed-literal::

    (33) -0.7454


.. parsed-literal::

    (35) -0.3239


.. parsed-literal::

    (37) -0.6735


.. parsed-literal::

    (39) -0.5815


.. parsed-literal::

    (41) -0.7599


.. parsed-literal::

    (43) -0.8674


.. parsed-literal::

    (45) -0.9404


.. parsed-literal::

    (47) -0.7904


.. parsed-literal::

    (49) -0.7422


.. parsed-literal::

    (50) -0.9549


.. parsed-literal::

    Inserting handle into data store.  model: inprogress_model.pkl, PZFlowInformer


.. code:: ipython3

    pzflow_dict = dict(hdf5_groupname="photometry")
    
    pzflow_estimator = ri.estimation.algos.pzflow_nf.pz_flow_estimator(
        input_data=test_data, model=pzflow_model["model"], **pzflow_dict, chunk_size=20000
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, PZFlowEstimator
    Inserting handle into data store.  model: <pzflow.flow.Flow object at 0x7f2bb95300d0>, PZFlowEstimator
    Process 0 running estimator on chunk 0 - 20,449


.. parsed-literal::

    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/qp/parameterizations/interp/interp.py:187: UserWarning: The distributions at indices = [ 6919 17601] have an integral of 0.
      warnings.warn(
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/qp/parameterizations/interp/interp.py:207: RuntimeWarning: invalid value encountered in divide
      new_yvals = (self._yvals.T / self._ycumul[:, -1]).T
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/qp/parameterizations/interp/interp.py:208: RuntimeWarning: invalid value encountered in divide
      self._ycumul = (self._ycumul.T / self._ycumul[:, -1]).T
    /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/qp/parameterizations/interp/interp.py:140: RuntimeWarning: There are non-finite values in the yvals for the following distributions: (array([ 6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,  6919,
            6919,  6919,  6919,  6919, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601,
           17601, 17601, 17601, 17601, 17601, 17601, 17601, 17601]), array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
            13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
            26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
            39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
            52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
            65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
            78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
            91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
           104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
           117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
           130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
           143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
           156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
           169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
           182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
           195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
           208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
           221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
           234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
           247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,
           260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272,
           273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
           286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
           299, 300,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,
            11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
            24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,
            37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
            50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,
            63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,
            76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,
            89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101,
           102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
           115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
           128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
           141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
           154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
           167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
           180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
           193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205,
           206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218,
           219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
           232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244,
           245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257,
           258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270,
           271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283,
           284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296,
           297, 298, 299, 300]))
      warnings.warn(


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PZFlowEstimator


.. code:: ipython3

    mode = pzflow_estimator["output"].ancil["zmode"]
    truth = np.array(test_data["photometry"]["redshift"])

.. code:: ipython3

    # visualize the prediction.
    plt.figure(figsize=(8, 8))
    plt.scatter(truth, mode, s=0.5)
    plt.xlabel("True Redshift")
    plt.ylabel("Mode of Estimated Redshift")




.. parsed-literal::

    Text(0, 0.5, 'Mode of Estimated Redshift')




.. image:: PZFlow_files/PZFlow_7_1.png


