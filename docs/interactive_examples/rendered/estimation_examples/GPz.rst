GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** Feb 9, 2026

**Note:** If you’re interested in running this in pipeline mode, see
`06_GPz.ipynb <https://github.com/LSSTDESC/rail/blob/main/pipeline_examples/estimation_examples/06_GPz.ipynb>`__
in the ``pipeline_examples/estimation_examples/`` folder.

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import matplotlib.pyplot as plt
    import rail.interactive as ri
    import tables_io
    
    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
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
    /home/runner/.cache/lephare/runs/20260413T123647


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
      File "/tmp/ipykernel_9185/232514726.py", line 2, in <module>
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

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(
        n_basis=60,
        trainfrac=0.8,
        csl_method="normal",
        max_iter=150,
        hdf5_groupname="photometry",
    )

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    model = ri.estimation.algos.gpz.gpz_informer(
        training_data=training_data, **gpz_train_dict
    )["model"]


.. parsed-literal::

    Inserting handle into data store.  input: None, GPzInformer
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.4261243e-01	 3.2001171e-01	-3.3216690e-01	 3.2376043e-01	[-3.3832840e-01]	 4.9890208e-01


.. parsed-literal::

       2	-2.6712601e-01	 3.0744482e-01	-2.4089379e-01	 3.1188770e-01	[-2.5252085e-01]	 2.4643016e-01


.. parsed-literal::

       3	-2.2193101e-01	 2.8720421e-01	-1.7791671e-01	 2.9597357e-01	[-2.0747162e-01]	 3.2829809e-01


.. parsed-literal::

       4	-1.8018333e-01	 2.6901492e-01	-1.3039495e-01	 2.7358283e-01	[-1.5267505e-01]	 3.2415271e-01


.. parsed-literal::

       5	-1.3162356e-01	 2.5630340e-01	-1.0187740e-01	 2.6143017e-01	[-1.2451708e-01]	 2.1194482e-01


.. parsed-literal::

       6	-7.0173801e-02	 2.5256879e-01	-4.2238131e-02	 2.5763629e-01	[-5.9553187e-02]	 2.1641612e-01


.. parsed-literal::

       7	-5.2481986e-02	 2.4858930e-01	-2.8239783e-02	 2.5176682e-01	[-3.8856080e-02]	 2.2120047e-01


.. parsed-literal::

       8	-4.1335957e-02	 2.4681722e-01	-2.0060907e-02	 2.4972682e-01	[-3.0601461e-02]	 2.1633434e-01


.. parsed-literal::

       9	-2.5131040e-02	 2.4378865e-01	-7.1602274e-03	 2.4677423e-01	[-1.8460702e-02]	 2.0739532e-01


.. parsed-literal::

      10	-1.5562643e-02	 2.4192393e-01	 5.7204737e-04	 2.4449583e-01	[-1.1041482e-02]	 2.2894502e-01


.. parsed-literal::

      11	-1.1308168e-02	 2.4120995e-01	 4.2218173e-03	 2.4369464e-01	[-6.6425438e-03]	 2.1297407e-01


.. parsed-literal::

      12	-7.0283077e-03	 2.4043366e-01	 7.9289066e-03	 2.4268417e-01	[-1.3080722e-03]	 2.1689892e-01


.. parsed-literal::

      13	-2.6315309e-03	 2.3958217e-01	 1.2173532e-02	 2.4192712e-01	[ 3.1699942e-03]	 2.1234274e-01


.. parsed-literal::

      14	 7.4575590e-02	 2.2676421e-01	 9.4663620e-02	 2.3532699e-01	[ 6.5647108e-02]	 3.3468843e-01


.. parsed-literal::

      15	 8.8506728e-02	 2.2276615e-01	 1.1203368e-01	 2.2635117e-01	[ 1.0080920e-01]	 3.4084630e-01


.. parsed-literal::

      16	 1.9382422e-01	 2.1582745e-01	 2.2101712e-01	 2.2175166e-01	[ 2.0217004e-01]	 2.1333432e-01


.. parsed-literal::

      17	 2.7461537e-01	 2.0969113e-01	 3.0613186e-01	 2.1544084e-01	[ 2.7447440e-01]	 3.3229756e-01
      18	 3.0626518e-01	 2.0601026e-01	 3.4108350e-01	 2.1247376e-01	[ 3.0333066e-01]	 1.9507766e-01


.. parsed-literal::

      19	 3.6035631e-01	 2.0509530e-01	 3.9534061e-01	 2.1031210e-01	[ 3.5727753e-01]	 1.8009686e-01
      20	 4.2370913e-01	 2.1005123e-01	 4.5893865e-01	 2.1816683e-01	[ 4.1724299e-01]	 1.7876339e-01


.. parsed-literal::

      21	 4.8012951e-01	 2.0728844e-01	 5.1660422e-01	 2.1556118e-01	[ 4.6606673e-01]	 2.0652795e-01


.. parsed-literal::

      22	 5.5116508e-01	 1.9909648e-01	 5.9223345e-01	 2.0616778e-01	[ 4.9293250e-01]	 2.0909929e-01


.. parsed-literal::

      23	 6.1285016e-01	 1.9608875e-01	 6.5509850e-01	 2.0252515e-01	[ 5.5781434e-01]	 2.1218157e-01
      24	 6.6760326e-01	 1.9021442e-01	 7.0884643e-01	 1.9591134e-01	[ 6.1246153e-01]	 1.9285107e-01


.. parsed-literal::

      25	 7.1407492e-01	 1.8368803e-01	 7.5591090e-01	 1.8935829e-01	[ 6.4150244e-01]	 2.1851802e-01


.. parsed-literal::

      26	 7.4061792e-01	 1.8291168e-01	 7.8194493e-01	 1.8722312e-01	[ 6.8333686e-01]	 2.2090888e-01


.. parsed-literal::

      27	 7.7137746e-01	 1.8020984e-01	 8.1209474e-01	 1.8458793e-01	[ 7.0932647e-01]	 2.0871091e-01
      28	 7.9857140e-01	 1.8137120e-01	 8.4021199e-01	 1.8656607e-01	[ 7.2529876e-01]	 1.9840717e-01


.. parsed-literal::

      29	 8.3775353e-01	 1.8589803e-01	 8.8171625e-01	 1.9484562e-01	[ 7.4991220e-01]	 1.9894385e-01


.. parsed-literal::

      30	 8.6865967e-01	 1.8214045e-01	 9.1310571e-01	 1.8882082e-01	[ 7.8545258e-01]	 2.0108604e-01


.. parsed-literal::

      31	 8.9146337e-01	 1.8060500e-01	 9.3508850e-01	 1.8694057e-01	[ 8.1866276e-01]	 2.1459627e-01


.. parsed-literal::

      32	 9.1375121e-01	 1.7931574e-01	 9.5736042e-01	 1.8580042e-01	[ 8.4641025e-01]	 2.1354389e-01


.. parsed-literal::

      33	 9.3039632e-01	 1.7917787e-01	 9.7449167e-01	 1.8530196e-01	[ 8.6583855e-01]	 2.0854545e-01
      34	 9.5363340e-01	 1.7587334e-01	 9.9814698e-01	 1.8164133e-01	[ 8.9028756e-01]	 2.0025706e-01


.. parsed-literal::

      35	 9.7190841e-01	 1.7223365e-01	 1.0174025e+00	 1.7774847e-01	[ 9.1061286e-01]	 1.9208670e-01
      36	 9.9137832e-01	 1.6798913e-01	 1.0384124e+00	 1.7236981e-01	[ 9.3520774e-01]	 1.9932079e-01


.. parsed-literal::

      37	 1.0064517e+00	 1.6460975e-01	 1.0544903e+00	 1.6777677e-01	[ 9.5413056e-01]	 2.1398854e-01


.. parsed-literal::

      38	 1.0178807e+00	 1.6277312e-01	 1.0658784e+00	 1.6568732e-01	[ 9.7020546e-01]	 2.0904398e-01


.. parsed-literal::

      39	 1.0396578e+00	 1.5927766e-01	 1.0876478e+00	 1.6090160e-01	[ 1.0028294e+00]	 2.2429276e-01


.. parsed-literal::

      40	 1.0607527e+00	 1.5556387e-01	 1.1092694e+00	 1.5628528e-01	[ 1.0294999e+00]	 2.4677420e-01


.. parsed-literal::

      41	 1.0790758e+00	 1.5235018e-01	 1.1293717e+00	 1.5191177e-01	[ 1.0462402e+00]	 2.1609950e-01


.. parsed-literal::

      42	 1.0956173e+00	 1.4984832e-01	 1.1455959e+00	 1.4979770e-01	[ 1.0652888e+00]	 2.1611524e-01


.. parsed-literal::

      43	 1.1071204e+00	 1.4873244e-01	 1.1564875e+00	 1.4982101e-01	[ 1.0730976e+00]	 2.2163582e-01


.. parsed-literal::

      44	 1.1247010e+00	 1.4649527e-01	 1.1740717e+00	 1.4798238e-01	[ 1.0845566e+00]	 2.1785426e-01


.. parsed-literal::

      45	 1.1370727e+00	 1.4630656e-01	 1.1870563e+00	 1.4902057e-01	[ 1.0905524e+00]	 2.0869184e-01


.. parsed-literal::

      46	 1.1497325e+00	 1.4543754e-01	 1.1997042e+00	 1.4813468e-01	[ 1.1083997e+00]	 2.0619702e-01


.. parsed-literal::

      47	 1.1592322e+00	 1.4498527e-01	 1.2091090e+00	 1.4766115e-01	[ 1.1214280e+00]	 2.1166658e-01
      48	 1.1734359e+00	 1.4411324e-01	 1.2235487e+00	 1.4563041e-01	[ 1.1388009e+00]	 2.0017505e-01


.. parsed-literal::

      49	 1.1884716e+00	 1.4231185e-01	 1.2397440e+00	 1.4203918e-01	[ 1.1539136e+00]	 2.0698071e-01


.. parsed-literal::

      50	 1.2038448e+00	 1.4161699e-01	 1.2548177e+00	 1.3998284e-01	[ 1.1688310e+00]	 2.1425128e-01


.. parsed-literal::

      51	 1.2125232e+00	 1.4098460e-01	 1.2633035e+00	 1.3919460e-01	[ 1.1751669e+00]	 2.1147871e-01


.. parsed-literal::

      52	 1.2284069e+00	 1.3890588e-01	 1.2797723e+00	 1.3686642e-01	[ 1.1817017e+00]	 2.1361470e-01


.. parsed-literal::

      53	 1.2391992e+00	 1.3752658e-01	 1.2908205e+00	 1.3472582e-01	[ 1.1838171e+00]	 2.1562052e-01
      54	 1.2499336e+00	 1.3656089e-01	 1.3016625e+00	 1.3416061e-01	[ 1.1929965e+00]	 2.0307851e-01


.. parsed-literal::

      55	 1.2646293e+00	 1.3491153e-01	 1.3165678e+00	 1.3420327e-01	[ 1.2012430e+00]	 2.0597291e-01


.. parsed-literal::

      56	 1.2759262e+00	 1.3507473e-01	 1.3282942e+00	 1.3526105e-01	[ 1.2093648e+00]	 2.0687413e-01


.. parsed-literal::

      57	 1.2863959e+00	 1.3408220e-01	 1.3387346e+00	 1.3580343e-01	[ 1.2124020e+00]	 2.1083903e-01


.. parsed-literal::

      58	 1.2959844e+00	 1.3314607e-01	 1.3484623e+00	 1.3489451e-01	[ 1.2203951e+00]	 2.0302558e-01


.. parsed-literal::

      59	 1.3083138e+00	 1.3211727e-01	 1.3610253e+00	 1.3386748e-01	[ 1.2353211e+00]	 2.1390700e-01


.. parsed-literal::

      60	 1.3148890e+00	 1.3153261e-01	 1.3676999e+00	 1.3357037e-01	[ 1.2438284e+00]	 2.1784735e-01


.. parsed-literal::

      61	 1.3233106e+00	 1.3127466e-01	 1.3759471e+00	 1.3339912e-01	[ 1.2526386e+00]	 2.1595764e-01


.. parsed-literal::

      62	 1.3330461e+00	 1.3132530e-01	 1.3858748e+00	 1.3431326e-01	[ 1.2589487e+00]	 2.0678639e-01


.. parsed-literal::

      63	 1.3391604e+00	 1.3128868e-01	 1.3921185e+00	 1.3507771e-01	[ 1.2617331e+00]	 2.1130300e-01


.. parsed-literal::

      64	 1.3460967e+00	 1.3204975e-01	 1.3995910e+00	 1.3747154e-01	  1.2576254e+00 	 2.1788096e-01


.. parsed-literal::

      65	 1.3540002e+00	 1.3107736e-01	 1.4074068e+00	 1.3657470e-01	  1.2615886e+00 	 2.1651864e-01


.. parsed-literal::

      66	 1.3610086e+00	 1.3050433e-01	 1.4143652e+00	 1.3595942e-01	[ 1.2632387e+00]	 2.1542525e-01


.. parsed-literal::

      67	 1.3715110e+00	 1.3058973e-01	 1.4251280e+00	 1.3633073e-01	  1.2594320e+00 	 2.0869136e-01


.. parsed-literal::

      68	 1.3734682e+00	 1.3228121e-01	 1.4273693e+00	 1.3804343e-01	  1.2505902e+00 	 2.0723844e-01
      69	 1.3823586e+00	 1.3174357e-01	 1.4360694e+00	 1.3703244e-01	  1.2603775e+00 	 1.9819522e-01


.. parsed-literal::

      70	 1.3856414e+00	 1.3163763e-01	 1.4393962e+00	 1.3646397e-01	[ 1.2636398e+00]	 2.0507431e-01


.. parsed-literal::

      71	 1.3906501e+00	 1.3208353e-01	 1.4447255e+00	 1.3583191e-01	[ 1.2677759e+00]	 2.1042681e-01


.. parsed-literal::

      72	 1.3939325e+00	 1.3231136e-01	 1.4485050e+00	 1.3472133e-01	  1.2609338e+00 	 2.0606828e-01


.. parsed-literal::

      73	 1.3983373e+00	 1.3241454e-01	 1.4528114e+00	 1.3472097e-01	[ 1.2677949e+00]	 2.0796800e-01
      74	 1.4026477e+00	 1.3264404e-01	 1.4571343e+00	 1.3482955e-01	[ 1.2719170e+00]	 1.9357324e-01


.. parsed-literal::

      75	 1.4064363e+00	 1.3262242e-01	 1.4608552e+00	 1.3471442e-01	[ 1.2754905e+00]	 2.0672059e-01


.. parsed-literal::

      76	 1.4134669e+00	 1.3275105e-01	 1.4680174e+00	 1.3410667e-01	  1.2743332e+00 	 2.1148896e-01


.. parsed-literal::

      77	 1.4150634e+00	 1.3310755e-01	 1.4696695e+00	 1.3392552e-01	  1.2690860e+00 	 2.1725941e-01


.. parsed-literal::

      78	 1.4210552e+00	 1.3238936e-01	 1.4754693e+00	 1.3338063e-01	[ 1.2767683e+00]	 2.2345972e-01


.. parsed-literal::

      79	 1.4237574e+00	 1.3223155e-01	 1.4782666e+00	 1.3294173e-01	[ 1.2774413e+00]	 2.1570015e-01


.. parsed-literal::

      80	 1.4282230e+00	 1.3204476e-01	 1.4829162e+00	 1.3221198e-01	[ 1.2779652e+00]	 2.2046900e-01


.. parsed-literal::

      81	 1.4348466e+00	 1.3203944e-01	 1.4896994e+00	 1.3144041e-01	[ 1.2807493e+00]	 2.1355700e-01


.. parsed-literal::

      82	 1.4381566e+00	 1.3240148e-01	 1.4934087e+00	 1.3088695e-01	  1.2636296e+00 	 2.0613647e-01


.. parsed-literal::

      83	 1.4467560e+00	 1.3218588e-01	 1.5017195e+00	 1.3090181e-01	  1.2786166e+00 	 2.0975184e-01


.. parsed-literal::

      84	 1.4494045e+00	 1.3196952e-01	 1.5041990e+00	 1.3119330e-01	[ 1.2820524e+00]	 2.0783448e-01


.. parsed-literal::

      85	 1.4537301e+00	 1.3180471e-01	 1.5085888e+00	 1.3172378e-01	  1.2803070e+00 	 2.1442056e-01


.. parsed-literal::

      86	 1.4556116e+00	 1.3150541e-01	 1.5107827e+00	 1.3224541e-01	  1.2655882e+00 	 2.2174764e-01
      87	 1.4598837e+00	 1.3137570e-01	 1.5149442e+00	 1.3202966e-01	  1.2712840e+00 	 1.8287063e-01


.. parsed-literal::

      88	 1.4618313e+00	 1.3126225e-01	 1.5169423e+00	 1.3188908e-01	  1.2718268e+00 	 2.0956421e-01


.. parsed-literal::

      89	 1.4653338e+00	 1.3092382e-01	 1.5205437e+00	 1.3177538e-01	  1.2733427e+00 	 2.2377968e-01


.. parsed-literal::

      90	 1.4703358e+00	 1.3037764e-01	 1.5257415e+00	 1.3152328e-01	  1.2738602e+00 	 2.0880485e-01


.. parsed-literal::

      91	 1.4734002e+00	 1.3004033e-01	 1.5289857e+00	 1.3149584e-01	  1.2780850e+00 	 3.4706950e-01


.. parsed-literal::

      92	 1.4766433e+00	 1.2984004e-01	 1.5322360e+00	 1.3149630e-01	  1.2790812e+00 	 2.1864295e-01


.. parsed-literal::

      93	 1.4796604e+00	 1.2966800e-01	 1.5353511e+00	 1.3166893e-01	  1.2783924e+00 	 2.0892286e-01


.. parsed-literal::

      94	 1.4822635e+00	 1.2990620e-01	 1.5380357e+00	 1.3221578e-01	  1.2766029e+00 	 2.2109365e-01


.. parsed-literal::

      95	 1.4849961e+00	 1.2980350e-01	 1.5407127e+00	 1.3187317e-01	  1.2787250e+00 	 2.1463299e-01


.. parsed-literal::

      96	 1.4873682e+00	 1.2971079e-01	 1.5430977e+00	 1.3196319e-01	  1.2802719e+00 	 2.1010494e-01


.. parsed-literal::

      97	 1.4896770e+00	 1.2966846e-01	 1.5453885e+00	 1.3222243e-01	  1.2811157e+00 	 2.1618652e-01


.. parsed-literal::

      98	 1.4927603e+00	 1.2963530e-01	 1.5484480e+00	 1.3313710e-01	[ 1.2822458e+00]	 2.1340632e-01


.. parsed-literal::

      99	 1.4954538e+00	 1.2953228e-01	 1.5510556e+00	 1.3352560e-01	  1.2812245e+00 	 2.0916033e-01


.. parsed-literal::

     100	 1.4971846e+00	 1.2943553e-01	 1.5527408e+00	 1.3343930e-01	  1.2819864e+00 	 2.0451713e-01


.. parsed-literal::

     101	 1.4998724e+00	 1.2930109e-01	 1.5554118e+00	 1.3365778e-01	[ 1.2827556e+00]	 2.1299028e-01


.. parsed-literal::

     102	 1.5020273e+00	 1.2921028e-01	 1.5576909e+00	 1.3406310e-01	  1.2809018e+00 	 2.0716596e-01
     103	 1.5045663e+00	 1.2922729e-01	 1.5602199e+00	 1.3424320e-01	[ 1.2827924e+00]	 2.0140553e-01


.. parsed-literal::

     104	 1.5063612e+00	 1.2929215e-01	 1.5620488e+00	 1.3458791e-01	[ 1.2841405e+00]	 2.0615721e-01


.. parsed-literal::

     105	 1.5077923e+00	 1.2927880e-01	 1.5635125e+00	 1.3455780e-01	[ 1.2852653e+00]	 2.1157598e-01


.. parsed-literal::

     106	 1.5107019e+00	 1.2922587e-01	 1.5665061e+00	 1.3431578e-01	  1.2841186e+00 	 2.1380711e-01


.. parsed-literal::

     107	 1.5120838e+00	 1.2893152e-01	 1.5679416e+00	 1.3365197e-01	  1.2798454e+00 	 3.2297587e-01


.. parsed-literal::

     108	 1.5138566e+00	 1.2879378e-01	 1.5697271e+00	 1.3321836e-01	  1.2777808e+00 	 2.0999169e-01
     109	 1.5158653e+00	 1.2858404e-01	 1.5717552e+00	 1.3271759e-01	  1.2740078e+00 	 2.0356679e-01


.. parsed-literal::

     110	 1.5179385e+00	 1.2838493e-01	 1.5738525e+00	 1.3234169e-01	  1.2710758e+00 	 2.1733379e-01
     111	 1.5207966e+00	 1.2816411e-01	 1.5767551e+00	 1.3202887e-01	  1.2677209e+00 	 2.0389414e-01


.. parsed-literal::

     112	 1.5219845e+00	 1.2792118e-01	 1.5780356e+00	 1.3204950e-01	  1.2652421e+00 	 2.1435475e-01


.. parsed-literal::

     113	 1.5239320e+00	 1.2799461e-01	 1.5798693e+00	 1.3218573e-01	  1.2711942e+00 	 2.1989036e-01


.. parsed-literal::

     114	 1.5249019e+00	 1.2804370e-01	 1.5808206e+00	 1.3233814e-01	  1.2734540e+00 	 2.1599150e-01
     115	 1.5267605e+00	 1.2811125e-01	 1.5826782e+00	 1.3258051e-01	  1.2751933e+00 	 1.9434881e-01


.. parsed-literal::

     116	 1.5290046e+00	 1.2811059e-01	 1.5850289e+00	 1.3292549e-01	  1.2751402e+00 	 2.1440291e-01


.. parsed-literal::

     117	 1.5313668e+00	 1.2819417e-01	 1.5873703e+00	 1.3315370e-01	  1.2720332e+00 	 2.1720672e-01


.. parsed-literal::

     118	 1.5329251e+00	 1.2806443e-01	 1.5889213e+00	 1.3288558e-01	  1.2685129e+00 	 2.0605874e-01


.. parsed-literal::

     119	 1.5344626e+00	 1.2793609e-01	 1.5904918e+00	 1.3288171e-01	  1.2632802e+00 	 2.1387744e-01


.. parsed-literal::

     120	 1.5356433e+00	 1.2781542e-01	 1.5917181e+00	 1.3267387e-01	  1.2562990e+00 	 2.1653700e-01


.. parsed-literal::

     121	 1.5371811e+00	 1.2782270e-01	 1.5932572e+00	 1.3282728e-01	  1.2562999e+00 	 2.1074414e-01


.. parsed-literal::

     122	 1.5393384e+00	 1.2787953e-01	 1.5954763e+00	 1.3340318e-01	  1.2548054e+00 	 2.0662284e-01


.. parsed-literal::

     123	 1.5404273e+00	 1.2783326e-01	 1.5965784e+00	 1.3346380e-01	  1.2549707e+00 	 2.0428753e-01


.. parsed-literal::

     124	 1.5417825e+00	 1.2769488e-01	 1.5979537e+00	 1.3345897e-01	  1.2546823e+00 	 2.1529436e-01
     125	 1.5442917e+00	 1.2734637e-01	 1.6005300e+00	 1.3330026e-01	  1.2564844e+00 	 1.9656515e-01


.. parsed-literal::

     126	 1.5451153e+00	 1.2694483e-01	 1.6014611e+00	 1.3307171e-01	  1.2502627e+00 	 2.1891737e-01


.. parsed-literal::

     127	 1.5470958e+00	 1.2699150e-01	 1.6033473e+00	 1.3289384e-01	  1.2552160e+00 	 2.0635104e-01


.. parsed-literal::

     128	 1.5479189e+00	 1.2698760e-01	 1.6041703e+00	 1.3290956e-01	  1.2551346e+00 	 2.0698333e-01


.. parsed-literal::

     129	 1.5493551e+00	 1.2693113e-01	 1.6056533e+00	 1.3283493e-01	  1.2533188e+00 	 2.1512723e-01


.. parsed-literal::

     130	 1.5514331e+00	 1.2684700e-01	 1.6078114e+00	 1.3282312e-01	  1.2514995e+00 	 2.0732784e-01
     131	 1.5532402e+00	 1.2676482e-01	 1.6096785e+00	 1.3272125e-01	  1.2494155e+00 	 1.9321585e-01


.. parsed-literal::

     132	 1.5542067e+00	 1.2670317e-01	 1.6107071e+00	 1.3279773e-01	  1.2507545e+00 	 2.0582724e-01
     133	 1.5555382e+00	 1.2664329e-01	 1.6119813e+00	 1.3253036e-01	  1.2517648e+00 	 2.0087171e-01


.. parsed-literal::

     134	 1.5562949e+00	 1.2661791e-01	 1.6127087e+00	 1.3241224e-01	  1.2534402e+00 	 2.0703959e-01
     135	 1.5577915e+00	 1.2655050e-01	 1.6141955e+00	 1.3219250e-01	  1.2543625e+00 	 1.7080021e-01


.. parsed-literal::

     136	 1.5595973e+00	 1.2654301e-01	 1.6160359e+00	 1.3219841e-01	  1.2501640e+00 	 1.7447400e-01
     137	 1.5605144e+00	 1.2652028e-01	 1.6171363e+00	 1.3194917e-01	  1.2411392e+00 	 2.0036173e-01


.. parsed-literal::

     138	 1.5620790e+00	 1.2651907e-01	 1.6186255e+00	 1.3216755e-01	  1.2420035e+00 	 1.9837642e-01
     139	 1.5626848e+00	 1.2651654e-01	 1.6192418e+00	 1.3225989e-01	  1.2415656e+00 	 2.0082521e-01


.. parsed-literal::

     140	 1.5638926e+00	 1.2646799e-01	 1.6204843e+00	 1.3229641e-01	  1.2408187e+00 	 2.0211601e-01
     141	 1.5648643e+00	 1.2637751e-01	 1.6215794e+00	 1.3226606e-01	  1.2404111e+00 	 1.9975638e-01


.. parsed-literal::

     142	 1.5667889e+00	 1.2626976e-01	 1.6234655e+00	 1.3215634e-01	  1.2416414e+00 	 2.0251346e-01
     143	 1.5676967e+00	 1.2618310e-01	 1.6243478e+00	 1.3199100e-01	  1.2427371e+00 	 1.9964719e-01


.. parsed-literal::

     144	 1.5686998e+00	 1.2605603e-01	 1.6253332e+00	 1.3182398e-01	  1.2435988e+00 	 2.0883226e-01


.. parsed-literal::

     145	 1.5702445e+00	 1.2588784e-01	 1.6268533e+00	 1.3172701e-01	  1.2432274e+00 	 2.1255827e-01


.. parsed-literal::

     146	 1.5709700e+00	 1.2580714e-01	 1.6275947e+00	 1.3159944e-01	  1.2399299e+00 	 3.2353067e-01


.. parsed-literal::

     147	 1.5722425e+00	 1.2573065e-01	 1.6288617e+00	 1.3170123e-01	  1.2382258e+00 	 2.1064806e-01


.. parsed-literal::

     148	 1.5731949e+00	 1.2571806e-01	 1.6298322e+00	 1.3177131e-01	  1.2365922e+00 	 2.1343851e-01


.. parsed-literal::

     149	 1.5746064e+00	 1.2570990e-01	 1.6313004e+00	 1.3171873e-01	  1.2348073e+00 	 2.0357060e-01


.. parsed-literal::

     150	 1.5751114e+00	 1.2566542e-01	 1.6319393e+00	 1.3144670e-01	  1.2326437e+00 	 2.0975375e-01
    Inserting handle into data store.  model: inprogress_model.pkl, GPzInformer


This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model=model)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    results = ri.estimation.algos.gpz.gpz_estimator(input_data=test_data, **gpz_test_dict)


.. parsed-literal::

    Inserting handle into data store.  input: None, GPzEstimator
    Inserting handle into data store.  model: <rail.estimation.algos._gpz_util.GP object at 0x7f24ecc287c0>, GPzEstimator
    Process 0 running estimator on chunk 0 - 20,449
    Process 0 estimating GPz PZ PDF for rows 0 - 20,449


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, GPzEstimator


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results["output"]

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12, 10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0, 3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: GPz_files/GPz_13_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data["photometry"]["redshift"]
    zmode = ens.ancil["zmode"].flatten()

.. code:: ipython3

    plt.figure(figsize=(12, 12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0, 3], [0, 3], "k--")
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: GPz_files/GPz_16_1.png

