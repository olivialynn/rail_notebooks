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
    /home/runner/.cache/lephare/runs/20260406T121811


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
      File "/tmp/ipykernel_6086/232514726.py", line 2, in <module>
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
       1	-3.5552454e-01	 3.2360977e-01	-3.4529203e-01	 3.1101591e-01	[-3.2280956e-01]	 4.4768429e-01


.. parsed-literal::

       2	-2.8289294e-01	 3.1292086e-01	-2.5809820e-01	 3.0112329e-01	[-2.2372932e-01]	 2.2541261e-01


.. parsed-literal::

       3	-2.3654405e-01	 2.9053657e-01	-1.9157745e-01	 2.8165759e-01	[-1.5690174e-01]	 3.1462526e-01
       4	-2.0379314e-01	 2.6777126e-01	-1.6203954e-01	 2.6361892e-01	[-1.3624084e-01]	 1.7080498e-01


.. parsed-literal::

       5	-1.0938112e-01	 2.5888504e-01	-7.4261106e-02	 2.5232315e-01	[-4.5863239e-02]	 1.9822216e-01
       6	-7.8359855e-02	 2.5385520e-01	-4.8174173e-02	 2.4872833e-01	[-2.9604630e-02]	 1.7997265e-01


.. parsed-literal::

       7	-5.8731301e-02	 2.5059393e-01	-3.5140672e-02	 2.4522386e-01	[-1.4453258e-02]	 2.0827675e-01


.. parsed-literal::

       8	-4.7766324e-02	 2.4870452e-01	-2.7333007e-02	 2.4314406e-01	[-5.7572529e-03]	 2.0528865e-01


.. parsed-literal::

       9	-3.4493177e-02	 2.4619195e-01	-1.6638894e-02	 2.4052201e-01	[ 5.5184460e-03]	 2.1202731e-01
      10	-2.2939907e-02	 2.4391618e-01	-6.6799571e-03	 2.3796209e-01	[ 1.6769351e-02]	 1.9788170e-01


.. parsed-literal::

      11	-2.0559154e-02	 2.4383006e-01	-5.8511760e-03	 2.3757102e-01	[ 2.0002060e-02]	 2.1915197e-01
      12	-1.6449203e-02	 2.4296433e-01	-1.9536029e-03	 2.3697724e-01	[ 2.1593944e-02]	 1.8084693e-01


.. parsed-literal::

      13	-1.3868991e-02	 2.4240931e-01	 6.0647374e-04	 2.3649284e-01	[ 2.4386189e-02]	 2.0417690e-01


.. parsed-literal::

      14	-1.0255848e-02	 2.4163886e-01	 4.5337118e-03	 2.3552718e-01	[ 2.9228991e-02]	 2.0363355e-01


.. parsed-literal::

      15	 1.3015319e-01	 2.2383241e-01	 1.5240186e-01	 2.1961243e-01	[ 1.6410638e-01]	 2.9704785e-01


.. parsed-literal::

      16	 1.5200348e-01	 2.2276700e-01	 1.7755486e-01	 2.2083521e-01	[ 1.7996182e-01]	 2.1373558e-01
      17	 2.8565974e-01	 2.1672355e-01	 3.1652205e-01	 2.1613167e-01	[ 3.1353510e-01]	 1.8162608e-01


.. parsed-literal::

      18	 3.4452714e-01	 2.1042419e-01	 3.8148549e-01	 2.1049896e-01	[ 3.6795914e-01]	 2.0968437e-01
      19	 3.9562952e-01	 2.0572487e-01	 4.3190205e-01	 2.0497953e-01	[ 4.2248506e-01]	 1.7896199e-01


.. parsed-literal::

      20	 4.2534674e-01	 2.0309636e-01	 4.6117482e-01	 2.0318944e-01	[ 4.5134477e-01]	 2.0337367e-01


.. parsed-literal::

      21	 5.1241094e-01	 1.9497142e-01	 5.4789542e-01	 1.9587637e-01	[ 5.3493828e-01]	 2.0105147e-01


.. parsed-literal::

      22	 6.2953363e-01	 1.9499606e-01	 6.6860590e-01	 1.9541515e-01	[ 6.4882229e-01]	 2.0572376e-01
      23	 6.6799425e-01	 1.9134731e-01	 7.1040185e-01	 1.9079475e-01	[ 6.8241840e-01]	 1.9613910e-01


.. parsed-literal::

      24	 7.1093084e-01	 1.8785890e-01	 7.5056801e-01	 1.8723794e-01	[ 7.3443649e-01]	 2.0209813e-01
      25	 7.3397378e-01	 1.8554152e-01	 7.7383923e-01	 1.8534863e-01	[ 7.5714447e-01]	 1.9926047e-01


.. parsed-literal::

      26	 7.7218317e-01	 1.8428895e-01	 8.1264341e-01	 1.8350091e-01	[ 7.9414661e-01]	 3.2468390e-01
      27	 8.0669065e-01	 1.8288726e-01	 8.4768750e-01	 1.8108557e-01	[ 8.2643522e-01]	 1.8797541e-01


.. parsed-literal::

      28	 8.3423681e-01	 1.8180012e-01	 8.7572544e-01	 1.7776887e-01	[ 8.5599652e-01]	 2.1505308e-01


.. parsed-literal::

      29	 8.6146024e-01	 1.7854824e-01	 9.0374981e-01	 1.7416559e-01	[ 8.8512236e-01]	 2.0376754e-01
      30	 8.8476184e-01	 1.7484532e-01	 9.2785542e-01	 1.7260325e-01	[ 9.0811701e-01]	 1.9146109e-01


.. parsed-literal::

      31	 9.0420992e-01	 1.7225923e-01	 9.4738694e-01	 1.7084613e-01	[ 9.2759089e-01]	 2.0193386e-01
      32	 9.2021958e-01	 1.7178640e-01	 9.6384391e-01	 1.7061385e-01	[ 9.4494184e-01]	 1.9438076e-01


.. parsed-literal::

      33	 9.4335311e-01	 1.6960152e-01	 9.8777641e-01	 1.6860752e-01	[ 9.7296299e-01]	 2.0449257e-01
      34	 9.6524356e-01	 1.6822635e-01	 1.0109449e+00	 1.6768345e-01	[ 9.9372367e-01]	 1.8070555e-01


.. parsed-literal::

      35	 9.8183359e-01	 1.6510148e-01	 1.0281828e+00	 1.6558931e-01	[ 1.0099057e+00]	 2.1421742e-01


.. parsed-literal::

      36	 9.9591031e-01	 1.6380569e-01	 1.0419745e+00	 1.6457336e-01	[ 1.0221053e+00]	 2.0820308e-01
      37	 1.0128753e+00	 1.6228755e-01	 1.0593257e+00	 1.6360337e-01	[ 1.0362292e+00]	 1.9578862e-01


.. parsed-literal::

      38	 1.0275600e+00	 1.6145963e-01	 1.0745357e+00	 1.6260487e-01	[ 1.0475382e+00]	 1.9824171e-01


.. parsed-literal::

      39	 1.0401943e+00	 1.6175547e-01	 1.0885599e+00	 1.6195222e-01	[ 1.0516768e+00]	 2.0339394e-01


.. parsed-literal::

      40	 1.0515229e+00	 1.6050803e-01	 1.0999804e+00	 1.6075014e-01	[ 1.0588398e+00]	 2.1033192e-01
      41	 1.0580124e+00	 1.5928605e-01	 1.1064378e+00	 1.5998991e-01	[ 1.0633621e+00]	 1.9444060e-01


.. parsed-literal::

      42	 1.0725878e+00	 1.5759010e-01	 1.1215238e+00	 1.5887083e-01	[ 1.0727739e+00]	 1.9633174e-01
      43	 1.0799972e+00	 1.5692980e-01	 1.1296064e+00	 1.5909942e-01	[ 1.0827379e+00]	 1.8901825e-01


.. parsed-literal::

      44	 1.0927898e+00	 1.5547473e-01	 1.1424968e+00	 1.5792227e-01	[ 1.0912958e+00]	 1.8353963e-01
      45	 1.1004351e+00	 1.5399830e-01	 1.1502057e+00	 1.5677041e-01	[ 1.0983664e+00]	 1.9836640e-01


.. parsed-literal::

      46	 1.1090507e+00	 1.5244856e-01	 1.1592382e+00	 1.5561421e-01	[ 1.1032796e+00]	 2.0883560e-01
      47	 1.1212159e+00	 1.5084379e-01	 1.1715893e+00	 1.5436581e-01	[ 1.1173931e+00]	 1.9563651e-01


.. parsed-literal::

      48	 1.1314272e+00	 1.4865819e-01	 1.1818957e+00	 1.5278089e-01	[ 1.1206856e+00]	 1.8816638e-01
      49	 1.1392794e+00	 1.4778591e-01	 1.1897626e+00	 1.5217074e-01	[ 1.1252450e+00]	 1.9918966e-01


.. parsed-literal::

      50	 1.1498403e+00	 1.4670971e-01	 1.2003931e+00	 1.5126196e-01	[ 1.1308161e+00]	 1.9641089e-01
      51	 1.1565032e+00	 1.4617484e-01	 1.2072170e+00	 1.5041572e-01	[ 1.1320181e+00]	 1.7855668e-01


.. parsed-literal::

      52	 1.1659677e+00	 1.4509195e-01	 1.2166013e+00	 1.4967598e-01	[ 1.1434674e+00]	 1.8362522e-01
      53	 1.1739481e+00	 1.4413626e-01	 1.2246875e+00	 1.4902230e-01	[ 1.1501716e+00]	 1.9843507e-01


.. parsed-literal::

      54	 1.1811223e+00	 1.4363316e-01	 1.2320923e+00	 1.4884806e-01	[ 1.1516530e+00]	 2.0477986e-01
      55	 1.1952035e+00	 1.4297044e-01	 1.2469066e+00	 1.4852730e-01	  1.1490439e+00 	 1.8117952e-01


.. parsed-literal::

      56	 1.2004730e+00	 1.4394868e-01	 1.2527816e+00	 1.4960480e-01	  1.1387107e+00 	 1.8003964e-01
      57	 1.2121553e+00	 1.4300837e-01	 1.2640547e+00	 1.4834240e-01	[ 1.1541503e+00]	 1.9679570e-01


.. parsed-literal::

      58	 1.2193177e+00	 1.4258546e-01	 1.2712965e+00	 1.4749023e-01	[ 1.1593045e+00]	 1.8691635e-01
      59	 1.2274188e+00	 1.4271530e-01	 1.2795336e+00	 1.4690263e-01	[ 1.1655801e+00]	 1.9842124e-01


.. parsed-literal::

      60	 1.2339716e+00	 1.4341871e-01	 1.2862269e+00	 1.4679409e-01	  1.1621890e+00 	 2.0501709e-01
      61	 1.2423388e+00	 1.4292040e-01	 1.2945583e+00	 1.4629548e-01	[ 1.1734646e+00]	 1.9719005e-01


.. parsed-literal::

      62	 1.2478962e+00	 1.4246876e-01	 1.3002132e+00	 1.4598881e-01	[ 1.1760246e+00]	 1.9610834e-01


.. parsed-literal::

      63	 1.2577739e+00	 1.4135492e-01	 1.3104599e+00	 1.4527234e-01	[ 1.1788281e+00]	 2.1244264e-01


.. parsed-literal::

      64	 1.2606742e+00	 1.4147066e-01	 1.3137803e+00	 1.4539895e-01	  1.1607610e+00 	 2.1543956e-01
      65	 1.2719866e+00	 1.4006585e-01	 1.3249555e+00	 1.4448768e-01	[ 1.1828971e+00]	 1.8417048e-01


.. parsed-literal::

      66	 1.2763801e+00	 1.3935094e-01	 1.3292661e+00	 1.4408073e-01	[ 1.1912210e+00]	 2.1473956e-01
      67	 1.2809910e+00	 1.3900301e-01	 1.3339540e+00	 1.4398378e-01	[ 1.1947088e+00]	 1.9722247e-01


.. parsed-literal::

      68	 1.2858422e+00	 1.3896768e-01	 1.3389975e+00	 1.4436117e-01	[ 1.2006344e+00]	 1.9755673e-01


.. parsed-literal::

      69	 1.2919609e+00	 1.3897813e-01	 1.3451742e+00	 1.4436685e-01	[ 1.2012656e+00]	 2.1044922e-01
      70	 1.2983253e+00	 1.3924146e-01	 1.3516750e+00	 1.4454849e-01	[ 1.2024455e+00]	 1.7820621e-01


.. parsed-literal::

      71	 1.3050860e+00	 1.3956116e-01	 1.3584693e+00	 1.4458345e-01	[ 1.2072888e+00]	 1.8497825e-01
      72	 1.3109444e+00	 1.4055927e-01	 1.3646160e+00	 1.4526594e-01	  1.1944173e+00 	 1.8341851e-01


.. parsed-literal::

      73	 1.3183335e+00	 1.3981437e-01	 1.3718901e+00	 1.4451426e-01	  1.2043574e+00 	 1.9881082e-01
      74	 1.3227023e+00	 1.3903490e-01	 1.3762490e+00	 1.4394222e-01	  1.2055003e+00 	 1.7915082e-01


.. parsed-literal::

      75	 1.3311229e+00	 1.3822572e-01	 1.3852518e+00	 1.4344927e-01	  1.1948846e+00 	 1.8313646e-01


.. parsed-literal::

      76	 1.3365543e+00	 1.3810798e-01	 1.3911061e+00	 1.4423898e-01	  1.1824689e+00 	 2.1556401e-01


.. parsed-literal::

      77	 1.3434112e+00	 1.3812790e-01	 1.3978778e+00	 1.4405662e-01	  1.1901699e+00 	 2.0439291e-01


.. parsed-literal::

      78	 1.3487793e+00	 1.3842575e-01	 1.4033302e+00	 1.4433563e-01	  1.1971762e+00 	 2.1291828e-01
      79	 1.3551385e+00	 1.3802382e-01	 1.4099422e+00	 1.4418888e-01	  1.1996203e+00 	 1.9195700e-01


.. parsed-literal::

      80	 1.3620499e+00	 1.3704153e-01	 1.4173725e+00	 1.4389080e-01	  1.1993994e+00 	 2.0274711e-01


.. parsed-literal::

      81	 1.3680920e+00	 1.3631427e-01	 1.4233551e+00	 1.4354593e-01	  1.2071875e+00 	 2.0432401e-01


.. parsed-literal::

      82	 1.3725500e+00	 1.3570706e-01	 1.4276211e+00	 1.4302169e-01	[ 1.2125431e+00]	 2.0241451e-01
      83	 1.3787671e+00	 1.3515126e-01	 1.4339617e+00	 1.4257155e-01	[ 1.2234209e+00]	 1.9654107e-01


.. parsed-literal::

      84	 1.3852192e+00	 1.3498557e-01	 1.4405328e+00	 1.4264013e-01	[ 1.2301881e+00]	 1.9656038e-01


.. parsed-literal::

      85	 1.3915537e+00	 1.3483940e-01	 1.4467255e+00	 1.4253112e-01	[ 1.2532996e+00]	 2.0365095e-01
      86	 1.3967612e+00	 1.3521942e-01	 1.4518675e+00	 1.4301708e-01	[ 1.2693098e+00]	 1.7105913e-01


.. parsed-literal::

      87	 1.4021722e+00	 1.3501144e-01	 1.4574771e+00	 1.4296317e-01	[ 1.2760112e+00]	 2.0253634e-01
      88	 1.4067635e+00	 1.3501056e-01	 1.4620979e+00	 1.4321703e-01	[ 1.2861244e+00]	 1.9078660e-01


.. parsed-literal::

      89	 1.4109550e+00	 1.3442349e-01	 1.4662837e+00	 1.4283016e-01	[ 1.2874700e+00]	 2.1069765e-01
      90	 1.4175579e+00	 1.3355015e-01	 1.4732112e+00	 1.4242580e-01	  1.2838702e+00 	 1.8288779e-01


.. parsed-literal::

      91	 1.4220767e+00	 1.3285609e-01	 1.4777528e+00	 1.4197122e-01	  1.2812175e+00 	 2.0666957e-01


.. parsed-literal::

      92	 1.4256265e+00	 1.3286012e-01	 1.4812261e+00	 1.4205715e-01	  1.2828137e+00 	 2.0167279e-01


.. parsed-literal::

      93	 1.4299686e+00	 1.3275741e-01	 1.4856212e+00	 1.4224280e-01	  1.2791829e+00 	 2.0575237e-01


.. parsed-literal::

      94	 1.4331474e+00	 1.3262568e-01	 1.4889203e+00	 1.4241241e-01	  1.2764202e+00 	 2.0939589e-01
      95	 1.4375479e+00	 1.3223722e-01	 1.4934246e+00	 1.4234979e-01	  1.2744767e+00 	 2.0495605e-01


.. parsed-literal::

      96	 1.4415696e+00	 1.3142954e-01	 1.4976205e+00	 1.4222531e-01	  1.2655684e+00 	 2.0290422e-01


.. parsed-literal::

      97	 1.4449164e+00	 1.3098905e-01	 1.5009849e+00	 1.4165057e-01	  1.2638787e+00 	 2.0660591e-01


.. parsed-literal::

      98	 1.4467795e+00	 1.3089244e-01	 1.5027480e+00	 1.4140971e-01	  1.2652684e+00 	 2.0869684e-01
      99	 1.4506226e+00	 1.3050431e-01	 1.5067801e+00	 1.4086596e-01	  1.2559088e+00 	 1.9802046e-01


.. parsed-literal::

     100	 1.4537086e+00	 1.3000223e-01	 1.5100060e+00	 1.4034125e-01	  1.2470118e+00 	 2.0259762e-01


.. parsed-literal::

     101	 1.4566110e+00	 1.2988881e-01	 1.5128959e+00	 1.4037888e-01	  1.2444361e+00 	 2.0425558e-01
     102	 1.4600488e+00	 1.2971428e-01	 1.5163323e+00	 1.4049219e-01	  1.2377490e+00 	 1.6568923e-01


.. parsed-literal::

     103	 1.4620671e+00	 1.2936021e-01	 1.5183658e+00	 1.3995865e-01	  1.2373711e+00 	 1.9660950e-01


.. parsed-literal::

     104	 1.4643311e+00	 1.2924535e-01	 1.5205613e+00	 1.3991144e-01	  1.2372263e+00 	 2.1013999e-01
     105	 1.4665551e+00	 1.2903826e-01	 1.5228057e+00	 1.3973920e-01	  1.2328123e+00 	 1.7804527e-01


.. parsed-literal::

     106	 1.4689042e+00	 1.2884012e-01	 1.5251675e+00	 1.3969102e-01	  1.2312940e+00 	 1.9755936e-01
     107	 1.4706828e+00	 1.2867222e-01	 1.5270116e+00	 1.3963434e-01	  1.2298149e+00 	 2.0332003e-01


.. parsed-literal::

     108	 1.4739732e+00	 1.2841167e-01	 1.5302368e+00	 1.3943377e-01	  1.2339854e+00 	 1.6066527e-01


.. parsed-literal::

     109	 1.4753405e+00	 1.2834681e-01	 1.5315663e+00	 1.3937583e-01	  1.2387055e+00 	 2.0236754e-01
     110	 1.4774643e+00	 1.2816008e-01	 1.5336583e+00	 1.3916981e-01	  1.2445282e+00 	 1.9621491e-01


.. parsed-literal::

     111	 1.4806488e+00	 1.2783690e-01	 1.5368607e+00	 1.3888731e-01	  1.2479142e+00 	 2.0886922e-01


.. parsed-literal::

     112	 1.4828109e+00	 1.2744707e-01	 1.5391483e+00	 1.3839206e-01	  1.2428917e+00 	 3.1217074e-01


.. parsed-literal::

     113	 1.4862927e+00	 1.2705569e-01	 1.5427279e+00	 1.3809057e-01	  1.2385857e+00 	 2.0660377e-01


.. parsed-literal::

     114	 1.4885857e+00	 1.2681445e-01	 1.5451361e+00	 1.3809896e-01	  1.2329906e+00 	 2.0160723e-01
     115	 1.4907118e+00	 1.2659800e-01	 1.5473869e+00	 1.3790762e-01	  1.2213066e+00 	 1.8628097e-01


.. parsed-literal::

     116	 1.4927527e+00	 1.2646826e-01	 1.5494862e+00	 1.3779885e-01	  1.2115894e+00 	 1.9242549e-01
     117	 1.4952334e+00	 1.2625282e-01	 1.5520671e+00	 1.3755033e-01	  1.2002024e+00 	 1.8700743e-01


.. parsed-literal::

     118	 1.4972527e+00	 1.2609896e-01	 1.5540946e+00	 1.3734266e-01	  1.1968925e+00 	 2.0810390e-01


.. parsed-literal::

     119	 1.4987128e+00	 1.2601536e-01	 1.5555185e+00	 1.3712746e-01	  1.1992789e+00 	 2.0930624e-01
     120	 1.5010628e+00	 1.2579140e-01	 1.5579204e+00	 1.3665982e-01	  1.1992308e+00 	 1.7956734e-01


.. parsed-literal::

     121	 1.5024321e+00	 1.2573289e-01	 1.5593596e+00	 1.3641460e-01	  1.1900743e+00 	 3.1159687e-01


.. parsed-literal::

     122	 1.5037667e+00	 1.2565200e-01	 1.5607208e+00	 1.3632975e-01	  1.1844150e+00 	 2.1647143e-01
     123	 1.5066074e+00	 1.2532286e-01	 1.5636836e+00	 1.3617224e-01	  1.1660208e+00 	 1.8296218e-01


.. parsed-literal::

     124	 1.5085346e+00	 1.2493749e-01	 1.5656574e+00	 1.3589316e-01	  1.1528795e+00 	 1.9795418e-01


.. parsed-literal::

     125	 1.5107011e+00	 1.2457151e-01	 1.5677987e+00	 1.3563628e-01	  1.1469232e+00 	 2.1376348e-01


.. parsed-literal::

     126	 1.5126525e+00	 1.2413630e-01	 1.5697800e+00	 1.3529334e-01	  1.1360780e+00 	 2.1680570e-01


.. parsed-literal::

     127	 1.5141897e+00	 1.2402146e-01	 1.5712648e+00	 1.3509845e-01	  1.1370043e+00 	 2.0652008e-01
     128	 1.5152033e+00	 1.2402330e-01	 1.5722863e+00	 1.3504222e-01	  1.1356206e+00 	 1.7737341e-01


.. parsed-literal::

     129	 1.5193402e+00	 1.2409179e-01	 1.5767080e+00	 1.3473628e-01	  1.0980380e+00 	 1.8939209e-01


.. parsed-literal::

     130	 1.5205156e+00	 1.2405243e-01	 1.5779233e+00	 1.3468488e-01	  1.0909508e+00 	 3.1377697e-01


.. parsed-literal::

     131	 1.5216874e+00	 1.2403703e-01	 1.5791374e+00	 1.3465863e-01	  1.0760202e+00 	 2.1171761e-01


.. parsed-literal::

     132	 1.5234808e+00	 1.2400360e-01	 1.5810079e+00	 1.3460042e-01	  1.0396322e+00 	 2.0926356e-01


.. parsed-literal::

     133	 1.5245525e+00	 1.2397579e-01	 1.5820514e+00	 1.3459838e-01	  1.0234435e+00 	 2.0660114e-01


.. parsed-literal::

     134	 1.5258725e+00	 1.2393381e-01	 1.5833406e+00	 1.3444901e-01	  1.0091337e+00 	 2.1576071e-01


.. parsed-literal::

     135	 1.5276877e+00	 1.2384644e-01	 1.5851051e+00	 1.3414609e-01	  9.8837303e-01 	 2.0935845e-01


.. parsed-literal::

     136	 1.5287510e+00	 1.2376434e-01	 1.5861755e+00	 1.3398482e-01	  9.7485398e-01 	 2.1928167e-01


.. parsed-literal::

     137	 1.5313788e+00	 1.2354022e-01	 1.5889030e+00	 1.3367180e-01	  9.1356491e-01 	 2.0904279e-01


.. parsed-literal::

     138	 1.5327149e+00	 1.2342191e-01	 1.5903477e+00	 1.3357361e-01	  8.6716974e-01 	 3.0095387e-01


.. parsed-literal::

     139	 1.5343534e+00	 1.2328818e-01	 1.5920285e+00	 1.3356504e-01	  8.3251013e-01 	 2.1707392e-01


.. parsed-literal::

     140	 1.5359189e+00	 1.2312594e-01	 1.5936657e+00	 1.3352203e-01	  7.8276283e-01 	 2.1058583e-01


.. parsed-literal::

     141	 1.5373649e+00	 1.2300806e-01	 1.5951718e+00	 1.3358602e-01	  7.3335984e-01 	 2.1494913e-01


.. parsed-literal::

     142	 1.5387714e+00	 1.2282692e-01	 1.5965772e+00	 1.3334649e-01	  6.8231094e-01 	 2.0411301e-01


.. parsed-literal::

     143	 1.5400899e+00	 1.2274911e-01	 1.5978415e+00	 1.3321394e-01	  6.6525058e-01 	 2.1355915e-01


.. parsed-literal::

     144	 1.5416827e+00	 1.2264269e-01	 1.5993910e+00	 1.3301494e-01	  6.4923652e-01 	 2.1568561e-01


.. parsed-literal::

     145	 1.5434210e+00	 1.2250140e-01	 1.6011458e+00	 1.3285884e-01	  6.1074388e-01 	 2.1018982e-01


.. parsed-literal::

     146	 1.5447150e+00	 1.2234991e-01	 1.6025029e+00	 1.3278038e-01	  5.6863206e-01 	 3.1242967e-01


.. parsed-literal::

     147	 1.5461466e+00	 1.2228440e-01	 1.6039996e+00	 1.3280514e-01	  5.2832425e-01 	 2.0719910e-01


.. parsed-literal::

     148	 1.5472832e+00	 1.2221250e-01	 1.6051962e+00	 1.3283282e-01	  4.9926403e-01 	 2.1483445e-01
     149	 1.5487018e+00	 1.2212872e-01	 1.6067333e+00	 1.3284440e-01	  4.5601919e-01 	 1.9858050e-01


.. parsed-literal::

     150	 1.5501409e+00	 1.2204929e-01	 1.6082403e+00	 1.3280098e-01	  4.1160700e-01 	 1.9923496e-01
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
    Inserting handle into data store.  model: <rail.estimation.algos._gpz_util.GP object at 0x7fdf1820b5b0>, GPzEstimator
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

