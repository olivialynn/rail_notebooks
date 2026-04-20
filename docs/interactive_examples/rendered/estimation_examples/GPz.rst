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
    /home/runner/.cache/lephare/runs/20260420T122630


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
      File "/tmp/ipykernel_8456/232514726.py", line 2, in <module>
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
       1	-3.4224637e-01	 3.2020635e-01	-3.3205557e-01	 3.2186213e-01	[-3.3524853e-01]	 4.5768905e-01


.. parsed-literal::

       2	-2.7523742e-01	 3.1106950e-01	-2.5192296e-01	 3.1213662e-01	[-2.5561233e-01]	 2.2574997e-01


.. parsed-literal::

       3	-2.3140767e-01	 2.9061662e-01	-1.8711816e-01	 2.9266053e-01	[-1.9463760e-01]	 3.1215525e-01


.. parsed-literal::

       4	-1.9707573e-01	 2.7329511e-01	-1.4693209e-01	 2.7755163e-01	[-1.6710725e-01]	 2.9005313e-01


.. parsed-literal::

       5	-1.3138833e-01	 2.5680927e-01	-9.2500784e-02	 2.6186180e-01	[-1.2070245e-01]	 2.1266794e-01


.. parsed-literal::

       6	-6.6702640e-02	 2.5062335e-01	-3.5470179e-02	 2.5570589e-01	[-5.6861575e-02]	 2.0158625e-01
       7	-4.6021654e-02	 2.4733806e-01	-2.1340814e-02	 2.5254657e-01	[-4.2779953e-02]	 1.9720960e-01


.. parsed-literal::

       8	-3.0439824e-02	 2.4451187e-01	-1.0909772e-02	 2.5011745e-01	[-3.5450430e-02]	 1.9648004e-01
       9	-2.2296378e-02	 2.4311842e-01	-4.4175675e-03	 2.4971032e-01	[-3.0856291e-02]	 1.8373060e-01


.. parsed-literal::

      10	-1.1490971e-02	 2.4102280e-01	 4.6999052e-03	 2.4902745e-01	[-2.7108934e-02]	 1.9703531e-01


.. parsed-literal::

      11	-4.4847776e-03	 2.4000982e-01	 1.0075070e-02	 2.4860138e-01	[-2.3139793e-02]	 2.0244670e-01
      12	 1.1467647e-03	 2.3878379e-01	 1.5616524e-02	 2.4792810e-01	[-2.0259356e-02]	 1.9367886e-01


.. parsed-literal::

      13	 4.2516947e-03	 2.3815028e-01	 1.8752772e-02	 2.4744252e-01	[-1.8359052e-02]	 2.1212006e-01


.. parsed-literal::

      14	 6.1775610e-02	 2.2961718e-01	 8.0532482e-02	 2.3737062e-01	[ 4.7908875e-02]	 3.1385374e-01


.. parsed-literal::

      15	 8.9162084e-02	 2.2404360e-01	 1.1210771e-01	 2.3089785e-01	[ 9.8379671e-02]	 3.1328011e-01


.. parsed-literal::

      16	 1.8173636e-01	 2.1689256e-01	 2.0923374e-01	 2.2484833e-01	[ 1.8705845e-01]	 2.1971297e-01


.. parsed-literal::

      17	 2.5536059e-01	 2.1211342e-01	 2.8621275e-01	 2.2067950e-01	[ 2.4479113e-01]	 3.1723928e-01


.. parsed-literal::

      18	 2.9530573e-01	 2.0784889e-01	 3.2793981e-01	 2.1689448e-01	[ 2.7720911e-01]	 2.0988488e-01


.. parsed-literal::

      19	 3.2882817e-01	 2.0717284e-01	 3.6213795e-01	 2.1454702e-01	[ 3.1944814e-01]	 2.0745444e-01
      20	 3.5958181e-01	 2.0482569e-01	 3.9252596e-01	 2.1198307e-01	[ 3.5362904e-01]	 1.9481254e-01


.. parsed-literal::

      21	 4.0314951e-01	 2.0347791e-01	 4.3634907e-01	 2.1190409e-01	[ 3.9913084e-01]	 2.0851874e-01


.. parsed-literal::

      22	 4.8152472e-01	 2.0035405e-01	 5.1678485e-01	 2.0596312e-01	[ 4.6917645e-01]	 2.1655989e-01


.. parsed-literal::

      23	 5.4955580e-01	 1.9987075e-01	 5.8896742e-01	 2.0489865e-01	[ 5.3578177e-01]	 2.0993233e-01
      24	 5.9015707e-01	 1.9675171e-01	 6.3134887e-01	 2.0328288e-01	[ 5.7036905e-01]	 1.9339418e-01


.. parsed-literal::

      25	 6.2140185e-01	 1.9488595e-01	 6.6143213e-01	 2.0080818e-01	[ 6.1777735e-01]	 2.0792317e-01


.. parsed-literal::

      26	 6.5180170e-01	 1.9385341e-01	 6.9225526e-01	 1.9921409e-01	[ 6.4818477e-01]	 2.0664549e-01


.. parsed-literal::

      27	 7.0062289e-01	 1.9364230e-01	 7.3848987e-01	 2.0442452e-01	[ 6.9748287e-01]	 2.0325208e-01


.. parsed-literal::

      28	 7.2783720e-01	 1.9665290e-01	 7.6654451e-01	 2.0674170e-01	[ 7.1858125e-01]	 2.0322561e-01
      29	 7.5192730e-01	 1.8996501e-01	 7.9191028e-01	 2.0215972e-01	[ 7.5602878e-01]	 2.0369315e-01


.. parsed-literal::

      30	 7.6738045e-01	 1.8809326e-01	 8.0713544e-01	 2.0018474e-01	[ 7.7603298e-01]	 2.1383047e-01


.. parsed-literal::

      31	 7.9031476e-01	 1.8627319e-01	 8.3108028e-01	 1.9653626e-01	[ 8.0076078e-01]	 2.0720172e-01


.. parsed-literal::

      32	 8.1541760e-01	 1.8482982e-01	 8.5668802e-01	 1.9493097e-01	[ 8.2517507e-01]	 2.0126605e-01


.. parsed-literal::

      33	 8.3923572e-01	 1.8493550e-01	 8.8149581e-01	 1.9616997e-01	[ 8.4991592e-01]	 2.1818471e-01


.. parsed-literal::

      34	 8.6027524e-01	 1.8753367e-01	 9.0327794e-01	 2.0091356e-01	[ 8.6494234e-01]	 2.0338631e-01
      35	 8.7310624e-01	 1.8798487e-01	 9.1650716e-01	 2.0314309e-01	[ 8.8178386e-01]	 1.9757867e-01


.. parsed-literal::

      36	 8.8619694e-01	 1.8612092e-01	 9.2939507e-01	 2.0000257e-01	[ 8.9852707e-01]	 1.9799447e-01
      37	 9.0458404e-01	 1.8387018e-01	 9.4825222e-01	 1.9680188e-01	[ 9.1169188e-01]	 1.9877005e-01


.. parsed-literal::

      38	 9.2107502e-01	 1.8437457e-01	 9.6564417e-01	 1.9707680e-01	[ 9.2746358e-01]	 2.1069789e-01
      39	 9.3669117e-01	 1.8473968e-01	 9.8262622e-01	 1.9929970e-01	[ 9.4071411e-01]	 1.9173002e-01


.. parsed-literal::

      40	 9.5258561e-01	 1.8336332e-01	 9.9916113e-01	 1.9803466e-01	[ 9.5509501e-01]	 2.0173478e-01


.. parsed-literal::

      41	 9.6626492e-01	 1.8145996e-01	 1.0132213e+00	 1.9593407e-01	[ 9.7292068e-01]	 2.0968580e-01


.. parsed-literal::

      42	 9.8082584e-01	 1.7750214e-01	 1.0280438e+00	 1.9232625e-01	[ 9.8921339e-01]	 2.0409632e-01
      43	 9.9647298e-01	 1.7599844e-01	 1.0447207e+00	 1.9072433e-01	[ 1.0026092e+00]	 1.9603729e-01


.. parsed-literal::

      44	 1.0121714e+00	 1.7432267e-01	 1.0604154e+00	 1.8832168e-01	[ 1.0166337e+00]	 2.0890284e-01


.. parsed-literal::

      45	 1.0230210e+00	 1.7414271e-01	 1.0719199e+00	 1.8857327e-01	[ 1.0174451e+00]	 2.1583343e-01
      46	 1.0331625e+00	 1.7462314e-01	 1.0820177e+00	 1.8832540e-01	[ 1.0201437e+00]	 2.0451927e-01


.. parsed-literal::

      47	 1.0414411e+00	 1.7430000e-01	 1.0900318e+00	 1.8815910e-01	[ 1.0263194e+00]	 2.0152497e-01
      48	 1.0567164e+00	 1.7181714e-01	 1.1054983e+00	 1.8666762e-01	[ 1.0348102e+00]	 1.9502950e-01


.. parsed-literal::

      49	 1.0644109e+00	 1.7023127e-01	 1.1130734e+00	 1.8511675e-01	[ 1.0376633e+00]	 1.7791796e-01
      50	 1.0757246e+00	 1.6730426e-01	 1.1242602e+00	 1.8179796e-01	[ 1.0518141e+00]	 1.9449592e-01


.. parsed-literal::

      51	 1.0824817e+00	 1.6535783e-01	 1.1311716e+00	 1.7937013e-01	[ 1.0605403e+00]	 2.0213985e-01
      52	 1.0919965e+00	 1.6261341e-01	 1.1411667e+00	 1.7554895e-01	[ 1.0705055e+00]	 1.8387508e-01


.. parsed-literal::

      53	 1.1009903e+00	 1.5962368e-01	 1.1504882e+00	 1.7001004e-01	[ 1.0841738e+00]	 2.1228862e-01


.. parsed-literal::

      54	 1.1111536e+00	 1.5840956e-01	 1.1610300e+00	 1.6709155e-01	[ 1.0923670e+00]	 2.0880914e-01
      55	 1.1195726e+00	 1.5785492e-01	 1.1696754e+00	 1.6692429e-01	[ 1.0956986e+00]	 2.0217204e-01


.. parsed-literal::

      56	 1.1248919e+00	 1.5763924e-01	 1.1754875e+00	 1.6724520e-01	  1.0947724e+00 	 2.0299172e-01


.. parsed-literal::

      57	 1.1326852e+00	 1.5732662e-01	 1.1831826e+00	 1.6729226e-01	[ 1.1016024e+00]	 2.0593739e-01


.. parsed-literal::

      58	 1.1382919e+00	 1.5708034e-01	 1.1888373e+00	 1.6649426e-01	[ 1.1072426e+00]	 2.0763135e-01
      59	 1.1475297e+00	 1.5672459e-01	 1.1983482e+00	 1.6525253e-01	[ 1.1133431e+00]	 1.9803214e-01


.. parsed-literal::

      60	 1.1567630e+00	 1.5654116e-01	 1.2082163e+00	 1.6333876e-01	[ 1.1199537e+00]	 2.0421124e-01
      61	 1.1655451e+00	 1.5711348e-01	 1.2170751e+00	 1.6456047e-01	[ 1.1237093e+00]	 1.9970155e-01


.. parsed-literal::

      62	 1.1716796e+00	 1.5652516e-01	 1.2232822e+00	 1.6329600e-01	[ 1.1302152e+00]	 2.0357609e-01
      63	 1.1784698e+00	 1.5543709e-01	 1.2302486e+00	 1.6261213e-01	[ 1.1347591e+00]	 1.8003392e-01


.. parsed-literal::

      64	 1.1846197e+00	 1.5396263e-01	 1.2368855e+00	 1.6168241e-01	[ 1.1367591e+00]	 2.0944190e-01
      65	 1.1915368e+00	 1.5313184e-01	 1.2437902e+00	 1.6229168e-01	[ 1.1383746e+00]	 1.8988109e-01


.. parsed-literal::

      66	 1.2015182e+00	 1.5183154e-01	 1.2541414e+00	 1.6333124e-01	  1.1371706e+00 	 1.9550824e-01


.. parsed-literal::

      67	 1.2091775e+00	 1.5193084e-01	 1.2620232e+00	 1.6460466e-01	[ 1.1416962e+00]	 2.0861530e-01
      68	 1.2161082e+00	 1.5144002e-01	 1.2689574e+00	 1.6445547e-01	[ 1.1463760e+00]	 1.9964743e-01


.. parsed-literal::

      69	 1.2244147e+00	 1.5066467e-01	 1.2775121e+00	 1.6379159e-01	[ 1.1600118e+00]	 1.9291902e-01
      70	 1.2276175e+00	 1.4983633e-01	 1.2810142e+00	 1.6295289e-01	[ 1.1616903e+00]	 1.9084263e-01


.. parsed-literal::

      71	 1.2329396e+00	 1.4934783e-01	 1.2860608e+00	 1.6239615e-01	[ 1.1682068e+00]	 2.0434570e-01
      72	 1.2369526e+00	 1.4861950e-01	 1.2900985e+00	 1.6156728e-01	[ 1.1731312e+00]	 1.8716574e-01


.. parsed-literal::

      73	 1.2406604e+00	 1.4803969e-01	 1.2938910e+00	 1.6079579e-01	[ 1.1771091e+00]	 2.1393752e-01


.. parsed-literal::

      74	 1.2484130e+00	 1.4715558e-01	 1.3017392e+00	 1.5928291e-01	[ 1.1864671e+00]	 2.1356106e-01
      75	 1.2513468e+00	 1.4745018e-01	 1.3051270e+00	 1.5895406e-01	[ 1.1898316e+00]	 2.0538902e-01


.. parsed-literal::

      76	 1.2598148e+00	 1.4622346e-01	 1.3135456e+00	 1.5742489e-01	[ 1.2007045e+00]	 2.0603609e-01


.. parsed-literal::

      77	 1.2627942e+00	 1.4633941e-01	 1.3163462e+00	 1.5796051e-01	[ 1.2033054e+00]	 2.1037602e-01
      78	 1.2685058e+00	 1.4583627e-01	 1.3221962e+00	 1.5803440e-01	[ 1.2066999e+00]	 1.9054079e-01


.. parsed-literal::

      79	 1.2761946e+00	 1.4451052e-01	 1.3306086e+00	 1.5692938e-01	[ 1.2176379e+00]	 2.0141339e-01


.. parsed-literal::

      80	 1.2823152e+00	 1.4337549e-01	 1.3368920e+00	 1.5619198e-01	[ 1.2178018e+00]	 2.1643090e-01


.. parsed-literal::

      81	 1.2866144e+00	 1.4278046e-01	 1.3410897e+00	 1.5563851e-01	[ 1.2230191e+00]	 2.1039701e-01


.. parsed-literal::

      82	 1.2923888e+00	 1.4178399e-01	 1.3471354e+00	 1.5487433e-01	[ 1.2258885e+00]	 2.0701671e-01
      83	 1.2970117e+00	 1.4123694e-01	 1.3519472e+00	 1.5489547e-01	[ 1.2276180e+00]	 1.9979763e-01


.. parsed-literal::

      84	 1.3017979e+00	 1.4122674e-01	 1.3568182e+00	 1.5561346e-01	[ 1.2315049e+00]	 2.0455980e-01


.. parsed-literal::

      85	 1.3065554e+00	 1.4118033e-01	 1.3617936e+00	 1.5613119e-01	[ 1.2392518e+00]	 2.1449137e-01
      86	 1.3116925e+00	 1.4116139e-01	 1.3666633e+00	 1.5613495e-01	[ 1.2512722e+00]	 1.9366312e-01


.. parsed-literal::

      87	 1.3141049e+00	 1.4111728e-01	 1.3689464e+00	 1.5588156e-01	[ 1.2559728e+00]	 2.0470738e-01
      88	 1.3212420e+00	 1.4012090e-01	 1.3762825e+00	 1.5401393e-01	[ 1.2674469e+00]	 1.9469094e-01


.. parsed-literal::

      89	 1.3263243e+00	 1.4037918e-01	 1.3814012e+00	 1.5426934e-01	[ 1.2726285e+00]	 1.9852972e-01


.. parsed-literal::

      90	 1.3309448e+00	 1.3975082e-01	 1.3860868e+00	 1.5374732e-01	[ 1.2756330e+00]	 2.0317769e-01
      91	 1.3354909e+00	 1.3911738e-01	 1.3907861e+00	 1.5370054e-01	  1.2744350e+00 	 2.0344901e-01


.. parsed-literal::

      92	 1.3397996e+00	 1.3834686e-01	 1.3952546e+00	 1.5352625e-01	  1.2732831e+00 	 2.0209765e-01
      93	 1.3438205e+00	 1.3758360e-01	 1.3998895e+00	 1.5403881e-01	  1.2586199e+00 	 1.9666219e-01


.. parsed-literal::

      94	 1.3494459e+00	 1.3696118e-01	 1.4053865e+00	 1.5355062e-01	  1.2707007e+00 	 2.0041561e-01
      95	 1.3522163e+00	 1.3686864e-01	 1.4080800e+00	 1.5335234e-01	[ 1.2771797e+00]	 1.8320227e-01


.. parsed-literal::

      96	 1.3581399e+00	 1.3604141e-01	 1.4142751e+00	 1.5261585e-01	[ 1.2835720e+00]	 1.8813229e-01
      97	 1.3617792e+00	 1.3524486e-01	 1.4184028e+00	 1.5220890e-01	[ 1.2854008e+00]	 1.9880939e-01


.. parsed-literal::

      98	 1.3663023e+00	 1.3499421e-01	 1.4227269e+00	 1.5189883e-01	[ 1.2890905e+00]	 1.9592571e-01


.. parsed-literal::

      99	 1.3695680e+00	 1.3454293e-01	 1.4259898e+00	 1.5163761e-01	  1.2883345e+00 	 2.0387888e-01
     100	 1.3728221e+00	 1.3409789e-01	 1.4293197e+00	 1.5148221e-01	  1.2854592e+00 	 1.9961143e-01


.. parsed-literal::

     101	 1.3758941e+00	 1.3355760e-01	 1.4324990e+00	 1.5114188e-01	  1.2877990e+00 	 2.0778036e-01


.. parsed-literal::

     102	 1.3790949e+00	 1.3345255e-01	 1.4356382e+00	 1.5111661e-01	[ 1.2908823e+00]	 2.0423436e-01


.. parsed-literal::

     103	 1.3829774e+00	 1.3326740e-01	 1.4394822e+00	 1.5097550e-01	[ 1.2971937e+00]	 2.0884013e-01
     104	 1.3863301e+00	 1.3313894e-01	 1.4427834e+00	 1.5081473e-01	[ 1.3036402e+00]	 1.9123077e-01


.. parsed-literal::

     105	 1.3922993e+00	 1.3302048e-01	 1.4487467e+00	 1.5083039e-01	[ 1.3158418e+00]	 1.7645288e-01


.. parsed-literal::

     106	 1.3939041e+00	 1.3390103e-01	 1.4504141e+00	 1.5186178e-01	[ 1.3169110e+00]	 2.0865774e-01
     107	 1.3986806e+00	 1.3330477e-01	 1.4550501e+00	 1.5150812e-01	[ 1.3228435e+00]	 2.0003963e-01


.. parsed-literal::

     108	 1.4006890e+00	 1.3313769e-01	 1.4570685e+00	 1.5156711e-01	  1.3222684e+00 	 1.9878197e-01


.. parsed-literal::

     109	 1.4036370e+00	 1.3298782e-01	 1.4600908e+00	 1.5181304e-01	  1.3203576e+00 	 2.0646358e-01
     110	 1.4079559e+00	 1.3276812e-01	 1.4645008e+00	 1.5188514e-01	  1.3181700e+00 	 1.9315648e-01


.. parsed-literal::

     111	 1.4124088e+00	 1.3269087e-01	 1.4691159e+00	 1.5208600e-01	  1.3217608e+00 	 2.0957851e-01
     112	 1.4159629e+00	 1.3265696e-01	 1.4726438e+00	 1.5188621e-01	[ 1.3257420e+00]	 1.9504690e-01


.. parsed-literal::

     113	 1.4188071e+00	 1.3275672e-01	 1.4755038e+00	 1.5185742e-01	[ 1.3318946e+00]	 1.8249297e-01


.. parsed-literal::

     114	 1.4209651e+00	 1.3271985e-01	 1.4776765e+00	 1.5184826e-01	[ 1.3320842e+00]	 2.0989895e-01


.. parsed-literal::

     115	 1.4231321e+00	 1.3262323e-01	 1.4798899e+00	 1.5181817e-01	  1.3313650e+00 	 2.1647906e-01


.. parsed-literal::

     116	 1.4269482e+00	 1.3244880e-01	 1.4839100e+00	 1.5162416e-01	  1.3249195e+00 	 2.1395111e-01
     117	 1.4290571e+00	 1.3221542e-01	 1.4861404e+00	 1.5149713e-01	  1.3220188e+00 	 1.9990134e-01


.. parsed-literal::

     118	 1.4313560e+00	 1.3207502e-01	 1.4883425e+00	 1.5131616e-01	  1.3244304e+00 	 2.0490623e-01


.. parsed-literal::

     119	 1.4343725e+00	 1.3190925e-01	 1.4913821e+00	 1.5097463e-01	  1.3253694e+00 	 2.0244789e-01
     120	 1.4363147e+00	 1.3177281e-01	 1.4933372e+00	 1.5090168e-01	  1.3269920e+00 	 2.0540714e-01


.. parsed-literal::

     121	 1.4386507e+00	 1.3132235e-01	 1.4960024e+00	 1.5033675e-01	  1.3278980e+00 	 2.0444465e-01
     122	 1.4424896e+00	 1.3115453e-01	 1.4997716e+00	 1.5054671e-01	[ 1.3323827e+00]	 1.9805002e-01


.. parsed-literal::

     123	 1.4439343e+00	 1.3109556e-01	 1.5012182e+00	 1.5051926e-01	[ 1.3346784e+00]	 2.0324898e-01


.. parsed-literal::

     124	 1.4468149e+00	 1.3075838e-01	 1.5041972e+00	 1.5014484e-01	[ 1.3367913e+00]	 2.0416379e-01


.. parsed-literal::

     125	 1.4489123e+00	 1.3055516e-01	 1.5064564e+00	 1.4986397e-01	  1.3343064e+00 	 2.0814610e-01


.. parsed-literal::

     126	 1.4513703e+00	 1.3023529e-01	 1.5088954e+00	 1.4952646e-01	[ 1.3369556e+00]	 2.0960808e-01


.. parsed-literal::

     127	 1.4534218e+00	 1.3003941e-01	 1.5109417e+00	 1.4937249e-01	[ 1.3377625e+00]	 2.1978092e-01
     128	 1.4556120e+00	 1.2989012e-01	 1.5131244e+00	 1.4943648e-01	  1.3332144e+00 	 2.0116353e-01


.. parsed-literal::

     129	 1.4581784e+00	 1.2979277e-01	 1.5156871e+00	 1.4962065e-01	  1.3336602e+00 	 2.0736718e-01


.. parsed-literal::

     130	 1.4607719e+00	 1.2982697e-01	 1.5182494e+00	 1.5009383e-01	  1.3330079e+00 	 2.0868134e-01
     131	 1.4635665e+00	 1.2977276e-01	 1.5211325e+00	 1.5041013e-01	  1.3302161e+00 	 1.8516326e-01


.. parsed-literal::

     132	 1.4654051e+00	 1.3003968e-01	 1.5231513e+00	 1.5135179e-01	  1.3231433e+00 	 2.0412207e-01


.. parsed-literal::

     133	 1.4675378e+00	 1.2981439e-01	 1.5253056e+00	 1.5090424e-01	  1.3224059e+00 	 2.1781826e-01


.. parsed-literal::

     134	 1.4691524e+00	 1.2962477e-01	 1.5269804e+00	 1.5043415e-01	  1.3205407e+00 	 2.1245050e-01
     135	 1.4710473e+00	 1.2951701e-01	 1.5289203e+00	 1.5014317e-01	  1.3186112e+00 	 1.8983364e-01


.. parsed-literal::

     136	 1.4720105e+00	 1.2933898e-01	 1.5301888e+00	 1.4969070e-01	  1.3093206e+00 	 2.0413923e-01
     137	 1.4755501e+00	 1.2940407e-01	 1.5335070e+00	 1.4981970e-01	  1.3137500e+00 	 1.9804215e-01


.. parsed-literal::

     138	 1.4766077e+00	 1.2941724e-01	 1.5345184e+00	 1.4991984e-01	  1.3158558e+00 	 2.0675850e-01


.. parsed-literal::

     139	 1.4784385e+00	 1.2935134e-01	 1.5363692e+00	 1.4991588e-01	  1.3151995e+00 	 2.1180320e-01


.. parsed-literal::

     140	 1.4802195e+00	 1.2916862e-01	 1.5383112e+00	 1.4945327e-01	  1.3155888e+00 	 2.0427418e-01
     141	 1.4827074e+00	 1.2903086e-01	 1.5407564e+00	 1.4939165e-01	  1.3133084e+00 	 1.9750261e-01


.. parsed-literal::

     142	 1.4844502e+00	 1.2893478e-01	 1.5425345e+00	 1.4919765e-01	  1.3083470e+00 	 1.9972444e-01
     143	 1.4862600e+00	 1.2878198e-01	 1.5443748e+00	 1.4897956e-01	  1.3040154e+00 	 1.9731712e-01


.. parsed-literal::

     144	 1.4873288e+00	 1.2860909e-01	 1.5456238e+00	 1.4876447e-01	  1.2932813e+00 	 2.0513010e-01
     145	 1.4897777e+00	 1.2850875e-01	 1.5479300e+00	 1.4882316e-01	  1.2969353e+00 	 1.9867206e-01


.. parsed-literal::

     146	 1.4911351e+00	 1.2840614e-01	 1.5492447e+00	 1.4892629e-01	  1.2989253e+00 	 2.0231891e-01
     147	 1.4926597e+00	 1.2827963e-01	 1.5507549e+00	 1.4911651e-01	  1.3000386e+00 	 1.9745851e-01


.. parsed-literal::

     148	 1.4945739e+00	 1.2800882e-01	 1.5527084e+00	 1.4915850e-01	  1.3006725e+00 	 2.0173430e-01
     149	 1.4960735e+00	 1.2806715e-01	 1.5542510e+00	 1.4984054e-01	  1.2963591e+00 	 1.9894409e-01


.. parsed-literal::

     150	 1.4970273e+00	 1.2801170e-01	 1.5551961e+00	 1.4962877e-01	  1.2967663e+00 	 1.9873762e-01
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
    Inserting handle into data store.  model: <rail.estimation.algos._gpz_util.GP object at 0x7f15a45572e0>, GPzEstimator
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

