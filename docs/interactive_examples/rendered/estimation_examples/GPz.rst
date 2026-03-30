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
    /home/runner/.cache/lephare/runs/20260330T122231


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
      File "/tmp/ipykernel_6081/232514726.py", line 2, in <module>
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
       1	-3.4077999e-01	 3.1987728e-01	-3.3051592e-01	 3.2336863e-01	[-3.3625561e-01]	 4.4559216e-01


.. parsed-literal::

       2	-2.7475831e-01	 3.1099612e-01	-2.5195710e-01	 3.1411998e-01	[-2.5940184e-01]	 2.2399497e-01


.. parsed-literal::

       3	-2.3349326e-01	 2.9202216e-01	-1.9166792e-01	 2.9412926e-01	[-1.9643814e-01]	 3.0800462e-01


.. parsed-literal::

       4	-1.9857626e-01	 2.7439556e-01	-1.5044630e-01	 2.7424281e-01	[-1.4735975e-01]	 2.8777552e-01


.. parsed-literal::

       5	-1.3579949e-01	 2.5804672e-01	-9.8619453e-02	 2.5983860e-01	[-1.0122832e-01]	 2.1084023e-01


.. parsed-literal::

       6	-7.1532330e-02	 2.5187488e-01	-4.0603221e-02	 2.5268997e-01	[-4.1901424e-02]	 2.1488857e-01
       7	-5.0767806e-02	 2.4853890e-01	-2.6326618e-02	 2.5210431e-01	[-3.4352462e-02]	 1.7490959e-01


.. parsed-literal::

       8	-3.6043091e-02	 2.4589265e-01	-1.6497606e-02	 2.4852738e-01	[-2.5273692e-02]	 1.9378591e-01
       9	-2.8369572e-02	 2.4460945e-01	-1.0438656e-02	 2.4754903e-01	[-1.9868735e-02]	 1.6755271e-01


.. parsed-literal::

      10	-1.7974652e-02	 2.4266303e-01	-1.8035635e-03	 2.4480719e-01	[-9.2005034e-03]	 2.0000100e-01
      11	-1.0499142e-02	 2.4150337e-01	 3.9313278e-03	 2.4398370e-01	[-3.8838263e-03]	 1.9044852e-01


.. parsed-literal::

      12	-4.7047757e-03	 2.4028562e-01	 9.6581073e-03	 2.4273470e-01	[ 1.9275515e-03]	 1.7994261e-01
      13	-8.6106637e-04	 2.3949684e-01	 1.3689944e-02	 2.4204569e-01	[ 5.7217249e-03]	 1.7643166e-01


.. parsed-literal::

      14	 1.3649458e-01	 2.2823233e-01	 1.6174822e-01	 2.3195945e-01	[ 1.6058050e-01]	 4.2797732e-01
      15	 2.0775266e-01	 2.2227263e-01	 2.3517143e-01	 2.2502697e-01	[ 2.3160218e-01]	 1.8601012e-01


.. parsed-literal::

      16	 3.0114531e-01	 2.1584831e-01	 3.3241351e-01	 2.1909433e-01	[ 3.2468540e-01]	 1.7319417e-01
      17	 3.5748301e-01	 2.1000143e-01	 3.9156254e-01	 2.1255915e-01	[ 3.8918715e-01]	 1.9997811e-01


.. parsed-literal::

      18	 4.0125921e-01	 2.0761800e-01	 4.3566207e-01	 2.0969677e-01	[ 4.3744560e-01]	 1.9965267e-01


.. parsed-literal::

      19	 4.5442567e-01	 2.0352020e-01	 4.9017722e-01	 2.0526416e-01	[ 4.9846397e-01]	 2.0898080e-01
      20	 5.5182295e-01	 1.9926407e-01	 5.9305727e-01	 2.0363575e-01	[ 6.1182999e-01]	 1.6816068e-01


.. parsed-literal::

      21	 5.8257856e-01	 2.0032363e-01	 6.2561987e-01	 2.0516466e-01	[ 6.4347420e-01]	 2.1028852e-01


.. parsed-literal::

      22	 6.2001061e-01	 1.9462279e-01	 6.6207980e-01	 1.9977506e-01	[ 6.7665075e-01]	 2.0431757e-01
      23	 6.3326118e-01	 2.0421188e-01	 6.7171706e-01	 2.0934853e-01	  6.6668035e-01 	 1.7065644e-01


.. parsed-literal::

      24	 6.7294589e-01	 2.0413657e-01	 7.1067415e-01	 2.0857076e-01	[ 7.1134510e-01]	 2.0916986e-01
      25	 7.2095852e-01	 1.9522855e-01	 7.6263427e-01	 2.0207190e-01	[ 7.6706102e-01]	 1.8463421e-01


.. parsed-literal::

      26	 7.6357324e-01	 1.9198884e-01	 8.0495938e-01	 2.0047136e-01	[ 8.0261778e-01]	 2.0329785e-01
      27	 7.9767204e-01	 1.9228177e-01	 8.3987572e-01	 2.0228304e-01	[ 8.3498432e-01]	 2.0485473e-01


.. parsed-literal::

      28	 8.2695625e-01	 1.9030446e-01	 8.7174369e-01	 2.0106685e-01	[ 8.6178310e-01]	 2.1462893e-01
      29	 8.5533945e-01	 1.8301156e-01	 9.0075324e-01	 1.9146265e-01	[ 8.9053078e-01]	 2.0067787e-01


.. parsed-literal::

      30	 8.7725190e-01	 1.7926048e-01	 9.2270863e-01	 1.8771623e-01	[ 9.1810721e-01]	 2.1040368e-01
      31	 9.0278827e-01	 1.7392171e-01	 9.4897023e-01	 1.8346833e-01	[ 9.4679109e-01]	 1.7357278e-01


.. parsed-literal::

      32	 9.2879526e-01	 1.7021633e-01	 9.7626589e-01	 1.8330260e-01	[ 9.6318610e-01]	 1.8325019e-01
      33	 9.4823560e-01	 1.6787243e-01	 9.9642475e-01	 1.8124192e-01	[ 9.7039477e-01]	 1.9573355e-01


.. parsed-literal::

      34	 9.6177550e-01	 1.6593553e-01	 1.0096600e+00	 1.7952975e-01	[ 9.8566436e-01]	 1.8232226e-01


.. parsed-literal::

      35	 9.7770125e-01	 1.6395672e-01	 1.0256498e+00	 1.7836046e-01	[ 1.0016384e+00]	 2.1800113e-01
      36	 9.9580716e-01	 1.6195110e-01	 1.0447132e+00	 1.7807637e-01	[ 1.0195411e+00]	 1.7839313e-01


.. parsed-literal::

      37	 1.0127089e+00	 1.5864331e-01	 1.0621213e+00	 1.7627231e-01	[ 1.0318014e+00]	 2.0172930e-01


.. parsed-literal::

      38	 1.0317306e+00	 1.5696409e-01	 1.0817157e+00	 1.7515221e-01	[ 1.0475422e+00]	 2.0863485e-01


.. parsed-literal::

      39	 1.0472719e+00	 1.5624099e-01	 1.0979850e+00	 1.7444717e-01	[ 1.0579176e+00]	 2.0569777e-01
      40	 1.0589520e+00	 1.5470905e-01	 1.1095859e+00	 1.7165956e-01	[ 1.0682809e+00]	 1.7757750e-01


.. parsed-literal::

      41	 1.0674866e+00	 1.5315119e-01	 1.1179502e+00	 1.6937738e-01	[ 1.0772214e+00]	 1.9027019e-01
      42	 1.0854749e+00	 1.4986278e-01	 1.1360999e+00	 1.6622438e-01	[ 1.0923647e+00]	 1.9483352e-01


.. parsed-literal::

      43	 1.0946793e+00	 1.4792003e-01	 1.1450747e+00	 1.6419515e-01	[ 1.1015427e+00]	 2.0656872e-01
      44	 1.1053803e+00	 1.4689429e-01	 1.1555499e+00	 1.6387377e-01	[ 1.1132977e+00]	 2.0266581e-01


.. parsed-literal::

      45	 1.1168871e+00	 1.4599010e-01	 1.1673530e+00	 1.6397077e-01	[ 1.1221322e+00]	 2.0796824e-01


.. parsed-literal::

      46	 1.1286453e+00	 1.4505948e-01	 1.1792065e+00	 1.6290222e-01	[ 1.1329137e+00]	 2.0611334e-01


.. parsed-literal::

      47	 1.1425344e+00	 1.4239385e-01	 1.1930549e+00	 1.5994808e-01	[ 1.1410450e+00]	 2.0680308e-01


.. parsed-literal::

      48	 1.1589818e+00	 1.4129048e-01	 1.2093936e+00	 1.5637960e-01	[ 1.1576113e+00]	 2.0575857e-01
      49	 1.1693798e+00	 1.3977298e-01	 1.2200423e+00	 1.5386180e-01	[ 1.1633662e+00]	 1.8333817e-01


.. parsed-literal::

      50	 1.1844371e+00	 1.3756900e-01	 1.2357930e+00	 1.5013372e-01	[ 1.1718969e+00]	 1.9497776e-01


.. parsed-literal::

      51	 1.2002846e+00	 1.3561559e-01	 1.2520516e+00	 1.4781400e-01	[ 1.1844661e+00]	 2.0271850e-01
      52	 1.2165401e+00	 1.3432055e-01	 1.2686859e+00	 1.4697944e-01	[ 1.2004972e+00]	 1.6189909e-01


.. parsed-literal::

      53	 1.2286927e+00	 1.3345665e-01	 1.2812775e+00	 1.4656202e-01	[ 1.2148242e+00]	 1.6126418e-01
      54	 1.2397608e+00	 1.3321032e-01	 1.2925035e+00	 1.4644139e-01	[ 1.2244969e+00]	 1.9768643e-01


.. parsed-literal::

      55	 1.2514476e+00	 1.3253720e-01	 1.3041576e+00	 1.4556451e-01	[ 1.2335315e+00]	 1.6712165e-01
      56	 1.2639848e+00	 1.3211524e-01	 1.3170699e+00	 1.4480368e-01	[ 1.2374899e+00]	 1.8059206e-01


.. parsed-literal::

      57	 1.2708851e+00	 1.3249034e-01	 1.3240296e+00	 1.4578734e-01	  1.2369922e+00 	 1.8289256e-01
      58	 1.2810247e+00	 1.3177138e-01	 1.3338073e+00	 1.4523179e-01	[ 1.2494371e+00]	 1.9849110e-01


.. parsed-literal::

      59	 1.2890432e+00	 1.3121109e-01	 1.3421306e+00	 1.4478987e-01	[ 1.2518901e+00]	 1.9706345e-01


.. parsed-literal::

      60	 1.3004122e+00	 1.3132369e-01	 1.3541353e+00	 1.4510953e-01	[ 1.2563586e+00]	 2.0465899e-01
      61	 1.3086197e+00	 1.3220876e-01	 1.3623570e+00	 1.4550595e-01	[ 1.2589830e+00]	 1.6466784e-01


.. parsed-literal::

      62	 1.3201924e+00	 1.3238075e-01	 1.3739976e+00	 1.4568822e-01	[ 1.2741906e+00]	 1.9874763e-01
      63	 1.3270505e+00	 1.3210025e-01	 1.3809517e+00	 1.4558843e-01	[ 1.2854289e+00]	 1.7354488e-01


.. parsed-literal::

      64	 1.3352193e+00	 1.3211200e-01	 1.3894175e+00	 1.4548330e-01	[ 1.2911702e+00]	 2.0587659e-01


.. parsed-literal::

      65	 1.3424941e+00	 1.3128983e-01	 1.3970410e+00	 1.4457265e-01	[ 1.3033971e+00]	 2.0804763e-01
      66	 1.3497416e+00	 1.3105737e-01	 1.4041155e+00	 1.4417397e-01	[ 1.3042706e+00]	 1.9373608e-01


.. parsed-literal::

      67	 1.3552829e+00	 1.3036016e-01	 1.4096972e+00	 1.4332368e-01	  1.3029768e+00 	 1.9494677e-01


.. parsed-literal::

      68	 1.3616648e+00	 1.2976813e-01	 1.4162761e+00	 1.4259515e-01	[ 1.3043253e+00]	 2.0196795e-01
      69	 1.3689188e+00	 1.2897971e-01	 1.4238524e+00	 1.4099083e-01	[ 1.3098932e+00]	 1.7848849e-01


.. parsed-literal::

      70	 1.3770790e+00	 1.2839027e-01	 1.4320214e+00	 1.4085014e-01	[ 1.3220187e+00]	 1.7019653e-01
      71	 1.3829463e+00	 1.2854020e-01	 1.4378868e+00	 1.4086440e-01	[ 1.3335656e+00]	 1.9810104e-01


.. parsed-literal::

      72	 1.3896499e+00	 1.2816183e-01	 1.4450111e+00	 1.4035807e-01	[ 1.3377469e+00]	 1.7761660e-01
      73	 1.3945123e+00	 1.2826081e-01	 1.4500087e+00	 1.4018549e-01	[ 1.3387976e+00]	 1.9965410e-01


.. parsed-literal::

      74	 1.3994004e+00	 1.2776138e-01	 1.4547638e+00	 1.3969083e-01	[ 1.3399908e+00]	 1.9747305e-01
      75	 1.4049120e+00	 1.2686617e-01	 1.4602472e+00	 1.3926330e-01	  1.3394448e+00 	 1.8328238e-01


.. parsed-literal::

      76	 1.4098675e+00	 1.2651132e-01	 1.4652989e+00	 1.3888735e-01	[ 1.3408452e+00]	 1.8082380e-01
      77	 1.4154279e+00	 1.2588189e-01	 1.4711734e+00	 1.3916783e-01	[ 1.3416848e+00]	 2.0003653e-01


.. parsed-literal::

      78	 1.4202014e+00	 1.2592116e-01	 1.4759472e+00	 1.3929586e-01	[ 1.3474058e+00]	 1.9999719e-01


.. parsed-literal::

      79	 1.4243013e+00	 1.2602650e-01	 1.4800870e+00	 1.3931212e-01	[ 1.3537521e+00]	 2.0593429e-01
      80	 1.4288697e+00	 1.2593039e-01	 1.4848348e+00	 1.3971748e-01	[ 1.3591044e+00]	 1.8779182e-01


.. parsed-literal::

      81	 1.4335827e+00	 1.2579112e-01	 1.4896340e+00	 1.3975455e-01	[ 1.3597612e+00]	 1.8079734e-01
      82	 1.4374208e+00	 1.2558519e-01	 1.4934193e+00	 1.3955899e-01	[ 1.3633838e+00]	 2.0010924e-01


.. parsed-literal::

      83	 1.4423805e+00	 1.2514539e-01	 1.4984363e+00	 1.3980700e-01	  1.3611979e+00 	 1.7477250e-01
      84	 1.4465994e+00	 1.2507608e-01	 1.5027749e+00	 1.3981397e-01	[ 1.3657292e+00]	 1.6535044e-01


.. parsed-literal::

      85	 1.4505747e+00	 1.2490906e-01	 1.5068317e+00	 1.3966601e-01	[ 1.3671835e+00]	 2.0546317e-01
      86	 1.4545767e+00	 1.2467685e-01	 1.5109612e+00	 1.3960896e-01	  1.3654318e+00 	 1.9182539e-01


.. parsed-literal::

      87	 1.4573028e+00	 1.2450127e-01	 1.5137903e+00	 1.3931742e-01	  1.3616553e+00 	 1.8810463e-01
      88	 1.4600556e+00	 1.2426464e-01	 1.5164809e+00	 1.3916938e-01	  1.3611697e+00 	 2.0457196e-01


.. parsed-literal::

      89	 1.4645035e+00	 1.2370840e-01	 1.5209891e+00	 1.3919492e-01	  1.3547543e+00 	 1.6965032e-01


.. parsed-literal::

      90	 1.4671892e+00	 1.2338614e-01	 1.5237484e+00	 1.3934517e-01	  1.3507726e+00 	 2.0946932e-01


.. parsed-literal::

      91	 1.4709252e+00	 1.2311650e-01	 1.5275646e+00	 1.3934589e-01	  1.3480686e+00 	 2.1632147e-01
      92	 1.4749922e+00	 1.2276602e-01	 1.5318533e+00	 1.3921988e-01	  1.3464155e+00 	 1.9308639e-01


.. parsed-literal::

      93	 1.4779363e+00	 1.2266648e-01	 1.5349859e+00	 1.3879906e-01	  1.3430486e+00 	 1.9802999e-01
      94	 1.4803056e+00	 1.2251590e-01	 1.5373276e+00	 1.3874470e-01	  1.3394034e+00 	 2.0055485e-01


.. parsed-literal::

      95	 1.4830370e+00	 1.2226864e-01	 1.5400498e+00	 1.3860065e-01	  1.3369630e+00 	 2.1464944e-01
      96	 1.4851993e+00	 1.2199629e-01	 1.5422342e+00	 1.3875724e-01	  1.3260399e+00 	 1.9045711e-01


.. parsed-literal::

      97	 1.4876804e+00	 1.2182920e-01	 1.5446510e+00	 1.3859917e-01	  1.3287557e+00 	 1.8903112e-01


.. parsed-literal::

      98	 1.4908819e+00	 1.2150958e-01	 1.5478407e+00	 1.3812519e-01	  1.3331155e+00 	 2.0791650e-01
      99	 1.4927003e+00	 1.2147367e-01	 1.5497232e+00	 1.3786297e-01	  1.3326666e+00 	 2.0484090e-01


.. parsed-literal::

     100	 1.4943181e+00	 1.2145615e-01	 1.5513427e+00	 1.3770246e-01	  1.3326396e+00 	 2.0006990e-01


.. parsed-literal::

     101	 1.4975367e+00	 1.2125171e-01	 1.5547207e+00	 1.3742476e-01	  1.3218652e+00 	 2.1382642e-01


.. parsed-literal::

     102	 1.4997180e+00	 1.2113773e-01	 1.5569625e+00	 1.3738686e-01	  1.3150683e+00 	 2.0527506e-01


.. parsed-literal::

     103	 1.5028815e+00	 1.2088389e-01	 1.5601821e+00	 1.3758014e-01	  1.2993423e+00 	 2.1135426e-01
     104	 1.5049855e+00	 1.2070411e-01	 1.5622774e+00	 1.3771697e-01	  1.2916899e+00 	 2.0389104e-01


.. parsed-literal::

     105	 1.5067416e+00	 1.2069379e-01	 1.5639428e+00	 1.3766221e-01	  1.2960314e+00 	 2.0074081e-01


.. parsed-literal::

     106	 1.5092048e+00	 1.2057642e-01	 1.5664239e+00	 1.3742277e-01	  1.2959315e+00 	 2.1375847e-01
     107	 1.5113975e+00	 1.2044909e-01	 1.5687607e+00	 1.3725281e-01	  1.2858012e+00 	 1.9804978e-01


.. parsed-literal::

     108	 1.5127624e+00	 1.2017038e-01	 1.5704151e+00	 1.3698969e-01	  1.2603000e+00 	 2.1043158e-01
     109	 1.5150977e+00	 1.2009007e-01	 1.5727279e+00	 1.3704245e-01	  1.2630220e+00 	 1.8212438e-01


.. parsed-literal::

     110	 1.5160948e+00	 1.2001146e-01	 1.5737041e+00	 1.3717098e-01	  1.2592096e+00 	 2.0710945e-01


.. parsed-literal::

     111	 1.5177991e+00	 1.1981501e-01	 1.5754622e+00	 1.3729147e-01	  1.2440216e+00 	 2.0477057e-01


.. parsed-literal::

     112	 1.5192045e+00	 1.1959139e-01	 1.5769463e+00	 1.3741797e-01	  1.2161110e+00 	 2.1302056e-01
     113	 1.5207819e+00	 1.1959719e-01	 1.5784819e+00	 1.3729602e-01	  1.2174496e+00 	 1.9769001e-01


.. parsed-literal::

     114	 1.5219693e+00	 1.1959351e-01	 1.5796616e+00	 1.3709075e-01	  1.2162445e+00 	 2.0747423e-01
     115	 1.5234360e+00	 1.1961029e-01	 1.5810909e+00	 1.3687760e-01	  1.2152510e+00 	 2.0003986e-01


.. parsed-literal::

     116	 1.5252041e+00	 1.1957037e-01	 1.5829005e+00	 1.3668709e-01	  1.2077209e+00 	 1.7988586e-01


.. parsed-literal::

     117	 1.5266344e+00	 1.1959004e-01	 1.5842955e+00	 1.3665315e-01	  1.2068916e+00 	 2.0526791e-01


.. parsed-literal::

     118	 1.5276314e+00	 1.1951901e-01	 1.5853004e+00	 1.3671004e-01	  1.2034583e+00 	 2.0406747e-01
     119	 1.5288599e+00	 1.1939995e-01	 1.5865912e+00	 1.3685452e-01	  1.1913396e+00 	 1.9712710e-01


.. parsed-literal::

     120	 1.5301095e+00	 1.1932544e-01	 1.5878940e+00	 1.3695513e-01	  1.1808578e+00 	 1.9402838e-01
     121	 1.5316037e+00	 1.1927229e-01	 1.5895106e+00	 1.3723887e-01	  1.1617032e+00 	 2.0018506e-01


.. parsed-literal::

     122	 1.5333872e+00	 1.1920447e-01	 1.5912190e+00	 1.3731361e-01	  1.1562336e+00 	 1.8396544e-01
     123	 1.5343132e+00	 1.1920497e-01	 1.5920640e+00	 1.3726085e-01	  1.1581615e+00 	 2.0105219e-01


.. parsed-literal::

     124	 1.5358003e+00	 1.1916018e-01	 1.5934483e+00	 1.3721615e-01	  1.1635251e+00 	 1.8917704e-01
     125	 1.5369289e+00	 1.1922265e-01	 1.5945289e+00	 1.3714145e-01	  1.1523531e+00 	 1.8271184e-01


.. parsed-literal::

     126	 1.5382118e+00	 1.1916761e-01	 1.5958013e+00	 1.3712937e-01	  1.1571957e+00 	 2.0523643e-01


.. parsed-literal::

     127	 1.5396148e+00	 1.1912736e-01	 1.5972793e+00	 1.3710419e-01	  1.1561223e+00 	 2.1537828e-01
     128	 1.5408961e+00	 1.1913879e-01	 1.5986406e+00	 1.3700432e-01	  1.1522945e+00 	 1.8001485e-01


.. parsed-literal::

     129	 1.5418166e+00	 1.1903081e-01	 1.5998576e+00	 1.3684628e-01	  1.1416020e+00 	 1.9240618e-01
     130	 1.5439564e+00	 1.1908090e-01	 1.6019100e+00	 1.3675381e-01	  1.1384349e+00 	 2.0047522e-01


.. parsed-literal::

     131	 1.5446085e+00	 1.1904537e-01	 1.6025116e+00	 1.3675908e-01	  1.1382351e+00 	 2.0704889e-01


.. parsed-literal::

     132	 1.5459896e+00	 1.1891865e-01	 1.6038900e+00	 1.3676250e-01	  1.1367685e+00 	 2.1351147e-01


.. parsed-literal::

     133	 1.5468178e+00	 1.1886999e-01	 1.6047361e+00	 1.3680453e-01	  1.1348019e+00 	 2.8894091e-01
     134	 1.5479800e+00	 1.1876014e-01	 1.6059090e+00	 1.3684746e-01	  1.1348501e+00 	 1.8148088e-01


.. parsed-literal::

     135	 1.5491441e+00	 1.1872618e-01	 1.6070991e+00	 1.3685308e-01	  1.1336414e+00 	 1.7709160e-01
     136	 1.5503985e+00	 1.1872946e-01	 1.6083792e+00	 1.3686907e-01	  1.1289089e+00 	 1.7190719e-01


.. parsed-literal::

     137	 1.5516838e+00	 1.1880361e-01	 1.6097369e+00	 1.3688446e-01	  1.1149590e+00 	 1.7268395e-01
     138	 1.5530101e+00	 1.1880642e-01	 1.6110573e+00	 1.3690475e-01	  1.1112673e+00 	 1.7822814e-01


.. parsed-literal::

     139	 1.5543606e+00	 1.1878091e-01	 1.6124374e+00	 1.3693732e-01	  1.0947449e+00 	 2.0590544e-01
     140	 1.5555662e+00	 1.1876195e-01	 1.6137184e+00	 1.3700717e-01	  1.0787902e+00 	 1.6721320e-01


.. parsed-literal::

     141	 1.5569785e+00	 1.1877446e-01	 1.6152488e+00	 1.3701075e-01	  1.0528423e+00 	 1.9787145e-01
     142	 1.5581044e+00	 1.1882224e-01	 1.6164214e+00	 1.3681994e-01	  1.0393749e+00 	 1.9139004e-01


.. parsed-literal::

     143	 1.5591631e+00	 1.1884711e-01	 1.6175119e+00	 1.3651694e-01	  1.0349964e+00 	 1.7196965e-01
     144	 1.5601024e+00	 1.1886228e-01	 1.6184514e+00	 1.3621857e-01	  1.0358147e+00 	 1.8020844e-01


.. parsed-literal::

     145	 1.5611730e+00	 1.1886225e-01	 1.6195288e+00	 1.3598126e-01	  1.0287512e+00 	 1.7699981e-01
     146	 1.5623717e+00	 1.1872493e-01	 1.6207485e+00	 1.3586366e-01	  1.0165870e+00 	 1.9648886e-01


.. parsed-literal::

     147	 1.5635451e+00	 1.1849217e-01	 1.6219319e+00	 1.3585160e-01	  9.9959002e-01 	 1.9804358e-01


.. parsed-literal::

     148	 1.5646651e+00	 1.1832776e-01	 1.6230313e+00	 1.3589221e-01	  9.8321704e-01 	 2.0740676e-01
     149	 1.5655718e+00	 1.1815415e-01	 1.6239452e+00	 1.3588681e-01	  9.6688098e-01 	 1.8550897e-01


.. parsed-literal::

     150	 1.5665960e+00	 1.1794946e-01	 1.6249878e+00	 1.3576827e-01	  9.5066301e-01 	 2.0447421e-01
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
    Inserting handle into data store.  model: <rail.estimation.algos._gpz_util.GP object at 0x7f67a0897970>, GPzEstimator
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

