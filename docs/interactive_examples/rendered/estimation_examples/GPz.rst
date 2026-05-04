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
    /home/runner/.cache/lephare/runs/20260504T123336


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
      File "/tmp/ipykernel_8398/232514726.py", line 2, in <module>
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
       1	-3.3244299e-01	 3.1633148e-01	-3.2214087e-01	 3.3344968e-01	[-3.5601254e-01]	 4.5086217e-01


.. parsed-literal::

       2	-2.5851411e-01	 3.0510344e-01	-2.3293516e-01	 3.2165277e-01	[-2.8628639e-01]	 2.2875237e-01


.. parsed-literal::

       3	-2.1290591e-01	 2.8427459e-01	-1.6856869e-01	 3.0194439e-01	[-2.4466883e-01]	 3.1807899e-01
       4	-1.8921109e-01	 2.6159315e-01	-1.4865398e-01	 2.7931613e-01	 -2.5689534e-01 	 1.6922545e-01


.. parsed-literal::

       5	-9.4041011e-02	 2.5443114e-01	-5.6978336e-02	 2.6882378e-01	[-1.2641340e-01]	 2.1776676e-01
       6	-6.2379901e-02	 2.4900499e-01	-2.9374428e-02	 2.6436835e-01	[-8.4814668e-02]	 1.9831944e-01


.. parsed-literal::

       7	-4.1984334e-02	 2.4579486e-01	-1.6312130e-02	 2.5969275e-01	[-7.0999595e-02]	 1.9452810e-01
       8	-3.0675281e-02	 2.4402470e-01	-8.7103812e-03	 2.5736090e-01	[-6.3343608e-02]	 1.8882847e-01


.. parsed-literal::

       9	-1.4854586e-02	 2.4105407e-01	 3.9773003e-03	 2.5454191e-01	[-5.1360933e-02]	 1.9515824e-01
      10	-3.5890328e-03	 2.3879312e-01	 1.2834816e-02	 2.5330358e-01	 -5.4543603e-02 	 1.9454694e-01


.. parsed-literal::

      11	 3.7661081e-03	 2.3772396e-01	 1.9151300e-02	 2.5323280e-01	[-4.6946082e-02]	 2.0162487e-01
      12	 6.1181452e-03	 2.3724106e-01	 2.1290830e-02	 2.5282609e-01	[-4.4851071e-02]	 1.9753337e-01


.. parsed-literal::

      13	 1.1024039e-02	 2.3639568e-01	 2.5959387e-02	 2.5219297e-01	[-4.3732700e-02]	 1.7006469e-01


.. parsed-literal::

      14	 5.1820292e-02	 2.2905001e-01	 6.9206765e-02	 2.4747104e-01	[-1.6278542e-03]	 3.1256342e-01


.. parsed-literal::

      15	 8.3794463e-02	 2.2216396e-01	 1.0448860e-01	 2.3838256e-01	[ 5.5373918e-02]	 3.1392884e-01


.. parsed-literal::

      16	 1.3917246e-01	 2.1620410e-01	 1.6197957e-01	 2.3404479e-01	[ 1.1106555e-01]	 2.1021962e-01
      17	 2.4738915e-01	 2.1341759e-01	 2.7633458e-01	 2.3127873e-01	[ 2.0615264e-01]	 2.0967412e-01


.. parsed-literal::

      18	 2.8270112e-01	 2.1300132e-01	 3.1501296e-01	 2.2781854e-01	[ 2.3916032e-01]	 2.1430707e-01


.. parsed-literal::

      19	 3.3329383e-01	 2.0768706e-01	 3.6720863e-01	 2.2097547e-01	[ 2.9672047e-01]	 2.1779799e-01


.. parsed-literal::

      20	 4.0487410e-01	 2.0418037e-01	 4.3882978e-01	 2.1781190e-01	[ 3.7341454e-01]	 2.0393872e-01
      21	 4.9604819e-01	 2.0516252e-01	 5.3166703e-01	 2.1738590e-01	[ 4.7955838e-01]	 2.0221686e-01


.. parsed-literal::

      22	 5.9067429e-01	 1.9886132e-01	 6.2951706e-01	 2.1226596e-01	[ 5.8790490e-01]	 2.0376062e-01
      23	 6.4614809e-01	 1.9193535e-01	 6.8818896e-01	 2.0632687e-01	[ 6.5367962e-01]	 2.0669699e-01


.. parsed-literal::

      24	 6.9031595e-01	 1.8943092e-01	 7.3219605e-01	 2.0319697e-01	[ 7.1206913e-01]	 2.1152687e-01
      25	 7.2816704e-01	 1.8753855e-01	 7.6935275e-01	 2.0355005e-01	[ 7.4189952e-01]	 1.8976259e-01


.. parsed-literal::

      26	 7.6917110e-01	 1.9005624e-01	 8.0951710e-01	 2.0822844e-01	[ 7.6579756e-01]	 1.9554353e-01
      27	 7.8512324e-01	 1.9408235e-01	 8.2489230e-01	 2.0924704e-01	[ 7.8366850e-01]	 1.9049120e-01


.. parsed-literal::

      28	 8.1801201e-01	 1.8792102e-01	 8.5876232e-01	 2.0229939e-01	[ 8.2456301e-01]	 2.0554543e-01
      29	 8.4260897e-01	 1.8517753e-01	 8.8460853e-01	 1.9904912e-01	[ 8.4743752e-01]	 1.9225073e-01


.. parsed-literal::

      30	 8.7079961e-01	 1.8451889e-01	 9.1420751e-01	 1.9770010e-01	[ 8.7251591e-01]	 1.9648147e-01


.. parsed-literal::

      31	 8.8881463e-01	 1.8557757e-01	 9.3206517e-01	 1.9830576e-01	[ 8.9452586e-01]	 2.0390582e-01
      32	 9.0254270e-01	 1.8363586e-01	 9.4602647e-01	 1.9674007e-01	[ 9.0943535e-01]	 1.7523074e-01


.. parsed-literal::

      33	 9.1827857e-01	 1.8288074e-01	 9.6188447e-01	 1.9687288e-01	[ 9.2189596e-01]	 1.9260001e-01
      34	 9.3825826e-01	 1.8235973e-01	 9.8294027e-01	 1.9751930e-01	[ 9.4048665e-01]	 1.9962931e-01


.. parsed-literal::

      35	 9.5062276e-01	 1.8254562e-01	 9.9672003e-01	 1.9801857e-01	[ 9.5164624e-01]	 1.9786763e-01
      36	 9.6529744e-01	 1.8143997e-01	 1.0114129e+00	 1.9717425e-01	[ 9.6633796e-01]	 1.8009281e-01


.. parsed-literal::

      37	 9.7615001e-01	 1.8070591e-01	 1.0225106e+00	 1.9624143e-01	[ 9.7439610e-01]	 2.0710206e-01
      38	 9.9052030e-01	 1.7964663e-01	 1.0372249e+00	 1.9495250e-01	[ 9.8232825e-01]	 1.7930698e-01


.. parsed-literal::

      39	 1.0034178e+00	 1.7854095e-01	 1.0506452e+00	 1.9187578e-01	[ 9.9405542e-01]	 1.9985652e-01


.. parsed-literal::

      40	 1.0156989e+00	 1.7637892e-01	 1.0628792e+00	 1.8945345e-01	[ 1.0030173e+00]	 2.0426822e-01


.. parsed-literal::

      41	 1.0233040e+00	 1.7551666e-01	 1.0705838e+00	 1.8839304e-01	[ 1.0120371e+00]	 2.1245551e-01
      42	 1.0343550e+00	 1.7420190e-01	 1.0821290e+00	 1.8667592e-01	[ 1.0177864e+00]	 2.0065856e-01


.. parsed-literal::

      43	 1.0444305e+00	 1.7369029e-01	 1.0925113e+00	 1.8593069e-01	[ 1.0292158e+00]	 1.9919538e-01
      44	 1.0519683e+00	 1.7286878e-01	 1.1001890e+00	 1.8467317e-01	[ 1.0359448e+00]	 1.9981670e-01


.. parsed-literal::

      45	 1.0597628e+00	 1.7256269e-01	 1.1080207e+00	 1.8430035e-01	[ 1.0400318e+00]	 1.9892073e-01
      46	 1.0673388e+00	 1.7260642e-01	 1.1158926e+00	 1.8387782e-01	[ 1.0448587e+00]	 1.9952941e-01


.. parsed-literal::

      47	 1.0722860e+00	 1.7337493e-01	 1.1209652e+00	 1.8513676e-01	[ 1.0481981e+00]	 2.1314311e-01


.. parsed-literal::

      48	 1.0766162e+00	 1.7280588e-01	 1.1251922e+00	 1.8438854e-01	[ 1.0532379e+00]	 2.1724296e-01
      49	 1.0836758e+00	 1.7225203e-01	 1.1324932e+00	 1.8334789e-01	[ 1.0616634e+00]	 2.0283246e-01


.. parsed-literal::

      50	 1.0897462e+00	 1.7220110e-01	 1.1388617e+00	 1.8322535e-01	[ 1.0694114e+00]	 2.0218801e-01


.. parsed-literal::

      51	 1.0996358e+00	 1.7107515e-01	 1.1495930e+00	 1.8137473e-01	[ 1.0816871e+00]	 2.0265055e-01


.. parsed-literal::

      52	 1.1080574e+00	 1.7073747e-01	 1.1582972e+00	 1.8102952e-01	[ 1.0860715e+00]	 2.0272589e-01
      53	 1.1128071e+00	 1.7015153e-01	 1.1629834e+00	 1.8039842e-01	[ 1.0897516e+00]	 1.8822145e-01


.. parsed-literal::

      54	 1.1216506e+00	 1.6793872e-01	 1.1722749e+00	 1.7819165e-01	[ 1.0970968e+00]	 2.0072222e-01
      55	 1.1259355e+00	 1.6780860e-01	 1.1770569e+00	 1.7742796e-01	[ 1.1024817e+00]	 1.9464803e-01


.. parsed-literal::

      56	 1.1314023e+00	 1.6660587e-01	 1.1824550e+00	 1.7627407e-01	[ 1.1064781e+00]	 2.0873499e-01
      57	 1.1365706e+00	 1.6526947e-01	 1.1878576e+00	 1.7486072e-01	[ 1.1130300e+00]	 1.9343042e-01


.. parsed-literal::

      58	 1.1406436e+00	 1.6443584e-01	 1.1920296e+00	 1.7379974e-01	[ 1.1173187e+00]	 2.0593476e-01


.. parsed-literal::

      59	 1.1483179e+00	 1.6274598e-01	 1.1999711e+00	 1.7143404e-01	[ 1.1280753e+00]	 2.0335817e-01
      60	 1.1538356e+00	 1.6197107e-01	 1.2056477e+00	 1.7055273e-01	[ 1.1318713e+00]	 2.0365310e-01


.. parsed-literal::

      61	 1.1584256e+00	 1.6151392e-01	 1.2101610e+00	 1.7010954e-01	[ 1.1341858e+00]	 2.0108676e-01


.. parsed-literal::

      62	 1.1644358e+00	 1.6062166e-01	 1.2162635e+00	 1.6892399e-01	[ 1.1403933e+00]	 2.0620179e-01
      63	 1.1694613e+00	 1.5973878e-01	 1.2213704e+00	 1.6810684e-01	[ 1.1445319e+00]	 1.9651461e-01


.. parsed-literal::

      64	 1.1736640e+00	 1.5898165e-01	 1.2257730e+00	 1.6715936e-01	[ 1.1513442e+00]	 2.0761132e-01


.. parsed-literal::

      65	 1.1802554e+00	 1.5816419e-01	 1.2321678e+00	 1.6652846e-01	[ 1.1576525e+00]	 2.1429896e-01


.. parsed-literal::

      66	 1.1841754e+00	 1.5780862e-01	 1.2359904e+00	 1.6629087e-01	[ 1.1606753e+00]	 2.1084023e-01
      67	 1.1916256e+00	 1.5668758e-01	 1.2436039e+00	 1.6499237e-01	[ 1.1672791e+00]	 1.9659758e-01


.. parsed-literal::

      68	 1.1939163e+00	 1.5573279e-01	 1.2466255e+00	 1.6443119e-01	  1.1640452e+00 	 1.9680691e-01


.. parsed-literal::

      69	 1.2028209e+00	 1.5497287e-01	 1.2552300e+00	 1.6294152e-01	[ 1.1769581e+00]	 2.1867800e-01
      70	 1.2066595e+00	 1.5452506e-01	 1.2590980e+00	 1.6236374e-01	[ 1.1803275e+00]	 1.9454980e-01


.. parsed-literal::

      71	 1.2123328e+00	 1.5358514e-01	 1.2649583e+00	 1.6146894e-01	[ 1.1858899e+00]	 2.0215964e-01


.. parsed-literal::

      72	 1.2191695e+00	 1.5258582e-01	 1.2718846e+00	 1.6070251e-01	[ 1.1940018e+00]	 2.1320701e-01
      73	 1.2215070e+00	 1.5150869e-01	 1.2749204e+00	 1.6098654e-01	  1.1916500e+00 	 1.9156551e-01


.. parsed-literal::

      74	 1.2323104e+00	 1.5086486e-01	 1.2853604e+00	 1.5958726e-01	[ 1.2043430e+00]	 2.0195651e-01


.. parsed-literal::

      75	 1.2361573e+00	 1.5054723e-01	 1.2892020e+00	 1.5889585e-01	[ 1.2086211e+00]	 2.0812249e-01


.. parsed-literal::

      76	 1.2425904e+00	 1.4983868e-01	 1.2959170e+00	 1.5769714e-01	[ 1.2131730e+00]	 2.0893431e-01


.. parsed-literal::

      77	 1.2502607e+00	 1.4911662e-01	 1.3041803e+00	 1.5656454e-01	[ 1.2142183e+00]	 2.0294476e-01


.. parsed-literal::

      78	 1.2574631e+00	 1.4851120e-01	 1.3118340e+00	 1.5561213e-01	[ 1.2150631e+00]	 2.1345663e-01


.. parsed-literal::

      79	 1.2636247e+00	 1.4835363e-01	 1.3179012e+00	 1.5595126e-01	[ 1.2191313e+00]	 2.0772672e-01


.. parsed-literal::

      80	 1.2703026e+00	 1.4804565e-01	 1.3247079e+00	 1.5624389e-01	[ 1.2227878e+00]	 2.0363903e-01


.. parsed-literal::

      81	 1.2761993e+00	 1.4750612e-01	 1.3305773e+00	 1.5619445e-01	[ 1.2270418e+00]	 2.0342302e-01
      82	 1.2831923e+00	 1.4649392e-01	 1.3377120e+00	 1.5569909e-01	[ 1.2288749e+00]	 1.9854760e-01


.. parsed-literal::

      83	 1.2884292e+00	 1.4586182e-01	 1.3428656e+00	 1.5514582e-01	[ 1.2324751e+00]	 2.0392656e-01


.. parsed-literal::

      84	 1.2949941e+00	 1.4498748e-01	 1.3495043e+00	 1.5422305e-01	[ 1.2358788e+00]	 2.1520185e-01


.. parsed-literal::

      85	 1.2994062e+00	 1.4479786e-01	 1.3541751e+00	 1.5400239e-01	[ 1.2383210e+00]	 2.0394015e-01


.. parsed-literal::

      86	 1.3034784e+00	 1.4461935e-01	 1.3581744e+00	 1.5409354e-01	[ 1.2409623e+00]	 2.0307994e-01
      87	 1.3080140e+00	 1.4464746e-01	 1.3627455e+00	 1.5417960e-01	[ 1.2430477e+00]	 2.0307326e-01


.. parsed-literal::

      88	 1.3127941e+00	 1.4464679e-01	 1.3675326e+00	 1.5405252e-01	[ 1.2466289e+00]	 1.9784570e-01


.. parsed-literal::

      89	 1.3172243e+00	 1.4500545e-01	 1.3723859e+00	 1.5454344e-01	  1.2447868e+00 	 2.1191907e-01


.. parsed-literal::

      90	 1.3234112e+00	 1.4488507e-01	 1.3784294e+00	 1.5374477e-01	[ 1.2507286e+00]	 2.0536566e-01


.. parsed-literal::

      91	 1.3263269e+00	 1.4460394e-01	 1.3813242e+00	 1.5328812e-01	[ 1.2538200e+00]	 2.1018553e-01


.. parsed-literal::

      92	 1.3305352e+00	 1.4406446e-01	 1.3857129e+00	 1.5275728e-01	[ 1.2560648e+00]	 2.1338940e-01


.. parsed-literal::

      93	 1.3342971e+00	 1.4372093e-01	 1.3897259e+00	 1.5242934e-01	[ 1.2604116e+00]	 2.0790315e-01


.. parsed-literal::

      94	 1.3389164e+00	 1.4335949e-01	 1.3944387e+00	 1.5244180e-01	[ 1.2626817e+00]	 2.0830989e-01


.. parsed-literal::

      95	 1.3439420e+00	 1.4316927e-01	 1.3996529e+00	 1.5277209e-01	[ 1.2635443e+00]	 2.0665598e-01
      96	 1.3475376e+00	 1.4307391e-01	 1.4033084e+00	 1.5305779e-01	[ 1.2640467e+00]	 1.7946887e-01


.. parsed-literal::

      97	 1.3506647e+00	 1.4316276e-01	 1.4067262e+00	 1.5367970e-01	[ 1.2641267e+00]	 2.0551634e-01


.. parsed-literal::

      98	 1.3552008e+00	 1.4308295e-01	 1.4110453e+00	 1.5348973e-01	[ 1.2683725e+00]	 2.1160555e-01


.. parsed-literal::

      99	 1.3574469e+00	 1.4289755e-01	 1.4131772e+00	 1.5313383e-01	[ 1.2728298e+00]	 2.0533371e-01
     100	 1.3623954e+00	 1.4228238e-01	 1.4181642e+00	 1.5229379e-01	[ 1.2797494e+00]	 1.8489623e-01


.. parsed-literal::

     101	 1.3673510e+00	 1.4125870e-01	 1.4232356e+00	 1.5135603e-01	[ 1.2872076e+00]	 1.8669248e-01
     102	 1.3720447e+00	 1.4044149e-01	 1.4280458e+00	 1.5031462e-01	[ 1.2909030e+00]	 1.7188311e-01


.. parsed-literal::

     103	 1.3751773e+00	 1.4005820e-01	 1.4312195e+00	 1.5014876e-01	[ 1.2918406e+00]	 2.0858598e-01


.. parsed-literal::

     104	 1.3792860e+00	 1.3940408e-01	 1.4355362e+00	 1.4980666e-01	[ 1.2922829e+00]	 2.0834136e-01
     105	 1.3825439e+00	 1.3871326e-01	 1.4389517e+00	 1.4947319e-01	  1.2918264e+00 	 1.9739556e-01


.. parsed-literal::

     106	 1.3865273e+00	 1.3816745e-01	 1.4430129e+00	 1.4911343e-01	[ 1.2952161e+00]	 2.0632720e-01


.. parsed-literal::

     107	 1.3914024e+00	 1.3770940e-01	 1.4479739e+00	 1.4901544e-01	[ 1.2986971e+00]	 2.1349072e-01


.. parsed-literal::

     108	 1.3936453e+00	 1.3744314e-01	 1.4502633e+00	 1.4879980e-01	[ 1.3025479e+00]	 2.1504331e-01
     109	 1.3963141e+00	 1.3727006e-01	 1.4529081e+00	 1.4877896e-01	[ 1.3040352e+00]	 2.0003438e-01


.. parsed-literal::

     110	 1.4015950e+00	 1.3695088e-01	 1.4582474e+00	 1.4898336e-01	  1.3020891e+00 	 1.9170356e-01


.. parsed-literal::

     111	 1.4051561e+00	 1.3645324e-01	 1.4619376e+00	 1.4861969e-01	  1.3008224e+00 	 2.0668983e-01


.. parsed-literal::

     112	 1.4095488e+00	 1.3578926e-01	 1.4665845e+00	 1.4848693e-01	  1.2909500e+00 	 2.1279645e-01


.. parsed-literal::

     113	 1.4136506e+00	 1.3543269e-01	 1.4706715e+00	 1.4769142e-01	  1.2967799e+00 	 2.0515347e-01


.. parsed-literal::

     114	 1.4161709e+00	 1.3530460e-01	 1.4731568e+00	 1.4739737e-01	  1.3001897e+00 	 2.0977068e-01


.. parsed-literal::

     115	 1.4203121e+00	 1.3473307e-01	 1.4775182e+00	 1.4679518e-01	  1.3012082e+00 	 2.1584225e-01


.. parsed-literal::

     116	 1.4225265e+00	 1.3436500e-01	 1.4798860e+00	 1.4642232e-01	  1.2957433e+00 	 2.1051669e-01
     117	 1.4256383e+00	 1.3424252e-01	 1.4829533e+00	 1.4644672e-01	  1.2979117e+00 	 1.9369507e-01


.. parsed-literal::

     118	 1.4279463e+00	 1.3412122e-01	 1.4852789e+00	 1.4645784e-01	  1.2980393e+00 	 1.9943333e-01
     119	 1.4296905e+00	 1.3402700e-01	 1.4870438e+00	 1.4637469e-01	  1.2983885e+00 	 2.0292950e-01


.. parsed-literal::

     120	 1.4324633e+00	 1.3377529e-01	 1.4898870e+00	 1.4583243e-01	  1.3011444e+00 	 1.8068790e-01
     121	 1.4351927e+00	 1.3352343e-01	 1.4926538e+00	 1.4535227e-01	[ 1.3041251e+00]	 1.9849253e-01


.. parsed-literal::

     122	 1.4381189e+00	 1.3328629e-01	 1.4955808e+00	 1.4478370e-01	[ 1.3074836e+00]	 1.9919610e-01
     123	 1.4412985e+00	 1.3290374e-01	 1.4988195e+00	 1.4422543e-01	[ 1.3101932e+00]	 1.8700647e-01


.. parsed-literal::

     124	 1.4438567e+00	 1.3273856e-01	 1.5014557e+00	 1.4405394e-01	  1.3087364e+00 	 2.0393038e-01
     125	 1.4460391e+00	 1.3259789e-01	 1.5035756e+00	 1.4411160e-01	[ 1.3105543e+00]	 1.9834948e-01


.. parsed-literal::

     126	 1.4487357e+00	 1.3240937e-01	 1.5062512e+00	 1.4432402e-01	[ 1.3110495e+00]	 2.0599389e-01
     127	 1.4515361e+00	 1.3230590e-01	 1.5090618e+00	 1.4441622e-01	[ 1.3133354e+00]	 2.0098591e-01


.. parsed-literal::

     128	 1.4544931e+00	 1.3241181e-01	 1.5122325e+00	 1.4499549e-01	  1.3108491e+00 	 1.9993019e-01


.. parsed-literal::

     129	 1.4580227e+00	 1.3227509e-01	 1.5156744e+00	 1.4459491e-01	[ 1.3168775e+00]	 2.0301199e-01
     130	 1.4596733e+00	 1.3220255e-01	 1.5173258e+00	 1.4431254e-01	[ 1.3182598e+00]	 1.7296648e-01


.. parsed-literal::

     131	 1.4625563e+00	 1.3201066e-01	 1.5203222e+00	 1.4394425e-01	  1.3179823e+00 	 1.9963241e-01


.. parsed-literal::

     132	 1.4631065e+00	 1.3189900e-01	 1.5212260e+00	 1.4372610e-01	  1.3125054e+00 	 2.0673847e-01


.. parsed-literal::

     133	 1.4669796e+00	 1.3172295e-01	 1.5249590e+00	 1.4373017e-01	  1.3136369e+00 	 2.0396924e-01
     134	 1.4683284e+00	 1.3162972e-01	 1.5262851e+00	 1.4378306e-01	  1.3137364e+00 	 1.9906044e-01


.. parsed-literal::

     135	 1.4707897e+00	 1.3145555e-01	 1.5288142e+00	 1.4380657e-01	  1.3117279e+00 	 2.0272112e-01
     136	 1.4731730e+00	 1.3110527e-01	 1.5313472e+00	 1.4346457e-01	  1.3096032e+00 	 1.8380833e-01


.. parsed-literal::

     137	 1.4760920e+00	 1.3097267e-01	 1.5343407e+00	 1.4340002e-01	  1.3061875e+00 	 1.9741344e-01
     138	 1.4783250e+00	 1.3081777e-01	 1.5365809e+00	 1.4315431e-01	  1.3039213e+00 	 1.8927789e-01


.. parsed-literal::

     139	 1.4810667e+00	 1.3047258e-01	 1.5393696e+00	 1.4270537e-01	  1.3016410e+00 	 2.0180702e-01


.. parsed-literal::

     140	 1.4828751e+00	 1.3021464e-01	 1.5413148e+00	 1.4246594e-01	  1.2841010e+00 	 2.1573853e-01


.. parsed-literal::

     141	 1.4852377e+00	 1.3011229e-01	 1.5435744e+00	 1.4245475e-01	  1.2900964e+00 	 2.1027589e-01
     142	 1.4871263e+00	 1.2998948e-01	 1.5454668e+00	 1.4248952e-01	  1.2908816e+00 	 1.8704748e-01


.. parsed-literal::

     143	 1.4889817e+00	 1.2990169e-01	 1.5473748e+00	 1.4255775e-01	  1.2875678e+00 	 2.1115422e-01
     144	 1.4902714e+00	 1.2989088e-01	 1.5489175e+00	 1.4257233e-01	  1.2830154e+00 	 1.9955516e-01


.. parsed-literal::

     145	 1.4933198e+00	 1.2981546e-01	 1.5518923e+00	 1.4261589e-01	  1.2779877e+00 	 1.9759774e-01


.. parsed-literal::

     146	 1.4946042e+00	 1.2981111e-01	 1.5531550e+00	 1.4255088e-01	  1.2782460e+00 	 2.0360732e-01
     147	 1.4965395e+00	 1.2977617e-01	 1.5551615e+00	 1.4244832e-01	  1.2757887e+00 	 1.9873095e-01


.. parsed-literal::

     148	 1.4985995e+00	 1.2960234e-01	 1.5573703e+00	 1.4213012e-01	  1.2729585e+00 	 2.1247172e-01


.. parsed-literal::

     149	 1.5012418e+00	 1.2961075e-01	 1.5600864e+00	 1.4215943e-01	  1.2676856e+00 	 2.0473313e-01


.. parsed-literal::

     150	 1.5029741e+00	 1.2959342e-01	 1.5618006e+00	 1.4216670e-01	  1.2676207e+00 	 2.0645308e-01
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
    Inserting handle into data store.  model: <rail.estimation.algos._gpz_util.GP object at 0x7fd514d33a60>, GPzEstimator
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

