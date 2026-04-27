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
       1	-3.3933696e-01	 3.1936640e-01	-3.2908234e-01	 3.2536605e-01	[-3.4037453e-01]	 4.4895267e-01


.. parsed-literal::

       2	-2.7383677e-01	 3.1105140e-01	-2.5176451e-01	 3.1485148e-01	[-2.6396709e-01]	 2.2334504e-01


.. parsed-literal::

       3	-2.2874452e-01	 2.9058185e-01	-1.8635565e-01	 2.9327753e-01	[-1.9357782e-01]	 2.9608965e-01


.. parsed-literal::

       4	-1.9721327e-01	 2.7474680e-01	-1.5016091e-01	 2.7642545e-01	[-1.5158886e-01]	 2.8755236e-01
       5	-1.3416974e-01	 2.5843317e-01	-9.5963667e-02	 2.5975361e-01	[-9.9760658e-02]	 2.0369291e-01


.. parsed-literal::

       6	-7.2811821e-02	 2.5226770e-01	-4.2281172e-02	 2.5222534e-01	[-4.1351910e-02]	 2.1552348e-01


.. parsed-literal::

       7	-5.3141635e-02	 2.4913905e-01	-2.8646128e-02	 2.5100474e-01	[-3.2147867e-02]	 2.0176387e-01


.. parsed-literal::

       8	-3.8050551e-02	 2.4645178e-01	-1.8557213e-02	 2.4676422e-01	[-1.8226384e-02]	 2.0828009e-01
       9	-3.0388508e-02	 2.4511945e-01	-1.2586297e-02	 2.4582314e-01	[-1.3525380e-02]	 2.0087481e-01


.. parsed-literal::

      10	-1.9317837e-02	 2.4302492e-01	-3.2577220e-03	 2.4438427e-01	[-5.7338652e-03]	 1.9864106e-01


.. parsed-literal::

      11	-1.1288997e-02	 2.4170058e-01	 3.2118384e-03	 2.4298104e-01	[ 8.4799615e-04]	 2.0592570e-01
      12	-6.7413455e-03	 2.4074723e-01	 7.7315902e-03	 2.4276833e-01	[ 3.7983257e-03]	 1.9818974e-01


.. parsed-literal::

      13	-1.9576655e-03	 2.3984888e-01	 1.2755108e-02	 2.4244984e-01	[ 7.9086127e-03]	 1.8973875e-01


.. parsed-literal::

      14	 1.0903978e-01	 2.2609796e-01	 1.2986974e-01	 2.2821257e-01	[ 1.2880359e-01]	 3.1672072e-01


.. parsed-literal::

      15	 1.5203535e-01	 2.2829032e-01	 1.7791752e-01	 2.2461757e-01	[ 1.7438405e-01]	 3.2601953e-01


.. parsed-literal::

      16	 1.9495742e-01	 2.2287745e-01	 2.2116229e-01	 2.2165864e-01	[ 2.1252500e-01]	 2.0373511e-01
      17	 2.8007875e-01	 2.1805343e-01	 3.0975868e-01	 2.1830303e-01	[ 2.8876576e-01]	 1.9615912e-01


.. parsed-literal::

      18	 3.4578518e-01	 2.1306888e-01	 3.7950598e-01	 2.1408487e-01	[ 3.4956431e-01]	 2.1031022e-01
      19	 3.9396068e-01	 2.0935874e-01	 4.2854712e-01	 2.1236349e-01	[ 3.9776568e-01]	 1.9183803e-01


.. parsed-literal::

      20	 4.4840633e-01	 2.0351189e-01	 4.8373650e-01	 2.0540652e-01	[ 4.5283797e-01]	 1.8042445e-01
      21	 5.2246040e-01	 1.9839384e-01	 5.6008035e-01	 1.9927407e-01	[ 5.3151147e-01]	 1.9231319e-01


.. parsed-literal::

      22	 5.6234916e-01	 2.0104475e-01	 6.0602330e-01	 2.0216037e-01	[ 5.8456222e-01]	 1.9645309e-01
      23	 6.0809608e-01	 1.9380546e-01	 6.5081169e-01	 1.9419743e-01	[ 6.1496655e-01]	 1.9299579e-01


.. parsed-literal::

      24	 6.4496403e-01	 1.9157256e-01	 6.8717412e-01	 1.9195156e-01	[ 6.5547447e-01]	 1.9474339e-01
      25	 6.8253785e-01	 2.0060910e-01	 7.2226132e-01	 1.9999824e-01	[ 6.9174906e-01]	 1.9041014e-01


.. parsed-literal::

      26	 7.3628432e-01	 2.0005151e-01	 7.7661967e-01	 1.9881591e-01	[ 7.4180337e-01]	 1.9322157e-01
      27	 7.7231483e-01	 2.0282344e-01	 8.1527471e-01	 2.0240355e-01	[ 7.7505534e-01]	 1.9581914e-01


.. parsed-literal::

      28	 8.0241961e-01	 2.0277045e-01	 8.4516386e-01	 2.0180927e-01	[ 8.0789360e-01]	 1.9777727e-01
      29	 8.3219008e-01	 2.0100671e-01	 8.7474069e-01	 2.0081490e-01	[ 8.3580072e-01]	 1.9779396e-01


.. parsed-literal::

      30	 8.6587223e-01	 1.9477113e-01	 9.0985156e-01	 1.9496413e-01	[ 8.6533179e-01]	 1.9664907e-01
      31	 8.9455531e-01	 1.8936770e-01	 9.3920310e-01	 1.9059673e-01	[ 8.9615100e-01]	 1.9711971e-01


.. parsed-literal::

      32	 9.1255128e-01	 1.8649743e-01	 9.5788102e-01	 1.8793646e-01	[ 9.1319467e-01]	 1.9607663e-01
      33	 9.2581524e-01	 1.8549569e-01	 9.7145703e-01	 1.8605535e-01	[ 9.3140127e-01]	 1.8443704e-01


.. parsed-literal::

      34	 9.4084554e-01	 1.8267889e-01	 9.8721062e-01	 1.8265340e-01	[ 9.4546738e-01]	 2.0490909e-01
      35	 9.6464463e-01	 1.7860619e-01	 1.0113030e+00	 1.7714310e-01	[ 9.7666112e-01]	 1.9291019e-01


.. parsed-literal::

      36	 9.8763032e-01	 1.7494308e-01	 1.0345052e+00	 1.7298987e-01	[ 9.9647810e-01]	 2.0936656e-01
      37	 1.0089273e+00	 1.6986946e-01	 1.0569285e+00	 1.6755074e-01	[ 1.0215329e+00]	 1.6664863e-01


.. parsed-literal::

      38	 1.0251763e+00	 1.6556906e-01	 1.0743799e+00	 1.6281438e-01	[ 1.0349351e+00]	 2.0201683e-01


.. parsed-literal::

      39	 1.0442340e+00	 1.6309280e-01	 1.0946604e+00	 1.6010633e-01	[ 1.0555191e+00]	 2.0948935e-01
      40	 1.0577952e+00	 1.6123996e-01	 1.1090194e+00	 1.5822732e-01	[ 1.0601835e+00]	 1.9915533e-01


.. parsed-literal::

      41	 1.0691084e+00	 1.6021928e-01	 1.1200490e+00	 1.5680159e-01	[ 1.0742706e+00]	 2.1192694e-01


.. parsed-literal::

      42	 1.0834177e+00	 1.5955774e-01	 1.1338880e+00	 1.5613448e-01	[ 1.0930298e+00]	 2.0663762e-01


.. parsed-literal::

      43	 1.0950867e+00	 1.5767685e-01	 1.1456640e+00	 1.5436359e-01	[ 1.0985759e+00]	 2.0683742e-01


.. parsed-literal::

      44	 1.1148256e+00	 1.5431212e-01	 1.1658967e+00	 1.5139553e-01	[ 1.1112549e+00]	 2.0636868e-01
      45	 1.1269352e+00	 1.5237725e-01	 1.1782474e+00	 1.4943184e-01	[ 1.1201213e+00]	 1.7773700e-01


.. parsed-literal::

      46	 1.1378445e+00	 1.5106773e-01	 1.1890012e+00	 1.4750049e-01	[ 1.1297672e+00]	 1.9774556e-01
      47	 1.1500983e+00	 1.5012469e-01	 1.2008324e+00	 1.4608695e-01	[ 1.1414982e+00]	 1.9940925e-01


.. parsed-literal::

      48	 1.1639751e+00	 1.4882356e-01	 1.2147714e+00	 1.4476394e-01	[ 1.1540914e+00]	 1.9968224e-01


.. parsed-literal::

      49	 1.1756880e+00	 1.4714266e-01	 1.2265091e+00	 1.4387947e-01	[ 1.1629466e+00]	 2.0391297e-01
      50	 1.1868007e+00	 1.4616141e-01	 1.2376020e+00	 1.4313474e-01	[ 1.1729495e+00]	 1.9970870e-01


.. parsed-literal::

      51	 1.1927209e+00	 1.4519356e-01	 1.2436858e+00	 1.4251355e-01	[ 1.1793986e+00]	 1.9519973e-01
      52	 1.2012378e+00	 1.4351365e-01	 1.2526097e+00	 1.4116856e-01	[ 1.1870103e+00]	 1.9074988e-01


.. parsed-literal::

      53	 1.2094019e+00	 1.4188815e-01	 1.2612686e+00	 1.4032527e-01	[ 1.1947174e+00]	 2.0831943e-01
      54	 1.2212570e+00	 1.4091838e-01	 1.2730540e+00	 1.3926221e-01	[ 1.2025705e+00]	 2.0016408e-01


.. parsed-literal::

      55	 1.2301776e+00	 1.4022409e-01	 1.2820223e+00	 1.3865587e-01	[ 1.2075450e+00]	 2.0025468e-01
      56	 1.2396958e+00	 1.3888138e-01	 1.2917125e+00	 1.3786217e-01	[ 1.2142029e+00]	 1.9931173e-01


.. parsed-literal::

      57	 1.2485076e+00	 1.3760341e-01	 1.3010266e+00	 1.3752318e-01	[ 1.2183594e+00]	 2.0377588e-01


.. parsed-literal::

      58	 1.2589798e+00	 1.3614369e-01	 1.3115119e+00	 1.3646782e-01	[ 1.2287810e+00]	 2.1707392e-01


.. parsed-literal::

      59	 1.2644791e+00	 1.3598341e-01	 1.3169145e+00	 1.3620671e-01	[ 1.2349159e+00]	 2.0582676e-01


.. parsed-literal::

      60	 1.2729509e+00	 1.3503589e-01	 1.3257483e+00	 1.3560326e-01	[ 1.2382364e+00]	 2.0222044e-01


.. parsed-literal::

      61	 1.2785495e+00	 1.3448967e-01	 1.3316865e+00	 1.3536620e-01	[ 1.2412471e+00]	 2.1674800e-01


.. parsed-literal::

      62	 1.2847012e+00	 1.3391226e-01	 1.3377756e+00	 1.3463618e-01	[ 1.2456224e+00]	 2.0071197e-01
      63	 1.2928430e+00	 1.3290633e-01	 1.3459490e+00	 1.3340553e-01	[ 1.2476047e+00]	 2.0425010e-01


.. parsed-literal::

      64	 1.2991219e+00	 1.3201468e-01	 1.3522757e+00	 1.3249978e-01	[ 1.2490205e+00]	 1.9273019e-01


.. parsed-literal::

      65	 1.3077069e+00	 1.3123745e-01	 1.3609360e+00	 1.3143708e-01	[ 1.2523182e+00]	 2.0465684e-01
      66	 1.3138235e+00	 1.3054391e-01	 1.3671231e+00	 1.3073230e-01	[ 1.2554716e+00]	 1.9614363e-01


.. parsed-literal::

      67	 1.3197790e+00	 1.2995895e-01	 1.3731307e+00	 1.3019104e-01	[ 1.2614524e+00]	 2.0615292e-01
      68	 1.3247739e+00	 1.2951758e-01	 1.3782008e+00	 1.2978046e-01	[ 1.2657418e+00]	 1.9319296e-01


.. parsed-literal::

      69	 1.3318339e+00	 1.2887549e-01	 1.3854930e+00	 1.2900459e-01	[ 1.2677023e+00]	 1.7949533e-01
      70	 1.3366350e+00	 1.2798327e-01	 1.3905362e+00	 1.2811596e-01	  1.2670513e+00 	 1.8500543e-01


.. parsed-literal::

      71	 1.3423952e+00	 1.2788050e-01	 1.3961808e+00	 1.2776621e-01	[ 1.2706291e+00]	 2.0496130e-01
      72	 1.3479245e+00	 1.2762048e-01	 1.4017226e+00	 1.2715610e-01	[ 1.2727330e+00]	 1.8248558e-01


.. parsed-literal::

      73	 1.3529368e+00	 1.2730009e-01	 1.4068738e+00	 1.2659041e-01	[ 1.2737684e+00]	 1.8480873e-01


.. parsed-literal::

      74	 1.3630365e+00	 1.2664468e-01	 1.4174084e+00	 1.2547424e-01	[ 1.2762625e+00]	 2.0772409e-01
      75	 1.3685242e+00	 1.2619304e-01	 1.4232023e+00	 1.2484549e-01	[ 1.2764604e+00]	 2.0129752e-01


.. parsed-literal::

      76	 1.3740044e+00	 1.2592253e-01	 1.4284472e+00	 1.2452650e-01	[ 1.2841663e+00]	 1.9769526e-01


.. parsed-literal::

      77	 1.3775953e+00	 1.2569559e-01	 1.4320140e+00	 1.2424092e-01	[ 1.2882559e+00]	 2.1324611e-01
      78	 1.3831377e+00	 1.2518828e-01	 1.4376826e+00	 1.2350252e-01	[ 1.2924448e+00]	 1.9953322e-01


.. parsed-literal::

      79	 1.3860026e+00	 1.2515995e-01	 1.4407991e+00	 1.2308051e-01	[ 1.2935497e+00]	 2.0450568e-01


.. parsed-literal::

      80	 1.3911316e+00	 1.2475400e-01	 1.4458526e+00	 1.2255436e-01	[ 1.2978727e+00]	 2.0425010e-01
      81	 1.3937152e+00	 1.2456927e-01	 1.4485245e+00	 1.2223370e-01	  1.2977781e+00 	 1.9967580e-01


.. parsed-literal::

      82	 1.3970220e+00	 1.2438367e-01	 1.4519200e+00	 1.2188914e-01	[ 1.2992208e+00]	 1.9778728e-01


.. parsed-literal::

      83	 1.4015459e+00	 1.2436076e-01	 1.4566719e+00	 1.2162983e-01	[ 1.2995720e+00]	 2.1076035e-01
      84	 1.4058437e+00	 1.2430726e-01	 1.4610052e+00	 1.2140072e-01	[ 1.3056275e+00]	 1.7809653e-01


.. parsed-literal::

      85	 1.4096634e+00	 1.2425071e-01	 1.4648351e+00	 1.2139858e-01	[ 1.3080974e+00]	 2.1676874e-01


.. parsed-literal::

      86	 1.4130430e+00	 1.2420770e-01	 1.4684272e+00	 1.2132051e-01	[ 1.3088757e+00]	 2.0408392e-01


.. parsed-literal::

      87	 1.4167221e+00	 1.2414721e-01	 1.4720233e+00	 1.2148726e-01	[ 1.3092415e+00]	 2.1775889e-01


.. parsed-literal::

      88	 1.4200946e+00	 1.2416270e-01	 1.4754347e+00	 1.2150561e-01	  1.3082983e+00 	 2.1151686e-01


.. parsed-literal::

      89	 1.4241781e+00	 1.2417942e-01	 1.4795999e+00	 1.2141438e-01	  1.3086466e+00 	 2.1223545e-01


.. parsed-literal::

      90	 1.4289382e+00	 1.2443327e-01	 1.4844486e+00	 1.2132878e-01	[ 1.3102343e+00]	 2.1200395e-01


.. parsed-literal::

      91	 1.4330098e+00	 1.2439741e-01	 1.4885226e+00	 1.2104511e-01	[ 1.3162914e+00]	 2.1777868e-01


.. parsed-literal::

      92	 1.4360718e+00	 1.2412912e-01	 1.4914232e+00	 1.2079216e-01	[ 1.3239586e+00]	 2.1001434e-01


.. parsed-literal::

      93	 1.4399097e+00	 1.2380491e-01	 1.4952123e+00	 1.2037850e-01	[ 1.3317138e+00]	 2.1598244e-01
      94	 1.4426872e+00	 1.2344654e-01	 1.4980941e+00	 1.2031080e-01	[ 1.3365062e+00]	 2.0467687e-01


.. parsed-literal::

      95	 1.4459064e+00	 1.2319623e-01	 1.5013237e+00	 1.2010506e-01	[ 1.3374363e+00]	 2.1555352e-01


.. parsed-literal::

      96	 1.4479791e+00	 1.2312817e-01	 1.5034119e+00	 1.2019574e-01	  1.3342550e+00 	 2.0891809e-01


.. parsed-literal::

      97	 1.4499398e+00	 1.2304835e-01	 1.5054140e+00	 1.2040789e-01	  1.3300040e+00 	 2.1000147e-01
      98	 1.4531351e+00	 1.2287307e-01	 1.5086431e+00	 1.2063328e-01	  1.3272384e+00 	 1.8561959e-01


.. parsed-literal::

      99	 1.4565807e+00	 1.2247710e-01	 1.5121507e+00	 1.2076537e-01	  1.3242583e+00 	 2.0681262e-01


.. parsed-literal::

     100	 1.4589357e+00	 1.2225249e-01	 1.5144786e+00	 1.2057674e-01	  1.3288503e+00 	 2.0704436e-01


.. parsed-literal::

     101	 1.4608919e+00	 1.2208760e-01	 1.5164275e+00	 1.2015808e-01	  1.3335598e+00 	 2.1269941e-01


.. parsed-literal::

     102	 1.4631542e+00	 1.2188896e-01	 1.5187420e+00	 1.1974074e-01	  1.3364218e+00 	 2.1574020e-01


.. parsed-literal::

     103	 1.4665850e+00	 1.2160335e-01	 1.5223823e+00	 1.1920903e-01	  1.3350617e+00 	 2.0983839e-01


.. parsed-literal::

     104	 1.4689256e+00	 1.2135870e-01	 1.5249523e+00	 1.1890969e-01	  1.3285843e+00 	 2.0232463e-01
     105	 1.4712330e+00	 1.2125183e-01	 1.5272159e+00	 1.1902137e-01	  1.3279109e+00 	 1.8705988e-01


.. parsed-literal::

     106	 1.4733085e+00	 1.2110688e-01	 1.5292904e+00	 1.1925734e-01	  1.3257837e+00 	 1.8825650e-01
     107	 1.4753617e+00	 1.2086780e-01	 1.5313985e+00	 1.1932039e-01	  1.3240361e+00 	 1.9789767e-01


.. parsed-literal::

     108	 1.4782370e+00	 1.2057891e-01	 1.5343292e+00	 1.1929083e-01	  1.3254427e+00 	 2.0104599e-01


.. parsed-literal::

     109	 1.4808933e+00	 1.2029866e-01	 1.5370097e+00	 1.1903104e-01	  1.3292814e+00 	 2.0571351e-01
     110	 1.4839657e+00	 1.1997630e-01	 1.5401384e+00	 1.1858023e-01	  1.3320552e+00 	 1.8750691e-01


.. parsed-literal::

     111	 1.4857307e+00	 1.1980436e-01	 1.5419802e+00	 1.1839201e-01	  1.3335995e+00 	 2.0784044e-01


.. parsed-literal::

     112	 1.4871928e+00	 1.1977336e-01	 1.5434089e+00	 1.1836837e-01	  1.3327384e+00 	 2.0623851e-01


.. parsed-literal::

     113	 1.4887074e+00	 1.1972661e-01	 1.5449354e+00	 1.1840334e-01	  1.3309130e+00 	 2.1198869e-01


.. parsed-literal::

     114	 1.4901588e+00	 1.1963946e-01	 1.5464392e+00	 1.1844077e-01	  1.3291461e+00 	 2.0549536e-01
     115	 1.4918779e+00	 1.1952764e-01	 1.5483059e+00	 1.1864251e-01	  1.3272591e+00 	 1.9344497e-01


.. parsed-literal::

     116	 1.4941713e+00	 1.1941294e-01	 1.5506060e+00	 1.1875794e-01	  1.3261762e+00 	 2.0454192e-01


.. parsed-literal::

     117	 1.4953206e+00	 1.1937129e-01	 1.5517218e+00	 1.1880188e-01	  1.3279935e+00 	 2.0280147e-01


.. parsed-literal::

     118	 1.4976025e+00	 1.1925463e-01	 1.5539892e+00	 1.1896192e-01	  1.3287684e+00 	 2.0500612e-01
     119	 1.4992156e+00	 1.1910785e-01	 1.5556222e+00	 1.1903411e-01	  1.3279123e+00 	 2.0010662e-01


.. parsed-literal::

     120	 1.5011627e+00	 1.1899900e-01	 1.5575573e+00	 1.1901843e-01	  1.3265713e+00 	 2.1044564e-01
     121	 1.5021043e+00	 1.1892885e-01	 1.5584930e+00	 1.1883862e-01	  1.3262664e+00 	 1.9719505e-01


.. parsed-literal::

     122	 1.5037793e+00	 1.1877412e-01	 1.5601957e+00	 1.1858532e-01	  1.3250774e+00 	 1.8481565e-01


.. parsed-literal::

     123	 1.5061108e+00	 1.1857220e-01	 1.5625887e+00	 1.1820657e-01	  1.3257098e+00 	 2.0234728e-01


.. parsed-literal::

     124	 1.5072313e+00	 1.1845661e-01	 1.5637867e+00	 1.1804766e-01	  1.3247295e+00 	 3.1654882e-01
     125	 1.5093479e+00	 1.1836703e-01	 1.5659761e+00	 1.1791444e-01	  1.3271337e+00 	 2.0022249e-01


.. parsed-literal::

     126	 1.5111504e+00	 1.1830529e-01	 1.5678547e+00	 1.1787085e-01	  1.3294071e+00 	 1.7798615e-01
     127	 1.5128309e+00	 1.1824263e-01	 1.5696378e+00	 1.1783172e-01	  1.3316669e+00 	 1.9335818e-01


.. parsed-literal::

     128	 1.5135081e+00	 1.1818651e-01	 1.5704354e+00	 1.1775854e-01	  1.3303180e+00 	 1.9416690e-01
     129	 1.5148682e+00	 1.1815434e-01	 1.5716986e+00	 1.1771748e-01	  1.3320051e+00 	 1.9000506e-01


.. parsed-literal::

     130	 1.5155742e+00	 1.1811517e-01	 1.5723843e+00	 1.1766983e-01	  1.3314676e+00 	 1.8626738e-01
     131	 1.5163348e+00	 1.1806440e-01	 1.5731369e+00	 1.1763958e-01	  1.3307724e+00 	 2.0140290e-01


.. parsed-literal::

     132	 1.5175986e+00	 1.1794711e-01	 1.5744019e+00	 1.1763167e-01	  1.3292110e+00 	 2.1820211e-01


.. parsed-literal::

     133	 1.5190612e+00	 1.1786298e-01	 1.5758713e+00	 1.1770324e-01	  1.3297728e+00 	 2.0467234e-01
     134	 1.5202491e+00	 1.1782081e-01	 1.5770537e+00	 1.1775385e-01	  1.3310906e+00 	 1.9624996e-01


.. parsed-literal::

     135	 1.5218563e+00	 1.1775664e-01	 1.5786713e+00	 1.1783774e-01	  1.3334057e+00 	 1.9445491e-01
     136	 1.5232026e+00	 1.1767222e-01	 1.5800450e+00	 1.1786983e-01	  1.3333278e+00 	 1.8822026e-01


.. parsed-literal::

     137	 1.5247169e+00	 1.1747380e-01	 1.5816965e+00	 1.1771682e-01	  1.3305350e+00 	 1.9446945e-01
     138	 1.5268491e+00	 1.1734500e-01	 1.5838097e+00	 1.1774602e-01	  1.3298457e+00 	 1.9314742e-01


.. parsed-literal::

     139	 1.5276195e+00	 1.1730096e-01	 1.5845615e+00	 1.1768015e-01	  1.3296045e+00 	 2.0291162e-01


.. parsed-literal::

     140	 1.5287076e+00	 1.1716538e-01	 1.5857123e+00	 1.1761661e-01	  1.3276579e+00 	 2.0634627e-01


.. parsed-literal::

     141	 1.5296766e+00	 1.1705351e-01	 1.5867263e+00	 1.1751651e-01	  1.3249083e+00 	 2.0544076e-01
     142	 1.5304068e+00	 1.1702693e-01	 1.5874632e+00	 1.1755698e-01	  1.3255385e+00 	 2.0207763e-01


.. parsed-literal::

     143	 1.5318842e+00	 1.1693678e-01	 1.5890139e+00	 1.1766366e-01	  1.3244025e+00 	 1.9319963e-01
     144	 1.5330773e+00	 1.1685371e-01	 1.5902632e+00	 1.1767099e-01	  1.3212175e+00 	 1.8834472e-01


.. parsed-literal::

     145	 1.5337484e+00	 1.1676440e-01	 1.5911383e+00	 1.1781568e-01	  1.3133376e+00 	 2.0049143e-01
     146	 1.5353976e+00	 1.1672326e-01	 1.5927020e+00	 1.1769934e-01	  1.3129847e+00 	 1.9566417e-01


.. parsed-literal::

     147	 1.5358330e+00	 1.1670414e-01	 1.5931133e+00	 1.1764489e-01	  1.3131051e+00 	 1.9276905e-01
     148	 1.5370177e+00	 1.1665261e-01	 1.5942884e+00	 1.1764795e-01	  1.3126563e+00 	 1.9368410e-01


.. parsed-literal::

     149	 1.5384737e+00	 1.1653754e-01	 1.5957370e+00	 1.1764089e-01	  1.3125193e+00 	 2.0190668e-01


.. parsed-literal::

     150	 1.5397179e+00	 1.1653372e-01	 1.5969611e+00	 1.1771698e-01	  1.3140649e+00 	 2.0785213e-01
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
    Inserting handle into data store.  model: <rail.estimation.algos._gpz_util.GP object at 0x7f1f9009a500>, GPzEstimator
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

