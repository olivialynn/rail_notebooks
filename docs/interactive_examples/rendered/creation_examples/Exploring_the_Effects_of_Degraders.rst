Exploring the Effects of Different Degraders on Estimated Redshifts
===================================================================

**Authors:** Jennifer Scora, Mubdi Rahman

**Last run successfully:** Feb 9, 2026

Thanks to Matteo Moretti and Biprateep Dey for inspiring the use case.

In this notebook, we’ll explore how to create simulated datasets with
the `RAIL creation
stage <https://rail-hub.readthedocs.io/en/latest/source/rail_stages/creation.html>`__,
in particular focusing on how data sets created using different
degradation algorithms can affect the calibration of models to estimate
photometric redshifts (photo-zs). Here “degradation” algorithms refer to
any algorithms applied to alter the “true” sample, for example to add
biases or cuts.

Here are the main steps we’ll be following:

1. Simulating galaxies with photometric data and redshifts
2. “Degrading” photometry and redshift information to create different
   calibration data
3. Calibrating the photometric redshift algorithms with the differently
   degraded data
4. Estimating the photometric redshifts of a set of target galaxies
   using the calibrated models
5. Seeing how the algorithm calibration affected the output redshift
   distributions

1. Simulating galaxies with photometric data and redshifts
----------------------------------------------------------

In this step we want to create the data sets of galaxy magnitudes and
corresponding redshifts that we will use to calibrate and estimate
photometric redshifts. We use the `PZflow
algorithm <https://rail-hub.readthedocs.io/en/latarget/source/rail_stages/creation.html#pzflow-engine>`__
to generate our model, which is a machine learning package that we’re
going to use in this context to model galaxies. Then we sample two data
sets from the model, a calibration dataset and a target dataset. The
calibration data set will be used to calibrate our models, and the
target data set is the data we will get photo-z estimates for. These
data sets will be considered our “true” data, which means they contain
the “real” redshifts before we have made any alterations to make the
data more realistic.

Set up
~~~~~~

Let’s start by importing the packages we’ll need to create and analyze
the data sets.

.. code:: ipython3

    import rail.interactive as ri
    import numpy as np
    from pzflow.examples import get_galaxy_data
    
    # for plotting
    import matplotlib.pyplot as plt
    
    %matplotlib inline


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
    /home/runner/.cache/lephare/runs/20260326T201727


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
      File "/tmp/ipykernel_5148/1847479680.py", line 1, in <module>
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


We need to set up some column name dictionaries, as the expected column
names vary between some of the codes. In order to handle this, we can
pass in dictionaries of expected column names and the column name that
exists in the input data (``band_dict`` and ``rename_dict`` below). In
this notebook, we are using bands ugrizy, and each band will have a name
‘mag_u_lsst’, for example, with the error column name being
‘mag_err_u_lsst’.

The initial data we pull from our model won’t have any associated
errors. Those will be created when we degrade the datasets, but the
error columns will need to be renamed with the ``rename_dict`` later on.

.. code:: ipython3

    bands = ["u", "g", "r", "i", "z", "y"]
    band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"mag_{band}_lsst_err": f"mag_err_{band}_lsst" for band in bands}

In order to generate the model with PZflow, we need to grab some sample
data to base the model off of. This sample data is only used to create
the model, and is seperate from the calibration and target data we’ll
get from the model later. We’ll rename the band columns in this data
table to match our desired band names as discussed above, using
``band_dict``. We can check that our columns have been renamed
appropriately by printing out the first few lines of the table:

.. code:: ipython3

    catalog = get_galaxy_data().rename(band_dict, axis=1)
    # let's take a look at the columns
    catalog.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>mag_u_lsst</th>
          <th>mag_g_lsst</th>
          <th>mag_r_lsst</th>
          <th>mag_i_lsst</th>
          <th>mag_z_lsst</th>
          <th>mag_y_lsst</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.287087</td>
          <td>26.759261</td>
          <td>25.901778</td>
          <td>25.187710</td>
          <td>24.932318</td>
          <td>24.736903</td>
          <td>24.671623</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.293313</td>
          <td>27.428358</td>
          <td>26.679299</td>
          <td>25.977161</td>
          <td>25.700094</td>
          <td>25.522763</td>
          <td>25.417632</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.497276</td>
          <td>27.294001</td>
          <td>26.068798</td>
          <td>25.450055</td>
          <td>24.460507</td>
          <td>23.887221</td>
          <td>23.206112</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.283310</td>
          <td>28.154075</td>
          <td>26.283166</td>
          <td>24.599570</td>
          <td>23.723491</td>
          <td>23.214108</td>
          <td>22.860012</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.545183</td>
          <td>29.276065</td>
          <td>27.878301</td>
          <td>27.333528</td>
          <td>26.543374</td>
          <td>26.061941</td>
          <td>25.383056</td>
        </tr>
      </tbody>
    </table>
    </div>



Looks like the column names are the way we want them!

Calibrate and sample the model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we need to use the galaxy data we retrieved to calibrate the model
that we’ll use to create our input galaxy magnitude data catalogues
later. We’re going to use the ``PZflow`` engine to do this, specifically
the ``modeler`` function. This will train the normalizing flow that
serves as the engine for the input data creation. To get a sense of what
it does and the parameters it needs, let’s check out its docstrings:

.. code:: ipython3

    ri.creation.engines.flowEngine.flow_modeler?

We’ll pass the modeler a few parameters: - **input_data:** this is the
input catalog that our modeler needs to train the data flow (the one we
retrieved above) - **seed (optional):** this is the random seed used for
training - **phys_cols (optional):** The names of any non-photometry
columns and their [min,max] values. - **phot_cols (optional):** This is
a dictionary of the names of the photometry columns and their
corresponding [min,max] values. - **calc_colors (optional):** Whether to
internally calculate colors (if phot_cols are magnitudes). Assumes that
you want to calculate colors from adjacent columns in phot_cols. If you
do not want to calculate colors, set False. Else, provide a dictionary
``{‘ref_column_name’: band}``, where band is a string corresponding to
the column in phot_cols you want to save as the overall galaxy
magnitude. We’re passing in the default value here just so you can see
how it works. - **num_training_epochs (optional):** By default 30, here
we’re doing fewer so that it doesn’t take as long.

**NOTE:** This calibration may take a while depending on your setup.

.. code:: ipython3

    flow_model = ri.creation.engines.flowEngine.flow_modeler(
        input_data=catalog,
        seed=0,
        phys_cols={"redshift": [0, 3]},
        phot_cols={
            "mag_u_lsst": [17, 35],
            "mag_g_lsst": [16, 32],
            "mag_r_lsst": [15, 30],
            "mag_i_lsst": [15, 30],
            "mag_z_lsst": [14, 29],
            "mag_y_lsst": [14, 28],
        },
        calc_colors={"ref_column_name": "mag_i_lsst"},
        num_training_epochs=10,
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, FlowModeler


.. parsed-literal::

    Training 30 epochs 
    Loss:


.. parsed-literal::

    (0) 17.6137


.. parsed-literal::

    (1) 2.3274


.. parsed-literal::

    (2) 0.2876


.. parsed-literal::

    (3) -0.0272


.. parsed-literal::

    (4) -0.1473


.. parsed-literal::

    (5) -2.1294


.. parsed-literal::

    (6) -1.7337


.. parsed-literal::

    (7) -1.5389


.. parsed-literal::

    (8) -2.2590


.. parsed-literal::

    (9) -1.9952


.. parsed-literal::

    (10) -3.0617


.. parsed-literal::

    (11) -3.3305


.. parsed-literal::

    (12) -2.5602


.. parsed-literal::

    (13) -3.1145


.. parsed-literal::

    (14) -2.3787


.. parsed-literal::

    (15) -3.8322


.. parsed-literal::

    (16) -3.4641


.. parsed-literal::

    (17) -3.1314


.. parsed-literal::

    (18) -3.6828


.. parsed-literal::

    (19) -2.9029


.. parsed-literal::

    (20) -3.5720


.. parsed-literal::

    (21) -4.0345


.. parsed-literal::

    (22) -4.3882


.. parsed-literal::

    (23) -4.5509


.. parsed-literal::

    (24) -3.9286


.. parsed-literal::

    (25) -3.7284


.. parsed-literal::

    (26) -4.3904


.. parsed-literal::

    (27) -4.3243


.. parsed-literal::

    (28) -4.7942


.. parsed-literal::

    (29) -4.7405


.. parsed-literal::

    (30) -4.7778


.. parsed-literal::

    Inserting handle into data store.  model: inprogress_model.pkl, FlowModeler


Now we’ll use the flow to produce some synthetic data for our
calibration data set and target data set. Since this is a test we’ll
create some small datasets, with 600 galaxies for this sample, so we’ll
pass in the argument: ``n_samples = 600``. We’ll also use a specific
seed for each one to ensure they’re reproducible but different from each
other.

**Note that when we pass the model to this function, we don’t pass the
dictionary, but the actual model object. This is true of all the
interactive functions.**

.. code:: ipython3

    # get sample calibration and target data sets
    calib_data_orig = ri.creation.engines.flowEngine.flow_creator(
        n_samples=600, model=flow_model["model"], seed=1235
    )
    targ_data_orig = ri.creation.engines.flowEngine.flow_creator(
        model=flow_model["model"], n_samples=600, seed=1234
    )


.. parsed-literal::

    Inserting handle into data store.  model: <pzflow.flow.Flow object at 0x7f1a019015a0>, FlowCreator


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.pq, FlowCreator
    Inserting handle into data store.  model: <pzflow.flow.Flow object at 0x7f1a019015a0>, FlowCreator
    Inserting handle into data store.  output: inprogress_output.pq, FlowCreator


Let’s plot these data sets to check that they are in fact different:

.. code:: ipython3

    hist_options = {"bins": np.linspace(0, 3, 30), "histtype": "stepfilled", "alpha": 0.5}
    
    plt.hist(calib_data_orig["output"]["redshift"], label="calibration", **hist_options)
    plt.hist(targ_data_orig["output"]["redshift"], label="target", **hist_options)
    plt.legend(loc="best")
    plt.xlabel("redshift")
    plt.ylabel("number of galaxies")




.. parsed-literal::

    Text(0, 0.5, 'number of galaxies')




.. image:: Exploring_the_Effects_of_Degraders_files/Exploring_the_Effects_of_Degraders_17_1.png


2. “Degrading” photometry and redshift information to create different calibration data
---------------------------------------------------------------------------------------

The goal of this step is to create a bunch of realistic galaxy
observations that have been degraded in a variety of ways that we’re
going to use as calibration sets for our favourite photometric redshift
algorithm, and to degrade one target data set we want to use to get
estimated redshifts.

So in this step, we’re going to create four different calibration data
sets, where each data set has had one more degrader applied. Thus, the
fourth data has all four degraders applied, while the first only has one
applied. We’ll also create a set of target data will all of the same
four degradations applied, such that the target data should most closely
resemble the most degraded calibration data set.

The degraders we’ll be using, in order, are:

1. ``lsst_error_model`` to add photometric errors that are modelled
   based on the Vera Rubin telescope
2. ``inv_redshift_incompleteness`` to mimic redshift dependent
   incompleteness
3. ``line_confusion`` to simulate the effect of misidentified lines
4. ``quantity_cut`` mimics a band-dependent brightness cut

1. LSST Error Model
~~~~~~~~~~~~~~~~~~~

This method adds photometric errors, non-detections and extended source
errors that are modelled based on the Vera Rubin telescope. We’re going
to apply it to both calibration and target data sets. Once again, we’re
supplying different seeds to ensure the results are reproducible and
different from each other. We need to supply the ``band_dict`` we
created earlier, which tells the code what the band column names should
be. We are also supplying ``ndFlag=np.nan``, which just tells the code
to make non-detections ``np.nan`` in the output.

.. code:: ipython3

    # calibration data
    calib_data_photerrs = ri.creation.degraders.photometric_errors.lsst_error_model(
        sample=calib_data_orig["output"], seed=66, renameDict=band_dict, ndFlag=np.nan
    )
    
    # target data set
    targ_data_photerrs = ri.creation.degraders.photometric_errors.lsst_error_model(
        sample=targ_data_orig["output"], seed=66, renameDict=band_dict, ndFlag=np.nan
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, LSSTErrorModel
    Inserting handle into data store.  output: inprogress_output.pq, LSSTErrorModel
    Inserting handle into data store.  input: None, LSSTErrorModel
    Inserting handle into data store.  output: inprogress_output.pq, LSSTErrorModel


.. code:: ipython3

    # let's see what the output looks like
    calib_data_photerrs["output"].head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>mag_u_lsst</th>
          <th>mag_u_lsst_err</th>
          <th>mag_g_lsst</th>
          <th>mag_g_lsst_err</th>
          <th>mag_r_lsst</th>
          <th>mag_r_lsst_err</th>
          <th>mag_i_lsst</th>
          <th>mag_i_lsst_err</th>
          <th>mag_z_lsst</th>
          <th>mag_z_lsst_err</th>
          <th>mag_y_lsst</th>
          <th>mag_y_lsst_err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.433343</td>
          <td>27.446394</td>
          <td>0.748055</td>
          <td>28.211738</td>
          <td>0.548112</td>
          <td>27.708995</td>
          <td>0.335556</td>
          <td>27.169004</td>
          <td>0.339186</td>
          <td>27.116238</td>
          <td>0.567046</td>
          <td>27.045358</td>
          <td>0.993960</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.323755</td>
          <td>26.091792</td>
          <td>0.272504</td>
          <td>25.932552</td>
          <td>0.084444</td>
          <td>25.796621</td>
          <td>0.065864</td>
          <td>25.265092</td>
          <td>0.067161</td>
          <td>24.815723</td>
          <td>0.086288</td>
          <td>24.249732</td>
          <td>0.117569</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.870260</td>
          <td>26.597977</td>
          <td>0.406650</td>
          <td>26.663111</td>
          <td>0.159298</td>
          <td>24.870533</td>
          <td>0.029014</td>
          <td>23.714600</td>
          <td>0.017316</td>
          <td>22.848013</td>
          <td>0.015531</td>
          <td>22.493640</td>
          <td>0.025000</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.793983</td>
          <td>29.076769</td>
          <td>1.840873</td>
          <td>27.276723</td>
          <td>0.266134</td>
          <td>27.889978</td>
          <td>0.386658</td>
          <td>27.504056</td>
          <td>0.439639</td>
          <td>26.477421</td>
          <td>0.350413</td>
          <td>25.714324</td>
          <td>0.395747</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.294506</td>
          <td>25.819838</td>
          <td>0.217892</td>
          <td>25.835736</td>
          <td>0.077541</td>
          <td>25.459463</td>
          <td>0.048832</td>
          <td>24.792595</td>
          <td>0.044158</td>
          <td>24.228459</td>
          <td>0.051305</td>
          <td>23.734367</td>
          <td>0.074790</td>
        </tr>
      </tbody>
    </table>
    </div>



You can see that error columns have been added in for each of the
magnitude columns.

Now let’s take a look at what’s happened to the magnitudes. Below we’ll
plot the u-band magnitudes before and after running the degrader. You
can see that the higher magnitude objects now have a much wider variance
in magnitude compared to their initial magnitudes, but at lower
magnitudes they’ve remained similar:

.. code:: ipython3

    # we have to set the range because there are nans in the new dataset with errors, which messes up plt.hist2d
    range = [
        [
            np.min(calib_data_orig["output"]["mag_u_lsst"]),
            np.max(calib_data_orig["output"]["mag_u_lsst"]),
        ],
        [
            np.min(calib_data_photerrs["output"]["mag_u_lsst"]),
            np.max(calib_data_photerrs["output"]["mag_u_lsst"]),
        ],
    ]
    plt.hist2d(
        calib_data_orig["output"]["mag_u_lsst"],
        calib_data_photerrs["output"]["mag_u_lsst"],
        range=range,
        bins=20,
        cmap="viridis",
    )
    plt.xlabel("original u-band magnitude")
    plt.ylabel("new u-band magnitude")
    plt.colorbar(label="number of galaxies")




.. parsed-literal::

    <matplotlib.colorbar.Colorbar at 0x7f1987f305b0>




.. image:: Exploring_the_Effects_of_Degraders_files/Exploring_the_Effects_of_Degraders_23_1.png


You can make this plot for all the other magnitudes if you’d like.

2. Redshift Incompleteness
~~~~~~~~~~~~~~~~~~~~~~~~~~

This method applies a selection function, which keeps galaxies with
probability

:math:`p_{\text{keep}}(z) = \min(1, \frac{z_p}{z})`,

where :math:`z_p` is the ‘’pivot’’ redshift. We’ll use
:math:`z_p = 1.0`.

**NOTE**:

As you’ll see later with the evaluators, they’ll require the samples
that we want to compare to be the same length. But if you’ve removed
galaxies due to incompleteness, they won’t inherently be the same
length. So instead, what we’re going to do is flag those galaxies that
are removed.

To do this, we can use the parameter ``drop_rows=False``. This will
return a data table of the same length as before, with a “flag” column
that identifies which galaxies are to be kept, and which are to be
dropped.

.. code:: ipython3

    # calibration data set
    calib_data_inc = (
        ri.creation.degraders.spectroscopic_degraders.inv_redshift_incompleteness(
            sample=calib_data_photerrs["output"], pivot_redshift=1.0
        )
    )
    
    # target data set - use drop_rows to ensure it's the same length
    targ_data_inc = (
        ri.creation.degraders.spectroscopic_degraders.inv_redshift_incompleteness(
            sample=targ_data_photerrs["output"], pivot_redshift=1.0, drop_rows=False
        )
    )
    targ_data_inc["output"]  # look at the output


.. parsed-literal::

    Inserting handle into data store.  input: None, InvRedshiftIncompleteness
    Inserting handle into data store.  output: inprogress_output.pq, InvRedshiftIncompleteness
    Inserting handle into data store.  input: None, InvRedshiftIncompleteness
    Inserting handle into data store.  output: inprogress_output.pq, InvRedshiftIncompleteness




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>flag</th>
          <th>redshift</th>
          <th>mag_u_lsst</th>
          <th>mag_u_lsst_err</th>
          <th>mag_g_lsst</th>
          <th>mag_g_lsst_err</th>
          <th>mag_r_lsst</th>
          <th>mag_r_lsst_err</th>
          <th>mag_i_lsst</th>
          <th>mag_i_lsst_err</th>
          <th>mag_z_lsst</th>
          <th>mag_z_lsst_err</th>
          <th>mag_y_lsst</th>
          <th>mag_y_lsst_err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>True</td>
          <td>0.710979</td>
          <td>26.762862</td>
          <td>0.460818</td>
          <td>26.802089</td>
          <td>0.179290</td>
          <td>26.148865</td>
          <td>0.089899</td>
          <td>25.530220</td>
          <td>0.084893</td>
          <td>25.525601</td>
          <td>0.159922</td>
          <td>25.866070</td>
          <td>0.444362</td>
        </tr>
        <tr>
          <th>1</th>
          <td>True</td>
          <td>0.684234</td>
          <td>23.210656</td>
          <td>0.022756</td>
          <td>22.488883</td>
          <td>0.006382</td>
          <td>21.626449</td>
          <td>0.005261</td>
          <td>20.815851</td>
          <td>0.005173</td>
          <td>20.562635</td>
          <td>0.005376</td>
          <td>20.348689</td>
          <td>0.006163</td>
        </tr>
        <tr>
          <th>2</th>
          <td>True</td>
          <td>0.249871</td>
          <td>27.112190</td>
          <td>0.594565</td>
          <td>28.039316</td>
          <td>0.483038</td>
          <td>26.920252</td>
          <td>0.175355</td>
          <td>26.965811</td>
          <td>0.288325</td>
          <td>27.001835</td>
          <td>0.521963</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>True</td>
          <td>1.751949</td>
          <td>27.919353</td>
          <td>1.008960</td>
          <td>25.954472</td>
          <td>0.086088</td>
          <td>25.496445</td>
          <td>0.050463</td>
          <td>24.900504</td>
          <td>0.048597</td>
          <td>24.270590</td>
          <td>0.053261</td>
          <td>23.881404</td>
          <td>0.085152</td>
        </tr>
        <tr>
          <th>4</th>
          <td>True</td>
          <td>0.478461</td>
          <td>27.142777</td>
          <td>0.607556</td>
          <td>27.380647</td>
          <td>0.289556</td>
          <td>25.807922</td>
          <td>0.066527</td>
          <td>25.244399</td>
          <td>0.065941</td>
          <td>24.948522</td>
          <td>0.096970</td>
          <td>25.168788</td>
          <td>0.256251</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>595</th>
          <td>True</td>
          <td>0.577203</td>
          <td>28.865338</td>
          <td>1.671480</td>
          <td>26.842217</td>
          <td>0.185480</td>
          <td>25.729843</td>
          <td>0.062078</td>
          <td>25.313037</td>
          <td>0.070074</td>
          <td>24.948450</td>
          <td>0.096964</td>
          <td>25.564396</td>
          <td>0.352132</td>
        </tr>
        <tr>
          <th>596</th>
          <td>True</td>
          <td>0.405416</td>
          <td>25.636360</td>
          <td>0.186861</td>
          <td>25.907552</td>
          <td>0.082606</td>
          <td>25.873413</td>
          <td>0.070499</td>
          <td>25.953150</td>
          <td>0.122920</td>
          <td>25.977916</td>
          <td>0.234033</td>
          <td>25.859538</td>
          <td>0.442173</td>
        </tr>
        <tr>
          <th>597</th>
          <td>True</td>
          <td>0.410133</td>
          <td>25.673685</td>
          <td>0.192824</td>
          <td>24.865269</td>
          <td>0.032888</td>
          <td>23.971060</td>
          <td>0.013630</td>
          <td>23.739090</td>
          <td>0.017672</td>
          <td>23.530157</td>
          <td>0.027685</td>
          <td>23.459727</td>
          <td>0.058640</td>
        </tr>
        <tr>
          <th>598</th>
          <td>True</td>
          <td>0.299733</td>
          <td>29.113775</td>
          <td>1.871178</td>
          <td>27.213368</td>
          <td>0.252691</td>
          <td>26.316164</td>
          <td>0.104110</td>
          <td>26.612522</td>
          <td>0.215678</td>
          <td>26.338080</td>
          <td>0.313754</td>
          <td>25.960055</td>
          <td>0.476829</td>
        </tr>
        <tr>
          <th>599</th>
          <td>True</td>
          <td>0.218481</td>
          <td>26.985461</td>
          <td>0.542968</td>
          <td>25.646029</td>
          <td>0.065578</td>
          <td>25.393283</td>
          <td>0.046046</td>
          <td>25.145220</td>
          <td>0.060390</td>
          <td>25.325217</td>
          <td>0.134623</td>
          <td>25.398899</td>
          <td>0.308790</td>
        </tr>
      </tbody>
    </table>
    <p>600 rows × 14 columns</p>
    </div>



We can see that, as expected, the target data set has the “flag” column,
and that the length of the data set is still 600. Now let’s take a look
at the calibration data set, where we left ``drop_rows`` as true:

.. code:: ipython3

    targ_data_inc["output"]  # look at the output




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>flag</th>
          <th>redshift</th>
          <th>mag_u_lsst</th>
          <th>mag_u_lsst_err</th>
          <th>mag_g_lsst</th>
          <th>mag_g_lsst_err</th>
          <th>mag_r_lsst</th>
          <th>mag_r_lsst_err</th>
          <th>mag_i_lsst</th>
          <th>mag_i_lsst_err</th>
          <th>mag_z_lsst</th>
          <th>mag_z_lsst_err</th>
          <th>mag_y_lsst</th>
          <th>mag_y_lsst_err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>True</td>
          <td>0.710979</td>
          <td>26.762862</td>
          <td>0.460818</td>
          <td>26.802089</td>
          <td>0.179290</td>
          <td>26.148865</td>
          <td>0.089899</td>
          <td>25.530220</td>
          <td>0.084893</td>
          <td>25.525601</td>
          <td>0.159922</td>
          <td>25.866070</td>
          <td>0.444362</td>
        </tr>
        <tr>
          <th>1</th>
          <td>True</td>
          <td>0.684234</td>
          <td>23.210656</td>
          <td>0.022756</td>
          <td>22.488883</td>
          <td>0.006382</td>
          <td>21.626449</td>
          <td>0.005261</td>
          <td>20.815851</td>
          <td>0.005173</td>
          <td>20.562635</td>
          <td>0.005376</td>
          <td>20.348689</td>
          <td>0.006163</td>
        </tr>
        <tr>
          <th>2</th>
          <td>True</td>
          <td>0.249871</td>
          <td>27.112190</td>
          <td>0.594565</td>
          <td>28.039316</td>
          <td>0.483038</td>
          <td>26.920252</td>
          <td>0.175355</td>
          <td>26.965811</td>
          <td>0.288325</td>
          <td>27.001835</td>
          <td>0.521963</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>True</td>
          <td>1.751949</td>
          <td>27.919353</td>
          <td>1.008960</td>
          <td>25.954472</td>
          <td>0.086088</td>
          <td>25.496445</td>
          <td>0.050463</td>
          <td>24.900504</td>
          <td>0.048597</td>
          <td>24.270590</td>
          <td>0.053261</td>
          <td>23.881404</td>
          <td>0.085152</td>
        </tr>
        <tr>
          <th>4</th>
          <td>True</td>
          <td>0.478461</td>
          <td>27.142777</td>
          <td>0.607556</td>
          <td>27.380647</td>
          <td>0.289556</td>
          <td>25.807922</td>
          <td>0.066527</td>
          <td>25.244399</td>
          <td>0.065941</td>
          <td>24.948522</td>
          <td>0.096970</td>
          <td>25.168788</td>
          <td>0.256251</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>595</th>
          <td>True</td>
          <td>0.577203</td>
          <td>28.865338</td>
          <td>1.671480</td>
          <td>26.842217</td>
          <td>0.185480</td>
          <td>25.729843</td>
          <td>0.062078</td>
          <td>25.313037</td>
          <td>0.070074</td>
          <td>24.948450</td>
          <td>0.096964</td>
          <td>25.564396</td>
          <td>0.352132</td>
        </tr>
        <tr>
          <th>596</th>
          <td>True</td>
          <td>0.405416</td>
          <td>25.636360</td>
          <td>0.186861</td>
          <td>25.907552</td>
          <td>0.082606</td>
          <td>25.873413</td>
          <td>0.070499</td>
          <td>25.953150</td>
          <td>0.122920</td>
          <td>25.977916</td>
          <td>0.234033</td>
          <td>25.859538</td>
          <td>0.442173</td>
        </tr>
        <tr>
          <th>597</th>
          <td>True</td>
          <td>0.410133</td>
          <td>25.673685</td>
          <td>0.192824</td>
          <td>24.865269</td>
          <td>0.032888</td>
          <td>23.971060</td>
          <td>0.013630</td>
          <td>23.739090</td>
          <td>0.017672</td>
          <td>23.530157</td>
          <td>0.027685</td>
          <td>23.459727</td>
          <td>0.058640</td>
        </tr>
        <tr>
          <th>598</th>
          <td>True</td>
          <td>0.299733</td>
          <td>29.113775</td>
          <td>1.871178</td>
          <td>27.213368</td>
          <td>0.252691</td>
          <td>26.316164</td>
          <td>0.104110</td>
          <td>26.612522</td>
          <td>0.215678</td>
          <td>26.338080</td>
          <td>0.313754</td>
          <td>25.960055</td>
          <td>0.476829</td>
        </tr>
        <tr>
          <th>599</th>
          <td>True</td>
          <td>0.218481</td>
          <td>26.985461</td>
          <td>0.542968</td>
          <td>25.646029</td>
          <td>0.065578</td>
          <td>25.393283</td>
          <td>0.046046</td>
          <td>25.145220</td>
          <td>0.060390</td>
          <td>25.325217</td>
          <td>0.134623</td>
          <td>25.398899</td>
          <td>0.308790</td>
        </tr>
      </tbody>
    </table>
    <p>600 rows × 14 columns</p>
    </div>



This data set is shorter than the target data set now, since those
galaxies have just been removed from the data entirely. This isn’t a
problem for the calibration data set, since we don’t need to compare it
to anything later. Let’s plot a histogram of the calibration data set
redshifts with just the photometric errors, and compare it to our new
data set with both that and the redshift incompleteness:

.. code:: ipython3

    plt.hist(calib_data_photerrs["output"]["redshift"], label="input", **hist_options)
    plt.hist(calib_data_inc["output"]["redshift"], label="ouput", **hist_options)
    plt.legend(loc="best")
    plt.xlabel("redshift")
    plt.ylabel("number of galaxies")




.. parsed-literal::

    Text(0, 0.5, 'number of galaxies')




.. image:: Exploring_the_Effects_of_Degraders_files/Exploring_the_Effects_of_Degraders_29_1.png


The output data set clearly has fewer galaxies than the input data set
above redshift of 1, and the distributions are the same for redshifts
less than 1, as expected.

For the target data set, we just have one more step that we need to do
before we can feed it into any other degraders. We use the “flag” column
to mask all of the “dropped” galaxy rows and set them all as ``np.nan``
- this keeps the indices the same, allowing us to compare to the truth
data set as is our goal.

.. code:: ipython3

    # save the column as a separate variable
    inc_flag = targ_data_inc["output"]["flag"]
    
    # drop the flag column from the dataframe entirely
    targ_data_inc["output"].drop(columns="flag", inplace=True)
    
    # replace the lines that are cut out by the degrader with np.nan
    new_targ_data_inc = targ_data_inc["output"].where(inc_flag, np.nan)
    
    # take a look at the result
    new_targ_data_inc




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>mag_u_lsst</th>
          <th>mag_u_lsst_err</th>
          <th>mag_g_lsst</th>
          <th>mag_g_lsst_err</th>
          <th>mag_r_lsst</th>
          <th>mag_r_lsst_err</th>
          <th>mag_i_lsst</th>
          <th>mag_i_lsst_err</th>
          <th>mag_z_lsst</th>
          <th>mag_z_lsst_err</th>
          <th>mag_y_lsst</th>
          <th>mag_y_lsst_err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.710979</td>
          <td>26.762862</td>
          <td>0.460818</td>
          <td>26.802089</td>
          <td>0.179290</td>
          <td>26.148865</td>
          <td>0.089899</td>
          <td>25.530220</td>
          <td>0.084893</td>
          <td>25.525601</td>
          <td>0.159922</td>
          <td>25.866070</td>
          <td>0.444362</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.684234</td>
          <td>23.210656</td>
          <td>0.022756</td>
          <td>22.488883</td>
          <td>0.006382</td>
          <td>21.626449</td>
          <td>0.005261</td>
          <td>20.815851</td>
          <td>0.005173</td>
          <td>20.562635</td>
          <td>0.005376</td>
          <td>20.348689</td>
          <td>0.006163</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.249871</td>
          <td>27.112190</td>
          <td>0.594565</td>
          <td>28.039316</td>
          <td>0.483038</td>
          <td>26.920252</td>
          <td>0.175355</td>
          <td>26.965811</td>
          <td>0.288325</td>
          <td>27.001835</td>
          <td>0.521963</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.751949</td>
          <td>27.919353</td>
          <td>1.008960</td>
          <td>25.954472</td>
          <td>0.086088</td>
          <td>25.496445</td>
          <td>0.050463</td>
          <td>24.900504</td>
          <td>0.048597</td>
          <td>24.270590</td>
          <td>0.053261</td>
          <td>23.881404</td>
          <td>0.085152</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.478461</td>
          <td>27.142777</td>
          <td>0.607556</td>
          <td>27.380647</td>
          <td>0.289556</td>
          <td>25.807922</td>
          <td>0.066527</td>
          <td>25.244399</td>
          <td>0.065941</td>
          <td>24.948522</td>
          <td>0.096970</td>
          <td>25.168788</td>
          <td>0.256251</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>595</th>
          <td>0.577203</td>
          <td>28.865338</td>
          <td>1.671480</td>
          <td>26.842217</td>
          <td>0.185480</td>
          <td>25.729843</td>
          <td>0.062078</td>
          <td>25.313037</td>
          <td>0.070074</td>
          <td>24.948450</td>
          <td>0.096964</td>
          <td>25.564396</td>
          <td>0.352132</td>
        </tr>
        <tr>
          <th>596</th>
          <td>0.405416</td>
          <td>25.636360</td>
          <td>0.186861</td>
          <td>25.907552</td>
          <td>0.082606</td>
          <td>25.873413</td>
          <td>0.070499</td>
          <td>25.953150</td>
          <td>0.122920</td>
          <td>25.977916</td>
          <td>0.234033</td>
          <td>25.859538</td>
          <td>0.442173</td>
        </tr>
        <tr>
          <th>597</th>
          <td>0.410133</td>
          <td>25.673685</td>
          <td>0.192824</td>
          <td>24.865269</td>
          <td>0.032888</td>
          <td>23.971060</td>
          <td>0.013630</td>
          <td>23.739090</td>
          <td>0.017672</td>
          <td>23.530157</td>
          <td>0.027685</td>
          <td>23.459727</td>
          <td>0.058640</td>
        </tr>
        <tr>
          <th>598</th>
          <td>0.299733</td>
          <td>29.113775</td>
          <td>1.871178</td>
          <td>27.213368</td>
          <td>0.252691</td>
          <td>26.316164</td>
          <td>0.104110</td>
          <td>26.612522</td>
          <td>0.215678</td>
          <td>26.338080</td>
          <td>0.313754</td>
          <td>25.960055</td>
          <td>0.476829</td>
        </tr>
        <tr>
          <th>599</th>
          <td>0.218481</td>
          <td>26.985461</td>
          <td>0.542968</td>
          <td>25.646029</td>
          <td>0.065578</td>
          <td>25.393283</td>
          <td>0.046046</td>
          <td>25.145220</td>
          <td>0.060390</td>
          <td>25.325217</td>
          <td>0.134623</td>
          <td>25.398899</td>
          <td>0.308790</td>
        </tr>
      </tbody>
    </table>
    <p>600 rows × 13 columns</p>
    </div>



The new dataframe is the same length as the old one, but without the
flag column, and now those rows will just be ``np.nan``.

3. Line Confusion
~~~~~~~~~~~~~~~~~

This method simulates the effect of misidentified lines. The degrader
will misidentify some percentage (``frac_wrong``) of the actual lines
(here we’re picking :math:`5007.0~\mathring{\mathrm{A}}`, which are OIII
lines) as the line we pick for ``wrong_wavelen``. In this case, we’ll
pick :math:`3727.0~\mathring{\mathrm{A}}`, which are OII lines.

This degrader doesn’t cut any galaxies, so we don’t have to worry about
the ``drop_rows`` parameter.

.. code:: ipython3

    # dataset 3: add in line confusion
    calib_data_conf = ri.creation.degraders.spectroscopic_degraders.line_confusion(
        sample=calib_data_inc["output"],
        true_wavelen=5007.0,
        wrong_wavelen=3727.0,
        frac_wrong=0.05,
        seed=1337,
    )
    
    # dataset 3: add in line confusion using the modified data set
    targ_data_conf = ri.creation.degraders.spectroscopic_degraders.line_confusion(
        sample=new_targ_data_inc,
        true_wavelen=5007.0,
        wrong_wavelen=3727.0,
        frac_wrong=0.05,
        seed=1450,
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, LineConfusion
    Inserting handle into data store.  output: inprogress_output.pq, LineConfusion
    Inserting handle into data store.  input: None, LineConfusion
    Inserting handle into data store.  output: inprogress_output.pq, LineConfusion


Now let’s take a look at what this has done to our redshift distribution
by plotting the input calibration data set against the one output by the
``line_confusion`` method:

.. code:: ipython3

    plt.hist(calib_data_inc["output"]["redshift"], label="input data", **hist_options)
    plt.hist(calib_data_conf["output"]["redshift"], label="output data", **hist_options)
    plt.legend(loc="best")
    plt.ylabel("redshift")
    plt.ylabel("number of galaxies")




.. parsed-literal::

    Text(0, 0.5, 'number of galaxies')




.. image:: Exploring_the_Effects_of_Degraders_files/Exploring_the_Effects_of_Degraders_36_1.png


We can see that the output data has a few small differences in the
distribution, spread across the whole range of redshifts.

4. Quantity Cut
~~~~~~~~~~~~~~~

This method cuts galaxies based on their band magnitudes. It takes a
dictionary of cuts, where you can provide the band name and the values
to cut that band on (for example, ``{"mag_i_lsst": 25.0}``). If one
value is given, it’s considered a maximum, and if a tuple is given, it’s
considered a range within which the sample is selected. For this, we’ll
just set a maximum magnitude for the i band of 25.

Since this method cuts galaxies, we’re going to follow the steps we used
for the ``inv_redshift_incompleteness`` method to keep our target
dataset at the same length:

.. code:: ipython3

    # cut some of the data below a certain magnitude
    calib_data_cut = ri.creation.degraders.quantityCut.quantity_cut(
        sample=calib_data_conf["output"], cuts={"mag_i_lsst": 25.0}
    )
    
    # cut some of the data below a certain magnitude, set drop_rows=False to keep data set the same length
    targ_data_cut = ri.creation.degraders.quantityCut.quantity_cut(
        sample=targ_data_conf["output"], cuts={"mag_i_lsst": 25.0}, drop_rows=False
    )
    targ_data_cut["output"]


.. parsed-literal::

    Inserting handle into data store.  input: None, QuantityCut
    Inserting handle into data store.  output: inprogress_output.pq, QuantityCut
    Inserting handle into data store.  input: None, QuantityCut
    Inserting handle into data store.  output: inprogress_output.pq, QuantityCut




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>flag</th>
          <th>redshift</th>
          <th>mag_u_lsst</th>
          <th>mag_u_lsst_err</th>
          <th>mag_g_lsst</th>
          <th>mag_g_lsst_err</th>
          <th>mag_r_lsst</th>
          <th>mag_r_lsst_err</th>
          <th>mag_i_lsst</th>
          <th>mag_i_lsst_err</th>
          <th>mag_z_lsst</th>
          <th>mag_z_lsst_err</th>
          <th>mag_y_lsst</th>
          <th>mag_y_lsst_err</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>0.710979</td>
          <td>26.762862</td>
          <td>0.460818</td>
          <td>26.802089</td>
          <td>0.179290</td>
          <td>26.148865</td>
          <td>0.089899</td>
          <td>25.530220</td>
          <td>0.084893</td>
          <td>25.525601</td>
          <td>0.159922</td>
          <td>25.866070</td>
          <td>0.444362</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>0.684234</td>
          <td>23.210656</td>
          <td>0.022756</td>
          <td>22.488883</td>
          <td>0.006382</td>
          <td>21.626449</td>
          <td>0.005261</td>
          <td>20.815851</td>
          <td>0.005173</td>
          <td>20.562635</td>
          <td>0.005376</td>
          <td>20.348689</td>
          <td>0.006163</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0</td>
          <td>0.249871</td>
          <td>27.112190</td>
          <td>0.594565</td>
          <td>28.039316</td>
          <td>0.483038</td>
          <td>26.920252</td>
          <td>0.175355</td>
          <td>26.965811</td>
          <td>0.288325</td>
          <td>27.001835</td>
          <td>0.521963</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1</td>
          <td>1.751949</td>
          <td>27.919353</td>
          <td>1.008960</td>
          <td>25.954472</td>
          <td>0.086088</td>
          <td>25.496445</td>
          <td>0.050463</td>
          <td>24.900504</td>
          <td>0.048597</td>
          <td>24.270590</td>
          <td>0.053261</td>
          <td>23.881404</td>
          <td>0.085152</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0</td>
          <td>0.478461</td>
          <td>27.142777</td>
          <td>0.607556</td>
          <td>27.380647</td>
          <td>0.289556</td>
          <td>25.807922</td>
          <td>0.066527</td>
          <td>25.244399</td>
          <td>0.065941</td>
          <td>24.948522</td>
          <td>0.096970</td>
          <td>25.168788</td>
          <td>0.256251</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>595</th>
          <td>0</td>
          <td>0.577203</td>
          <td>28.865338</td>
          <td>1.671480</td>
          <td>26.842217</td>
          <td>0.185480</td>
          <td>25.729843</td>
          <td>0.062078</td>
          <td>25.313037</td>
          <td>0.070074</td>
          <td>24.948450</td>
          <td>0.096964</td>
          <td>25.564396</td>
          <td>0.352132</td>
        </tr>
        <tr>
          <th>596</th>
          <td>0</td>
          <td>0.405416</td>
          <td>25.636360</td>
          <td>0.186861</td>
          <td>25.907552</td>
          <td>0.082606</td>
          <td>25.873413</td>
          <td>0.070499</td>
          <td>25.953150</td>
          <td>0.122920</td>
          <td>25.977916</td>
          <td>0.234033</td>
          <td>25.859538</td>
          <td>0.442173</td>
        </tr>
        <tr>
          <th>597</th>
          <td>1</td>
          <td>0.410133</td>
          <td>25.673685</td>
          <td>0.192824</td>
          <td>24.865269</td>
          <td>0.032888</td>
          <td>23.971060</td>
          <td>0.013630</td>
          <td>23.739090</td>
          <td>0.017672</td>
          <td>23.530157</td>
          <td>0.027685</td>
          <td>23.459727</td>
          <td>0.058640</td>
        </tr>
        <tr>
          <th>598</th>
          <td>0</td>
          <td>0.299733</td>
          <td>29.113775</td>
          <td>1.871178</td>
          <td>27.213368</td>
          <td>0.252691</td>
          <td>26.316164</td>
          <td>0.104110</td>
          <td>26.612522</td>
          <td>0.215678</td>
          <td>26.338080</td>
          <td>0.313754</td>
          <td>25.960055</td>
          <td>0.476829</td>
        </tr>
        <tr>
          <th>599</th>
          <td>0</td>
          <td>0.636955</td>
          <td>26.985461</td>
          <td>0.542968</td>
          <td>25.646029</td>
          <td>0.065578</td>
          <td>25.393283</td>
          <td>0.046046</td>
          <td>25.145220</td>
          <td>0.060390</td>
          <td>25.325217</td>
          <td>0.134623</td>
          <td>25.398899</td>
          <td>0.308790</td>
        </tr>
      </tbody>
    </table>
    <p>600 rows × 14 columns</p>
    </div>



We can see that there’s been a flag column added to the target data
again, but this time the flags are 1 and 0 instead of True and False.
Let’s save the flag column and drop it from the main DataFrame. We’re
going to do something a little different with the data later so we don’t
need do the ``np.nan`` substitution from earlier.

.. code:: ipython3

    # save flag column
    cut_flag = targ_data_cut["output"]["flag"]
    
    # drop flag column from dataframe
    targ_data_cut["output"].drop(columns="flag", inplace=True)

Now let’s plot a histogram of the calibration data set we input into the
``quantity_cut`` method compared to the output calibration data set to
see how it’s changed the number and distribution of galaxies:

.. code:: ipython3

    plt.hist(calib_data_conf["output"]["redshift"], label="input data", **hist_options)
    plt.hist(calib_data_cut["output"]["redshift"], label="output data", **hist_options)
    plt.legend(loc="best")
    plt.xlabel("redshift")
    plt.ylabel("number of galaxies")




.. parsed-literal::

    Text(0, 0.5, 'number of galaxies')




.. image:: Exploring_the_Effects_of_Degraders_files/Exploring_the_Effects_of_Degraders_42_1.png


We can see our output distribution has roughly the same shape, but with
significantly fewer galaxies overall.

Now we have applied four different degraders, so we’ve set up our
various calibration data sets, and our target data set. The final step
is to use the dictionary we made earlier of error column names
(``rename_dict``) and the RAIL function ``column_mapper`` to rename the
error columns, so they match the expected names for the later steps:

.. code:: ipython3

    # renames error columns to match DC2 for calibration data sets
    
    # photerrs
    df_calib_data_photerrs = ri.tools.table_tools.column_mapper(
        data=calib_data_photerrs["output"], columns=rename_dict
    )
    
    # photerrs
    df_calib_data_inc = ri.tools.table_tools.column_mapper(
        data=targ_data_inc["output"], columns=rename_dict
    )
    
    # photerrs
    df_calib_data_conf = ri.tools.table_tools.column_mapper(
        data=calib_data_conf["output"], columns=rename_dict
    )
    
    # photerrs
    df_calib_data_cut = ri.tools.table_tools.column_mapper(
        data=calib_data_cut["output"], columns=rename_dict
    )
    
    
    # renames error columns for target data set
    df_targ_data = ri.tools.table_tools.column_mapper(
        data=targ_data_cut["output"], columns=rename_dict
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, ColumnMapper
    Inserting handle into data store.  output: inprogress_output.pq, ColumnMapper
    Inserting handle into data store.  input: None, ColumnMapper
    Inserting handle into data store.  output: inprogress_output.pq, ColumnMapper
    Inserting handle into data store.  input: None, ColumnMapper
    Inserting handle into data store.  output: inprogress_output.pq, ColumnMapper
    Inserting handle into data store.  input: None, ColumnMapper
    Inserting handle into data store.  output: inprogress_output.pq, ColumnMapper
    Inserting handle into data store.  input: None, ColumnMapper
    Inserting handle into data store.  output: inprogress_output.pq, ColumnMapper


Now that we have all four of our calibration data sets, let’s plot them
all together to get a final look at their differences:

.. code:: ipython3

    plt.hist(
        df_calib_data_photerrs["output"]["redshift"],
        label="photometric errors",
        **hist_options,
    )
    plt.hist(
        df_calib_data_inc["output"]["redshift"], label="z incompleteness", **hist_options
    )
    plt.hist(
        df_calib_data_conf["output"]["redshift"], label="line confusion", **hist_options
    )
    plt.hist(df_calib_data_cut["output"]["redshift"], label="quantity cut", **hist_options)
    
    plt.legend(loc="best")
    plt.xlabel("redshift")
    plt.ylabel("number of galaxies")




.. parsed-literal::

    Text(0, 0.5, 'number of galaxies')




.. image:: Exploring_the_Effects_of_Degraders_files/Exploring_the_Effects_of_Degraders_46_1.png


We have one final step to do to our target data set before we can use it
in the estimation and evaluation stages. For this data set to work with
the RAIL evaluate stages, we want a couple of things:

1. Our degraded (and cut down) target DataFrame indices to match up with
   our original target DataFrame indices
2. Our target data sets to not have columns with all NaNs
3. Our target data to have linearly increasing indices (i.e. not retain
   the masked indices)

In order to accomplish this, we’re going to do the following: 1. Mask
the degraded target data set using our existing masks 2. Mask the
“truth” target data set using the existing masks 3. Reindex both of
these arrays

.. code:: ipython3

    # mask the degraded test data
    masked_targ_data = df_targ_data["output"][cut_flag & inc_flag]
    
    # reset the index
    reindexed_targ_data = masked_targ_data.reset_index(drop=True)
    reindexed_targ_data




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>mag_u_lsst</th>
          <th>mag_err_u_lsst</th>
          <th>mag_g_lsst</th>
          <th>mag_err_g_lsst</th>
          <th>mag_r_lsst</th>
          <th>mag_err_r_lsst</th>
          <th>mag_i_lsst</th>
          <th>mag_err_i_lsst</th>
          <th>mag_z_lsst</th>
          <th>mag_err_z_lsst</th>
          <th>mag_y_lsst</th>
          <th>mag_err_y_lsst</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.684234</td>
          <td>23.210656</td>
          <td>0.022756</td>
          <td>22.488883</td>
          <td>0.006382</td>
          <td>21.626449</td>
          <td>0.005261</td>
          <td>20.815851</td>
          <td>0.005173</td>
          <td>20.562635</td>
          <td>0.005376</td>
          <td>20.348689</td>
          <td>0.006163</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.751949</td>
          <td>27.919353</td>
          <td>1.008960</td>
          <td>25.954472</td>
          <td>0.086088</td>
          <td>25.496445</td>
          <td>0.050463</td>
          <td>24.900504</td>
          <td>0.048597</td>
          <td>24.270590</td>
          <td>0.053261</td>
          <td>23.881404</td>
          <td>0.085152</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.334738</td>
          <td>22.874410</td>
          <td>0.017247</td>
          <td>22.589696</td>
          <td>0.006609</td>
          <td>22.079401</td>
          <td>0.005541</td>
          <td>22.081291</td>
          <td>0.006311</td>
          <td>21.733536</td>
          <td>0.007373</td>
          <td>21.967633</td>
          <td>0.016030</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.508261</td>
          <td>28.081741</td>
          <td>1.110059</td>
          <td>26.276645</td>
          <td>0.114130</td>
          <td>25.395028</td>
          <td>0.046117</td>
          <td>24.961183</td>
          <td>0.051288</td>
          <td>24.863762</td>
          <td>0.090014</td>
          <td>24.410283</td>
          <td>0.135127</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.202286</td>
          <td>25.221688</td>
          <td>0.131189</td>
          <td>24.293276</td>
          <td>0.020071</td>
          <td>23.534810</td>
          <td>0.009916</td>
          <td>23.200063</td>
          <td>0.011558</td>
          <td>22.954672</td>
          <td>0.016944</td>
          <td>22.812649</td>
          <td>0.033050</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>230</th>
          <td>0.498327</td>
          <td>25.262660</td>
          <td>0.135901</td>
          <td>24.728088</td>
          <td>0.029166</td>
          <td>23.968463</td>
          <td>0.013603</td>
          <td>23.650719</td>
          <td>0.016428</td>
          <td>23.688417</td>
          <td>0.031807</td>
          <td>23.429337</td>
          <td>0.057080</td>
        </tr>
        <tr>
          <th>231</th>
          <td>0.351069</td>
          <td>25.236311</td>
          <td>0.132853</td>
          <td>23.537328</td>
          <td>0.011081</td>
          <td>22.117558</td>
          <td>0.005575</td>
          <td>21.565970</td>
          <td>0.005575</td>
          <td>21.224826</td>
          <td>0.006085</td>
          <td>21.011602</td>
          <td>0.008167</td>
        </tr>
        <tr>
          <th>232</th>
          <td>0.631504</td>
          <td>27.523146</td>
          <td>0.786924</td>
          <td>25.309303</td>
          <td>0.048671</td>
          <td>23.763543</td>
          <td>0.011646</td>
          <td>22.631465</td>
          <td>0.008012</td>
          <td>22.201093</td>
          <td>0.009619</td>
          <td>21.919523</td>
          <td>0.015417</td>
        </tr>
        <tr>
          <th>233</th>
          <td>1.020108</td>
          <td>24.253126</td>
          <td>0.056253</td>
          <td>23.911289</td>
          <td>0.014679</td>
          <td>23.228338</td>
          <td>0.008209</td>
          <td>22.512949</td>
          <td>0.007533</td>
          <td>21.869678</td>
          <td>0.007898</td>
          <td>21.642636</td>
          <td>0.012414</td>
        </tr>
        <tr>
          <th>234</th>
          <td>0.410133</td>
          <td>25.673685</td>
          <td>0.192824</td>
          <td>24.865269</td>
          <td>0.032888</td>
          <td>23.971060</td>
          <td>0.013630</td>
          <td>23.739090</td>
          <td>0.017672</td>
          <td>23.530157</td>
          <td>0.027685</td>
          <td>23.459727</td>
          <td>0.058640</td>
        </tr>
      </tbody>
    </table>
    <p>235 rows × 13 columns</p>
    </div>



.. code:: ipython3

    # mask the degraded target data
    masked_targ_data_orig = targ_data_orig["output"][cut_flag & inc_flag]
    
    # reset the index
    reindexed_targ_data_orig = masked_targ_data_orig.reset_index(drop=True)
    reindexed_targ_data_orig




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>mag_u_lsst</th>
          <th>mag_g_lsst</th>
          <th>mag_r_lsst</th>
          <th>mag_i_lsst</th>
          <th>mag_z_lsst</th>
          <th>mag_y_lsst</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.684234</td>
          <td>23.222248</td>
          <td>22.496042</td>
          <td>21.628773</td>
          <td>20.817329</td>
          <td>20.564213</td>
          <td>20.347912</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.751949</td>
          <td>28.061226</td>
          <td>26.201729</td>
          <td>25.492706</td>
          <td>24.880798</td>
          <td>24.292238</td>
          <td>23.972845</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.334738</td>
          <td>22.901604</td>
          <td>22.585258</td>
          <td>22.078140</td>
          <td>22.091541</td>
          <td>21.725985</td>
          <td>21.967663</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.508261</td>
          <td>27.219675</td>
          <td>26.349281</td>
          <td>25.363613</td>
          <td>24.975111</td>
          <td>24.789568</td>
          <td>24.562302</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.202286</td>
          <td>25.150200</td>
          <td>24.267548</td>
          <td>23.537415</td>
          <td>23.175728</td>
          <td>22.982109</td>
          <td>22.828064</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>230</th>
          <td>0.498327</td>
          <td>25.256531</td>
          <td>24.734894</td>
          <td>23.983690</td>
          <td>23.683472</td>
          <td>23.646606</td>
          <td>23.460381</td>
        </tr>
        <tr>
          <th>231</th>
          <td>0.351069</td>
          <td>25.185009</td>
          <td>23.521603</td>
          <td>22.113777</td>
          <td>21.557377</td>
          <td>21.228889</td>
          <td>21.010700</td>
        </tr>
        <tr>
          <th>232</th>
          <td>0.631504</td>
          <td>27.546143</td>
          <td>25.364649</td>
          <td>23.775986</td>
          <td>22.639019</td>
          <td>22.233021</td>
          <td>21.900209</td>
        </tr>
        <tr>
          <th>233</th>
          <td>1.020108</td>
          <td>24.216951</td>
          <td>23.881901</td>
          <td>23.230240</td>
          <td>22.513334</td>
          <td>21.873619</td>
          <td>21.645157</td>
        </tr>
        <tr>
          <th>234</th>
          <td>0.410133</td>
          <td>25.853966</td>
          <td>24.891123</td>
          <td>23.988443</td>
          <td>23.726196</td>
          <td>23.565538</td>
          <td>23.432323</td>
        </tr>
      </tbody>
    </table>
    <p>235 rows × 7 columns</p>
    </div>



We can see that these DataFrames are now the same length, with indices
that actually match the length of the arrays and that are linearly
increasing, which is what we wanted. Now these can be appropriately
compared to each other in the later steps.

3. Calibrating the photometric redshift algorithms with the differently degraded data
-------------------------------------------------------------------------------------

Now we can loop through each of the calibration datasets to calibrate
our algorithms. We’ll use all four of our calibration data sets to
calibrate our models.

For this notebook, we’ll use the `K-Nearest
Neighbours <https://rail-hub.readthedocs.io/en/latarget/source/rail_stages/estimation.html#k-nearest-neighbor>`__
(KNN) algorithm, which is a wrapper around ``sklearn``\ ’s nearest
neighbour (NN) machine learning model. Essentially, it takes a given
galaxy, identifies its nearest neighbours in the space, in this case
galaxies that have similar colours, and then constructs the photometric
redshift PDF as a sum of Gaussians from each neighbour. For more details
on how this algorithm works, you can see the `wikipedia
page <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`__ or
the `Quick Start in
Estimation <https://rail-hub.readthedocs.io/projects/rail-notebooks/en/latest/interactive_examples/rendered/estimation_examples/00_Quick_Start_in_Estimation.html>`__
notebook.

The calibration methods of RAIL algorithms are called *informers*, so
the function we want to use is called ``k_near_neigh_informer()``.

Useful parameters: - ``nondetect_val``: This tells the code which values
are considered non-detections. We pass in ``np.nan`` here, since that’s
what we used as the ``ndFlag`` in the degradation stage for
non-detections. - ``hdf5_groupname``: the dictionary key the code will
find the data under. Set to ``""`` if the data is passed in directly.

First, we’ll set up a dictionary with all four of the calibration
datasets, and empty dictionaries to store the calibrated models:

.. code:: ipython3

    # make a dictionary of the calibration datasets to iterate through
    calib_datasets = {
        "lsst_error_model": df_calib_data_photerrs,
        "inv_redshift_inc": df_calib_data_inc,
        "line_confusion": df_calib_data_conf,
        "quantity_cut": df_calib_data_cut,
    }
    
    # set up dictionary for output
    knn_models = {}

Now we’ll iterate through the datasets, calibrating a model for each
calibration set:

.. code:: ipython3

    for key, item in calib_datasets.items():
    
        # calibrate the model
        inform_knn = ri.estimation.algos.k_nearneigh.k_near_neigh_informer(
            training_data=item["output"], nondetect_val=np.nan, hdf5_groupname=""
        )
        
        knn_models[key] = inform_knn


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 450 training and 150 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.06777777777777778 and numneigh=7
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer
    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 450 training and 150 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.06777777777777778 and numneigh=7
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer
    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 386 training and 128 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.075 and numneigh=4
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer
    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 191 training and 64 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.075 and numneigh=3
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer


.. code:: ipython3

    # let's see what the output looks like 
    knn_models["lsst_error_model"]




.. parsed-literal::

    {'model': {'kdtree': <sklearn.neighbors._kd_tree.KDTree at 0x55d9fc4664e0>,
      'bestsig': np.float64(0.06777777777777778),
      'nneigh': 7,
      'truezs': array([1.4333429 , 1.3237551 , 0.8702599 , 1.7939827 , 1.2945056 ,
             1.1320935 , 1.1123738 , 0.14820874, 1.7649848 , 0.13836348,
             0.5463327 , 0.33181095, 0.45686185, 0.7568008 , 1.8688486 ,
             0.51444894, 1.1470054 , 0.56240654, 0.77783823, 0.47601438,
             0.8391931 , 0.9542949 , 0.52233857, 2.049638  , 0.9337821 ,
             1.5828538 , 1.1982977 , 0.5996851 , 0.34287202, 1.302322  ,
             0.5765392 , 0.5861906 , 0.9636896 , 1.3210725 , 0.29344237,
             1.5826114 , 2.2057626 , 0.73928225, 1.2017496 , 0.29615378,
             1.1129823 , 0.285105  , 0.9201316 , 0.7032695 , 1.9016705 ,
             0.19354403, 1.4168835 , 0.73398304, 1.1979619 , 1.232079  ,
             0.5940706 , 1.5156059 , 1.9105101 , 0.2790072 , 1.4457306 ,
             1.5527472 , 0.41600168, 0.9682647 , 0.5611768 , 2.1933482 ,
             1.3341204 , 1.3705808 , 1.7518663 , 0.59243715, 0.52366143,
             0.5632213 , 1.650192  , 0.5890905 , 1.5205675 , 1.0699294 ,
             0.73326874, 1.835554  , 1.7176813 , 1.743949  , 0.9451206 ,
             0.08180296, 0.49115348, 1.4235613 , 1.1388881 , 0.38673675,
             1.1898067 , 1.0973556 , 0.8509266 , 0.67183864, 0.8128478 ,
             1.3921927 , 1.4698129 , 0.72021025, 1.3031769 , 0.32609248,
             1.210638  , 0.42013288, 2.242836  , 1.3876585 , 0.78741646,
             0.29916453, 2.1097922 , 1.3746064 , 0.26279032, 0.3636477 ,
             0.3132515 , 2.2058485 , 0.98319566, 0.5067608 , 1.0027157 ,
             0.849437  , 1.4602329 , 1.3697542 , 1.2450745 , 2.1913164 ,
             1.6064138 , 0.93563014, 1.1443644 , 0.33537817, 0.8218638 ,
             0.70383537, 0.90621483, 1.2114272 , 1.0715214 , 0.9545132 ,
             1.065865  , 1.2047305 , 0.45711076, 0.66132927, 0.31336808,
             0.17441905, 2.043468  , 0.17591262, 0.30481195, 0.470649  ,
             0.7549343 , 0.43365705, 1.1348412 , 2.1660342 , 0.6524892 ,
             0.8562118 , 0.14503407, 0.47894454, 2.1233435 , 1.7707397 ,
             0.21947038, 2.0594726 , 2.201642  , 1.4173121 , 0.5161568 ,
             0.8191002 , 2.1764154 , 0.65915453, 0.30754578, 1.3682023 ,
             0.13578796, 1.7152648 , 1.2880605 , 1.4056685 , 0.24158561,
             0.28185225, 0.7170991 , 1.9746878 , 0.32469094, 0.4641285 ,
             2.1328204 , 0.7931153 , 0.71383107, 2.0965385 , 0.2099607 ,
             1.1762638 , 0.35085678, 0.37228823, 1.5243781 , 1.7920742 ,
             1.768083  , 2.241077  , 0.6705948 , 1.2002186 , 1.842099  ,
             0.87937343, 0.29588675, 0.9430428 , 0.900398  , 1.7555468 ,
             0.72097045, 1.2434773 , 1.4596146 , 1.3867974 , 1.0127914 ,
             0.21674824, 1.9922367 , 2.217565  , 0.4885862 , 1.1067638 ,
             0.6340991 , 1.6274004 , 1.7903895 , 2.2525907 , 1.1592224 ,
             0.6944686 , 0.6926007 , 0.9301318 , 0.16400123, 0.9422506 ,
             1.0048239 , 1.2398766 , 1.9664532 , 0.7971904 , 1.1084152 ,
             0.3231094 , 0.8011063 , 0.16412175, 1.3823427 , 0.3111242 ,
             0.66980475, 2.1324353 , 0.36510038, 0.42826843, 0.06771374,
             1.0254345 , 0.90034324, 1.0493418 , 0.14361739, 0.17504692,
             1.0162743 , 1.7410797 , 0.5967882 , 0.95617354, 0.7685839 ,
             1.572384  , 0.9912834 , 1.8263315 , 2.157621  , 0.3042667 ,
             0.5038768 , 1.7457547 , 2.29274   , 0.483999  , 1.3042163 ,
             0.17721963, 0.37443542, 2.2124104 , 0.707572  , 1.8924183 ,
             1.1780736 , 0.19370699, 1.0474126 , 0.29545975, 1.4300001 ,
             0.44004273, 0.9724295 , 0.9186385 , 0.17414916, 0.63105214,
             1.7826929 , 0.6058014 , 2.2586625 , 0.83745164, 0.2669332 ,
             1.1646124 , 0.975093  , 2.257729  , 0.4385332 , 0.4218998 ,
             0.30188823, 2.0057054 , 0.26030195, 2.2154443 , 0.89930093,
             1.7500795 , 1.3424792 , 0.48091793, 0.5691763 , 0.8420398 ,
             1.2838188 , 1.7509391 , 0.40002012, 1.4081954 , 0.72034645,
             2.2843187 , 0.57547367, 0.30207837, 1.3668653 , 1.80824   ,
             0.6391039 , 0.40295982, 0.7391731 , 0.7412892 , 2.1955771 ,
             0.3370936 , 0.9087808 , 0.38845444, 0.30060232, 0.475186  ,
             0.5246581 , 0.5085225 , 1.1101241 , 1.1015421 , 0.1892674 ,
             0.5671846 , 0.787998  , 0.27489257, 1.9335248 , 0.4364879 ,
             0.7695458 , 0.9013622 , 1.2346406 , 0.32413328, 0.9574801 ,
             0.3013389 , 0.70873785, 0.52307767, 0.27721393, 0.5818955 ,
             2.072917  , 0.35308433, 0.55400753, 0.33471656, 0.20966005,
             0.60201263, 0.2505467 , 1.1141411 , 0.2803557 , 1.5231757 ,
             2.0144129 , 1.6401707 , 0.73318845, 1.0635017 , 1.200102  ,
             0.3425616 , 0.6631957 , 0.2206223 , 1.3993084 , 0.35734427,
             1.4942942 , 1.1615012 , 1.3075851 , 1.0389373 , 1.223887  ,
             1.0996472 , 0.6812018 , 2.0574875 , 1.3219174 , 1.1511052 ,
             0.34176445, 1.3045671 , 0.87984765, 0.32177424, 0.3763888 ,
             0.8202743 , 0.29169178, 0.31402266, 0.3444091 , 0.27668357,
             0.2959299 , 1.5235835 , 1.5358413 , 0.67979324, 0.30667782,
             0.30161166, 1.9743047 , 2.2682843 , 1.4258395 , 0.9695651 ,
             0.5671768 , 1.0727217 , 0.46502316, 0.34059918, 0.23844993,
             0.81917554, 0.16491187, 2.1633694 , 1.7594773 , 0.5566076 ,
             1.6650255 , 0.21200144, 0.45102406, 0.38877988, 1.2053446 ,
             0.34152937, 1.5321474 , 0.29181147, 0.4799868 , 1.2575852 ,
             2.1957371 , 0.34061432, 1.1815947 , 0.84324145, 0.31398714,
             0.51589644, 1.3904884 , 1.5146952 , 0.98825103, 1.3888367 ,
             1.2802653 , 0.8861996 , 0.15926027, 0.9070977 , 0.7667237 ,
             0.7266838 , 0.14852488, 0.69021386, 0.2171812 , 0.28067517,
             0.65013725, 0.73057395, 1.2387667 , 0.2594204 , 2.2343667 ,
             0.6506754 , 1.7647612 , 1.5926791 , 0.47002256, 0.30704796,
             1.5875505 , 0.5977346 , 0.30942369, 0.43926072, 1.4083784 ,
             0.7819265 , 1.6856236 , 0.1792171 , 1.3056164 , 0.76581246,
             0.25475788, 1.1734301 , 1.6756759 , 0.28719735, 0.22969246,
             0.6003391 , 0.17246795, 0.82271415, 0.527624  , 0.20761871,
             1.2142472 , 0.5452797 , 0.28257442, 0.5789856 , 0.34235108,
             1.0498717 , 0.6847771 , 1.140883  , 0.61571866, 0.49521637,
             0.9500967 , 1.5125798 , 0.28669   , 0.35282803, 1.7318455 ,
             0.47037816, 2.2011223 , 0.9508046 , 0.36787045, 0.29350114,
             0.488513  , 0.332191  , 1.3395056 , 0.2969067 , 1.338268  ,
             2.1542578 , 0.7112359 , 0.34530675, 2.0404677 , 0.6228099 ,
             0.1826657 , 0.09311604, 1.8493752 , 1.7863531 , 0.3320719 ,
             1.882168  , 0.56478727, 1.2733663 , 0.30870986, 0.6073284 ,
             0.42434287, 1.2378035 , 0.5975763 , 1.3544034 , 0.9600128 ,
             0.9768367 , 0.34299016, 0.8634549 , 0.23797214, 0.11863601,
             1.0072093 , 1.1916834 , 1.0649809 , 1.4154159 , 1.3555586 ,
             1.34472   , 1.6318725 , 1.8455282 , 0.39839816, 1.6346661 ,
             1.6372824 , 2.0320172 , 0.5530062 , 0.34578633, 1.1159678 ,
             1.2040006 , 0.27654958, 1.1554179 , 0.7724498 , 0.5336855 ,
             0.7548364 , 2.108087  , 0.2729175 , 0.58194035, 0.33421624,
             0.5175764 , 0.48491   , 1.8840697 , 0.7763976 , 0.23653305,
             0.414626  , 0.42791784, 2.0598073 , 0.43945396, 1.5376469 ,
             0.5895858 , 1.0193623 , 0.6483196 , 1.6718389 , 0.52963954,
             0.3067937 , 1.0020282 , 0.9428348 , 0.2929634 , 1.5957819 ,
             1.0945596 , 1.397035  , 1.5963764 , 1.0290524 , 0.22328591,
             0.70576704, 0.22334087, 0.8895134 , 1.913652  , 1.8872316 ,
             1.1563677 , 0.551071  , 0.74864054, 0.19263005, 0.59278095,
             1.7926812 , 1.5917898 , 0.891467  , 0.33411133, 2.1682367 ,
             0.50696313, 0.25047553, 0.9364835 , 0.2743677 , 0.61948425,
             0.29832304, 0.31371355, 1.2902012 , 1.9265969 , 1.4590098 ,
             0.41096365, 0.2730738 , 0.48057532, 1.2671596 , 2.2282958 ,
             0.29613948, 1.2573075 , 1.7920887 , 0.6521886 , 1.3949696 ,
             0.20531785, 1.2290087 , 0.5943552 , 1.4014667 , 1.1682425 ,
             0.7588887 , 2.078631  , 1.2710853 , 1.7710841 , 0.77392304,
             1.7186058 , 0.9631001 , 1.7394538 , 0.33084118, 0.78578913,
             0.64480174, 1.0032451 , 1.4184198 , 0.43717694, 1.0607342 ,
             0.6782961 , 1.9529852 , 0.75885445, 0.39110053, 1.1039946 ,
             0.9746763 , 2.2530441 , 1.4716698 , 0.33372307, 0.39481735,
             0.74744   , 0.33060288, 1.1515912 , 0.7316464 , 1.5216072 ],
            dtype=float32),
      'only_colors': False}}



We can see that the models output by this algorithm include a dictionary
of data and an ``sklearn`` object.

Estimating the photometric redshifts of a set of target galaxies using the calibrated models
--------------------------------------------------------------------------------------------

Now that we’ve got all four of our models, we can use the *Estimator* of
the KNN Algorithm on our target data set to get our photometric redshift
probability distribution functions. It takes the same parameters we gave
to the *informer* above that relate to the data format, as well as the
galaxy data to estimate redshifts for as ``input_data``, and the model
from the *informer* stage as ``model``.

We’ll iterate over each of the models, storing the estimated redshifts
in a dictionary:

.. code:: ipython3

    estimated_photoz = {} # set up a dictionary to store estimates in 
    
    for key, item in knn_models.items():
    
        # estimate the photozs
        knn_estimated = ri.estimation.algos.k_nearneigh.k_near_neigh_estimator(
            input_data=reindexed_targ_data,
            model=item["model"],
            nondetect_val=np.nan,
            hdf5_groupname="",
        )
    
        # add estimates to dictionary under the appropriate key
        estimated_photoz[key] = knn_estimated


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighEstimator
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x55d9fc4664e0>, 'bestsig': np.float64(0.06777777777777778), 'nneigh': 7, 'truezs': array([1.4333429 , 1.3237551 , 0.8702599 , 1.7939827 , 1.2945056 ,
           1.1320935 , 1.1123738 , 0.14820874, 1.7649848 , 0.13836348,
           0.5463327 , 0.33181095, 0.45686185, 0.7568008 , 1.8688486 ,
           0.51444894, 1.1470054 , 0.56240654, 0.77783823, 0.47601438,
           0.8391931 , 0.9542949 , 0.52233857, 2.049638  , 0.9337821 ,
           1.5828538 , 1.1982977 , 0.5996851 , 0.34287202, 1.302322  ,
           0.5765392 , 0.5861906 , 0.9636896 , 1.3210725 , 0.29344237,
           1.5826114 , 2.2057626 , 0.73928225, 1.2017496 , 0.29615378,
           1.1129823 , 0.285105  , 0.9201316 , 0.7032695 , 1.9016705 ,
           0.19354403, 1.4168835 , 0.73398304, 1.1979619 , 1.232079  ,
           0.5940706 , 1.5156059 , 1.9105101 , 0.2790072 , 1.4457306 ,
           1.5527472 , 0.41600168, 0.9682647 , 0.5611768 , 2.1933482 ,
           1.3341204 , 1.3705808 , 1.7518663 , 0.59243715, 0.52366143,
           0.5632213 , 1.650192  , 0.5890905 , 1.5205675 , 1.0699294 ,
           0.73326874, 1.835554  , 1.7176813 , 1.743949  , 0.9451206 ,
           0.08180296, 0.49115348, 1.4235613 , 1.1388881 , 0.38673675,
           1.1898067 , 1.0973556 , 0.8509266 , 0.67183864, 0.8128478 ,
           1.3921927 , 1.4698129 , 0.72021025, 1.3031769 , 0.32609248,
           1.210638  , 0.42013288, 2.242836  , 1.3876585 , 0.78741646,
           0.29916453, 2.1097922 , 1.3746064 , 0.26279032, 0.3636477 ,
           0.3132515 , 2.2058485 , 0.98319566, 0.5067608 , 1.0027157 ,
           0.849437  , 1.4602329 , 1.3697542 , 1.2450745 , 2.1913164 ,
           1.6064138 , 0.93563014, 1.1443644 , 0.33537817, 0.8218638 ,
           0.70383537, 0.90621483, 1.2114272 , 1.0715214 , 0.9545132 ,
           1.065865  , 1.2047305 , 0.45711076, 0.66132927, 0.31336808,
           0.17441905, 2.043468  , 0.17591262, 0.30481195, 0.470649  ,
           0.7549343 , 0.43365705, 1.1348412 , 2.1660342 , 0.6524892 ,
           0.8562118 , 0.14503407, 0.47894454, 2.1233435 , 1.7707397 ,
           0.21947038, 2.0594726 , 2.201642  , 1.4173121 , 0.5161568 ,
           0.8191002 , 2.1764154 , 0.65915453, 0.30754578, 1.3682023 ,
           0.13578796, 1.7152648 , 1.2880605 , 1.4056685 , 0.24158561,
           0.28185225, 0.7170991 , 1.9746878 , 0.32469094, 0.4641285 ,
           2.1328204 , 0.7931153 , 0.71383107, 2.0965385 , 0.2099607 ,
           1.1762638 , 0.35085678, 0.37228823, 1.5243781 , 1.7920742 ,
           1.768083  , 2.241077  , 0.6705948 , 1.2002186 , 1.842099  ,
           0.87937343, 0.29588675, 0.9430428 , 0.900398  , 1.7555468 ,
           0.72097045, 1.2434773 , 1.4596146 , 1.3867974 , 1.0127914 ,
           0.21674824, 1.9922367 , 2.217565  , 0.4885862 , 1.1067638 ,
           0.6340991 , 1.6274004 , 1.7903895 , 2.2525907 , 1.1592224 ,
           0.6944686 , 0.6926007 , 0.9301318 , 0.16400123, 0.9422506 ,
           1.0048239 , 1.2398766 , 1.9664532 , 0.7971904 , 1.1084152 ,
           0.3231094 , 0.8011063 , 0.16412175, 1.3823427 , 0.3111242 ,
           0.66980475, 2.1324353 , 0.36510038, 0.42826843, 0.06771374,
           1.0254345 , 0.90034324, 1.0493418 , 0.14361739, 0.17504692,
           1.0162743 , 1.7410797 , 0.5967882 , 0.95617354, 0.7685839 ,
           1.572384  , 0.9912834 , 1.8263315 , 2.157621  , 0.3042667 ,
           0.5038768 , 1.7457547 , 2.29274   , 0.483999  , 1.3042163 ,
           0.17721963, 0.37443542, 2.2124104 , 0.707572  , 1.8924183 ,
           1.1780736 , 0.19370699, 1.0474126 , 0.29545975, 1.4300001 ,
           0.44004273, 0.9724295 , 0.9186385 , 0.17414916, 0.63105214,
           1.7826929 , 0.6058014 , 2.2586625 , 0.83745164, 0.2669332 ,
           1.1646124 , 0.975093  , 2.257729  , 0.4385332 , 0.4218998 ,
           0.30188823, 2.0057054 , 0.26030195, 2.2154443 , 0.89930093,
           1.7500795 , 1.3424792 , 0.48091793, 0.5691763 , 0.8420398 ,
           1.2838188 , 1.7509391 , 0.40002012, 1.4081954 , 0.72034645,
           2.2843187 , 0.57547367, 0.30207837, 1.3668653 , 1.80824   ,
           0.6391039 , 0.40295982, 0.7391731 , 0.7412892 , 2.1955771 ,
           0.3370936 , 0.9087808 , 0.38845444, 0.30060232, 0.475186  ,
           0.5246581 , 0.5085225 , 1.1101241 , 1.1015421 , 0.1892674 ,
           0.5671846 , 0.787998  , 0.27489257, 1.9335248 , 0.4364879 ,
           0.7695458 , 0.9013622 , 1.2346406 , 0.32413328, 0.9574801 ,
           0.3013389 , 0.70873785, 0.52307767, 0.27721393, 0.5818955 ,
           2.072917  , 0.35308433, 0.55400753, 0.33471656, 0.20966005,
           0.60201263, 0.2505467 , 1.1141411 , 0.2803557 , 1.5231757 ,
           2.0144129 , 1.6401707 , 0.73318845, 1.0635017 , 1.200102  ,
           0.3425616 , 0.6631957 , 0.2206223 , 1.3993084 , 0.35734427,
           1.4942942 , 1.1615012 , 1.3075851 , 1.0389373 , 1.223887  ,
           1.0996472 , 0.6812018 , 2.0574875 , 1.3219174 , 1.1511052 ,
           0.34176445, 1.3045671 , 0.87984765, 0.32177424, 0.3763888 ,
           0.8202743 , 0.29169178, 0.31402266, 0.3444091 , 0.27668357,
           0.2959299 , 1.5235835 , 1.5358413 , 0.67979324, 0.30667782,
           0.30161166, 1.9743047 , 2.2682843 , 1.4258395 , 0.9695651 ,
           0.5671768 , 1.0727217 , 0.46502316, 0.34059918, 0.23844993,
           0.81917554, 0.16491187, 2.1633694 , 1.7594773 , 0.5566076 ,
           1.6650255 , 0.21200144, 0.45102406, 0.38877988, 1.2053446 ,
           0.34152937, 1.5321474 , 0.29181147, 0.4799868 , 1.2575852 ,
           2.1957371 , 0.34061432, 1.1815947 , 0.84324145, 0.31398714,
           0.51589644, 1.3904884 , 1.5146952 , 0.98825103, 1.3888367 ,
           1.2802653 , 0.8861996 , 0.15926027, 0.9070977 , 0.7667237 ,
           0.7266838 , 0.14852488, 0.69021386, 0.2171812 , 0.28067517,
           0.65013725, 0.73057395, 1.2387667 , 0.2594204 , 2.2343667 ,
           0.6506754 , 1.7647612 , 1.5926791 , 0.47002256, 0.30704796,
           1.5875505 , 0.5977346 , 0.30942369, 0.43926072, 1.4083784 ,
           0.7819265 , 1.6856236 , 0.1792171 , 1.3056164 , 0.76581246,
           0.25475788, 1.1734301 , 1.6756759 , 0.28719735, 0.22969246,
           0.6003391 , 0.17246795, 0.82271415, 0.527624  , 0.20761871,
           1.2142472 , 0.5452797 , 0.28257442, 0.5789856 , 0.34235108,
           1.0498717 , 0.6847771 , 1.140883  , 0.61571866, 0.49521637,
           0.9500967 , 1.5125798 , 0.28669   , 0.35282803, 1.7318455 ,
           0.47037816, 2.2011223 , 0.9508046 , 0.36787045, 0.29350114,
           0.488513  , 0.332191  , 1.3395056 , 0.2969067 , 1.338268  ,
           2.1542578 , 0.7112359 , 0.34530675, 2.0404677 , 0.6228099 ,
           0.1826657 , 0.09311604, 1.8493752 , 1.7863531 , 0.3320719 ,
           1.882168  , 0.56478727, 1.2733663 , 0.30870986, 0.6073284 ,
           0.42434287, 1.2378035 , 0.5975763 , 1.3544034 , 0.9600128 ,
           0.9768367 , 0.34299016, 0.8634549 , 0.23797214, 0.11863601,
           1.0072093 , 1.1916834 , 1.0649809 , 1.4154159 , 1.3555586 ,
           1.34472   , 1.6318725 , 1.8455282 , 0.39839816, 1.6346661 ,
           1.6372824 , 2.0320172 , 0.5530062 , 0.34578633, 1.1159678 ,
           1.2040006 , 0.27654958, 1.1554179 , 0.7724498 , 0.5336855 ,
           0.7548364 , 2.108087  , 0.2729175 , 0.58194035, 0.33421624,
           0.5175764 , 0.48491   , 1.8840697 , 0.7763976 , 0.23653305,
           0.414626  , 0.42791784, 2.0598073 , 0.43945396, 1.5376469 ,
           0.5895858 , 1.0193623 , 0.6483196 , 1.6718389 , 0.52963954,
           0.3067937 , 1.0020282 , 0.9428348 , 0.2929634 , 1.5957819 ,
           1.0945596 , 1.397035  , 1.5963764 , 1.0290524 , 0.22328591,
           0.70576704, 0.22334087, 0.8895134 , 1.913652  , 1.8872316 ,
           1.1563677 , 0.551071  , 0.74864054, 0.19263005, 0.59278095,
           1.7926812 , 1.5917898 , 0.891467  , 0.33411133, 2.1682367 ,
           0.50696313, 0.25047553, 0.9364835 , 0.2743677 , 0.61948425,
           0.29832304, 0.31371355, 1.2902012 , 1.9265969 , 1.4590098 ,
           0.41096365, 0.2730738 , 0.48057532, 1.2671596 , 2.2282958 ,
           0.29613948, 1.2573075 , 1.7920887 , 0.6521886 , 1.3949696 ,
           0.20531785, 1.2290087 , 0.5943552 , 1.4014667 , 1.1682425 ,
           0.7588887 , 2.078631  , 1.2710853 , 1.7710841 , 0.77392304,
           1.7186058 , 0.9631001 , 1.7394538 , 0.33084118, 0.78578913,
           0.64480174, 1.0032451 , 1.4184198 , 0.43717694, 1.0607342 ,
           0.6782961 , 1.9529852 , 0.75885445, 0.39110053, 1.1039946 ,
           0.9746763 , 2.2530441 , 1.4716698 , 0.33372307, 0.39481735,
           0.74744   , 0.33060288, 1.1515912 , 0.7316464 , 1.5216072 ],
          dtype=float32), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 235
    Process 0 estimating PZ PDF for rows 0 - 235
    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighEstimator
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x55da0a1d4c10>, 'bestsig': np.float64(0.06777777777777778), 'nneigh': 7, 'truezs': array([0.7109789 , 0.684234  , 0.24987125, 1.7519494 , 0.47846103,
           0.33473837, 0.50826055, 0.20228612, 0.6769716 , 1.3265918 ,
           1.7202301 , 0.87294644, 1.1373162 , 1.8772999 , 0.08898091,
           0.45685887, 1.9146216 , 0.22230697, 0.72233576, 0.5414646 ,
           1.1675177 , 0.6698597 , 1.0965223 , 2.0495567 , 1.0193622 ,
           1.833097  , 0.7588972 , 0.45343232, 2.1920176 , 1.216041  ,
           0.6625742 , 1.1159935 , 0.34173703, 1.0484967 , 0.94162714,
           1.6057475 , 0.7500063 , 0.71602046, 1.8231468 , 1.182788  ,
           0.25448942, 0.69419223, 1.6918525 , 0.9241866 , 0.74404824,
           1.2289805 , 0.21024513, 0.38334322, 1.0288469 , 1.911262  ,
           1.0010829 , 0.90705556, 1.1687441 , 1.6545238 , 0.5605775 ,
           0.784747  , 0.5195216 , 1.0560476 , 0.27467442, 0.23919082,
           1.2195003 , 0.21392596, 0.37882733, 0.68585557, 0.5490009 ,
           0.42329884, 1.555593  , 1.210321  , 0.81382084, 1.3347794 ,
           0.6548316 , 1.5099897 , 1.858717  , 1.9246919 , 0.7614285 ,
           1.9462461 , 0.7767667 , 0.32316816, 1.7933766 , 0.81352246,
           1.1640702 , 1.1437674 , 0.69064885, 0.63813937, 0.5217388 ,
           1.2677696 , 1.1091589 , 0.25915003, 2.0554237 , 1.523815  ,
           1.3465053 , 1.7464467 , 0.3149128 , 0.7638346 , 0.25076175,
           0.32721925, 1.2614264 , 1.0190805 , 0.26458716, 0.52258766,
           0.801554  , 1.8529932 , 1.4897779 , 1.8443738 , 1.3805554 ,
           1.9923592 , 0.4518503 , 0.2975279 , 1.00295   , 0.6558908 ,
           0.9030685 , 0.30417752, 0.5760741 , 1.3896191 , 1.5866966 ,
           1.0772498 , 0.3231529 , 0.9319731 , 1.8586497 , 0.19883513,
           0.22013259, 0.33062005, 0.8908531 , 1.6711466 , 0.54070127,
           1.9315952 , 1.7293675 , 0.49456823, 1.5056812 , 1.8751185 ,
           2.096883  , 0.5033921 , 0.45904768, 0.84505564, 1.4364039 ,
           0.7246009 , 1.2622792 , 1.5738457 , 0.03295648, 2.197969  ,
           0.1921562 , 2.258078  , 1.2803447 , 1.2486078 , 0.3592161 ,
           1.8083658 , 1.3751208 , 0.783929  , 0.7064227 , 1.6213154 ,
           0.5070284 , 1.2207518 , 1.0689307 , 0.08825755, 1.1680539 ,
           0.12144101, 0.98300195, 1.6140265 , 1.6799926 , 0.11701906,
           1.2279174 , 1.6100335 , 0.4352889 , 1.8487916 , 0.4424889 ,
           1.6828479 , 1.2518253 , 1.2403201 , 0.761552  , 1.0323439 ,
           0.69296837, 0.81484187, 1.2265118 , 2.0952249 , 0.436754  ,
           0.48351014, 1.0136161 , 2.2209527 , 0.3960607 , 2.081249  ,
           1.0378922 , 0.6808912 , 1.1873897 , 0.5193564 , 0.32013834,
           0.34706223, 1.219157  , 0.646887  , 2.1160793 , 1.2688411 ,
           1.414916  , 1.3863642 , 2.0398643 , 0.7865466 , 1.1997736 ,
           0.8022462 , 0.85921365, 0.30757582, 0.8409442 , 1.2399282 ,
           0.35271335, 1.0404111 , 0.7396241 , 1.5488262 , 1.6761452 ,
           0.5482834 , 1.9496062 , 1.5166888 , 0.3129294 , 1.7768048 ,
           0.8412742 , 1.0084713 , 1.5489537 , 0.48848116, 0.14496863,
           0.23794246, 0.3788036 , 0.46359015, 0.62428343, 1.9253665 ,
           1.0352235 , 0.26477253, 0.66503334, 0.26914525, 0.34013057,
           0.2453798 , 0.34928298, 0.52387416, 1.6353551 , 0.20927262,
           1.7007978 , 1.2357035 , 0.78999966, 0.21197438, 0.34859014,
           1.3615644 , 0.49246132, 0.6402361 , 1.0550171 , 0.4397025 ,
           0.06682682, 0.2235887 , 0.6409944 , 1.1568516 , 0.34310257,
           2.1020665 , 0.6676136 , 0.2892561 , 1.0816255 , 1.1975572 ,
           0.21113086, 0.7874545 , 1.6710181 , 1.641495  , 0.47862422,
           1.1729175 , 0.26468492, 0.66300887, 0.8555204 , 1.021383  ,
           0.8085413 , 0.798252  , 0.4025681 , 0.448745  , 0.3244543 ,
           1.3469714 , 1.0412824 , 0.1996845 , 0.28143764, 1.0004673 ,
           1.8122331 , 1.2829037 , 0.12390375, 0.7586328 , 1.3448102 ,
           0.17554522, 1.2371709 , 1.6698164 , 0.76215774, 0.71107644,
           2.246666  , 0.69113314, 0.3506186 , 1.1662769 , 0.38393402,
           0.4805318 , 0.38080978, 1.7821834 , 0.1840899 , 0.6428973 ,
           1.2902075 , 1.3799711 , 1.4608349 , 0.14815784, 0.7778674 ,
           0.4101075 , 2.1689503 , 2.165155  , 0.4863118 , 1.0381601 ,
           1.6407136 , 0.67955685, 0.5607626 , 0.2334609 , 1.9129939 ,
           0.6962531 , 0.5966457 , 0.9401247 , 1.2061448 , 1.1276329 ,
           0.23477018, 0.9950513 , 0.9737336 , 0.2295376 , 0.78468144,
           1.6631842 , 1.2192425 , 0.28074265, 0.95340025, 2.2354558 ,
           1.7337321 , 1.5350442 , 1.5514237 , 0.5914608 , 0.30340016,
           0.71777415, 2.157378  , 0.6288412 , 0.06341338, 1.8084018 ,
           0.90526235, 1.3380194 , 0.9498703 , 0.4846537 , 0.8246906 ,
           1.7548978 , 0.75522506, 0.6282366 , 1.0004065 , 1.4370538 ,
           0.32454205, 0.5432165 , 0.3164065 , 0.57056123, 1.8136976 ,
           1.4192182 , 1.4117311 , 0.84531534, 2.0079808 , 0.3490708 ,
           0.254853  , 1.3801321 , 0.7821099 , 0.87743366, 0.2606058 ,
           0.52621615, 1.6324197 , 1.2050378 , 0.337636  , 0.67370814,
           1.3140535 , 1.6535616 , 0.7328976 , 0.26897717, 2.1861548 ,
           0.2761942 , 0.72272885, 0.8308658 , 0.41968846, 1.0286633 ,
           0.8081778 , 0.18077457, 0.2731949 , 2.0020351 , 1.5735263 ,
           1.7204025 , 0.7102272 , 0.25153983, 0.28826392, 0.32783568,
           2.1266177 , 0.8596837 , 0.79027414, 1.4484913 , 1.8321487 ,
           1.2935405 , 1.7014916 , 0.82449925, 1.289315  , 0.7623329 ,
           0.946247  , 0.44006944, 1.5548165 , 0.5012544 , 0.49273002,
           0.34791315, 1.1781442 , 1.2139969 , 0.32177198, 1.9339446 ,
           1.03222   , 1.1280472 , 2.1892319 , 0.5223257 , 0.48564768,
           0.71715933, 0.48294783, 0.24194098, 0.30748248, 1.6466292 ,
           0.764136  , 0.90872896, 1.3862547 , 1.6997713 , 0.825963  ,
           0.65866965, 1.587491  , 0.5147842 , 1.0834266 , 0.34743237,
           0.7781678 , 0.60572624, 1.1225178 , 1.5855931 , 0.68468964,
           0.2807387 , 0.14923072, 1.1438118 , 1.2508092 , 0.18814528,
           0.6471189 , 0.2442944 , 0.33424544, 0.30661714, 1.3162143 ,
           0.31114745, 1.8743091 , 0.6789422 , 1.5728709 , 0.46619964,
           0.4442432 , 0.37189484, 1.0652593 , 1.8779815 , 0.62861633,
           1.1786792 , 0.32380974, 1.1075592 , 1.4031479 , 0.73665506,
           1.8058543 , 0.68590575, 1.7401445 , 1.264687  , 2.0463743 ,
           1.3998047 , 0.4694059 , 0.39519513, 0.5710615 , 0.1873436 ,
           0.35757995, 0.7391028 , 0.23277509, 0.2373817 , 0.67686623,
           1.9920671 , 0.47692108, 0.58705056, 0.17037988, 1.6296208 ,
           0.09747732, 1.0740004 , 1.1446515 , 1.5588969 , 1.6577418 ,
           0.7235913 , 2.0230772 , 1.3389658 , 0.3568672 , 0.82200575,
           0.8137377 , 0.32149458, 2.262886  , 0.8974304 , 0.7102419 ,
           0.27450657, 1.0300833 , 0.38317144, 1.3828422 , 0.3850013 ,
           0.6407984 , 1.74429   , 2.071599  , 0.80022395, 1.4374872 ,
           0.3299694 , 0.16367114, 0.6610519 , 1.0559413 , 1.1803398 ,
           1.1252072 , 0.298823  , 1.3336755 , 1.1032159 , 0.99269354,
           1.6185905 , 1.2204965 , 1.1078162 , 1.8344034 , 0.7249699 ,
           0.26785254, 1.3186297 , 0.4636271 , 0.29694188, 1.0729576 ,
           0.2956568 , 0.24910796, 1.3125856 , 0.20031476, 1.8605608 ,
           1.216285  , 1.0338118 , 0.16625631, 0.6050299 , 1.8145348 ,
           0.37368417, 1.6761639 , 1.571375  , 0.29964614, 0.75524414,
           2.0574937 , 0.3185619 , 0.8659217 , 2.126754  , 2.0602818 ,
           1.380269  , 0.25527608, 0.28007066, 0.23432124, 0.61618555,
           1.0591956 , 0.36815393, 1.1343762 , 1.9245627 , 1.5464859 ,
           1.059783  , 0.41786635, 1.6939487 , 2.0981293 , 0.78470504,
           1.2277493 , 1.1570475 , 0.8899392 , 0.34040666, 1.1850611 ,
           1.5435475 , 1.4924233 , 0.30082107, 1.2135372 , 1.5336401 ,
           2.0479455 , 0.32986522, 0.1614908 , 1.717179  , 0.9428643 ,
           0.24671948, 0.84659   , 0.30413032, 1.8622115 , 0.7089313 ,
           0.5617754 , 0.6803961 , 1.9109143 , 1.2743742 , 1.9122435 ,
           0.32125604, 0.78256035, 1.215565  , 1.8659166 , 1.8824646 ,
           0.57487124, 1.903805  , 1.3982508 , 0.7146926 , 0.22318244,
           0.58959544, 0.22787619, 1.5225413 , 0.7986866 , 0.42288804,
           0.49832654, 0.8555077 , 0.35106885, 0.6315036 , 1.0201079 ,
           0.57720315, 0.4054165 , 0.41013336, 0.29973304, 0.21848059],
          dtype=float32), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 235
    Process 0 estimating PZ PDF for rows 0 - 235


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighEstimator
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x55da0cf41390>, 'bestsig': np.float64(0.075), 'nneigh': 4, 'truezs': array([1.43334293, 1.32375515, 0.87025988, 2.75354752, 1.2945056 ,
           1.13209355, 1.11237383, 0.14820874, 1.76498485, 0.13836348,
           0.54633272, 0.33181095, 0.45686185, 0.75680077, 1.86884856,
           0.51444894, 1.14700544, 0.56240654, 0.77783823, 0.47601438,
           0.83919311, 0.95429492, 0.52233857, 2.04963803, 0.9337821 ,
           1.58285379, 0.59968507, 0.34287202, 0.57653922, 0.58619058,
           1.63809873, 1.32107246, 0.29344237, 2.20576262, 0.73928225,
           1.20174956, 0.29615378, 1.11298227, 0.28510499, 0.92013162,
           0.70326948, 1.90167046, 0.19354403, 1.41688347, 0.73398304,
           1.19796193, 0.59407061, 1.51560593, 1.91051006, 0.2790072 ,
           0.41600168, 0.9682647 , 0.56117678, 1.33412039, 1.37058079,
           1.75186634, 0.59243715, 1.04694736, 0.56322128, 1.65019202,
           0.58909053, 1.52056754, 1.06992936, 0.73326874, 0.94512057,
           0.08180296, 0.49115348, 1.42356133, 1.13888812, 0.38673675,
           1.1898067 , 1.0973556 , 0.85092658, 0.67183864, 0.81284779,
           1.39219272, 1.46981287, 0.72021025, 0.32609248, 1.21063805,
           0.42013288, 2.242836  , 1.38765848, 1.40128634, 0.29916453,
           1.37460637, 0.26279032, 0.3636477 , 0.3132515 , 0.98319566,
           0.50676078, 1.69052792, 0.849437  , 1.46023285, 1.3697542 ,
           1.24507451, 2.19131637, 2.50156   , 0.93563014, 1.14436436,
           0.33537817, 0.82186377, 0.70383537, 0.90621483, 1.0715214 ,
           0.95451319, 1.06586504, 1.20473051, 0.45711076, 0.66132927,
           0.31336808, 0.17441905, 2.043468  , 0.17591262, 0.30481195,
           0.470649  , 1.35764854, 0.43365705, 1.1348412 , 0.65248919,
           0.85621178, 0.14503407, 0.9868729 , 2.12334347, 1.77073967,
           0.21947038, 2.05947256, 2.20164204, 1.41731215, 0.51615679,
           0.8191002 , 0.65915453, 0.30754578, 1.36820233, 0.13578796,
           1.28806055, 1.4056685 , 0.24158561, 0.28185225, 1.30681917,
           1.97468781, 0.32469094, 0.46412849, 0.79311532, 1.3024288 ,
           0.2099607 , 1.17626381, 0.35085678, 0.37228823, 1.52437806,
           0.67059481, 1.20021856, 1.84209895, 0.87937343, 0.29588675,
           0.94304281, 0.90039802, 0.72097045, 1.24347734, 1.45961463,
           1.38679743, 1.0127914 , 0.21674824, 1.99223673, 0.48858619,
           1.10676384, 0.63409913, 1.15922236, 0.69446862, 0.69260073,
           0.93013179, 0.16400123, 1.6092967 , 1.00482392, 1.96645319,
           0.79719043, 0.32310939, 0.80110627, 0.16412175, 1.3823427 ,
           0.31112421, 0.66980475, 2.13243532, 0.36510038, 0.42826843,
           0.06771374, 1.02543449, 0.90034324, 1.0493418 , 0.14361739,
           0.17504692, 1.01627433, 1.74107969, 0.59678823, 1.62800132,
           0.76858389, 0.99128342, 1.8263315 , 2.15762091, 0.30426669,
           0.50387681, 0.48399901, 1.30421627, 0.17721963, 0.37443542,
           2.21241045, 0.70757198, 0.19370699, 1.04741263, 0.29545975,
           1.43000007, 0.44004273, 0.97242951, 0.91863853, 0.17414916,
           0.63105214, 0.6058014 , 2.25866246, 0.83745164, 0.2669332 ,
           1.16461241, 0.97509301, 0.43853319, 0.4218998 , 0.30188823,
           0.26030195, 0.89930093, 1.34247923, 0.48091793, 0.56917632,
           1.47466955, 1.28381884, 0.40002012, 1.40819538, 0.72034645,
           0.57547367, 0.30207837, 1.36686528, 1.80824006, 0.63910389,
           0.40295982, 0.73917311, 0.7412892 , 0.33709359, 0.90878081,
           0.38845444, 0.30060232, 0.47518599, 0.52465808, 0.50852251,
           1.10154212, 0.59770911, 0.56718463, 0.78799802, 0.27489257,
           1.93352485, 0.43648791, 0.76954579, 0.90136218, 1.2346406 ,
           0.32413328, 0.95748007, 0.30133891, 0.70873785, 0.52307767,
           0.27721393, 1.12518128, 0.81778729, 0.55400753, 0.33471656,
           0.20966005, 0.60201263, 0.25054669, 1.11414111, 0.28035569,
           0.73318845, 1.06350172, 1.20010197, 0.3425616 , 0.66319573,
           0.2206223 , 1.39930844, 0.35734427, 1.49429417, 1.16150117,
           1.30758512, 1.73918949, 1.22388697, 0.68120182, 1.32191741,
           1.15110517, 0.34176445, 1.3045671 , 0.87984765, 0.32177424,
           0.37638879, 0.82027429, 0.29169178, 0.31402266, 0.34440911,
           0.27668357, 0.29592991, 1.53584135, 0.67979324, 0.30667782,
           0.30161166, 1.42583954, 0.96956509, 0.56717682, 1.07272172,
           0.46502316, 0.34059918, 0.23844993, 0.81917554, 0.16491187,
           0.5566076 , 1.66502547, 0.21200144, 0.45102406, 0.38877988,
           1.20534456, 0.34152937, 1.53214741, 0.29181147, 0.47998679,
           1.25758517, 0.34061432, 1.18159473, 0.84324145, 0.31398714,
           0.51589644, 1.39048839, 1.51469517, 0.98825103, 1.38883674,
           1.28026533, 0.88619959, 0.15926027, 0.9070977 , 0.76672369,
           0.7266838 , 0.14852488, 0.69021386, 0.21718121, 0.28067517,
           0.65013725, 0.73057395, 1.23876667, 0.25942039, 0.65067542,
           0.47002256, 0.30704796, 0.59773457, 0.30942369, 0.43926072,
           0.78192651, 1.68562365, 0.58420714, 1.30561638, 0.76581246,
           0.25475788, 0.28719735, 0.22969246, 0.60033911, 0.17246795,
           0.82271415, 0.52762401, 0.20761871, 0.54527968, 0.28257442,
           0.57898557, 0.34235108, 1.04987168, 0.68477708, 1.14088297,
           0.61571866, 0.49521637, 0.95009673, 1.5125798 , 0.28669   ,
           0.35282803, 1.7318455 , 0.47037816, 0.95080459, 0.36787045,
           0.29350114, 0.48851299, 0.33219099, 1.33950555, 0.29690671,
           1.33826804, 0.71123588, 0.34530675, 2.04046774, 0.62280989,
           0.18266571, 0.09311604, 0.3320719 , 0.56478727, 1.27336633,
           0.30870986, 0.60732841, 0.42434287, 1.23780346, 0.59757632,
           1.35440338, 0.96001279, 0.97683668, 0.34299016, 0.86345488,
           0.23797214, 0.11863601, 1.0072093 , 1.19168341, 1.06498086,
           1.35555863, 1.34472001, 1.63187253, 1.84552824, 0.39839816,
           1.63728237, 0.55300617, 0.34578633, 1.20400059, 0.27654958,
           1.15541792, 0.77244979, 0.53368551, 0.75483638, 0.27291751,
           0.58194035, 0.79243915, 0.5175764 , 0.48491001, 1.88406968,
           0.77639759, 0.66120766, 0.414626  , 0.42791784, 2.0598073 ,
           0.43945396, 0.58958578, 1.01936233, 0.6483196 , 1.67183888,
           0.52963954, 0.30679369, 1.00202823, 0.94283479, 0.29296339,
           1.59578192, 1.09455955, 1.59637642, 1.02905238, 0.22328591,
           1.29159526, 0.22334087, 1.5384474 , 2.91431588, 1.88723159,
           1.15636766, 0.55107099, 0.74864054, 0.19263005, 0.59278095,
           0.89146698, 0.33411133, 0.50696313, 0.25047553, 0.9364835 ,
           0.71203623, 0.61948425, 0.29832304, 0.31371355, 1.29020119,
           0.41096365, 0.27307379, 0.48057532, 1.26715958, 0.29613948,
           0.6521886 , 1.39496958, 0.20531785, 1.22900867, 0.59435523,
           1.40146673, 1.16824245, 0.75888872, 2.07863092, 1.27108526,
           1.77108407, 0.77392304, 0.96310008, 0.33084118, 0.78578913,
           0.64480174, 1.00324512, 1.41841984, 0.43717694, 1.06073415,
           0.67829609, 1.95298517, 0.75885445, 0.39110053, 1.10399461,
           0.97467631, 2.25304413, 1.47166979, 0.33372307, 0.39481735,
           0.74743998, 0.78758482, 1.15159118, 0.73164642]), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 235
    Process 0 estimating PZ PDF for rows 0 - 235
    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighEstimator
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x55da042d8e30>, 'bestsig': np.float64(0.075), 'nneigh': 3, 'truezs': array([0.87025988, 1.2945056 , 1.13209355, 0.13836348, 0.45686185,
           0.75680077, 0.51444894, 1.14700544, 0.47601438, 0.83919311,
           0.52233857, 0.59968507, 0.34287202, 0.57653922, 1.63809873,
           0.29344237, 0.73928225, 0.29615378, 1.11298227, 0.92013162,
           0.19354403, 0.59407061, 0.56117678, 1.33412039, 0.59243715,
           1.04694736, 0.56322128, 0.58909053, 0.94512057, 0.49115348,
           1.13888812, 1.1898067 , 0.85092658, 0.67183864, 0.81284779,
           0.72021025, 0.32609248, 1.21063805, 1.40128634, 0.29916453,
           0.3636477 , 0.3132515 , 0.50676078, 1.69052792, 1.46023285,
           1.24507451, 0.93563014, 1.14436436, 0.82186377, 0.70383537,
           0.90621483, 0.95451319, 1.06586504, 1.20473051, 0.45711076,
           0.66132927, 0.31336808, 0.17441905, 0.30481195, 0.470649  ,
           1.35764854, 0.65248919, 0.9868729 , 1.41731215, 0.51615679,
           0.65915453, 0.30754578, 0.13578796, 0.24158561, 0.28185225,
           0.46412849, 1.3024288 , 0.2099607 , 0.35085678, 1.52437806,
           0.67059481, 1.20021856, 0.94304281, 0.90039802, 0.21674824,
           0.48858619, 0.69446862, 0.93013179, 1.6092967 , 1.00482392,
           0.79719043, 0.80110627, 0.16412175, 0.36510038, 0.42826843,
           0.06771374, 1.02543449, 0.90034324, 1.0493418 , 1.01627433,
           0.59678823, 1.62800132, 0.30426669, 0.50387681, 0.48399901,
           0.37443542, 0.70757198, 0.19370699, 0.44004273, 0.97242951,
           0.63105214, 0.83745164, 0.2669332 , 0.4218998 , 0.26030195,
           1.34247923, 0.48091793, 0.56917632, 0.40002012, 0.72034645,
           0.30207837, 0.63910389, 0.73917311, 0.7412892 , 0.33709359,
           0.30060232, 0.47518599, 0.59770911, 0.56718463, 0.78799802,
           0.27489257, 0.43648791, 1.2346406 , 0.95748007, 0.30133891,
           0.55400753, 0.33471656, 0.60201263, 0.25054669, 1.11414111,
           0.73318845, 0.2206223 , 1.22388697, 0.68120182, 1.32191741,
           1.15110517, 0.34176445, 0.37638879, 0.82027429, 0.34440911,
           0.27668357, 0.29592991, 0.67979324, 0.30161166, 1.07272172,
           0.46502316, 0.34059918, 0.81917554, 0.45102406, 0.38877988,
           1.20534456, 0.47998679, 1.25758517, 0.34061432, 0.84324145,
           0.31398714, 0.51589644, 1.51469517, 0.98825103, 1.28026533,
           0.88619959, 0.14852488, 0.28067517, 0.65013725, 0.73057395,
           0.25942039, 0.65067542, 0.47002256, 0.59773457, 0.30942369,
           0.43926072, 0.78192651, 0.58420714, 0.76581246, 0.25475788,
           0.22969246, 0.60033911, 0.17246795, 0.82271415, 0.52762401,
           0.34235108, 1.14088297, 0.61571866, 0.49521637, 0.95009673,
           0.28669   , 0.35282803, 0.47037816, 0.36787045, 0.48851299,
           0.33219099, 1.33950555, 0.34530675, 0.62280989, 0.18266571,
           0.09311604, 0.3320719 , 0.56478727, 0.60732841, 0.59757632,
           1.35440338, 0.96001279, 0.97683668, 0.34299016, 0.11863601,
           1.19168341, 1.35555863, 1.20400059, 0.27654958, 1.15541792,
           0.77244979, 0.75483638, 0.27291751, 0.5175764 , 0.48491001,
           0.77639759, 0.6483196 , 0.52963954, 0.30679369, 0.22328591,
           1.15636766, 0.55107099, 0.74864054, 0.19263005, 0.59278095,
           0.89146698, 0.33411133, 0.50696313, 0.9364835 , 0.61948425,
           0.29832304, 1.29020119, 0.48057532, 1.26715958, 0.29613948,
           0.6521886 , 1.16824245, 0.75888872, 0.77392304, 0.78578913,
           0.64480174, 1.00324512, 0.43717694, 0.75885445, 0.39110053,
           0.97467631, 1.47166979, 0.74743998, 0.78758482, 0.73164642]), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 235
    Process 0 estimating PZ PDF for rows 0 - 235


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator


Now let’s take a look at what the output of the estimation stage
actually looks like. Most estimation stages output an ``Ensemble``,
which is a data structure from the package ``qp``. For more information,
see `the qp
documentation <https://qp.readthedocs.io/en/main/user_guide/datastructure.html>`__.

An Ensemble acts a bit like a ```scipy`` probability
distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`__,
so you can easily calculate statistics from it, such as the mean,
median, or mode of all of the photo-z PDFs. For a full list of the
functions available see `the list of methods
here <https://qp.readthedocs.io/en/main/user_guide/methods.html>`__.

In this case, we’re using an ``Ensemble`` to hold a redshift
distribution for each of the galaxies we’re estimating. If you’d like to
see the underlying data points that make up the ``Ensemble``\ ’s PDFs,
you can take a look at the three data dictionaries it’s made up of: -
``.metadata``: Contains information about the whole data structure, like
the Ensemble type, and any shared parameters such as the bins of
histograms. This is not per-object metadata. - ``.objdata``: The main
data points of the distributions for each object, where each object is a
row. - ``.ancil``: the optional dictionary, containing extra information
about each object. It can have arrays that have one or more data points
per distribution.

.. code:: ipython3

    # estimated_photoz contains the output of the KNN estimate function for each of our
    # parameter sets. Here we print out the result for just one of them. We can see that
    # the Ensemble has the same number of rows as galaxies that we input, and some number
    # of points per row
    print(estimated_photoz["lsst_error_model"])


.. parsed-literal::

    {'output': Ensemble(the_class=mixmod,shape=(235, 7))}


We can see that this algorithm outputs Ensembles of class ``mixmod``,
which are just combinations of Gaussians (for more info see the `qp
docs <https://qp.readthedocs.io/en/main/user_guide/parameterizations/mixmod.html>`__).

So each distribution in this Ensemble has a set of Gaussians that, added
together, make up the distribution. Each distribution is therefore
described by a set of means, weights, and standard deviations. The shape
portion of the print statement tells us two things: the first number is
the number of photo-z distributions, or galaxies, in this ``Ensemble``,
and the second number tells us how many Gaussians are combined to make
up each photo-z distribution.

Let’s take a look at what the different dictionaries look like for this
``Ensemble``:

.. code:: ipython3

    # this is the metadata dictionary of that output Ensemble
    print(estimated_photoz["lsst_error_model"]["output"].metadata)


.. parsed-literal::

    {'pdf_name': array([b'mixmod'], dtype='|S6'), 'pdf_version': array([0])}


.. code:: ipython3

    # this is the actual distribution data of that output Ensemble, which contains
    # the data points that describe each photometric redshift probability distribution
    print(estimated_photoz["lsst_error_model"]["output"].objdata)


.. parsed-literal::

    {'weights': array([[0.23629855, 0.15147082, 0.15088038, ..., 0.12592662, 0.11959536,
            0.0845446 ],
           [0.25903318, 0.14618087, 0.13981088, ..., 0.11317833, 0.1083862 ,
            0.10432916],
           [0.29664524, 0.16999572, 0.12086626, ..., 0.10369715, 0.09583377,
            0.09428205],
           ...,
           [0.17956374, 0.15587108, 0.14357747, ..., 0.13113376, 0.1303593 ,
            0.12812151],
           [0.20232662, 0.1720883 , 0.15061593, ..., 0.11939139, 0.11831609,
            0.10934293],
           [0.21170527, 0.15228697, 0.15037734, ..., 0.13314878, 0.10882771,
            0.09870863]], shape=(235, 7)), 'stds': array([[0.06777778, 0.06777778, 0.06777778, ..., 0.06777778, 0.06777778,
            0.06777778],
           [0.06777778, 0.06777778, 0.06777778, ..., 0.06777778, 0.06777778,
            0.06777778],
           [0.06777778, 0.06777778, 0.06777778, ..., 0.06777778, 0.06777778,
            0.06777778],
           ...,
           [0.06777778, 0.06777778, 0.06777778, ..., 0.06777778, 0.06777778,
            0.06777778],
           [0.06777778, 0.06777778, 0.06777778, ..., 0.06777778, 0.06777778,
            0.06777778],
           [0.06777778, 0.06777778, 0.06777778, ..., 0.06777778, 0.06777778,
            0.06777778]], shape=(235, 7)), 'means': array([[0.6944686 , 0.7548364 , 0.6506754 , ..., 0.73318845, 0.7391731 ,
            0.7588887 ],
           [1.2575852 , 2.2586625 , 1.0607342 , ..., 1.4056685 , 0.14361739,
            1.3746064 ],
           [0.29344237, 0.29613948, 0.3111242 , ..., 0.33411133, 2.29274   ,
            0.31402266],
           ...,
           [0.77392304, 0.47002256, 0.5890905 , ..., 0.72021025, 0.5611768 ,
            0.48057532],
           [0.98825103, 0.9574801 , 0.900398  , ..., 0.9636896 , 1.2573075 ,
            1.1320935 ],
           [0.33060288, 0.33471656, 0.34059918, ..., 0.45711076, 0.2505467 ,
            0.30161166]], shape=(235, 7), dtype=float32)}


Typically the ancillary data table includes a photo-z point estimate
derived from the PDFs, by default this is the mode of the distribution,
called ‘zmode’ in the ancillary dictionary below:

.. code:: ipython3

    # this is the ancillary dictionary of the output Ensemble, which in this case
    # contains the zmode, redshift, and distribution type
    print(estimated_photoz["lsst_error_model"]["output"].ancil)


.. parsed-literal::

    {'zmode': array([[0.71],
           [1.29],
           [0.3 ],
           [0.54],
           [0.28],
           [0.62],
           [1.4 ],
           [0.75],
           [0.68],
           [1.19],
           [0.68],
           [1.12],
           [1.13],
           [0.65],
           [1.16],
           [0.78],
           [0.67],
           [0.86],
           [1.15],
           [0.21],
           [0.47],
           [1.13],
           [1.1 ],
           [1.06],
           [0.53],
           [0.71],
           [0.26],
           [0.35],
           [0.35],
           [0.61],
           [0.43],
           [0.74],
           [0.77],
           [0.83],
           [0.58],
           [0.5 ],
           [1.38],
           [0.28],
           [1.4 ],
           [0.99],
           [0.53],
           [0.93],
           [1.44],
           [0.32],
           [0.31],
           [0.57],
           [0.81],
           [0.76],
           [1.99],
           [0.51],
           [0.37],
           [0.95],
           [0.25],
           [0.98],
           [0.44],
           [0.53],
           [0.71],
           [1.27],
           [0.15],
           [0.28],
           [0.78],
           [0.32],
           [1.23],
           [1.08],
           [0.23],
           [1.17],
           [1.09],
           [1.41],
           [0.32],
           [0.32],
           [0.55],
           [0.58],
           [0.81],
           [1.2 ],
           [0.77],
           [0.31],
           [0.86],
           [0.23],
           [0.99],
           [0.52],
           [0.93],
           [1.43],
           [0.36],
           [0.28],
           [0.59],
           [0.65],
           [0.27],
           [0.49],
           [0.52],
           [0.43],
           [0.24],
           [0.26],
           [0.59],
           [0.71],
           [0.29],
           [0.69],
           [0.81],
           [0.83],
           [0.88],
           [1.12],
           [0.87],
           [0.34],
           [0.33],
           [0.33],
           [0.21],
           [0.38],
           [0.21],
           [0.17],
           [0.76],
           [0.32],
           [0.41],
           [0.43],
           [0.61],
           [1.36],
           [0.24],
           [0.78],
           [0.4 ],
           [1.58],
           [0.52],
           [0.25],
           [0.98],
           [1.18],
           [1.11],
           [0.95],
           [0.31],
           [0.97],
           [1.36],
           [1.54],
           [0.78],
           [0.17],
           [0.73],
           [1.1 ],
           [1.31],
           [0.64],
           [0.28],
           [1.37],
           [0.98],
           [0.3 ],
           [0.4 ],
           [2.15],
           [0.76],
           [0.77],
           [0.32],
           [0.67],
           [0.34],
           [0.97],
           [0.77],
           [0.45],
           [0.25],
           [0.68],
           [0.29],
           [0.33],
           [0.28],
           [0.99],
           [0.94],
           [1.4 ],
           [1.22],
           [0.82],
           [0.56],
           [0.48],
           [1.39],
           [0.24],
           [0.52],
           [0.73],
           [0.32],
           [0.32],
           [0.73],
           [0.92],
           [0.77],
           [0.68],
           [0.68],
           [0.59],
           [0.58],
           [1.11],
           [0.59],
           [0.34],
           [0.31],
           [1.14],
           [0.97],
           [0.67],
           [0.72],
           [0.67],
           [0.27],
           [0.27],
           [0.56],
           [0.23],
           [0.99],
           [1.2 ],
           [0.68],
           [1.29],
           [0.99],
           [0.77],
           [1.  ],
           [0.31],
           [0.44],
           [0.66],
           [0.31],
           [0.23],
           [0.33],
           [0.69],
           [1.3 ],
           [0.52],
           [0.23],
           [0.99],
           [0.55],
           [1.23],
           [0.43],
           [0.75],
           [0.26],
           [0.3 ],
           [0.25],
           [0.73],
           [0.99],
           [0.34],
           [0.37],
           [0.75],
           [0.99],
           [1.1 ],
           [0.96],
           [0.29],
           [0.84],
           [0.7 ],
           [0.33],
           [0.73],
           [0.58],
           [0.7 ],
           [0.21],
           [0.65],
           [0.8 ],
           [0.34],
           [0.48],
           [0.45],
           [0.52],
           [0.97],
           [0.32]]), 'redshift': 0      0.684234
    1      1.751949
    2      0.334738
    3      0.508261
    4      0.202286
             ...   
    230    0.498327
    231    0.351069
    232    0.631504
    233    1.020108
    234    0.410133
    Name: redshift, Length: 235, dtype: float64, 'distribution_type': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}


The nice thing about the ``Ensemble`` is that you don’t actually need to
access these dictionaries at all. You can just use the ``.pdf()``
method, which calculates the value(s) of the distribution(s) at any
redshift(s) you provide, and works the same for all types of
``Ensemble``. This is quite useful for plotting, since the different
estimation algorithms can store data in different ways, which makes
plotting the actual data points more confusing, but we can always use
the ``.pdf()`` method.

Now let’s use this method to plot one redshift PDF from each of our four
estimated redshift distribution datasets to compare them:

.. code:: ipython3

    xvals = np.linspace(0, 3, 200)  # we want to cover the whole available redshift space
    gal_id = 100 # the galaxy we'll look at 
    for key, df in estimated_photoz.items():
        plt.plot(xvals, df["output"][gal_id].pdf(xvals), label=key)
    
    # plot the true redshift
    plt.axvline(
        targ_data_orig["output"]["redshift"].iloc[gal_id],
        color="k",
        ls="--",
        label="true redshift",
    )
    
    plt.legend(loc="best", title="calibration dataset")
    plt.xlabel("redshift")
    plt.ylabel("p(z)")




.. parsed-literal::

    Text(0, 0.5, 'p(z)')




.. image:: Exploring_the_Effects_of_Degraders_files/Exploring_the_Effects_of_Degraders_66_1.png


This plot shows us the estimated photo-z PDF for the first galaxy with
each of the different calibration sets, compared to the redshift from
the “true” target dataset we sampled at the beginning.

Plotting one distribution at a time isn’t the best way to get a sense of
how the whole set of galaxy redshift distributions changes, so let’s
summarize these distributions. This will give us a sense of how all of
the estimated redshift distributions change with each different
calibration data set. There are a number of summarizing algorithms, but
here we’ll use two of the most basic:

1. `Point Estimate
   Histogram <https://rail-hub.readthedocs.io/en/latest/source/rail_stages/estimation.html#point-estimate-histogram>`__:
   This algorithm creates a histogram of all the point estimates of the
   photometric redshifts. By default, the point estimate used is
   ``zmode``, which is usually found in the ancillary dictionary of the
   distributions.
2. `Naive
   Stacking <https://rail-hub.readthedocs.io/en/latest/source/rail_stages/estimation.html#naive-stacking>`__:
   This algorithm stacks the PDFs of the estimated photometric redshifts
   together and normalizes the stacked distribution.

.. code:: ipython3

    # set up dictionaries for output
    point_est_dict = {}
    naive_stack_dict = {}
    
    for key, item in estimated_photoz.items():
    
        # get the summary of the point estimates
        point_estimate_ens = ri.estimation.algos.point_est_hist.point_est_hist_summarizer(
            input_data=item["output"]
        )
        point_est_dict[key] = point_estimate_ens
    
        # get a summary of the PDFs
        naive_stack_ens = ri.estimation.algos.naive_stack.naive_stack_summarizer(
            input_data=item["output"]
        )
        naive_stack_dict[key] = naive_stack_ens


.. parsed-literal::

    Inserting handle into data store.  input: None, PointEstHistSummarizer
    Process 0 running estimator on chunk 0 - 235


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  input: None, NaiveStackSummarizer
    Process 0 running estimator on chunk 0 - 235
    Inserting handle into data store.  output: inprogress_output.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  input: None, PointEstHistSummarizer
    Process 0 running estimator on chunk 0 - 235


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  input: None, NaiveStackSummarizer
    Process 0 running estimator on chunk 0 - 235
    Inserting handle into data store.  output: inprogress_output.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  input: None, PointEstHistSummarizer
    Process 0 running estimator on chunk 0 - 235


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  input: None, NaiveStackSummarizer
    Process 0 running estimator on chunk 0 - 235
    Inserting handle into data store.  output: inprogress_output.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  input: None, PointEstHistSummarizer
    Process 0 running estimator on chunk 0 - 235


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  input: None, NaiveStackSummarizer
    Process 0 running estimator on chunk 0 - 235
    Inserting handle into data store.  output: inprogress_output.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, NaiveStackSummarizer


Now let’s take a look at the output dictionaries for both these
functions for one of the distributions:

.. code:: ipython3

    print(point_est_dict["lsst_error_model"])
    print(naive_stack_dict["lsst_error_model"])


.. parsed-literal::

    {'output': Ensemble(the_class=hist,shape=(1000, 301)), 'single_NZ': Ensemble(the_class=hist,shape=(1, 301))}
    {'output': Ensemble(the_class=interp,shape=(1000, 302)), 'single_NZ': Ensemble(the_class=interp,shape=(1, 302))}


These functions output ``Ensembles``, just like the KNN estimation
algorithm. However, they output two separate ``Ensembles``: the
“single_NZ” one contains just one distribution, the actual stacked
distribution that has been created. The ‘output’ one contains a number
of bootstrapped distributions, to make further analysis easier.

We’re going to focus on the “single_NZ” distribution here. We’ll start
by plotting the point estimate summarized distributions for all of the
runs, which are histograms:

.. code:: ipython3

    # get bin centers and widths
    bin_width = (
        point_est_dict["lsst_error_model"]["single_NZ"].metadata["bins"][1]
        - point_est_dict["lsst_error_model"]["single_NZ"].metadata["bins"][0]
    )
    bin_centers = (
        point_est_dict["lsst_error_model"]["single_NZ"].metadata["bins"][:-1]
        + point_est_dict["lsst_error_model"]["single_NZ"].metadata["bins"][1:]
    ) / 2
    
    for key, df in point_est_dict.items():
        plt.bar(
            bin_centers,
            df["single_NZ"].objdata["pdfs"],
            width=bin_width,
            alpha=0.7,
            label=key,
        )
    
    plt.legend(loc="best")
    plt.xlabel("redshift")
    plt.ylabel("N(z)")




.. parsed-literal::

    Text(0, 0.5, 'N(z)')




.. image:: Exploring_the_Effects_of_Degraders_files/Exploring_the_Effects_of_Degraders_72_1.png


It’s a little difficult to see the differences between so many
distributions in this format, but you can get a sense that there are
some distinct differences in the distributions of redshifts.

Let’s plot the summarized distributions from the Naive Stacking
algorithm, which are smoothed distributions since they are created by
stacking the full photo-z PDFs instead of point estimates:

.. code:: ipython3

    for key, df in naive_stack_dict.items():
        plt.plot(
            df["single_NZ"].metadata["xvals"], df["single_NZ"].objdata["yvals"], label=key
        )
    
    plt.legend(loc="best")
    plt.xlabel("redshift")
    plt.ylabel("N(z)")




.. parsed-literal::

    Text(0, 0.5, 'N(z)')




.. image:: Exploring_the_Effects_of_Degraders_files/Exploring_the_Effects_of_Degraders_74_1.png


It’s a bit easier to see the differences between the distributions of
redshifts in this plot. We can see that the overall shape of the
distributions is the same, but there are some significant differences,
in particular at higher redshifts.

If you’d like to save these summarized distributions so you can use them
elsewhere, or compare them using your own algorithms, you likely want
them as just an array instead of an Ensemble. These Ensembles are of the
type “interp”, which means that they have their data already stored in a
grid of x and y values. So it’s easy to just pull those values out into
an array, like so:

.. code:: ipython3

    # returns the array of y values of the summarized photo-z distribution for all the galaxies 
    y_arr = naive_stack_dict["lsst_error_model"]["single_NZ"].objdata['yvals']
    type(y_arr) 




.. parsed-literal::

    numpy.ndarray



However, if you want to do this for any type of Ensemble, the method
that works the most consistently is to use the ``.pdf()`` function,
which will return the values of the PDF at a given set of redshifts. We
can use this to get a dictionary of arrays, one for each of the
distributions we’ve summarized, and turn it into a pandas DataFrame:

.. code:: ipython3

    import pandas as pd
    
    array_dict = {}
    z_grid_out = np.linspace(0,3,301) # create a set of z values to sample the PDFs on 
    # add the z grid into the dictionary
    array_dict["z_grid_values"] = z_grid_out
    
    # calculate the PDF values for each of the different distributions 
    for key, item in naive_stack_dict.items():
        array_dict[key] = item["single_NZ"].pdf(z_grid_out) 
    
    df_summarized_dist = pd.DataFrame(array_dict)
    df_summarized_dist.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>z_grid_values</th>
          <th>lsst_error_model</th>
          <th>inv_redshift_inc</th>
          <th>line_confusion</th>
          <th>quantity_cut</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.00</td>
          <td>0.018251</td>
          <td>0.091872</td>
          <td>0.024438</td>
          <td>0.027158</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.01</td>
          <td>0.024925</td>
          <td>0.108324</td>
          <td>0.031755</td>
          <td>0.034996</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.02</td>
          <td>0.033536</td>
          <td>0.126094</td>
          <td>0.040757</td>
          <td>0.044551</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.03</td>
          <td>0.044467</td>
          <td>0.145059</td>
          <td>0.051683</td>
          <td>0.056047</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.04</td>
          <td>0.058111</td>
          <td>0.165115</td>
          <td>0.064771</td>
          <td>0.069705</td>
        </tr>
      </tbody>
    </table>
    </div>



Now you can save the pandas DataFrame to whatever file type you would
like, or pass it on to another algorithm.

4. Seeing how the algorithm calibration affected the output redshift distributions
----------------------------------------------------------------------------------

You can compare your estimated redshift distributions however you want,
but RAIL has a bunch of built in metrics that compare the distributions,
so we’ll use that here. They are a part of the `evaluation
stage <https://rail-hub.readthedocs.io/en/latest/source/rail_stages/creation.html>`__.
For a more detailed look at all of the available metrics and how to use
them, take a look at the ``01_Evaluation_by_Type.ipynb`` notebook.

| Here we’re just going to use two of the available metrics: 1. The
  `Brier score <https://en.wikipedia.org/wiki/Brier_score>`__, which
  assesses the accuracy of probabilistic predictions. The lower the
  score, the better the predictions.
| 2. The `Conditional Density Estimation
  loss <https://vitaliset.github.io/conditional-density-estimation/>`__,
  which is the averaged squared loss between the true and predicted
  conditional probability density functions. The lower the score, the
  better the predicted probability density, in this case, the
  photometric redshift distributions.

For the evaluation metrics, in general we need the estimated redshift
distributions, and the actual redshifts – these are the pre-degradation
redshifts from our initially sampled distribution. This is why we did
all of that data wrangling earlier to get our estimated redshifts to
line up with our pre-degradation photometry data.

.. code:: ipython3

    # set up dictionaries for output
    eval_dict = {}
    
    for key, item in estimated_photoz.items():
        # evaluate the results
        evaluator_stage_dict = dict(
            metrics=["cdeloss", "brier"],
            _random_state=None,
            metric_config={
                "brier": {"limits": (0, 3.1)},
            },
        )
    
        the_eval = ri.evaluation.dist_to_point_evaluator.dist_to_point_evaluator(
            data=item["output"],
            truth=reindexed_targ_data_orig,
            **evaluator_stage_dict,
            hdf5_groupname="",
        )
    
        # put the evaluation results in a dictionary so we have them
        eval_dict[key] = the_eval


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.684234   23.222248   22.496042   21.628773   20.817329   20.564213   
    1    1.751949   28.061226   26.201729   25.492706   24.880798   24.292238   
    2    0.334738   22.901604   22.585258   22.078140   22.091541   21.725985   
    3    0.508261   27.219675   26.349281   25.363613   24.975111   24.789568   
    4    0.202286   25.150200   24.267548   23.537415   23.175728   22.982109   
    ..        ...         ...         ...         ...         ...         ...   
    230  0.498327   25.256531   24.734894   23.983690   23.683472   23.646606   
    231  0.351069   25.185009   23.521603   22.113777   21.557377   21.228889   
    232  0.631504   27.546143   25.364649   23.775986   22.639019   22.233021   
    233  1.020108   24.216951   23.881901   23.230240   22.513334   21.873619   
    234  0.410133   25.853966   24.891123   23.988443   23.726196   23.565538   
    
         mag_y_lsst  
    0     20.347912  
    1     23.972845  
    2     21.967663  
    3     24.562302  
    4     22.828064  
    ..          ...  
    230   23.460381  
    231   21.010700  
    232   21.900209  
    233   21.645157  
    234   23.432323  
    
    [235 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.684234   23.222248   22.496042   21.628773   20.817329   20.564213   
    1    1.751949   28.061226   26.201729   25.492706   24.880798   24.292238   
    2    0.334738   22.901604   22.585258   22.078140   22.091541   21.725985   
    3    0.508261   27.219675   26.349281   25.363613   24.975111   24.789568   
    4    0.202286   25.150200   24.267548   23.537415   23.175728   22.982109   
    ..        ...         ...         ...         ...         ...         ...   
    230  0.498327   25.256531   24.734894   23.983690   23.683472   23.646606   
    231  0.351069   25.185009   23.521603   22.113777   21.557377   21.228889   
    232  0.631504   27.546143   25.364649   23.775986   22.639019   22.233021   
    233  1.020108   24.216951   23.881901   23.230240   22.513334   21.873619   
    234  0.410133   25.853966   24.891123   23.988443   23.726196   23.565538   
    
         mag_y_lsst  
    0     20.347912  
    1     23.972845  
    2     21.967663  
    3     24.562302  
    4     22.828064  
    ..          ...  
    230   23.460381  
    231   21.010700  
    232   21.900209  
    233   21.645157  
    234   23.432323  
    
    [235 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.684234   23.222248   22.496042   21.628773   20.817329   20.564213   
    1    1.751949   28.061226   26.201729   25.492706   24.880798   24.292238   
    2    0.334738   22.901604   22.585258   22.078140   22.091541   21.725985   
    3    0.508261   27.219675   26.349281   25.363613   24.975111   24.789568   
    4    0.202286   25.150200   24.267548   23.537415   23.175728   22.982109   
    ..        ...         ...         ...         ...         ...         ...   
    230  0.498327   25.256531   24.734894   23.983690   23.683472   23.646606   
    231  0.351069   25.185009   23.521603   22.113777   21.557377   21.228889   
    232  0.631504   27.546143   25.364649   23.775986   22.639019   22.233021   
    233  1.020108   24.216951   23.881901   23.230240   22.513334   21.873619   
    234  0.410133   25.853966   24.891123   23.988443   23.726196   23.565538   
    
         mag_y_lsst  
    0     20.347912  
    1     23.972845  
    2     21.967663  
    3     24.562302  
    4     22.828064  
    ..          ...  
    230   23.460381  
    231   21.010700  
    232   21.900209  
    233   21.645157  
    234   23.432323  
    
    [235 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.684234   23.222248   22.496042   21.628773   20.817329   20.564213   
    1    1.751949   28.061226   26.201729   25.492706   24.880798   24.292238   
    2    0.334738   22.901604   22.585258   22.078140   22.091541   21.725985   
    3    0.508261   27.219675   26.349281   25.363613   24.975111   24.789568   
    4    0.202286   25.150200   24.267548   23.537415   23.175728   22.982109   
    ..        ...         ...         ...         ...         ...         ...   
    230  0.498327   25.256531   24.734894   23.983690   23.683472   23.646606   
    231  0.351069   25.185009   23.521603   22.113777   21.557377   21.228889   
    232  0.631504   27.546143   25.364649   23.775986   22.639019   22.233021   
    233  1.020108   24.216951   23.881901   23.230240   22.513334   21.873619   
    234  0.410133   25.853966   24.891123   23.988443   23.726196   23.565538   
    
         mag_y_lsst  
    0     20.347912  
    1     23.972845  
    2     21.967663  
    3     24.562302  
    4     22.828064  
    ..          ...  
    230   23.460381  
    231   21.010700  
    232   21.900209  
    233   21.645157  
    234   23.432323  
    
    [235 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator


Now let’s take a look at the metrics we calculated, and compare them.
The code below just selects the one dictionary output per run that we
want to look at, to make the dictionary a little easier to read.

.. code:: ipython3

    # pull data out of the sub-directory to make the dictionaries easier to read
    results_dict = {key: val["summary"] for key, val in eval_dict.items()}
    
    results_dict




.. parsed-literal::

    {'lsst_error_model': {'cdeloss': array([-3.06683668]),
      'brier': array([228.01555861])},
     'inv_redshift_inc': {'cdeloss': array([-7.61215538]),
      'brier': array([403.38103869])},
     'line_confusion': {'cdeloss': array([-2.89249239]),
      'brier': array([233.53128453])},
     'quantity_cut': {'cdeloss': array([-2.80675096]),
      'brier': array([253.3444831])}}



We can also plot these metrics to better visualize which trianing data
sets gave better scores, ‘better’ here meaning lower for both of the
metrics:

.. code:: ipython3

    for key, value in eval_dict.items():
        plt.scatter(value["summary"]["brier"], value["summary"]["cdeloss"], label=key)
    
    plt.legend(loc="best")
    plt.xlabel("Brier score")
    plt.ylabel("CDE loss")




.. parsed-literal::

    Text(0, 0.5, 'CDE loss')




.. image:: Exploring_the_Effects_of_Degraders_files/Exploring_the_Effects_of_Degraders_85_1.png


This gives us a bit of a clearer picture of which calibration
distributions did better than others. It’s also clear from this why
multiple metrics can be useful, since some of these distributions do
better in one metric than the other.

Next Steps
----------

If you’d like to parallelize your iteration in order to speed things up,
take a look at the `introduction to RAIL
interactive <https://rail-hub.readthedocs.io/projects/rail-notebooks/en/latest/interactive_examples/rendered/estimation_examples/Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters.html>`__
notebook.

To learn more about the creation stage of RAIL, and the available
degraders, take a look at the `RAIL Creation
docs <https://rail-hub.readthedocs.io/en/latest/source/rail_stages/creation.html>`__.

Similarly, if you’d like to learn more about the Evaluation stage, you
can take a look at the `RAIL Evaluation
docs <https://rail-hub.readthedocs.io/en/latest/source/rail_stages/evaluation.html>`__,
or try out the `Evaluation by
type <https://rail-hub.readthedocs.io/projects/rail-notebooks/en/latest/interactive_examples/rendered/evaluation_examples/01_Evaluation_by_Type.html>`__
notebook.
