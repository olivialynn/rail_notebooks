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
    /home/runner/.cache/lephare/runs/20260511T124646


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
      File "/tmp/ipykernel_4529/1847479680.py", line 1, in <module>
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

    (1) 1.4431


.. parsed-literal::

    (2) -0.0247


.. parsed-literal::

    (3) -1.2539


.. parsed-literal::

    (4) -1.4070


.. parsed-literal::

    (5) -2.0570


.. parsed-literal::

    (6) -0.5525


.. parsed-literal::

    (7) -1.6905


.. parsed-literal::

    (8) -1.7447


.. parsed-literal::

    (9) -2.6114


.. parsed-literal::

    (10) -3.0154


.. parsed-literal::

    (11) -2.6144


.. parsed-literal::

    (12) -2.3561


.. parsed-literal::

    (13) -3.0956


.. parsed-literal::

    (14) -3.3651


.. parsed-literal::

    (15) -3.3584


.. parsed-literal::

    (16) -3.6009


.. parsed-literal::

    (17) -3.2523


.. parsed-literal::

    (18) -3.4311


.. parsed-literal::

    (19) -3.7140


.. parsed-literal::

    (20) -2.5754


.. parsed-literal::

    (21) -3.7610


.. parsed-literal::

    (22) -4.3439


.. parsed-literal::

    (23) -4.2062


.. parsed-literal::

    (24) -4.0795


.. parsed-literal::

    (25) -3.8448


.. parsed-literal::

    (26) -4.5009


.. parsed-literal::

    (27) -4.4987


.. parsed-literal::

    (28) -3.8959


.. parsed-literal::

    (29) -4.4735


.. parsed-literal::

    (30) -4.3248
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

    Inserting handle into data store.  model: <pzflow.flow.Flow object at 0x7f5a14ae2290>, FlowCreator


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.pq, FlowCreator
    Inserting handle into data store.  model: <pzflow.flow.Flow object at 0x7f5a14ae2290>, FlowCreator
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
          <td>0.338784</td>
          <td>25.911770</td>
          <td>0.235140</td>
          <td>26.162720</td>
          <td>0.103334</td>
          <td>25.738584</td>
          <td>0.062561</td>
          <td>25.870381</td>
          <td>0.114382</td>
          <td>25.688351</td>
          <td>0.183659</td>
          <td>26.886563</td>
          <td>0.901562</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.248485</td>
          <td>25.048114</td>
          <td>0.112895</td>
          <td>24.944219</td>
          <td>0.035251</td>
          <td>24.921420</td>
          <td>0.030338</td>
          <td>24.687588</td>
          <td>0.040231</td>
          <td>24.138941</td>
          <td>0.047386</td>
          <td>23.883616</td>
          <td>0.085318</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2.153771</td>
          <td>26.781072</td>
          <td>0.467142</td>
          <td>27.689342</td>
          <td>0.369984</td>
          <td>26.979072</td>
          <td>0.184317</td>
          <td>27.176490</td>
          <td>0.341198</td>
          <td>26.895242</td>
          <td>0.482519</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.058963</td>
          <td>26.089085</td>
          <td>0.271905</td>
          <td>25.738733</td>
          <td>0.071178</td>
          <td>25.606092</td>
          <td>0.055622</td>
          <td>25.143436</td>
          <td>0.060294</td>
          <td>24.506536</td>
          <td>0.065661</td>
          <td>24.197618</td>
          <td>0.112352</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.260936</td>
          <td>26.923663</td>
          <td>0.519092</td>
          <td>27.599141</td>
          <td>0.344719</td>
          <td>26.861468</td>
          <td>0.166803</td>
          <td>26.362330</td>
          <td>0.174710</td>
          <td>25.705027</td>
          <td>0.186267</td>
          <td>26.096104</td>
          <td>0.527140</td>
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

    <matplotlib.colorbar.Colorbar at 0x7f59927aed40>




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
          <td>0.558813</td>
          <td>24.666773</td>
          <td>0.080926</td>
          <td>23.577384</td>
          <td>0.011402</td>
          <td>22.230680</td>
          <td>0.005690</td>
          <td>21.364320</td>
          <td>0.005415</td>
          <td>21.004385</td>
          <td>0.005764</td>
          <td>20.714733</td>
          <td>0.007046</td>
        </tr>
        <tr>
          <th>1</th>
          <td>True</td>
          <td>0.750511</td>
          <td>26.151982</td>
          <td>0.286121</td>
          <td>26.005485</td>
          <td>0.090035</td>
          <td>25.617993</td>
          <td>0.056213</td>
          <td>24.982373</td>
          <td>0.052262</td>
          <td>24.739239</td>
          <td>0.080663</td>
          <td>24.672967</td>
          <td>0.169272</td>
        </tr>
        <tr>
          <th>2</th>
          <td>True</td>
          <td>0.902206</td>
          <td>25.814563</td>
          <td>0.216938</td>
          <td>25.839181</td>
          <td>0.077777</td>
          <td>25.248637</td>
          <td>0.040501</td>
          <td>24.605841</td>
          <td>0.037421</td>
          <td>24.191651</td>
          <td>0.049656</td>
          <td>24.151599</td>
          <td>0.107930</td>
        </tr>
        <tr>
          <th>3</th>
          <td>False</td>
          <td>1.533603</td>
          <td>29.117307</td>
          <td>1.874080</td>
          <td>27.221108</td>
          <td>0.254301</td>
          <td>27.795433</td>
          <td>0.359200</td>
          <td>26.846600</td>
          <td>0.261693</td>
          <td>25.994994</td>
          <td>0.237361</td>
          <td>25.158003</td>
          <td>0.253995</td>
        </tr>
        <tr>
          <th>4</th>
          <td>True</td>
          <td>2.207015</td>
          <td>27.231466</td>
          <td>0.646419</td>
          <td>28.532864</td>
          <td>0.686894</td>
          <td>27.685246</td>
          <td>0.329296</td>
          <td>27.340211</td>
          <td>0.387806</td>
          <td>26.513153</td>
          <td>0.360381</td>
          <td>NaN</td>
          <td>NaN</td>
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
          <td>0.755889</td>
          <td>25.461619</td>
          <td>0.161143</td>
          <td>24.457381</td>
          <td>0.023075</td>
          <td>23.625050</td>
          <td>0.010547</td>
          <td>22.770888</td>
          <td>0.008676</td>
          <td>22.456852</td>
          <td>0.011482</td>
          <td>22.353575</td>
          <td>0.022155</td>
        </tr>
        <tr>
          <th>596</th>
          <td>True</td>
          <td>1.236990</td>
          <td>24.034745</td>
          <td>0.046423</td>
          <td>23.843580</td>
          <td>0.013920</td>
          <td>23.583893</td>
          <td>0.010251</td>
          <td>23.060909</td>
          <td>0.010466</td>
          <td>21.651609</td>
          <td>0.007099</td>
          <td>21.170114</td>
          <td>0.008962</td>
        </tr>
        <tr>
          <th>597</th>
          <td>True</td>
          <td>0.482431</td>
          <td>26.406270</td>
          <td>0.350404</td>
          <td>25.724428</td>
          <td>0.070284</td>
          <td>24.610976</td>
          <td>0.023151</td>
          <td>24.292106</td>
          <td>0.028386</td>
          <td>24.050732</td>
          <td>0.043818</td>
          <td>23.922825</td>
          <td>0.088314</td>
        </tr>
        <tr>
          <th>598</th>
          <td>True</td>
          <td>0.316736</td>
          <td>27.535139</td>
          <td>0.793120</td>
          <td>26.206112</td>
          <td>0.107325</td>
          <td>25.270233</td>
          <td>0.041284</td>
          <td>25.241629</td>
          <td>0.065780</td>
          <td>24.928589</td>
          <td>0.095289</td>
          <td>24.823965</td>
          <td>0.192367</td>
        </tr>
        <tr>
          <th>599</th>
          <td>True</td>
          <td>1.381295</td>
          <td>27.400163</td>
          <td>0.725298</td>
          <td>25.967997</td>
          <td>0.087118</td>
          <td>25.600417</td>
          <td>0.055343</td>
          <td>24.922868</td>
          <td>0.049572</td>
          <td>24.723836</td>
          <td>0.079574</td>
          <td>24.109730</td>
          <td>0.104052</td>
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
          <td>0.558813</td>
          <td>24.666773</td>
          <td>0.080926</td>
          <td>23.577384</td>
          <td>0.011402</td>
          <td>22.230680</td>
          <td>0.005690</td>
          <td>21.364320</td>
          <td>0.005415</td>
          <td>21.004385</td>
          <td>0.005764</td>
          <td>20.714733</td>
          <td>0.007046</td>
        </tr>
        <tr>
          <th>1</th>
          <td>True</td>
          <td>0.750511</td>
          <td>26.151982</td>
          <td>0.286121</td>
          <td>26.005485</td>
          <td>0.090035</td>
          <td>25.617993</td>
          <td>0.056213</td>
          <td>24.982373</td>
          <td>0.052262</td>
          <td>24.739239</td>
          <td>0.080663</td>
          <td>24.672967</td>
          <td>0.169272</td>
        </tr>
        <tr>
          <th>2</th>
          <td>True</td>
          <td>0.902206</td>
          <td>25.814563</td>
          <td>0.216938</td>
          <td>25.839181</td>
          <td>0.077777</td>
          <td>25.248637</td>
          <td>0.040501</td>
          <td>24.605841</td>
          <td>0.037421</td>
          <td>24.191651</td>
          <td>0.049656</td>
          <td>24.151599</td>
          <td>0.107930</td>
        </tr>
        <tr>
          <th>3</th>
          <td>False</td>
          <td>1.533603</td>
          <td>29.117307</td>
          <td>1.874080</td>
          <td>27.221108</td>
          <td>0.254301</td>
          <td>27.795433</td>
          <td>0.359200</td>
          <td>26.846600</td>
          <td>0.261693</td>
          <td>25.994994</td>
          <td>0.237361</td>
          <td>25.158003</td>
          <td>0.253995</td>
        </tr>
        <tr>
          <th>4</th>
          <td>True</td>
          <td>2.207015</td>
          <td>27.231466</td>
          <td>0.646419</td>
          <td>28.532864</td>
          <td>0.686894</td>
          <td>27.685246</td>
          <td>0.329296</td>
          <td>27.340211</td>
          <td>0.387806</td>
          <td>26.513153</td>
          <td>0.360381</td>
          <td>NaN</td>
          <td>NaN</td>
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
          <td>0.755889</td>
          <td>25.461619</td>
          <td>0.161143</td>
          <td>24.457381</td>
          <td>0.023075</td>
          <td>23.625050</td>
          <td>0.010547</td>
          <td>22.770888</td>
          <td>0.008676</td>
          <td>22.456852</td>
          <td>0.011482</td>
          <td>22.353575</td>
          <td>0.022155</td>
        </tr>
        <tr>
          <th>596</th>
          <td>True</td>
          <td>1.236990</td>
          <td>24.034745</td>
          <td>0.046423</td>
          <td>23.843580</td>
          <td>0.013920</td>
          <td>23.583893</td>
          <td>0.010251</td>
          <td>23.060909</td>
          <td>0.010466</td>
          <td>21.651609</td>
          <td>0.007099</td>
          <td>21.170114</td>
          <td>0.008962</td>
        </tr>
        <tr>
          <th>597</th>
          <td>True</td>
          <td>0.482431</td>
          <td>26.406270</td>
          <td>0.350404</td>
          <td>25.724428</td>
          <td>0.070284</td>
          <td>24.610976</td>
          <td>0.023151</td>
          <td>24.292106</td>
          <td>0.028386</td>
          <td>24.050732</td>
          <td>0.043818</td>
          <td>23.922825</td>
          <td>0.088314</td>
        </tr>
        <tr>
          <th>598</th>
          <td>True</td>
          <td>0.316736</td>
          <td>27.535139</td>
          <td>0.793120</td>
          <td>26.206112</td>
          <td>0.107325</td>
          <td>25.270233</td>
          <td>0.041284</td>
          <td>25.241629</td>
          <td>0.065780</td>
          <td>24.928589</td>
          <td>0.095289</td>
          <td>24.823965</td>
          <td>0.192367</td>
        </tr>
        <tr>
          <th>599</th>
          <td>True</td>
          <td>1.381295</td>
          <td>27.400163</td>
          <td>0.725298</td>
          <td>25.967997</td>
          <td>0.087118</td>
          <td>25.600417</td>
          <td>0.055343</td>
          <td>24.922868</td>
          <td>0.049572</td>
          <td>24.723836</td>
          <td>0.079574</td>
          <td>24.109730</td>
          <td>0.104052</td>
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
          <td>0.558813</td>
          <td>24.666773</td>
          <td>0.080926</td>
          <td>23.577384</td>
          <td>0.011402</td>
          <td>22.230680</td>
          <td>0.005690</td>
          <td>21.364320</td>
          <td>0.005415</td>
          <td>21.004385</td>
          <td>0.005764</td>
          <td>20.714733</td>
          <td>0.007046</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.750511</td>
          <td>26.151982</td>
          <td>0.286121</td>
          <td>26.005485</td>
          <td>0.090035</td>
          <td>25.617993</td>
          <td>0.056213</td>
          <td>24.982373</td>
          <td>0.052262</td>
          <td>24.739239</td>
          <td>0.080663</td>
          <td>24.672967</td>
          <td>0.169272</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.902206</td>
          <td>25.814563</td>
          <td>0.216938</td>
          <td>25.839181</td>
          <td>0.077777</td>
          <td>25.248637</td>
          <td>0.040501</td>
          <td>24.605841</td>
          <td>0.037421</td>
          <td>24.191651</td>
          <td>0.049656</td>
          <td>24.151599</td>
          <td>0.107930</td>
        </tr>
        <tr>
          <th>3</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4</th>
          <td>2.207015</td>
          <td>27.231466</td>
          <td>0.646419</td>
          <td>28.532864</td>
          <td>0.686894</td>
          <td>27.685246</td>
          <td>0.329296</td>
          <td>27.340211</td>
          <td>0.387806</td>
          <td>26.513153</td>
          <td>0.360381</td>
          <td>NaN</td>
          <td>NaN</td>
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
          <td>0.755889</td>
          <td>25.461619</td>
          <td>0.161143</td>
          <td>24.457381</td>
          <td>0.023075</td>
          <td>23.625050</td>
          <td>0.010547</td>
          <td>22.770888</td>
          <td>0.008676</td>
          <td>22.456852</td>
          <td>0.011482</td>
          <td>22.353575</td>
          <td>0.022155</td>
        </tr>
        <tr>
          <th>596</th>
          <td>1.236990</td>
          <td>24.034745</td>
          <td>0.046423</td>
          <td>23.843580</td>
          <td>0.013920</td>
          <td>23.583893</td>
          <td>0.010251</td>
          <td>23.060909</td>
          <td>0.010466</td>
          <td>21.651609</td>
          <td>0.007099</td>
          <td>21.170114</td>
          <td>0.008962</td>
        </tr>
        <tr>
          <th>597</th>
          <td>0.482431</td>
          <td>26.406270</td>
          <td>0.350404</td>
          <td>25.724428</td>
          <td>0.070284</td>
          <td>24.610976</td>
          <td>0.023151</td>
          <td>24.292106</td>
          <td>0.028386</td>
          <td>24.050732</td>
          <td>0.043818</td>
          <td>23.922825</td>
          <td>0.088314</td>
        </tr>
        <tr>
          <th>598</th>
          <td>0.316736</td>
          <td>27.535139</td>
          <td>0.793120</td>
          <td>26.206112</td>
          <td>0.107325</td>
          <td>25.270233</td>
          <td>0.041284</td>
          <td>25.241629</td>
          <td>0.065780</td>
          <td>24.928589</td>
          <td>0.095289</td>
          <td>24.823965</td>
          <td>0.192367</td>
        </tr>
        <tr>
          <th>599</th>
          <td>1.381295</td>
          <td>27.400163</td>
          <td>0.725298</td>
          <td>25.967997</td>
          <td>0.087118</td>
          <td>25.600417</td>
          <td>0.055343</td>
          <td>24.922868</td>
          <td>0.049572</td>
          <td>24.723836</td>
          <td>0.079574</td>
          <td>24.109730</td>
          <td>0.104052</td>
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
          <td>1</td>
          <td>0.558813</td>
          <td>24.666773</td>
          <td>0.080926</td>
          <td>23.577384</td>
          <td>0.011402</td>
          <td>22.230680</td>
          <td>0.005690</td>
          <td>21.364320</td>
          <td>0.005415</td>
          <td>21.004385</td>
          <td>0.005764</td>
          <td>20.714733</td>
          <td>0.007046</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>0.750511</td>
          <td>26.151982</td>
          <td>0.286121</td>
          <td>26.005485</td>
          <td>0.090035</td>
          <td>25.617993</td>
          <td>0.056213</td>
          <td>24.982373</td>
          <td>0.052262</td>
          <td>24.739239</td>
          <td>0.080663</td>
          <td>24.672967</td>
          <td>0.169272</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1</td>
          <td>0.902206</td>
          <td>25.814563</td>
          <td>0.216938</td>
          <td>25.839181</td>
          <td>0.077777</td>
          <td>25.248637</td>
          <td>0.040501</td>
          <td>24.605841</td>
          <td>0.037421</td>
          <td>24.191651</td>
          <td>0.049656</td>
          <td>24.151599</td>
          <td>0.107930</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0</td>
          <td>2.207015</td>
          <td>27.231466</td>
          <td>0.646419</td>
          <td>28.532864</td>
          <td>0.686894</td>
          <td>27.685246</td>
          <td>0.329296</td>
          <td>27.340211</td>
          <td>0.387806</td>
          <td>26.513153</td>
          <td>0.360381</td>
          <td>NaN</td>
          <td>NaN</td>
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
          <td>1</td>
          <td>0.755889</td>
          <td>25.461619</td>
          <td>0.161143</td>
          <td>24.457381</td>
          <td>0.023075</td>
          <td>23.625050</td>
          <td>0.010547</td>
          <td>22.770888</td>
          <td>0.008676</td>
          <td>22.456852</td>
          <td>0.011482</td>
          <td>22.353575</td>
          <td>0.022155</td>
        </tr>
        <tr>
          <th>596</th>
          <td>1</td>
          <td>2.005262</td>
          <td>24.034745</td>
          <td>0.046423</td>
          <td>23.843580</td>
          <td>0.013920</td>
          <td>23.583893</td>
          <td>0.010251</td>
          <td>23.060909</td>
          <td>0.010466</td>
          <td>21.651609</td>
          <td>0.007099</td>
          <td>21.170114</td>
          <td>0.008962</td>
        </tr>
        <tr>
          <th>597</th>
          <td>1</td>
          <td>0.482431</td>
          <td>26.406270</td>
          <td>0.350404</td>
          <td>25.724428</td>
          <td>0.070284</td>
          <td>24.610976</td>
          <td>0.023151</td>
          <td>24.292106</td>
          <td>0.028386</td>
          <td>24.050732</td>
          <td>0.043818</td>
          <td>23.922825</td>
          <td>0.088314</td>
        </tr>
        <tr>
          <th>598</th>
          <td>0</td>
          <td>0.316736</td>
          <td>27.535139</td>
          <td>0.793120</td>
          <td>26.206112</td>
          <td>0.107325</td>
          <td>25.270233</td>
          <td>0.041284</td>
          <td>25.241629</td>
          <td>0.065780</td>
          <td>24.928589</td>
          <td>0.095289</td>
          <td>24.823965</td>
          <td>0.192367</td>
        </tr>
        <tr>
          <th>599</th>
          <td>1</td>
          <td>2.199126</td>
          <td>27.400163</td>
          <td>0.725298</td>
          <td>25.967997</td>
          <td>0.087118</td>
          <td>25.600417</td>
          <td>0.055343</td>
          <td>24.922868</td>
          <td>0.049572</td>
          <td>24.723836</td>
          <td>0.079574</td>
          <td>24.109730</td>
          <td>0.104052</td>
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
          <td>0.558813</td>
          <td>24.666773</td>
          <td>0.080926</td>
          <td>23.577384</td>
          <td>0.011402</td>
          <td>22.230680</td>
          <td>0.005690</td>
          <td>21.364320</td>
          <td>0.005415</td>
          <td>21.004385</td>
          <td>0.005764</td>
          <td>20.714733</td>
          <td>0.007046</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.750511</td>
          <td>26.151982</td>
          <td>0.286121</td>
          <td>26.005485</td>
          <td>0.090035</td>
          <td>25.617993</td>
          <td>0.056213</td>
          <td>24.982373</td>
          <td>0.052262</td>
          <td>24.739239</td>
          <td>0.080663</td>
          <td>24.672967</td>
          <td>0.169272</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.902206</td>
          <td>25.814563</td>
          <td>0.216938</td>
          <td>25.839181</td>
          <td>0.077777</td>
          <td>25.248637</td>
          <td>0.040501</td>
          <td>24.605841</td>
          <td>0.037421</td>
          <td>24.191651</td>
          <td>0.049656</td>
          <td>24.151599</td>
          <td>0.107930</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.895582</td>
          <td>24.393195</td>
          <td>0.063633</td>
          <td>24.188828</td>
          <td>0.018391</td>
          <td>23.568735</td>
          <td>0.010146</td>
          <td>22.809988</td>
          <td>0.008883</td>
          <td>22.380937</td>
          <td>0.010873</td>
          <td>22.198871</td>
          <td>0.019422</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.646287</td>
          <td>25.933927</td>
          <td>0.239477</td>
          <td>25.728043</td>
          <td>0.070509</td>
          <td>25.087468</td>
          <td>0.035117</td>
          <td>24.593527</td>
          <td>0.037016</td>
          <td>24.418115</td>
          <td>0.060710</td>
          <td>24.321316</td>
          <td>0.125112</td>
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
          <th>226</th>
          <td>0.813069</td>
          <td>26.115540</td>
          <td>0.277807</td>
          <td>26.128128</td>
          <td>0.100255</td>
          <td>25.602908</td>
          <td>0.055465</td>
          <td>24.974285</td>
          <td>0.051888</td>
          <td>24.607750</td>
          <td>0.071816</td>
          <td>24.859044</td>
          <td>0.198132</td>
        </tr>
        <tr>
          <th>227</th>
          <td>0.755889</td>
          <td>25.461619</td>
          <td>0.161143</td>
          <td>24.457381</td>
          <td>0.023075</td>
          <td>23.625050</td>
          <td>0.010547</td>
          <td>22.770888</td>
          <td>0.008676</td>
          <td>22.456852</td>
          <td>0.011482</td>
          <td>22.353575</td>
          <td>0.022155</td>
        </tr>
        <tr>
          <th>228</th>
          <td>2.005262</td>
          <td>24.034745</td>
          <td>0.046423</td>
          <td>23.843580</td>
          <td>0.013920</td>
          <td>23.583893</td>
          <td>0.010251</td>
          <td>23.060909</td>
          <td>0.010466</td>
          <td>21.651609</td>
          <td>0.007099</td>
          <td>21.170114</td>
          <td>0.008962</td>
        </tr>
        <tr>
          <th>229</th>
          <td>0.482431</td>
          <td>26.406270</td>
          <td>0.350404</td>
          <td>25.724428</td>
          <td>0.070284</td>
          <td>24.610976</td>
          <td>0.023151</td>
          <td>24.292106</td>
          <td>0.028386</td>
          <td>24.050732</td>
          <td>0.043818</td>
          <td>23.922825</td>
          <td>0.088314</td>
        </tr>
        <tr>
          <th>230</th>
          <td>2.199126</td>
          <td>27.400163</td>
          <td>0.725298</td>
          <td>25.967997</td>
          <td>0.087118</td>
          <td>25.600417</td>
          <td>0.055343</td>
          <td>24.922868</td>
          <td>0.049572</td>
          <td>24.723836</td>
          <td>0.079574</td>
          <td>24.109730</td>
          <td>0.104052</td>
        </tr>
      </tbody>
    </table>
    <p>231 rows × 13 columns</p>
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
          <td>0.558813</td>
          <td>24.713184</td>
          <td>23.571205</td>
          <td>22.227867</td>
          <td>21.358557</td>
          <td>20.996706</td>
          <td>20.703762</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.750511</td>
          <td>26.330038</td>
          <td>26.115959</td>
          <td>25.643736</td>
          <td>24.997763</td>
          <td>24.764048</td>
          <td>24.650198</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.902206</td>
          <td>26.077417</td>
          <td>25.778303</td>
          <td>25.293756</td>
          <td>24.599886</td>
          <td>24.188847</td>
          <td>24.020731</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.895582</td>
          <td>24.499634</td>
          <td>24.176474</td>
          <td>23.566423</td>
          <td>22.824486</td>
          <td>22.369812</td>
          <td>22.198908</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.646287</td>
          <td>26.026688</td>
          <td>25.736584</td>
          <td>25.132135</td>
          <td>24.595406</td>
          <td>24.405972</td>
          <td>24.248049</td>
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
          <th>226</th>
          <td>0.813069</td>
          <td>26.210548</td>
          <td>26.045996</td>
          <td>25.626324</td>
          <td>24.914921</td>
          <td>24.677912</td>
          <td>24.663494</td>
        </tr>
        <tr>
          <th>227</th>
          <td>0.755889</td>
          <td>25.345015</td>
          <td>24.436749</td>
          <td>23.613729</td>
          <td>22.758505</td>
          <td>22.476389</td>
          <td>22.314535</td>
        </tr>
        <tr>
          <th>228</th>
          <td>1.236990</td>
          <td>24.090199</td>
          <td>23.853308</td>
          <td>23.583414</td>
          <td>23.060661</td>
          <td>21.655033</td>
          <td>21.169876</td>
        </tr>
        <tr>
          <th>229</th>
          <td>0.482431</td>
          <td>26.793308</td>
          <td>25.781473</td>
          <td>24.640827</td>
          <td>24.271374</td>
          <td>24.107733</td>
          <td>23.881256</td>
        </tr>
        <tr>
          <th>230</th>
          <td>1.381295</td>
          <td>26.768213</td>
          <td>26.117926</td>
          <td>25.567503</td>
          <td>24.973419</td>
          <td>24.633711</td>
          <td>24.049259</td>
        </tr>
      </tbody>
    </table>
    <p>231 rows × 7 columns</p>
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

    
    
    
    best fit values are sigma=0.075 and numneigh=7
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer
    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 450 training and 150 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.06777777777777778 and numneigh=6
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer
    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 386 training and 128 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.075 and numneigh=6
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer
    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 179 training and 60 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.060555555555555564 and numneigh=6
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer


.. code:: ipython3

    # let's see what the output looks like 
    knn_models["lsst_error_model"]




.. parsed-literal::

    {'model': {'kdtree': <sklearn.neighbors._kd_tree.KDTree at 0x555e2f09c0d0>,
      'bestsig': np.float64(0.075),
      'nneigh': 7,
      'truezs': array([0.3387837 , 1.2484847 , 2.153771  , 1.0589626 , 1.2609361 ,
             1.9168605 , 1.4007992 , 1.9427918 , 2.0654135 , 0.17257373,
             0.8977886 , 0.26945314, 0.26148003, 1.0591877 , 1.1169813 ,
             0.27561224, 1.6678991 , 1.8990567 , 1.6929686 , 1.1173662 ,
             1.7225233 , 1.3819587 , 1.2092649 , 1.0769911 , 0.7295847 ,
             0.7972144 , 1.9422759 , 1.5583638 , 0.37218407, 1.1825606 ,
             1.2922283 , 2.185101  , 1.660346  , 0.22092588, 0.68016213,
             1.8995954 , 1.8868735 , 0.48495978, 1.9762305 , 0.29599017,
             0.92012775, 1.5730174 , 0.7104293 , 0.29740044, 1.7267003 ,
             0.45903644, 0.21176746, 0.49612883, 2.1182091 , 0.4143995 ,
             1.2394212 , 1.577477  , 2.1925735 , 1.5085403 , 2.2599053 ,
             1.4495722 , 0.33966503, 1.2594613 , 0.46935967, 1.5197884 ,
             1.408432  , 0.33716878, 0.33912238, 1.9967526 , 1.1096987 ,
             0.69080484, 1.2366139 , 1.5823141 , 1.5005487 , 0.41091365,
             1.5099218 , 2.1247592 , 0.8427471 , 1.9725356 , 1.6986089 ,
             0.81424224, 2.0235112 , 0.43857667, 1.5166559 , 1.6643689 ,
             1.2386088 , 1.385784  , 1.0357916 , 1.586993  , 1.4581815 ,
             0.8329263 , 0.6347715 , 0.4416393 , 1.6619018 , 1.1954021 ,
             1.710019  , 0.28416213, 1.15485   , 1.0471253 , 0.36942825,
             0.5142037 , 0.81722975, 1.1447295 , 0.7169016 , 0.15013102,
             0.7603108 , 0.3026472 , 0.52148837, 0.37467706, 2.046937  ,
             0.17641734, 0.47197208, 1.3768597 , 1.5150669 , 0.772303  ,
             0.77221745, 1.3901622 , 1.1456535 , 1.3942121 , 0.73172015,
             1.796524  , 0.5268111 , 1.2542303 , 1.2184184 , 1.8905294 ,
             1.3014894 , 0.74375266, 0.6096792 , 0.8230558 , 0.5392863 ,
             0.12460821, 1.5852716 , 1.4283535 , 1.0215803 , 1.7241379 ,
             1.9044843 , 0.7803245 , 1.9492304 , 2.0915785 , 1.570083  ,
             1.5308751 , 0.11890943, 0.4212692 , 1.7925823 , 2.2249663 ,
             0.1842451 , 0.7113709 , 0.23686388, 0.80016905, 0.73088247,
             1.0766165 , 0.41537255, 1.6848027 , 0.9766652 , 0.31980702,
             1.3925503 , 0.41526327, 0.8231582 , 0.36595705, 1.0231842 ,
             1.2365012 , 1.6561979 , 0.42219388, 0.65557814, 0.6922684 ,
             1.0122771 , 2.1075711 , 0.19901502, 0.47973642, 0.19558333,
             0.9865092 , 1.1884375 , 1.5470389 , 0.32296866, 0.48241863,
             0.45952928, 2.1969178 , 0.35712716, 1.0781019 , 1.0852157 ,
             1.2209102 , 0.23388898, 0.5836977 , 1.0109735 , 0.6622701 ,
             0.300294  , 0.3356636 , 1.1564184 , 0.22877395, 0.2577274 ,
             1.4687084 , 0.31583545, 0.9865428 , 0.3240504 , 1.9025545 ,
             0.26419064, 1.7401885 , 1.2884661 , 0.13800065, 0.4142787 ,
             0.35887143, 0.70411986, 0.9345084 , 0.19424494, 0.5825124 ,
             1.3206912 , 1.0387331 , 0.98988897, 1.2181985 , 1.8136814 ,
             2.004329  , 0.4122194 , 1.0917231 , 0.33617753, 0.6843047 ,
             0.3690189 , 1.0357112 , 1.0776194 , 0.3369648 , 0.9748512 ,
             1.9367574 , 0.35112265, 0.715802  , 0.29704168, 0.12714212,
             0.47079867, 1.5402744 , 1.2326618 , 0.55935544, 1.0218351 ,
             2.151322  , 0.43930042, 0.30776164, 0.74550116, 1.0954635 ,
             2.210278  , 0.69323355, 0.660379  , 1.0471896 , 0.24121569,
             0.19651906, 1.4974073 , 0.9445937 , 1.5089995 , 0.9598642 ,
             0.9005455 , 1.9031551 , 0.38123366, 0.28945082, 1.0247388 ,
             0.3174387 , 0.38283414, 0.5029381 , 0.84094846, 0.29326314,
             0.46964473, 1.2063687 , 0.817962  , 0.67825496, 1.4622391 ,
             1.9775689 , 1.0692734 , 1.85006   , 0.27684304, 1.4282855 ,
             1.0894117 , 1.2913512 , 1.1350669 , 1.5418271 , 0.6917444 ,
             1.2094436 , 1.0492188 , 1.0524127 , 0.12256391, 1.3322802 ,
             1.7683057 , 1.4954526 , 0.23115957, 0.61865664, 0.96132666,
             0.5199129 , 1.7646323 , 0.05754659, 0.9210199 , 1.0734234 ,
             1.2824069 , 1.1859305 , 0.5288401 , 0.4788778 , 1.7773004 ,
             2.1948323 , 0.25191057, 0.39952275, 0.32592618, 2.2187638 ,
             1.453592  , 0.24295595, 0.14553945, 1.6446583 , 0.95385367,
             1.356936  , 1.3907412 , 2.2001233 , 1.9908433 , 1.7229935 ,
             2.0994332 , 0.44873235, 1.4893026 , 0.3521081 , 0.24306677,
             0.37484702, 0.7519174 , 1.4333534 , 0.9700511 , 0.88261896,
             0.949674  , 0.9186223 , 0.29579   , 1.7426188 , 1.5048388 ,
             1.0374715 , 2.0043445 , 0.6813067 , 1.8925664 , 1.3435515 ,
             0.80627555, 0.82640475, 1.7222879 , 0.51165456, 2.1689281 ,
             0.6841359 , 1.2628349 , 0.42523935, 0.13694493, 0.26267603,
             1.6763788 , 0.9370012 , 1.6882255 , 0.230941  , 1.073324  ,
             0.8316961 , 2.24691   , 1.187081  , 1.0098666 , 0.2512058 ,
             0.24307631, 0.71475106, 0.7024954 , 0.44530925, 0.46709555,
             1.5220389 , 1.0647475 , 1.6761858 , 0.65311486, 1.4649705 ,
             1.5797095 , 1.8619741 , 1.5866327 , 0.2673171 , 0.38398167,
             0.42573032, 0.31027374, 1.8861971 , 1.1731787 , 0.7971668 ,
             0.87635094, 1.9429008 , 0.35762537, 0.9308536 , 0.6092299 ,
             1.5966717 , 0.3755022 , 1.2034929 , 0.17367361, 0.31361863,
             1.4443353 , 0.57207286, 1.1167314 , 1.5383885 , 2.1446722 ,
             2.165035  , 0.63585186, 0.2859163 , 0.71970755, 0.6129777 ,
             1.3133336 , 0.9370098 , 0.5676254 , 0.42082402, 1.4824944 ,
             2.2655153 , 0.9016719 , 0.16962297, 1.484996  , 1.6049935 ,
             2.1962085 , 0.65253437, 0.2792837 , 0.93705606, 0.67907447,
             1.2947209 , 0.8118757 , 1.6488634 , 1.3994403 , 0.7008775 ,
             1.6342213 , 1.2435552 , 0.29685268, 0.82000196, 1.6279699 ,
             2.013835  , 1.0658953 , 1.2250209 , 0.29793736, 0.6884921 ,
             0.88409185, 0.7543896 , 0.5572138 , 0.6110236 , 2.174003  ,
             0.6401175 , 0.07403553, 0.3358718 , 1.6684875 , 2.1915298 ,
             1.4325726 , 0.50202614, 1.0128951 , 0.65083647, 0.99642175,
             0.2585532 , 0.60083276, 0.32165393, 1.1643121 , 0.6478291 ,
             1.0469615 , 0.6963149 , 0.08231371, 0.3075918 , 1.6456468 ,
             0.5748981 , 1.9606504 , 0.84170216, 0.5282795 , 1.2481147 ,
             1.9890461 , 0.21737956, 1.7133806 , 0.998776  , 0.91377544,
             0.28719673, 1.3026854 , 1.4113157 , 1.7375115 , 1.8534294 ,
             0.32929924, 0.18261755, 0.20783126, 0.70749456, 0.51771104,
             1.1084858 , 0.80397975, 0.44082564, 0.46448323, 2.1709962 ,
             0.40037915, 1.2334458 , 0.7789311 , 0.5157913 , 0.9123784 ,
             1.8994789 , 0.4295467 , 1.5511605 , 0.37756938, 0.5866857 ,
             0.7362395 , 0.16729401, 0.32926682, 2.1852376 , 1.2001705 ,
             0.990963  , 1.5574335 , 1.019879  , 0.9907941 , 0.7852963 ,
             0.3522209 , 1.2676425 , 0.5657136 , 1.8390493 , 1.3716091 ,
             1.954355  , 0.40749767, 1.457467  , 0.20244673, 0.3017453 ,
             0.34874973, 0.55647886, 2.0248137 , 0.14586294, 0.8868226 ,
             0.43397102, 1.7854514 , 0.9049927 , 1.6703786 , 1.5607324 ,
             0.21144807, 0.52272874, 0.27734432, 0.77333516, 1.7356288 ,
             0.24240845, 1.5862732 , 1.727943  , 0.81161827, 0.38072985,
             0.7859878 , 0.26325673, 0.3207219 , 0.77366334, 2.0242143 ,
             0.37477273, 0.88826793, 1.0190021 , 1.904946  , 1.9518087 ,
             1.2677785 , 0.4039423 , 1.3294607 , 0.47779796, 0.95710194,
             1.23786   , 1.0939436 , 1.6338733 , 0.23312815, 0.19628933,
             1.1870846 , 1.0277696 , 0.40081662, 0.15940216, 0.30246064,
             0.8417037 , 0.95927805, 0.95212346, 0.5326788 , 0.64528245,
             0.7276664 , 0.31976745, 1.8555865 , 0.2859598 , 0.5046692 ,
             0.7332148 , 0.40533826, 1.1204103 , 0.4433315 , 0.3248412 ,
             2.192293  , 1.3484986 , 0.31520498, 1.9443543 , 1.6622114 ,
             0.8135123 , 1.5823334 , 0.63508326, 0.2618248 , 0.3287813 ,
             0.3535491 , 0.28178594, 0.71620005, 0.17928712, 0.29929218,
             0.9006489 , 0.7308033 , 0.39481208, 0.9249152 , 1.7869917 ,
             1.0647364 , 1.6364329 , 1.5271417 , 0.22749327, 0.81578225,
             0.7379882 , 0.66389453, 0.6628717 , 0.2733819 , 0.28549173,
             0.23606737, 1.3886667 , 0.16803949, 0.9386295 , 2.2172847 ,
             1.2751876 , 0.93685156, 0.34067133, 0.44561365, 0.72134036,
             0.3757531 , 0.37668836, 1.8723054 , 0.8309141 , 0.7979397 ,
             2.1456444 , 1.1374543 , 1.9793335 , 2.0710495 , 1.2848771 ],
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
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x555e2f09c0d0>, 'bestsig': np.float64(0.075), 'nneigh': 7, 'truezs': array([0.3387837 , 1.2484847 , 2.153771  , 1.0589626 , 1.2609361 ,
           1.9168605 , 1.4007992 , 1.9427918 , 2.0654135 , 0.17257373,
           0.8977886 , 0.26945314, 0.26148003, 1.0591877 , 1.1169813 ,
           0.27561224, 1.6678991 , 1.8990567 , 1.6929686 , 1.1173662 ,
           1.7225233 , 1.3819587 , 1.2092649 , 1.0769911 , 0.7295847 ,
           0.7972144 , 1.9422759 , 1.5583638 , 0.37218407, 1.1825606 ,
           1.2922283 , 2.185101  , 1.660346  , 0.22092588, 0.68016213,
           1.8995954 , 1.8868735 , 0.48495978, 1.9762305 , 0.29599017,
           0.92012775, 1.5730174 , 0.7104293 , 0.29740044, 1.7267003 ,
           0.45903644, 0.21176746, 0.49612883, 2.1182091 , 0.4143995 ,
           1.2394212 , 1.577477  , 2.1925735 , 1.5085403 , 2.2599053 ,
           1.4495722 , 0.33966503, 1.2594613 , 0.46935967, 1.5197884 ,
           1.408432  , 0.33716878, 0.33912238, 1.9967526 , 1.1096987 ,
           0.69080484, 1.2366139 , 1.5823141 , 1.5005487 , 0.41091365,
           1.5099218 , 2.1247592 , 0.8427471 , 1.9725356 , 1.6986089 ,
           0.81424224, 2.0235112 , 0.43857667, 1.5166559 , 1.6643689 ,
           1.2386088 , 1.385784  , 1.0357916 , 1.586993  , 1.4581815 ,
           0.8329263 , 0.6347715 , 0.4416393 , 1.6619018 , 1.1954021 ,
           1.710019  , 0.28416213, 1.15485   , 1.0471253 , 0.36942825,
           0.5142037 , 0.81722975, 1.1447295 , 0.7169016 , 0.15013102,
           0.7603108 , 0.3026472 , 0.52148837, 0.37467706, 2.046937  ,
           0.17641734, 0.47197208, 1.3768597 , 1.5150669 , 0.772303  ,
           0.77221745, 1.3901622 , 1.1456535 , 1.3942121 , 0.73172015,
           1.796524  , 0.5268111 , 1.2542303 , 1.2184184 , 1.8905294 ,
           1.3014894 , 0.74375266, 0.6096792 , 0.8230558 , 0.5392863 ,
           0.12460821, 1.5852716 , 1.4283535 , 1.0215803 , 1.7241379 ,
           1.9044843 , 0.7803245 , 1.9492304 , 2.0915785 , 1.570083  ,
           1.5308751 , 0.11890943, 0.4212692 , 1.7925823 , 2.2249663 ,
           0.1842451 , 0.7113709 , 0.23686388, 0.80016905, 0.73088247,
           1.0766165 , 0.41537255, 1.6848027 , 0.9766652 , 0.31980702,
           1.3925503 , 0.41526327, 0.8231582 , 0.36595705, 1.0231842 ,
           1.2365012 , 1.6561979 , 0.42219388, 0.65557814, 0.6922684 ,
           1.0122771 , 2.1075711 , 0.19901502, 0.47973642, 0.19558333,
           0.9865092 , 1.1884375 , 1.5470389 , 0.32296866, 0.48241863,
           0.45952928, 2.1969178 , 0.35712716, 1.0781019 , 1.0852157 ,
           1.2209102 , 0.23388898, 0.5836977 , 1.0109735 , 0.6622701 ,
           0.300294  , 0.3356636 , 1.1564184 , 0.22877395, 0.2577274 ,
           1.4687084 , 0.31583545, 0.9865428 , 0.3240504 , 1.9025545 ,
           0.26419064, 1.7401885 , 1.2884661 , 0.13800065, 0.4142787 ,
           0.35887143, 0.70411986, 0.9345084 , 0.19424494, 0.5825124 ,
           1.3206912 , 1.0387331 , 0.98988897, 1.2181985 , 1.8136814 ,
           2.004329  , 0.4122194 , 1.0917231 , 0.33617753, 0.6843047 ,
           0.3690189 , 1.0357112 , 1.0776194 , 0.3369648 , 0.9748512 ,
           1.9367574 , 0.35112265, 0.715802  , 0.29704168, 0.12714212,
           0.47079867, 1.5402744 , 1.2326618 , 0.55935544, 1.0218351 ,
           2.151322  , 0.43930042, 0.30776164, 0.74550116, 1.0954635 ,
           2.210278  , 0.69323355, 0.660379  , 1.0471896 , 0.24121569,
           0.19651906, 1.4974073 , 0.9445937 , 1.5089995 , 0.9598642 ,
           0.9005455 , 1.9031551 , 0.38123366, 0.28945082, 1.0247388 ,
           0.3174387 , 0.38283414, 0.5029381 , 0.84094846, 0.29326314,
           0.46964473, 1.2063687 , 0.817962  , 0.67825496, 1.4622391 ,
           1.9775689 , 1.0692734 , 1.85006   , 0.27684304, 1.4282855 ,
           1.0894117 , 1.2913512 , 1.1350669 , 1.5418271 , 0.6917444 ,
           1.2094436 , 1.0492188 , 1.0524127 , 0.12256391, 1.3322802 ,
           1.7683057 , 1.4954526 , 0.23115957, 0.61865664, 0.96132666,
           0.5199129 , 1.7646323 , 0.05754659, 0.9210199 , 1.0734234 ,
           1.2824069 , 1.1859305 , 0.5288401 , 0.4788778 , 1.7773004 ,
           2.1948323 , 0.25191057, 0.39952275, 0.32592618, 2.2187638 ,
           1.453592  , 0.24295595, 0.14553945, 1.6446583 , 0.95385367,
           1.356936  , 1.3907412 , 2.2001233 , 1.9908433 , 1.7229935 ,
           2.0994332 , 0.44873235, 1.4893026 , 0.3521081 , 0.24306677,
           0.37484702, 0.7519174 , 1.4333534 , 0.9700511 , 0.88261896,
           0.949674  , 0.9186223 , 0.29579   , 1.7426188 , 1.5048388 ,
           1.0374715 , 2.0043445 , 0.6813067 , 1.8925664 , 1.3435515 ,
           0.80627555, 0.82640475, 1.7222879 , 0.51165456, 2.1689281 ,
           0.6841359 , 1.2628349 , 0.42523935, 0.13694493, 0.26267603,
           1.6763788 , 0.9370012 , 1.6882255 , 0.230941  , 1.073324  ,
           0.8316961 , 2.24691   , 1.187081  , 1.0098666 , 0.2512058 ,
           0.24307631, 0.71475106, 0.7024954 , 0.44530925, 0.46709555,
           1.5220389 , 1.0647475 , 1.6761858 , 0.65311486, 1.4649705 ,
           1.5797095 , 1.8619741 , 1.5866327 , 0.2673171 , 0.38398167,
           0.42573032, 0.31027374, 1.8861971 , 1.1731787 , 0.7971668 ,
           0.87635094, 1.9429008 , 0.35762537, 0.9308536 , 0.6092299 ,
           1.5966717 , 0.3755022 , 1.2034929 , 0.17367361, 0.31361863,
           1.4443353 , 0.57207286, 1.1167314 , 1.5383885 , 2.1446722 ,
           2.165035  , 0.63585186, 0.2859163 , 0.71970755, 0.6129777 ,
           1.3133336 , 0.9370098 , 0.5676254 , 0.42082402, 1.4824944 ,
           2.2655153 , 0.9016719 , 0.16962297, 1.484996  , 1.6049935 ,
           2.1962085 , 0.65253437, 0.2792837 , 0.93705606, 0.67907447,
           1.2947209 , 0.8118757 , 1.6488634 , 1.3994403 , 0.7008775 ,
           1.6342213 , 1.2435552 , 0.29685268, 0.82000196, 1.6279699 ,
           2.013835  , 1.0658953 , 1.2250209 , 0.29793736, 0.6884921 ,
           0.88409185, 0.7543896 , 0.5572138 , 0.6110236 , 2.174003  ,
           0.6401175 , 0.07403553, 0.3358718 , 1.6684875 , 2.1915298 ,
           1.4325726 , 0.50202614, 1.0128951 , 0.65083647, 0.99642175,
           0.2585532 , 0.60083276, 0.32165393, 1.1643121 , 0.6478291 ,
           1.0469615 , 0.6963149 , 0.08231371, 0.3075918 , 1.6456468 ,
           0.5748981 , 1.9606504 , 0.84170216, 0.5282795 , 1.2481147 ,
           1.9890461 , 0.21737956, 1.7133806 , 0.998776  , 0.91377544,
           0.28719673, 1.3026854 , 1.4113157 , 1.7375115 , 1.8534294 ,
           0.32929924, 0.18261755, 0.20783126, 0.70749456, 0.51771104,
           1.1084858 , 0.80397975, 0.44082564, 0.46448323, 2.1709962 ,
           0.40037915, 1.2334458 , 0.7789311 , 0.5157913 , 0.9123784 ,
           1.8994789 , 0.4295467 , 1.5511605 , 0.37756938, 0.5866857 ,
           0.7362395 , 0.16729401, 0.32926682, 2.1852376 , 1.2001705 ,
           0.990963  , 1.5574335 , 1.019879  , 0.9907941 , 0.7852963 ,
           0.3522209 , 1.2676425 , 0.5657136 , 1.8390493 , 1.3716091 ,
           1.954355  , 0.40749767, 1.457467  , 0.20244673, 0.3017453 ,
           0.34874973, 0.55647886, 2.0248137 , 0.14586294, 0.8868226 ,
           0.43397102, 1.7854514 , 0.9049927 , 1.6703786 , 1.5607324 ,
           0.21144807, 0.52272874, 0.27734432, 0.77333516, 1.7356288 ,
           0.24240845, 1.5862732 , 1.727943  , 0.81161827, 0.38072985,
           0.7859878 , 0.26325673, 0.3207219 , 0.77366334, 2.0242143 ,
           0.37477273, 0.88826793, 1.0190021 , 1.904946  , 1.9518087 ,
           1.2677785 , 0.4039423 , 1.3294607 , 0.47779796, 0.95710194,
           1.23786   , 1.0939436 , 1.6338733 , 0.23312815, 0.19628933,
           1.1870846 , 1.0277696 , 0.40081662, 0.15940216, 0.30246064,
           0.8417037 , 0.95927805, 0.95212346, 0.5326788 , 0.64528245,
           0.7276664 , 0.31976745, 1.8555865 , 0.2859598 , 0.5046692 ,
           0.7332148 , 0.40533826, 1.1204103 , 0.4433315 , 0.3248412 ,
           2.192293  , 1.3484986 , 0.31520498, 1.9443543 , 1.6622114 ,
           0.8135123 , 1.5823334 , 0.63508326, 0.2618248 , 0.3287813 ,
           0.3535491 , 0.28178594, 0.71620005, 0.17928712, 0.29929218,
           0.9006489 , 0.7308033 , 0.39481208, 0.9249152 , 1.7869917 ,
           1.0647364 , 1.6364329 , 1.5271417 , 0.22749327, 0.81578225,
           0.7379882 , 0.66389453, 0.6628717 , 0.2733819 , 0.28549173,
           0.23606737, 1.3886667 , 0.16803949, 0.9386295 , 2.2172847 ,
           1.2751876 , 0.93685156, 0.34067133, 0.44561365, 0.72134036,
           0.3757531 , 0.37668836, 1.8723054 , 0.8309141 , 0.7979397 ,
           2.1456444 , 1.1374543 , 1.9793335 , 2.0710495 , 1.2848771 ],
          dtype=float32), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 231
    Process 0 estimating PZ PDF for rows 0 - 231
    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighEstimator
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x555e29ed76a0>, 'bestsig': np.float64(0.06777777777777778), 'nneigh': 6, 'truezs': array([0.55881345, 0.7505109 , 0.90220624, 1.5336026 , 2.2070146 ,
           0.8955822 , 1.6249051 , 0.23629159, 0.6462868 , 1.7353387 ,
           0.763272  , 0.47564086, 2.2530859 , 1.8554956 , 0.7982221 ,
           0.76877576, 1.3000743 , 1.8611133 , 2.0920973 , 1.43776   ,
           2.001961  , 0.6372048 , 0.9936522 , 0.7628601 , 0.29599035,
           0.38150662, 1.6021278 , 0.238391  , 1.0948358 , 1.1219006 ,
           1.1072104 , 1.0287111 , 0.34911022, 0.25646636, 0.22051866,
           0.3539296 , 2.0317266 , 1.2601942 , 1.6730524 , 1.7939835 ,
           1.1406909 , 1.470882  , 0.4616059 , 1.6799297 , 0.7618903 ,
           1.1093786 , 1.2332758 , 2.2344105 , 1.7830278 , 1.647311  ,
           0.3059654 , 0.5192312 , 0.36335027, 1.003161  , 0.7951077 ,
           0.33331382, 0.24545115, 0.2706954 , 0.23922813, 1.3995671 ,
           0.76424795, 0.3454027 , 0.92014045, 1.50908   , 0.7286523 ,
           0.39165866, 0.37877578, 0.6143071 , 2.0869694 , 0.33980843,
           0.8983433 , 1.452438  , 1.2582    , 0.471642  , 1.2365409 ,
           1.6166989 , 0.38236007, 0.4854227 , 0.3159622 , 1.4187093 ,
           0.33032903, 0.32984808, 0.12638023, 0.90142775, 0.83803195,
           1.1367068 , 0.99584633, 0.8486199 , 0.30471608, 2.241398  ,
           2.0443535 , 1.7040839 , 0.82886666, 2.0554793 , 0.85184926,
           1.5541309 , 0.2941999 , 1.4971268 , 1.3535663 , 1.8222005 ,
           0.7513285 , 0.20425871, 1.0594712 , 2.0609205 , 2.061831  ,
           0.6413929 , 0.81413555, 1.2428627 , 0.33295354, 0.6061036 ,
           1.8412706 , 1.2596604 , 1.165445  , 0.816052  , 1.2004609 ,
           0.11277711, 1.2267913 , 0.89664143, 1.6502886 , 1.232932  ,
           0.20811279, 0.7456885 , 1.1457629 , 1.5560873 , 0.3354844 ,
           1.0612944 , 2.0645301 , 1.7534913 , 0.45086992, 0.86530596,
           2.037356  , 1.753183  , 2.1295207 , 1.5796303 , 0.7473981 ,
           0.37370917, 1.5270216 , 1.2306724 , 0.45983312, 1.8729159 ,
           0.23071945, 0.48873422, 1.1552161 , 0.9053792 , 1.2459372 ,
           0.31662253, 0.9299229 , 2.2198892 , 0.7545864 , 1.1526842 ,
           0.44138792, 1.0930884 , 0.6091568 , 0.27900597, 0.84303904,
           1.1493393 , 1.0321171 , 0.9747351 , 0.31416434, 0.4387843 ,
           0.21525839, 1.221034  , 0.7443083 , 0.75047857, 1.4105809 ,
           1.9781985 , 1.5346011 , 1.9001346 , 2.1914194 , 1.3437325 ,
           0.6514119 , 0.9928134 , 1.2821323 , 1.0787287 , 1.2485256 ,
           0.420011  , 1.1290407 , 2.1706157 , 0.6505394 , 0.49841812,
           1.5165782 , 1.0859427 , 0.7566858 , 1.4679477 , 1.4946868 ,
           1.197381  , 1.3461198 , 0.48871514, 0.6662773 , 0.9728182 ,
           1.9987504 , 2.1722655 , 1.1541808 , 0.20163687, 1.6158202 ,
           1.7021348 , 1.4898676 , 2.062999  , 2.0930305 , 1.5780787 ,
           1.093081  , 1.9427973 , 0.8451926 , 1.0181175 , 0.68932855,
           0.79173756, 1.0233458 , 1.0444417 , 1.1015977 , 2.2592466 ,
           1.6614345 , 1.1971903 , 1.0617965 , 1.1481383 , 1.379136  ,
           1.9159787 , 1.774878  , 1.92184   , 1.4973985 , 0.33018225,
           1.4189883 , 0.26245782, 0.24482591, 0.6538787 , 1.9068872 ,
           0.28651056, 0.11907032, 0.3221812 , 0.34916314, 2.1723266 ,
           0.69444734, 2.1961155 , 1.5877504 , 0.9934564 , 1.422497  ,
           0.8755203 , 1.5608668 , 0.816333  , 1.6358231 , 0.65158004,
           2.220551  , 2.125428  , 0.6436674 , 0.3926067 , 0.33824775,
           1.2296118 , 1.8858044 , 1.8675022 , 1.9799294 , 1.1597168 ,
           0.47570485, 0.16822317, 1.0948414 , 0.23735416, 1.4079769 ,
           0.28268525, 1.3731358 , 1.6045098 , 1.6970054 , 1.9559264 ,
           1.1236519 , 0.65292805, 0.8595312 , 0.41416624, 1.9941795 ,
           1.9751544 , 0.7504347 , 0.54048526, 1.531439  , 1.2437488 ,
           0.86788386, 1.9859314 , 0.2783632 , 2.0742245 , 0.4171769 ,
           0.30643004, 1.45176   , 1.7225975 , 1.1763982 , 0.80028504,
           0.4008354 , 1.221183  , 1.4293817 , 1.3292282 , 0.99560416,
           1.6900824 , 0.32389772, 1.1118728 , 0.31973514, 0.2515174 ,
           0.2248721 , 0.8342227 , 1.5321685 , 0.46760604, 0.7861448 ,
           0.32378957, 1.347512  , 1.9815885 , 0.3396715 , 1.2084031 ,
           0.33192623, 0.38968304, 1.5378948 , 0.7972447 , 0.7408778 ,
           0.6221331 , 1.359355  , 0.21444824, 0.39070052, 1.0248024 ,
           0.35231084, 1.5085105 , 0.9286954 , 2.0569828 , 1.2976897 ,
           0.3316737 , 0.28869924, 1.7134411 , 1.1984626 , 0.7041289 ,
           2.1323972 , 0.67136174, 0.4414976 , 0.67100734, 1.9189782 ,
           0.91290563, 1.4474343 , 0.8390846 , 2.2402608 , 0.35923252,
           0.49004754, 0.8563014 , 0.33946437, 1.7026358 , 0.7837244 ,
           0.6555414 , 0.8881492 , 1.1780074 , 0.2967447 , 0.23448676,
           1.941983  , 0.09235552, 1.7943003 , 1.2103118 , 0.40359628,
           0.27364957, 0.9181916 , 1.910843  , 0.82584083, 0.69151133,
           0.399703  , 1.4629658 , 0.29331177, 0.96935105, 1.3542042 ,
           0.3684387 , 0.22159994, 2.1127956 , 0.33806035, 0.09834345,
           0.360427  , 1.3642906 , 1.8208151 , 1.44029   , 0.08175142,
           0.30219734, 1.7140306 , 0.58381945, 0.33770522, 2.0024023 ,
           1.9155238 , 0.9219813 , 1.7099419 , 0.99845845, 0.8231765 ,
           0.8660293 , 1.319631  , 1.586565  , 0.39935434, 1.1870159 ,
           0.9662966 , 2.0945237 , 1.8463886 , 1.9988604 , 1.6981459 ,
           0.6800698 , 0.9099942 , 0.29400748, 0.37295017, 1.3174622 ,
           0.8960492 , 0.22452182, 1.4450642 , 1.1864197 , 1.6658671 ,
           1.0363033 , 0.78465307, 0.6374488 , 1.7577441 , 0.41653097,
           1.2125803 , 0.87100905, 0.99364805, 0.41182813, 0.5279117 ,
           1.1613647 , 1.5512142 , 0.2661973 , 1.3286062 , 0.6854428 ,
           2.0866234 , 0.4290453 , 1.4687241 , 1.3025283 , 1.5350864 ,
           0.4851526 , 0.6641179 , 0.3803746 , 1.4565063 , 1.356751  ,
           1.0601317 , 2.146504  , 2.2208934 , 1.0648258 , 0.28255364,
           0.775051  , 1.9923803 , 1.3968794 , 0.95884186, 0.48211107,
           1.4530034 , 0.88962495, 0.7468036 , 0.9992443 , 0.7882017 ,
           0.30071828, 0.3206147 , 0.7301343 , 0.8851671 , 1.1096519 ,
           0.35125157, 0.38926303, 0.3848606 , 0.18119505, 0.28397596,
           1.2652915 , 0.6110066 , 0.82539684, 1.7837977 , 1.1145703 ,
           1.0566006 , 1.090981  , 0.29408664, 0.913708  , 0.32547042,
           2.2083452 , 2.0368187 , 0.5154389 , 2.1807764 , 1.2733608 ,
           0.3564471 , 1.2835176 , 0.3303169 , 0.67847925, 1.7083668 ,
           0.6936529 , 0.40413797, 0.4825075 , 1.8901287 , 1.8899792 ,
           0.85065496, 1.642328  , 0.64819735, 0.83763504, 0.5239894 ,
           0.24686363, 1.652617  , 1.4569668 , 0.47304437, 1.3852047 ,
           1.2664164 , 1.0792382 , 1.5192379 , 0.46851328, 0.6742487 ,
           0.5550146 , 1.2255925 , 1.1507967 , 2.2522726 , 0.80655825,
           0.7275481 , 0.28981665, 0.23223399, 0.7194287 , 0.30890796,
           1.0619966 , 0.21677263, 0.31772107, 0.9234608 , 1.0953987 ,
           1.294335  , 0.6379765 , 1.2150372 , 1.002516  , 0.6264234 ,
           0.54890937, 2.2230918 , 0.20849846, 1.620249  , 1.037871  ,
           1.6168774 , 1.6477047 , 1.3238137 , 1.924463  , 0.263706  ,
           2.1617622 , 0.52372473, 0.9449945 , 0.4304488 , 0.7854042 ,
           1.2131426 , 1.8966483 , 0.39485738, 0.33011445, 1.1509819 ,
           1.1336427 , 1.3028526 , 1.9506515 , 0.44942892, 0.7285169 ,
           1.0477815 , 1.4772081 , 1.0688604 , 1.42989   , 1.1834953 ,
           1.4556056 , 1.5882937 , 1.569488  , 1.4288738 , 0.5626003 ,
           1.6023797 , 1.5206873 , 1.4187769 , 1.533052  , 1.16254   ,
           0.3184059 , 2.0222063 , 0.27451056, 1.2514484 , 2.1886182 ,
           0.73146   , 1.5212272 , 0.99738055, 1.0780188 , 0.2693759 ,
           1.3091061 , 1.075804  , 1.2388315 , 0.2629362 , 1.2962663 ,
           0.27403134, 0.40829715, 0.34115484, 1.885298  , 0.59261376,
           1.2963498 , 1.0130947 , 1.8573694 , 0.7822347 , 0.35462645,
           0.84497976, 0.2270768 , 1.6226087 , 1.1656309 , 1.3629788 ,
           1.4145454 , 0.9358979 , 0.48121575, 1.0258199 , 0.23826845,
           1.953058  , 0.507933  , 0.32615516, 1.2145613 , 0.3293191 ,
           0.49019334, 0.19838522, 0.44218338, 1.0391501 , 0.677462  ,
           0.71728635, 0.8130692 , 0.2359062 , 0.491691  , 1.4609389 ,
           0.7558888 , 1.2369905 , 0.48243064, 0.3167364 , 1.3812947 ],
          dtype=float32), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 231
    Process 0 estimating PZ PDF for rows 0 - 231


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighEstimator
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x555e3b8f81d0>, 'bestsig': np.float64(0.075), 'nneigh': 6, 'truezs': array([0.33878371, 1.24848473, 1.05896258, 2.03743152, 1.40079916,
           0.17257373, 0.89778858, 0.26945314, 0.26148003, 1.05918765,
           1.11698127, 0.27561224, 1.11736619, 1.20926487, 1.07699108,
           0.72958469, 0.79721439, 1.94227588, 0.37218407, 1.18256056,
           1.29222834, 1.66034603, 0.22092588, 0.68016213, 0.48495978,
           1.9762305 , 0.29599017, 0.92012775, 0.71042931, 0.29740044,
           0.96012757, 0.21176746, 0.49612883, 0.4143995 , 1.23942125,
           1.57747698, 2.19257355, 0.33966503, 0.46935967, 1.51978838,
           0.33716878, 0.33912238, 1.10969865, 0.69080484, 1.23661387,
           1.58231413, 0.41091365, 1.50992179, 0.84274709, 1.97253561,
           0.81424224, 2.02351117, 0.43857667, 1.66436887, 1.23860884,
           1.38578403, 1.03579164, 2.47546924, 0.83292627, 0.63477153,
           0.4416393 , 1.19540215, 0.28416213, 1.15485001, 1.04712534,
           0.36942825, 0.51420373, 0.81722975, 1.1447295 , 0.7169016 ,
           0.15013102, 0.76031083, 0.3026472 , 0.52148837, 0.37467706,
           2.04693699, 0.17641734, 0.47197208, 1.37685966, 1.51506686,
           0.77230299, 0.77221745, 1.39016223, 1.88255621, 1.39421213,
           0.73172015, 1.79652405, 0.52681112, 1.25423026, 1.21841836,
           1.89052939, 2.09191231, 0.74375266, 0.60967922, 0.8230558 ,
           0.53928632, 0.12460821, 2.26234672, 1.02158034, 1.90448427,
           0.78032452, 1.94923043, 2.09157848, 0.11890943, 0.42126921,
           1.79258227, 2.22496629, 0.18424509, 0.71137089, 0.23686388,
           0.80016905, 0.73088247, 1.07661653, 0.41537255, 0.9766652 ,
           0.31980702, 2.21424728, 0.41526327, 0.8231582 , 0.36595705,
           1.02318418, 1.65619791, 0.91063182, 0.65557814, 0.69226837,
           1.01227713, 2.10757113, 0.19901502, 0.47973642, 0.19558333,
           0.9865092 , 1.18843746, 0.32296866, 0.48241863, 0.45952928,
           2.19691777, 0.35712716, 1.07810187, 1.08521569, 1.98365906,
           0.23388898, 0.58369768, 1.01097345, 0.66227013, 0.74686668,
           0.33566359, 1.15641844, 0.22877395, 0.25772741, 0.31583545,
           0.98654282, 0.3240504 , 0.26419064, 1.2884661 , 0.13800065,
           0.41427869, 0.35887143, 0.70411986, 0.93450838, 0.19424494,
           0.58251238, 1.32069123, 1.03873312, 0.98988897, 1.21819854,
           1.81368136, 2.00432897, 0.41221941, 1.09172308, 0.33617753,
           0.68430471, 0.36901891, 1.73485533, 1.07761943, 0.33696479,
           0.97485119, 0.35112265, 0.71580201, 0.29704168, 0.12714212,
           0.47079867, 1.54027438, 1.23266184, 0.55935544, 1.02183509,
           0.43930042, 0.30776164, 0.74550116, 1.09546351, 2.21027803,
           0.69323355, 0.66037899, 1.04718959, 0.24121569, 0.60745129,
           0.94459373, 1.50899947, 0.9598642 , 0.90054548, 1.90315509,
           0.38123366, 0.28945082, 1.02473879, 0.31743869, 0.38283414,
           0.50293809, 0.84094846, 0.29326314, 0.46964473, 1.20636868,
           0.81796199, 0.67825496, 1.46223915, 1.06927335, 1.85005999,
           0.27684304, 1.42828548, 1.08941174, 1.13506687, 0.69174439,
           1.20944357, 1.04921877, 1.05241275, 0.12256391, 1.33228016,
           1.76830566, 0.23115957, 0.61865664, 0.96132666, 0.5199129 ,
           2.71411702, 0.05754659, 0.92101991, 1.07342339, 1.18593049,
           0.52884012, 0.47887781, 1.77730036, 2.19483232, 0.25191057,
           0.39952275, 0.32592618, 1.45359194, 0.24295595, 0.14553945,
           1.64465833, 0.95385367, 1.35693598, 1.39074123, 1.72299349,
           0.44873235, 2.34422814, 0.35210809, 0.24306677, 0.37484702,
           0.75191742, 1.43335342, 0.97005111, 0.88261896, 0.94967401,
           0.91862231, 0.29578999, 1.50483882, 1.03747153, 0.68130672,
           0.80627555, 1.45366477, 1.03081684, 0.68413591, 1.26283491,
           0.42523935, 0.13694493, 0.26267603, 1.67637885, 0.93700123,
           1.68822551, 0.230941  , 1.07332397, 0.83169609, 2.2469101 ,
           1.18708098, 1.0098666 , 0.2512058 , 0.24307631, 0.71475106,
           0.7024954 , 0.94168592, 0.46709555, 1.52203894, 1.06474745,
           1.67618585, 0.65311486, 1.46497047, 1.57970953, 1.86197412,
           1.58663273, 0.26731709, 0.38398167, 0.42573032, 0.31027374,
           1.88619709, 1.17317867, 0.79716682, 0.87635094, 0.35762537,
           0.93085361, 0.60922992, 0.3755022 , 1.20349288, 0.17367361,
           0.31361863, 1.44433534, 0.57207286, 1.11673141, 1.53838849,
           0.63585186, 0.2859163 , 0.71970755, 0.61297768, 1.31333363,
           0.93700981, 0.5676254 , 0.42082402, 0.90167189, 0.16962297,
           1.60499346, 2.19620848, 0.65253437, 0.2792837 , 0.93705606,
           0.67907447, 1.29472089, 0.8118757 , 1.39944029, 0.70087749,
           1.63422132, 1.24355519, 0.29685268, 0.82000196, 1.62796986,
           1.06589532, 1.22502089, 0.29793736, 0.68849212, 0.88409185,
           0.75438958, 0.55721378, 0.6110236 , 2.17400289, 0.64011753,
           0.07403553, 0.33587179, 1.66848755, 2.19152975, 1.4325726 ,
           0.50202614, 1.01289511, 1.21779935, 0.99642175, 0.25855321,
           0.60083276, 0.32165393, 1.16431212, 0.64782912, 1.04696155,
           0.69631487, 0.08231371, 0.3075918 , 1.64564681, 0.57489812,
           0.84170216, 0.52827948, 0.21737956, 1.71338058, 0.99877602,
           0.91377544, 0.28719673, 1.30268538, 1.41131568, 1.85342944,
           0.32929924, 0.18261755, 0.20783126, 0.70749456, 0.51771104,
           1.10848582, 0.80397975, 0.44082564, 0.46448323, 2.17099619,
           0.40037915, 1.23344576, 0.77893108, 0.5157913 , 0.91237837,
           1.89947891, 0.42954671, 1.55116045, 0.37756938, 0.58668572,
           0.73623949, 0.16729401, 0.32926682, 0.99096298, 1.55743349,
           1.01987898, 0.99079412, 0.78529632, 0.35222089, 0.56571358,
           1.83904934, 1.37160909, 0.40749767, 0.20244673, 0.3017453 ,
           0.34874973, 0.55647886, 0.14586294, 0.88682258, 0.43397102,
           0.9049927 , 1.67037857, 0.21144807, 0.52272874, 0.27734432,
           0.77333516, 1.73562884, 0.24240845, 1.58627319, 0.81161827,
           0.38072985, 1.39936702, 0.26325673, 0.32072189, 0.77366334,
           0.37477273, 1.53677423, 1.01900208, 1.90494597, 1.26777852,
           0.40394229, 1.32946074, 0.47779796, 0.95710194, 1.0939436 ,
           1.63387334, 0.23312815, 0.19628933, 1.18708456, 1.02776957,
           0.40081662, 0.15940216, 0.30246064, 0.84170371, 0.95927805,
           1.62256028, 0.53267878, 1.21033786, 1.32101571, 0.31976745,
           0.28595981, 0.50466919, 0.7332148 , 0.40533826, 1.12041032,
           0.44333151, 0.3248412 , 2.19229293, 1.34849858, 0.31520498,
           2.95556264, 1.66221142, 0.81351233, 1.58233345, 0.63508326,
           0.26182479, 0.32878131, 0.35354909, 0.28178594, 0.71620005,
           0.17928712, 0.29929218, 0.90064889, 0.73080331, 0.39481208,
           0.92491519, 1.06473637, 1.63643289, 1.52714169, 0.22749327,
           0.81578225, 0.73798817, 0.66389453, 0.66287172, 0.27338189,
           0.28549173, 0.23606737, 1.38866675, 0.16803949, 0.93862951,
           2.21728468, 1.27518761, 0.93685156, 0.34067133, 0.44561365,
           0.72134036, 0.3757531 , 0.37668836, 0.83091408, 0.79793972,
           1.13745427, 3.00255512, 2.07104945, 1.28487706]), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 231
    Process 0 estimating PZ PDF for rows 0 - 231
    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighEstimator
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x555e3c4ac9d0>, 'bestsig': np.float64(0.060555555555555564), 'nneigh': 6, 'truezs': array([1.24848473, 0.17257373, 1.05918765, 1.11698127, 0.72958469,
           1.29222834, 0.22092588, 0.29599017, 0.29740044, 0.96012757,
           0.4143995 , 0.46935967, 0.33912238, 0.69080484, 0.41091365,
           0.84274709, 0.81424224, 0.43857667, 1.03579164, 0.83292627,
           0.36942825, 1.1447295 , 0.7169016 , 0.15013102, 0.76031083,
           0.52148837, 0.37467706, 0.17641734, 0.47197208, 1.37685966,
           1.51506686, 0.73172015, 0.52681112, 1.25423026, 0.74375266,
           0.8230558 , 0.12460821, 2.26234672, 1.02158034, 0.11890943,
           0.42126921, 0.80016905, 0.73088247, 0.41537255, 0.31980702,
           0.41526327, 1.02318418, 1.65619791, 1.01227713, 0.47973642,
           1.18843746, 0.32296866, 0.48241863, 0.35712716, 1.07810187,
           1.98365906, 0.58369768, 1.01097345, 0.66227013, 0.22877395,
           0.3240504 , 0.26419064, 1.2884661 , 0.41427869, 0.70411986,
           1.03873312, 0.98988897, 1.21819854, 2.00432897, 0.41221941,
           1.09172308, 0.33617753, 1.73485533, 1.07761943, 0.33696479,
           0.97485119, 0.71580201, 0.12714212, 0.47079867, 1.09546351,
           1.04718959, 0.24121569, 0.60745129, 0.94459373, 1.50899947,
           0.38123366, 0.28945082, 1.02473879, 0.31743869, 0.38283414,
           0.84094846, 0.29326314, 1.20636868, 0.81796199, 0.67825496,
           1.46223915, 1.06927335, 1.13506687, 0.69174439, 1.20944357,
           1.05241275, 0.23115957, 0.61865664, 0.96132666, 0.5199129 ,
           0.05754659, 1.07342339, 1.18593049, 0.52884012, 0.47887781,
           0.25191057, 0.39952275, 1.64465833, 0.95385367, 1.35693598,
           1.39074123, 0.44873235, 0.35210809, 0.37484702, 0.75191742,
           0.94967401, 0.91862231, 0.29578999, 1.03747153, 0.68130672,
           0.80627555, 1.03081684, 0.68413591, 0.230941  , 1.07332397,
           1.18708098, 1.0098666 , 0.24307631, 0.94168592, 0.46709555,
           1.52203894, 0.65311486, 0.42573032, 0.79716682, 0.35762537,
           0.93085361, 0.60922992, 0.3755022 , 1.20349288, 0.57207286,
           1.53838849, 0.63585186, 0.2859163 , 0.71970755, 0.61297768,
           0.90167189, 0.16962297, 1.60499346, 0.67907447, 0.70087749,
           0.82000196, 1.62796986, 0.68849212, 0.88409185, 0.75438958,
           0.6110236 , 0.64011753, 0.07403553, 0.33587179, 0.50202614,
           1.01289511, 0.99642175, 0.25855321, 0.60083276, 0.32165393,
           0.69631487, 0.08231371, 0.57489812, 0.21737956, 1.71338058,
           0.99877602, 0.91377544, 0.28719673, 0.20783126, 0.70749456,
           0.51771104, 0.80397975, 0.44082564, 0.40037915, 0.77893108,
           0.91237837, 0.42954671, 0.16729401, 0.99096298, 0.99079412,
           0.35222089, 0.40749767, 0.34874973, 0.55647886, 0.14586294,
           0.88682258, 0.43397102, 0.52272874, 0.77333516, 1.73562884,
           0.24240845, 0.38072985, 0.26325673, 0.32072189, 0.37477273,
           1.26777852, 0.40394229, 0.47779796, 0.23312815, 0.15940216,
           0.30246064, 0.84170371, 0.53267878, 1.21033786, 0.31976745,
           0.50466919, 0.7332148 , 0.40533826, 0.3248412 , 0.31520498,
           0.63508326, 0.32878131, 0.28178594, 0.71620005, 0.17928712,
           0.92491519, 1.06473637, 1.52714169, 0.81578225, 0.66287172,
           0.28549173, 0.23606737, 1.38866675, 0.16803949, 0.34067133,
           0.44561365, 0.72134036, 0.37668836, 0.79793972]), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 231
    Process 0 estimating PZ PDF for rows 0 - 231


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

    {'output': Ensemble(the_class=mixmod,shape=(231, 7))}


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

    {'weights': array([[0.16375052, 0.15837549, 0.15124112, ..., 0.13704563, 0.12764297,
            0.12101177],
           [0.20693113, 0.16172259, 0.13711467, ..., 0.13518609, 0.1227389 ,
            0.10047821],
           [0.17928897, 0.17477274, 0.1637517 , ..., 0.11954436, 0.11787295,
            0.11630643],
           ...,
           [0.16207927, 0.15072217, 0.14882856, ..., 0.13551712, 0.13391748,
            0.13341235],
           [0.19380311, 0.14105894, 0.13944058, ..., 0.13667272, 0.13048491,
            0.12001908],
           [0.16612199, 0.15728006, 0.13990405, ..., 0.1361796 , 0.13602757,
            0.1255849 ]], shape=(231, 7)), 'stds': array([[0.075, 0.075, 0.075, ..., 0.075, 0.075, 0.075],
           [0.075, 0.075, 0.075, ..., 0.075, 0.075, 0.075],
           [0.075, 0.075, 0.075, ..., 0.075, 0.075, 0.075],
           ...,
           [0.075, 0.075, 0.075, ..., 0.075, 0.075, 0.075],
           [0.075, 0.075, 0.075, ..., 0.075, 0.075, 0.075],
           [0.075, 0.075, 0.075, ..., 0.075, 0.075, 0.075]], shape=(231, 7)), 'means': array([[0.81578225, 0.63508326, 0.37477273, ..., 0.80397975, 0.71620005,
            0.7979397 ],
           [0.8427471 , 0.7972144 , 0.73088247, ..., 0.7008775 , 0.6917444 ,
            0.9016719 ],
           [0.70749456, 0.95212346, 0.6917444 , ..., 0.69080484, 1.0122771 ,
            0.9016719 ],
           ...,
           [1.0647475 , 2.0994332 , 1.710019  , ..., 0.96132666, 1.0769911 ,
            1.2676425 ],
           [0.4143995 , 0.14553945, 0.47079867, ..., 0.4142787 , 0.35712716,
            0.37467706],
           [0.9598642 , 1.6929686 , 0.65083647, ..., 1.3819587 , 1.3925503 ,
            1.2913512 ]], shape=(231, 7), dtype=float32)}


Typically the ancillary data table includes a photo-z point estimate
derived from the PDFs, by default this is the mode of the distribution,
called ‘zmode’ in the ancillary dictionary below:

.. code:: ipython3

    # this is the ancillary dictionary of the output Ensemble, which in this case
    # contains the zmode, redshift, and distribution type
    print(estimated_photoz["lsst_error_model"]["output"].ancil)


.. parsed-literal::

    {'zmode': array([[0.78],
           [0.74],
           [0.69],
           [0.93],
           [0.7 ],
           [1.26],
           [0.72],
           [0.82],
           [1.39],
           [0.8 ],
           [1.02],
           [0.31],
           [0.44],
           [1.09],
           [1.04],
           [0.3 ],
           [0.45],
           [1.04],
           [0.39],
           [0.38],
           [1.73],
           [0.72],
           [0.32],
           [0.22],
           [0.5 ],
           [2.12],
           [0.36],
           [1.54],
           [0.34],
           [1.08],
           [0.34],
           [0.38],
           [0.3 ],
           [0.29],
           [0.13],
           [0.81],
           [0.78],
           [0.76],
           [0.3 ],
           [0.43],
           [0.99],
           [0.75],
           [0.23],
           [1.04],
           [0.7 ],
           [0.7 ],
           [0.41],
           [1.26],
           [0.97],
           [1.06],
           [1.28],
           [0.17],
           [0.79],
           [0.25],
           [0.7 ],
           [0.28],
           [1.05],
           [0.52],
           [0.71],
           [0.3 ],
           [1.02],
           [0.78],
           [0.4 ],
           [0.69],
           [0.33],
           [0.37],
           [0.25],
           [0.7 ],
           [1.08],
           [1.28],
           [1.03],
           [0.53],
           [1.23],
           [1.56],
           [0.28],
           [1.23],
           [0.29],
           [0.7 ],
           [1.05],
           [0.74],
           [1.05],
           [1.02],
           [1.43],
           [0.23],
           [0.3 ],
           [0.65],
           [0.33],
           [0.26],
           [0.45],
           [0.74],
           [1.02],
           [0.38],
           [0.33],
           [0.3 ],
           [1.17],
           [1.13],
           [0.42],
           [0.24],
           [0.21],
           [2.01],
           [1.02],
           [0.72],
           [0.76],
           [0.39],
           [0.72],
           [0.5 ],
           [0.72],
           [0.4 ],
           [0.42],
           [0.3 ],
           [0.75],
           [0.3 ],
           [1.02],
           [0.95],
           [0.63],
           [1.09],
           [0.32],
           [0.26],
           [0.39],
           [0.41],
           [2.14],
           [0.18],
           [0.45],
           [0.7 ],
           [0.45],
           [0.92],
           [0.38],
           [0.43],
           [0.39],
           [0.65],
           [0.27],
           [0.17],
           [2.01],
           [0.94],
           [0.38],
           [0.3 ],
           [1.03],
           [0.28],
           [0.3 ],
           [0.2 ],
           [0.37],
           [1.03],
           [1.07],
           [0.97],
           [1.16],
           [0.38],
           [0.4 ],
           [1.4 ],
           [1.07],
           [0.72],
           [0.34],
           [1.01],
           [0.29],
           [0.38],
           [0.28],
           [0.32],
           [1.35],
           [2.14],
           [0.26],
           [0.84],
           [0.68],
           [0.39],
           [0.73],
           [0.4 ],
           [0.14],
           [0.29],
           [0.78],
           [0.77],
           [0.3 ],
           [1.03],
           [0.33],
           [0.29],
           [0.75],
           [0.89],
           [0.31],
           [0.43],
           [0.76],
           [0.5 ],
           [0.8 ],
           [0.46],
           [1.05],
           [0.78],
           [0.49],
           [0.32],
           [0.25],
           [1.04],
           [0.45],
           [1.02],
           [1.02],
           [0.7 ],
           [1.23],
           [0.38],
           [0.3 ],
           [0.22],
           [1.28],
           [0.45],
           [1.02],
           [0.27],
           [0.75],
           [0.42],
           [1.21],
           [0.99],
           [1.11],
           [0.53],
           [1.43],
           [0.28],
           [1.1 ],
           [0.69],
           [1.04],
           [0.98],
           [1.6 ],
           [1.05],
           [0.42],
           [0.46],
           [0.47],
           [1.29],
           [1.08],
           [0.78],
           [0.22],
           [0.56],
           [0.22],
           [1.2 ],
           [0.33],
           [0.26],
           [1.01],
           [0.64],
           [0.73],
           [0.69],
           [1.05],
           [0.4 ],
           [1.37]]), 'redshift': 0      0.558813
    1      0.750511
    2      0.902206
    3      0.895582
    4      0.646287
             ...   
    226    0.813069
    227    0.755889
    228    2.005262
    229    0.482431
    230    2.199126
    Name: redshift, Length: 231, dtype: float64, 'distribution_type': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}


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
    Process 0 running estimator on chunk 0 - 231


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  input: None, NaiveStackSummarizer
    Process 0 running estimator on chunk 0 - 231
    Inserting handle into data store.  output: inprogress_output.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  input: None, PointEstHistSummarizer
    Process 0 running estimator on chunk 0 - 231


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  input: None, NaiveStackSummarizer
    Process 0 running estimator on chunk 0 - 231
    Inserting handle into data store.  output: inprogress_output.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  input: None, PointEstHistSummarizer
    Process 0 running estimator on chunk 0 - 231


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  input: None, NaiveStackSummarizer
    Process 0 running estimator on chunk 0 - 231
    Inserting handle into data store.  output: inprogress_output.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  input: None, PointEstHistSummarizer
    Process 0 running estimator on chunk 0 - 231


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  input: None, NaiveStackSummarizer
    Process 0 running estimator on chunk 0 - 231
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
          <td>0.042493</td>
          <td>0.027213</td>
          <td>0.042968</td>
          <td>0.025888</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.01</td>
          <td>0.053777</td>
          <td>0.034604</td>
          <td>0.054636</td>
          <td>0.035446</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.02</td>
          <td>0.067273</td>
          <td>0.043354</td>
          <td>0.068639</td>
          <td>0.047673</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.03</td>
          <td>0.083214</td>
          <td>0.053573</td>
          <td>0.085219</td>
          <td>0.063012</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.04</td>
          <td>0.101813</td>
          <td>0.065379</td>
          <td>0.104596</td>
          <td>0.081880</td>
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
    0    0.558813   24.713184   23.571205   22.227867   21.358557   20.996706   
    1    0.750511   26.330038   26.115959   25.643736   24.997763   24.764048   
    2    0.902206   26.077417   25.778303   25.293756   24.599886   24.188847   
    3    0.895582   24.499634   24.176474   23.566423   22.824486   22.369812   
    4    0.646287   26.026688   25.736584   25.132135   24.595406   24.405972   
    ..        ...         ...         ...         ...         ...         ...   
    226  0.813069   26.210548   26.045996   25.626324   24.914921   24.677912   
    227  0.755889   25.345015   24.436749   23.613729   22.758505   22.476389   
    228  1.236990   24.090199   23.853308   23.583414   23.060661   21.655033   
    229  0.482431   26.793308   25.781473   24.640827   24.271374   24.107733   
    230  1.381295   26.768213   26.117926   25.567503   24.973419   24.633711   
    
         mag_y_lsst  
    0     20.703762  
    1     24.650198  
    2     24.020731  
    3     22.198908  
    4     24.248049  
    ..          ...  
    226   24.663494  
    227   22.314535  
    228   21.169876  
    229   23.881256  
    230   24.049259  
    
    [231 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.558813   24.713184   23.571205   22.227867   21.358557   20.996706   
    1    0.750511   26.330038   26.115959   25.643736   24.997763   24.764048   
    2    0.902206   26.077417   25.778303   25.293756   24.599886   24.188847   
    3    0.895582   24.499634   24.176474   23.566423   22.824486   22.369812   
    4    0.646287   26.026688   25.736584   25.132135   24.595406   24.405972   
    ..        ...         ...         ...         ...         ...         ...   
    226  0.813069   26.210548   26.045996   25.626324   24.914921   24.677912   
    227  0.755889   25.345015   24.436749   23.613729   22.758505   22.476389   
    228  1.236990   24.090199   23.853308   23.583414   23.060661   21.655033   
    229  0.482431   26.793308   25.781473   24.640827   24.271374   24.107733   
    230  1.381295   26.768213   26.117926   25.567503   24.973419   24.633711   
    
         mag_y_lsst  
    0     20.703762  
    1     24.650198  
    2     24.020731  
    3     22.198908  
    4     24.248049  
    ..          ...  
    226   24.663494  
    227   22.314535  
    228   21.169876  
    229   23.881256  
    230   24.049259  
    
    [231 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.558813   24.713184   23.571205   22.227867   21.358557   20.996706   
    1    0.750511   26.330038   26.115959   25.643736   24.997763   24.764048   
    2    0.902206   26.077417   25.778303   25.293756   24.599886   24.188847   
    3    0.895582   24.499634   24.176474   23.566423   22.824486   22.369812   
    4    0.646287   26.026688   25.736584   25.132135   24.595406   24.405972   
    ..        ...         ...         ...         ...         ...         ...   
    226  0.813069   26.210548   26.045996   25.626324   24.914921   24.677912   
    227  0.755889   25.345015   24.436749   23.613729   22.758505   22.476389   
    228  1.236990   24.090199   23.853308   23.583414   23.060661   21.655033   
    229  0.482431   26.793308   25.781473   24.640827   24.271374   24.107733   
    230  1.381295   26.768213   26.117926   25.567503   24.973419   24.633711   
    
         mag_y_lsst  
    0     20.703762  
    1     24.650198  
    2     24.020731  
    3     22.198908  
    4     24.248049  
    ..          ...  
    226   24.663494  
    227   22.314535  
    228   21.169876  
    229   23.881256  
    230   24.049259  
    
    [231 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.558813   24.713184   23.571205   22.227867   21.358557   20.996706   
    1    0.750511   26.330038   26.115959   25.643736   24.997763   24.764048   
    2    0.902206   26.077417   25.778303   25.293756   24.599886   24.188847   
    3    0.895582   24.499634   24.176474   23.566423   22.824486   22.369812   
    4    0.646287   26.026688   25.736584   25.132135   24.595406   24.405972   
    ..        ...         ...         ...         ...         ...         ...   
    226  0.813069   26.210548   26.045996   25.626324   24.914921   24.677912   
    227  0.755889   25.345015   24.436749   23.613729   22.758505   22.476389   
    228  1.236990   24.090199   23.853308   23.583414   23.060661   21.655033   
    229  0.482431   26.793308   25.781473   24.640827   24.271374   24.107733   
    230  1.381295   26.768213   26.117926   25.567503   24.973419   24.633711   
    
         mag_y_lsst  
    0     20.703762  
    1     24.650198  
    2     24.020731  
    3     22.198908  
    4     24.248049  
    ..          ...  
    226   24.663494  
    227   22.314535  
    228   21.169876  
    229   23.881256  
    230   24.049259  
    
    [231 rows x 7 columns], DistToPointEvaluator
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

    {'lsst_error_model': {'cdeloss': array([-2.83263549]),
      'brier': array([213.40963265])},
     'inv_redshift_inc': {'cdeloss': array([-7.6031252]),
      'brier': array([404.03050759])},
     'line_confusion': {'cdeloss': array([-2.71549204]),
      'brier': array([216.94598826])},
     'quantity_cut': {'cdeloss': array([-2.70200076]),
      'brier': array([258.3239495])}}



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
