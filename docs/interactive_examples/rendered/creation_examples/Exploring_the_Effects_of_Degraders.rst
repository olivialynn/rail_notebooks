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
    /home/runner/.cache/lephare/runs/20260406T120756


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
      File "/tmp/ipykernel_5091/1847479680.py", line 1, in <module>
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

    (1) 0.8740


.. parsed-literal::

    (2) -0.5913


.. parsed-literal::

    (3) 0.7176


.. parsed-literal::

    (4) -0.9093


.. parsed-literal::

    (5) -0.4305


.. parsed-literal::

    (6) -1.7633


.. parsed-literal::

    (7) -1.4254


.. parsed-literal::

    (8) -2.6962


.. parsed-literal::

    (9) -2.8001


.. parsed-literal::

    (10) -1.5271


.. parsed-literal::

    (11) -3.1819


.. parsed-literal::

    (12) -1.8189


.. parsed-literal::

    (13) -2.6294


.. parsed-literal::

    (14) -3.5693


.. parsed-literal::

    (15) -3.5883


.. parsed-literal::

    (16) -3.2828


.. parsed-literal::

    (17) -3.5843


.. parsed-literal::

    (18) -2.2753


.. parsed-literal::

    (19) -4.0064


.. parsed-literal::

    (20) -2.5626


.. parsed-literal::

    (21) -3.7406


.. parsed-literal::

    (22) -3.2074


.. parsed-literal::

    (23) -4.3113


.. parsed-literal::

    (24) -3.1268


.. parsed-literal::

    (25) -4.0936


.. parsed-literal::

    (26) -3.1948


.. parsed-literal::

    (27) -4.4349


.. parsed-literal::

    (28) -4.5118


.. parsed-literal::

    (29) -4.7653


.. parsed-literal::

    (30) -4.3649


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

    Inserting handle into data store.  model: <pzflow.flow.Flow object at 0x7f4350915ff0>, FlowCreator


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.pq, FlowCreator
    Inserting handle into data store.  model: <pzflow.flow.Flow object at 0x7f4350915ff0>, FlowCreator
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
          <td>1.457567</td>
          <td>27.440129</td>
          <td>0.744942</td>
          <td>28.107785</td>
          <td>0.508110</td>
          <td>27.597370</td>
          <td>0.307001</td>
          <td>26.951300</td>
          <td>0.284961</td>
          <td>26.776778</td>
          <td>0.441503</td>
          <td>26.471038</td>
          <td>0.686953</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.325402</td>
          <td>26.085218</td>
          <td>0.271052</td>
          <td>25.873620</td>
          <td>0.080174</td>
          <td>25.699163</td>
          <td>0.060412</td>
          <td>25.084903</td>
          <td>0.057243</td>
          <td>24.589847</td>
          <td>0.070688</td>
          <td>23.996821</td>
          <td>0.094249</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.893867</td>
          <td>26.444374</td>
          <td>0.361029</td>
          <td>26.441589</td>
          <td>0.131683</td>
          <td>24.714692</td>
          <td>0.025325</td>
          <td>23.586648</td>
          <td>0.015592</td>
          <td>22.720113</td>
          <td>0.014023</td>
          <td>22.409525</td>
          <td>0.023246</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.787176</td>
          <td>28.984102</td>
          <td>1.765826</td>
          <td>27.223043</td>
          <td>0.254704</td>
          <td>27.740689</td>
          <td>0.344067</td>
          <td>27.287371</td>
          <td>0.372212</td>
          <td>26.305232</td>
          <td>0.305609</td>
          <td>25.592537</td>
          <td>0.359995</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.302208</td>
          <td>25.703932</td>
          <td>0.197784</td>
          <td>25.659319</td>
          <td>0.066354</td>
          <td>25.283083</td>
          <td>0.041757</td>
          <td>24.517841</td>
          <td>0.034620</td>
          <td>23.917928</td>
          <td>0.038952</td>
          <td>23.361045</td>
          <td>0.053723</td>
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

    <matplotlib.colorbar.Colorbar at 0x7f42ce387850>




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
          <td>0.750908</td>
          <td>26.788129</td>
          <td>0.469612</td>
          <td>26.883947</td>
          <td>0.192125</td>
          <td>26.276444</td>
          <td>0.100552</td>
          <td>25.603776</td>
          <td>0.090571</td>
          <td>25.548348</td>
          <td>0.163059</td>
          <td>25.937608</td>
          <td>0.468907</td>
        </tr>
        <tr>
          <th>1</th>
          <td>True</td>
          <td>0.648564</td>
          <td>23.139859</td>
          <td>0.021444</td>
          <td>22.359534</td>
          <td>0.006135</td>
          <td>21.489193</td>
          <td>0.005210</td>
          <td>20.729761</td>
          <td>0.005151</td>
          <td>20.490030</td>
          <td>0.005334</td>
          <td>20.287871</td>
          <td>0.006056</td>
        </tr>
        <tr>
          <th>2</th>
          <td>True</td>
          <td>0.275019</td>
          <td>27.072953</td>
          <td>0.578208</td>
          <td>27.961646</td>
          <td>0.455798</td>
          <td>26.862689</td>
          <td>0.166976</td>
          <td>26.894715</td>
          <td>0.272169</td>
          <td>26.887629</td>
          <td>0.479795</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>False</td>
          <td>1.727078</td>
          <td>27.958106</td>
          <td>1.032567</td>
          <td>25.900969</td>
          <td>0.082128</td>
          <td>25.423261</td>
          <td>0.047288</td>
          <td>24.785029</td>
          <td>0.043862</td>
          <td>24.145750</td>
          <td>0.047673</td>
          <td>23.723136</td>
          <td>0.074051</td>
        </tr>
        <tr>
          <th>4</th>
          <td>True</td>
          <td>0.499235</td>
          <td>27.149999</td>
          <td>0.610654</td>
          <td>27.340766</td>
          <td>0.280362</td>
          <td>25.738337</td>
          <td>0.062548</td>
          <td>25.074234</td>
          <td>0.056703</td>
          <td>24.745917</td>
          <td>0.081139</td>
          <td>24.890352</td>
          <td>0.203409</td>
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
          <td>0.569906</td>
          <td>29.198308</td>
          <td>1.941090</td>
          <td>26.983756</td>
          <td>0.208913</td>
          <td>25.857585</td>
          <td>0.069518</td>
          <td>25.425932</td>
          <td>0.077431</td>
          <td>25.000691</td>
          <td>0.101506</td>
          <td>25.682421</td>
          <td>0.386107</td>
        </tr>
        <tr>
          <th>596</th>
          <td>True</td>
          <td>0.423343</td>
          <td>25.361537</td>
          <td>0.147941</td>
          <td>25.543017</td>
          <td>0.059864</td>
          <td>25.313284</td>
          <td>0.042890</td>
          <td>25.283270</td>
          <td>0.068252</td>
          <td>25.294399</td>
          <td>0.131084</td>
          <td>24.966467</td>
          <td>0.216777</td>
        </tr>
        <tr>
          <th>597</th>
          <td>True</td>
          <td>0.366884</td>
          <td>25.424068</td>
          <td>0.156065</td>
          <td>24.547564</td>
          <td>0.024935</td>
          <td>23.696877</td>
          <td>0.011096</td>
          <td>23.479646</td>
          <td>0.014310</td>
          <td>23.238418</td>
          <td>0.021510</td>
          <td>23.213067</td>
          <td>0.047109</td>
        </tr>
        <tr>
          <th>598</th>
          <td>True</td>
          <td>0.313108</td>
          <td>29.073612</td>
          <td>1.838296</td>
          <td>27.301175</td>
          <td>0.271489</td>
          <td>26.387450</td>
          <td>0.110799</td>
          <td>26.691282</td>
          <td>0.230277</td>
          <td>26.346142</td>
          <td>0.315782</td>
          <td>25.976749</td>
          <td>0.482789</td>
        </tr>
        <tr>
          <th>599</th>
          <td>True</td>
          <td>0.222359</td>
          <td>27.747188</td>
          <td>0.908112</td>
          <td>26.091389</td>
          <td>0.097082</td>
          <td>25.928403</td>
          <td>0.074013</td>
          <td>25.612108</td>
          <td>0.091237</td>
          <td>25.832657</td>
          <td>0.207377</td>
          <td>25.967632</td>
          <td>0.479527</td>
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
          <td>0.750908</td>
          <td>26.788129</td>
          <td>0.469612</td>
          <td>26.883947</td>
          <td>0.192125</td>
          <td>26.276444</td>
          <td>0.100552</td>
          <td>25.603776</td>
          <td>0.090571</td>
          <td>25.548348</td>
          <td>0.163059</td>
          <td>25.937608</td>
          <td>0.468907</td>
        </tr>
        <tr>
          <th>1</th>
          <td>True</td>
          <td>0.648564</td>
          <td>23.139859</td>
          <td>0.021444</td>
          <td>22.359534</td>
          <td>0.006135</td>
          <td>21.489193</td>
          <td>0.005210</td>
          <td>20.729761</td>
          <td>0.005151</td>
          <td>20.490030</td>
          <td>0.005334</td>
          <td>20.287871</td>
          <td>0.006056</td>
        </tr>
        <tr>
          <th>2</th>
          <td>True</td>
          <td>0.275019</td>
          <td>27.072953</td>
          <td>0.578208</td>
          <td>27.961646</td>
          <td>0.455798</td>
          <td>26.862689</td>
          <td>0.166976</td>
          <td>26.894715</td>
          <td>0.272169</td>
          <td>26.887629</td>
          <td>0.479795</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>False</td>
          <td>1.727078</td>
          <td>27.958106</td>
          <td>1.032567</td>
          <td>25.900969</td>
          <td>0.082128</td>
          <td>25.423261</td>
          <td>0.047288</td>
          <td>24.785029</td>
          <td>0.043862</td>
          <td>24.145750</td>
          <td>0.047673</td>
          <td>23.723136</td>
          <td>0.074051</td>
        </tr>
        <tr>
          <th>4</th>
          <td>True</td>
          <td>0.499235</td>
          <td>27.149999</td>
          <td>0.610654</td>
          <td>27.340766</td>
          <td>0.280362</td>
          <td>25.738337</td>
          <td>0.062548</td>
          <td>25.074234</td>
          <td>0.056703</td>
          <td>24.745917</td>
          <td>0.081139</td>
          <td>24.890352</td>
          <td>0.203409</td>
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
          <td>0.569906</td>
          <td>29.198308</td>
          <td>1.941090</td>
          <td>26.983756</td>
          <td>0.208913</td>
          <td>25.857585</td>
          <td>0.069518</td>
          <td>25.425932</td>
          <td>0.077431</td>
          <td>25.000691</td>
          <td>0.101506</td>
          <td>25.682421</td>
          <td>0.386107</td>
        </tr>
        <tr>
          <th>596</th>
          <td>True</td>
          <td>0.423343</td>
          <td>25.361537</td>
          <td>0.147941</td>
          <td>25.543017</td>
          <td>0.059864</td>
          <td>25.313284</td>
          <td>0.042890</td>
          <td>25.283270</td>
          <td>0.068252</td>
          <td>25.294399</td>
          <td>0.131084</td>
          <td>24.966467</td>
          <td>0.216777</td>
        </tr>
        <tr>
          <th>597</th>
          <td>True</td>
          <td>0.366884</td>
          <td>25.424068</td>
          <td>0.156065</td>
          <td>24.547564</td>
          <td>0.024935</td>
          <td>23.696877</td>
          <td>0.011096</td>
          <td>23.479646</td>
          <td>0.014310</td>
          <td>23.238418</td>
          <td>0.021510</td>
          <td>23.213067</td>
          <td>0.047109</td>
        </tr>
        <tr>
          <th>598</th>
          <td>True</td>
          <td>0.313108</td>
          <td>29.073612</td>
          <td>1.838296</td>
          <td>27.301175</td>
          <td>0.271489</td>
          <td>26.387450</td>
          <td>0.110799</td>
          <td>26.691282</td>
          <td>0.230277</td>
          <td>26.346142</td>
          <td>0.315782</td>
          <td>25.976749</td>
          <td>0.482789</td>
        </tr>
        <tr>
          <th>599</th>
          <td>True</td>
          <td>0.222359</td>
          <td>27.747188</td>
          <td>0.908112</td>
          <td>26.091389</td>
          <td>0.097082</td>
          <td>25.928403</td>
          <td>0.074013</td>
          <td>25.612108</td>
          <td>0.091237</td>
          <td>25.832657</td>
          <td>0.207377</td>
          <td>25.967632</td>
          <td>0.479527</td>
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
          <td>0.750908</td>
          <td>26.788129</td>
          <td>0.469612</td>
          <td>26.883947</td>
          <td>0.192125</td>
          <td>26.276444</td>
          <td>0.100552</td>
          <td>25.603776</td>
          <td>0.090571</td>
          <td>25.548348</td>
          <td>0.163059</td>
          <td>25.937608</td>
          <td>0.468907</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.648564</td>
          <td>23.139859</td>
          <td>0.021444</td>
          <td>22.359534</td>
          <td>0.006135</td>
          <td>21.489193</td>
          <td>0.005210</td>
          <td>20.729761</td>
          <td>0.005151</td>
          <td>20.490030</td>
          <td>0.005334</td>
          <td>20.287871</td>
          <td>0.006056</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.275019</td>
          <td>27.072953</td>
          <td>0.578208</td>
          <td>27.961646</td>
          <td>0.455798</td>
          <td>26.862689</td>
          <td>0.166976</td>
          <td>26.894715</td>
          <td>0.272169</td>
          <td>26.887629</td>
          <td>0.479795</td>
          <td>NaN</td>
          <td>NaN</td>
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
          <td>0.499235</td>
          <td>27.149999</td>
          <td>0.610654</td>
          <td>27.340766</td>
          <td>0.280362</td>
          <td>25.738337</td>
          <td>0.062548</td>
          <td>25.074234</td>
          <td>0.056703</td>
          <td>24.745917</td>
          <td>0.081139</td>
          <td>24.890352</td>
          <td>0.203409</td>
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
          <td>0.569906</td>
          <td>29.198308</td>
          <td>1.941090</td>
          <td>26.983756</td>
          <td>0.208913</td>
          <td>25.857585</td>
          <td>0.069518</td>
          <td>25.425932</td>
          <td>0.077431</td>
          <td>25.000691</td>
          <td>0.101506</td>
          <td>25.682421</td>
          <td>0.386107</td>
        </tr>
        <tr>
          <th>596</th>
          <td>0.423343</td>
          <td>25.361537</td>
          <td>0.147941</td>
          <td>25.543017</td>
          <td>0.059864</td>
          <td>25.313284</td>
          <td>0.042890</td>
          <td>25.283270</td>
          <td>0.068252</td>
          <td>25.294399</td>
          <td>0.131084</td>
          <td>24.966467</td>
          <td>0.216777</td>
        </tr>
        <tr>
          <th>597</th>
          <td>0.366884</td>
          <td>25.424068</td>
          <td>0.156065</td>
          <td>24.547564</td>
          <td>0.024935</td>
          <td>23.696877</td>
          <td>0.011096</td>
          <td>23.479646</td>
          <td>0.014310</td>
          <td>23.238418</td>
          <td>0.021510</td>
          <td>23.213067</td>
          <td>0.047109</td>
        </tr>
        <tr>
          <th>598</th>
          <td>0.313108</td>
          <td>29.073612</td>
          <td>1.838296</td>
          <td>27.301175</td>
          <td>0.271489</td>
          <td>26.387450</td>
          <td>0.110799</td>
          <td>26.691282</td>
          <td>0.230277</td>
          <td>26.346142</td>
          <td>0.315782</td>
          <td>25.976749</td>
          <td>0.482789</td>
        </tr>
        <tr>
          <th>599</th>
          <td>0.222359</td>
          <td>27.747188</td>
          <td>0.908112</td>
          <td>26.091389</td>
          <td>0.097082</td>
          <td>25.928403</td>
          <td>0.074013</td>
          <td>25.612108</td>
          <td>0.091237</td>
          <td>25.832657</td>
          <td>0.207377</td>
          <td>25.967632</td>
          <td>0.479527</td>
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
          <td>0.750908</td>
          <td>26.788129</td>
          <td>0.469612</td>
          <td>26.883947</td>
          <td>0.192125</td>
          <td>26.276444</td>
          <td>0.100552</td>
          <td>25.603776</td>
          <td>0.090571</td>
          <td>25.548348</td>
          <td>0.163059</td>
          <td>25.937608</td>
          <td>0.468907</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>0.648564</td>
          <td>23.139859</td>
          <td>0.021444</td>
          <td>22.359534</td>
          <td>0.006135</td>
          <td>21.489193</td>
          <td>0.005210</td>
          <td>20.729761</td>
          <td>0.005151</td>
          <td>20.490030</td>
          <td>0.005334</td>
          <td>20.287871</td>
          <td>0.006056</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0</td>
          <td>0.275019</td>
          <td>27.072953</td>
          <td>0.578208</td>
          <td>27.961646</td>
          <td>0.455798</td>
          <td>26.862689</td>
          <td>0.166976</td>
          <td>26.894715</td>
          <td>0.272169</td>
          <td>26.887629</td>
          <td>0.479795</td>
          <td>NaN</td>
          <td>NaN</td>
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
          <td>0.499235</td>
          <td>27.149999</td>
          <td>0.610654</td>
          <td>27.340766</td>
          <td>0.280362</td>
          <td>25.738337</td>
          <td>0.062548</td>
          <td>25.074234</td>
          <td>0.056703</td>
          <td>24.745917</td>
          <td>0.081139</td>
          <td>24.890352</td>
          <td>0.203409</td>
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
          <td>0.569906</td>
          <td>29.198308</td>
          <td>1.941090</td>
          <td>26.983756</td>
          <td>0.208913</td>
          <td>25.857585</td>
          <td>0.069518</td>
          <td>25.425932</td>
          <td>0.077431</td>
          <td>25.000691</td>
          <td>0.101506</td>
          <td>25.682421</td>
          <td>0.386107</td>
        </tr>
        <tr>
          <th>596</th>
          <td>0</td>
          <td>0.912176</td>
          <td>25.361537</td>
          <td>0.147941</td>
          <td>25.543017</td>
          <td>0.059864</td>
          <td>25.313284</td>
          <td>0.042890</td>
          <td>25.283270</td>
          <td>0.068252</td>
          <td>25.294399</td>
          <td>0.131084</td>
          <td>24.966467</td>
          <td>0.216777</td>
        </tr>
        <tr>
          <th>597</th>
          <td>1</td>
          <td>0.366884</td>
          <td>25.424068</td>
          <td>0.156065</td>
          <td>24.547564</td>
          <td>0.024935</td>
          <td>23.696877</td>
          <td>0.011096</td>
          <td>23.479646</td>
          <td>0.014310</td>
          <td>23.238418</td>
          <td>0.021510</td>
          <td>23.213067</td>
          <td>0.047109</td>
        </tr>
        <tr>
          <th>598</th>
          <td>0</td>
          <td>0.313108</td>
          <td>29.073612</td>
          <td>1.838296</td>
          <td>27.301175</td>
          <td>0.271489</td>
          <td>26.387450</td>
          <td>0.110799</td>
          <td>26.691282</td>
          <td>0.230277</td>
          <td>26.346142</td>
          <td>0.315782</td>
          <td>25.976749</td>
          <td>0.482789</td>
        </tr>
        <tr>
          <th>599</th>
          <td>0</td>
          <td>0.642166</td>
          <td>27.747188</td>
          <td>0.908112</td>
          <td>26.091389</td>
          <td>0.097082</td>
          <td>25.928403</td>
          <td>0.074013</td>
          <td>25.612108</td>
          <td>0.091237</td>
          <td>25.832657</td>
          <td>0.207377</td>
          <td>25.967632</td>
          <td>0.479527</td>
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
          <td>0.648564</td>
          <td>23.139859</td>
          <td>0.021444</td>
          <td>22.359534</td>
          <td>0.006135</td>
          <td>21.489193</td>
          <td>0.005210</td>
          <td>20.729761</td>
          <td>0.005151</td>
          <td>20.490030</td>
          <td>0.005334</td>
          <td>20.287871</td>
          <td>0.006056</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.341317</td>
          <td>23.995016</td>
          <td>0.044831</td>
          <td>23.835708</td>
          <td>0.013835</td>
          <td>23.407375</td>
          <td>0.009130</td>
          <td>23.394428</td>
          <td>0.013385</td>
          <td>23.005858</td>
          <td>0.017678</td>
          <td>23.288929</td>
          <td>0.050391</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.491075</td>
          <td>27.684474</td>
          <td>0.873034</td>
          <td>26.037720</td>
          <td>0.092619</td>
          <td>25.055265</td>
          <td>0.034133</td>
          <td>24.608385</td>
          <td>0.037506</td>
          <td>24.438054</td>
          <td>0.061793</td>
          <td>24.059722</td>
          <td>0.099595</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.196691</td>
          <td>24.949134</td>
          <td>0.103585</td>
          <td>24.003613</td>
          <td>0.015802</td>
          <td>23.307856</td>
          <td>0.008594</td>
          <td>22.958861</td>
          <td>0.009767</td>
          <td>22.700154</td>
          <td>0.013805</td>
          <td>22.604590</td>
          <td>0.027534</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.324315</td>
          <td>27.041117</td>
          <td>0.565188</td>
          <td>25.761018</td>
          <td>0.072593</td>
          <td>25.057992</td>
          <td>0.034215</td>
          <td>23.955027</td>
          <td>0.021202</td>
          <td>23.350773</td>
          <td>0.023690</td>
          <td>22.616143</td>
          <td>0.027814</td>
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
          <td>0.480807</td>
          <td>24.781846</td>
          <td>0.089509</td>
          <td>24.253105</td>
          <td>0.019405</td>
          <td>23.533419</td>
          <td>0.009907</td>
          <td>23.183154</td>
          <td>0.011417</td>
          <td>23.190153</td>
          <td>0.020643</td>
          <td>22.976959</td>
          <td>0.038212</td>
        </tr>
        <tr>
          <th>227</th>
          <td>0.350717</td>
          <td>25.287372</td>
          <td>0.138820</td>
          <td>23.579408</td>
          <td>0.011418</td>
          <td>22.180189</td>
          <td>0.005636</td>
          <td>21.592349</td>
          <td>0.005600</td>
          <td>21.207141</td>
          <td>0.006056</td>
          <td>21.034529</td>
          <td>0.008273</td>
        </tr>
        <tr>
          <th>228</th>
          <td>0.617275</td>
          <td>27.365007</td>
          <td>0.708321</td>
          <td>25.133651</td>
          <td>0.041664</td>
          <td>23.654719</td>
          <td>0.010768</td>
          <td>22.553688</td>
          <td>0.007690</td>
          <td>22.128707</td>
          <td>0.009182</td>
          <td>21.888231</td>
          <td>0.015033</td>
        </tr>
        <tr>
          <th>229</th>
          <td>1.072344</td>
          <td>24.622990</td>
          <td>0.077876</td>
          <td>24.265253</td>
          <td>0.019603</td>
          <td>23.689445</td>
          <td>0.011037</td>
          <td>23.159791</td>
          <td>0.011225</td>
          <td>22.509084</td>
          <td>0.011932</td>
          <td>22.381054</td>
          <td>0.022684</td>
        </tr>
        <tr>
          <th>230</th>
          <td>0.366884</td>
          <td>25.424068</td>
          <td>0.156065</td>
          <td>24.547564</td>
          <td>0.024935</td>
          <td>23.696877</td>
          <td>0.011096</td>
          <td>23.479646</td>
          <td>0.014310</td>
          <td>23.238418</td>
          <td>0.021510</td>
          <td>23.213067</td>
          <td>0.047109</td>
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
          <td>0.648564</td>
          <td>23.150772</td>
          <td>22.366415</td>
          <td>21.491493</td>
          <td>20.731236</td>
          <td>20.491596</td>
          <td>20.287109</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.341317</td>
          <td>24.068222</td>
          <td>23.826418</td>
          <td>23.405294</td>
          <td>23.416424</td>
          <td>22.987785</td>
          <td>23.289028</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.491075</td>
          <td>27.021898</td>
          <td>26.095686</td>
          <td>25.032051</td>
          <td>24.618488</td>
          <td>24.387232</td>
          <td>24.167744</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.196691</td>
          <td>24.893013</td>
          <td>23.983337</td>
          <td>23.310110</td>
          <td>22.938263</td>
          <td>22.722412</td>
          <td>22.617384</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.324315</td>
          <td>27.208641</td>
          <td>25.888857</td>
          <td>25.021338</td>
          <td>23.973490</td>
          <td>23.354897</td>
          <td>22.628229</td>
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
          <td>0.480807</td>
          <td>24.777891</td>
          <td>24.257607</td>
          <td>23.544462</td>
          <td>23.205730</td>
          <td>23.162960</td>
          <td>22.997458</td>
        </tr>
        <tr>
          <th>227</th>
          <td>0.350717</td>
          <td>25.233673</td>
          <td>23.563206</td>
          <td>22.176367</td>
          <td>21.583717</td>
          <td>21.211185</td>
          <td>21.033613</td>
        </tr>
        <tr>
          <th>228</th>
          <td>0.617275</td>
          <td>27.384861</td>
          <td>25.180693</td>
          <td>23.666212</td>
          <td>22.560936</td>
          <td>22.159149</td>
          <td>21.869398</td>
        </tr>
        <tr>
          <th>229</th>
          <td>1.072344</td>
          <td>24.572737</td>
          <td>24.226124</td>
          <td>23.692007</td>
          <td>23.160368</td>
          <td>22.515059</td>
          <td>22.385687</td>
        </tr>
        <tr>
          <th>230</th>
          <td>0.366884</td>
          <td>25.564947</td>
          <td>24.567034</td>
          <td>23.710983</td>
          <td>23.469208</td>
          <td>23.265720</td>
          <td>23.191113</td>
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

    
    
    
    best fit values are sigma=0.075 and numneigh=7
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer
    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 388 training and 130 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.075 and numneigh=7
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer
    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 190 training and 63 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.05333333333333334 and numneigh=3
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer


.. code:: ipython3

    # let's see what the output looks like 
    knn_models["lsst_error_model"]




.. parsed-literal::

    {'model': {'kdtree': <sklearn.neighbors._kd_tree.KDTree at 0x5562220186b0>,
      'bestsig': np.float64(0.075),
      'nneigh': 7,
      'truezs': array([1.4575667 , 1.3254019 , 0.8938675 , 1.7871764 , 1.3022082 ,
             1.2871565 , 1.1447735 , 0.12723911, 1.8323886 , 0.13669944,
             0.5284236 , 0.27324033, 0.4471414 , 0.82454646, 1.8187356 ,
             0.51238316, 0.8822826 , 0.60520804, 0.7676796 , 0.45275044,
             0.9548153 , 0.9622477 , 0.5167731 , 2.058155  , 0.99046284,
             1.4791145 , 1.6850917 , 0.5816809 , 0.31295252, 1.3985872 ,
             0.5568371 , 0.6027714 , 0.97304213, 1.3225304 , 0.74346644,
             1.6440561 , 2.1583824 , 0.72943944, 1.2203817 , 0.652326  ,
             1.2514406 , 0.28268826, 0.94577205, 0.723452  , 2.0185044 ,
             0.19236302, 1.3841392 , 0.752143  , 1.2254468 , 1.2517804 ,
             0.56882685, 1.5172755 , 1.8563595 , 0.29535747, 1.5213935 ,
             1.5474918 , 0.4186144 , 1.0833172 , 0.5815803 , 2.1780775 ,
             1.0250716 , 1.3788006 , 1.8038442 , 0.57718736, 0.51131564,
             0.5176462 , 1.7182266 , 0.6126933 , 1.6489189 , 1.1260153 ,
             0.7447854 , 2.0196667 , 1.687067  , 1.8242302 , 0.92453974,
             0.08758879, 0.46544194, 1.3557566 , 1.1979079 , 0.37931514,
             1.2566574 , 1.308829  , 0.9502593 , 0.65654165, 0.9082115 ,
             0.89423275, 1.464293  , 0.6992428 , 1.3241165 , 0.38057172,
             1.2135875 , 0.7546936 , 2.1444125 , 1.3916032 , 0.7782356 ,
             0.31368136, 2.1346319 , 1.3375509 , 0.273921  , 0.3500061 ,
             0.30414927, 2.203555  , 1.027878  , 0.47388995, 1.2794236 ,
             0.98023736, 1.4020702 , 1.3251982 , 1.2835758 , 2.2032716 ,
             1.6549317 , 1.0038908 , 1.2294555 , 0.7139947 , 0.82936347,
             0.7357193 , 0.9528853 , 1.2270715 , 1.1164203 , 1.0199636 ,
             1.104983  , 1.6740468 , 0.47896004, 0.6616947 , 0.31951833,
             0.19057477, 2.0415504 , 0.19163287, 0.31050837, 0.44918132,
             0.8313323 , 0.43797266, 1.2257323 , 2.1874743 , 0.6854433 ,
             0.9061391 , 0.17675316, 0.51041174, 2.1499166 , 1.7623742 ,
             0.25641   , 2.053147  , 2.0700443 , 1.4082597 , 0.47210896,
             0.8639988 , 2.1309052 , 0.67155504, 0.5714797 , 1.5322901 ,
             0.1458056 , 1.7100867 , 1.6858023 , 1.3852688 , 0.2649294 ,
             0.2639078 , 0.7322086 , 2.0209517 , 0.33337343, 0.4937464 ,
             2.2139778 , 0.8745914 , 0.72961587, 2.1336741 , 0.21930182,
             1.4785273 , 0.3310789 , 0.4109348 , 1.5498259 , 1.8007388 ,
             1.8720315 , 2.2577333 , 0.6700262 , 1.2111015 , 1.8562785 ,
             0.9083506 , 0.29047775, 1.0140748 , 1.0111526 , 1.8541086 ,
             0.7483253 , 1.3566431 , 1.4793339 , 1.4667773 , 1.0862784 ,
             0.23729324, 0.5662998 , 2.1948264 , 0.5153098 , 1.1620181 ,
             0.7217192 , 1.6348987 , 1.7578868 , 2.2113328 , 1.185142  ,
             0.66059667, 0.7399084 , 0.8994029 , 0.17477322, 1.0126058 ,
             1.0211793 , 1.2585114 , 2.0413594 , 0.8658266 , 1.1437713 ,
             0.54520833, 0.6949835 , 0.2046473 , 1.4243658 , 0.36589777,
             0.65094453, 2.1635377 , 0.34921026, 0.3559624 , 0.06207943,
             1.0691171 , 0.95071113, 1.1032231 , 0.19231594, 0.19316709,
             1.0684927 , 1.7314166 , 0.64692485, 1.5400673 , 0.8265726 ,
             1.5882561 , 1.0415562 , 1.7893887 , 2.2085586 , 0.3265648 ,
             0.49142575, 1.9761553 , 2.1890483 , 0.4804567 , 1.3557078 ,
             0.17260182, 0.34545076, 2.2364736 , 0.70189863, 1.8659621 ,
             1.2281982 , 0.17718768, 1.1142792 , 0.2900548 , 1.4359069 ,
             0.5518607 , 1.3820024 , 1.0249273 , 0.17632508, 0.6297785 ,
             1.8313813 , 0.6010807 , 2.2544782 , 0.8551384 , 0.27807105,
             1.200532  , 1.0281208 , 2.2649999 , 0.43679428, 0.35992527,
             0.7661513 , 1.9824528 , 0.29422593, 2.2103033 , 0.7839007 ,
             1.9174744 , 1.8278486 , 0.4401505 , 0.6401528 , 0.9407756 ,
             1.5475633 , 1.7343954 , 0.37179923, 1.3946618 , 0.7471809 ,
             2.2655299 , 0.554232  , 0.35659623, 1.381664  , 1.7699106 ,
             0.60914344, 0.42628634, 0.7627975 , 0.7419727 , 2.088419  ,
             0.37222052, 0.97797817, 0.43334973, 0.34014833, 0.4547485 ,
             0.5360031 , 0.49932325, 1.206409  , 1.1535699 , 0.15389156,
             0.5777842 , 0.72689766, 0.26679778, 1.9508541 , 0.39875436,
             0.8295778 , 0.993333  , 1.2747929 , 0.32616198, 0.94644463,
             0.281296  , 0.7269047 , 0.5130308 , 0.27702117, 0.58606493,
             0.5399356 , 0.34621716, 0.5576269 , 0.3203405 , 0.22125661,
             0.5805454 , 0.23907995, 1.0953357 , 0.29229963, 1.491581  ,
             2.1682792 , 1.724986  , 0.76470405, 1.1354841 , 1.2281108 ,
             0.36220884, 0.66774464, 0.18750942, 1.3721446 , 0.35170412,
             1.5298799 , 1.0906045 , 1.3567997 , 1.0865414 , 1.2564151 ,
             0.91625017, 0.6815893 , 2.1057582 , 1.39133   , 1.1901782 ,
             0.33725595, 1.4671648 , 0.97342956, 0.32748342, 0.29285145,
             0.8617269 , 0.32840693, 0.38082767, 0.2558304 , 0.28015506,
             0.2851802 , 1.6437423 , 1.5495169 , 0.6993024 , 0.3113973 ,
             0.2945242 , 1.9896207 , 2.2355928 , 1.4535533 , 1.0377462 ,
             0.57291067, 1.1146835 , 0.44586074, 0.31297922, 0.24485064,
             0.8324529 , 0.16677165, 2.2044773 , 1.9613445 , 0.5684309 ,
             0.640288  , 0.21630156, 0.4349811 , 0.40336752, 1.2391386 ,
             0.3598318 , 1.4698677 , 0.31637847, 0.46535468, 1.53786   ,
             2.0877047 , 0.34686935, 1.1987302 , 0.98761255, 0.5196041 ,
             0.4861629 , 1.4217974 , 1.4955932 , 1.0676541 , 1.3316078 ,
             1.3643768 , 0.9488969 , 0.16774845, 1.0244675 , 0.8440034 ,
             0.86729765, 0.2003715 , 0.7023953 , 0.26743448, 0.2787304 ,
             0.62572163, 0.74388325, 1.2570525 , 0.2567829 , 2.1638703 ,
             0.6205842 , 1.7355294 , 1.5650795 , 0.46541715, 0.3220868 ,
             1.5719006 , 0.57792366, 0.3072163 , 0.42273712, 0.8419348 ,
             0.88949144, 1.6528194 , 0.15954518, 1.6330723 , 0.7780057 ,
             0.26169693, 1.6551992 , 1.5724477 , 0.3380767 , 0.21921611,
             0.6138441 , 0.14574707, 1.154194  , 0.52505016, 0.20260882,
             0.8913227 , 0.5341736 , 0.28514588, 0.59255743, 0.31618166,
             1.2586774 , 0.73114645, 1.5118434 , 0.648918  , 0.43458962,
             0.9386279 , 1.5234374 , 0.3221773 , 0.33446097, 1.7691178 ,
             0.4477265 , 2.1450996 , 1.021512  , 0.3726771 , 0.30175018,
             0.46907663, 0.30592418, 1.1224805 , 0.29202497, 1.43495   ,
             2.0833702 , 0.73687506, 0.5897093 , 2.1081824 , 0.60214007,
             0.17416728, 0.11336923, 1.8178264 , 1.7733105 , 0.3455541 ,
             1.8338467 , 0.56675404, 1.2686193 , 0.32640696, 0.56540984,
             0.38020515, 1.324671  , 0.5739554 , 1.3083127 , 0.9130395 ,
             0.99835724, 0.33403754, 0.9141597 , 0.27690053, 0.14257753,
             1.0399185 , 1.1823943 , 1.2235447 , 1.4557323 , 1.2509599 ,
             0.9438342 , 1.6004602 , 1.8238353 , 0.7655238 , 1.4813395 ,
             1.6609927 , 2.020829  , 0.6104919 , 0.54624903, 1.7785156 ,
             1.5719788 , 0.26386952, 1.214011  , 0.8009981 , 0.5171038 ,
             0.79824793, 2.0033674 , 0.28447795, 0.5842887 , 0.52964056,
             0.49456775, 0.48134172, 2.0258524 , 0.8102676 , 0.24379098,
             0.50191575, 0.32375944, 2.0361476 , 0.4259448 , 1.5501468 ,
             0.5609606 , 1.3186553 , 0.6481654 , 1.4727288 , 0.52530813,
             0.30161166, 1.8054986 , 0.9689349 , 0.28358138, 1.5862015 ,
             1.2461551 , 1.0939054 , 1.5959108 , 1.1531341 , 0.23085725,
             0.74615943, 0.22021604, 0.92097086, 1.9158221 , 1.8485302 ,
             1.2094257 , 0.54062045, 0.76542664, 0.19027257, 0.58722305,
             1.7474982 , 1.5219986 , 0.9435741 , 0.34590602, 2.1108525 ,
             0.4989978 , 0.28908253, 0.91092587, 0.29988897, 0.589214  ,
             0.29844642, 0.5673281 , 1.3654486 , 1.8625295 , 1.4441293 ,
             0.35949934, 0.28131962, 0.5466361 , 1.6434426 , 2.229297  ,
             0.27927268, 1.2960224 , 1.8249247 , 0.6589263 , 1.3768171 ,
             0.20149231, 1.2584286 , 0.56529313, 1.4166747 , 1.2047888 ,
             0.80600727, 2.1160157 , 1.2633454 , 1.8177421 , 0.76136184,
             1.6857423 , 1.0009352 , 1.7527928 , 0.36117136, 0.86685115,
             0.6599654 , 1.0534489 , 1.4405056 , 0.3983264 , 1.1197848 ,
             0.689267  , 1.9299375 , 0.76282245, 0.419829  , 1.1727688 ,
             1.0236223 , 2.2424622 , 1.5145707 , 0.37699008, 0.41640508,
             0.68884885, 0.34693825, 1.254465  , 0.7958047 , 1.5862162 ],
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
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x5562220186b0>, 'bestsig': np.float64(0.075), 'nneigh': 7, 'truezs': array([1.4575667 , 1.3254019 , 0.8938675 , 1.7871764 , 1.3022082 ,
           1.2871565 , 1.1447735 , 0.12723911, 1.8323886 , 0.13669944,
           0.5284236 , 0.27324033, 0.4471414 , 0.82454646, 1.8187356 ,
           0.51238316, 0.8822826 , 0.60520804, 0.7676796 , 0.45275044,
           0.9548153 , 0.9622477 , 0.5167731 , 2.058155  , 0.99046284,
           1.4791145 , 1.6850917 , 0.5816809 , 0.31295252, 1.3985872 ,
           0.5568371 , 0.6027714 , 0.97304213, 1.3225304 , 0.74346644,
           1.6440561 , 2.1583824 , 0.72943944, 1.2203817 , 0.652326  ,
           1.2514406 , 0.28268826, 0.94577205, 0.723452  , 2.0185044 ,
           0.19236302, 1.3841392 , 0.752143  , 1.2254468 , 1.2517804 ,
           0.56882685, 1.5172755 , 1.8563595 , 0.29535747, 1.5213935 ,
           1.5474918 , 0.4186144 , 1.0833172 , 0.5815803 , 2.1780775 ,
           1.0250716 , 1.3788006 , 1.8038442 , 0.57718736, 0.51131564,
           0.5176462 , 1.7182266 , 0.6126933 , 1.6489189 , 1.1260153 ,
           0.7447854 , 2.0196667 , 1.687067  , 1.8242302 , 0.92453974,
           0.08758879, 0.46544194, 1.3557566 , 1.1979079 , 0.37931514,
           1.2566574 , 1.308829  , 0.9502593 , 0.65654165, 0.9082115 ,
           0.89423275, 1.464293  , 0.6992428 , 1.3241165 , 0.38057172,
           1.2135875 , 0.7546936 , 2.1444125 , 1.3916032 , 0.7782356 ,
           0.31368136, 2.1346319 , 1.3375509 , 0.273921  , 0.3500061 ,
           0.30414927, 2.203555  , 1.027878  , 0.47388995, 1.2794236 ,
           0.98023736, 1.4020702 , 1.3251982 , 1.2835758 , 2.2032716 ,
           1.6549317 , 1.0038908 , 1.2294555 , 0.7139947 , 0.82936347,
           0.7357193 , 0.9528853 , 1.2270715 , 1.1164203 , 1.0199636 ,
           1.104983  , 1.6740468 , 0.47896004, 0.6616947 , 0.31951833,
           0.19057477, 2.0415504 , 0.19163287, 0.31050837, 0.44918132,
           0.8313323 , 0.43797266, 1.2257323 , 2.1874743 , 0.6854433 ,
           0.9061391 , 0.17675316, 0.51041174, 2.1499166 , 1.7623742 ,
           0.25641   , 2.053147  , 2.0700443 , 1.4082597 , 0.47210896,
           0.8639988 , 2.1309052 , 0.67155504, 0.5714797 , 1.5322901 ,
           0.1458056 , 1.7100867 , 1.6858023 , 1.3852688 , 0.2649294 ,
           0.2639078 , 0.7322086 , 2.0209517 , 0.33337343, 0.4937464 ,
           2.2139778 , 0.8745914 , 0.72961587, 2.1336741 , 0.21930182,
           1.4785273 , 0.3310789 , 0.4109348 , 1.5498259 , 1.8007388 ,
           1.8720315 , 2.2577333 , 0.6700262 , 1.2111015 , 1.8562785 ,
           0.9083506 , 0.29047775, 1.0140748 , 1.0111526 , 1.8541086 ,
           0.7483253 , 1.3566431 , 1.4793339 , 1.4667773 , 1.0862784 ,
           0.23729324, 0.5662998 , 2.1948264 , 0.5153098 , 1.1620181 ,
           0.7217192 , 1.6348987 , 1.7578868 , 2.2113328 , 1.185142  ,
           0.66059667, 0.7399084 , 0.8994029 , 0.17477322, 1.0126058 ,
           1.0211793 , 1.2585114 , 2.0413594 , 0.8658266 , 1.1437713 ,
           0.54520833, 0.6949835 , 0.2046473 , 1.4243658 , 0.36589777,
           0.65094453, 2.1635377 , 0.34921026, 0.3559624 , 0.06207943,
           1.0691171 , 0.95071113, 1.1032231 , 0.19231594, 0.19316709,
           1.0684927 , 1.7314166 , 0.64692485, 1.5400673 , 0.8265726 ,
           1.5882561 , 1.0415562 , 1.7893887 , 2.2085586 , 0.3265648 ,
           0.49142575, 1.9761553 , 2.1890483 , 0.4804567 , 1.3557078 ,
           0.17260182, 0.34545076, 2.2364736 , 0.70189863, 1.8659621 ,
           1.2281982 , 0.17718768, 1.1142792 , 0.2900548 , 1.4359069 ,
           0.5518607 , 1.3820024 , 1.0249273 , 0.17632508, 0.6297785 ,
           1.8313813 , 0.6010807 , 2.2544782 , 0.8551384 , 0.27807105,
           1.200532  , 1.0281208 , 2.2649999 , 0.43679428, 0.35992527,
           0.7661513 , 1.9824528 , 0.29422593, 2.2103033 , 0.7839007 ,
           1.9174744 , 1.8278486 , 0.4401505 , 0.6401528 , 0.9407756 ,
           1.5475633 , 1.7343954 , 0.37179923, 1.3946618 , 0.7471809 ,
           2.2655299 , 0.554232  , 0.35659623, 1.381664  , 1.7699106 ,
           0.60914344, 0.42628634, 0.7627975 , 0.7419727 , 2.088419  ,
           0.37222052, 0.97797817, 0.43334973, 0.34014833, 0.4547485 ,
           0.5360031 , 0.49932325, 1.206409  , 1.1535699 , 0.15389156,
           0.5777842 , 0.72689766, 0.26679778, 1.9508541 , 0.39875436,
           0.8295778 , 0.993333  , 1.2747929 , 0.32616198, 0.94644463,
           0.281296  , 0.7269047 , 0.5130308 , 0.27702117, 0.58606493,
           0.5399356 , 0.34621716, 0.5576269 , 0.3203405 , 0.22125661,
           0.5805454 , 0.23907995, 1.0953357 , 0.29229963, 1.491581  ,
           2.1682792 , 1.724986  , 0.76470405, 1.1354841 , 1.2281108 ,
           0.36220884, 0.66774464, 0.18750942, 1.3721446 , 0.35170412,
           1.5298799 , 1.0906045 , 1.3567997 , 1.0865414 , 1.2564151 ,
           0.91625017, 0.6815893 , 2.1057582 , 1.39133   , 1.1901782 ,
           0.33725595, 1.4671648 , 0.97342956, 0.32748342, 0.29285145,
           0.8617269 , 0.32840693, 0.38082767, 0.2558304 , 0.28015506,
           0.2851802 , 1.6437423 , 1.5495169 , 0.6993024 , 0.3113973 ,
           0.2945242 , 1.9896207 , 2.2355928 , 1.4535533 , 1.0377462 ,
           0.57291067, 1.1146835 , 0.44586074, 0.31297922, 0.24485064,
           0.8324529 , 0.16677165, 2.2044773 , 1.9613445 , 0.5684309 ,
           0.640288  , 0.21630156, 0.4349811 , 0.40336752, 1.2391386 ,
           0.3598318 , 1.4698677 , 0.31637847, 0.46535468, 1.53786   ,
           2.0877047 , 0.34686935, 1.1987302 , 0.98761255, 0.5196041 ,
           0.4861629 , 1.4217974 , 1.4955932 , 1.0676541 , 1.3316078 ,
           1.3643768 , 0.9488969 , 0.16774845, 1.0244675 , 0.8440034 ,
           0.86729765, 0.2003715 , 0.7023953 , 0.26743448, 0.2787304 ,
           0.62572163, 0.74388325, 1.2570525 , 0.2567829 , 2.1638703 ,
           0.6205842 , 1.7355294 , 1.5650795 , 0.46541715, 0.3220868 ,
           1.5719006 , 0.57792366, 0.3072163 , 0.42273712, 0.8419348 ,
           0.88949144, 1.6528194 , 0.15954518, 1.6330723 , 0.7780057 ,
           0.26169693, 1.6551992 , 1.5724477 , 0.3380767 , 0.21921611,
           0.6138441 , 0.14574707, 1.154194  , 0.52505016, 0.20260882,
           0.8913227 , 0.5341736 , 0.28514588, 0.59255743, 0.31618166,
           1.2586774 , 0.73114645, 1.5118434 , 0.648918  , 0.43458962,
           0.9386279 , 1.5234374 , 0.3221773 , 0.33446097, 1.7691178 ,
           0.4477265 , 2.1450996 , 1.021512  , 0.3726771 , 0.30175018,
           0.46907663, 0.30592418, 1.1224805 , 0.29202497, 1.43495   ,
           2.0833702 , 0.73687506, 0.5897093 , 2.1081824 , 0.60214007,
           0.17416728, 0.11336923, 1.8178264 , 1.7733105 , 0.3455541 ,
           1.8338467 , 0.56675404, 1.2686193 , 0.32640696, 0.56540984,
           0.38020515, 1.324671  , 0.5739554 , 1.3083127 , 0.9130395 ,
           0.99835724, 0.33403754, 0.9141597 , 0.27690053, 0.14257753,
           1.0399185 , 1.1823943 , 1.2235447 , 1.4557323 , 1.2509599 ,
           0.9438342 , 1.6004602 , 1.8238353 , 0.7655238 , 1.4813395 ,
           1.6609927 , 2.020829  , 0.6104919 , 0.54624903, 1.7785156 ,
           1.5719788 , 0.26386952, 1.214011  , 0.8009981 , 0.5171038 ,
           0.79824793, 2.0033674 , 0.28447795, 0.5842887 , 0.52964056,
           0.49456775, 0.48134172, 2.0258524 , 0.8102676 , 0.24379098,
           0.50191575, 0.32375944, 2.0361476 , 0.4259448 , 1.5501468 ,
           0.5609606 , 1.3186553 , 0.6481654 , 1.4727288 , 0.52530813,
           0.30161166, 1.8054986 , 0.9689349 , 0.28358138, 1.5862015 ,
           1.2461551 , 1.0939054 , 1.5959108 , 1.1531341 , 0.23085725,
           0.74615943, 0.22021604, 0.92097086, 1.9158221 , 1.8485302 ,
           1.2094257 , 0.54062045, 0.76542664, 0.19027257, 0.58722305,
           1.7474982 , 1.5219986 , 0.9435741 , 0.34590602, 2.1108525 ,
           0.4989978 , 0.28908253, 0.91092587, 0.29988897, 0.589214  ,
           0.29844642, 0.5673281 , 1.3654486 , 1.8625295 , 1.4441293 ,
           0.35949934, 0.28131962, 0.5466361 , 1.6434426 , 2.229297  ,
           0.27927268, 1.2960224 , 1.8249247 , 0.6589263 , 1.3768171 ,
           0.20149231, 1.2584286 , 0.56529313, 1.4166747 , 1.2047888 ,
           0.80600727, 2.1160157 , 1.2633454 , 1.8177421 , 0.76136184,
           1.6857423 , 1.0009352 , 1.7527928 , 0.36117136, 0.86685115,
           0.6599654 , 1.0534489 , 1.4405056 , 0.3983264 , 1.1197848 ,
           0.689267  , 1.9299375 , 0.76282245, 0.419829  , 1.1727688 ,
           1.0236223 , 2.2424622 , 1.5145707 , 0.37699008, 0.41640508,
           0.68884885, 0.34693825, 1.254465  , 0.7958047 , 1.5862162 ],
          dtype=float32), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 231
    Process 0 estimating PZ PDF for rows 0 - 231
    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x55620b608710>, 'bestsig': np.float64(0.075), 'nneigh': 7, 'truezs': array([0.7509079 , 0.64856386, 0.27501893, 1.7270784 , 0.49923456,
           0.34131694, 0.49107456, 0.19669104, 0.68063   , 1.3243154 ,
           1.5690708 , 0.94031143, 1.3222268 , 1.8482414 , 0.10947442,
           0.44441795, 0.5719139 , 0.2320149 , 0.7590655 , 0.50075614,
           1.1899778 , 0.6688264 , 1.1402568 , 2.034984  , 1.2302194 ,
           1.8602103 , 0.7940375 , 0.49264932, 2.163286  , 1.2482784 ,
           0.65717155, 1.1286031 , 0.3655461 , 1.2945307 , 1.0124041 ,
           1.2655163 , 0.7799204 , 0.72486293, 1.6730183 , 1.2040765 ,
           0.29499555, 0.8598248 , 0.5811117 , 1.0079627 , 0.7839023 ,
           1.5040898 , 0.19833839, 0.3641243 , 1.0699039 , 2.0049148 ,
           1.0512375 , 1.0130191 , 0.977039  , 1.6852964 , 0.5351938 ,
           0.87444913, 0.6633852 , 1.1152002 , 0.27860236, 0.27408767,
           1.2258346 , 0.22099543, 0.37300777, 0.69973516, 0.5051955 ,
           0.40547454, 1.586558  , 1.1978953 , 0.8481954 , 1.3823315 ,
           0.7186409 , 1.499964  , 1.8874849 , 1.8998837 , 0.73322594,
           2.0316916 , 0.86967075, 0.3517443 , 1.8481865 , 0.8677667 ,
           1.2646464 , 1.1628019 , 0.71207875, 0.63848305, 0.49140382,
           1.3518044 , 1.1720487 , 0.25023675, 2.0395994 , 1.5040977 ,
           1.3844157 , 1.7504225 , 0.3208294 , 0.7749364 , 0.2707306 ,
           0.34326088, 1.3048459 , 1.1196127 , 0.32044792, 0.48277915,
           0.8804904 , 1.9946897 , 1.4457994 , 1.8159542 , 1.3982546 ,
           0.5782739 , 0.4247055 , 0.30203664, 1.0419514 , 0.6375356 ,
           0.9837023 , 0.31565166, 0.6038319 , 1.2770414 , 1.607333  ,
           1.1012123 , 0.2862991 , 0.91292155, 1.9246125 , 0.2050327 ,
           0.21128106, 0.60300714, 0.92902094, 1.6759135 , 0.6721448 ,
           1.8983424 , 1.573741  , 0.47639036, 1.5046899 , 0.60835093,
           2.073361  , 0.47884107, 0.4525504 , 0.935526  , 1.249265  ,
           0.7246314 , 1.0954314 , 1.5557518 , 0.04437506, 2.2233949 ,
           0.19662118, 2.2380438 , 1.3238128 , 1.3808628 , 0.35275698,
           1.7665598 , 1.3939935 , 0.8008949 , 0.692803  , 1.6233435 ,
           0.5196067 , 1.2036526 , 1.1200435 , 0.09098494, 1.1269304 ,
           0.1731242 , 1.0081686 , 1.6923593 , 1.737859  , 0.1415689 ,
           1.4324336 , 1.6301638 , 0.41800284, 1.8247597 , 0.3597448 ,
           1.6799692 , 1.2519487 , 1.388462  , 0.8341284 , 1.2356384 ,
           0.70903623, 0.92613685, 1.2624303 , 2.052319  , 0.42295802,
           0.47042882, 1.0931101 , 2.2369432 , 0.40498495, 2.0274017 ,
           1.1171969 , 0.6855167 , 1.2497094 , 0.52586997, 0.32986665,
           0.7100953 , 1.2352313 , 0.6265608 , 2.1519914 , 1.3553787 ,
           1.3960954 , 1.3478462 , 1.8974652 , 0.8129996 , 1.1918653 ,
           0.80644786, 0.9694023 , 0.6688755 , 0.8952673 , 1.544577  ,
           0.35564065, 1.0472128 , 0.91116446, 1.571813  , 1.6761749 ,
           0.53566855, 2.0151792 , 1.5396993 , 0.31581545, 1.81194   ,
           0.880221  , 1.060499  , 0.8301916 , 0.4817537 , 0.17633653,
           0.24423599, 0.37152028, 0.4538076 , 0.6046192 , 2.1038504 ,
           1.0719593 , 0.31062198, 0.6389825 , 0.28426862, 0.3448646 ,
           0.24209881, 0.3408916 , 0.48278522, 1.6716807 , 0.24768281,
           1.7123325 , 1.2886477 , 0.8215641 , 0.22030282, 0.3779739 ,
           1.4148338 , 0.47465324, 0.6471168 , 1.0979984 , 0.41957152,
           0.06494439, 0.24611795, 0.6569177 , 1.7181219 , 0.69799757,
           2.012556  , 0.6651492 , 0.3496039 , 1.1342088 , 1.3061376 ,
           0.19477487, 0.8838167 , 1.6716948 , 1.6599975 , 0.3905996 ,
           1.1406928 , 0.33647263, 0.6972396 , 0.9603098 , 0.90957177,
           0.86597884, 0.83918285, 0.3891691 , 0.42035353, 0.32300937,
           1.3670679 , 1.0882114 , 0.1932075 , 0.28778052, 1.0775149 ,
           1.7742798 , 1.3634543 , 0.14922333, 0.8318142 , 1.3320235 ,
           0.18061554, 1.4581984 , 1.6579394 , 0.8504089 , 0.7093656 ,
           2.2185655 , 0.72250575, 0.9340246 , 1.207407  , 0.33605266,
           0.40376544, 0.3367417 , 1.7527099 , 0.18367052, 0.6362902 ,
           1.3291239 , 1.3824226 , 1.4546529 , 0.14693093, 0.8607303 ,
           0.39750457, 2.1248128 , 2.2001376 , 0.4735781 , 1.0962887 ,
           1.6864796 , 0.7008854 , 0.54542744, 0.2259171 , 2.1036599 ,
           0.7160775 , 0.5938034 , 0.85677296, 1.8499906 , 1.1863334 ,
           0.24228597, 1.0443172 , 1.0325204 , 0.2916491 , 0.86447537,
           1.6458129 , 1.441298  , 0.29436457, 1.0167453 , 2.1428852 ,
           1.721051  , 1.6346577 , 1.5554229 , 0.6494462 , 0.74497575,
           0.8567695 , 2.15687   , 0.59277236, 0.0743897 , 2.0148222 ,
           0.9956508 , 1.5869482 , 0.99034005, 0.5250659 , 0.94613516,
           1.762905  , 1.4356766 , 0.67307633, 1.4360083 , 1.4572653 ,
           0.22226214, 0.50866336, 0.33183217, 0.57955205, 1.7685014 ,
           1.2846231 , 1.3788173 , 0.8747939 , 0.5075159 , 0.3576218 ,
           0.33196926, 1.3688413 , 0.7576706 , 0.9585252 , 0.27926588,
           0.5033353 , 1.7129889 , 1.6079798 , 0.35150182, 0.6999687 ,
           1.3321447 , 1.6698563 , 0.70452195, 0.28319418, 2.1310909 ,
           0.3258345 , 0.7243432 , 0.81880844, 0.411018  , 0.9769152 ,
           0.9223339 , 0.21013546, 0.23374689, 2.0538142 , 0.6565119 ,
           1.7624803 , 0.72002494, 0.22492659, 0.28156114, 0.5680841 ,
           2.11473   , 0.83155173, 0.7891407 , 1.4250221 , 1.8838968 ,
           1.3587425 , 1.7310114 , 0.85265195, 1.6449909 , 0.8109962 ,
           0.9811071 , 0.33558047, 1.5699701 , 0.47167754, 0.47530818,
           0.3742907 , 1.5779842 , 1.2365133 , 0.3067701 , 1.9186394 ,
           1.1169693 , 1.1770847 , 2.1893818 , 0.5341126 , 0.5353838 ,
           0.7029604 , 0.47277987, 0.24633026, 0.34929144, 1.6542519 ,
           0.79824054, 0.83787334, 1.3696426 , 1.7420716 , 0.91617846,
           0.67199004, 1.5849122 , 0.50078475, 1.1804256 , 0.35167813,
           0.8526281 , 0.59335136, 1.157247  , 1.6209148 , 0.7201484 ,
           0.29634917, 0.18856645, 1.7960072 , 1.3005323 , 0.19037902,
           0.629537  , 0.27001452, 0.5634093 , 0.30821908, 1.3296134 ,
           0.30536044, 1.896461  , 0.6864689 , 1.480893  , 0.48073304,
           0.38314748, 0.39025724, 1.119745  , 1.7424088 , 1.5668273 ,
           1.7645733 , 0.35064256, 1.0467646 , 1.3997321 , 0.7668346 ,
           1.8283119 , 0.69808376, 1.7548176 , 1.5536597 , 2.0474243 ,
           0.8440059 , 0.43517923, 0.39780498, 0.5602253 , 0.2136991 ,
           0.39766693, 0.83407414, 0.23081744, 0.2224611 , 0.67571974,
           1.9920452 , 0.45233023, 0.5646688 , 0.17855966, 1.6331127 ,
           0.09237194, 1.0797889 , 1.1809235 , 1.5649515 , 1.6333115 ,
           0.8126156 , 2.0062203 , 1.365101  , 0.3629043 , 0.91218174,
           0.841331  , 0.31149578, 2.2821321 , 0.8903425 , 0.69834614,
           0.30012155, 1.0642179 , 0.40089548, 1.4090189 , 0.280069  ,
           0.64791775, 1.7222848 , 2.002755  , 0.9041139 , 1.4027407 ,
           0.34044242, 0.17405033, 0.6791674 , 1.0952946 , 1.2847446 ,
           1.3378326 , 0.31677747, 1.2994996 , 1.1552294 , 1.0569918 ,
           1.6636094 , 1.2850146 , 1.5512612 , 1.8328991 , 0.7594153 ,
           0.2673508 , 0.76500595, 0.50905526, 0.31360626, 1.1438482 ,
           0.30131602, 0.22309995, 1.307293  , 0.20865846, 1.9162115 ,
           1.4958813 , 1.0869291 , 0.1811465 , 0.6047617 , 1.8611648 ,
           0.3963859 , 1.7208407 , 1.6073371 , 0.328732  , 0.8205782 ,
           2.1674337 , 0.36139798, 0.9740199 , 2.1371102 , 0.50472754,
           1.4593083 , 0.60614526, 0.30098987, 0.25089943, 0.60791457,
           1.0653917 , 0.34885752, 1.2164313 , 2.170782  , 1.4974097 ,
           1.2155995 , 0.36493075, 1.7254338 , 2.128921  , 0.83513516,
           1.1619282 , 1.6369752 , 0.9647597 , 0.61466855, 1.1292769 ,
           1.5490702 , 1.4468782 , 0.32296348, 1.245134  , 1.5757453 ,
           2.1907597 , 0.34313035, 0.15663695, 1.6838946 , 1.0441709 ,
           0.259982  , 0.8482543 , 0.62684137, 1.8578389 , 0.7054815 ,
           0.5754708 , 0.72365856, 1.6065819 , 1.2941504 , 1.8039846 ,
           0.2948767 , 0.8349498 , 1.5456029 , 1.8169351 , 1.8941436 ,
           0.56431925, 1.9805108 , 1.3215396 , 0.7111058 , 0.21661925,
           0.5816016 , 0.27320838, 1.5008984 , 0.769746  , 0.38831842,
           0.4808067 , 0.97581434, 0.3507173 , 0.61727524, 1.0723442 ,
           0.56990594, 0.42334294, 0.36688423, 0.31310773, 0.22235906],
          dtype=float32), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 231
    Process 0 estimating PZ PDF for rows 0 - 231
    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x55620b6023a0>, 'bestsig': np.float64(0.075), 'nneigh': 7, 'truezs': array([1.3254019 , 0.89386749, 1.78717637, 2.09287802, 1.28715646,
           1.14477348, 0.12723911, 0.13669944, 0.52842361, 0.27324033,
           0.44714141, 0.82454646, 1.8187356 , 0.51238316, 0.88228261,
           0.60520804, 0.76767957, 0.45275044, 0.95481533, 0.96224773,
           0.5167731 , 2.05815506, 0.99046284, 1.47911453, 0.58168089,
           0.31295252, 1.39858723, 0.55683708, 0.6027714 , 0.97304213,
           2.12017968, 0.74346644, 2.15838242, 0.72943944, 1.22038174,
           0.65232599, 1.25144064, 0.28268826, 0.94577205, 0.72345197,
           2.01850438, 0.19236302, 1.38413918, 0.75214303, 1.22544682,
           1.25178039, 0.56882685, 1.51727545, 0.29535747, 1.52139354,
           1.54749179, 0.41861439, 1.08331716, 0.58158028, 1.02507162,
           1.80384421, 0.57718736, 0.51131564, 1.03886624, 1.71822655,
           0.61269331, 1.64891887, 1.12601531, 0.74478543, 1.68706703,
           0.92453974, 0.08758879, 0.46544194, 0.37931514, 1.25665736,
           1.30882895, 0.95025933, 0.65654165, 0.90821153, 0.89423275,
           1.464293  , 0.69924277, 1.32411647, 0.38057172, 1.21358752,
           0.75469363, 0.77823561, 0.31368136, 3.2111891 , 1.33755088,
           0.27392101, 0.3500061 , 0.30414927, 1.02787805, 0.47388995,
           1.27942359, 0.98023736, 2.22703657, 1.28357577, 1.00389075,
           1.22945547, 0.71399468, 0.82936347, 1.33183436, 0.95288533,
           1.11642027, 1.01996362, 1.10498297, 1.67404675, 0.47896004,
           0.66169471, 0.31951833, 0.19057477, 0.19163287, 0.31050837,
           0.44918132, 0.83133233, 0.43797266, 1.22573233, 0.68544328,
           0.90613908, 0.17675316, 1.02914719, 1.76237416, 0.25641   ,
           2.05314708, 0.47210896, 1.50417007, 0.67155504, 0.57147968,
           1.5322901 , 0.1458056 , 1.7100867 , 0.26492941, 0.26390779,
           0.73220861, 2.02095175, 0.33337343, 0.4937464 , 0.87459141,
           0.72961587, 0.21930182, 0.33107889, 0.41093481, 1.80073881,
           0.67002618, 1.97048172, 1.85627854, 0.90835059, 0.29047775,
           1.70578818, 1.01115263, 1.85410857, 0.74832529, 1.35664308,
           1.47933388, 1.08627844, 0.23729324, 0.5662998 , 2.19482636,
           0.51530981, 1.16201806, 0.72171921, 1.63489866, 2.2113328 ,
           1.18514204, 0.66059667, 0.7399084 , 0.89940292, 0.17477322,
           1.01260579, 1.02117932, 1.25851142, 2.04135942, 0.86582661,
           1.14377129, 0.54520833, 0.69498348, 0.61837109, 1.42436576,
           0.36589777, 0.65094453, 2.16353774, 0.34921026, 0.3559624 ,
           0.06207943, 1.06911707, 0.95071113, 0.19231594, 0.19316709,
           1.06849265, 1.73141658, 0.64692485, 1.54006732, 0.8265726 ,
           1.58825612, 1.04155624, 0.32656479, 0.49142575, 1.97615528,
           3.28429428, 0.48045671, 1.35570776, 0.17260182, 0.34545076,
           2.23647356, 0.70189863, 1.86596215, 1.22819817, 0.17718768,
           1.11427915, 0.2900548 , 1.43590689, 0.55186069, 1.02492726,
           0.17632508, 0.6297785 , 0.60108072, 0.85513842, 0.27807105,
           1.20053196, 1.02812076, 0.43679428, 0.35992527, 0.76615131,
           1.98245275, 0.29422593, 0.78390068, 1.82784855, 0.4401505 ,
           0.64015281, 0.94077557, 1.73439538, 0.37179923, 0.74718088,
           0.554232  , 0.35659623, 2.19962217, 1.76991057, 0.60914344,
           0.42628634, 0.76279747, 0.74197268, 2.08841896, 0.37222052,
           0.97797817, 0.43334973, 0.34014833, 0.45474851, 0.53600311,
           0.49932325, 1.20640898, 1.15356994, 0.15389156, 0.57778418,
           0.72689766, 0.26679778, 0.39875436, 1.45792757, 0.99333298,
           1.27479291, 0.32616198, 0.94644463, 0.28129601, 0.72690469,
           0.51303083, 0.27702117, 0.58606493, 0.53993559, 0.34621716,
           0.5576269 , 0.32034051, 0.22125661, 0.58054543, 0.66462928,
           1.09533572, 0.29229963, 0.76470405, 1.1354841 , 1.22811079,
           0.36220884, 0.66774464, 0.18750942, 1.37214458, 0.35170412,
           1.52987993, 1.09060454, 1.08654141, 1.25641513, 0.91625017,
           0.68158931, 2.10575819, 1.39133   , 1.19017816, 0.79652282,
           1.46716475, 0.97342956, 0.32748342, 0.29285145, 0.86172688,
           0.32840693, 0.38082767, 0.25583041, 0.28015506, 0.28518021,
           1.64374232, 1.54951692, 0.69930238, 0.31139731, 0.29452419,
           1.45355332, 1.03774619, 0.57291067, 1.11468351, 0.44586074,
           0.31297922, 0.24485064, 0.83245289, 0.16677165, 1.96134448,
           0.5684309 , 0.640288  , 0.21630156, 0.43498111, 0.40336752,
           1.2391386 , 0.35983181, 0.31637847, 0.46535468, 1.53786004,
           2.08770466, 0.34686935, 1.19873023, 0.98761255, 0.51960409,
           0.4861629 , 1.42179739, 1.06765413, 1.33160782, 0.94889688,
           0.16774845, 1.02446747, 0.84400338, 0.86729765, 0.2003715 ,
           0.70239532, 0.26743448, 0.27873039, 0.62572163, 0.74388325,
           0.25678289, 2.16387033, 0.62058419, 1.73552942, 1.56507945,
           0.46541715, 0.32208681, 0.57792366, 0.30721629, 0.42273712,
           0.8419348 , 0.88949144, 1.6528194 , 0.15954518, 1.63307226,
           1.38864358, 0.26169693, 1.65519917, 1.57244766, 0.33807671,
           0.21921611, 0.6138441 , 0.14574707, 1.154194  , 0.52505016,
           0.20260882, 0.89132267, 0.53417361, 0.28514588, 0.59255743,
           0.31618166, 1.25867736, 0.73114645, 0.64891797, 0.43458962,
           0.9386279 , 1.52343738, 0.32217729, 0.33446097, 1.76911783,
           0.44772649, 2.14509964, 1.02151203, 0.37267709, 0.30175018,
           0.46907663, 0.30592418, 1.12248051, 0.29202497, 1.43494999,
           2.08337021, 0.73687506, 0.58970928, 2.10818243, 0.60214007,
           0.17416728, 0.11336923, 1.81782639, 0.34555411, 1.83384669,
           0.56675404, 1.2686193 , 0.32640696, 0.56540984, 0.38020515,
           1.32467103, 0.57395542, 1.30831265, 0.91303951, 0.99835724,
           0.33403754, 0.91415972, 0.27690053, 0.14257753, 1.03991854,
           1.18239427, 1.22354472, 0.94383419, 1.60046017, 1.82383525,
           0.76552379, 1.48133945, 1.66099274, 2.02082896, 0.61049187,
           0.54624903, 1.77851558, 1.57197881, 0.26386952, 1.97439035,
           0.80099809, 0.51710379, 0.79824793, 2.00336742, 0.72561875,
           0.58428872, 0.52964056, 0.49456775, 0.48134172, 2.02585244,
           0.81026763, 0.24379098, 0.50191575, 0.32375944, 0.42594481,
           0.56096059, 1.31865525, 0.6481654 , 1.47272885, 0.52530813,
           0.30161166, 0.96893489, 0.28358138, 1.24615514, 1.81303584,
           2.48744978, 1.15313411, 0.65358258, 0.74615943, 0.22021604,
           0.92097086, 1.91582215, 1.20942569, 0.54062045, 0.76542664,
           0.19027257, 0.58722305, 1.74749815, 1.52199864, 1.61107469,
           0.34590602, 0.49899781, 0.28908253, 0.91092587, 0.29988897,
           0.58921403, 0.29844642, 0.5673281 , 1.36544859, 1.44412935,
           0.35949934, 0.28131962, 0.5466361 , 0.27927268, 1.82492471,
           0.65892631, 0.20149231, 0.56529313, 1.41667473, 1.2047888 ,
           0.80600727, 2.11601567, 2.04066816, 0.76136184, 1.0009352 ,
           0.36117136, 0.86685115, 0.6599654 , 1.05344892, 1.44050562,
           0.3983264 , 1.11978483, 0.68926698, 0.76282245, 0.41982901,
           1.17276883, 1.02362227, 1.51457071, 0.37699008, 0.41640508,
           1.2688667 , 0.34693825, 0.79580468]), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 231
    Process 0 estimating PZ PDF for rows 0 - 231
    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighEstimator
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x5562286da3c0>, 'bestsig': np.float64(0.05333333333333334), 'nneigh': 3, 'truezs': array([0.89386749, 2.09287802, 1.28715646, 0.13669944, 0.52842361,
           0.44714141, 0.82454646, 0.51238316, 0.88228261, 0.76767957,
           0.45275044, 0.95481533, 0.5167731 , 0.58168089, 0.31295252,
           0.55683708, 0.97304213, 0.72943944, 0.65232599, 0.94577205,
           0.19236302, 1.38413918, 0.56882685, 0.58158028, 1.02507162,
           0.57718736, 0.51131564, 1.03886624, 0.61269331, 0.92453974,
           0.46544194, 1.25665736, 0.95025933, 0.65654165, 0.90821153,
           0.89423275, 1.464293  , 0.69924277, 0.38057172, 1.21358752,
           0.77823561, 0.31368136, 0.3500061 , 0.30414927, 0.47388995,
           1.27942359, 1.28357577, 1.22945547, 0.82936347, 1.33183436,
           0.95288533, 1.01996362, 1.10498297, 0.47896004, 0.66169471,
           0.31951833, 0.19057477, 0.31050837, 0.44918132, 0.83133233,
           0.68544328, 1.02914719, 0.47210896, 1.50417007, 0.67155504,
           0.57147968, 0.1458056 , 0.26492941, 0.26390779, 0.4937464 ,
           0.72961587, 0.21930182, 0.33107889, 0.67002618, 1.97048172,
           0.23729324, 0.5662998 , 0.51530981, 1.63489866, 0.66059667,
           0.89940292, 1.01260579, 1.02117932, 1.25851142, 0.86582661,
           0.69498348, 0.61837109, 0.34921026, 0.3559624 , 0.06207943,
           1.06911707, 0.95071113, 0.19316709, 0.64692485, 0.32656479,
           0.49142575, 0.48045671, 0.34545076, 0.70189863, 0.17718768,
           0.55186069, 0.6297785 , 0.85513842, 0.27807105, 0.35992527,
           0.29422593, 0.78390068, 0.4401505 , 0.64015281, 0.37179923,
           0.74718088, 0.60914344, 0.76279747, 0.74197268, 0.37222052,
           0.34014833, 0.45474851, 0.49932325, 0.15389156, 0.57778418,
           0.72689766, 0.26679778, 0.39875436, 1.27479291, 0.94644463,
           0.28129601, 0.53993559, 0.5576269 , 0.32034051, 0.58054543,
           0.66462928, 1.09533572, 0.76470405, 0.18750942, 1.37214458,
           1.09060454, 1.25641513, 0.91625017, 0.68158931, 1.39133   ,
           1.19017816, 0.79652282, 0.29285145, 0.86172688, 0.25583041,
           0.28015506, 0.69930238, 0.29452419, 1.11468351, 0.44586074,
           0.31297922, 0.83245289, 0.640288  , 0.43498111, 0.40336752,
           1.2391386 , 0.46535468, 0.34686935, 0.51960409, 0.4861629 ,
           1.06765413, 1.33160782, 0.94889688, 0.2003715 , 0.27873039,
           0.62572163, 0.74388325, 0.25678289, 0.62058419, 1.56507945,
           0.46541715, 0.57792366, 0.42273712, 0.8419348 , 0.88949144,
           0.15954518, 1.38864358, 0.26169693, 0.21921611, 0.6138441 ,
           0.14574707, 1.154194  , 0.52505016, 0.89132267, 0.31618166,
           0.64891797, 0.43458962, 0.9386279 , 0.32217729, 0.33446097,
           0.44772649, 0.37267709, 0.46907663, 0.30592418, 1.12248051,
           0.73687506, 0.58970928, 0.60214007, 0.17416728, 0.11336923,
           0.56675404, 0.56540984, 0.38020515, 0.57395542, 1.30831265,
           0.91303951, 0.99835724, 0.33403754, 0.14257753, 1.18239427,
           0.94383419, 0.26386952, 0.80099809, 0.79824793, 0.72561875,
           0.49456775, 0.48134172, 0.81026763, 0.32375944, 0.52530813,
           0.30161166, 0.28358138, 1.81303584, 0.65358258, 0.22021604,
           1.20942569, 0.54062045, 0.76542664, 0.19027257, 0.58722305,
           1.61107469, 0.34590602, 0.49899781, 0.91092587, 0.58921403,
           0.29844642, 1.36544859, 0.35949934, 0.5466361 , 0.65892631,
           0.56529313, 1.2047888 , 0.80600727, 2.04066816, 0.76136184,
           0.86685115, 1.05344892, 0.3983264 , 0.76282245, 1.02362227,
           1.51457071, 1.2688667 , 0.34693825]), 'only_colors': False}, KNearNeighEstimator
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

    {'weights': array([[0.2201824 , 0.20534083, 0.15971718, ..., 0.10103702, 0.10102804,
            0.09997908],
           [0.16764358, 0.14324085, 0.14302643, ..., 0.14207148, 0.13267892,
            0.12872115],
           [0.32757241, 0.15800938, 0.1314739 , ..., 0.09295102, 0.09240935,
            0.08924883],
           ...,
           [0.2404289 , 0.14322931, 0.13790509, ..., 0.12291801, 0.11204479,
            0.11149682],
           [0.18226461, 0.15397391, 0.15201749, ..., 0.13088583, 0.12795103,
            0.10786032],
           [0.1912003 , 0.16586703, 0.15148472, ..., 0.13376674, 0.11143036,
            0.10579118]], shape=(231, 7)), 'stds': array([[0.075, 0.075, 0.075, ..., 0.075, 0.075, 0.075],
           [0.075, 0.075, 0.075, ..., 0.075, 0.075, 0.075],
           [0.075, 0.075, 0.075, ..., 0.075, 0.075, 0.075],
           ...,
           [0.075, 0.075, 0.075, ..., 0.075, 0.075, 0.075],
           [0.075, 0.075, 0.075, ..., 0.075, 0.075, 0.075],
           [0.075, 0.075, 0.075, ..., 0.075, 0.075, 0.075]], shape=(231, 7)), 'means': array([[0.60914344, 0.66059667, 0.6205842 , ..., 0.4937464 , 0.76470405,
            0.57718736],
           [0.27927268, 0.36589777, 0.34590602, ..., 1.5495169 , 0.38082767,
            1.7691178 ],
           [0.60214007, 0.2003715 , 0.4861629 , ..., 0.5609606 , 0.45275044,
            0.2639078 ],
           ...,
           [0.62572163, 0.6992428 , 0.4471414 , ..., 0.51041174, 0.5466361 ,
            0.5815803 ],
           [1.0250716 , 1.1535699 , 1.0691171 , ..., 1.0249273 , 1.0199636 ,
            0.94644463],
           [0.31297922, 0.27807105, 0.34693825, ..., 0.31951833, 0.23907995,
            0.3559624 ]], shape=(231, 7), dtype=float32)}


Typically the ancillary data table includes a photo-z point estimate
derived from the PDFs, by default this is the mode of the distribution,
called ‘zmode’ in the ancillary dictionary below:

.. code:: ipython3

    # this is the ancillary dictionary of the output Ensemble, which in this case
    # contains the zmode, redshift, and distribution type
    print(estimated_photoz["lsst_error_model"]["output"].ancil)


.. parsed-literal::

    {'zmode': array([[0.63],
           [0.34],
           [0.57],
           [0.27],
           [1.37],
           [0.45],
           [0.87],
           [0.58],
           [1.2 ],
           [0.68],
           [1.12],
           [1.08],
           [0.67],
           [1.11],
           [0.87],
           [0.67],
           [0.54],
           [1.04],
           [0.22],
           [0.31],
           [1.1 ],
           [1.09],
           [0.88],
           [0.53],
           [0.86],
           [0.26],
           [0.38],
           [0.49],
           [0.32],
           [0.31],
           [1.31],
           [0.86],
           [0.76],
           [0.87],
           [0.49],
           [1.52],
           [1.37],
           [0.58],
           [1.35],
           [1.08],
           [0.52],
           [0.61],
           [0.56],
           [0.36],
           [0.26],
           [0.59],
           [0.78],
           [1.3 ],
           [0.33],
           [0.42],
           [0.88],
           [0.28],
           [0.86],
           [0.45],
           [0.57],
           [0.53],
           [0.63],
           [0.93],
           [0.15],
           [0.33],
           [0.71],
           [0.53],
           [1.13],
           [1.07],
           [0.18],
           [0.85],
           [1.09],
           [1.51],
           [0.51],
           [0.5 ],
           [1.11],
           [0.23],
           [0.86],
           [1.22],
           [0.78],
           [0.9 ],
           [1.02],
           [2.06],
           [0.49],
           [0.43],
           [0.79],
           [0.79],
           [0.34],
           [0.31],
           [0.57],
           [0.6 ],
           [0.24],
           [0.52],
           [0.26],
           [0.29],
           [0.3 ],
           [0.24],
           [0.55],
           [0.68],
           [0.28],
           [0.43],
           [0.9 ],
           [0.87],
           [1.03],
           [0.9 ],
           [0.93],
           [0.37],
           [0.34],
           [0.32],
           [0.2 ],
           [0.45],
           [0.18],
           [1.83],
           [0.18],
           [0.96],
           [0.28],
           [0.26],
           [0.32],
           [0.58],
           [0.21],
           [0.43],
           [0.53],
           [0.24],
           [0.95],
           [1.09],
           [0.94],
           [0.31],
           [0.33],
           [0.74],
           [0.2 ],
           [0.75],
           [1.53],
           [1.31],
           [0.22],
           [0.26],
           [0.32],
           [1.09],
           [1.14],
           [1.34],
           [0.84],
           [0.27],
           [0.32],
           [2.13],
           [0.8 ],
           [0.73],
           [0.42],
           [0.71],
           [0.5 ],
           [0.9 ],
           [0.25],
           [0.24],
           [0.77],
           [0.57],
           [0.28],
           [0.27],
           [0.57],
           [0.9 ],
           [0.93],
           [1.52],
           [0.3 ],
           [0.86],
           [0.24],
           [0.55],
           [0.46],
           [0.23],
           [0.52],
           [0.77],
           [0.5 ],
           [0.36],
           [0.65],
           [0.78],
           [0.91],
           [0.79],
           [0.61],
           [0.63],
           [0.52],
           [0.6 ],
           [0.28],
           [0.35],
           [0.3 ],
           [0.92],
           [0.7 ],
           [0.65],
           [0.9 ],
           [0.37],
           [0.24],
           [0.3 ],
           [0.56],
           [0.67],
           [1.02],
           [0.69],
           [0.91],
           [0.74],
           [1.08],
           [0.27],
           [0.49],
           [0.28],
           [0.22],
           [1.03],
           [1.31],
           [0.68],
           [0.78],
           [0.51],
           [0.19],
           [0.19],
           [1.04],
           [0.56],
           [0.99],
           [0.24],
           [0.8 ],
           [0.36],
           [0.38],
           [0.23],
           [0.72],
           [0.35],
           [1.12],
           [0.38],
           [1.31],
           [0.88],
           [1.05],
           [0.77],
           [0.79],
           [0.61],
           [0.29],
           [0.55],
           [1.38],
           [0.76],
           [0.21],
           [0.62],
           [0.88],
           [0.36],
           [0.51],
           [0.39],
           [0.59],
           [1.04],
           [0.31]]), 'redshift': 0      0.648564
    1      0.341317
    2      0.491075
    3      0.196691
    4      1.324315
             ...   
    226    0.480807
    227    0.350717
    228    0.617275
    229    1.072344
    230    0.366884
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
          <td>0.032832</td>
          <td>0.076333</td>
          <td>0.032554</td>
          <td>0.014760</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.01</td>
          <td>0.042564</td>
          <td>0.090581</td>
          <td>0.042121</td>
          <td>0.021217</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.02</td>
          <td>0.054592</td>
          <td>0.106574</td>
          <td>0.053910</td>
          <td>0.030010</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.03</td>
          <td>0.069263</td>
          <td>0.124419</td>
          <td>0.068245</td>
          <td>0.041758</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.04</td>
          <td>0.086922</td>
          <td>0.144236</td>
          <td>0.085441</td>
          <td>0.057139</td>
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
    0    0.648564   23.150772   22.366415   21.491493   20.731236   20.491596   
    1    0.341317   24.068222   23.826418   23.405294   23.416424   22.987785   
    2    0.491075   27.021898   26.095686   25.032051   24.618488   24.387232   
    3    0.196691   24.893013   23.983337   23.310110   22.938263   22.722412   
    4    1.324315   27.208641   25.888857   25.021338   23.973490   23.354897   
    ..        ...         ...         ...         ...         ...         ...   
    226  0.480807   24.777891   24.257607   23.544462   23.205730   23.162960   
    227  0.350717   25.233673   23.563206   22.176367   21.583717   21.211185   
    228  0.617275   27.384861   25.180693   23.666212   22.560936   22.159149   
    229  1.072344   24.572737   24.226124   23.692007   23.160368   22.515059   
    230  0.366884   25.564947   24.567034   23.710983   23.469208   23.265720   
    
         mag_y_lsst  
    0     20.287109  
    1     23.289028  
    2     24.167744  
    3     22.617384  
    4     22.628229  
    ..          ...  
    226   22.997458  
    227   21.033613  
    228   21.869398  
    229   22.385687  
    230   23.191113  
    
    [231 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.648564   23.150772   22.366415   21.491493   20.731236   20.491596   
    1    0.341317   24.068222   23.826418   23.405294   23.416424   22.987785   
    2    0.491075   27.021898   26.095686   25.032051   24.618488   24.387232   
    3    0.196691   24.893013   23.983337   23.310110   22.938263   22.722412   
    4    1.324315   27.208641   25.888857   25.021338   23.973490   23.354897   
    ..        ...         ...         ...         ...         ...         ...   
    226  0.480807   24.777891   24.257607   23.544462   23.205730   23.162960   
    227  0.350717   25.233673   23.563206   22.176367   21.583717   21.211185   
    228  0.617275   27.384861   25.180693   23.666212   22.560936   22.159149   
    229  1.072344   24.572737   24.226124   23.692007   23.160368   22.515059   
    230  0.366884   25.564947   24.567034   23.710983   23.469208   23.265720   
    
         mag_y_lsst  
    0     20.287109  
    1     23.289028  
    2     24.167744  
    3     22.617384  
    4     22.628229  
    ..          ...  
    226   22.997458  
    227   21.033613  
    228   21.869398  
    229   22.385687  
    230   23.191113  
    
    [231 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.648564   23.150772   22.366415   21.491493   20.731236   20.491596   
    1    0.341317   24.068222   23.826418   23.405294   23.416424   22.987785   
    2    0.491075   27.021898   26.095686   25.032051   24.618488   24.387232   
    3    0.196691   24.893013   23.983337   23.310110   22.938263   22.722412   
    4    1.324315   27.208641   25.888857   25.021338   23.973490   23.354897   
    ..        ...         ...         ...         ...         ...         ...   
    226  0.480807   24.777891   24.257607   23.544462   23.205730   23.162960   
    227  0.350717   25.233673   23.563206   22.176367   21.583717   21.211185   
    228  0.617275   27.384861   25.180693   23.666212   22.560936   22.159149   
    229  1.072344   24.572737   24.226124   23.692007   23.160368   22.515059   
    230  0.366884   25.564947   24.567034   23.710983   23.469208   23.265720   
    
         mag_y_lsst  
    0     20.287109  
    1     23.289028  
    2     24.167744  
    3     22.617384  
    4     22.628229  
    ..          ...  
    226   22.997458  
    227   21.033613  
    228   21.869398  
    229   22.385687  
    230   23.191113  
    
    [231 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.648564   23.150772   22.366415   21.491493   20.731236   20.491596   
    1    0.341317   24.068222   23.826418   23.405294   23.416424   22.987785   
    2    0.491075   27.021898   26.095686   25.032051   24.618488   24.387232   
    3    0.196691   24.893013   23.983337   23.310110   22.938263   22.722412   
    4    1.324315   27.208641   25.888857   25.021338   23.973490   23.354897   
    ..        ...         ...         ...         ...         ...         ...   
    226  0.480807   24.777891   24.257607   23.544462   23.205730   23.162960   
    227  0.350717   25.233673   23.563206   22.176367   21.583717   21.211185   
    228  0.617275   27.384861   25.180693   23.666212   22.560936   22.159149   
    229  1.072344   24.572737   24.226124   23.692007   23.160368   22.515059   
    230  0.366884   25.564947   24.567034   23.710983   23.469208   23.265720   
    
         mag_y_lsst  
    0     20.287109  
    1     23.289028  
    2     24.167744  
    3     22.617384  
    4     22.628229  
    ..          ...  
    226   22.997458  
    227   21.033613  
    228   21.869398  
    229   22.385687  
    230   23.191113  
    
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

    {'lsst_error_model': {'cdeloss': array([-2.86276227]),
      'brier': array([215.05790881])},
     'inv_redshift_inc': {'cdeloss': array([-6.87797104]),
      'brier': array([364.76679897])},
     'line_confusion': {'cdeloss': array([-2.74641839]),
      'brier': array([203.2424497])},
     'quantity_cut': {'cdeloss': array([-2.45543461]),
      'brier': array([325.42226963])}}



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
