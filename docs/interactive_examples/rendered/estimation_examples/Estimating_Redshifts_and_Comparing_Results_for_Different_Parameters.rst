Estimating photometric redshifts with RAIL stages and comparing results for different parameters
================================================================================================

**Authors:** Jennifer Scora, Tai Withers, Mubdi Rahman

**Last run successfully:** Feb 9, 2026

This notebook shows how to run through the various `stages of
RAIL <https://rail-hub.readthedocs.io/en/latest/source/rail_stages/what_are_rail_stages.html>`__
(creation, estimation, evaluation) in order to create a simulated
dataset of galaxy magnitudes and redshifts, use the magnitudes to
estimate photometric redshifts, and then compare the resulting estimated
photometric redshifts to the *true* redshifts. We will be using the
`K-Nearest Neighbour
algorithm <https://rail-hub.readthedocs.io/en/latest/source/rail_stages/estimation.html#k-nearest-neighbor>`__
(KNN) to estimate redshifts, and testing out how the limits on the
number of nearest neighbours affect the resulting esimated redshift
distributions.

To do this, we loop over the estimation and evaluation stages while
varying these parameters to test their effect. We will also then be
exploring how we can parallelize this loop within the notebook to speed
things up a little. However, if you are running on very large datasets,
we recommend running in pipeline mode instead (see instructions
`here <https://rail-hub.readthedocs.io/en/latest/source/user_guide/pipeline_usage.html>`__),
as it is not possible to loop over large files with the interactive mode
of RAIL.

Here are the steps that we’re going to cover in this notebook:

1. Creating a realistic data set of galaxy magnitudes and true redshifts
2. Estimating the photometric redshifts
3. Summarizing the redshift distributions
4. Evaluating the photometric redshifts against the *true* values
5. Repeating steps 2-4 with a parallelized loop

Before we get started, here’s a quick introduction to some of the
features of RAIL interactive mode. The only RAIL package you need to
import is the ``rail.interactive`` package. This contains all of the
interactive functions for all of the RAIL algorithms. You may need to
import supporting functions that are not part of a stage separately. To
get a sense of what functions/stages are available and for some more
detailed instructions, see `the RAIL
documentation <https://rail-hub.readthedocs.io/en/latest/source/user_guide/interactive_usage.html>`__.

.. code:: ipython3

    # import the packages we'll need
    import rail.interactive as ri


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
      File "/tmp/ipykernel_6942/3510305779.py", line 2, in <module>
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


To get the docstrings for a function, including what parameters it needs
and what it returns, you can just put a question mark after the function
call or use the ``help()`` function, as you would with any python
function.

.. code:: ipython3

    ri.creation.engines.flowEngine.flow_modeler?

1. Creating a realistic data set of galaxy magnitudes and true redshifts
------------------------------------------------------------------------

First we want to create the data sets of galaxy magnitudes that we will
use to estimate photometric redshifts. We will use the `PZflow
algorithm <https://rail-hub.readthedocs.io/en/latest/source/rail_stages/creation.html#pzflow-engine>`__,
which is a machine-learning algorithm, to generate our model. Then we
pull two data sets from the model, a calibration dataset and a target
dataset. The calibration data set will be used to calibrate our models,
and the target data is what we’ll get photo-z estimates for. We’ll then
degrade these datasets so that they better resemble real data from the
Rubin telescope.

.. code:: ipython3

    # importing some supplementary packages and functions
    import numpy as np
    from pzflow.examples import get_galaxy_data
    
    # plotting imports
    import matplotlib.pyplot as plt
    
    %matplotlib inline

Here we first need to set up some column name dictionaries, as the
expected column names vary between some of the codes. In order to handle
this, we can pass in dictionaries of expected column names and the
column name that exists in the input data (``band_dict`` and
``rename_dict`` below). In this notebook, we are using bands ugrizy, and
each band will have a name ‘mag_u_lsst’, for example, with the error
column name being ‘mag_err_u_lsst’.

The initial data we pull from our model won’t have any associated
errors. Those will be created when we degrade the datasets, but the
error columns will need to be renamed with the ``rename_dict`` later on.

.. code:: ipython3

    bands = ["u", "g", "r", "i", "z", "y"]
    band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"mag_{band}_lsst_err": f"mag_err_{band}_lsst" for band in bands}

In order to generate the model with PZflow, we need to grab some sample
data to base the model off of. This sample data is only used to create
the model, and is seperate from the training and test data we’ll get
from the model later. We’ll rename the band columns in this data table
to match our desired band names as discussed above, using ``band_dict``.
We can check that our columns have been renamed appropriately by
printing out the first few lines of the table:

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
magnitude. - **num_training_epochs (optional):** By default 30, here
we’re doing fewer so that it doesn’t take as long.

**NOTE:** This training may take a while depending on your setup.

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
    Inserting handle into data store.  model: inprogress_model.pkl, FlowModeler


Now, if you take a look at the output of this function, you can see that
it’s a dictionary with the key ‘model’, since that’s what we’re
generating, and the actual model object as the value. If there were
multiple outputs for this function, they would all be collected in this
dictionary:

.. code:: ipython3

    print(flow_model)


.. parsed-literal::

    {'model': <pzflow.flow.Flow object at 0x7f285d951f00>}


Now we’ll use the flow to produce some synthetic data for our
calibration data set, which we’ll need to calibrate the KNN estimation
algorithm later. We’ll just create a small dataset of 250 galaxies for
this sample, so we’ll pass in the argument: ``n_samples = 250``. We’ll
also (optionally) use a specific seed for this so that it’s
reproducible.

**Note that when we pass the model to this function, we don’t pass the
dictionary, but the actual model object. This is true of all the
interactive functions.**

.. code:: ipython3

    # sample calibration data set from the model
    calib_data_orig = ri.creation.engines.flowEngine.flow_creator(
        n_samples=250, model=flow_model["model"], seed=1235
    )


.. parsed-literal::

    Inserting handle into data store.  model: <pzflow.flow.Flow object at 0x7f285d951f00>, FlowCreator


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.pq, FlowCreator


Now we can look at the output from this function – as before, it’s a
dictionary. Here the key is ‘output’ instead of model, and the data is
just given as a table:

.. code:: ipython3

    print(calib_data_orig)


.. parsed-literal::

    {'output':      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    1.457567   28.267445   27.804485   27.435261   26.651432   26.222874   
    1    1.325402   26.251822   25.970957   25.726908   25.101812   24.611454   
    2    0.893867   26.969597   26.337978   24.742476   23.584187   22.719334   
    3    1.787176   29.413265   28.371691   27.711725   27.122871   26.453514   
    4    1.302208   25.956703   25.612457   25.233931   24.481222   23.915579   
    ..        ...         ...         ...         ...         ...         ...   
    245  0.551861   22.804483   21.715130   20.476379   19.528458   19.157227   
    246  1.382002   26.747984   26.366240   26.045853   25.634527   25.366415   
    247  1.024927   27.088902   26.794044   26.218945   25.538654   24.945810   
    248  0.176325   27.062140   25.895752   25.386101   25.088024   24.981846   
    249  0.629779   26.145018   25.360502   24.390930   23.710407   23.538721   
    
         mag_y_lsst  
    0     25.534056  
    1     23.984524  
    2     22.381075  
    3     26.127983  
    4     23.255400  
    ..          ...  
    245   18.923553  
    246   24.823549  
    247   24.730619  
    248   24.936480  
    249   23.396826  
    
    [250 rows x 7 columns]}


Now let’s do the same thing, except this time we’re going to grab our
target data set. This data set is our ‘actual’ data set, that we’ll feed
into the KNN estimation model to get our redshifts. Again, we’ll just
create a small dataset of 250 galaxies, and we’ll use a different seed
to ensure that the data won’t be the same as the calibration set.

.. code:: ipython3

    # sample target data set from the model
    targ_data_orig = ri.creation.engines.flowEngine.flow_creator(
        model=flow_model["model"], n_samples=250, seed=1234
    )


.. parsed-literal::

    Inserting handle into data store.  model: <pzflow.flow.Flow object at 0x7f285d951f00>, FlowCreator
    Inserting handle into data store.  output: inprogress_output.pq, FlowCreator


Let’s check out the distributions of galaxy redshifts, just to make sure
they aren’t the same:

.. code:: ipython3

    hist_options = {"bins": np.linspace(0, 3, 30), "histtype": "stepfilled", "alpha": 0.5}
    
    plt.hist(calib_data_orig["output"]["redshift"], label="calibration data", **hist_options)
    plt.hist(targ_data_orig["output"]["redshift"], label="target data", **hist_options)
    plt.legend(loc="best")
    plt.xlabel("redshift")
    plt.ylabel("number of galaxies")




.. parsed-literal::

    Text(0, 0.5, 'number of galaxies')




.. image:: Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_files/Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_26_1.png


Degrade the data sets
~~~~~~~~~~~~~~~~~~~~~

Next we will apply some degradation functions to the data, to make it
look more like real observations. We apply the following functions to
the calibration data set: 1. ``lsst_error_model`` to add photometric
errors that are modelled based on the telescope 2.
``inv_redshift_incompleteness`` to mimic redshift dependent
incompleteness 3. ``line_confusion`` to simulate the effect of
misidentified lines 4. ``quantity_cut`` mimics a band-dependent
brightness cut

We then use the administrative function ``column_mapper`` to rename the
error columns so that they match the names in DC2.

For the target data set, we only apply the ``lsst_error_model``
degradations, as well as making the above structural changes to get the
data in the same output format as the calibration data set. This is
beause we want to be able to compare the estimated redshifts to the true
redshifts later on, and to do this when applying cuts can get a bit
complicated. If you want to see how this works, we go into detail about
this in the the
`Exploring_the_Effects_of_Degraders.ipynb <https://rail-hub.readthedocs.io/projects/rail-notebooks/en/latest/interactive_examples/rendered/creation_examples/Exploring_the_Effects_of_Degraders.html>`__
notebook.

1. Apply the ``lsst_error_model`` to both calibration and target data
   sets. Once again, we’re supplying different seeds to ensure the
   results are reproducible and different from each other. We are also
   using the ``band_dict`` we created earlier, which tells the code what
   the band column names should be. We are also supplying
   ``ndFlag=np.nan``, which just tells the code to make non-detections
   ``np.nan`` in the output.

.. code:: ipython3

    # add photometric errors modelled on LSST to the data
    calib_data_errs = ri.creation.degraders.photometric_errors.lsst_error_model(
        sample=calib_data_orig["output"], seed=66, renameDict=band_dict, ndFlag=np.nan
    )
    
    targ_data_errs = ri.creation.degraders.photometric_errors.lsst_error_model(
        sample=targ_data_orig["output"], seed=58, renameDict=band_dict, ndFlag=np.nan
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, LSSTErrorModel
    Inserting handle into data store.  output: inprogress_output.pq, LSSTErrorModel
    Inserting handle into data store.  input: None, LSSTErrorModel
    Inserting handle into data store.  output: inprogress_output.pq, LSSTErrorModel


.. code:: ipython3

    # let's see what the output looks like
    calib_data_errs["output"].head()




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



You can see that the error columns have been added in for each of the
magnitude columns.

Now let’s take a look at what’s happened to the magnitudes. Below we’ll
plot the u-band magnitudes before and after running the degrader. You
can see that the higher magnitude objects now have a much wider variance
in magnitude compared to their initial magnitudes, but at lower
magnitudes they’ve remained similar:

.. code:: ipython3

    # we have to set the range because there are nans in the new
    # dataset with errors, which messes up plt.hist2d
    range = [
        [
            np.min(calib_data_orig["output"]["mag_u_lsst"]),
            np.max(calib_data_orig["output"]["mag_u_lsst"]),
        ],
        [
            np.min(calib_data_errs["output"]["mag_u_lsst"]),
            np.max(calib_data_errs["output"]["mag_u_lsst"]),
        ],
    ]
    plt.hist2d(
        calib_data_orig["output"]["mag_u_lsst"],
        calib_data_errs["output"]["mag_u_lsst"],
        range=range,
        bins=20,
        cmap="viridis",
    )
    plt.xlabel("original u-band magnitude")
    plt.ylabel("new u-band magnitude")
    plt.colorbar(label="number of galaxies")




.. parsed-literal::

    <matplotlib.colorbar.Colorbar at 0x7f27ec4e1030>




.. image:: Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_files/Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_32_1.png


You can make this plot for all the other magnitudes if you’d like.

2. Use ``inv_redshift_incompleteness`` to mimic redshift dependent
   incompleteness by removing some galaxies above a redshift threshold.
   The threshold is given as ``pivot_redshift``:

.. code:: ipython3

    # randomly removes some galaxies above certain redshift threshold
    calib_data_inc = (
        ri.creation.degraders.spectroscopic_degraders.inv_redshift_incompleteness(
            sample=calib_data_errs["output"], pivot_redshift=1.0
        )
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, InvRedshiftIncompleteness
    Inserting handle into data store.  output: inprogress_output.pq, InvRedshiftIncompleteness


Now let’s take a look at what’s happened to the data. We can easily see
that this has resulted in a smaller sample of galaxies:

.. code:: ipython3

    print(f"Number of galaxies after cut: {len(calib_data_inc['output'])}")
    print(f"Number of galaxies in original: {len(calib_data_errs['output'])}")


.. parsed-literal::

    Number of galaxies after cut: 211
    Number of galaxies in original: 250


Now let’s plot the redshift distributions of our input and output
sample. We can see that the distribution is the same below our redshift
threshold of 1, and above redshift 1 is where some galaxies are no
longer present:

.. code:: ipython3

    plt.hist(calib_data_errs["output"]["redshift"], label="original", **hist_options)
    plt.hist(calib_data_inc["output"]["redshift"], label="cut", **hist_options)
    plt.legend(loc="best")
    plt.xlabel("redshift")
    plt.ylabel("number of galaxies")




.. parsed-literal::

    Text(0, 0.5, 'number of galaxies')




.. image:: Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_files/Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_38_1.png


3. Apply ``line_confusion`` to simulate the effect of misidentified
   lines. The degrader will misidentify some percentage (``frac_wrong``)
   of the actual lines (here we’re picking ``5007.0`` Angstroms, which
   are OIII lines) as the line we pick for ``wrong_wavelen``. In this
   case, we’ll pick ``3727.0`` Angstroms, which are OII lines.

.. code:: ipython3

    # simulates the effect of misidentified lines
    calib_data_conf = ri.creation.degraders.spectroscopic_degraders.line_confusion(
        sample=calib_data_inc["output"],
        true_wavelen=5007.0,
        wrong_wavelen=3727.0,
        frac_wrong=0.05,
        seed=1337,
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, LineConfusion
    Inserting handle into data store.  output: inprogress_output.pq, LineConfusion


Now let’s plot the distribution of redshifts we passed into this stage
compared to what’s been output by the ``line_confusion`` function. We
can see that the output data has a few differences in the distribution,
spread across the whole range of redshifts:

.. code:: ipython3

    plt.hist(calib_data_inc["output"]["redshift"], label="input data", **hist_options)
    plt.hist(calib_data_conf["output"]["redshift"], label="output data", **hist_options)
    plt.legend(loc="best")
    plt.ylabel("redshift")
    plt.ylabel("number of galaxies")




.. parsed-literal::

    Text(0, 0.5, 'number of galaxies')




.. image:: Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_files/Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_42_1.png


4. We use ``quantity_cut`` to cut galaxies based on their specific band
   magnitudes. This function takes a dictionary of cuts, where you can
   provide the band name and the values to cut that band on. If one
   value is given, it’s considered a maximum, and if a tuple is given,
   it’s considered a range within which the sample is selected. For
   this, we’ll just set a maximum magnitude for the i band of 25.

.. code:: ipython3

    # cut some of the data below a certain magnitude
    calib_data_cut = ri.creation.degraders.quantityCut.quantity_cut(
        sample=calib_data_conf["output"], cuts={"mag_i_lsst": 25.0}
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, QuantityCut
    Inserting handle into data store.  output: inprogress_output.pq, QuantityCut


Now let’s check how this has affected the number of galaxies in our
sample:

.. code:: ipython3

    print(f"Number of galaxies after cut: {len(calib_data_cut['output'])}")
    print(f"Number of galaxies in original: {len(calib_data_conf['output'])}")


.. parsed-literal::

    Number of galaxies after cut: 102
    Number of galaxies in original: 211


We can see that this cut the sample down significantly – now we’re at
about half the galaxies!

Now let’s plot the distributions to see once again how they compare:

.. code:: ipython3

    plt.hist(calib_data_conf["output"]["redshift"], label="input data", **hist_options)
    plt.hist(calib_data_cut["output"]["redshift"], label="output data", **hist_options)
    plt.legend(loc="best")
    plt.xlabel("redshift")
    plt.ylabel("number of galaxies")




.. parsed-literal::

    Text(0, 0.5, 'number of galaxies')




.. image:: Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_files/Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_48_1.png


Now we just need to use the dictionary we made earlier of error column
names (``rename_dict``) to rename the error columns, so they match the
expected names for the later steps:

.. code:: ipython3

    # renames error columns to match DC2
    calib_data = ri.tools.table_tools.column_mapper(
        data=calib_data_cut["output"], columns=rename_dict
    )
    
    # renames error columns to match DC2
    targ_data = ri.tools.table_tools.column_mapper(
        data=targ_data_errs["output"], columns=rename_dict, hdf5_groupname=""
    )


.. parsed-literal::

    Inserting handle into data store.  input: None, ColumnMapper
    Inserting handle into data store.  output: inprogress_output.pq, ColumnMapper
    Inserting handle into data store.  input: None, ColumnMapper
    Inserting handle into data store.  output: inprogress_output.pq, ColumnMapper


We can compare the tables before and after we used the ``column_mapper``
function to see the effect on the column names:

.. code:: ipython3

    calib_data_cut["output"].head()




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
          <td>1.287156</td>
          <td>24.796497</td>
          <td>0.090664</td>
          <td>24.474573</td>
          <td>0.023417</td>
          <td>23.933826</td>
          <td>0.013242</td>
          <td>23.541721</td>
          <td>0.015037</td>
          <td>23.169586</td>
          <td>0.020285</td>
          <td>22.796036</td>
          <td>0.032570</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.136699</td>
          <td>25.557074</td>
          <td>0.174753</td>
          <td>24.917891</td>
          <td>0.034444</td>
          <td>24.815432</td>
          <td>0.027648</td>
          <td>24.561186</td>
          <td>0.035972</td>
          <td>24.793751</td>
          <td>0.084634</td>
          <td>24.787004</td>
          <td>0.186461</td>
        </tr>
        <tr>
          <th>8</th>
          <td>0.528424</td>
          <td>26.239728</td>
          <td>0.307040</td>
          <td>26.112581</td>
          <td>0.098900</td>
          <td>24.967011</td>
          <td>0.031579</td>
          <td>24.423155</td>
          <td>0.031847</td>
          <td>24.356228</td>
          <td>0.057467</td>
          <td>24.028407</td>
          <td>0.096898</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.447141</td>
          <td>24.440489</td>
          <td>0.066336</td>
          <td>22.173527</td>
          <td>0.005853</td>
          <td>20.683201</td>
          <td>0.005061</td>
          <td>19.904717</td>
          <td>0.005044</td>
          <td>19.589198</td>
          <td>0.005082</td>
          <td>19.403434</td>
          <td>0.005257</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    calib_data["output"].head()




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
          <td>1.287156</td>
          <td>24.796497</td>
          <td>0.090664</td>
          <td>24.474573</td>
          <td>0.023417</td>
          <td>23.933826</td>
          <td>0.013242</td>
          <td>23.541721</td>
          <td>0.015037</td>
          <td>23.169586</td>
          <td>0.020285</td>
          <td>22.796036</td>
          <td>0.032570</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.136699</td>
          <td>25.557074</td>
          <td>0.174753</td>
          <td>24.917891</td>
          <td>0.034444</td>
          <td>24.815432</td>
          <td>0.027648</td>
          <td>24.561186</td>
          <td>0.035972</td>
          <td>24.793751</td>
          <td>0.084634</td>
          <td>24.787004</td>
          <td>0.186461</td>
        </tr>
        <tr>
          <th>8</th>
          <td>0.528424</td>
          <td>26.239728</td>
          <td>0.307040</td>
          <td>26.112581</td>
          <td>0.098900</td>
          <td>24.967011</td>
          <td>0.031579</td>
          <td>24.423155</td>
          <td>0.031847</td>
          <td>24.356228</td>
          <td>0.057467</td>
          <td>24.028407</td>
          <td>0.096898</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.447141</td>
          <td>24.440489</td>
          <td>0.066336</td>
          <td>22.173527</td>
          <td>0.005853</td>
          <td>20.683201</td>
          <td>0.005061</td>
          <td>19.904717</td>
          <td>0.005044</td>
          <td>19.589198</td>
          <td>0.005082</td>
          <td>19.403434</td>
          <td>0.005257</td>
        </tr>
      </tbody>
    </table>
    </div>



2. Estimate the redshifts
-------------------------

Now, we estimate our photometric redshifts. We use the `KNN
algorithm <https://rail-hub.readthedocs.io/en/latest/source/rail_stages/estimation.html#k-nearest-neighbor>`__
to estimate our redshifts, varying the minimum and maximum allowed
number of neighbours to see its effect on the final result (see
`here <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`__
for more of an explanation of how KNN works).

To do this, we iterate over a list of the different parameter inputs we
want to use for the estimator. In each loop, we estimate the redshifts
with the chosen parameters.

First, we need to pick a few (min, max) neighbour limits that we can
iterate over. The default values are (3,7), so let’s try values that are
around that. We also need a dictionary where we can store the estimated
redshifts once we have them:

.. code:: ipython3

    # set up parameters to iterate over and the dictionary to store data
    nb_params = [(3, 7), (2, 6), (2, 8), (4, 9)]
    estimated_photoz = {}

Now we can loop through the two steps required to estimate redshifts:
calibrating the model and using the model to estimate. In RAIL, all
estimation algorithms have an **informer** method that calibrates, and
an **estimator** method that uses the model to estimate the redshift
distributions.

**The algorithm**: The ``K-Nearest Neighbours`` algorithm we’re using
(see
`here <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`__
for more of an explanation of how it works) is a wrapper around
``sklearn``\ ’s nearest neighbour (NN) machine learning model.
Essentially, it takes a given galaxy, identifies its nearest neighbours
in the space, in this case galaxies that have similar colours, and then
constructs the photometric redshift PDF as a sum of Gaussians from each
neighbour.

**Informer**: This method is training the model that we will use to
estimate the redshifts. We will plug in our calibration data set, and
any parameters the model needs. The parameters that we need for this
algorithm are the minimum and maximum neighbour limits, which we’ll be
iterating over. These set the minimum and maximum possible number of
neighbours that the model will use to estimate a galaxy’s redshift. They
do not set the specific number of neighbours it will use, just the range
it will test. A larger range will require more computing time. The
inform method will also set aside some of the calibration data set as a
validation data set.

**Estimator**: Once our model is trained, we can then use it to estimate
the redshifts of the target data set. We provide the estimate algorithm
with the target data set, and the model that we’ve calibrated, and any
other necessary parameters.

Common parameters: - ``nondetect_val``: This tells the code which values
are considered non-detections. We pass in ``np.nan`` here, since that’s
what we used as the ``ndFlag`` in the degradation stage for
non-detections. - ``hdf5_groupname``: the dictionary key the code will
find the data under. Set to ``""`` if the data is passed in directly.

.. code:: ipython3

    for nb_min, nb_max in nb_params:
    
        # use training data to train the informer or model that we will use to estimate redshifts
        inform_knn = ri.estimation.algos.k_nearneigh.k_near_neigh_informer(
            training_data=calib_data["output"],
            nondetect_val=np.nan,
            hdf5_groupname="",
            nneigh_min=nb_min,
            nneigh_max=nb_max,
        )
        # use the trained model to estimate the redshifts of the test data
        knn_estimated = ri.estimation.algos.k_nearneigh.k_near_neigh_estimator(
            input_data=targ_data["output"],
            model=inform_knn["model"],
            nondetect_val=np.nan,
            hdf5_groupname="",
        )
    
        # add estimated redshifts to a dictionary to store them
        estimated_photoz[(nb_min, nb_max)] = knn_estimated


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 76 training and 26 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.075 and numneigh=6
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer
    Inserting handle into data store.  input: None, KNearNeighEstimator
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x558d0f30eb10>, 'bestsig': np.float64(0.075), 'nneigh': 6, 'truezs': array([0.89386749, 1.28715646, 0.13669944, 0.52842361, 0.44714141,
           0.82454646, 0.51238316, 0.88228261, 0.76767957, 0.45275044,
           0.95481533, 0.5167731 , 0.58168089, 0.31295252, 0.55683708,
           0.97304213, 0.72943944, 0.65232599, 1.61402755, 0.19236302,
           0.56882685, 0.58158028, 1.02507162, 0.57718736, 0.51131564,
           0.51764619, 0.61269331, 0.92453974, 0.46544194, 1.35575664,
           1.25665736, 0.95025933, 0.65654165, 0.90821153, 0.89423275,
           1.464293  , 0.69924277, 0.38057172, 1.21358752, 0.77823561,
           0.76485178, 0.3500061 , 0.30414927, 0.47388995, 1.27942359,
           1.28357577, 1.22945547, 0.82936347, 0.73571932, 0.95288533,
           1.01996362, 1.10498297, 0.47896004, 0.66169471, 0.31951833,
           0.19057477, 0.31050837, 0.44918132, 0.83133233, 0.68544328,
           0.51041174, 1.40825975, 0.47210896, 0.86399877, 0.67155504,
           0.57147968, 0.1458056 , 1.38526881, 0.26492941, 0.26390779,
           0.4937464 , 0.72961587, 0.21930182, 0.33107889, 1.54982591,
           0.67002618, 0.23729324, 0.5662998 , 0.51530981, 1.63489866,
           0.66059667, 0.89940292, 1.01260579, 1.02117932, 1.25851142,
           0.86582661, 0.69498348, 0.2046473 , 0.34921026, 0.3559624 ,
           0.06207943, 0.95071113, 0.19316709, 0.64692485, 0.32656479,
           0.49142575, 0.98890441, 0.34545076, 0.70189863, 0.17718768,
           0.55186069, 0.6297785 ]), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 250
    Process 0 estimating PZ PDF for rows 0 - 250
    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 76 training and 26 validation samples
    finding best fit sigma and NNeigh...
    
    
    
    best fit values are sigma=0.075 and numneigh=6
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer
    Inserting handle into data store.  input: None, KNearNeighEstimator
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x558d17e5c270>, 'bestsig': np.float64(0.075), 'nneigh': 6, 'truezs': array([0.89386749, 1.28715646, 0.13669944, 0.52842361, 0.44714141,
           0.82454646, 0.51238316, 0.88228261, 0.76767957, 0.45275044,
           0.95481533, 0.5167731 , 0.58168089, 0.31295252, 0.55683708,
           0.97304213, 0.72943944, 0.65232599, 1.61402755, 0.19236302,
           0.56882685, 0.58158028, 1.02507162, 0.57718736, 0.51131564,
           0.51764619, 0.61269331, 0.92453974, 0.46544194, 1.35575664,
           1.25665736, 0.95025933, 0.65654165, 0.90821153, 0.89423275,
           1.464293  , 0.69924277, 0.38057172, 1.21358752, 0.77823561,
           0.76485178, 0.3500061 , 0.30414927, 0.47388995, 1.27942359,
           1.28357577, 1.22945547, 0.82936347, 0.73571932, 0.95288533,
           1.01996362, 1.10498297, 0.47896004, 0.66169471, 0.31951833,
           0.19057477, 0.31050837, 0.44918132, 0.83133233, 0.68544328,
           0.51041174, 1.40825975, 0.47210896, 0.86399877, 0.67155504,
           0.57147968, 0.1458056 , 1.38526881, 0.26492941, 0.26390779,
           0.4937464 , 0.72961587, 0.21930182, 0.33107889, 1.54982591,
           0.67002618, 0.23729324, 0.5662998 , 0.51530981, 1.63489866,
           0.66059667, 0.89940292, 1.01260579, 1.02117932, 1.25851142,
           0.86582661, 0.69498348, 0.2046473 , 0.34921026, 0.3559624 ,
           0.06207943, 0.95071113, 0.19316709, 0.64692485, 0.32656479,
           0.49142575, 0.98890441, 0.34545076, 0.70189863, 0.17718768,
           0.55186069, 0.6297785 ]), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 250
    Process 0 estimating PZ PDF for rows 0 - 250


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator
    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 76 training and 26 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.075 and numneigh=6
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer
    Inserting handle into data store.  input: None, KNearNeighEstimator
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x558d225b98c0>, 'bestsig': np.float64(0.075), 'nneigh': 6, 'truezs': array([0.89386749, 1.28715646, 0.13669944, 0.52842361, 0.44714141,
           0.82454646, 0.51238316, 0.88228261, 0.76767957, 0.45275044,
           0.95481533, 0.5167731 , 0.58168089, 0.31295252, 0.55683708,
           0.97304213, 0.72943944, 0.65232599, 1.61402755, 0.19236302,
           0.56882685, 0.58158028, 1.02507162, 0.57718736, 0.51131564,
           0.51764619, 0.61269331, 0.92453974, 0.46544194, 1.35575664,
           1.25665736, 0.95025933, 0.65654165, 0.90821153, 0.89423275,
           1.464293  , 0.69924277, 0.38057172, 1.21358752, 0.77823561,
           0.76485178, 0.3500061 , 0.30414927, 0.47388995, 1.27942359,
           1.28357577, 1.22945547, 0.82936347, 0.73571932, 0.95288533,
           1.01996362, 1.10498297, 0.47896004, 0.66169471, 0.31951833,
           0.19057477, 0.31050837, 0.44918132, 0.83133233, 0.68544328,
           0.51041174, 1.40825975, 0.47210896, 0.86399877, 0.67155504,
           0.57147968, 0.1458056 , 1.38526881, 0.26492941, 0.26390779,
           0.4937464 , 0.72961587, 0.21930182, 0.33107889, 1.54982591,
           0.67002618, 0.23729324, 0.5662998 , 0.51530981, 1.63489866,
           0.66059667, 0.89940292, 1.01260579, 1.02117932, 1.25851142,
           0.86582661, 0.69498348, 0.2046473 , 0.34921026, 0.3559624 ,
           0.06207943, 0.95071113, 0.19316709, 0.64692485, 0.32656479,
           0.49142575, 0.98890441, 0.34545076, 0.70189863, 0.17718768,
           0.55186069, 0.6297785 ]), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 250
    Process 0 estimating PZ PDF for rows 0 - 250
    Inserting handle into data store.  output: inprogress_output.hdf5, KNearNeighEstimator


.. parsed-literal::

    Inserting handle into data store.  input: None, KNearNeighInformer
    split into 76 training and 26 validation samples
    finding best fit sigma and NNeigh...


.. parsed-literal::

    
    
    
    best fit values are sigma=0.075 and numneigh=6
    
    
    
    Inserting handle into data store.  model: inprogress_model.pkl, KNearNeighInformer
    Inserting handle into data store.  input: None, KNearNeighEstimator
    Inserting handle into data store.  model: {'kdtree': <sklearn.neighbors._kd_tree.KDTree object at 0x558d088a4460>, 'bestsig': np.float64(0.075), 'nneigh': 6, 'truezs': array([0.89386749, 1.28715646, 0.13669944, 0.52842361, 0.44714141,
           0.82454646, 0.51238316, 0.88228261, 0.76767957, 0.45275044,
           0.95481533, 0.5167731 , 0.58168089, 0.31295252, 0.55683708,
           0.97304213, 0.72943944, 0.65232599, 1.61402755, 0.19236302,
           0.56882685, 0.58158028, 1.02507162, 0.57718736, 0.51131564,
           0.51764619, 0.61269331, 0.92453974, 0.46544194, 1.35575664,
           1.25665736, 0.95025933, 0.65654165, 0.90821153, 0.89423275,
           1.464293  , 0.69924277, 0.38057172, 1.21358752, 0.77823561,
           0.76485178, 0.3500061 , 0.30414927, 0.47388995, 1.27942359,
           1.28357577, 1.22945547, 0.82936347, 0.73571932, 0.95288533,
           1.01996362, 1.10498297, 0.47896004, 0.66169471, 0.31951833,
           0.19057477, 0.31050837, 0.44918132, 0.83133233, 0.68544328,
           0.51041174, 1.40825975, 0.47210896, 0.86399877, 0.67155504,
           0.57147968, 0.1458056 , 1.38526881, 0.26492941, 0.26390779,
           0.4937464 , 0.72961587, 0.21930182, 0.33107889, 1.54982591,
           0.67002618, 0.23729324, 0.5662998 , 0.51530981, 1.63489866,
           0.66059667, 0.89940292, 1.01260579, 1.02117932, 1.25851142,
           0.86582661, 0.69498348, 0.2046473 , 0.34921026, 0.3559624 ,
           0.06207943, 0.95071113, 0.19316709, 0.64692485, 0.32656479,
           0.49142575, 0.98890441, 0.34545076, 0.70189863, 0.17718768,
           0.55186069, 0.6297785 ]), 'only_colors': False}, KNearNeighEstimator
    Process 0 running estimator on chunk 0 - 250
    Process 0 estimating PZ PDF for rows 0 - 250
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
    print(estimated_photoz[(3, 7)])


.. parsed-literal::

    {'output': Ensemble(the_class=mixmod,shape=(250, 6))}


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
    print(estimated_photoz[(3, 7)]["output"].metadata)


.. parsed-literal::

    {'pdf_name': array([b'mixmod'], dtype='|S6'), 'pdf_version': array([0])}


.. code:: ipython3

    # this is the actual distribution data of that output Ensemble, which contains
    # the data points that describe each photometric redshift probability distribution
    print(estimated_photoz[(3, 7)]["output"].objdata)


.. parsed-literal::

    {'weights': array([[0.2918513 , 0.16278321, 0.15516035, 0.14697529, 0.12250347,
            0.12072638],
           [0.34602373, 0.15792845, 0.15367406, 0.13880513, 0.10205082,
            0.10151783],
           [0.1793383 , 0.17367583, 0.16761759, 0.16358461, 0.15814113,
            0.15764253],
           ...,
           [0.22545074, 0.17418616, 0.15100546, 0.15088091, 0.14982755,
            0.14864918],
           [0.183507  , 0.17294289, 0.1688335 , 0.1637988 , 0.16218283,
            0.14873497],
           [0.17188943, 0.17161809, 0.16714341, 0.16374974, 0.16315386,
            0.16244548]], shape=(250, 6)), 'stds': array([[0.075, 0.075, 0.075, 0.075, 0.075, 0.075],
           [0.075, 0.075, 0.075, 0.075, 0.075, 0.075],
           [0.075, 0.075, 0.075, 0.075, 0.075, 0.075],
           ...,
           [0.075, 0.075, 0.075, 0.075, 0.075, 0.075],
           [0.075, 0.075, 0.075, 0.075, 0.075, 0.075],
           [0.075, 0.075, 0.075, 0.075, 0.075, 0.075]], shape=(250, 6)), 'means': array([[0.86582661, 1.63489866, 0.95071113, 0.82936347, 0.76767957,
            1.38526881],
           [0.66059667, 0.57718736, 0.4937464 , 0.55683708, 0.6297785 ,
            0.51131564],
           [0.52842361, 0.44918132, 0.3500061 , 0.5167731 , 0.49142575,
            0.65232599],
           ...,
           [0.30414927, 0.67002618, 0.31951833, 0.19236302, 0.17718768,
            0.95071113],
           [1.21358752, 1.02507162, 0.95481533, 0.97304213, 1.22945547,
            1.10498297],
           [0.82454646, 1.61402755, 0.95481533, 1.464293  , 1.40825975,
            0.90821153]], shape=(250, 6))}


Typically the ancillary data table includes a photo-z point estimate
derived from the PDFs, by default this is the mode of the distribution,
called ‘zmode’ in the ancillary dictionary below:

.. code:: ipython3

    # this is the ancillary dictionary of the output Ensemble, which in this case
    # contains the zmode, redshift, and distribution type
    print(estimated_photoz[(3, 7)]["output"].ancil)


.. parsed-literal::

    {'zmode': array([[0.86],
           [0.6 ],
           [0.5 ],
           [0.99],
           [0.49],
           [0.33],
           [0.26],
           [0.21],
           [0.52],
           [0.71],
           [0.9 ],
           [0.97],
           [0.2 ],
           [0.91],
           [0.87],
           [0.21],
           [0.5 ],
           [0.49],
           [0.66],
           [0.48],
           [0.99],
           [0.65],
           [1.26],
           [0.95],
           [0.19],
           [0.89],
           [0.89],
           [0.71],
           [0.52],
           [1.26],
           [0.66],
           [1.04],
           [0.34],
           [0.48],
           [0.94],
           [0.9 ],
           [0.89],
           [0.62],
           [0.91],
           [1.26],
           [0.64],
           [0.27],
           [0.55],
           [0.98],
           [0.91],
           [1.32],
           [0.21],
           [0.49],
           [0.98],
           [1.29],
           [1.03],
           [0.94],
           [0.89],
           [1.31],
           [0.51],
           [0.74],
           [0.9 ],
           [1.27],
           [0.31],
           [0.32],
           [1.25],
           [0.49],
           [0.49],
           [0.82],
           [0.29],
           [0.5 ],
           [0.93],
           [0.96],
           [0.56],
           [1.4 ],
           [0.89],
           [1.29],
           [1.26],
           [0.91],
           [0.54],
           [1.28],
           [0.37],
           [0.21],
           [0.81],
           [0.57],
           [0.46],
           [1.24],
           [0.38],
           [0.54],
           [0.52],
           [1.31],
           [1.26],
           [0.7 ],
           [1.25],
           [0.89],
           [1.32],
           [0.11],
           [0.25],
           [0.85],
           [0.5 ],
           [0.19],
           [1.37],
           [1.04],
           [0.2 ],
           [0.55],
           [0.64],
           [1.29],
           [1.3 ],
           [0.24],
           [1.26],
           [0.54],
           [0.5 ],
           [0.27],
           [1.02],
           [0.56],
           [0.93],
           [0.67],
           [0.58],
           [1.29],
           [1.39],
           [0.96],
           [0.54],
           [0.95],
           [0.19],
           [0.67],
           [0.24],
           [0.51],
           [0.89],
           [0.16],
           [0.81],
           [0.33],
           [0.9 ],
           [0.52],
           [1.28],
           [0.54],
           [1.57],
           [0.49],
           [0.26],
           [0.88],
           [0.99],
           [0.64],
           [0.91],
           [1.44],
           [0.15],
           [0.67],
           [0.35],
           [0.45],
           [0.9 ],
           [1.27],
           [0.69],
           [0.31],
           [1.59],
           [0.96],
           [0.69],
           [0.8 ],
           [0.49],
           [1.23],
           [1.  ],
           [0.68],
           [0.89],
           [0.17],
           [1.28],
           [1.59],
           [1.52],
           [0.45],
           [0.77],
           [1.25],
           [0.9 ],
           [0.9 ],
           [0.53],
           [0.89],
           [1.26],
           [0.89],
           [0.31],
           [0.93],
           [0.71],
           [0.75],
           [1.29],
           [1.28],
           [0.49],
           [0.58],
           [1.04],
           [0.46],
           [0.59],
           [0.28],
           [0.9 ],
           [0.72],
           [1.26],
           [0.63],
           [0.48],
           [0.36],
           [0.72],
           [0.53],
           [0.21],
           [1.27],
           [1.41],
           [1.3 ],
           [1.24],
           [0.68],
           [1.26],
           [0.57],
           [0.92],
           [0.49],
           [0.54],
           [1.3 ],
           [0.31],
           [1.01],
           [0.94],
           [1.56],
           [0.72],
           [0.5 ],
           [1.45],
           [1.44],
           [0.66],
           [1.26],
           [0.83],
           [0.27],
           [0.92],
           [0.51],
           [0.22],
           [0.14],
           [0.23],
           [0.5 ],
           [0.54],
           [0.89],
           [0.85],
           [0.67],
           [0.63],
           [0.31],
           [0.2 ],
           [0.21],
           [0.5 ],
           [0.25],
           [0.14],
           [0.66],
           [1.29],
           [1.29],
           [0.75],
           [0.36],
           [0.51],
           [1.36],
           [0.52],
           [0.53],
           [1.26],
           [0.35],
           [0.68],
           [0.71],
           [0.23],
           [0.96],
           [0.54],
           [1.24],
           [0.93],
           [0.27],
           [1.  ],
           [0.9 ]]), 'redshift': 0      0.750908
    1      0.648564
    2      0.275019
    3      1.727078
    4      0.499235
             ...   
    245    2.012556
    246    0.665149
    247    0.349604
    248    1.134209
    249    1.306138
    Name: redshift, Length: 250, dtype: float32, 'distribution_type': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0])}


Converting an Ensemble output to an array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The nice thing about the ``Ensemble`` is that you don’t actually need to
access these dictionaries at all. You can just use the ``.pdf()``
method, which calculates the value(s) of the distribution(s) at any
redshift(s) you provide, and works the same for all types of
``Ensemble``. Here we’ll use it to get an array of PDF values at a set
of redshift grid points:

.. code:: ipython3

    # create the points to evaluate the PDFs at 
    output_zgrid = np.linspace(0,3,200) 
    # evaluate all the PDFs at the given redshifts
    output_photoz_pdfvals = estimated_photoz[(3,7)]["output"].pdf(output_zgrid) 
    output_photoz_pdfvals




.. parsed-literal::

    array([[1.15754366e-023, 8.87790685e-023, 6.53942688e-022, ...,
            1.38554284e-069, 3.79355537e-071, 9.97530282e-073],
           [3.61745159e-010, 1.33981747e-009, 4.76744227e-009, ...,
            2.53324875e-206, 5.09421019e-209, 9.83850425e-212],
           [1.66503378e-005, 4.17023108e-005, 1.00320426e-004, ...,
            3.84594827e-208, 7.56442514e-211, 1.42889926e-213],
           ...,
           [7.92526025e-002, 1.27105035e-001, 1.96014660e-001, ...,
            3.25708058e-158, 1.42528887e-160, 5.99005037e-163],
           [5.98838839e-036, 7.59927868e-035, 9.26255103e-034, ...,
            1.02128266e-117, 9.43007558e-120, 8.36264441e-122],
           [5.18996060e-027, 4.63566589e-026, 3.97661412e-025, ...,
            9.92196455e-072, 2.56880683e-073, 6.38731620e-075]],
          shape=(250, 200))



This array now contains the PDF values of the photo-z probability
distributions for each galaxy, where each row is a distribution, and
each column is the value of the distribution at one of the redshift grid
points in ``output_zgrid``.

Now let’s plot some of these estimated photo-z probability
distributions:

.. code:: ipython3

    xvals = np.linspace(0, 3, 200)  # we want to cover the whole available redshift space
    plt.plot(xvals, estimated_photoz[(3, 7)]["output"][0].pdf(xvals), label="(3,7)")
    plt.plot(xvals, estimated_photoz[(2, 6)]["output"][0].pdf(xvals), label="(2,6)")
    plt.plot(xvals, estimated_photoz[(2, 8)]["output"][0].pdf(xvals), label="(2,8)")
    plt.plot(xvals, estimated_photoz[(4, 9)]["output"][0].pdf(xvals), label="(4,9)")
    
    plt.legend(loc="best", title="(min,max) neighbours")
    plt.xlabel("redshift")
    plt.ylabel("p(z)")




.. parsed-literal::

    Text(0, 0.5, 'p(z)')




.. image:: Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_files/Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_69_1.png


You can see that the distributions are all quite similar to each other.
Some of the runs have little to no difference in their photometric
redshift distribution for this galaxy, and any differences that do exist
are typically small. This makes sense, since the algorithm is averaging
out a set of neighbours, and one more or less shouldn’t significantly
change that for each individual galaxy.

Now let’s summarize these distributions, so we can get a sense of how
the whole distribution of redshift distributions changes with the
different parameters. There are a number of summarizing algorithms. Here
we’ll use two of the most basic:

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

    naive_dict = {}
    point_est_dict = {}
    
    for nb_min, nb_max in nb_params:
        # summarize the distributions using point estimate and naive stack summarizers
        point_estimate_ens = ri.estimation.algos.point_est_hist.point_est_hist_summarizer(
            input_data=estimated_photoz[(nb_min, nb_max)]["output"]
        )
        point_est_dict[(nb_min, nb_max)] = point_estimate_ens
        naive_stack_ens = ri.estimation.algos.naive_stack.naive_stack_summarizer(
            input_data=estimated_photoz[(nb_min, nb_max)]["output"]
        )
        naive_dict[(nb_min, nb_max)] = naive_stack_ens


.. parsed-literal::

    Inserting handle into data store.  input: None, PointEstHistSummarizer
    Process 0 running estimator on chunk 0 - 250


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  input: None, NaiveStackSummarizer
    Process 0 running estimator on chunk 0 - 250
    Inserting handle into data store.  output: inprogress_output.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  input: None, PointEstHistSummarizer
    Process 0 running estimator on chunk 0 - 250


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  input: None, NaiveStackSummarizer
    Process 0 running estimator on chunk 0 - 250
    Inserting handle into data store.  output: inprogress_output.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  input: None, PointEstHistSummarizer
    Process 0 running estimator on chunk 0 - 250


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  input: None, NaiveStackSummarizer
    Process 0 running estimator on chunk 0 - 250
    Inserting handle into data store.  output: inprogress_output.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  input: None, PointEstHistSummarizer
    Process 0 running estimator on chunk 0 - 250


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, PointEstHistSummarizer
    Inserting handle into data store.  input: None, NaiveStackSummarizer
    Process 0 running estimator on chunk 0 - 250


.. parsed-literal::

    Inserting handle into data store.  output: inprogress_output.hdf5, NaiveStackSummarizer
    Inserting handle into data store.  single_NZ: inprogress_single_NZ.hdf5, NaiveStackSummarizer


Now let’s take a look at the output dictionaries for both these
functions for one of the distributions:

.. code:: ipython3

    print(point_est_dict[(3, 7)])
    print(naive_dict[(3, 7)])


.. parsed-literal::

    {'output': Ensemble(the_class=hist,shape=(1000, 301)), 'single_NZ': Ensemble(the_class=hist,shape=(1, 301))}
    {'output': Ensemble(the_class=interp,shape=(1000, 302)), 'single_NZ': Ensemble(the_class=interp,shape=(1, 302))}


These functions output ``Ensembles``, just like the KNN estimation
algorithm. However, they output two separate ``Ensembles``: the
‘single_NZ’ one contains just one distribution, the actual stacked
distribution that has been created. The ‘output’ one contains a number
of bootstrapped distributions, to make further analysis easier.

We’re going to focus on the ‘single_NZ’ distribution here. We’ll start
by plotting the point estimate summarized distributions for all of the
runs, which are histograms:

.. code:: ipython3

    # get bin centers and widths
    bin_width = (
        point_est_dict[(3, 7)]["single_NZ"].metadata["bins"][1]
        - point_est_dict[(3, 7)]["single_NZ"].metadata["bins"][0]
    )
    bin_centers = (
        point_est_dict[(3, 7)]["single_NZ"].metadata["bins"][:-1]
        + point_est_dict[(3, 7)]["single_NZ"].metadata["bins"][1:]
    ) / 2
    
    # plot both histograms to compare
    plt.bar(
        bin_centers,
        point_est_dict[(3, 7)]["single_NZ"].objdata["pdfs"],
        width=bin_width,
        alpha=0.7,
        label="(3,7)",
    )
    plt.bar(
        bin_centers,
        point_est_dict[(2, 6)]["single_NZ"].objdata["pdfs"],
        width=bin_width,
        alpha=0.7,
        label="(2,6)",
    )
    plt.bar(
        bin_centers,
        point_est_dict[(2, 8)]["single_NZ"].objdata["pdfs"],
        width=bin_width,
        alpha=0.7,
        label="(2,8)",
    )
    plt.bar(
        bin_centers,
        point_est_dict[(4, 9)]["single_NZ"].objdata["pdfs"],
        width=bin_width,
        alpha=0.7,
        label="(4,9)",
    )
    
    plt.legend(loc="best", title="(min, max) neighbours")
    plt.xlabel("redshift")
    plt.ylabel("N(z)")




.. parsed-literal::

    Text(0, 0.5, 'N(z)')




.. image:: Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_files/Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_76_1.png


You can see that the distributions are not all the same, but the
variations that do exist are small.

Let’s plot the summarized distributions from the Naive Stacking
algorithm, which are smoothed distributions since it stacked the actual
distributions instead of point estimates:

.. code:: ipython3

    # Plot of naive stack summarized distribution
    plt.plot(
        naive_dict[(3, 7)]["single_NZ"].metadata["xvals"],
        naive_dict[(3, 7)]["single_NZ"].objdata["yvals"],
        label="(3,7)",
    )
    plt.plot(
        naive_dict[(2, 6)]["single_NZ"].metadata["xvals"],
        naive_dict[(2, 6)]["single_NZ"].objdata["yvals"],
        label="(2,6)",
    )
    plt.plot(
        naive_dict[(2, 8)]["single_NZ"].metadata["xvals"],
        naive_dict[(2, 8)]["single_NZ"].objdata["yvals"],
        label="(2,8)",
    )
    plt.plot(
        naive_dict[(4, 9)]["single_NZ"].metadata["xvals"],
        naive_dict[(4, 9)]["single_NZ"].objdata["yvals"],
        label="(4,9)",
    )
    
    plt.legend(loc="best", title="(min, max) neighbours")
    plt.xlabel("redshift")
    plt.ylabel("N(z)")




.. parsed-literal::

    Text(0, 0.5, 'N(z)')




.. image:: Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_files/Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_78_1.png


Similar to the histograms above, the summarized distributions of all the
galaxy photometric redshift probability density functions have some
slight differences but are overall quite similar.

4. Use the ``Evaluator`` stage to calculate some metrics
--------------------------------------------------------

Now that we have a sense of how the distributions of the estimated
photometric redshift probability distributions differ, let’s get a
little more technical. We’ll use the `evaluation
stage <https://rail-hub.readthedocs.io/en/latest/source/rail_stages/creation.html>`__
to calculate some metrics for each of the distributions of redshifts.
For a more detailed look at the available metrics and how to use them,
take a look at the ``Evaluation_by_Type.ipynb`` notebook.

| Here are the metrics we’ll calculate: 1. The `Brier
  score <https://en.wikipedia.org/wiki/Brier_score>`__, which assesses
  the accuracy of probabilistic predictions. The lower the score, the
  better the predictions.
| 2. The `Conditional Density Estimation
  loss <https://vitaliset.github.io/conditional-density-estimation/>`__,
  which is the averaged squared loss between the true and predicted
  conditional probability density functions. The lower the score, the
  better the predicted probability density, in this case, the
  photometric redshift distributions.

For the evaluation metrics, in general we need the estimated redshift
distributions, and the actual redshifts – these are the pre-degradation
redshifts from our initially sampled distribution.

.. code:: ipython3

    eval_dict = {}
    
    for nb_min, nb_max in nb_params:
        ### Evaluate the results
        evaluator_stage_dict = dict(
            metrics=["cdeloss", "brier"],
            _random_state=None,
            metric_config={
                # the limits of the redshifts to evaluate the distributions on
                "brier": {"limits": (0, 3.1)},
            },
        )
    
        the_eval = ri.evaluation.dist_to_point_evaluator.dist_to_point_evaluator(
            data=estimated_photoz[(nb_min, nb_max)]["output"],
            truth=targ_data_orig["output"],
            **evaluator_stage_dict,
            hdf5_groupname="",
        )
    
        # put the evaluation results in a dictionary so we have them
        eval_dict[(nb_min, nb_max)] = the_eval


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
    0    0.750908   27.164078   26.775993   26.225727   25.507996   25.336931   
    1    0.648564   23.150772   22.366415   21.491493   20.731236   20.491596   
    2    0.275019   28.322945   27.592419   27.075672   26.847256   26.854797   
    3    1.727078   28.105497   26.135040   25.419762   24.767265   24.165056   
    4    0.499235   28.817802   27.137247   25.664852   25.014309   24.740934   
    ..        ...         ...         ...         ...         ...         ...   
    245  2.012556   30.255859   28.493664   27.708603   27.206402   26.577536   
    246  0.665149   26.606745   26.018583   25.270943   24.608408   24.401495   
    247  0.349604   27.350212   26.747803   26.110224   25.893808   25.810492   
    248  1.134209   28.412712   27.714722   26.864374   26.204229   25.371664   
    249  1.306138   28.965940   28.081882   27.370880   26.612272   25.999018   
    
         mag_y_lsst  
    0     25.276443  
    1     20.287109  
    2     26.788618  
    3     23.801785  
    4     24.513580  
    ..          ...  
    245   26.203823  
    246   24.269236  
    247   25.664831  
    248   25.018898  
    249   25.375439  
    
    [250 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.750908   27.164078   26.775993   26.225727   25.507996   25.336931   
    1    0.648564   23.150772   22.366415   21.491493   20.731236   20.491596   
    2    0.275019   28.322945   27.592419   27.075672   26.847256   26.854797   
    3    1.727078   28.105497   26.135040   25.419762   24.767265   24.165056   
    4    0.499235   28.817802   27.137247   25.664852   25.014309   24.740934   
    ..        ...         ...         ...         ...         ...         ...   
    245  2.012556   30.255859   28.493664   27.708603   27.206402   26.577536   
    246  0.665149   26.606745   26.018583   25.270943   24.608408   24.401495   
    247  0.349604   27.350212   26.747803   26.110224   25.893808   25.810492   
    248  1.134209   28.412712   27.714722   26.864374   26.204229   25.371664   
    249  1.306138   28.965940   28.081882   27.370880   26.612272   25.999018   
    
         mag_y_lsst  
    0     25.276443  
    1     20.287109  
    2     26.788618  
    3     23.801785  
    4     24.513580  
    ..          ...  
    245   26.203823  
    246   24.269236  
    247   25.664831  
    248   25.018898  
    249   25.375439  
    
    [250 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.750908   27.164078   26.775993   26.225727   25.507996   25.336931   
    1    0.648564   23.150772   22.366415   21.491493   20.731236   20.491596   
    2    0.275019   28.322945   27.592419   27.075672   26.847256   26.854797   
    3    1.727078   28.105497   26.135040   25.419762   24.767265   24.165056   
    4    0.499235   28.817802   27.137247   25.664852   25.014309   24.740934   
    ..        ...         ...         ...         ...         ...         ...   
    245  2.012556   30.255859   28.493664   27.708603   27.206402   26.577536   
    246  0.665149   26.606745   26.018583   25.270943   24.608408   24.401495   
    247  0.349604   27.350212   26.747803   26.110224   25.893808   25.810492   
    248  1.134209   28.412712   27.714722   26.864374   26.204229   25.371664   
    249  1.306138   28.965940   28.081882   27.370880   26.612272   25.999018   
    
         mag_y_lsst  
    0     25.276443  
    1     20.287109  
    2     26.788618  
    3     23.801785  
    4     24.513580  
    ..          ...  
    245   26.203823  
    246   24.269236  
    247   25.664831  
    248   25.018898  
    249   25.375439  
    
    [250 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  input: None, DistToPointEvaluator
    Inserting handle into data store.  truth:      redshift  mag_u_lsst  mag_g_lsst  mag_r_lsst  mag_i_lsst  mag_z_lsst  \
    0    0.750908   27.164078   26.775993   26.225727   25.507996   25.336931   
    1    0.648564   23.150772   22.366415   21.491493   20.731236   20.491596   
    2    0.275019   28.322945   27.592419   27.075672   26.847256   26.854797   
    3    1.727078   28.105497   26.135040   25.419762   24.767265   24.165056   
    4    0.499235   28.817802   27.137247   25.664852   25.014309   24.740934   
    ..        ...         ...         ...         ...         ...         ...   
    245  2.012556   30.255859   28.493664   27.708603   27.206402   26.577536   
    246  0.665149   26.606745   26.018583   25.270943   24.608408   24.401495   
    247  0.349604   27.350212   26.747803   26.110224   25.893808   25.810492   
    248  1.134209   28.412712   27.714722   26.864374   26.204229   25.371664   
    249  1.306138   28.965940   28.081882   27.370880   26.612272   25.999018   
    
         mag_y_lsst  
    0     25.276443  
    1     20.287109  
    2     26.788618  
    3     23.801785  
    4     24.513580  
    ..          ...  
    245   26.203823  
    246   24.269236  
    247   25.664831  
    248   25.018898  
    249   25.375439  
    
    [250 rows x 7 columns], DistToPointEvaluator
    Requested metrics: ['cdeloss', 'brier']
    Inserting handle into data store.  output: inprogress_output.hdf5, DistToPointEvaluator
    Inserting handle into data store.  summary: inprogress_summary.hdf5, DistToPointEvaluator
    Inserting handle into data store.  single_distribution_summary: inprogress_single_distribution_summary.hdf5, DistToPointEvaluator


Now let’s take a look at the metrics we calculated, and compare them.
The code below just selects the one dictionary output per run that we
want to look at, to make the dictionary a little easier to read.

.. code:: ipython3

    results_dict = {key: val["summary"] for key, val in eval_dict.items()}
    
    print(results_dict)


.. parsed-literal::

    {(3, 7): {'cdeloss': array([-1.21461823]), 'brier': array([172.23436667])}, (2, 6): {'cdeloss': array([-1.21461823]), 'brier': array([172.23436667])}, (2, 8): {'cdeloss': array([-1.21461823]), 'brier': array([172.23436667])}, (4, 9): {'cdeloss': array([-1.21461823]), 'brier': array([172.23436667])}}


We can see that the metrics are actually quite similar across most of
the runs. Typically, when there is a variation it’s because the actual
number of neighbours chosen is different. This doesn’t always happen
just because you change the limits, hence the lack of variation across
some of these runs. As mentioned above, smaller metrics are better – in
the case of CDE loss that can mean more negative. So those that have the
lower CDE loss and Brier scores are the runs that are better estimates

5. Using multiprocessing
------------------------

Let’s say we wanted to do the same as above but with a lot more
parameters. We can use the python
```multiprocessing`` <https://docs.python.org/3/library/multiprocessing.html#module-multiprocessing>`__
module to run the whole loop concurrently, and speed up the process a
little. To do this, we need to turn our loop above into its own
function, which takes a the tuple of parameters as an input, and outputs
the data that we want to keep.

.. code:: ipython3

    def estimate_photoz(nb_lims):
        """A function to estimate photo-zs using the KNN algorithm, given a minimum
        and maximum number of nearest neighbours. It will then evaluate the performance"""
    
        # train the informer
        inform_knn = ri.estimation.algos.k_nearneigh.k_near_neigh_informer(
            training_data=calib_data["output"],
            nondetect_val=np.nan,
            model="bpz.pkl",
            hdf5_groupname="",
            nneigh_min=nb_lims[0],
            nneigh_max=nb_lims[1],
        )
        # estimate redshifts
        knn_estimated = ri.estimation.algos.k_nearneigh.k_near_neigh_estimator(
            input_data=targ_data["output"],
            model=inform_knn["model"],
            nondetect_val=np.nan,
            hdf5_groupname="",
        )
    
        ### Evaluate the results
        evaluator_stage_dict = dict(
            metrics=["cdeloss", "brier"],
            _random_state=None,
            metric_config={
                "brier": {"limits": (0, 3.1)},
            },
        )
    
        the_eval = ri.evaluation.dist_to_point_evaluator.dist_to_point_evaluator(
            data=knn_estimated["output"],
            truth=targ_data_orig["output"],
            **evaluator_stage_dict,
            hdf5_groupname="",
        )
    
        # summarize the distributions using point estimate and naive stack summarizers
        point_estimate_ens = ri.estimation.algos.point_est_hist.point_est_hist_summarizer(
            input_data=knn_estimated["output"]
        )
        naive_stack_ens = ri.estimation.algos.naive_stack.naive_stack_summarizer(
            input_data=knn_estimated["output"]
        )
    
        return nb_lims, the_eval, point_estimate_ens, naive_stack_ens

Now we need to set up the parameters that we’ll be iterating over –
we’ll add another four sets of limits to iterate over, with a larger
range of possible neighbours, to slow it down a bit. We also need to set
up the dictionaries for all the outputs we’ll be storing. We’ll just
store the summarized distributions and evaluator results here, since
we’re looking for the differences, not the actual redshifts.

.. code:: ipython3

    from multiprocessing.pool import ThreadPool
    
    # set up parameters to iterate over and dictionaries to store data
    nb_params = [(3, 7), (2, 6), (2, 8), (4, 9), (5, 10), (1, 9), (2, 9), (3, 10)]
    eval_dict_lg = {}
    naive_dict_lg = {}
    point_est_dict_lg = {}

Now let’s actually run our loop.

We’re going to capture the output into the ``iter_out_2`` variable using
the ``%%capture`` magic command, since running it with this many
parameters is going to produce more output than we really need. The
``%%timeit -o`` magic command times how long the cell takes to run, and
stores that result in ``iter_out_2``.

The ``ThreadPool`` class takes the number of processes to use as a
parameter, and lets us use the
```imap_unordered`` <https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.imap_unordered>`__
utility that accepts the function we’d like to run, as well as the
parameters to run it on. Calling ``imap_unordered(func, [a, b, c])``
returns the iterable ``[func(a), func(b), func(c)]`` – though the
ordering of the result is not guaranteed.

We’ll try the parallelization with 2 cores first and see how fast it is:

.. code:: ipython3

    %%capture iter_out_2
    %%timeit -o
    pool = ThreadPool(2) # use 2 cores
    for result in pool.imap_unordered(estimate_photoz, nb_params):
        # store the outputs in the dictionaries
        eval_dict_lg[result[0]] = result[1]
        point_est_dict_lg[result[0]] = result[2]
        naive_dict_lg[result[0]] = result[3]


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. code:: ipython3

    # print out the first entry of iter_out_2, which is the %%timeit result, stored with the -o option
    iter_out_2.outputs[0]




.. parsed-literal::

    <TimeitResult : 7.51 s +- 51.1 ms per loop (mean +- std. dev. of 7 runs, 1 loop each)>



If you want to take a look at the code output, you can do so by
uncommenting the line below.

.. code:: ipython3

    # show the entire captured output
    # iter_out_2.show()

Now let’s try re-running on the same set of parameters, but this time
with four cores. We’ll time this and compare it with the time it took
with two cores. As before, we need to set up some dictionaries to store
the data in:

.. code:: ipython3

    eval_dict_lg_2 = {}
    naive_dict_lg_2 = {}
    point_est_dict_lg_2 = {}

.. code:: ipython3

    %%capture iter_out_4
    %%timeit -o
    pool = ThreadPool(4) # use 4 cores
    for result in pool.imap_unordered(estimate_photoz, nb_params):
        eval_dict_lg_2[result[0]] = result[1]
        point_est_dict_lg_2[result[0]] = result[2]
        naive_dict_lg_2[result[0]] = result[3]


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. parsed-literal::

    WARNING:root:Input predictions do not sum to 1.


.. code:: ipython3

    iter_out_4.outputs[0]




.. parsed-literal::

    <TimeitResult : 7.69 s +- 44.7 ms per loop (mean +- std. dev. of 7 runs, 1 loop each)>



We can see it’s actually faster with two cores than four, just because
the overhead of assigning the work to each core is still dominating over
the actual work being done. If given a larger parameter set, or more
data, then the four core parallelization may be worth it.

Comparing results
~~~~~~~~~~~~~~~~~

We can use the dictionaries of data we stored to compare the results
we’ve gotten. We’ll follow the same method as above, so see there for
more explanation if needed.

First we’ll take a look at the dictionary of evaluation metrics we
calculated, and compare the results for the different parameters:

.. code:: ipython3

    results_dict_lg = {key: val["summary"] for key, val in eval_dict_lg.items()}
    
    print(results_dict_lg)


.. parsed-literal::

    {(2, 6): {'cdeloss': array([-1.21461823]), 'brier': array([172.23436667])}, (3, 7): {'cdeloss': array([-1.21461823]), 'brier': array([172.23436667])}, (2, 8): {'cdeloss': array([-1.21461823]), 'brier': array([172.23436667])}, (4, 9): {'cdeloss': array([-1.21461823]), 'brier': array([172.23436667])}, (5, 10): {'cdeloss': array([-1.21461823]), 'brier': array([172.23436667])}, (1, 9): {'cdeloss': array([-1.21461823]), 'brier': array([172.23436667])}, (2, 9): {'cdeloss': array([-1.21461823]), 'brier': array([172.23436667])}, (3, 10): {'cdeloss': array([-1.21461823]), 'brier': array([172.23436667])}}


We can see that as before, most of the distributions have similar
metrics. Let’s plot the summarized distributions of redshift estimates
from two different runs with different evalutation metrics to see how
the differences have manifested:

.. code:: ipython3

    # Plot of naive stack summarized distribution
    plt.plot(
        naive_dict_lg[(3, 7)]["single_NZ"].metadata["xvals"],
        naive_dict_lg[(3, 7)]["single_NZ"].objdata["yvals"],
        label="(3,7)",
    )
    plt.plot(
        naive_dict_lg[(5, 10)]["single_NZ"].metadata["xvals"],
        naive_dict_lg[(5, 10)]["single_NZ"].objdata["yvals"],
        label="(5,10)",
    )
    
    plt.legend(loc="best", title="(min, max) neighbours")
    plt.xlabel("redshift")
    plt.ylabel("p(z)")




.. parsed-literal::

    Text(0, 0.5, 'p(z)')




.. image:: Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_files/Estimating_Redshifts_and_Comparing_Results_for_Different_Parameters_102_1.png


As expected, the differences are relatively small, since the differences
in the evaluation metrics are relatively small.

Next Steps
----------

This is clearly a toy problem with a small number of galaxies, but
presumably you may want to do something similar with a larger dataset
and a more intensive algorithm. If you are planning to use very large
datasets, we recommend running using a pipeline instead of the
interactive notebook method (see instructions
`here <https://rail-hub.readthedocs.io/en/latest/source/user_guide/pipeline_usage.html>`__).
This method is made for production-level use, and so should be able to
operate more efficiently.

If you’d like to explore a little more with RAIL, we recommend taking a
look at the
`Exploring_the_Effects_of_Degraders <https://rail-hub.readthedocs.io/projects/rail-notebooks/en/latest/interactive_examples/rendered/creation_examples/Exploring_the_Effects_of_Degraders.html>`__
notebook, which takes a more in-depth look at the creation of datasets
and the effect the training data set has on the estimated redshift
distributions.
