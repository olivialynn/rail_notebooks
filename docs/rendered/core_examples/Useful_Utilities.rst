Useful Utilities
================

**Authors:** Olivia Lynn

**Last Run Successfully:** September 20, 2023

This is a notebook that contains various utilities that may be used when
working with RAIL.

Setting Things Up
-----------------

.. code:: ipython3

    import rail

Listing imported stages (1/2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s list out our currently imported stages. Right now, this will only
be what we get by importing ``rail`` and ``rail.stages``.

.. code:: ipython3

    import rail.stages
    for val in rail.core.stage.RailStage.pipeline_stages.values():
        print(val[0])


.. parsed-literal::

    <class 'rail.estimation.estimator.CatEstimator'>
    <class 'rail.estimation.classifier.PZClassifier'>
    <class 'rail.estimation.algos.naive_stack.NaiveStackInformer'>
    <class 'rail.estimation.algos.naive_stack.NaiveStackSummarizer'>
    <class 'rail.estimation.algos.naive_stack.NaiveStackMaskedSummarizer'>
    <class 'rail.estimation.algos.random_gauss.RandomGaussInformer'>
    <class 'rail.estimation.algos.random_gauss.RandomGaussEstimator'>
    <class 'rail.estimation.algos.point_est_hist.PointEstHistInformer'>
    <class 'rail.estimation.algos.point_est_hist.PointEstHistSummarizer'>
    <class 'rail.estimation.algos.point_est_hist.PointEstHistMaskedSummarizer'>
    <class 'rail.estimation.algos.train_z.TrainZInformer'>
    <class 'rail.estimation.algos.train_z.TrainZEstimator'>
    <class 'rail.estimation.algos.var_inf.VarInfStackInformer'>
    <class 'rail.estimation.algos.var_inf.VarInfStackSummarizer'>
    <class 'rail.estimation.algos.uniform_binning.UniformBinningClassifier'>
    <class 'rail.estimation.algos.equal_count.EqualCountClassifier'>
    <class 'rail.estimation.algos.true_nz.TrueNZHistogrammer'>
    <class 'rail.creation.noisifier.Noisifier'>
    <class 'rail.creation.degraders.addRandom.AddColumnOfRandom'>
    <class 'rail.creation.selector.Selector'>
    <class 'rail.creation.degraders.quantityCut.QuantityCut'>
    <class 'rail.evaluation.evaluator.OldEvaluator'>
    <class 'rail.evaluation.dist_to_dist_evaluator.DistToDistEvaluator'>
    <class 'rail.evaluation.dist_to_point_evaluator.DistToPointEvaluator'>
    <class 'rail.evaluation.point_to_point_evaluator.PointToPointEvaluator'>
    <class 'rail.evaluation.single_evaluator.SingleEvaluator'>
    <class 'rail.tools.table_tools.ColumnMapper'>
    <class 'rail.tools.table_tools.RowSelector'>
    <class 'rail.tools.table_tools.TableConverter'>


Import and attach all
~~~~~~~~~~~~~~~~~~~~~

Using ``rail.stages.import_and_attach_all()`` lets you import all
packages within the RAIL ecosystem at once.

This kind of blanket import is a useful shortcut; however, it will be
slower than specific imports, as you will import things you’ll never
need.

As of such, ``import_and_attach_all`` is recommended for new users and
those who wish to do rapid exploration with notebooks; pipelines
designed to be run at scale would generally prefer lightweight, specific
imports.

.. code:: ipython3

    import rail
    import rail.stages
    rail.stages.import_and_attach_all()


.. parsed-literal::

    Imported rail.astro_tools
    Imported rail.bpz
    Imported rail.cmnn
    Imported rail.core


.. parsed-literal::

    Imported rail.dsps
    Imported rail.flexzboost


.. parsed-literal::

    Install FSPS with the following commands:
    pip uninstall fsps
    git clone --recursive https://github.com/dfm/python-fsps.git
    cd python-fsps
    python -m pip install .
    export SPS_HOME=$(pwd)/src/fsps/libfsps
    
    Imported rail.fsps
    Imported rail.gpz
    Imported rail.hub
    Imported rail.interfaces


.. parsed-literal::

    Imported rail.pzflow
    Imported rail.sklearn
    Imported rail.som
    Imported rail.stages
    Attached 12 base classes and 81 fully formed stages to rail.stages


Now that we’ve attached all available stages to rail.stages, we can use
``from rail.stages import *`` to let us omit prefixes.

To see this in action:

.. code:: ipython3

    # with prefix
    
    print(rail.tools.table_tools.ColumnMapper)


.. parsed-literal::

    <class 'rail.tools.table_tools.ColumnMapper'>


.. code:: ipython3

    # without prefix
    
    try:
        print(ColumnMapper)
    except Exception as e: 
        print(e)


.. parsed-literal::

    name 'ColumnMapper' is not defined


.. code:: ipython3

    from rail.stages import *

.. code:: ipython3

    print(ColumnMapper)


.. parsed-literal::

    <class 'rail.tools.table_tools.ColumnMapper'>


Listing imported stages (2/2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, let’s try listing imported stages again, and notice how many more
we get.

.. code:: ipython3

    for val in rail.core.stage.RailStage.pipeline_stages.values():
        print(val[0])


.. parsed-literal::

    <class 'rail.estimation.estimator.CatEstimator'>
    <class 'rail.estimation.classifier.PZClassifier'>
    <class 'rail.estimation.algos.naive_stack.NaiveStackInformer'>
    <class 'rail.estimation.algos.naive_stack.NaiveStackSummarizer'>
    <class 'rail.estimation.algos.naive_stack.NaiveStackMaskedSummarizer'>
    <class 'rail.estimation.algos.random_gauss.RandomGaussInformer'>
    <class 'rail.estimation.algos.random_gauss.RandomGaussEstimator'>
    <class 'rail.estimation.algos.point_est_hist.PointEstHistInformer'>
    <class 'rail.estimation.algos.point_est_hist.PointEstHistSummarizer'>
    <class 'rail.estimation.algos.point_est_hist.PointEstHistMaskedSummarizer'>
    <class 'rail.estimation.algos.train_z.TrainZInformer'>
    <class 'rail.estimation.algos.train_z.TrainZEstimator'>
    <class 'rail.estimation.algos.var_inf.VarInfStackInformer'>
    <class 'rail.estimation.algos.var_inf.VarInfStackSummarizer'>
    <class 'rail.estimation.algos.uniform_binning.UniformBinningClassifier'>
    <class 'rail.estimation.algos.equal_count.EqualCountClassifier'>
    <class 'rail.estimation.algos.true_nz.TrueNZHistogrammer'>
    <class 'rail.creation.noisifier.Noisifier'>
    <class 'rail.creation.degraders.addRandom.AddColumnOfRandom'>
    <class 'rail.creation.selector.Selector'>
    <class 'rail.creation.degraders.quantityCut.QuantityCut'>
    <class 'rail.evaluation.evaluator.OldEvaluator'>
    <class 'rail.evaluation.dist_to_dist_evaluator.DistToDistEvaluator'>
    <class 'rail.evaluation.dist_to_point_evaluator.DistToPointEvaluator'>
    <class 'rail.evaluation.point_to_point_evaluator.PointToPointEvaluator'>
    <class 'rail.evaluation.single_evaluator.SingleEvaluator'>
    <class 'rail.tools.table_tools.ColumnMapper'>
    <class 'rail.tools.table_tools.RowSelector'>
    <class 'rail.tools.table_tools.TableConverter'>
    <class 'rail.creation.degraders.grid_selection.GridSelection'>
    <class 'rail.creation.degraders.observing_condition_degrader.ObsCondition'>
    <class 'rail.creation.degraders.spectroscopic_degraders.LineConfusion'>
    <class 'rail.creation.degraders.spectroscopic_degraders.InvRedshiftIncompleteness'>
    <class 'rail.creation.degraders.spectroscopic_selections.SpecSelection'>
    <class 'rail.creation.degraders.spectroscopic_selections.SpecSelection_GAMA'>
    <class 'rail.creation.degraders.spectroscopic_selections.SpecSelection_BOSS'>
    <class 'rail.creation.degraders.spectroscopic_selections.SpecSelection_DEEP2'>
    <class 'rail.creation.degraders.spectroscopic_selections.SpecSelection_VVDSf02'>
    <class 'rail.creation.degraders.spectroscopic_selections.SpecSelection_zCOSMOS'>
    <class 'rail.creation.degraders.spectroscopic_selections.SpecSelection_HSC'>
    <class 'rail.creation.degraders.photometric_errors.PhotoErrorModel'>
    <class 'rail.creation.degraders.photometric_errors.LSSTErrorModel'>
    <class 'rail.creation.degraders.photometric_errors.RomanErrorModel'>
    <class 'rail.creation.degraders.photometric_errors.EuclidErrorModel'>
    <class 'rail.creation.degraders.unrec_bl_model.UnrecBlModel'>
    <class 'rail.tools.photometry_tools.HyperbolicSmoothing'>
    <class 'rail.tools.photometry_tools.HyperbolicMagnitudes'>
    <class 'rail.tools.photometry_tools.LSSTFluxToMagConverter'>
    <class 'rail.tools.photometry_tools.DustMapBase'>
    <class 'rail.tools.photometry_tools.Dereddener'>
    <class 'rail.tools.photometry_tools.Reddener'>
    <class 'rail.estimation.algos.bpz_lite.BPZliteInformer'>
    <class 'rail.estimation.algos.bpz_lite.BPZliteEstimator'>
    <class 'rail.estimation.algos.cmnn.CMNNInformer'>
    <class 'rail.estimation.algos.cmnn.CMNNEstimator'>
    <class 'rail.creation.engines.dsps_photometry_creator.DSPSPhotometryCreator'>
    <class 'rail.creation.engines.dsps_sed_modeler.DSPSSingleSedModeler'>
    <class 'rail.creation.engines.dsps_sed_modeler.DSPSPopulationSedModeler'>
    <class 'rail.estimation.algos.flexzboost.FlexZBoostInformer'>
    <class 'rail.estimation.algos.flexzboost.FlexZBoostEstimator'>
    <class 'rail.creation.engines.fsps_photometry_creator.FSPSPhotometryCreator'>
    <class 'rail.creation.engines.fsps_sed_modeler.FSPSSedModeler'>
    <class 'rail.estimation.algos.gpz.GPzInformer'>
    <class 'rail.estimation.algos.gpz.GPzEstimator'>
    <class 'rail.creation.engines.flowEngine.FlowModeler'>
    <class 'rail.creation.engines.flowEngine.FlowCreator'>
    <class 'rail.creation.engines.flowEngine.FlowPosterior'>
    <class 'rail.estimation.algos.pzflow_nf.PZFlowInformer'>
    <class 'rail.estimation.algos.pzflow_nf.PZFlowEstimator'>
    <class 'rail.estimation.algos.k_nearneigh.KNearNeighInformer'>
    <class 'rail.estimation.algos.k_nearneigh.KNearNeighEstimator'>
    <class 'rail.estimation.algos.sklearn_neurnet.SklNeurNetInformer'>
    <class 'rail.estimation.algos.sklearn_neurnet.SklNeurNetEstimator'>
    <class 'rail.estimation.algos.nz_dir.NZDirInformer'>
    <class 'rail.estimation.algos.nz_dir.NZDirSummarizer'>
    <class 'rail.estimation.algos.random_forest.RandomForestInformer'>
    <class 'rail.estimation.algos.random_forest.RandomForestClassifier'>
    <class 'rail.estimation.algos.minisom_som.MiniSOMInformer'>
    <class 'rail.estimation.algos.minisom_som.MiniSOMSummarizer'>
    <class 'rail.estimation.algos.somoclu_som.SOMocluInformer'>
    <class 'rail.estimation.algos.somoclu_som.SOMocluSummarizer'>


We can use this list of imported stages to browse for specifics, such as
looking through our available estimators.

**Note:** this will only filter through what you’ve imported, so if you
haven’t imported everything above, this will not be a complete list of
all estimators available in RAIL.

.. code:: ipython3

    for val in rail.core.stage.RailStage.pipeline_stages.values():
        if issubclass(val[0], rail.estimation.estimator.CatEstimator):
            print(val[0])


.. parsed-literal::

    <class 'rail.estimation.estimator.CatEstimator'>
    <class 'rail.estimation.algos.random_gauss.RandomGaussEstimator'>
    <class 'rail.estimation.algos.train_z.TrainZEstimator'>
    <class 'rail.estimation.algos.bpz_lite.BPZliteEstimator'>
    <class 'rail.estimation.algos.cmnn.CMNNEstimator'>
    <class 'rail.estimation.algos.flexzboost.FlexZBoostEstimator'>
    <class 'rail.estimation.algos.gpz.GPzEstimator'>
    <class 'rail.estimation.algos.pzflow_nf.PZFlowEstimator'>
    <class 'rail.estimation.algos.k_nearneigh.KNearNeighEstimator'>
    <class 'rail.estimation.algos.sklearn_neurnet.SklNeurNetEstimator'>
    <class 'rail.estimation.algos.nz_dir.NZDirSummarizer'>


Listing keys in the Data Store (1/2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s list out the keys in the Data Store to see what data we have
stored.

First, we must set up the Data Store:

.. code:: ipython3

    DS = rail.core.stage.RailStage.data_store
    DS.__class__.allow_overwrite = True

And because we’ve only just created the store, as you may have guessed,
it is empty.

We’ll come back to this in a bit.

.. code:: ipython3

    DS.keys()




.. parsed-literal::

    dict_keys([])



Finding data files with find_rail_file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We need to define our flow file that we’ll use in our pipeline

If we already know its path, we can just point directly to the file
(relative to the directory that holds our ``rail/`` directory):

.. code:: ipython3

    import os
    from rail.utils.path_utils import RAILDIR
    
    flow_file = os.path.join(
        RAILDIR, "rail/examples_data/goldenspike_data/data/pretrained_flow.pkl"
    )

But if we aren’t sure where our file is (or we’re just feeling lazy) we
can use ``find_rail_file``.

This is especially helpful in cases where our installation is spread
out, and some rail modules are located separately from others.

.. code:: ipython3

    from rail.utils.path_utils import find_rail_file
    
    flow_file = find_rail_file('examples_data/goldenspike_data/data/pretrained_flow.pkl')

We can set our FLOWDIR based on the location of our flow file, too.

.. code:: ipython3

    os.environ['FLOWDIR'] = os.path.dirname(flow_file)

.. code:: ipython3

    # Now, we have to set up some other variables for our pipeline:
    import numpy as np
    
    bands = ["u", "g", "r", "i", "z", "y"]
    band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"mag_{band}_lsst_err": f"mag_err_{band}_lsst" for band in bands}
    post_grid = [float(x) for x in np.linspace(0.0, 5, 21)]

Creating the Pipeline
---------------------

.. code:: ipython3

    import ceci

.. code:: ipython3

    # Make some stages
    
    flow_engine_test = FlowCreator.make_stage(
        name="flow_engine_test", model=flow_file, n_samples=50
    )
    col_remapper_test = ColumnMapper.make_stage(
        name="col_remapper_test", hdf5_groupname="", columns=rename_dict
    )
    #flow_engine_test.sample(6, seed=0).data


.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/rail/examples_data/goldenspike_data/data/pretrained_flow.pkl, flow_engine_test


.. code:: ipython3

    # Add the stages to the pipeline
    
    pipe = ceci.Pipeline.interactive()
    stages = [flow_engine_test, col_remapper_test]
    for stage in stages:
        pipe.add_stage(stage)

.. code:: ipython3

    # Connect stages
    
    col_remapper_test.connect_input(flow_engine_test)


.. parsed-literal::

    Inserting handle into data store.  output_flow_engine_test: inprogress_output_flow_engine_test.pq, flow_engine_test


Introspecting the Pipeline
--------------------------

Listing keys in the Data Store (2/2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that we have a some data in the Data Store, let’s take another look
at it.

.. code:: ipython3

    DS.keys()




.. parsed-literal::

    dict_keys(['model', 'output_flow_engine_test'])



Getting names of stages in the pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    pipe.stage_names




.. parsed-literal::

    ['flow_engine_test', 'col_remapper_test']



Getting the configuration of a particular stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s take a look a the config of the first stage we just listed above.

.. code:: ipython3

    pipe.flow_engine_test.config




.. parsed-literal::

    StageConfig{output_mode:default,n_samples:50,seed:12345,name:flow_engine_test,model:/opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/rail/examples_data/goldenspike_data/data/pretrained_flow.pkl,config:None,}



Updating a configuration value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can update config values even after the stage has been created. Let’s
give it a try.

.. code:: ipython3

    pipe.flow_engine_test.config.update(seed=42)
    
    pipe.flow_engine_test.config




.. parsed-literal::

    StageConfig{output_mode:default,n_samples:50,seed:42,name:flow_engine_test,model:/opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/rail/examples_data/goldenspike_data/data/pretrained_flow.pkl,config:None,}



Listing stage outputs (as both tags and aliased tags)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s get the list of outputs as ‘tags’.

These are how the stage thinks of the outputs, as a list names
associated to DataHandle types.

.. code:: ipython3

    pipe.flow_engine_test.outputs




.. parsed-literal::

    [('output', rail.core.data.PqHandle)]



We can also get the list of outputs as ‘aliased tags’.

These are how the pipeline thinks of the outputs, as a unique key that
points to a particular file

.. code:: ipython3

    pipe.flow_engine_test._outputs




.. parsed-literal::

    {'output_flow_engine_test': 'output_flow_engine_test.pq'}



Listing all pipeline methods and parameters that can be set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you’d like to take a closer look at what you can do with a pipeline,
use ``dir(pipe)`` to list out available methods and parameters.

.. code:: ipython3

    for item in dir(pipe):
        if '__' not in item:
            print(item)


.. parsed-literal::

    add_stage
    build_config
    build_dag
    build_stage
    callback
    create
    data_registry
    data_registry_lookup
    enqueue_job
    find_all_outputs
    generate_stage_command
    global_config
    initialize
    initialize_stages
    initiate_run
    interactive
    launcher_config
    make_flow_chart
    modules
    ordered_stages
    overall_inputs
    pipeline_files
    pipeline_outputs
    print_stages
    process_overall_inputs
    read
    remove_stage
    run
    run_config
    run_info
    run_jobs
    save
    setup_data_registry
    should_skip_stage
    sleep
    stage_config_data
    stage_execution_config
    stage_names
    stages
    stages_config


Initializing the Pipeline
-------------------------

Toggling resume mode
~~~~~~~~~~~~~~~~~~~~

We can turn ‘resume mode’ on when initializing a pipeline.

Resume mode lets us skip stages that already have output files, so we
don’t have to rerun the same stages as we iterate on a pipeline.

Just add a ``resume=True`` to do so.

.. code:: ipython3

    pipe.initialize(
        dict(model=flow_file), dict(output_dir=".", log_dir=".", resume=True), None
    )




.. parsed-literal::

    (({'flow_engine_test': <Job flow_engine_test>,
       'col_remapper_test': <Job col_remapper_test>},
      [<rail.creation.engines.flowEngine.FlowCreator at 0x7f15986b6620>,
       Stage that applies remaps the following column names in a pandas DataFrame:
       f{str(self.config.columns)}]),
     {'output_dir': '.', 'log_dir': '.', 'resume': True})



Running ``pipe.stages`` should show order of classes, or all the stages
this pipeline will run.

.. code:: ipython3

    pipe.stages




.. parsed-literal::

    [<rail.creation.engines.flowEngine.FlowCreator at 0x7f15986b6620>,
     Stage that applies remaps the following column names in a pandas DataFrame:
     f{str(self.config.columns)}]



Managing notebooks with git
---------------------------

*(thank you to https://stackoverflow.com/a/58004619)*

You can modify your git settings to run a filter over certain files
before they are added to git. This will leave the original file on disk
as-is, but commit the “cleaned” version.

First, add the following to your local ``.git/config`` file (or global
``~/.gitconfig``):

[filter "strip-notebook-output"]
    clean = "jupyter nbconvert --ClearOutputPreprocessor.enabled=True --to=notebook --stdin --stdout --log-level=ERROR"

Then, create a ``.gitattributes`` file in your directory with notebooks
and add the following line:

*.ipynb filter=strip-notebook-output
