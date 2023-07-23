RAIL Pipeline: Build, Save, Load, and Run
=========================================

author: Eric Charles

last run successfully: April 26, 2023

This notebook shows how to:

1. Build a simple interactive rail pipeline,

2. Save that pipeline (including configuration) to a yaml file,

3. Load that pipeline from the saved yaml file,

4. Run the loaded pipeline.

.. code:: ipython3

    import os
    import numpy as np
    import ceci
    import rail
    from rail.core.stage import RailStage
    from rail.creation.degradation.spectroscopic_degraders import LineConfusion 
    from rail.creation.degradation.quantityCut import QuantityCut
    from rail.creation.degradation.lsst_error_model import LSSTErrorModel
    from rail.creation.engines.flowEngine import FlowCreator, FlowPosterior
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.core.utilStages import ColumnMapper, TableConverter

We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks; the data store will work
around that and enable us to use data interactively.

When working interactively, we want to allow overwriting data in the
RAIL data store to avoid errors if we re-run cells.

See the ``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

Some configuration setup
~~~~~~~~~~~~~~~~~~~~~~~~

The example pipeline builds some of the RAIL creation functionality into
a pipeline.

Here we are defining:

1. The location of the pretrained PZFlow file used with this example.

2. The bands we will be generating data for.

3. The names of the columns where we will be writing the error
   estimates.

4. The grid of redshifts we use for posterior estimation.

.. code:: ipython3

    from rail.core.utils import RAILDIR
    flow_file = os.path.join(RAILDIR, 'rail/examples_data/goldenspike_data/data/pretrained_flow.pkl')
    bands = ['u','g','r','i','z','y']
    band_dict = {band:f'mag_{band}_lsst' for band in bands}
    rename_dict = {f'mag_{band}_lsst_err':f'mag_err_{band}_lsst' for band in bands}
    post_grid = [float(x) for x in np.linspace(0., 5, 21)]

Define the pipeline stages
~~~~~~~~~~~~~~~~~~~~~~~~~~

The RailStage base class defines the ``make_stage`` “classmethod”
function, which allows us to make a stage of that particular type in a
general way.

Note that that we are passing in the configuration parameters to each
pipeline stage as keyword arguments.

The names of the parameters will depend on the stage type.

A couple of things are important:

1. Each stage should have a unique name. In ``ceci``, stage names
   default to the name of the class (e.g., FlowCreator, or
   LSSTErrorModel); this would be problematic if you wanted two stages
   of the same type in a given pipeline, so be sure to assign each stage
   its own name.

2. At this point, we aren’t actually worrying about the connections
   between the stages.

.. code:: ipython3

    flow_engine_test = FlowCreator.make_stage(name='flow_engine_test', 
                                             model=flow_file, n_samples=50)
          
    lsst_error_model_test = LSSTErrorModel.make_stage(name='lsst_error_model_test',
                                                      bandNames=band_dict)
                    
    col_remapper_test = ColumnMapper.make_stage(name='col_remapper_test', hdf5_groupname='',
                                                columns=rename_dict)
    
    flow_post_test = FlowPosterior.make_stage(name='flow_post_test',
                                              column='redshift', flow=flow_file,
                                              grid=post_grid)
    
    table_conv_test = TableConverter.make_stage(name='table_conv_test', output_format='numpyDict', 
                                                seed=12345)



.. parsed-literal::

    No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)


.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.12/x64/lib/python3.10/site-packages/rail/examples_data/goldenspike_data/data/pretrained_flow.pkl, flow_engine_test


Make the pipeline and add the stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we make an empty interactive pipeline (interactive in the sense
that it will be run locally, rather than using the batch submission
mechanisms built into ``ceci``), and add the stages to that pipeline.

.. code:: ipython3

    pipe = ceci.Pipeline.interactive()
    stages = [flow_engine_test, lsst_error_model_test, col_remapper_test, table_conv_test]
    for stage in stages:
        pipe.add_stage(stage)

Here are some examples of interactive introspection into the pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

I.e., some functions that you can use to figure out what the pipeline is
doing.

.. code:: ipython3

    # Get the names of the stages
    pipe.stage_names




.. parsed-literal::

    ['flow_engine_test',
     'lsst_error_model_test',
     'col_remapper_test',
     'table_conv_test']



.. code:: ipython3

    # Get the configuration of a particular stage
    pipe.flow_engine_test.config




.. parsed-literal::

    StageConfig{output_mode:default,n_samples:50,seed:12345,name:flow_engine_test,model:/opt/hostedtoolcache/Python/3.10.12/x64/lib/python3.10/site-packages/rail/examples_data/goldenspike_data/data/pretrained_flow.pkl,config:None,aliases:{'output': 'output_flow_engine_test'},}



.. code:: ipython3

    # Get the list of outputs 'tags' 
    # These are how the stage thinks of the outputs, as a list names associated to DataHandle types.
    pipe.flow_engine_test.outputs




.. parsed-literal::

    [('output', rail.core.data.PqHandle)]



.. code:: ipython3

    # Get the list of outputs 'aliased tags'
    # These are how the pipeline things of the outputs, as a unique key that points to a particular file
    pipe.flow_engine_test._outputs




.. parsed-literal::

    {'output_flow_engine_test': 'output_flow_engine_test.pq'}



Okay, now let’s connect up the pipeline stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can use the ``RailStage.connect_input`` function to connect one stage
to another. By default, this will connect the output data product called
``output`` for one stage.

.. code:: ipython3

    lsst_error_model_test.connect_input(flow_engine_test)
    col_remapper_test.connect_input(lsst_error_model_test)
    #flow_post_test.connect_input(col_remapper_test, inputTag='input')
    table_conv_test.connect_input(col_remapper_test)


.. parsed-literal::

    Inserting handle into data store.  output_flow_engine_test: inprogress_output_flow_engine_test.pq, flow_engine_test
    Inserting handle into data store.  output_lsst_error_model_test: inprogress_output_lsst_error_model_test.pq, lsst_error_model_test
    Inserting handle into data store.  output_col_remapper_test: inprogress_output_col_remapper_test.pq, col_remapper_test


Initialize the pipeline
~~~~~~~~~~~~~~~~~~~~~~~

This will do a few things:

1. Attach any global pipeline inputs that were not specified in the
   connections above. In our case, the input flow file is pre-existing
   and must be specified as a global input.

2. Specifiy output and logging directories.

3. Optionally, create the pipeline in ‘resume’ mode, where it will
   ignore stages if all of their output already exists.

.. code:: ipython3

    pipe.initialize(dict(model=flow_file), dict(output_dir='.', log_dir='.', resume=False), None)




.. parsed-literal::

    (({'flow_engine_test': <Job flow_engine_test>,
       'lsst_error_model_test': <Job lsst_error_model_test>,
       'col_remapper_test': <Job col_remapper_test>,
       'table_conv_test': <Job table_conv_test>},
      [<rail.creation.engines.flowEngine.FlowCreator at 0x7f379613fdc0>,
       LSSTErrorModel parameters:
       
       Model for bands: mag_u_lsst, mag_g_lsst, mag_r_lsst, mag_i_lsst, mag_z_lsst, mag_y_lsst
       
       Using error type point
       Exposure time = 30.0 s
       Number of years of observations = 10.0
       Mean visits per year per band:
          mag_u_lsst: 5.6, mag_g_lsst: 8.0, mag_r_lsst: 18.4, mag_i_lsst: 18.4, mag_z_lsst: 16.0, mag_y_lsst: 16.0
       Airmass = 1.2
       Irreducible system error = 0.005
       Magnitudes dimmer than 30.0 are set to nan
       gamma for each band:
          mag_u_lsst: 0.038, mag_g_lsst: 0.039, mag_r_lsst: 0.039, mag_i_lsst: 0.039, mag_z_lsst: 0.039, mag_y_lsst: 0.039
       
       The coadded 5-sigma limiting magnitudes are:
       mag_u_lsst: 26.04, mag_g_lsst: 27.29, mag_r_lsst: 27.31, mag_i_lsst: 26.87, mag_z_lsst: 26.23, mag_y_lsst: 25.30
       
       The following single-visit 5-sigma limiting magnitudes are
       calculated using the parameters that follow them:
          mag_u_lsst: 23.83, mag_g_lsst: 24.90, mag_r_lsst: 24.47, mag_i_lsst: 24.03, mag_z_lsst: 23.46, mag_y_lsst: 22.53
       Cm for each band:
          mag_u_lsst: 23.09, mag_g_lsst: 24.42, mag_r_lsst: 24.44, mag_i_lsst: 24.32, mag_z_lsst: 24.16, mag_y_lsst: 23.73
       Median zenith sky brightness in each band:
          mag_u_lsst: 22.99, mag_g_lsst: 22.26, mag_r_lsst: 21.2, mag_i_lsst: 20.48, mag_z_lsst: 19.6, mag_y_lsst: 18.61
       Median zenith seeing FWHM (in arcseconds) for each band:
          mag_u_lsst: 0.81, mag_g_lsst: 0.77, mag_r_lsst: 0.73, mag_i_lsst: 0.71, mag_z_lsst: 0.69, mag_y_lsst: 0.68
       Extinction coefficient for each band:
          mag_u_lsst: 0.491, mag_g_lsst: 0.213, mag_r_lsst: 0.126, mag_i_lsst: 0.096, mag_z_lsst: 0.069, mag_y_lsst: 0.17,
       Stage that applies remaps the following column names in a pandas DataFrame:
       f{str(self.config.columns)},
       <rail.core.utilStages.TableConverter at 0x7f3795fe4610>]),
     {'output_dir': '.', 'log_dir': '.', 'resume': False})



Save the pipeline
~~~~~~~~~~~~~~~~~

This will actually write two files (as this is what ``ceci`` wants)

1. ``pipe_example.yml``, which will have a list of stages, with
   instructions on how to execute the stages (e.g., run this stage in
   parallel on 20 processors). For an interactive pipeline, those
   instructions will be trivial.

2. ``pipe_example_config.yml``, which will have a dictionary of
   configurations for each stage.

.. code:: ipython3

    pipe.save('pipe_saved.yml')

Read the saved pipeline
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    pr = ceci.Pipeline.read('pipe_saved.yml')

Run the newly read pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will actually launch Unix process to individually run each stage of
the pipeline; you can see the commands that are being executed in each
case.

.. code:: ipython3

    pr.run()


.. parsed-literal::

    
    Executing flow_engine_test
    Command is:
    OMP_NUM_THREADS=1   python3 -m ceci rail.creation.engines.flowEngine.FlowCreator   --model=/opt/hostedtoolcache/Python/3.10.12/x64/lib/python3.10/site-packages/rail/examples_data/goldenspike_data/data/pretrained_flow.pkl   --name=flow_engine_test   --config=pipe_saved_config.yml   --output=./output_flow_engine_test.pq 
    Output writing to ./flow_engine_test.out
    
    Job flow_engine_test has completed successfully!
    
    Executing lsst_error_model_test
    Command is:
    OMP_NUM_THREADS=1   python3 -m ceci rail.creation.degradation.lsst_error_model.LSSTErrorModel   --input=./output_flow_engine_test.pq   --name=lsst_error_model_test   --config=pipe_saved_config.yml   --output=./output_lsst_error_model_test.pq 
    Output writing to ./lsst_error_model_test.out
    
    Job lsst_error_model_test has completed successfully!
    
    Executing col_remapper_test
    Command is:
    OMP_NUM_THREADS=1   python3 -m ceci rail.core.utilStages.ColumnMapper   --input=./output_lsst_error_model_test.pq   --name=col_remapper_test   --config=pipe_saved_config.yml   --output=./output_col_remapper_test.pq 
    Output writing to ./col_remapper_test.out
    
    Job col_remapper_test has completed successfully!
    
    Executing table_conv_test
    Command is:
    OMP_NUM_THREADS=1   python3 -m ceci rail.core.utilStages.TableConverter   --input=./output_col_remapper_test.pq   --name=table_conv_test   --config=pipe_saved_config.yml   --output=./output_table_conv_test.hdf5 
    Output writing to ./table_conv_test.out
    
    Job table_conv_test has completed successfully!




.. parsed-literal::

    0


