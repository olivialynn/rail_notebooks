Run_Pipe, example of running a rail pipeline
============================================

author: Eric Charles last run successfully: April 26, 2023

This notbook shows how to: 1. Load that pipeline from a saved yaml file
4. Run the loaded pipeline

.. code:: ipython3

    import os
    import ceci
    from rail.core.utils import find_rail_file
    flow_file = find_rail_file('examples_data/goldenspike_data/data/pretrained_flow.pkl')
    os.environ['FLOWDIR'] = os.path.dirname(flow_file)

.. code:: ipython3

    p = ceci.Pipeline.read('pipe_example.yml')


.. parsed-literal::

    No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)


.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.12/x64/lib/python3.10/site-packages/rail/examples_data/goldenspike_data/data/pretrained_flow.pkl, flow_engine_test


.. code:: ipython3

    p.run()


.. parsed-literal::

    
    Executing flow_engine_test
    Command is:
    OMP_NUM_THREADS=1   python3 -m ceci rail.creation.engines.flowEngine.FlowCreator   --model=/opt/hostedtoolcache/Python/3.10.12/x64/lib/python3.10/site-packages/rail/examples_data/goldenspike_data/data/pretrained_flow.pkl   --name=flow_engine_test   --config=pipe_example_config.yml   --output=./output_flow_engine_test.pq 
    Output writing to ./flow_engine_test.out
    
    Job flow_engine_test has completed successfully!
    
    Executing lsst_error_model_test
    Command is:
    OMP_NUM_THREADS=1   python3 -m ceci rail.creation.degradation.lsst_error_model.LSSTErrorModel   --input=./output_flow_engine_test.pq   --name=lsst_error_model_test   --config=pipe_example_config.yml   --output=./output_lsst_error_model_test.pq 
    Output writing to ./lsst_error_model_test.out
    
    Job lsst_error_model_test has completed successfully!
    
    Executing col_remapper_test
    Command is:
    OMP_NUM_THREADS=1   python3 -m ceci rail.core.utilStages.ColumnMapper   --input=./output_lsst_error_model_test.pq   --name=col_remapper_test   --config=pipe_example_config.yml   --output=./output_col_remapper_test.pq 
    Output writing to ./col_remapper_test.out
    
    Job col_remapper_test has completed successfully!
    
    Executing table_conv_test
    Command is:
    OMP_NUM_THREADS=1   python3 -m ceci rail.core.utilStages.TableConverter   --input=./output_col_remapper_test.pq   --name=table_conv_test   --config=pipe_example_config.yml   --output=./output_table_conv_test.hdf5 
    Output writing to ./table_conv_test.out
    
    Job table_conv_test has completed successfully!




.. parsed-literal::

    0



Yep, that’s it.
~~~~~~~~~~~~~~~

