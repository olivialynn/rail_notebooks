Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fae48d2b7f0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




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
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>27.804766</td>
          <td>0.941101</td>
          <td>26.488514</td>
          <td>0.137127</td>
          <td>26.024167</td>
          <td>0.080546</td>
          <td>25.150730</td>
          <td>0.060686</td>
          <td>24.698623</td>
          <td>0.077822</td>
          <td>24.084930</td>
          <td>0.101818</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.436614</td>
          <td>0.743200</td>
          <td>27.566755</td>
          <td>0.336013</td>
          <td>26.721633</td>
          <td>0.147992</td>
          <td>26.572472</td>
          <td>0.208580</td>
          <td>25.994316</td>
          <td>0.237228</td>
          <td>25.092222</td>
          <td>0.240613</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.252444</td>
          <td>0.564422</td>
          <td>27.417391</td>
          <td>0.265397</td>
          <td>26.220979</td>
          <td>0.154866</td>
          <td>24.910619</td>
          <td>0.093798</td>
          <td>24.120071</td>
          <td>0.104997</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.105056</td>
          <td>0.507091</td>
          <td>27.135341</td>
          <td>0.210206</td>
          <td>26.242575</td>
          <td>0.157755</td>
          <td>25.590485</td>
          <td>0.169022</td>
          <td>25.021512</td>
          <td>0.226936</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.947096</td>
          <td>0.242089</td>
          <td>26.162454</td>
          <td>0.103310</td>
          <td>25.825562</td>
          <td>0.067575</td>
          <td>25.541881</td>
          <td>0.085770</td>
          <td>25.474578</td>
          <td>0.153088</td>
          <td>24.598643</td>
          <td>0.158872</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>28.136687</td>
          <td>1.145545</td>
          <td>26.480531</td>
          <td>0.136186</td>
          <td>25.408112</td>
          <td>0.046656</td>
          <td>25.081544</td>
          <td>0.057072</td>
          <td>24.719422</td>
          <td>0.079265</td>
          <td>24.801436</td>
          <td>0.188747</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.166243</td>
          <td>0.617666</td>
          <td>27.164006</td>
          <td>0.242640</td>
          <td>26.146344</td>
          <td>0.089700</td>
          <td>25.284718</td>
          <td>0.068339</td>
          <td>24.835375</td>
          <td>0.087794</td>
          <td>24.408574</td>
          <td>0.134927</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.404969</td>
          <td>0.727641</td>
          <td>26.562125</td>
          <td>0.146096</td>
          <td>26.558904</td>
          <td>0.128609</td>
          <td>26.114253</td>
          <td>0.141298</td>
          <td>26.587150</td>
          <td>0.381785</td>
          <td>24.976901</td>
          <td>0.218671</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.582400</td>
          <td>0.401815</td>
          <td>26.118882</td>
          <td>0.099447</td>
          <td>26.230231</td>
          <td>0.096560</td>
          <td>25.779597</td>
          <td>0.105669</td>
          <td>25.573773</td>
          <td>0.166633</td>
          <td>25.651504</td>
          <td>0.376952</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.467811</td>
          <td>0.758765</td>
          <td>26.532456</td>
          <td>0.142416</td>
          <td>26.719370</td>
          <td>0.147705</td>
          <td>26.209909</td>
          <td>0.153404</td>
          <td>25.835494</td>
          <td>0.207870</td>
          <td>25.698445</td>
          <td>0.390924</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




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
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>27.121564</td>
          <td>0.657045</td>
          <td>26.628554</td>
          <td>0.177659</td>
          <td>26.105826</td>
          <td>0.101740</td>
          <td>25.226546</td>
          <td>0.076926</td>
          <td>24.735775</td>
          <td>0.094546</td>
          <td>23.998253</td>
          <td>0.111454</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.842984</td>
          <td>0.212813</td>
          <td>26.645414</td>
          <td>0.162319</td>
          <td>26.555624</td>
          <td>0.241230</td>
          <td>25.575636</td>
          <td>0.195029</td>
          <td>25.784502</td>
          <td>0.482017</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.165953</td>
          <td>0.605609</td>
          <td>29.479611</td>
          <td>1.276811</td>
          <td>26.104934</td>
          <td>0.168945</td>
          <td>25.170440</td>
          <td>0.141085</td>
          <td>24.470190</td>
          <td>0.171201</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.581522</td>
          <td>0.923298</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.032483</td>
          <td>0.239567</td>
          <td>26.308894</td>
          <td>0.209802</td>
          <td>25.642610</td>
          <td>0.219848</td>
          <td>25.042461</td>
          <td>0.287644</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.615121</td>
          <td>0.456026</td>
          <td>26.298746</td>
          <td>0.134018</td>
          <td>26.122969</td>
          <td>0.103311</td>
          <td>26.015542</td>
          <td>0.153105</td>
          <td>25.703719</td>
          <td>0.217141</td>
          <td>25.473788</td>
          <td>0.380644</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.684613</td>
          <td>0.486772</td>
          <td>26.377802</td>
          <td>0.146109</td>
          <td>25.455568</td>
          <td>0.058549</td>
          <td>24.921019</td>
          <td>0.060001</td>
          <td>24.898523</td>
          <td>0.111316</td>
          <td>25.170361</td>
          <td>0.305321</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.606139</td>
          <td>0.174949</td>
          <td>25.874113</td>
          <td>0.083352</td>
          <td>25.241416</td>
          <td>0.078282</td>
          <td>24.689313</td>
          <td>0.091146</td>
          <td>23.955005</td>
          <td>0.107786</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.020305</td>
          <td>0.617066</td>
          <td>26.526713</td>
          <td>0.164761</td>
          <td>26.242832</td>
          <td>0.116131</td>
          <td>26.289972</td>
          <td>0.195726</td>
          <td>26.493086</td>
          <td>0.413656</td>
          <td>25.543726</td>
          <td>0.406307</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.548795</td>
          <td>0.442743</td>
          <td>26.477905</td>
          <td>0.160655</td>
          <td>26.143255</td>
          <td>0.108465</td>
          <td>25.677036</td>
          <td>0.117967</td>
          <td>25.443783</td>
          <td>0.179789</td>
          <td>25.783552</td>
          <td>0.494872</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.300517</td>
          <td>1.354358</td>
          <td>26.641529</td>
          <td>0.181197</td>
          <td>26.875793</td>
          <td>0.199177</td>
          <td>26.097063</td>
          <td>0.165766</td>
          <td>26.334804</td>
          <td>0.365099</td>
          <td>25.710971</td>
          <td>0.460192</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




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
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>27.956177</td>
          <td>1.031447</td>
          <td>26.616872</td>
          <td>0.153137</td>
          <td>26.088636</td>
          <td>0.085268</td>
          <td>25.172562</td>
          <td>0.061881</td>
          <td>24.630583</td>
          <td>0.073291</td>
          <td>23.870382</td>
          <td>0.084340</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.158963</td>
          <td>1.160601</td>
          <td>27.558387</td>
          <td>0.334041</td>
          <td>26.585764</td>
          <td>0.131757</td>
          <td>26.185555</td>
          <td>0.150379</td>
          <td>25.675412</td>
          <td>0.181825</td>
          <td>25.444460</td>
          <td>0.320524</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.091173</td>
          <td>0.613994</td>
          <td>27.776836</td>
          <td>0.421603</td>
          <td>29.518098</td>
          <td>1.223833</td>
          <td>25.882192</td>
          <td>0.125770</td>
          <td>24.984987</td>
          <td>0.108616</td>
          <td>24.257302</td>
          <td>0.128683</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.332313</td>
          <td>0.699874</td>
          <td>26.951441</td>
          <td>0.223259</td>
          <td>26.375534</td>
          <td>0.221021</td>
          <td>25.173208</td>
          <td>0.147250</td>
          <td>25.233603</td>
          <td>0.334076</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.764944</td>
          <td>0.208342</td>
          <td>26.103751</td>
          <td>0.098259</td>
          <td>25.831136</td>
          <td>0.068007</td>
          <td>25.765080</td>
          <td>0.104491</td>
          <td>25.598646</td>
          <td>0.170434</td>
          <td>24.845968</td>
          <td>0.196242</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.782505</td>
          <td>0.965191</td>
          <td>26.308480</td>
          <td>0.125615</td>
          <td>25.383969</td>
          <td>0.049459</td>
          <td>25.110705</td>
          <td>0.063657</td>
          <td>24.762137</td>
          <td>0.089067</td>
          <td>24.992584</td>
          <td>0.239261</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.834051</td>
          <td>0.965833</td>
          <td>26.682020</td>
          <td>0.164158</td>
          <td>26.097537</td>
          <td>0.087360</td>
          <td>25.165852</td>
          <td>0.062590</td>
          <td>24.825335</td>
          <td>0.088469</td>
          <td>24.149109</td>
          <td>0.109544</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.617754</td>
          <td>0.425393</td>
          <td>26.427752</td>
          <td>0.135704</td>
          <td>26.197250</td>
          <td>0.098505</td>
          <td>25.962954</td>
          <td>0.130398</td>
          <td>25.695029</td>
          <td>0.193573</td>
          <td>26.099365</td>
          <td>0.550983</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.789407</td>
          <td>0.503801</td>
          <td>26.199874</td>
          <td>0.117986</td>
          <td>26.240372</td>
          <td>0.109216</td>
          <td>25.935037</td>
          <td>0.136170</td>
          <td>25.404726</td>
          <td>0.161225</td>
          <td>25.577562</td>
          <td>0.395201</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.468057</td>
          <td>0.774583</td>
          <td>26.847475</td>
          <td>0.192471</td>
          <td>26.492717</td>
          <td>0.126193</td>
          <td>25.955352</td>
          <td>0.128216</td>
          <td>25.820423</td>
          <td>0.212999</td>
          <td>25.403999</td>
          <td>0.321639</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
