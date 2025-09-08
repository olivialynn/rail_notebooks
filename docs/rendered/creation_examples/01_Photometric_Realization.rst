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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f5805c76ad0>



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
    0      23.994413  0.011563  0.006033  
    1      25.391064  0.018934  0.011699  
    2      24.304707  0.029622  0.021949  
    3      25.291103  0.067541  0.041539  
    4      25.096743  0.051852  0.049732  
    ...          ...       ...       ...  
    99995  24.737946  0.048326  0.027367  
    99996  24.224169  0.003654  0.002141  
    99997  25.613836  0.018253  0.010538  
    99998  25.274899  0.116899  0.059667  
    99999  25.699642  0.030367  0.019719  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>26.695413</td>
          <td>0.437991</td>
          <td>26.729385</td>
          <td>0.168558</td>
          <td>26.050061</td>
          <td>0.082407</td>
          <td>25.189161</td>
          <td>0.062790</td>
          <td>24.698781</td>
          <td>0.077833</td>
          <td>24.062481</td>
          <td>0.099836</td>
          <td>0.011563</td>
          <td>0.006033</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.331964</td>
          <td>0.278368</td>
          <td>26.808701</td>
          <td>0.159457</td>
          <td>26.114856</td>
          <td>0.141372</td>
          <td>25.707877</td>
          <td>0.186716</td>
          <td>25.649649</td>
          <td>0.376408</td>
          <td>0.018934</td>
          <td>0.011699</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.728963</td>
          <td>0.783216</td>
          <td>27.848163</td>
          <td>0.374305</td>
          <td>26.124256</td>
          <td>0.142521</td>
          <td>24.908352</td>
          <td>0.093611</td>
          <td>24.316969</td>
          <td>0.124641</td>
          <td>0.029622</td>
          <td>0.021949</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.104124</td>
          <td>0.992018</td>
          <td>27.331115</td>
          <td>0.247278</td>
          <td>26.517489</td>
          <td>0.199181</td>
          <td>25.484699</td>
          <td>0.154422</td>
          <td>25.332427</td>
          <td>0.292723</td>
          <td>0.067541</td>
          <td>0.041539</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.983072</td>
          <td>0.249354</td>
          <td>26.004812</td>
          <td>0.089982</td>
          <td>25.947060</td>
          <td>0.075244</td>
          <td>25.808774</td>
          <td>0.108398</td>
          <td>25.615175</td>
          <td>0.172609</td>
          <td>25.279076</td>
          <td>0.280361</td>
          <td>0.051852</td>
          <td>0.049732</td>
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
          <td>27.001963</td>
          <td>0.549485</td>
          <td>26.488846</td>
          <td>0.137166</td>
          <td>25.425325</td>
          <td>0.047375</td>
          <td>25.079696</td>
          <td>0.056979</td>
          <td>24.818920</td>
          <td>0.086531</td>
          <td>24.873721</td>
          <td>0.200590</td>
          <td>0.048326</td>
          <td>0.027367</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.271448</td>
          <td>0.664525</td>
          <td>26.580003</td>
          <td>0.148356</td>
          <td>26.062912</td>
          <td>0.083346</td>
          <td>25.310800</td>
          <td>0.069936</td>
          <td>24.928532</td>
          <td>0.095284</td>
          <td>23.887355</td>
          <td>0.085599</td>
          <td>0.003654</td>
          <td>0.002141</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.820518</td>
          <td>0.950256</td>
          <td>26.628750</td>
          <td>0.154685</td>
          <td>26.380492</td>
          <td>0.110129</td>
          <td>26.577831</td>
          <td>0.209517</td>
          <td>25.693642</td>
          <td>0.184483</td>
          <td>25.427377</td>
          <td>0.315903</td>
          <td>0.018253</td>
          <td>0.010538</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.280054</td>
          <td>0.317091</td>
          <td>26.190199</td>
          <td>0.105845</td>
          <td>25.963767</td>
          <td>0.076363</td>
          <td>25.985071</td>
          <td>0.126371</td>
          <td>26.108523</td>
          <td>0.260585</td>
          <td>25.313746</td>
          <td>0.288342</td>
          <td>0.116899</td>
          <td>0.059667</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.283131</td>
          <td>0.317870</td>
          <td>26.732745</td>
          <td>0.169041</td>
          <td>26.553821</td>
          <td>0.128044</td>
          <td>26.620556</td>
          <td>0.217128</td>
          <td>25.834835</td>
          <td>0.207755</td>
          <td>25.041356</td>
          <td>0.230702</td>
          <td>0.030367</td>
          <td>0.019719</td>
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
          <td>26.754865</td>
          <td>0.505912</td>
          <td>26.782464</td>
          <td>0.202316</td>
          <td>26.098118</td>
          <td>0.101082</td>
          <td>25.232656</td>
          <td>0.077363</td>
          <td>24.581614</td>
          <td>0.082580</td>
          <td>23.944889</td>
          <td>0.106411</td>
          <td>0.011563</td>
          <td>0.006033</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.965461</td>
          <td>0.235722</td>
          <td>26.477010</td>
          <td>0.140572</td>
          <td>26.690506</td>
          <td>0.269596</td>
          <td>25.943893</td>
          <td>0.264858</td>
          <td>25.201380</td>
          <td>0.307145</td>
          <td>0.018934</td>
          <td>0.011699</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.885361</td>
          <td>0.889971</td>
          <td>26.180299</td>
          <td>0.176547</td>
          <td>25.063055</td>
          <td>0.126077</td>
          <td>24.464404</td>
          <td>0.167000</td>
          <td>0.029622</td>
          <td>0.021949</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.169337</td>
          <td>1.884402</td>
          <td>27.002107</td>
          <td>0.221470</td>
          <td>26.153941</td>
          <td>0.174074</td>
          <td>25.501911</td>
          <td>0.185098</td>
          <td>25.468963</td>
          <td>0.382752</td>
          <td>0.067541</td>
          <td>0.041539</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.932447</td>
          <td>1.104706</td>
          <td>26.151629</td>
          <td>0.118873</td>
          <td>25.939052</td>
          <td>0.088656</td>
          <td>25.512936</td>
          <td>0.099863</td>
          <td>25.584078</td>
          <td>0.198019</td>
          <td>24.944064</td>
          <td>0.251125</td>
          <td>0.051852</td>
          <td>0.049732</td>
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
          <td>26.826137</td>
          <td>0.534630</td>
          <td>26.387717</td>
          <td>0.145313</td>
          <td>25.517888</td>
          <td>0.060893</td>
          <td>25.059248</td>
          <td>0.066708</td>
          <td>24.830463</td>
          <td>0.103257</td>
          <td>25.106216</td>
          <td>0.285662</td>
          <td>0.048326</td>
          <td>0.027367</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.436498</td>
          <td>0.811282</td>
          <td>27.030393</td>
          <td>0.248509</td>
          <td>26.020670</td>
          <td>0.094422</td>
          <td>25.201560</td>
          <td>0.075247</td>
          <td>24.888964</td>
          <td>0.108113</td>
          <td>24.221230</td>
          <td>0.135247</td>
          <td>0.003654</td>
          <td>0.002141</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.836296</td>
          <td>1.040082</td>
          <td>26.743343</td>
          <td>0.195857</td>
          <td>26.292382</td>
          <td>0.119805</td>
          <td>26.143234</td>
          <td>0.170809</td>
          <td>26.188345</td>
          <td>0.322552</td>
          <td>25.441579</td>
          <td>0.371359</td>
          <td>0.018253</td>
          <td>0.010538</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.576636</td>
          <td>0.451188</td>
          <td>26.119441</td>
          <td>0.117635</td>
          <td>26.240550</td>
          <td>0.117683</td>
          <td>25.830602</td>
          <td>0.134319</td>
          <td>25.988291</td>
          <td>0.281638</td>
          <td>25.482380</td>
          <td>0.392963</td>
          <td>0.116899</td>
          <td>0.059667</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.986866</td>
          <td>0.598811</td>
          <td>26.840918</td>
          <td>0.212809</td>
          <td>26.214358</td>
          <td>0.112102</td>
          <td>26.239724</td>
          <td>0.185643</td>
          <td>25.593619</td>
          <td>0.198378</td>
          <td>25.471071</td>
          <td>0.380487</td>
          <td>0.030367</td>
          <td>0.019719</td>
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
          <td>26.245924</td>
          <td>0.308791</td>
          <td>26.604306</td>
          <td>0.151629</td>
          <td>26.023043</td>
          <td>0.080560</td>
          <td>25.223762</td>
          <td>0.064826</td>
          <td>24.773212</td>
          <td>0.083212</td>
          <td>24.038037</td>
          <td>0.097837</td>
          <td>0.011563</td>
          <td>0.006033</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.876122</td>
          <td>0.428326</td>
          <td>26.455652</td>
          <td>0.117974</td>
          <td>26.066370</td>
          <td>0.136051</td>
          <td>25.856950</td>
          <td>0.212309</td>
          <td>25.538975</td>
          <td>0.346235</td>
          <td>0.018934</td>
          <td>0.011699</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.393088</td>
          <td>0.627573</td>
          <td>27.643638</td>
          <td>0.321224</td>
          <td>25.949517</td>
          <td>0.123701</td>
          <td>24.943843</td>
          <td>0.097458</td>
          <td>24.077156</td>
          <td>0.102090</td>
          <td>0.029622</td>
          <td>0.021949</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.472115</td>
          <td>0.677348</td>
          <td>27.380926</td>
          <td>0.267511</td>
          <td>26.199004</td>
          <td>0.158412</td>
          <td>25.382469</td>
          <td>0.147170</td>
          <td>25.562948</td>
          <td>0.365196</td>
          <td>0.067541</td>
          <td>0.041539</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.203993</td>
          <td>0.304891</td>
          <td>26.000956</td>
          <td>0.092365</td>
          <td>25.699353</td>
          <td>0.062527</td>
          <td>25.647748</td>
          <td>0.097541</td>
          <td>25.257523</td>
          <td>0.131269</td>
          <td>25.064357</td>
          <td>0.243032</td>
          <td>0.051852</td>
          <td>0.049732</td>
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
          <td>28.129179</td>
          <td>1.151207</td>
          <td>26.401257</td>
          <td>0.129414</td>
          <td>25.470079</td>
          <td>0.050325</td>
          <td>25.007598</td>
          <td>0.054617</td>
          <td>24.928553</td>
          <td>0.097247</td>
          <td>24.395395</td>
          <td>0.136209</td>
          <td>0.048326</td>
          <td>0.027367</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.896293</td>
          <td>0.508819</td>
          <td>26.636417</td>
          <td>0.155720</td>
          <td>26.116482</td>
          <td>0.087384</td>
          <td>25.221292</td>
          <td>0.064613</td>
          <td>25.026059</td>
          <td>0.103798</td>
          <td>24.155248</td>
          <td>0.108288</td>
          <td>0.003654</td>
          <td>0.002141</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.305768</td>
          <td>0.681460</td>
          <td>26.629654</td>
          <td>0.155199</td>
          <td>26.228202</td>
          <td>0.096678</td>
          <td>25.997721</td>
          <td>0.128163</td>
          <td>26.012497</td>
          <td>0.241497</td>
          <td>25.410289</td>
          <td>0.312506</td>
          <td>0.018253</td>
          <td>0.010538</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.389284</td>
          <td>0.369046</td>
          <td>26.092276</td>
          <td>0.106225</td>
          <td>26.151494</td>
          <td>0.099735</td>
          <td>25.942052</td>
          <td>0.135152</td>
          <td>25.827781</td>
          <td>0.227389</td>
          <td>24.918524</td>
          <td>0.230064</td>
          <td>0.116899</td>
          <td>0.059667</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.615758</td>
          <td>0.414430</td>
          <td>26.735751</td>
          <td>0.170734</td>
          <td>26.349098</td>
          <td>0.108095</td>
          <td>26.557742</td>
          <td>0.207848</td>
          <td>25.758994</td>
          <td>0.196591</td>
          <td>25.892517</td>
          <td>0.456878</td>
          <td>0.030367</td>
          <td>0.019719</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
