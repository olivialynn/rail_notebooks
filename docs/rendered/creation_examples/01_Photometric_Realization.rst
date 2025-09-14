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

    <pzflow.flow.Flow at 0x7f7d5bc24070>



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
    0      23.994413  0.178948  0.132303  
    1      25.391064  0.057589  0.034968  
    2      24.304707  0.170229  0.142484  
    3      25.291103  0.079376  0.066848  
    4      25.096743  0.079298  0.045968  
    ...          ...       ...       ...  
    99995  24.737946  0.070099  0.040949  
    99996  24.224169  0.055451  0.043254  
    99997  25.613836  0.025733  0.025427  
    99998  25.274899  0.026456  0.023190  
    99999  25.699642  0.084614  0.078683  
    
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
          <td>26.860643</td>
          <td>0.495597</td>
          <td>26.973115</td>
          <td>0.207062</td>
          <td>25.974718</td>
          <td>0.077106</td>
          <td>25.284787</td>
          <td>0.068343</td>
          <td>24.678917</td>
          <td>0.076480</td>
          <td>24.072769</td>
          <td>0.100740</td>
          <td>0.178948</td>
          <td>0.132303</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.490270</td>
          <td>1.388545</td>
          <td>27.308123</td>
          <td>0.273028</td>
          <td>26.943386</td>
          <td>0.178831</td>
          <td>26.258771</td>
          <td>0.159955</td>
          <td>26.515846</td>
          <td>0.361142</td>
          <td>25.257447</td>
          <td>0.275481</td>
          <td>0.057589</td>
          <td>0.034968</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.016191</td>
          <td>2.465659</td>
          <td>28.280734</td>
          <td>0.519084</td>
          <td>25.983071</td>
          <td>0.126152</td>
          <td>24.909451</td>
          <td>0.093701</td>
          <td>24.452180</td>
          <td>0.140101</td>
          <td>0.170229</td>
          <td>0.142484</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.539846</td>
          <td>1.424539</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.111806</td>
          <td>0.206106</td>
          <td>26.055069</td>
          <td>0.134264</td>
          <td>25.695602</td>
          <td>0.184789</td>
          <td>25.206912</td>
          <td>0.264369</td>
          <td>0.079376</td>
          <td>0.066848</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.688977</td>
          <td>0.435862</td>
          <td>26.170567</td>
          <td>0.104045</td>
          <td>25.844323</td>
          <td>0.068707</td>
          <td>25.491279</td>
          <td>0.082028</td>
          <td>25.699047</td>
          <td>0.185328</td>
          <td>25.011287</td>
          <td>0.225017</td>
          <td>0.079298</td>
          <td>0.045968</td>
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
          <td>27.276947</td>
          <td>0.667044</td>
          <td>26.330770</td>
          <td>0.119628</td>
          <td>25.452412</td>
          <td>0.048528</td>
          <td>24.956868</td>
          <td>0.051091</td>
          <td>24.848011</td>
          <td>0.088775</td>
          <td>24.743429</td>
          <td>0.179711</td>
          <td>0.070099</td>
          <td>0.040949</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.331399</td>
          <td>1.276309</td>
          <td>26.982307</td>
          <td>0.208660</td>
          <td>26.079205</td>
          <td>0.084552</td>
          <td>25.233979</td>
          <td>0.065335</td>
          <td>24.779573</td>
          <td>0.083583</td>
          <td>24.273033</td>
          <td>0.119976</td>
          <td>0.055451</td>
          <td>0.043254</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.692122</td>
          <td>1.537841</td>
          <td>26.562371</td>
          <td>0.146127</td>
          <td>26.489997</td>
          <td>0.121147</td>
          <td>26.407341</td>
          <td>0.181508</td>
          <td>26.037096</td>
          <td>0.245749</td>
          <td>25.851010</td>
          <td>0.439329</td>
          <td>0.025733</td>
          <td>0.025427</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.058647</td>
          <td>0.265252</td>
          <td>26.159640</td>
          <td>0.103056</td>
          <td>26.065059</td>
          <td>0.083504</td>
          <td>25.972117</td>
          <td>0.124960</td>
          <td>25.746710</td>
          <td>0.192934</td>
          <td>25.066769</td>
          <td>0.235607</td>
          <td>0.026456</td>
          <td>0.023190</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.882577</td>
          <td>0.191904</td>
          <td>26.725510</td>
          <td>0.148486</td>
          <td>26.153826</td>
          <td>0.146194</td>
          <td>26.243019</td>
          <td>0.290686</td>
          <td>25.150678</td>
          <td>0.252473</td>
          <td>0.084614</td>
          <td>0.078683</td>
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
          <td>29.660878</td>
          <td>2.518496</td>
          <td>26.765324</td>
          <td>0.213174</td>
          <td>25.878717</td>
          <td>0.090056</td>
          <td>25.200313</td>
          <td>0.081445</td>
          <td>24.721983</td>
          <td>0.100897</td>
          <td>23.899755</td>
          <td>0.110657</td>
          <td>0.178948</td>
          <td>0.132303</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.079457</td>
          <td>0.641084</td>
          <td>27.890602</td>
          <td>0.490858</td>
          <td>26.620819</td>
          <td>0.160097</td>
          <td>26.452788</td>
          <td>0.223147</td>
          <td>26.234853</td>
          <td>0.336773</td>
          <td>25.072427</td>
          <td>0.278591</td>
          <td>0.057589</td>
          <td>0.034968</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.087828</td>
          <td>0.596546</td>
          <td>27.977481</td>
          <td>0.507124</td>
          <td>26.177195</td>
          <td>0.189698</td>
          <td>24.839481</td>
          <td>0.111795</td>
          <td>24.597797</td>
          <td>0.201292</td>
          <td>0.170229</td>
          <td>0.142484</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.633582</td>
          <td>0.928305</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.176957</td>
          <td>0.257656</td>
          <td>26.317903</td>
          <td>0.201405</td>
          <td>25.481626</td>
          <td>0.183254</td>
          <td>24.949228</td>
          <td>0.254447</td>
          <td>0.079376</td>
          <td>0.066848</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.899925</td>
          <td>0.262488</td>
          <td>26.301526</td>
          <td>0.135984</td>
          <td>26.026582</td>
          <td>0.096254</td>
          <td>25.684429</td>
          <td>0.116633</td>
          <td>25.455106</td>
          <td>0.178506</td>
          <td>24.896070</td>
          <td>0.242645</td>
          <td>0.079298</td>
          <td>0.045968</td>
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
          <td>26.930898</td>
          <td>0.578619</td>
          <td>26.496997</td>
          <td>0.160393</td>
          <td>25.471627</td>
          <td>0.058793</td>
          <td>25.087219</td>
          <td>0.068799</td>
          <td>24.985969</td>
          <td>0.118942</td>
          <td>24.676068</td>
          <td>0.201475</td>
          <td>0.070099</td>
          <td>0.040949</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.568567</td>
          <td>0.886752</td>
          <td>26.861113</td>
          <td>0.217560</td>
          <td>26.049011</td>
          <td>0.097608</td>
          <td>25.213703</td>
          <td>0.076721</td>
          <td>25.018902</td>
          <td>0.122065</td>
          <td>23.965936</td>
          <td>0.109281</td>
          <td>0.055451</td>
          <td>0.043254</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.454744</td>
          <td>0.404230</td>
          <td>27.129390</td>
          <td>0.269969</td>
          <td>26.381730</td>
          <td>0.129645</td>
          <td>26.103272</td>
          <td>0.165338</td>
          <td>26.193718</td>
          <td>0.324365</td>
          <td>25.896563</td>
          <td>0.524404</td>
          <td>0.025733</td>
          <td>0.025427</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.952729</td>
          <td>0.584440</td>
          <td>26.218502</td>
          <td>0.125229</td>
          <td>26.183185</td>
          <td>0.109082</td>
          <td>25.869301</td>
          <td>0.135244</td>
          <td>26.205974</td>
          <td>0.327505</td>
          <td>24.710343</td>
          <td>0.205533</td>
          <td>0.026456</td>
          <td>0.023190</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.334465</td>
          <td>0.768539</td>
          <td>26.750266</td>
          <td>0.200688</td>
          <td>26.583330</td>
          <td>0.157261</td>
          <td>26.356369</td>
          <td>0.208871</td>
          <td>25.626803</td>
          <td>0.207905</td>
          <td>25.040927</td>
          <td>0.275334</td>
          <td>0.084614</td>
          <td>0.078683</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.556037</td>
          <td>0.180799</td>
          <td>26.003254</td>
          <td>0.101710</td>
          <td>25.172411</td>
          <td>0.080491</td>
          <td>24.819193</td>
          <td>0.111211</td>
          <td>23.996765</td>
          <td>0.121926</td>
          <td>0.178948</td>
          <td>0.132303</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.344204</td>
          <td>0.339968</td>
          <td>27.735417</td>
          <td>0.392314</td>
          <td>26.706560</td>
          <td>0.150394</td>
          <td>26.316533</td>
          <td>0.173182</td>
          <td>26.058184</td>
          <td>0.257074</td>
          <td>25.552728</td>
          <td>0.358664</td>
          <td>0.057589</td>
          <td>0.034968</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.243631</td>
          <td>2.160970</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.738291</td>
          <td>1.520353</td>
          <td>25.742945</td>
          <td>0.132713</td>
          <td>25.393460</td>
          <td>0.182467</td>
          <td>24.168493</td>
          <td>0.141632</td>
          <td>0.170229</td>
          <td>0.142484</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.969202</td>
          <td>0.482151</td>
          <td>26.991438</td>
          <td>0.198801</td>
          <td>26.205204</td>
          <td>0.163747</td>
          <td>25.362129</td>
          <td>0.148542</td>
          <td>25.904763</td>
          <td>0.485728</td>
          <td>0.079376</td>
          <td>0.066848</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.597291</td>
          <td>0.848738</td>
          <td>26.051488</td>
          <td>0.098211</td>
          <td>25.911343</td>
          <td>0.076925</td>
          <td>25.640157</td>
          <td>0.098875</td>
          <td>25.434983</td>
          <td>0.155864</td>
          <td>25.526099</td>
          <td>0.358978</td>
          <td>0.079298</td>
          <td>0.045968</td>
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
          <td>28.433134</td>
          <td>1.371830</td>
          <td>26.524857</td>
          <td>0.146710</td>
          <td>25.384380</td>
          <td>0.047689</td>
          <td>25.050010</td>
          <td>0.058050</td>
          <td>25.024681</td>
          <td>0.108124</td>
          <td>24.881192</td>
          <td>0.210495</td>
          <td>0.070099</td>
          <td>0.040949</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.156535</td>
          <td>0.624670</td>
          <td>27.112436</td>
          <td>0.238782</td>
          <td>26.136059</td>
          <td>0.091827</td>
          <td>25.211301</td>
          <td>0.066274</td>
          <td>24.819556</td>
          <td>0.089438</td>
          <td>24.268442</td>
          <td>0.123544</td>
          <td>0.055451</td>
          <td>0.043254</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.696092</td>
          <td>0.440529</td>
          <td>27.001782</td>
          <td>0.213631</td>
          <td>26.262450</td>
          <td>0.100204</td>
          <td>26.542913</td>
          <td>0.205282</td>
          <td>26.091380</td>
          <td>0.259070</td>
          <td>25.368756</td>
          <td>0.303945</td>
          <td>0.025733</td>
          <td>0.025427</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.267376</td>
          <td>0.315555</td>
          <td>26.078361</td>
          <td>0.096676</td>
          <td>26.177720</td>
          <td>0.092982</td>
          <td>25.768293</td>
          <td>0.105546</td>
          <td>25.372334</td>
          <td>0.141359</td>
          <td>25.006416</td>
          <td>0.225951</td>
          <td>0.026456</td>
          <td>0.023190</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.473132</td>
          <td>0.388975</td>
          <td>26.631682</td>
          <td>0.166446</td>
          <td>26.834658</td>
          <td>0.176689</td>
          <td>26.255236</td>
          <td>0.173495</td>
          <td>25.881098</td>
          <td>0.233564</td>
          <td>25.159346</td>
          <td>0.275437</td>
          <td>0.084614</td>
          <td>0.078683</td>
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
