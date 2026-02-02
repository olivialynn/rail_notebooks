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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f245b0f7850>



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
    0      23.994413  0.056321  0.048566  
    1      25.391064  0.125728  0.092677  
    2      24.304707  0.064788  0.042801  
    3      25.291103  0.041003  0.020516  
    4      25.096743  0.052938  0.035649  
    ...          ...       ...       ...  
    99995  24.737946  0.119153  0.104567  
    99996  24.224169  0.216656  0.186960  
    99997  25.613836  0.130272  0.100009  
    99998  25.274899  0.036179  0.025622  
    99999  25.699642  0.142566  0.103843  
    
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
          <td>27.569622</td>
          <td>0.811119</td>
          <td>26.818560</td>
          <td>0.181807</td>
          <td>25.982508</td>
          <td>0.077638</td>
          <td>25.109503</td>
          <td>0.058506</td>
          <td>24.698644</td>
          <td>0.077824</td>
          <td>23.873185</td>
          <td>0.084537</td>
          <td>0.056321</td>
          <td>0.048566</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.770614</td>
          <td>0.174569</td>
          <td>26.565218</td>
          <td>0.129314</td>
          <td>26.138286</td>
          <td>0.144253</td>
          <td>25.933247</td>
          <td>0.225524</td>
          <td>25.580173</td>
          <td>0.356522</td>
          <td>0.125728</td>
          <td>0.092677</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.288178</td>
          <td>1.105936</td>
          <td>28.800388</td>
          <td>0.746669</td>
          <td>25.939971</td>
          <td>0.121521</td>
          <td>25.037794</td>
          <td>0.104856</td>
          <td>24.245061</td>
          <td>0.117092</td>
          <td>0.064788</td>
          <td>0.042801</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.262749</td>
          <td>0.568610</td>
          <td>27.738278</td>
          <td>0.343413</td>
          <td>26.414180</td>
          <td>0.182562</td>
          <td>25.452709</td>
          <td>0.150244</td>
          <td>25.348342</td>
          <td>0.296502</td>
          <td>0.041003</td>
          <td>0.020516</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.916381</td>
          <td>0.236037</td>
          <td>26.297368</td>
          <td>0.116206</td>
          <td>25.902598</td>
          <td>0.072343</td>
          <td>25.688037</td>
          <td>0.097527</td>
          <td>25.382960</td>
          <td>0.141498</td>
          <td>25.088302</td>
          <td>0.239836</td>
          <td>0.052938</td>
          <td>0.035649</td>
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
          <td>26.415043</td>
          <td>0.352826</td>
          <td>26.391414</td>
          <td>0.126089</td>
          <td>25.448895</td>
          <td>0.048376</td>
          <td>25.079218</td>
          <td>0.056954</td>
          <td>24.761752</td>
          <td>0.082280</td>
          <td>24.816459</td>
          <td>0.191154</td>
          <td>0.119153</td>
          <td>0.104567</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.141559</td>
          <td>0.238189</td>
          <td>26.035056</td>
          <td>0.081324</td>
          <td>25.258164</td>
          <td>0.066750</td>
          <td>24.823747</td>
          <td>0.086900</td>
          <td>24.282914</td>
          <td>0.121010</td>
          <td>0.216656</td>
          <td>0.186960</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.050456</td>
          <td>0.568984</td>
          <td>26.639058</td>
          <td>0.156056</td>
          <td>26.684358</td>
          <td>0.143324</td>
          <td>26.015855</td>
          <td>0.129787</td>
          <td>25.993363</td>
          <td>0.237042</td>
          <td>25.276788</td>
          <td>0.279841</td>
          <td>0.130272</td>
          <td>0.100009</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.039254</td>
          <td>0.261089</td>
          <td>26.242672</td>
          <td>0.110802</td>
          <td>26.156779</td>
          <td>0.090527</td>
          <td>25.805684</td>
          <td>0.108105</td>
          <td>25.885739</td>
          <td>0.216781</td>
          <td>25.462013</td>
          <td>0.324745</td>
          <td>0.036179</td>
          <td>0.025622</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.946901</td>
          <td>0.527972</td>
          <td>26.736977</td>
          <td>0.169650</td>
          <td>26.600972</td>
          <td>0.133377</td>
          <td>26.328599</td>
          <td>0.169771</td>
          <td>26.056262</td>
          <td>0.249654</td>
          <td>26.025959</td>
          <td>0.500704</td>
          <td>0.142566</td>
          <td>0.103843</td>
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
          <td>27.937244</td>
          <td>1.108110</td>
          <td>26.656978</td>
          <td>0.183462</td>
          <td>26.045640</td>
          <td>0.097415</td>
          <td>25.135221</td>
          <td>0.071654</td>
          <td>24.545098</td>
          <td>0.080693</td>
          <td>23.911308</td>
          <td>0.104294</td>
          <td>0.056321</td>
          <td>0.048566</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.602205</td>
          <td>0.404839</td>
          <td>26.836114</td>
          <td>0.198190</td>
          <td>26.310477</td>
          <td>0.204491</td>
          <td>25.834874</td>
          <td>0.251171</td>
          <td>25.407424</td>
          <td>0.374663</td>
          <td>0.125728</td>
          <td>0.092677</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.582358</td>
          <td>0.389433</td>
          <td>27.799050</td>
          <td>0.418957</td>
          <td>25.822466</td>
          <td>0.130948</td>
          <td>25.092616</td>
          <td>0.130349</td>
          <td>24.584254</td>
          <td>0.186317</td>
          <td>0.064788</td>
          <td>0.042801</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.013009</td>
          <td>0.610456</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.225056</td>
          <td>0.264416</td>
          <td>26.206964</td>
          <td>0.180810</td>
          <td>25.433684</td>
          <td>0.173524</td>
          <td>25.262999</td>
          <td>0.323460</td>
          <td>0.041003</td>
          <td>0.020516</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.571653</td>
          <td>0.443224</td>
          <td>26.463326</td>
          <td>0.155271</td>
          <td>25.881497</td>
          <td>0.084122</td>
          <td>25.704435</td>
          <td>0.117811</td>
          <td>25.755571</td>
          <td>0.228113</td>
          <td>24.952739</td>
          <td>0.252475</td>
          <td>0.052938</td>
          <td>0.035649</td>
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
          <td>27.277827</td>
          <td>0.748360</td>
          <td>26.301297</td>
          <td>0.139256</td>
          <td>25.453654</td>
          <td>0.059643</td>
          <td>25.083929</td>
          <td>0.070774</td>
          <td>25.074083</td>
          <td>0.132230</td>
          <td>24.697264</td>
          <td>0.211175</td>
          <td>0.119153</td>
          <td>0.104567</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.879908</td>
          <td>0.598727</td>
          <td>26.395431</td>
          <td>0.162323</td>
          <td>25.852421</td>
          <td>0.092049</td>
          <td>25.172299</td>
          <td>0.083236</td>
          <td>24.977634</td>
          <td>0.131771</td>
          <td>24.498990</td>
          <td>0.193584</td>
          <td>0.216656</td>
          <td>0.186960</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.407668</td>
          <td>0.400908</td>
          <td>26.640728</td>
          <td>0.186477</td>
          <td>26.289697</td>
          <td>0.124719</td>
          <td>26.313297</td>
          <td>0.205790</td>
          <td>25.635069</td>
          <td>0.213677</td>
          <td>25.899971</td>
          <td>0.544574</td>
          <td>0.130272</td>
          <td>0.100009</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.298567</td>
          <td>0.358394</td>
          <td>26.192033</td>
          <td>0.122524</td>
          <td>26.125609</td>
          <td>0.103858</td>
          <td>25.925350</td>
          <td>0.142119</td>
          <td>25.616857</td>
          <td>0.202501</td>
          <td>25.223361</td>
          <td>0.313327</td>
          <td>0.036179</td>
          <td>0.025622</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.232869</td>
          <td>0.730142</td>
          <td>26.569034</td>
          <td>0.176475</td>
          <td>26.530466</td>
          <td>0.154436</td>
          <td>26.570302</td>
          <td>0.256208</td>
          <td>25.979657</td>
          <td>0.285356</td>
          <td>26.014352</td>
          <td>0.594199</td>
          <td>0.142566</td>
          <td>0.103843</td>
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
          <td>26.321986</td>
          <td>0.122477</td>
          <td>26.072348</td>
          <td>0.087149</td>
          <td>25.246693</td>
          <td>0.068655</td>
          <td>24.691105</td>
          <td>0.080169</td>
          <td>23.852534</td>
          <td>0.086201</td>
          <td>0.056321</td>
          <td>0.048566</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.511874</td>
          <td>1.489546</td>
          <td>27.682138</td>
          <td>0.410511</td>
          <td>26.816093</td>
          <td>0.183917</td>
          <td>26.240014</td>
          <td>0.181506</td>
          <td>26.342961</td>
          <td>0.357560</td>
          <td>25.710276</td>
          <td>0.447706</td>
          <td>0.125728</td>
          <td>0.092677</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.452910</td>
          <td>3.092751</td>
          <td>27.874932</td>
          <td>0.439609</td>
          <td>27.515273</td>
          <td>0.297860</td>
          <td>26.318434</td>
          <td>0.175116</td>
          <td>25.073117</td>
          <td>0.112422</td>
          <td>24.475942</td>
          <td>0.148789</td>
          <td>0.064788</td>
          <td>0.042801</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.998041</td>
          <td>1.619798</td>
          <td>27.403559</td>
          <td>0.265853</td>
          <td>26.107162</td>
          <td>0.142477</td>
          <td>25.941952</td>
          <td>0.230179</td>
          <td>25.126872</td>
          <td>0.250974</td>
          <td>0.041003</td>
          <td>0.020516</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.845366</td>
          <td>0.226539</td>
          <td>26.243391</td>
          <td>0.113470</td>
          <td>25.898385</td>
          <td>0.074049</td>
          <td>25.601137</td>
          <td>0.092942</td>
          <td>25.782667</td>
          <td>0.204014</td>
          <td>25.298838</td>
          <td>0.292271</td>
          <td>0.052938</td>
          <td>0.035649</td>
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
          <td>26.583962</td>
          <td>0.440837</td>
          <td>26.460197</td>
          <td>0.151811</td>
          <td>25.349841</td>
          <td>0.051358</td>
          <td>24.999835</td>
          <td>0.061912</td>
          <td>24.780126</td>
          <td>0.096759</td>
          <td>24.352141</td>
          <td>0.148992</td>
          <td>0.119153</td>
          <td>0.104567</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.198487</td>
          <td>0.784561</td>
          <td>26.462553</td>
          <td>0.185268</td>
          <td>25.905172</td>
          <td>0.104820</td>
          <td>25.101830</td>
          <td>0.085275</td>
          <td>24.711706</td>
          <td>0.113654</td>
          <td>24.385667</td>
          <td>0.190986</td>
          <td>0.216656</td>
          <td>0.186960</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.306925</td>
          <td>0.741944</td>
          <td>26.795903</td>
          <td>0.203251</td>
          <td>26.465478</td>
          <td>0.138037</td>
          <td>26.269003</td>
          <td>0.188435</td>
          <td>25.981745</td>
          <td>0.271037</td>
          <td>30.659485</td>
          <td>4.255353</td>
          <td>0.130272</td>
          <td>0.100009</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.011745</td>
          <td>0.257469</td>
          <td>26.445009</td>
          <td>0.133567</td>
          <td>26.167718</td>
          <td>0.092614</td>
          <td>25.899067</td>
          <td>0.118890</td>
          <td>25.623262</td>
          <td>0.176021</td>
          <td>25.781327</td>
          <td>0.421620</td>
          <td>0.036179</td>
          <td>0.025622</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.343761</td>
          <td>0.373425</td>
          <td>26.840823</td>
          <td>0.214514</td>
          <td>26.449516</td>
          <td>0.138714</td>
          <td>25.848262</td>
          <td>0.134065</td>
          <td>25.982828</td>
          <td>0.276083</td>
          <td>26.074109</td>
          <td>0.600379</td>
          <td>0.142566</td>
          <td>0.103843</td>
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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_24_0.png


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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_25_0.png


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
