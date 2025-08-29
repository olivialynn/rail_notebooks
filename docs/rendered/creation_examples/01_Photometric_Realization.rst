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

    <pzflow.flow.Flow at 0x7f01fbde22c0>



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
    0      23.994413  0.054215  0.046940  
    1      25.391064  0.002536  0.001868  
    2      24.304707  0.015630  0.011947  
    3      25.291103  0.019484  0.010890  
    4      25.096743  0.073157  0.055111  
    ...          ...       ...       ...  
    99995  24.737946  0.087400  0.053245  
    99996  24.224169  0.006709  0.006319  
    99997  25.613836  0.006807  0.006117  
    99998  25.274899  0.009416  0.009138  
    99999  25.699642  0.128433  0.081509  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.637376</td>
          <td>0.155831</td>
          <td>26.056745</td>
          <td>0.082894</td>
          <td>25.310720</td>
          <td>0.069931</td>
          <td>24.655620</td>
          <td>0.074921</td>
          <td>24.114408</td>
          <td>0.104478</td>
          <td>0.054215</td>
          <td>0.046940</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.664116</td>
          <td>0.362765</td>
          <td>26.473558</td>
          <td>0.119429</td>
          <td>26.493865</td>
          <td>0.195263</td>
          <td>26.431389</td>
          <td>0.337918</td>
          <td>25.136461</td>
          <td>0.249541</td>
          <td>0.002536</td>
          <td>0.001868</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.511271</td>
          <td>0.676828</td>
          <td>27.684245</td>
          <td>0.329035</td>
          <td>25.944722</td>
          <td>0.122023</td>
          <td>24.820825</td>
          <td>0.086676</td>
          <td>24.353897</td>
          <td>0.128695</td>
          <td>0.015630</td>
          <td>0.011947</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.757405</td>
          <td>0.458936</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.454964</td>
          <td>0.273648</td>
          <td>26.354829</td>
          <td>0.173600</td>
          <td>25.448957</td>
          <td>0.149761</td>
          <td>25.021242</td>
          <td>0.226885</td>
          <td>0.019484</td>
          <td>0.010890</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.634159</td>
          <td>0.418066</td>
          <td>25.982345</td>
          <td>0.088223</td>
          <td>25.902409</td>
          <td>0.072331</td>
          <td>25.661772</td>
          <td>0.095306</td>
          <td>25.386868</td>
          <td>0.141975</td>
          <td>25.004644</td>
          <td>0.223778</td>
          <td>0.073157</td>
          <td>0.055111</td>
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
          <td>27.054713</td>
          <td>0.570721</td>
          <td>26.305564</td>
          <td>0.117037</td>
          <td>25.460417</td>
          <td>0.048874</td>
          <td>25.099039</td>
          <td>0.057965</td>
          <td>24.821068</td>
          <td>0.086695</td>
          <td>24.752777</td>
          <td>0.181140</td>
          <td>0.087400</td>
          <td>0.053245</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.031877</td>
          <td>0.561452</td>
          <td>26.672421</td>
          <td>0.160569</td>
          <td>26.030513</td>
          <td>0.080999</td>
          <td>25.229161</td>
          <td>0.065057</td>
          <td>24.796763</td>
          <td>0.084859</td>
          <td>24.131913</td>
          <td>0.106090</td>
          <td>0.006709</td>
          <td>0.006319</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.496244</td>
          <td>0.375929</td>
          <td>26.572610</td>
          <td>0.147417</td>
          <td>26.341875</td>
          <td>0.106477</td>
          <td>26.174493</td>
          <td>0.148813</td>
          <td>25.994442</td>
          <td>0.237253</td>
          <td>25.340972</td>
          <td>0.294747</td>
          <td>0.006807</td>
          <td>0.006117</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.211251</td>
          <td>0.300109</td>
          <td>26.182248</td>
          <td>0.105112</td>
          <td>25.945692</td>
          <td>0.075153</td>
          <td>25.634739</td>
          <td>0.093070</td>
          <td>25.582617</td>
          <td>0.167893</td>
          <td>25.234740</td>
          <td>0.270438</td>
          <td>0.009416</td>
          <td>0.009138</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.259426</td>
          <td>0.311915</td>
          <td>26.978900</td>
          <td>0.208067</td>
          <td>26.511237</td>
          <td>0.123402</td>
          <td>26.297821</td>
          <td>0.165378</td>
          <td>26.227771</td>
          <td>0.287126</td>
          <td>25.767902</td>
          <td>0.412386</td>
          <td>0.128433</td>
          <td>0.081509</td>
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
          <td>26.948599</td>
          <td>0.585093</td>
          <td>26.463494</td>
          <td>0.155539</td>
          <td>26.042484</td>
          <td>0.097083</td>
          <td>25.182072</td>
          <td>0.074633</td>
          <td>24.664282</td>
          <td>0.089563</td>
          <td>24.269781</td>
          <td>0.142266</td>
          <td>0.054215</td>
          <td>0.046940</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.776321</td>
          <td>0.448043</td>
          <td>26.787921</td>
          <td>0.183179</td>
          <td>26.273418</td>
          <td>0.190580</td>
          <td>25.838835</td>
          <td>0.242799</td>
          <td>25.375539</td>
          <td>0.352413</td>
          <td>0.002536</td>
          <td>0.001868</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.393551</td>
          <td>1.415518</td>
          <td>33.302492</td>
          <td>4.809750</td>
          <td>27.455071</td>
          <td>0.317532</td>
          <td>25.994104</td>
          <td>0.150362</td>
          <td>25.144969</td>
          <td>0.135115</td>
          <td>24.167894</td>
          <td>0.129235</td>
          <td>0.015630</td>
          <td>0.011947</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.803406</td>
          <td>0.457563</td>
          <td>27.175179</td>
          <td>0.253196</td>
          <td>26.122356</td>
          <td>0.167817</td>
          <td>25.310887</td>
          <td>0.155861</td>
          <td>25.694350</td>
          <td>0.450809</td>
          <td>0.019484</td>
          <td>0.010890</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.949139</td>
          <td>0.587270</td>
          <td>26.068995</td>
          <td>0.111162</td>
          <td>25.884247</td>
          <td>0.084937</td>
          <td>25.562779</td>
          <td>0.104893</td>
          <td>25.753034</td>
          <td>0.229182</td>
          <td>25.841688</td>
          <td>0.508876</td>
          <td>0.073157</td>
          <td>0.055111</td>
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
          <td>27.942877</td>
          <td>1.116253</td>
          <td>26.547574</td>
          <td>0.168398</td>
          <td>25.509811</td>
          <td>0.061212</td>
          <td>25.201603</td>
          <td>0.076629</td>
          <td>24.832977</td>
          <td>0.104753</td>
          <td>25.016558</td>
          <td>0.268707</td>
          <td>0.087400</td>
          <td>0.053245</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.375348</td>
          <td>0.379676</td>
          <td>26.731188</td>
          <td>0.193763</td>
          <td>25.982317</td>
          <td>0.091306</td>
          <td>25.078353</td>
          <td>0.067486</td>
          <td>24.824091</td>
          <td>0.102165</td>
          <td>24.404101</td>
          <td>0.158280</td>
          <td>0.006709</td>
          <td>0.006319</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.944161</td>
          <td>0.580184</td>
          <td>26.687075</td>
          <td>0.186691</td>
          <td>26.335192</td>
          <td>0.124266</td>
          <td>26.504355</td>
          <td>0.231198</td>
          <td>26.572398</td>
          <td>0.434630</td>
          <td>28.316807</td>
          <td>2.073966</td>
          <td>0.006807</td>
          <td>0.006117</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.374764</td>
          <td>0.379542</td>
          <td>26.339410</td>
          <td>0.138792</td>
          <td>26.106976</td>
          <td>0.101869</td>
          <td>25.761833</td>
          <td>0.123001</td>
          <td>25.391817</td>
          <td>0.166923</td>
          <td>25.677378</td>
          <td>0.444857</td>
          <td>0.009416</td>
          <td>0.009138</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.623683</td>
          <td>0.932526</td>
          <td>26.919795</td>
          <td>0.234219</td>
          <td>26.451601</td>
          <td>0.142584</td>
          <td>26.530371</td>
          <td>0.244987</td>
          <td>25.949823</td>
          <td>0.275366</td>
          <td>25.546035</td>
          <td>0.416166</td>
          <td>0.128433</td>
          <td>0.081509</td>
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
          <td>26.698582</td>
          <td>0.448030</td>
          <td>26.763729</td>
          <td>0.178542</td>
          <td>25.938508</td>
          <td>0.077257</td>
          <td>25.299839</td>
          <td>0.071775</td>
          <td>24.930868</td>
          <td>0.098747</td>
          <td>23.990704</td>
          <td>0.097083</td>
          <td>0.054215</td>
          <td>0.046940</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.132570</td>
          <td>1.142898</td>
          <td>28.072773</td>
          <td>0.495188</td>
          <td>26.811952</td>
          <td>0.159911</td>
          <td>26.231156</td>
          <td>0.156232</td>
          <td>25.912068</td>
          <td>0.221602</td>
          <td>25.824274</td>
          <td>0.430535</td>
          <td>0.002536</td>
          <td>0.001868</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.998945</td>
          <td>1.059085</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.088778</td>
          <td>0.451110</td>
          <td>25.844030</td>
          <td>0.112091</td>
          <td>24.947705</td>
          <td>0.097155</td>
          <td>24.470763</td>
          <td>0.142742</td>
          <td>0.015630</td>
          <td>0.011947</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.766152</td>
          <td>0.393718</td>
          <td>27.217977</td>
          <td>0.225914</td>
          <td>26.418824</td>
          <td>0.183906</td>
          <td>25.619748</td>
          <td>0.173847</td>
          <td>25.214844</td>
          <td>0.266954</td>
          <td>0.019484</td>
          <td>0.010890</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.207278</td>
          <td>0.309656</td>
          <td>26.105950</td>
          <td>0.103034</td>
          <td>25.879523</td>
          <td>0.074822</td>
          <td>25.597542</td>
          <td>0.095289</td>
          <td>25.255113</td>
          <td>0.133572</td>
          <td>25.210730</td>
          <td>0.279220</td>
          <td>0.073157</td>
          <td>0.055111</td>
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
          <td>26.636532</td>
          <td>0.435997</td>
          <td>26.304211</td>
          <td>0.123727</td>
          <td>25.372776</td>
          <td>0.048320</td>
          <td>24.933322</td>
          <td>0.053634</td>
          <td>24.891336</td>
          <td>0.098459</td>
          <td>24.865640</td>
          <td>0.212557</td>
          <td>0.087400</td>
          <td>0.053245</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.933348</td>
          <td>0.522952</td>
          <td>26.744735</td>
          <td>0.170856</td>
          <td>26.049950</td>
          <td>0.082447</td>
          <td>25.297545</td>
          <td>0.069162</td>
          <td>24.982482</td>
          <td>0.099958</td>
          <td>24.239493</td>
          <td>0.116596</td>
          <td>0.006709</td>
          <td>0.006319</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.232797</td>
          <td>0.647217</td>
          <td>26.603026</td>
          <td>0.151387</td>
          <td>26.377494</td>
          <td>0.109903</td>
          <td>26.035334</td>
          <td>0.132071</td>
          <td>25.850833</td>
          <td>0.210668</td>
          <td>25.985552</td>
          <td>0.486199</td>
          <td>0.006807</td>
          <td>0.006117</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.731831</td>
          <td>0.450512</td>
          <td>26.061545</td>
          <td>0.094671</td>
          <td>26.045344</td>
          <td>0.082162</td>
          <td>25.945952</td>
          <td>0.122302</td>
          <td>25.684309</td>
          <td>0.183238</td>
          <td>25.283406</td>
          <td>0.281662</td>
          <td>0.009416</td>
          <td>0.009138</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.350756</td>
          <td>1.367157</td>
          <td>26.394740</td>
          <td>0.141984</td>
          <td>26.591207</td>
          <td>0.150724</td>
          <td>25.932862</td>
          <td>0.138509</td>
          <td>26.123012</td>
          <td>0.298152</td>
          <td>24.751879</td>
          <td>0.206557</td>
          <td>0.128433</td>
          <td>0.081509</td>
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
