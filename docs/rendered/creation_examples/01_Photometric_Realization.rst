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

    <pzflow.flow.Flow at 0x7f0c8ac4e950>



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
          <td>27.190965</td>
          <td>0.628451</td>
          <td>26.745765</td>
          <td>0.170923</td>
          <td>26.029471</td>
          <td>0.080924</td>
          <td>25.277513</td>
          <td>0.067904</td>
          <td>24.599364</td>
          <td>0.071285</td>
          <td>23.820147</td>
          <td>0.080675</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.892662</td>
          <td>0.432654</td>
          <td>26.447408</td>
          <td>0.116742</td>
          <td>26.324338</td>
          <td>0.169157</td>
          <td>25.907275</td>
          <td>0.220706</td>
          <td>25.296736</td>
          <td>0.284401</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.851870</td>
          <td>0.968643</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.176637</td>
          <td>0.480704</td>
          <td>26.020216</td>
          <td>0.130278</td>
          <td>25.049666</td>
          <td>0.105950</td>
          <td>24.252685</td>
          <td>0.117872</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.448767</td>
          <td>0.749236</td>
          <td>29.862658</td>
          <td>1.505989</td>
          <td>27.225520</td>
          <td>0.226610</td>
          <td>26.307822</td>
          <td>0.166793</td>
          <td>25.629014</td>
          <td>0.174651</td>
          <td>25.545019</td>
          <td>0.346803</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.405481</td>
          <td>0.350187</td>
          <td>26.022215</td>
          <td>0.091367</td>
          <td>25.901723</td>
          <td>0.072288</td>
          <td>25.631203</td>
          <td>0.092781</td>
          <td>25.436346</td>
          <td>0.148148</td>
          <td>25.272251</td>
          <td>0.278813</td>
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
          <td>26.765905</td>
          <td>0.461870</td>
          <td>26.317267</td>
          <td>0.118233</td>
          <td>25.445580</td>
          <td>0.048234</td>
          <td>25.164703</td>
          <td>0.061443</td>
          <td>24.767031</td>
          <td>0.082664</td>
          <td>24.718564</td>
          <td>0.175961</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.576798</td>
          <td>0.400088</td>
          <td>26.813634</td>
          <td>0.181051</td>
          <td>26.182899</td>
          <td>0.092629</td>
          <td>25.232926</td>
          <td>0.065274</td>
          <td>25.004597</td>
          <td>0.101854</td>
          <td>24.242831</td>
          <td>0.116865</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.159069</td>
          <td>0.614562</td>
          <td>26.431355</td>
          <td>0.130524</td>
          <td>26.251339</td>
          <td>0.098364</td>
          <td>26.536488</td>
          <td>0.202384</td>
          <td>25.681895</td>
          <td>0.182659</td>
          <td>25.059396</td>
          <td>0.234174</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.369096</td>
          <td>0.710281</td>
          <td>26.078989</td>
          <td>0.096032</td>
          <td>26.018132</td>
          <td>0.080119</td>
          <td>25.896019</td>
          <td>0.116964</td>
          <td>25.662395</td>
          <td>0.179667</td>
          <td>25.074790</td>
          <td>0.237174</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.740805</td>
          <td>0.904501</td>
          <td>26.730245</td>
          <td>0.168682</td>
          <td>26.332351</td>
          <td>0.105594</td>
          <td>26.138505</td>
          <td>0.144280</td>
          <td>25.924436</td>
          <td>0.223879</td>
          <td>25.870519</td>
          <td>0.445857</td>
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
          <td>27.423550</td>
          <td>0.804492</td>
          <td>26.402646</td>
          <td>0.146515</td>
          <td>26.059451</td>
          <td>0.097690</td>
          <td>25.171505</td>
          <td>0.073275</td>
          <td>24.576804</td>
          <td>0.082209</td>
          <td>24.261300</td>
          <td>0.140004</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.620626</td>
          <td>0.457879</td>
          <td>27.538015</td>
          <td>0.373302</td>
          <td>26.717109</td>
          <td>0.172541</td>
          <td>26.337085</td>
          <td>0.201111</td>
          <td>26.306885</td>
          <td>0.354086</td>
          <td>25.594814</td>
          <td>0.417791</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.442035</td>
          <td>0.320487</td>
          <td>26.617331</td>
          <td>0.259257</td>
          <td>25.145196</td>
          <td>0.138048</td>
          <td>24.347658</td>
          <td>0.154202</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.081039</td>
          <td>1.236512</td>
          <td>27.655190</td>
          <td>0.430808</td>
          <td>27.380032</td>
          <td>0.317690</td>
          <td>26.183020</td>
          <td>0.188749</td>
          <td>25.584300</td>
          <td>0.209407</td>
          <td>24.880267</td>
          <td>0.252040</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>29.749829</td>
          <td>2.537664</td>
          <td>26.190421</td>
          <td>0.122028</td>
          <td>25.937241</td>
          <td>0.087776</td>
          <td>25.975590</td>
          <td>0.147946</td>
          <td>25.187611</td>
          <td>0.140136</td>
          <td>24.842620</td>
          <td>0.229101</td>
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
          <td>26.887459</td>
          <td>0.564403</td>
          <td>26.254514</td>
          <td>0.131388</td>
          <td>25.465811</td>
          <td>0.059083</td>
          <td>25.028996</td>
          <td>0.066026</td>
          <td>24.832432</td>
          <td>0.105076</td>
          <td>25.354240</td>
          <td>0.353315</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.812879</td>
          <td>1.027538</td>
          <td>26.480517</td>
          <td>0.157199</td>
          <td>25.988430</td>
          <td>0.092171</td>
          <td>25.220403</td>
          <td>0.076843</td>
          <td>24.818692</td>
          <td>0.102096</td>
          <td>24.276451</td>
          <td>0.142440</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.053098</td>
          <td>1.185603</td>
          <td>26.627264</td>
          <td>0.179451</td>
          <td>26.309130</td>
          <td>0.123019</td>
          <td>26.062387</td>
          <td>0.161379</td>
          <td>26.058971</td>
          <td>0.294005</td>
          <td>25.253246</td>
          <td>0.323733</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.044275</td>
          <td>0.298677</td>
          <td>25.860357</td>
          <td>0.094101</td>
          <td>26.083546</td>
          <td>0.102950</td>
          <td>25.647942</td>
          <td>0.115018</td>
          <td>25.750392</td>
          <td>0.232454</td>
          <td>25.581908</td>
          <td>0.425363</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.261272</td>
          <td>0.726880</td>
          <td>26.526881</td>
          <td>0.164388</td>
          <td>26.721705</td>
          <td>0.174870</td>
          <td>26.397366</td>
          <td>0.213581</td>
          <td>25.633457</td>
          <td>0.206660</td>
          <td>25.668680</td>
          <td>0.445774</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.574887</td>
          <td>0.147722</td>
          <td>25.945460</td>
          <td>0.075148</td>
          <td>25.126305</td>
          <td>0.059393</td>
          <td>24.715832</td>
          <td>0.079024</td>
          <td>23.917390</td>
          <td>0.087905</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.171263</td>
          <td>0.620168</td>
          <td>27.520608</td>
          <td>0.324173</td>
          <td>26.361308</td>
          <td>0.108402</td>
          <td>26.424191</td>
          <td>0.184291</td>
          <td>25.797924</td>
          <td>0.201607</td>
          <td>25.252010</td>
          <td>0.274516</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.030425</td>
          <td>0.588196</td>
          <td>28.584394</td>
          <td>0.751417</td>
          <td>28.717443</td>
          <td>0.751708</td>
          <td>25.929003</td>
          <td>0.130974</td>
          <td>24.899563</td>
          <td>0.100799</td>
          <td>24.338201</td>
          <td>0.138002</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.058185</td>
          <td>1.218891</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.977757</td>
          <td>0.983589</td>
          <td>25.945105</td>
          <td>0.153624</td>
          <td>25.362282</td>
          <td>0.173066</td>
          <td>25.650632</td>
          <td>0.460922</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.697798</td>
          <td>0.439154</td>
          <td>26.048921</td>
          <td>0.093649</td>
          <td>25.910581</td>
          <td>0.072960</td>
          <td>25.565104</td>
          <td>0.087673</td>
          <td>25.295674</td>
          <td>0.131412</td>
          <td>25.014699</td>
          <td>0.225971</td>
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
          <td>26.403050</td>
          <td>0.367361</td>
          <td>26.202940</td>
          <td>0.114617</td>
          <td>25.473988</td>
          <td>0.053574</td>
          <td>24.943022</td>
          <td>0.054859</td>
          <td>24.926147</td>
          <td>0.102850</td>
          <td>25.020862</td>
          <td>0.244905</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.662657</td>
          <td>0.868203</td>
          <td>26.876947</td>
          <td>0.193636</td>
          <td>26.029891</td>
          <td>0.082306</td>
          <td>25.206973</td>
          <td>0.064914</td>
          <td>24.856465</td>
          <td>0.090925</td>
          <td>24.309043</td>
          <td>0.125897</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.352392</td>
          <td>1.318236</td>
          <td>26.672384</td>
          <td>0.167363</td>
          <td>26.446686</td>
          <td>0.122457</td>
          <td>26.112897</td>
          <td>0.148395</td>
          <td>26.109904</td>
          <td>0.272981</td>
          <td>25.432653</td>
          <td>0.332141</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.960403</td>
          <td>0.570385</td>
          <td>26.345193</td>
          <td>0.133810</td>
          <td>25.819402</td>
          <td>0.075447</td>
          <td>25.772328</td>
          <td>0.118264</td>
          <td>25.810118</td>
          <td>0.226875</td>
          <td>25.097625</td>
          <td>0.269975</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.463139</td>
          <td>0.772080</td>
          <td>26.720418</td>
          <td>0.172860</td>
          <td>26.467781</td>
          <td>0.123493</td>
          <td>26.225768</td>
          <td>0.161800</td>
          <td>26.009273</td>
          <td>0.249087</td>
          <td>25.829557</td>
          <td>0.447532</td>
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
