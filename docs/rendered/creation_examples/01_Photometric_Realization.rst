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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f007c395f90>



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
          <td>27.706285</td>
          <td>0.885132</td>
          <td>26.765332</td>
          <td>0.173788</td>
          <td>25.873639</td>
          <td>0.070513</td>
          <td>25.191594</td>
          <td>0.062926</td>
          <td>24.714941</td>
          <td>0.078952</td>
          <td>24.200252</td>
          <td>0.112611</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.660780</td>
          <td>0.860014</td>
          <td>28.049120</td>
          <td>0.486566</td>
          <td>26.412993</td>
          <td>0.113295</td>
          <td>26.591047</td>
          <td>0.211845</td>
          <td>25.754738</td>
          <td>0.194243</td>
          <td>25.456019</td>
          <td>0.323200</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.357797</td>
          <td>1.150920</td>
          <td>27.928409</td>
          <td>0.398310</td>
          <td>25.958751</td>
          <td>0.123519</td>
          <td>24.764753</td>
          <td>0.082498</td>
          <td>24.063883</td>
          <td>0.099958</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.882363</td>
          <td>0.864727</td>
          <td>27.457811</td>
          <td>0.274282</td>
          <td>26.197876</td>
          <td>0.151829</td>
          <td>26.058718</td>
          <td>0.250159</td>
          <td>25.778339</td>
          <td>0.415693</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.580470</td>
          <td>0.401220</td>
          <td>26.100042</td>
          <td>0.097820</td>
          <td>26.085187</td>
          <td>0.084999</td>
          <td>25.624340</td>
          <td>0.092223</td>
          <td>25.452065</td>
          <td>0.150161</td>
          <td>25.384300</td>
          <td>0.305197</td>
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
          <td>28.095999</td>
          <td>1.119206</td>
          <td>26.444430</td>
          <td>0.132007</td>
          <td>25.442834</td>
          <td>0.048117</td>
          <td>24.986851</td>
          <td>0.052470</td>
          <td>24.764199</td>
          <td>0.082458</td>
          <td>24.347629</td>
          <td>0.127998</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.985371</td>
          <td>0.542932</td>
          <td>26.730783</td>
          <td>0.168759</td>
          <td>26.261219</td>
          <td>0.099219</td>
          <td>25.294403</td>
          <td>0.068928</td>
          <td>24.871724</td>
          <td>0.090646</td>
          <td>24.200039</td>
          <td>0.112590</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.787755</td>
          <td>0.931277</td>
          <td>26.957367</td>
          <td>0.204349</td>
          <td>26.579876</td>
          <td>0.130966</td>
          <td>26.324725</td>
          <td>0.169212</td>
          <td>26.226918</td>
          <td>0.286928</td>
          <td>25.664844</td>
          <td>0.380879</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.866223</td>
          <td>0.226446</td>
          <td>26.048674</td>
          <td>0.093514</td>
          <td>26.070277</td>
          <td>0.083889</td>
          <td>25.989879</td>
          <td>0.126899</td>
          <td>25.489768</td>
          <td>0.155094</td>
          <td>25.030169</td>
          <td>0.228572</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.434013</td>
          <td>0.358113</td>
          <td>26.968765</td>
          <td>0.206309</td>
          <td>26.490600</td>
          <td>0.121210</td>
          <td>26.620596</td>
          <td>0.217135</td>
          <td>26.162150</td>
          <td>0.272241</td>
          <td>25.788301</td>
          <td>0.418870</td>
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
          <td>27.136764</td>
          <td>0.663966</td>
          <td>26.986502</td>
          <td>0.239689</td>
          <td>26.260315</td>
          <td>0.116426</td>
          <td>25.081111</td>
          <td>0.067643</td>
          <td>24.784169</td>
          <td>0.098645</td>
          <td>23.886640</td>
          <td>0.101100</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.039803</td>
          <td>0.250481</td>
          <td>26.727097</td>
          <td>0.174011</td>
          <td>26.258304</td>
          <td>0.188206</td>
          <td>26.192534</td>
          <td>0.323477</td>
          <td>25.458917</td>
          <td>0.376229</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.790163</td>
          <td>0.460612</td>
          <td>28.698863</td>
          <td>0.801450</td>
          <td>25.943288</td>
          <td>0.147132</td>
          <td>24.961264</td>
          <td>0.117719</td>
          <td>23.979093</td>
          <td>0.112130</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.479390</td>
          <td>0.376353</td>
          <td>27.364075</td>
          <td>0.313668</td>
          <td>26.638228</td>
          <td>0.275287</td>
          <td>25.549273</td>
          <td>0.203354</td>
          <td>24.759644</td>
          <td>0.228159</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.326538</td>
          <td>0.365578</td>
          <td>26.097511</td>
          <td>0.112564</td>
          <td>25.889131</td>
          <td>0.084136</td>
          <td>25.638399</td>
          <td>0.110482</td>
          <td>26.107573</td>
          <td>0.302266</td>
          <td>25.024484</td>
          <td>0.266076</td>
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
          <td>26.672939</td>
          <td>0.482575</td>
          <td>26.508787</td>
          <td>0.163437</td>
          <td>25.470721</td>
          <td>0.059341</td>
          <td>25.049212</td>
          <td>0.067219</td>
          <td>24.753779</td>
          <td>0.098086</td>
          <td>24.464233</td>
          <td>0.170116</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.854784</td>
          <td>1.053357</td>
          <td>26.926351</td>
          <td>0.228867</td>
          <td>25.944446</td>
          <td>0.088676</td>
          <td>25.216528</td>
          <td>0.076581</td>
          <td>24.737654</td>
          <td>0.095098</td>
          <td>24.215663</td>
          <td>0.135168</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.320952</td>
          <td>0.367072</td>
          <td>26.532607</td>
          <td>0.165591</td>
          <td>26.339636</td>
          <td>0.126317</td>
          <td>26.270877</td>
          <td>0.192604</td>
          <td>25.850820</td>
          <td>0.248163</td>
          <td>27.631958</td>
          <td>1.533162</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.346723</td>
          <td>0.379276</td>
          <td>26.197146</td>
          <td>0.126197</td>
          <td>26.034478</td>
          <td>0.098621</td>
          <td>25.846396</td>
          <td>0.136611</td>
          <td>25.329843</td>
          <td>0.163189</td>
          <td>25.235599</td>
          <td>0.324793</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.172366</td>
          <td>0.325948</td>
          <td>26.979534</td>
          <td>0.240350</td>
          <td>26.612082</td>
          <td>0.159279</td>
          <td>27.428368</td>
          <td>0.483810</td>
          <td>26.034861</td>
          <td>0.287613</td>
          <td>26.520677</td>
          <td>0.811946</td>
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
          <td>27.296006</td>
          <td>0.675874</td>
          <td>26.932714</td>
          <td>0.200189</td>
          <td>26.082126</td>
          <td>0.084781</td>
          <td>25.116222</td>
          <td>0.058864</td>
          <td>24.821045</td>
          <td>0.086705</td>
          <td>23.843288</td>
          <td>0.082350</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.254507</td>
          <td>2.878582</td>
          <td>27.541544</td>
          <td>0.329610</td>
          <td>26.429675</td>
          <td>0.115062</td>
          <td>26.467249</td>
          <td>0.191115</td>
          <td>25.633267</td>
          <td>0.175443</td>
          <td>25.297787</td>
          <td>0.284902</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.563791</td>
          <td>1.347348</td>
          <td>28.297273</td>
          <td>0.562036</td>
          <td>25.979239</td>
          <td>0.136784</td>
          <td>25.225167</td>
          <td>0.133815</td>
          <td>24.252112</td>
          <td>0.128106</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.582131</td>
          <td>1.598268</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.862807</td>
          <td>0.460381</td>
          <td>26.476419</td>
          <td>0.240293</td>
          <td>25.392680</td>
          <td>0.177589</td>
          <td>25.508260</td>
          <td>0.413782</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.712760</td>
          <td>0.199441</td>
          <td>26.071113</td>
          <td>0.095489</td>
          <td>25.946125</td>
          <td>0.075290</td>
          <td>25.839384</td>
          <td>0.111498</td>
          <td>25.350563</td>
          <td>0.137793</td>
          <td>25.312450</td>
          <td>0.288431</td>
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
          <td>26.780746</td>
          <td>0.489643</td>
          <td>26.249419</td>
          <td>0.119343</td>
          <td>25.477959</td>
          <td>0.053763</td>
          <td>25.176502</td>
          <td>0.067478</td>
          <td>25.070545</td>
          <td>0.116662</td>
          <td>24.633751</td>
          <td>0.177164</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.664898</td>
          <td>0.869437</td>
          <td>26.662702</td>
          <td>0.161475</td>
          <td>25.975134</td>
          <td>0.078424</td>
          <td>25.149986</td>
          <td>0.061716</td>
          <td>24.762207</td>
          <td>0.083685</td>
          <td>24.074521</td>
          <td>0.102631</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>30.193813</td>
          <td>2.857882</td>
          <td>27.102089</td>
          <td>0.239993</td>
          <td>26.368790</td>
          <td>0.114436</td>
          <td>26.491280</td>
          <td>0.204619</td>
          <td>25.849946</td>
          <td>0.220387</td>
          <td>24.925623</td>
          <td>0.219850</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.479421</td>
          <td>0.398909</td>
          <td>26.317434</td>
          <td>0.130640</td>
          <td>26.007677</td>
          <td>0.089069</td>
          <td>25.833303</td>
          <td>0.124695</td>
          <td>25.868988</td>
          <td>0.238208</td>
          <td>25.053808</td>
          <td>0.260489</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.248330</td>
          <td>0.316921</td>
          <td>26.801320</td>
          <td>0.185121</td>
          <td>26.696408</td>
          <td>0.150435</td>
          <td>26.137727</td>
          <td>0.150053</td>
          <td>25.871912</td>
          <td>0.222339</td>
          <td>28.391455</td>
          <td>2.015164</td>
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
