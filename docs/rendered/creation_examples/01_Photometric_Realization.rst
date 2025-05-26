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

    <pzflow.flow.Flow at 0x7fdcc005a9e0>



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
    0      23.994413  0.019827  0.010832  
    1      25.391064  0.084127  0.067608  
    2      24.304707  0.071459  0.058653  
    3      25.291103  0.065878  0.057320  
    4      25.096743  0.203899  0.163618  
    ...          ...       ...       ...  
    99995  24.737946  0.052940  0.034076  
    99996  24.224169  0.041438  0.022488  
    99997  25.613836  0.003102  0.002626  
    99998  25.274899  0.096420  0.067579  
    99999  25.699642  0.070185  0.051520  
    
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
          <td>26.981233</td>
          <td>0.541308</td>
          <td>26.707902</td>
          <td>0.165503</td>
          <td>26.139234</td>
          <td>0.089141</td>
          <td>25.126628</td>
          <td>0.059402</td>
          <td>24.751214</td>
          <td>0.081519</td>
          <td>24.126515</td>
          <td>0.105590</td>
          <td>0.019827</td>
          <td>0.010832</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.778431</td>
          <td>0.925920</td>
          <td>27.286548</td>
          <td>0.268275</td>
          <td>26.409915</td>
          <td>0.112992</td>
          <td>26.200765</td>
          <td>0.152206</td>
          <td>25.685035</td>
          <td>0.183145</td>
          <td>25.506033</td>
          <td>0.336289</td>
          <td>0.084127</td>
          <td>0.067608</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.465211</td>
          <td>1.222300</td>
          <td>27.580640</td>
          <td>0.302907</td>
          <td>25.781887</td>
          <td>0.105881</td>
          <td>25.016059</td>
          <td>0.102881</td>
          <td>24.389720</td>
          <td>0.132747</td>
          <td>0.071459</td>
          <td>0.058653</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.182205</td>
          <td>0.536507</td>
          <td>27.189949</td>
          <td>0.220006</td>
          <td>26.278886</td>
          <td>0.162728</td>
          <td>25.588267</td>
          <td>0.168703</td>
          <td>25.402412</td>
          <td>0.309660</td>
          <td>0.065878</td>
          <td>0.057320</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.693347</td>
          <td>0.437307</td>
          <td>26.191835</td>
          <td>0.105996</td>
          <td>25.963715</td>
          <td>0.076360</td>
          <td>25.634515</td>
          <td>0.093052</td>
          <td>25.578909</td>
          <td>0.167364</td>
          <td>25.077304</td>
          <td>0.237668</td>
          <td>0.203899</td>
          <td>0.163618</td>
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
          <td>28.147634</td>
          <td>1.152691</td>
          <td>26.374773</td>
          <td>0.124284</td>
          <td>25.499257</td>
          <td>0.050589</td>
          <td>25.094151</td>
          <td>0.057714</td>
          <td>24.872177</td>
          <td>0.090682</td>
          <td>24.679271</td>
          <td>0.170183</td>
          <td>0.052940</td>
          <td>0.034076</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.329014</td>
          <td>0.329676</td>
          <td>26.821253</td>
          <td>0.182222</td>
          <td>26.139400</td>
          <td>0.089154</td>
          <td>25.126865</td>
          <td>0.059414</td>
          <td>25.018824</td>
          <td>0.103130</td>
          <td>24.194302</td>
          <td>0.112028</td>
          <td>0.041438</td>
          <td>0.022488</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.291279</td>
          <td>1.248740</td>
          <td>26.886999</td>
          <td>0.192620</td>
          <td>26.216942</td>
          <td>0.095440</td>
          <td>26.287154</td>
          <td>0.163880</td>
          <td>25.640734</td>
          <td>0.176397</td>
          <td>25.316166</td>
          <td>0.288906</td>
          <td>0.003102</td>
          <td>0.002626</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.769810</td>
          <td>0.463223</td>
          <td>26.225387</td>
          <td>0.109145</td>
          <td>26.089819</td>
          <td>0.085346</td>
          <td>26.039220</td>
          <td>0.132437</td>
          <td>25.851552</td>
          <td>0.210681</td>
          <td>25.300568</td>
          <td>0.285285</td>
          <td>0.096420</td>
          <td>0.067579</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.893454</td>
          <td>0.507723</td>
          <td>26.783479</td>
          <td>0.176484</td>
          <td>26.473622</td>
          <td>0.119435</td>
          <td>26.380049</td>
          <td>0.177358</td>
          <td>25.915105</td>
          <td>0.222148</td>
          <td>25.129758</td>
          <td>0.248170</td>
          <td>0.070185</td>
          <td>0.051520</td>
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
          <td>27.489362</td>
          <td>0.839796</td>
          <td>26.665006</td>
          <td>0.183358</td>
          <td>25.968744</td>
          <td>0.090288</td>
          <td>25.147599</td>
          <td>0.071805</td>
          <td>24.712609</td>
          <td>0.092720</td>
          <td>23.986520</td>
          <td>0.110414</td>
          <td>0.019827</td>
          <td>0.010832</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.841322</td>
          <td>0.545408</td>
          <td>27.586669</td>
          <td>0.393624</td>
          <td>26.535407</td>
          <td>0.150525</td>
          <td>26.139002</td>
          <td>0.173409</td>
          <td>25.843329</td>
          <td>0.248170</td>
          <td>24.696231</td>
          <td>0.206595</td>
          <td>0.084127</td>
          <td>0.067608</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>32.535204</td>
          <td>4.069381</td>
          <td>27.913603</td>
          <td>0.458579</td>
          <td>25.778070</td>
          <td>0.126550</td>
          <td>25.092257</td>
          <td>0.130847</td>
          <td>24.599645</td>
          <td>0.189536</td>
          <td>0.071459</td>
          <td>0.058653</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.286004</td>
          <td>1.219859</td>
          <td>27.719878</td>
          <td>0.395173</td>
          <td>26.320659</td>
          <td>0.200840</td>
          <td>25.046344</td>
          <td>0.125559</td>
          <td>25.799389</td>
          <td>0.492714</td>
          <td>0.065878</td>
          <td>0.057320</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.230636</td>
          <td>0.752856</td>
          <td>26.415905</td>
          <td>0.162491</td>
          <td>25.939630</td>
          <td>0.097558</td>
          <td>25.520593</td>
          <td>0.110848</td>
          <td>25.721948</td>
          <td>0.243028</td>
          <td>25.389783</td>
          <td>0.391591</td>
          <td>0.203899</td>
          <td>0.163618</td>
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
          <td>26.268961</td>
          <td>0.350971</td>
          <td>26.610506</td>
          <td>0.175980</td>
          <td>25.384077</td>
          <td>0.054160</td>
          <td>25.194705</td>
          <td>0.075315</td>
          <td>24.868226</td>
          <td>0.106880</td>
          <td>25.444426</td>
          <td>0.374204</td>
          <td>0.052940</td>
          <td>0.034076</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.256900</td>
          <td>0.346966</td>
          <td>26.643811</td>
          <td>0.180555</td>
          <td>26.069187</td>
          <td>0.098895</td>
          <td>25.165126</td>
          <td>0.073147</td>
          <td>24.770877</td>
          <td>0.097866</td>
          <td>24.249809</td>
          <td>0.139148</td>
          <td>0.041438</td>
          <td>0.022488</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.130706</td>
          <td>0.661200</td>
          <td>26.723268</td>
          <td>0.192456</td>
          <td>26.455259</td>
          <td>0.137851</td>
          <td>26.156093</td>
          <td>0.172561</td>
          <td>26.069491</td>
          <td>0.293055</td>
          <td>25.175554</td>
          <td>0.300614</td>
          <td>0.003102</td>
          <td>0.002626</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.417310</td>
          <td>0.398261</td>
          <td>26.286659</td>
          <td>0.135324</td>
          <td>26.221302</td>
          <td>0.115132</td>
          <td>25.690994</td>
          <td>0.118380</td>
          <td>25.698497</td>
          <td>0.220886</td>
          <td>25.172606</td>
          <td>0.306437</td>
          <td>0.096420</td>
          <td>0.067579</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.750161</td>
          <td>0.508179</td>
          <td>26.642627</td>
          <td>0.181786</td>
          <td>26.613289</td>
          <td>0.159857</td>
          <td>26.251867</td>
          <td>0.189526</td>
          <td>26.413179</td>
          <td>0.388957</td>
          <td>25.519625</td>
          <td>0.398816</td>
          <td>0.070185</td>
          <td>0.051520</td>
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
          <td>28.552885</td>
          <td>1.436074</td>
          <td>26.991871</td>
          <td>0.210936</td>
          <td>25.922104</td>
          <td>0.073859</td>
          <td>25.127575</td>
          <td>0.059671</td>
          <td>24.766474</td>
          <td>0.082911</td>
          <td>23.885384</td>
          <td>0.085759</td>
          <td>0.019827</td>
          <td>0.010832</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.698481</td>
          <td>3.351864</td>
          <td>27.203982</td>
          <td>0.266140</td>
          <td>26.655089</td>
          <td>0.150116</td>
          <td>26.110639</td>
          <td>0.151799</td>
          <td>26.223974</td>
          <td>0.306035</td>
          <td>25.634682</td>
          <td>0.397723</td>
          <td>0.084127</td>
          <td>0.067608</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.028170</td>
          <td>0.450647</td>
          <td>26.212630</td>
          <td>0.162585</td>
          <td>24.969784</td>
          <td>0.104330</td>
          <td>24.396296</td>
          <td>0.141159</td>
          <td>0.071459</td>
          <td>0.058653</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.447848</td>
          <td>0.768526</td>
          <td>27.703071</td>
          <td>0.388397</td>
          <td>27.151773</td>
          <td>0.223282</td>
          <td>26.530005</td>
          <td>0.211416</td>
          <td>25.624450</td>
          <td>0.182419</td>
          <td>25.227944</td>
          <td>0.281933</td>
          <td>0.065878</td>
          <td>0.057320</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.197462</td>
          <td>0.369838</td>
          <td>26.058711</td>
          <td>0.125817</td>
          <td>25.834434</td>
          <td>0.094122</td>
          <td>25.470995</td>
          <td>0.112437</td>
          <td>25.468416</td>
          <td>0.207657</td>
          <td>25.190314</td>
          <td>0.352765</td>
          <td>0.203899</td>
          <td>0.163618</td>
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
          <td>26.174777</td>
          <td>0.296344</td>
          <td>26.336905</td>
          <td>0.122999</td>
          <td>25.400598</td>
          <td>0.047588</td>
          <td>25.091478</td>
          <td>0.059195</td>
          <td>24.829441</td>
          <td>0.089649</td>
          <td>24.653766</td>
          <td>0.170956</td>
          <td>0.052940</td>
          <td>0.034076</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.507039</td>
          <td>0.784667</td>
          <td>26.507794</td>
          <td>0.141196</td>
          <td>26.016142</td>
          <td>0.081178</td>
          <td>25.218175</td>
          <td>0.065446</td>
          <td>24.927439</td>
          <td>0.096611</td>
          <td>24.314076</td>
          <td>0.126232</td>
          <td>0.041438</td>
          <td>0.022488</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.234010</td>
          <td>0.305659</td>
          <td>27.178245</td>
          <td>0.245524</td>
          <td>26.357561</td>
          <td>0.107958</td>
          <td>26.287129</td>
          <td>0.163895</td>
          <td>26.163842</td>
          <td>0.272644</td>
          <td>25.332818</td>
          <td>0.292847</td>
          <td>0.003102</td>
          <td>0.002626</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.540959</td>
          <td>0.410436</td>
          <td>26.245256</td>
          <td>0.119602</td>
          <td>26.056985</td>
          <td>0.090315</td>
          <td>25.788447</td>
          <td>0.116349</td>
          <td>25.962190</td>
          <td>0.250233</td>
          <td>25.195786</td>
          <td>0.284288</td>
          <td>0.096420</td>
          <td>0.067579</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.014572</td>
          <td>0.264169</td>
          <td>26.760404</td>
          <td>0.180317</td>
          <td>26.329784</td>
          <td>0.110590</td>
          <td>26.794772</td>
          <td>0.263033</td>
          <td>25.784313</td>
          <td>0.208603</td>
          <td>26.858552</td>
          <td>0.918279</td>
          <td>0.070185</td>
          <td>0.051520</td>
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
