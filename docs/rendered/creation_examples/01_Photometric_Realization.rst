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

    <pzflow.flow.Flow at 0x7f94a20d9930>



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
    0      23.994413  0.151775  0.149760  
    1      25.391064  0.123072  0.064907  
    2      24.304707  0.011014  0.008170  
    3      25.291103  0.059330  0.056816  
    4      25.096743  0.002767  0.001882  
    ...          ...       ...       ...  
    99995  24.737946  0.186307  0.159794  
    99996  24.224169  0.045070  0.035462  
    99997  25.613836  0.031065  0.018584  
    99998  25.274899  0.020376  0.018827  
    99999  25.699642  0.170531  0.123937  
    
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
          <td>27.176953</td>
          <td>0.622321</td>
          <td>27.015382</td>
          <td>0.214505</td>
          <td>25.934507</td>
          <td>0.074414</td>
          <td>25.178401</td>
          <td>0.062194</td>
          <td>24.687732</td>
          <td>0.077077</td>
          <td>23.976825</td>
          <td>0.092608</td>
          <td>0.151775</td>
          <td>0.149760</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.537348</td>
          <td>0.794265</td>
          <td>27.255525</td>
          <td>0.261567</td>
          <td>26.455685</td>
          <td>0.117586</td>
          <td>26.180231</td>
          <td>0.149548</td>
          <td>26.767907</td>
          <td>0.438548</td>
          <td>24.849610</td>
          <td>0.196566</td>
          <td>0.123072</td>
          <td>0.064907</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.363821</td>
          <td>0.254013</td>
          <td>26.020186</td>
          <td>0.130274</td>
          <td>24.942582</td>
          <td>0.096466</td>
          <td>24.543659</td>
          <td>0.151565</td>
          <td>0.011014</td>
          <td>0.008170</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.100071</td>
          <td>1.121827</td>
          <td>30.591937</td>
          <td>2.093717</td>
          <td>27.466350</td>
          <td>0.276193</td>
          <td>26.367856</td>
          <td>0.175532</td>
          <td>25.654276</td>
          <td>0.178435</td>
          <td>25.438296</td>
          <td>0.318668</td>
          <td>0.059330</td>
          <td>0.056816</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.287328</td>
          <td>0.318934</td>
          <td>26.217353</td>
          <td>0.108383</td>
          <td>25.911389</td>
          <td>0.072908</td>
          <td>25.738947</td>
          <td>0.101977</td>
          <td>25.341185</td>
          <td>0.136492</td>
          <td>25.160316</td>
          <td>0.254477</td>
          <td>0.002767</td>
          <td>0.001882</td>
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
          <td>28.362556</td>
          <td>1.297938</td>
          <td>26.522678</td>
          <td>0.141223</td>
          <td>25.411024</td>
          <td>0.046777</td>
          <td>25.096761</td>
          <td>0.057848</td>
          <td>24.805600</td>
          <td>0.085522</td>
          <td>24.677456</td>
          <td>0.169920</td>
          <td>0.186307</td>
          <td>0.159794</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.844661</td>
          <td>0.489774</td>
          <td>26.819758</td>
          <td>0.181992</td>
          <td>25.913913</td>
          <td>0.073071</td>
          <td>25.184365</td>
          <td>0.062524</td>
          <td>24.784380</td>
          <td>0.083938</td>
          <td>24.290503</td>
          <td>0.121811</td>
          <td>0.045070</td>
          <td>0.035462</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.852300</td>
          <td>0.492551</td>
          <td>26.768658</td>
          <td>0.174279</td>
          <td>26.391140</td>
          <td>0.111157</td>
          <td>26.469378</td>
          <td>0.191276</td>
          <td>25.859476</td>
          <td>0.212081</td>
          <td>25.947356</td>
          <td>0.472334</td>
          <td>0.031065</td>
          <td>0.018584</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.518620</td>
          <td>0.169147</td>
          <td>26.433007</td>
          <td>0.130710</td>
          <td>25.938787</td>
          <td>0.074696</td>
          <td>25.721970</td>
          <td>0.100472</td>
          <td>26.373706</td>
          <td>0.322799</td>
          <td>25.416531</td>
          <td>0.313178</td>
          <td>0.020376</td>
          <td>0.018827</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.190861</td>
          <td>1.181150</td>
          <td>27.278245</td>
          <td>0.266465</td>
          <td>26.541991</td>
          <td>0.126738</td>
          <td>26.550192</td>
          <td>0.204724</td>
          <td>26.014956</td>
          <td>0.241306</td>
          <td>27.015105</td>
          <td>0.975922</td>
          <td>0.170531</td>
          <td>0.123937</td>
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
          <td>28.604782</td>
          <td>1.621427</td>
          <td>26.719874</td>
          <td>0.204246</td>
          <td>26.010874</td>
          <td>0.100575</td>
          <td>25.177567</td>
          <td>0.079379</td>
          <td>24.531521</td>
          <td>0.084896</td>
          <td>24.034890</td>
          <td>0.123771</td>
          <td>0.151775</td>
          <td>0.149760</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.764782</td>
          <td>0.455171</td>
          <td>26.549313</td>
          <td>0.154148</td>
          <td>25.929687</td>
          <td>0.146789</td>
          <td>26.013335</td>
          <td>0.288297</td>
          <td>26.726525</td>
          <td>0.939635</td>
          <td>0.123072</td>
          <td>0.064907</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.181588</td>
          <td>0.684800</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.032845</td>
          <td>0.495267</td>
          <td>25.964381</td>
          <td>0.146522</td>
          <td>25.053685</td>
          <td>0.124811</td>
          <td>24.470986</td>
          <td>0.167606</td>
          <td>0.011014</td>
          <td>0.008170</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.524049</td>
          <td>0.767507</td>
          <td>27.186048</td>
          <td>0.257980</td>
          <td>26.169595</td>
          <td>0.176542</td>
          <td>25.737892</td>
          <td>0.225742</td>
          <td>25.164110</td>
          <td>0.301052</td>
          <td>0.059330</td>
          <td>0.056816</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.633921</td>
          <td>0.462400</td>
          <td>26.271777</td>
          <td>0.130894</td>
          <td>25.882679</td>
          <td>0.083631</td>
          <td>25.525208</td>
          <td>0.100039</td>
          <td>25.192615</td>
          <td>0.140695</td>
          <td>25.037245</td>
          <td>0.268772</td>
          <td>0.002767</td>
          <td>0.001882</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.204713</td>
          <td>0.134233</td>
          <td>25.331205</td>
          <td>0.056390</td>
          <td>25.131779</td>
          <td>0.077915</td>
          <td>24.780675</td>
          <td>0.107867</td>
          <td>24.931186</td>
          <td>0.269260</td>
          <td>0.186307</td>
          <td>0.159794</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.599269</td>
          <td>0.452156</td>
          <td>26.815800</td>
          <td>0.208988</td>
          <td>26.026449</td>
          <td>0.095430</td>
          <td>25.270265</td>
          <td>0.080415</td>
          <td>24.944299</td>
          <td>0.114084</td>
          <td>24.169278</td>
          <td>0.130036</td>
          <td>0.045070</td>
          <td>0.035462</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.335444</td>
          <td>0.368592</td>
          <td>26.740809</td>
          <td>0.195689</td>
          <td>26.309213</td>
          <td>0.121746</td>
          <td>26.422095</td>
          <td>0.216360</td>
          <td>25.707502</td>
          <td>0.218213</td>
          <td>27.502018</td>
          <td>1.427289</td>
          <td>0.031065</td>
          <td>0.018584</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.500139</td>
          <td>0.418268</td>
          <td>26.217787</td>
          <td>0.125063</td>
          <td>25.977525</td>
          <td>0.091028</td>
          <td>25.986341</td>
          <td>0.149462</td>
          <td>25.696510</td>
          <td>0.216035</td>
          <td>25.210112</td>
          <td>0.309436</td>
          <td>0.020376</td>
          <td>0.018827</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.282849</td>
          <td>0.370608</td>
          <td>26.942023</td>
          <td>0.245226</td>
          <td>26.809039</td>
          <td>0.199399</td>
          <td>26.267321</td>
          <td>0.203162</td>
          <td>25.805718</td>
          <td>0.252230</td>
          <td>25.658316</td>
          <td>0.466130</td>
          <td>0.170531</td>
          <td>0.123937</td>
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
          <td>26.624516</td>
          <td>0.482899</td>
          <td>26.731786</td>
          <td>0.207191</td>
          <td>26.042524</td>
          <td>0.103914</td>
          <td>25.161316</td>
          <td>0.078654</td>
          <td>24.714442</td>
          <td>0.100185</td>
          <td>23.932525</td>
          <td>0.113808</td>
          <td>0.151775</td>
          <td>0.149760</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.003628</td>
          <td>1.856706</td>
          <td>27.331163</td>
          <td>0.304774</td>
          <td>26.659548</td>
          <td>0.156628</td>
          <td>26.192427</td>
          <td>0.169425</td>
          <td>25.756694</td>
          <td>0.216602</td>
          <td>25.435070</td>
          <td>0.353013</td>
          <td>0.123072</td>
          <td>0.064907</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.193566</td>
          <td>1.183611</td>
          <td>28.140465</td>
          <td>0.520911</td>
          <td>27.679012</td>
          <td>0.328048</td>
          <td>25.979638</td>
          <td>0.125944</td>
          <td>25.080383</td>
          <td>0.108969</td>
          <td>24.304201</td>
          <td>0.123430</td>
          <td>0.011014</td>
          <td>0.008170</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.409498</td>
          <td>0.274619</td>
          <td>26.072034</td>
          <td>0.142525</td>
          <td>25.441096</td>
          <td>0.155243</td>
          <td>25.258069</td>
          <td>0.287417</td>
          <td>0.059330</td>
          <td>0.056816</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.026564</td>
          <td>0.559336</td>
          <td>26.217417</td>
          <td>0.108396</td>
          <td>26.027548</td>
          <td>0.080793</td>
          <td>25.534231</td>
          <td>0.085200</td>
          <td>25.428436</td>
          <td>0.147155</td>
          <td>25.610874</td>
          <td>0.365224</td>
          <td>0.002767</td>
          <td>0.001882</td>
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
          <td>26.977387</td>
          <td>0.645571</td>
          <td>26.444922</td>
          <td>0.170965</td>
          <td>25.419189</td>
          <td>0.063493</td>
          <td>25.092530</td>
          <td>0.078454</td>
          <td>24.895299</td>
          <td>0.124021</td>
          <td>24.588831</td>
          <td>0.210948</td>
          <td>0.186307</td>
          <td>0.159794</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.875068</td>
          <td>0.507319</td>
          <td>26.496849</td>
          <td>0.140712</td>
          <td>26.037046</td>
          <td>0.083273</td>
          <td>25.242589</td>
          <td>0.067376</td>
          <td>24.802396</td>
          <td>0.087167</td>
          <td>24.197992</td>
          <td>0.114946</td>
          <td>0.045070</td>
          <td>0.035462</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.473684</td>
          <td>0.765229</td>
          <td>26.645471</td>
          <td>0.158086</td>
          <td>26.196466</td>
          <td>0.094569</td>
          <td>26.190188</td>
          <td>0.152199</td>
          <td>25.723253</td>
          <td>0.190760</td>
          <td>26.665656</td>
          <td>0.787887</td>
          <td>0.031065</td>
          <td>0.018584</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.508987</td>
          <td>0.380878</td>
          <td>26.300294</td>
          <td>0.117024</td>
          <td>26.045723</td>
          <td>0.082523</td>
          <td>26.065569</td>
          <td>0.136217</td>
          <td>25.422116</td>
          <td>0.147093</td>
          <td>25.297612</td>
          <td>0.286024</td>
          <td>0.020376</td>
          <td>0.018827</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.092567</td>
          <td>0.671833</td>
          <td>26.767919</td>
          <td>0.212177</td>
          <td>26.948548</td>
          <td>0.223889</td>
          <td>26.152880</td>
          <td>0.184341</td>
          <td>25.862641</td>
          <td>0.264076</td>
          <td>25.192441</td>
          <td>0.324969</td>
          <td>0.170531</td>
          <td>0.123937</td>
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
