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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.19/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fbfe44322c0>



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
    0      23.994413  0.056906  0.052643  
    1      25.391064  0.001667  0.000939  
    2      24.304707  0.004579  0.003872  
    3      25.291103  0.105657  0.086087  
    4      25.096743  0.051853  0.050378  
    ...          ...       ...       ...  
    99995  24.737946  0.105616  0.061327  
    99996  24.224169  0.110688  0.057660  
    99997  25.613836  0.013119  0.008875  
    99998  25.274899  0.004360  0.003427  
    99999  25.699642  0.080715  0.061348  
    
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
          <td>33.101725</td>
          <td>5.651532</td>
          <td>27.483585</td>
          <td>0.314512</td>
          <td>25.978570</td>
          <td>0.077369</td>
          <td>25.229904</td>
          <td>0.065100</td>
          <td>24.672615</td>
          <td>0.076055</td>
          <td>24.086030</td>
          <td>0.101916</td>
          <td>0.056906</td>
          <td>0.052643</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.373960</td>
          <td>0.287996</td>
          <td>26.615626</td>
          <td>0.135076</td>
          <td>26.201023</td>
          <td>0.152240</td>
          <td>26.079341</td>
          <td>0.254430</td>
          <td>25.518285</td>
          <td>0.339564</td>
          <td>0.001667</td>
          <td>0.000939</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.822696</td>
          <td>3.211749</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.862591</td>
          <td>0.113608</td>
          <td>24.988991</td>
          <td>0.100472</td>
          <td>24.456898</td>
          <td>0.140671</td>
          <td>0.004579</td>
          <td>0.003872</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.806218</td>
          <td>0.475990</td>
          <td>27.889167</td>
          <td>0.431508</td>
          <td>27.418339</td>
          <td>0.265602</td>
          <td>26.270617</td>
          <td>0.161583</td>
          <td>25.509212</td>
          <td>0.157696</td>
          <td>25.428056</td>
          <td>0.316075</td>
          <td>0.105657</td>
          <td>0.086087</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.754882</td>
          <td>0.458068</td>
          <td>25.981551</td>
          <td>0.088162</td>
          <td>25.944337</td>
          <td>0.075063</td>
          <td>25.610144</td>
          <td>0.091080</td>
          <td>25.355005</td>
          <td>0.138129</td>
          <td>25.597523</td>
          <td>0.361404</td>
          <td>0.051853</td>
          <td>0.050378</td>
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
          <td>26.838099</td>
          <td>0.487399</td>
          <td>26.529622</td>
          <td>0.142070</td>
          <td>25.433259</td>
          <td>0.047709</td>
          <td>25.054797</td>
          <td>0.055733</td>
          <td>24.931599</td>
          <td>0.095541</td>
          <td>25.112609</td>
          <td>0.244692</td>
          <td>0.105616</td>
          <td>0.061327</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.642476</td>
          <td>0.156513</td>
          <td>26.027051</td>
          <td>0.080752</td>
          <td>25.231541</td>
          <td>0.065194</td>
          <td>24.716637</td>
          <td>0.079070</td>
          <td>24.180324</td>
          <td>0.110671</td>
          <td>0.110688</td>
          <td>0.057660</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.595421</td>
          <td>0.824764</td>
          <td>26.454918</td>
          <td>0.133209</td>
          <td>26.481578</td>
          <td>0.120264</td>
          <td>26.406355</td>
          <td>0.181356</td>
          <td>26.174207</td>
          <td>0.274924</td>
          <td>26.010060</td>
          <td>0.494860</td>
          <td>0.013119</td>
          <td>0.008875</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.206360</td>
          <td>0.635237</td>
          <td>26.139461</td>
          <td>0.101254</td>
          <td>26.157962</td>
          <td>0.090621</td>
          <td>25.877913</td>
          <td>0.115135</td>
          <td>25.573008</td>
          <td>0.166525</td>
          <td>25.521936</td>
          <td>0.340545</td>
          <td>0.004360</td>
          <td>0.003427</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.105729</td>
          <td>0.275605</td>
          <td>26.619477</td>
          <td>0.153462</td>
          <td>26.491531</td>
          <td>0.121309</td>
          <td>26.063756</td>
          <td>0.135276</td>
          <td>26.030748</td>
          <td>0.244468</td>
          <td>26.002014</td>
          <td>0.491922</td>
          <td>0.080715</td>
          <td>0.061348</td>
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
          <td>27.197801</td>
          <td>0.696419</td>
          <td>26.755557</td>
          <td>0.199485</td>
          <td>25.974100</td>
          <td>0.091560</td>
          <td>25.193974</td>
          <td>0.075536</td>
          <td>24.645038</td>
          <td>0.088188</td>
          <td>24.044702</td>
          <td>0.117256</td>
          <td>0.056906</td>
          <td>0.052643</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.899081</td>
          <td>0.222947</td>
          <td>26.522477</td>
          <td>0.146062</td>
          <td>26.192299</td>
          <td>0.177943</td>
          <td>26.762901</td>
          <td>0.501146</td>
          <td>25.440999</td>
          <td>0.370938</td>
          <td>0.001667</td>
          <td>0.000939</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.769639</td>
          <td>0.511358</td>
          <td>28.611976</td>
          <td>0.806848</td>
          <td>29.006622</td>
          <td>0.957957</td>
          <td>25.937918</td>
          <td>0.143188</td>
          <td>25.350862</td>
          <td>0.161157</td>
          <td>24.258323</td>
          <td>0.139649</td>
          <td>0.004579</td>
          <td>0.003872</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.633174</td>
          <td>1.615199</td>
          <td>28.196700</td>
          <td>0.622563</td>
          <td>27.396241</td>
          <td>0.311315</td>
          <td>26.109153</td>
          <td>0.170966</td>
          <td>25.445324</td>
          <td>0.179880</td>
          <td>26.319312</td>
          <td>0.722179</td>
          <td>0.105657</td>
          <td>0.086087</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.221618</td>
          <td>0.338619</td>
          <td>26.082995</td>
          <td>0.111997</td>
          <td>25.948804</td>
          <td>0.089430</td>
          <td>25.529740</td>
          <td>0.101355</td>
          <td>25.331045</td>
          <td>0.159807</td>
          <td>25.278956</td>
          <td>0.329194</td>
          <td>0.051853</td>
          <td>0.050378</td>
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
          <td>26.561964</td>
          <td>0.445162</td>
          <td>26.505937</td>
          <td>0.163558</td>
          <td>25.400884</td>
          <td>0.055980</td>
          <td>25.055903</td>
          <td>0.067872</td>
          <td>24.761388</td>
          <td>0.099097</td>
          <td>24.819967</td>
          <td>0.230187</td>
          <td>0.105616</td>
          <td>0.061327</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.468335</td>
          <td>0.158534</td>
          <td>26.058078</td>
          <td>0.100097</td>
          <td>25.121161</td>
          <td>0.071980</td>
          <td>24.955155</td>
          <td>0.117477</td>
          <td>24.487689</td>
          <td>0.174334</td>
          <td>0.110688</td>
          <td>0.057660</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.730696</td>
          <td>0.226177</td>
          <td>27.090777</td>
          <td>0.261204</td>
          <td>26.330468</td>
          <td>0.123793</td>
          <td>26.151526</td>
          <td>0.171961</td>
          <td>26.078571</td>
          <td>0.295315</td>
          <td>25.665495</td>
          <td>0.440931</td>
          <td>0.013119</td>
          <td>0.008875</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.323253</td>
          <td>0.364567</td>
          <td>26.249287</td>
          <td>0.128377</td>
          <td>26.055406</td>
          <td>0.097346</td>
          <td>25.579809</td>
          <td>0.104939</td>
          <td>25.614464</td>
          <td>0.201463</td>
          <td>24.833185</td>
          <td>0.227247</td>
          <td>0.004360</td>
          <td>0.003427</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.195182</td>
          <td>0.698083</td>
          <td>26.772338</td>
          <td>0.203548</td>
          <td>26.598792</td>
          <td>0.158574</td>
          <td>26.085933</td>
          <td>0.165378</td>
          <td>26.446421</td>
          <td>0.400625</td>
          <td>24.977048</td>
          <td>0.260108</td>
          <td>0.080715</td>
          <td>0.061348</td>
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
          <td>27.025654</td>
          <td>0.571609</td>
          <td>26.786090</td>
          <td>0.182795</td>
          <td>26.053495</td>
          <td>0.085975</td>
          <td>25.250151</td>
          <td>0.069086</td>
          <td>24.764779</td>
          <td>0.085808</td>
          <td>23.890314</td>
          <td>0.089396</td>
          <td>0.056906</td>
          <td>0.052643</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.296220</td>
          <td>0.270402</td>
          <td>26.671202</td>
          <td>0.141713</td>
          <td>26.438334</td>
          <td>0.186334</td>
          <td>26.097540</td>
          <td>0.258260</td>
          <td>25.113490</td>
          <td>0.244875</td>
          <td>0.001667</td>
          <td>0.000939</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.327900</td>
          <td>1.274025</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.712634</td>
          <td>0.336597</td>
          <td>25.873340</td>
          <td>0.114707</td>
          <td>25.018626</td>
          <td>0.103138</td>
          <td>24.464702</td>
          <td>0.141655</td>
          <td>0.004579</td>
          <td>0.003872</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.227348</td>
          <td>1.138657</td>
          <td>27.404308</td>
          <td>0.291100</td>
          <td>26.709459</td>
          <td>0.260728</td>
          <td>25.402749</td>
          <td>0.160477</td>
          <td>25.566786</td>
          <td>0.390840</td>
          <td>0.105657</td>
          <td>0.086087</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.116600</td>
          <td>0.284259</td>
          <td>26.148782</td>
          <td>0.105163</td>
          <td>25.976273</td>
          <td>0.079921</td>
          <td>25.637066</td>
          <td>0.096672</td>
          <td>25.654411</td>
          <td>0.184440</td>
          <td>25.040318</td>
          <td>0.238354</td>
          <td>0.051853</td>
          <td>0.050378</td>
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
          <td>29.158649</td>
          <td>1.969175</td>
          <td>26.345306</td>
          <td>0.131003</td>
          <td>25.447314</td>
          <td>0.052929</td>
          <td>25.087324</td>
          <td>0.063102</td>
          <td>24.879518</td>
          <td>0.099867</td>
          <td>24.603728</td>
          <td>0.174677</td>
          <td>0.105616</td>
          <td>0.061327</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.792812</td>
          <td>0.498583</td>
          <td>26.966653</td>
          <td>0.222601</td>
          <td>26.159181</td>
          <td>0.099563</td>
          <td>25.372027</td>
          <td>0.081415</td>
          <td>24.900555</td>
          <td>0.102018</td>
          <td>24.172249</td>
          <td>0.120904</td>
          <td>0.110688</td>
          <td>0.057660</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.139541</td>
          <td>0.606744</td>
          <td>26.859901</td>
          <td>0.188537</td>
          <td>26.380680</td>
          <td>0.110333</td>
          <td>26.646685</td>
          <td>0.222280</td>
          <td>25.976301</td>
          <td>0.234095</td>
          <td>25.529889</td>
          <td>0.343235</td>
          <td>0.013119</td>
          <td>0.008875</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.861955</td>
          <td>0.496138</td>
          <td>25.992229</td>
          <td>0.089009</td>
          <td>26.134627</td>
          <td>0.088799</td>
          <td>25.752889</td>
          <td>0.103252</td>
          <td>25.546933</td>
          <td>0.162895</td>
          <td>24.936799</td>
          <td>0.211518</td>
          <td>0.004360</td>
          <td>0.003427</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.616684</td>
          <td>0.429283</td>
          <td>26.893830</td>
          <td>0.204541</td>
          <td>26.759401</td>
          <td>0.162843</td>
          <td>26.162688</td>
          <td>0.157415</td>
          <td>26.081546</td>
          <td>0.270734</td>
          <td>25.661476</td>
          <td>0.403041</td>
          <td>0.080715</td>
          <td>0.061348</td>
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
