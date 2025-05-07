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

    <pzflow.flow.Flow at 0x7fa41e5b1750>



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
          <td>28.237449</td>
          <td>1.212254</td>
          <td>26.447201</td>
          <td>0.132323</td>
          <td>25.960414</td>
          <td>0.076138</td>
          <td>25.168258</td>
          <td>0.061637</td>
          <td>24.740731</td>
          <td>0.080769</td>
          <td>24.040310</td>
          <td>0.097914</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.562497</td>
          <td>0.334882</td>
          <td>26.620979</td>
          <td>0.135702</td>
          <td>26.155156</td>
          <td>0.146361</td>
          <td>25.904470</td>
          <td>0.220191</td>
          <td>25.179814</td>
          <td>0.258576</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.258279</td>
          <td>0.566790</td>
          <td>27.813551</td>
          <td>0.364331</td>
          <td>26.310188</td>
          <td>0.167130</td>
          <td>24.900328</td>
          <td>0.092954</td>
          <td>24.343544</td>
          <td>0.127546</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.026012</td>
          <td>2.667215</td>
          <td>29.065900</td>
          <td>0.969290</td>
          <td>27.659226</td>
          <td>0.322554</td>
          <td>26.587400</td>
          <td>0.211200</td>
          <td>25.847726</td>
          <td>0.210008</td>
          <td>24.980442</td>
          <td>0.219317</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.917180</td>
          <td>0.236193</td>
          <td>25.992228</td>
          <td>0.088993</td>
          <td>25.980817</td>
          <td>0.077522</td>
          <td>25.738957</td>
          <td>0.101978</td>
          <td>25.550095</td>
          <td>0.163302</td>
          <td>25.148683</td>
          <td>0.252060</td>
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
          <td>29.114212</td>
          <td>1.871537</td>
          <td>26.493897</td>
          <td>0.137765</td>
          <td>25.404064</td>
          <td>0.046489</td>
          <td>25.168488</td>
          <td>0.061649</td>
          <td>24.911678</td>
          <td>0.093885</td>
          <td>24.887149</td>
          <td>0.202863</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>36.768789</td>
          <td>9.312821</td>
          <td>26.635222</td>
          <td>0.155545</td>
          <td>25.952193</td>
          <td>0.075586</td>
          <td>25.139358</td>
          <td>0.060077</td>
          <td>24.782452</td>
          <td>0.083796</td>
          <td>24.217325</td>
          <td>0.114299</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.727537</td>
          <td>0.448746</td>
          <td>26.527103</td>
          <td>0.141762</td>
          <td>26.226135</td>
          <td>0.096213</td>
          <td>26.130495</td>
          <td>0.143289</td>
          <td>25.896374</td>
          <td>0.218711</td>
          <td>25.367254</td>
          <td>0.301048</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.150677</td>
          <td>0.285820</td>
          <td>26.501927</td>
          <td>0.138722</td>
          <td>26.160315</td>
          <td>0.090809</td>
          <td>25.963887</td>
          <td>0.124070</td>
          <td>25.703012</td>
          <td>0.185950</td>
          <td>25.493562</td>
          <td>0.332984</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.935985</td>
          <td>0.523786</td>
          <td>26.720309</td>
          <td>0.167261</td>
          <td>26.582419</td>
          <td>0.131254</td>
          <td>26.003214</td>
          <td>0.128374</td>
          <td>25.899501</td>
          <td>0.219282</td>
          <td>25.484181</td>
          <td>0.330515</td>
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
          <td>26.528431</td>
          <td>0.427020</td>
          <td>26.890238</td>
          <td>0.221319</td>
          <td>25.955491</td>
          <td>0.089168</td>
          <td>25.129980</td>
          <td>0.070633</td>
          <td>24.686293</td>
          <td>0.090526</td>
          <td>23.880212</td>
          <td>0.100532</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.710540</td>
          <td>0.489618</td>
          <td>27.503142</td>
          <td>0.363281</td>
          <td>26.648231</td>
          <td>0.162710</td>
          <td>26.365802</td>
          <td>0.206012</td>
          <td>25.469404</td>
          <td>0.178291</td>
          <td>25.945118</td>
          <td>0.542339</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.208726</td>
          <td>1.174789</td>
          <td>27.813747</td>
          <td>0.428174</td>
          <td>25.697410</td>
          <td>0.118960</td>
          <td>25.088295</td>
          <td>0.131429</td>
          <td>24.256471</td>
          <td>0.142590</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.234184</td>
          <td>0.656052</td>
          <td>27.602944</td>
          <td>0.378663</td>
          <td>26.423706</td>
          <td>0.230843</td>
          <td>25.560389</td>
          <td>0.205258</td>
          <td>24.673793</td>
          <td>0.212424</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.746230</td>
          <td>0.502729</td>
          <td>26.134082</td>
          <td>0.116203</td>
          <td>25.912647</td>
          <td>0.085897</td>
          <td>25.639837</td>
          <td>0.110621</td>
          <td>25.518205</td>
          <td>0.185830</td>
          <td>25.189076</td>
          <td>0.303990</td>
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
          <td>27.702695</td>
          <td>0.970096</td>
          <td>26.508092</td>
          <td>0.163340</td>
          <td>25.373060</td>
          <td>0.054417</td>
          <td>25.252797</td>
          <td>0.080469</td>
          <td>24.740237</td>
          <td>0.096929</td>
          <td>24.857744</td>
          <td>0.236675</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.423531</td>
          <td>0.395161</td>
          <td>26.698331</td>
          <td>0.189136</td>
          <td>25.969485</td>
          <td>0.090649</td>
          <td>25.236667</td>
          <td>0.077955</td>
          <td>24.719000</td>
          <td>0.093553</td>
          <td>24.064045</td>
          <td>0.118528</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.538107</td>
          <td>1.530611</td>
          <td>26.695299</td>
          <td>0.190067</td>
          <td>26.195459</td>
          <td>0.111436</td>
          <td>26.205235</td>
          <td>0.182218</td>
          <td>25.762067</td>
          <td>0.230631</td>
          <td>26.727713</td>
          <td>0.927686</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.406505</td>
          <td>0.397213</td>
          <td>26.127224</td>
          <td>0.118774</td>
          <td>26.054250</td>
          <td>0.100344</td>
          <td>25.809899</td>
          <td>0.132372</td>
          <td>25.437834</td>
          <td>0.178885</td>
          <td>25.307464</td>
          <td>0.343818</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.853628</td>
          <td>0.547049</td>
          <td>26.636580</td>
          <td>0.180439</td>
          <td>26.683771</td>
          <td>0.169322</td>
          <td>26.347255</td>
          <td>0.204813</td>
          <td>26.136329</td>
          <td>0.312063</td>
          <td>25.244377</td>
          <td>0.320634</td>
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
          <td>26.655419</td>
          <td>0.424929</td>
          <td>27.063249</td>
          <td>0.223251</td>
          <td>26.133766</td>
          <td>0.088725</td>
          <td>25.097685</td>
          <td>0.057904</td>
          <td>24.673320</td>
          <td>0.076112</td>
          <td>24.115370</td>
          <td>0.104581</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.054244</td>
          <td>0.221735</td>
          <td>26.608977</td>
          <td>0.134428</td>
          <td>26.207288</td>
          <td>0.153208</td>
          <td>25.850719</td>
          <td>0.210724</td>
          <td>25.220663</td>
          <td>0.267597</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.475740</td>
          <td>0.796837</td>
          <td>28.838189</td>
          <td>0.885534</td>
          <td>27.923831</td>
          <td>0.426247</td>
          <td>26.186842</td>
          <td>0.163465</td>
          <td>24.989886</td>
          <td>0.109081</td>
          <td>24.373254</td>
          <td>0.142234</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.646582</td>
          <td>2.339543</td>
          <td>28.246298</td>
          <td>0.608653</td>
          <td>26.109673</td>
          <td>0.176764</td>
          <td>25.884943</td>
          <td>0.267551</td>
          <td>25.163381</td>
          <td>0.315931</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.444990</td>
          <td>0.361519</td>
          <td>26.268448</td>
          <td>0.113457</td>
          <td>26.027462</td>
          <td>0.080896</td>
          <td>25.627028</td>
          <td>0.092580</td>
          <td>25.253308</td>
          <td>0.126678</td>
          <td>25.363847</td>
          <td>0.300630</td>
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
          <td>27.819080</td>
          <td>0.986869</td>
          <td>26.801713</td>
          <td>0.191524</td>
          <td>25.491233</td>
          <td>0.054400</td>
          <td>25.095725</td>
          <td>0.062817</td>
          <td>24.837713</td>
          <td>0.095181</td>
          <td>24.824281</td>
          <td>0.208020</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.495845</td>
          <td>0.139950</td>
          <td>26.118887</td>
          <td>0.089017</td>
          <td>25.316080</td>
          <td>0.071499</td>
          <td>24.891304</td>
          <td>0.093751</td>
          <td>24.001488</td>
          <td>0.096269</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.470098</td>
          <td>0.379762</td>
          <td>26.855006</td>
          <td>0.195336</td>
          <td>26.552796</td>
          <td>0.134246</td>
          <td>26.430903</td>
          <td>0.194498</td>
          <td>25.990818</td>
          <td>0.247639</td>
          <td>25.670410</td>
          <td>0.399976</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.861935</td>
          <td>1.030891</td>
          <td>26.384862</td>
          <td>0.138467</td>
          <td>26.024337</td>
          <td>0.090384</td>
          <td>25.973566</td>
          <td>0.140771</td>
          <td>25.871666</td>
          <td>0.238735</td>
          <td>25.158591</td>
          <td>0.283679</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.240731</td>
          <td>0.664661</td>
          <td>26.837921</td>
          <td>0.190928</td>
          <td>26.569888</td>
          <td>0.134909</td>
          <td>26.084921</td>
          <td>0.143394</td>
          <td>26.151481</td>
          <td>0.279763</td>
          <td>25.506249</td>
          <td>0.348766</td>
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
