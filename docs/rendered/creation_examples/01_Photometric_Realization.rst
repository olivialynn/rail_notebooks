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

    <pzflow.flow.Flow at 0x7ff03e82eb00>



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
          <td>26.763831</td>
          <td>0.461153</td>
          <td>26.505653</td>
          <td>0.139168</td>
          <td>25.941791</td>
          <td>0.074895</td>
          <td>25.057023</td>
          <td>0.055843</td>
          <td>24.639027</td>
          <td>0.073830</td>
          <td>23.949548</td>
          <td>0.090414</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.425316</td>
          <td>1.342077</td>
          <td>27.404519</td>
          <td>0.295184</td>
          <td>26.847056</td>
          <td>0.164766</td>
          <td>26.050314</td>
          <td>0.133714</td>
          <td>26.033190</td>
          <td>0.244960</td>
          <td>25.531074</td>
          <td>0.343011</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.135695</td>
          <td>1.011034</td>
          <td>28.729084</td>
          <td>0.711803</td>
          <td>26.246703</td>
          <td>0.158313</td>
          <td>25.020611</td>
          <td>0.103292</td>
          <td>24.354499</td>
          <td>0.128762</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.797417</td>
          <td>1.457422</td>
          <td>27.099790</td>
          <td>0.204040</td>
          <td>25.842557</td>
          <td>0.111641</td>
          <td>25.386503</td>
          <td>0.141931</td>
          <td>25.009144</td>
          <td>0.224617</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.259668</td>
          <td>0.311975</td>
          <td>26.105579</td>
          <td>0.098296</td>
          <td>25.931222</td>
          <td>0.074198</td>
          <td>25.967638</td>
          <td>0.124475</td>
          <td>25.496676</td>
          <td>0.156014</td>
          <td>25.540263</td>
          <td>0.345506</td>
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
          <td>29.198461</td>
          <td>1.941218</td>
          <td>26.535900</td>
          <td>0.142839</td>
          <td>25.448306</td>
          <td>0.048351</td>
          <td>25.012161</td>
          <td>0.053663</td>
          <td>24.887995</td>
          <td>0.091952</td>
          <td>25.020019</td>
          <td>0.226655</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.939853</td>
          <td>0.525267</td>
          <td>26.729925</td>
          <td>0.168636</td>
          <td>25.997013</td>
          <td>0.078639</td>
          <td>25.166138</td>
          <td>0.061521</td>
          <td>24.836406</td>
          <td>0.087873</td>
          <td>24.155678</td>
          <td>0.108315</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.726028</td>
          <td>0.168077</td>
          <td>26.319962</td>
          <td>0.104456</td>
          <td>26.295623</td>
          <td>0.165068</td>
          <td>25.838642</td>
          <td>0.208418</td>
          <td>25.332871</td>
          <td>0.292828</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.197978</td>
          <td>0.631536</td>
          <td>26.309228</td>
          <td>0.117410</td>
          <td>26.108211</td>
          <td>0.086740</td>
          <td>25.841206</td>
          <td>0.111510</td>
          <td>25.726400</td>
          <td>0.189659</td>
          <td>25.210150</td>
          <td>0.265069</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.107177</td>
          <td>0.592456</td>
          <td>26.788903</td>
          <td>0.177298</td>
          <td>26.550991</td>
          <td>0.127731</td>
          <td>26.328077</td>
          <td>0.169696</td>
          <td>26.066498</td>
          <td>0.251763</td>
          <td>26.755278</td>
          <td>0.829475</td>
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
          <td>29.421870</td>
          <td>2.245853</td>
          <td>26.987043</td>
          <td>0.239796</td>
          <td>25.872173</td>
          <td>0.082862</td>
          <td>25.166447</td>
          <td>0.072948</td>
          <td>24.631997</td>
          <td>0.086304</td>
          <td>23.991913</td>
          <td>0.110840</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.547950</td>
          <td>0.376199</td>
          <td>26.667619</td>
          <td>0.165423</td>
          <td>26.181987</td>
          <td>0.176434</td>
          <td>25.629922</td>
          <td>0.204128</td>
          <td>25.012374</td>
          <td>0.263428</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.350203</td>
          <td>0.776611</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.804615</td>
          <td>0.425208</td>
          <td>25.844582</td>
          <td>0.135139</td>
          <td>24.866820</td>
          <td>0.108420</td>
          <td>24.320614</td>
          <td>0.150669</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.469255</td>
          <td>0.768945</td>
          <td>27.905696</td>
          <td>0.476829</td>
          <td>26.272927</td>
          <td>0.203577</td>
          <td>26.336131</td>
          <td>0.384413</td>
          <td>25.810436</td>
          <td>0.520452</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.979392</td>
          <td>0.277312</td>
          <td>26.101102</td>
          <td>0.112917</td>
          <td>25.837890</td>
          <td>0.080421</td>
          <td>25.551890</td>
          <td>0.102439</td>
          <td>25.481764</td>
          <td>0.180188</td>
          <td>25.655220</td>
          <td>0.437487</td>
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
          <td>26.615636</td>
          <td>0.462387</td>
          <td>26.457553</td>
          <td>0.156442</td>
          <td>25.442863</td>
          <td>0.057893</td>
          <td>25.146915</td>
          <td>0.073287</td>
          <td>24.646257</td>
          <td>0.089252</td>
          <td>24.702884</td>
          <td>0.208073</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.657453</td>
          <td>0.471880</td>
          <td>26.914580</td>
          <td>0.226644</td>
          <td>26.007147</td>
          <td>0.093699</td>
          <td>25.180482</td>
          <td>0.074181</td>
          <td>24.853226</td>
          <td>0.105226</td>
          <td>24.164610</td>
          <td>0.129332</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.667365</td>
          <td>0.185640</td>
          <td>26.448158</td>
          <td>0.138741</td>
          <td>26.020273</td>
          <td>0.155672</td>
          <td>25.661349</td>
          <td>0.212091</td>
          <td>25.126576</td>
          <td>0.292489</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.189106</td>
          <td>0.335222</td>
          <td>26.044049</td>
          <td>0.110485</td>
          <td>26.150650</td>
          <td>0.109167</td>
          <td>25.940408</td>
          <td>0.148127</td>
          <td>26.268634</td>
          <td>0.353314</td>
          <td>24.687412</td>
          <td>0.207498</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.215969</td>
          <td>0.705025</td>
          <td>26.775208</td>
          <td>0.202789</td>
          <td>26.889650</td>
          <td>0.201508</td>
          <td>26.540180</td>
          <td>0.240462</td>
          <td>25.808236</td>
          <td>0.238990</td>
          <td>25.817714</td>
          <td>0.498248</td>
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
          <td>27.114357</td>
          <td>0.595522</td>
          <td>26.601736</td>
          <td>0.151164</td>
          <td>25.953015</td>
          <td>0.075651</td>
          <td>25.163968</td>
          <td>0.061411</td>
          <td>24.662651</td>
          <td>0.075398</td>
          <td>24.129406</td>
          <td>0.105872</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.629756</td>
          <td>0.353384</td>
          <td>26.705277</td>
          <td>0.146062</td>
          <td>26.303771</td>
          <td>0.166379</td>
          <td>25.769042</td>
          <td>0.196774</td>
          <td>25.344739</td>
          <td>0.295910</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.535445</td>
          <td>2.115355</td>
          <td>27.963742</td>
          <td>0.439361</td>
          <td>25.984683</td>
          <td>0.137429</td>
          <td>25.196009</td>
          <td>0.130484</td>
          <td>24.411628</td>
          <td>0.147007</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.800191</td>
          <td>0.544539</td>
          <td>29.516137</td>
          <td>1.418323</td>
          <td>27.115032</td>
          <td>0.255544</td>
          <td>26.617635</td>
          <td>0.269792</td>
          <td>25.667653</td>
          <td>0.223718</td>
          <td>25.023315</td>
          <td>0.282269</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.957784</td>
          <td>0.244452</td>
          <td>26.107764</td>
          <td>0.098605</td>
          <td>25.870628</td>
          <td>0.070426</td>
          <td>25.701375</td>
          <td>0.098822</td>
          <td>25.584072</td>
          <td>0.168333</td>
          <td>24.708307</td>
          <td>0.174685</td>
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
          <td>31.257947</td>
          <td>3.895719</td>
          <td>26.314716</td>
          <td>0.126295</td>
          <td>25.481092</td>
          <td>0.053912</td>
          <td>25.044894</td>
          <td>0.060049</td>
          <td>24.830848</td>
          <td>0.094609</td>
          <td>24.900708</td>
          <td>0.221718</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.837245</td>
          <td>0.491825</td>
          <td>26.540448</td>
          <td>0.145424</td>
          <td>26.030305</td>
          <td>0.082336</td>
          <td>25.276902</td>
          <td>0.069062</td>
          <td>24.850802</td>
          <td>0.090473</td>
          <td>24.335085</td>
          <td>0.128770</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.368542</td>
          <td>0.350814</td>
          <td>26.645937</td>
          <td>0.163635</td>
          <td>26.315360</td>
          <td>0.109227</td>
          <td>26.141116</td>
          <td>0.152033</td>
          <td>26.390876</td>
          <td>0.341964</td>
          <td>25.606327</td>
          <td>0.380640</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.534952</td>
          <td>0.416258</td>
          <td>25.956495</td>
          <td>0.095409</td>
          <td>26.119827</td>
          <td>0.098287</td>
          <td>25.828031</td>
          <td>0.124126</td>
          <td>25.424210</td>
          <td>0.163929</td>
          <td>24.899450</td>
          <td>0.229390</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.785665</td>
          <td>0.479714</td>
          <td>27.135222</td>
          <td>0.244608</td>
          <td>26.540484</td>
          <td>0.131523</td>
          <td>26.363734</td>
          <td>0.181940</td>
          <td>25.692838</td>
          <td>0.191373</td>
          <td>26.572097</td>
          <td>0.758330</td>
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
