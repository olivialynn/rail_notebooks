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

    <pzflow.flow.Flow at 0x7fb56879cdf0>



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
          <td>26.651919</td>
          <td>0.423765</td>
          <td>26.850358</td>
          <td>0.186760</td>
          <td>25.972235</td>
          <td>0.076937</td>
          <td>25.243608</td>
          <td>0.065895</td>
          <td>24.663991</td>
          <td>0.075478</td>
          <td>24.021662</td>
          <td>0.096326</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.207820</td>
          <td>0.635884</td>
          <td>27.589642</td>
          <td>0.342146</td>
          <td>26.519139</td>
          <td>0.124251</td>
          <td>26.133567</td>
          <td>0.143668</td>
          <td>26.010352</td>
          <td>0.240391</td>
          <td>25.172590</td>
          <td>0.257051</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.721233</td>
          <td>0.893487</td>
          <td>27.624827</td>
          <td>0.351760</td>
          <td>28.484938</td>
          <td>0.601247</td>
          <td>25.904029</td>
          <td>0.117782</td>
          <td>24.971653</td>
          <td>0.098957</td>
          <td>24.304878</td>
          <td>0.123340</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.083650</td>
          <td>0.201296</td>
          <td>26.153852</td>
          <td>0.146197</td>
          <td>25.780003</td>
          <td>0.198416</td>
          <td>25.667238</td>
          <td>0.381588</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.843609</td>
          <td>0.489393</td>
          <td>26.372771</td>
          <td>0.124069</td>
          <td>26.073365</td>
          <td>0.084118</td>
          <td>25.836686</td>
          <td>0.111071</td>
          <td>25.479015</td>
          <td>0.153672</td>
          <td>25.374360</td>
          <td>0.302772</td>
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
          <td>27.415305</td>
          <td>0.732697</td>
          <td>26.300324</td>
          <td>0.116505</td>
          <td>25.501211</td>
          <td>0.050677</td>
          <td>25.286736</td>
          <td>0.068461</td>
          <td>24.738899</td>
          <td>0.080639</td>
          <td>24.804024</td>
          <td>0.189160</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.808001</td>
          <td>0.180190</td>
          <td>26.167880</td>
          <td>0.091415</td>
          <td>25.176971</td>
          <td>0.062115</td>
          <td>24.870934</td>
          <td>0.090583</td>
          <td>24.104961</td>
          <td>0.103619</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.831647</td>
          <td>1.645109</td>
          <td>27.233576</td>
          <td>0.256912</td>
          <td>26.571317</td>
          <td>0.129999</td>
          <td>26.180454</td>
          <td>0.149576</td>
          <td>26.445842</td>
          <td>0.341800</td>
          <td>25.580308</td>
          <td>0.356560</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.733078</td>
          <td>0.450622</td>
          <td>25.995909</td>
          <td>0.089281</td>
          <td>26.003324</td>
          <td>0.079078</td>
          <td>25.903929</td>
          <td>0.117772</td>
          <td>25.909154</td>
          <td>0.221051</td>
          <td>24.942788</td>
          <td>0.212536</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.108961</td>
          <td>0.593206</td>
          <td>26.876896</td>
          <td>0.190987</td>
          <td>26.798404</td>
          <td>0.158059</td>
          <td>26.249942</td>
          <td>0.158753</td>
          <td>26.051944</td>
          <td>0.248770</td>
          <td>25.328866</td>
          <td>0.291884</td>
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
          <td>26.815649</td>
          <td>0.528837</td>
          <td>26.728340</td>
          <td>0.193280</td>
          <td>26.052524</td>
          <td>0.097098</td>
          <td>25.256520</td>
          <td>0.078989</td>
          <td>24.888580</td>
          <td>0.108077</td>
          <td>23.991361</td>
          <td>0.110787</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.439597</td>
          <td>0.345604</td>
          <td>26.500750</td>
          <td>0.143390</td>
          <td>26.286617</td>
          <td>0.192754</td>
          <td>25.875472</td>
          <td>0.250283</td>
          <td>25.785646</td>
          <td>0.482427</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.961586</td>
          <td>1.131061</td>
          <td>28.418363</td>
          <td>0.720710</td>
          <td>27.380205</td>
          <td>0.305029</td>
          <td>25.937200</td>
          <td>0.146364</td>
          <td>24.760278</td>
          <td>0.098774</td>
          <td>24.467274</td>
          <td>0.170777</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.687198</td>
          <td>0.502597</td>
          <td>28.304369</td>
          <td>0.688430</td>
          <td>27.542992</td>
          <td>0.361364</td>
          <td>26.537890</td>
          <td>0.253631</td>
          <td>25.437265</td>
          <td>0.185055</td>
          <td>26.199993</td>
          <td>0.685601</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.065440</td>
          <td>0.297257</td>
          <td>26.218188</td>
          <td>0.125001</td>
          <td>25.800733</td>
          <td>0.077827</td>
          <td>25.803911</td>
          <td>0.127580</td>
          <td>25.452176</td>
          <td>0.175724</td>
          <td>25.663124</td>
          <td>0.440113</td>
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
          <td>27.283069</td>
          <td>0.742273</td>
          <td>26.389658</td>
          <td>0.147604</td>
          <td>25.431196</td>
          <td>0.057297</td>
          <td>25.021850</td>
          <td>0.065610</td>
          <td>24.818116</td>
          <td>0.103769</td>
          <td>24.928102</td>
          <td>0.250800</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.842814</td>
          <td>1.045943</td>
          <td>26.890796</td>
          <td>0.222211</td>
          <td>26.145003</td>
          <td>0.105725</td>
          <td>25.300197</td>
          <td>0.082448</td>
          <td>24.700827</td>
          <td>0.092072</td>
          <td>24.198581</td>
          <td>0.133188</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.325684</td>
          <td>0.759969</td>
          <td>26.837969</td>
          <td>0.214222</td>
          <td>26.344202</td>
          <td>0.126818</td>
          <td>26.296373</td>
          <td>0.196782</td>
          <td>25.680461</td>
          <td>0.215502</td>
          <td>25.370644</td>
          <td>0.355210</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.901795</td>
          <td>0.573958</td>
          <td>26.214410</td>
          <td>0.128097</td>
          <td>26.011240</td>
          <td>0.096633</td>
          <td>25.881716</td>
          <td>0.140835</td>
          <td>25.869130</td>
          <td>0.256339</td>
          <td>24.586899</td>
          <td>0.190696</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.438261</td>
          <td>0.401234</td>
          <td>26.512456</td>
          <td>0.162379</td>
          <td>26.471786</td>
          <td>0.141213</td>
          <td>26.901811</td>
          <td>0.322440</td>
          <td>25.950013</td>
          <td>0.268471</td>
          <td>26.274500</td>
          <td>0.689308</td>
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
          <td>26.670728</td>
          <td>0.160355</td>
          <td>26.079691</td>
          <td>0.084599</td>
          <td>25.195707</td>
          <td>0.063164</td>
          <td>24.814745</td>
          <td>0.086225</td>
          <td>24.105283</td>
          <td>0.103662</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.257787</td>
          <td>0.566970</td>
          <td>26.710163</td>
          <td>0.146677</td>
          <td>26.547042</td>
          <td>0.204378</td>
          <td>25.937656</td>
          <td>0.226553</td>
          <td>25.221362</td>
          <td>0.267750</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.376371</td>
          <td>0.308194</td>
          <td>29.456908</td>
          <td>1.182849</td>
          <td>26.114082</td>
          <td>0.153604</td>
          <td>24.896850</td>
          <td>0.100559</td>
          <td>24.415984</td>
          <td>0.147558</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.046861</td>
          <td>1.972515</td>
          <td>27.997483</td>
          <td>0.553616</td>
          <td>27.033492</td>
          <td>0.238964</td>
          <td>26.170998</td>
          <td>0.186183</td>
          <td>25.649753</td>
          <td>0.220412</td>
          <td>25.406468</td>
          <td>0.382566</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.454064</td>
          <td>0.364092</td>
          <td>26.121030</td>
          <td>0.099757</td>
          <td>26.001771</td>
          <td>0.079083</td>
          <td>25.715173</td>
          <td>0.100024</td>
          <td>25.318335</td>
          <td>0.134012</td>
          <td>25.321118</td>
          <td>0.290457</td>
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
          <td>27.077681</td>
          <td>0.606888</td>
          <td>26.324612</td>
          <td>0.127382</td>
          <td>25.514233</td>
          <td>0.055522</td>
          <td>25.105589</td>
          <td>0.063369</td>
          <td>24.853566</td>
          <td>0.096514</td>
          <td>24.231373</td>
          <td>0.125432</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.200032</td>
          <td>1.195931</td>
          <td>26.543342</td>
          <td>0.145786</td>
          <td>26.043313</td>
          <td>0.083286</td>
          <td>25.268874</td>
          <td>0.068573</td>
          <td>24.711305</td>
          <td>0.080012</td>
          <td>24.298004</td>
          <td>0.124697</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.739038</td>
          <td>0.466153</td>
          <td>26.576867</td>
          <td>0.154258</td>
          <td>26.444710</td>
          <td>0.122247</td>
          <td>26.306356</td>
          <td>0.175057</td>
          <td>25.573002</td>
          <td>0.174591</td>
          <td>26.586485</td>
          <td>0.771554</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.503481</td>
          <td>0.406351</td>
          <td>26.289792</td>
          <td>0.127553</td>
          <td>25.886923</td>
          <td>0.080081</td>
          <td>25.965729</td>
          <td>0.139824</td>
          <td>25.801186</td>
          <td>0.225198</td>
          <td>25.414397</td>
          <td>0.347997</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.708671</td>
          <td>0.452873</td>
          <td>26.787992</td>
          <td>0.183048</td>
          <td>26.433527</td>
          <td>0.119873</td>
          <td>26.115516</td>
          <td>0.147218</td>
          <td>26.259888</td>
          <td>0.305332</td>
          <td>inf</td>
          <td>inf</td>
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
