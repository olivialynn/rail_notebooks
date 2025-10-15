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

    <pzflow.flow.Flow at 0x7fc5d0422410>



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
    0      23.994413  0.096267  0.091511  
    1      25.391064  0.095410  0.091229  
    2      24.304707  0.207371  0.109119  
    3      25.291103  0.039753  0.021989  
    4      25.096743  0.042489  0.033168  
    ...          ...       ...       ...  
    99995  24.737946  0.206096  0.133699  
    99996  24.224169  0.062931  0.038914  
    99997  25.613836  0.146896  0.126198  
    99998  25.274899  0.012929  0.009003  
    99999  25.699642  0.037473  0.021721  
    
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
          <td>27.740188</td>
          <td>0.904153</td>
          <td>26.913014</td>
          <td>0.196883</td>
          <td>25.934400</td>
          <td>0.074407</td>
          <td>25.266611</td>
          <td>0.067252</td>
          <td>24.527595</td>
          <td>0.066897</td>
          <td>24.030378</td>
          <td>0.097065</td>
          <td>0.096267</td>
          <td>0.091511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.146642</td>
          <td>0.609212</td>
          <td>27.759397</td>
          <td>0.390664</td>
          <td>26.761070</td>
          <td>0.153086</td>
          <td>26.024609</td>
          <td>0.130774</td>
          <td>25.734702</td>
          <td>0.190992</td>
          <td>25.103139</td>
          <td>0.242790</td>
          <td>0.095410</td>
          <td>0.091229</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.705328</td>
          <td>1.547856</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.812334</td>
          <td>0.752624</td>
          <td>25.924833</td>
          <td>0.119933</td>
          <td>25.065869</td>
          <td>0.107461</td>
          <td>24.385184</td>
          <td>0.132227</td>
          <td>0.207371</td>
          <td>0.109119</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.654376</td>
          <td>0.856517</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.479181</td>
          <td>0.598804</td>
          <td>26.092709</td>
          <td>0.138699</td>
          <td>25.762351</td>
          <td>0.195492</td>
          <td>25.173758</td>
          <td>0.257297</td>
          <td>0.039753</td>
          <td>0.021989</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.744778</td>
          <td>0.454606</td>
          <td>26.223188</td>
          <td>0.108936</td>
          <td>25.880107</td>
          <td>0.070918</td>
          <td>25.937331</td>
          <td>0.121242</td>
          <td>25.526454</td>
          <td>0.160038</td>
          <td>25.169849</td>
          <td>0.256474</td>
          <td>0.042489</td>
          <td>0.033168</td>
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
          <td>26.426676</td>
          <td>0.356060</td>
          <td>26.365642</td>
          <td>0.123304</td>
          <td>25.431332</td>
          <td>0.047628</td>
          <td>24.973655</td>
          <td>0.051859</td>
          <td>24.738080</td>
          <td>0.080580</td>
          <td>24.808071</td>
          <td>0.189807</td>
          <td>0.206096</td>
          <td>0.133699</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.253190</td>
          <td>0.656212</td>
          <td>27.056504</td>
          <td>0.221979</td>
          <td>25.898140</td>
          <td>0.072059</td>
          <td>25.220745</td>
          <td>0.064573</td>
          <td>24.977032</td>
          <td>0.099424</td>
          <td>24.345476</td>
          <td>0.127760</td>
          <td>0.062931</td>
          <td>0.038914</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.743077</td>
          <td>0.454025</td>
          <td>27.437991</td>
          <td>0.303237</td>
          <td>26.359566</td>
          <td>0.108135</td>
          <td>26.377984</td>
          <td>0.177047</td>
          <td>25.905215</td>
          <td>0.220327</td>
          <td>25.583495</td>
          <td>0.357453</td>
          <td>0.146896</td>
          <td>0.126198</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.306687</td>
          <td>0.323885</td>
          <td>26.100164</td>
          <td>0.097831</td>
          <td>26.015993</td>
          <td>0.079968</td>
          <td>25.679054</td>
          <td>0.096762</td>
          <td>25.545871</td>
          <td>0.162715</td>
          <td>25.064675</td>
          <td>0.235199</td>
          <td>0.012929</td>
          <td>0.009003</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.592893</td>
          <td>0.405067</td>
          <td>26.554046</td>
          <td>0.145085</td>
          <td>26.417038</td>
          <td>0.113695</td>
          <td>26.408447</td>
          <td>0.181678</td>
          <td>25.787689</td>
          <td>0.199702</td>
          <td>26.775482</td>
          <td>0.840313</td>
          <td>0.037473</td>
          <td>0.021721</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.929645</td>
          <td>0.234425</td>
          <td>25.970562</td>
          <td>0.093031</td>
          <td>25.198096</td>
          <td>0.077328</td>
          <td>24.764367</td>
          <td>0.099806</td>
          <td>23.995026</td>
          <td>0.114491</td>
          <td>0.096267</td>
          <td>0.091511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.198506</td>
          <td>0.704489</td>
          <td>27.419750</td>
          <td>0.348211</td>
          <td>26.808008</td>
          <td>0.191542</td>
          <td>26.111844</td>
          <td>0.171049</td>
          <td>26.265417</td>
          <td>0.351594</td>
          <td>24.957647</td>
          <td>0.258873</td>
          <td>0.095410</td>
          <td>0.091229</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.694754</td>
          <td>0.510970</td>
          <td>28.268048</td>
          <td>0.679434</td>
          <td>27.987296</td>
          <td>0.513443</td>
          <td>25.970858</td>
          <td>0.160189</td>
          <td>25.102164</td>
          <td>0.141234</td>
          <td>24.215883</td>
          <td>0.146398</td>
          <td>0.207371</td>
          <td>0.109119</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.244202</td>
          <td>1.310545</td>
          <td>30.069262</td>
          <td>1.796784</td>
          <td>27.429702</td>
          <td>0.311966</td>
          <td>26.551324</td>
          <td>0.241140</td>
          <td>25.573087</td>
          <td>0.195220</td>
          <td>25.034938</td>
          <td>0.269159</td>
          <td>0.039753</td>
          <td>0.021989</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.158387</td>
          <td>0.321206</td>
          <td>25.945815</td>
          <td>0.099011</td>
          <td>25.926045</td>
          <td>0.087315</td>
          <td>25.743580</td>
          <td>0.121643</td>
          <td>25.437522</td>
          <td>0.174319</td>
          <td>24.860259</td>
          <td>0.233504</td>
          <td>0.042489</td>
          <td>0.033168</td>
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
          <td>27.804352</td>
          <td>1.070369</td>
          <td>26.406793</td>
          <td>0.159539</td>
          <td>25.358556</td>
          <td>0.057720</td>
          <td>24.996425</td>
          <td>0.069063</td>
          <td>24.918134</td>
          <td>0.121468</td>
          <td>25.089868</td>
          <td>0.305840</td>
          <td>0.206096</td>
          <td>0.133699</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.173779</td>
          <td>1.265090</td>
          <td>26.396097</td>
          <td>0.146881</td>
          <td>26.037879</td>
          <td>0.096744</td>
          <td>25.283700</td>
          <td>0.081683</td>
          <td>24.858037</td>
          <td>0.106199</td>
          <td>24.258226</td>
          <td>0.140934</td>
          <td>0.062931</td>
          <td>0.038914</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.420215</td>
          <td>0.409253</td>
          <td>26.688152</td>
          <td>0.196831</td>
          <td>26.459104</td>
          <td>0.146672</td>
          <td>26.257163</td>
          <td>0.199473</td>
          <td>25.831519</td>
          <td>0.255242</td>
          <td>25.816161</td>
          <td>0.519417</td>
          <td>0.146896</td>
          <td>0.126198</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.878559</td>
          <td>1.066132</td>
          <td>26.251585</td>
          <td>0.128675</td>
          <td>26.280549</td>
          <td>0.118540</td>
          <td>25.729171</td>
          <td>0.119577</td>
          <td>25.740321</td>
          <td>0.223872</td>
          <td>25.361331</td>
          <td>0.348627</td>
          <td>0.012929</td>
          <td>0.009003</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.896227</td>
          <td>0.561679</td>
          <td>26.704619</td>
          <td>0.189974</td>
          <td>26.573642</td>
          <td>0.153094</td>
          <td>26.329617</td>
          <td>0.200437</td>
          <td>25.900146</td>
          <td>0.256105</td>
          <td>25.478937</td>
          <td>0.383149</td>
          <td>0.037473</td>
          <td>0.021721</td>
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
          <td>26.351758</td>
          <td>0.359469</td>
          <td>26.581707</td>
          <td>0.162827</td>
          <td>26.086051</td>
          <td>0.094687</td>
          <td>25.229901</td>
          <td>0.072875</td>
          <td>24.707897</td>
          <td>0.087354</td>
          <td>23.918068</td>
          <td>0.098253</td>
          <td>0.096267</td>
          <td>0.091511</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.257842</td>
          <td>0.698246</td>
          <td>27.007891</td>
          <td>0.232769</td>
          <td>26.838188</td>
          <td>0.181149</td>
          <td>26.409263</td>
          <td>0.202143</td>
          <td>25.622879</td>
          <td>0.192357</td>
          <td>25.618977</td>
          <td>0.404756</td>
          <td>0.095410</td>
          <td>0.091229</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.694111</td>
          <td>1.712932</td>
          <td>28.866335</td>
          <td>1.009150</td>
          <td>27.579733</td>
          <td>0.382523</td>
          <td>25.667521</td>
          <td>0.125290</td>
          <td>25.172308</td>
          <td>0.152322</td>
          <td>24.468903</td>
          <td>0.184432</td>
          <td>0.207371</td>
          <td>0.109119</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.196178</td>
          <td>1.191923</td>
          <td>31.746705</td>
          <td>3.152687</td>
          <td>27.342813</td>
          <td>0.252902</td>
          <td>26.471547</td>
          <td>0.194300</td>
          <td>25.897840</td>
          <td>0.221850</td>
          <td>25.494088</td>
          <td>0.337443</td>
          <td>0.039753</td>
          <td>0.021989</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.184426</td>
          <td>0.632312</td>
          <td>25.896836</td>
          <td>0.083223</td>
          <td>25.946955</td>
          <td>0.076716</td>
          <td>25.715888</td>
          <td>0.101981</td>
          <td>25.453614</td>
          <td>0.153221</td>
          <td>25.909737</td>
          <td>0.467144</td>
          <td>0.042489</td>
          <td>0.033168</td>
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
          <td>27.008127</td>
          <td>0.657120</td>
          <td>26.458722</td>
          <td>0.171985</td>
          <td>25.433538</td>
          <td>0.063848</td>
          <td>25.016263</td>
          <td>0.072797</td>
          <td>24.722035</td>
          <td>0.105916</td>
          <td>25.231501</td>
          <td>0.353115</td>
          <td>0.206096</td>
          <td>0.133699</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.809535</td>
          <td>0.487337</td>
          <td>26.757310</td>
          <td>0.177859</td>
          <td>26.241095</td>
          <td>0.101010</td>
          <td>25.070178</td>
          <td>0.058674</td>
          <td>24.696660</td>
          <td>0.080516</td>
          <td>24.077440</td>
          <td>0.104946</td>
          <td>0.062931</td>
          <td>0.038914</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.342567</td>
          <td>0.781629</td>
          <td>26.571694</td>
          <td>0.175649</td>
          <td>26.197626</td>
          <td>0.114941</td>
          <td>26.329868</td>
          <td>0.208321</td>
          <td>26.022126</td>
          <td>0.293168</td>
          <td>26.638645</td>
          <td>0.896495</td>
          <td>0.146896</td>
          <td>0.126198</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.661510</td>
          <td>0.427300</td>
          <td>26.301522</td>
          <td>0.116795</td>
          <td>25.989133</td>
          <td>0.078226</td>
          <td>25.906819</td>
          <td>0.118276</td>
          <td>25.424826</td>
          <td>0.146930</td>
          <td>25.478990</td>
          <td>0.329677</td>
          <td>0.012929</td>
          <td>0.009003</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.287362</td>
          <td>0.321475</td>
          <td>26.693279</td>
          <td>0.165191</td>
          <td>26.329985</td>
          <td>0.106700</td>
          <td>26.405396</td>
          <td>0.183522</td>
          <td>25.871252</td>
          <td>0.216740</td>
          <td>26.569596</td>
          <td>0.741574</td>
          <td>0.037473</td>
          <td>0.021721</td>
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
