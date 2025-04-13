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

    <pzflow.flow.Flow at 0x7f548d5cfaf0>



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
          <td>26.498076</td>
          <td>0.376465</td>
          <td>26.810319</td>
          <td>0.180544</td>
          <td>25.980154</td>
          <td>0.077477</td>
          <td>25.046941</td>
          <td>0.055346</td>
          <td>24.754408</td>
          <td>0.081749</td>
          <td>23.945298</td>
          <td>0.090077</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.266914</td>
          <td>0.662453</td>
          <td>27.259537</td>
          <td>0.262426</td>
          <td>26.644566</td>
          <td>0.138493</td>
          <td>26.459112</td>
          <td>0.189627</td>
          <td>25.856853</td>
          <td>0.211617</td>
          <td>25.619217</td>
          <td>0.367586</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.784661</td>
          <td>0.929497</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.958896</td>
          <td>0.407758</td>
          <td>26.013692</td>
          <td>0.129544</td>
          <td>25.054310</td>
          <td>0.106381</td>
          <td>24.341595</td>
          <td>0.127331</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.477192</td>
          <td>1.379126</td>
          <td>29.438466</td>
          <td>1.204307</td>
          <td>27.256129</td>
          <td>0.232436</td>
          <td>26.331670</td>
          <td>0.170215</td>
          <td>25.501792</td>
          <td>0.156698</td>
          <td>26.114731</td>
          <td>0.534339</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>28.238074</td>
          <td>1.212674</td>
          <td>26.045980</td>
          <td>0.093293</td>
          <td>26.012452</td>
          <td>0.079718</td>
          <td>25.725562</td>
          <td>0.100789</td>
          <td>25.235003</td>
          <td>0.124509</td>
          <td>25.822328</td>
          <td>0.429873</td>
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
          <td>27.791799</td>
          <td>0.933607</td>
          <td>26.423064</td>
          <td>0.129591</td>
          <td>25.435962</td>
          <td>0.047824</td>
          <td>25.137147</td>
          <td>0.059959</td>
          <td>24.967312</td>
          <td>0.098581</td>
          <td>24.545904</td>
          <td>0.151857</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.441563</td>
          <td>0.360236</td>
          <td>26.639224</td>
          <td>0.156078</td>
          <td>26.111855</td>
          <td>0.087019</td>
          <td>25.310038</td>
          <td>0.069889</td>
          <td>24.856086</td>
          <td>0.089408</td>
          <td>24.340312</td>
          <td>0.127190</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.441274</td>
          <td>0.745511</td>
          <td>26.781529</td>
          <td>0.176193</td>
          <td>26.293448</td>
          <td>0.102060</td>
          <td>26.838094</td>
          <td>0.259879</td>
          <td>25.654195</td>
          <td>0.178423</td>
          <td>25.195197</td>
          <td>0.261851</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.808654</td>
          <td>0.215873</td>
          <td>26.296791</td>
          <td>0.116148</td>
          <td>26.190407</td>
          <td>0.093242</td>
          <td>25.850940</td>
          <td>0.112460</td>
          <td>25.678296</td>
          <td>0.182103</td>
          <td>25.657430</td>
          <td>0.378692</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.826660</td>
          <td>0.483281</td>
          <td>26.801345</td>
          <td>0.179177</td>
          <td>26.668206</td>
          <td>0.141344</td>
          <td>26.469071</td>
          <td>0.191227</td>
          <td>26.088359</td>
          <td>0.256318</td>
          <td>25.471432</td>
          <td>0.327186</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.653235</td>
          <td>0.181410</td>
          <td>26.018242</td>
          <td>0.094222</td>
          <td>25.227901</td>
          <td>0.077018</td>
          <td>24.625227</td>
          <td>0.085791</td>
          <td>23.907665</td>
          <td>0.102977</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.106732</td>
          <td>0.307240</td>
          <td>27.624273</td>
          <td>0.399088</td>
          <td>26.829479</td>
          <td>0.189764</td>
          <td>26.176620</td>
          <td>0.175633</td>
          <td>25.778375</td>
          <td>0.231014</td>
          <td>25.351302</td>
          <td>0.345824</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.947332</td>
          <td>0.517518</td>
          <td>28.609408</td>
          <td>0.755685</td>
          <td>25.821211</td>
          <td>0.132438</td>
          <td>24.679456</td>
          <td>0.092014</td>
          <td>24.188564</td>
          <td>0.134481</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.080867</td>
          <td>0.315439</td>
          <td>27.991469</td>
          <td>0.552695</td>
          <td>27.270409</td>
          <td>0.290935</td>
          <td>26.484096</td>
          <td>0.242656</td>
          <td>25.569597</td>
          <td>0.206847</td>
          <td>26.981572</td>
          <td>1.121031</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.430854</td>
          <td>0.396378</td>
          <td>25.974443</td>
          <td>0.101104</td>
          <td>26.048630</td>
          <td>0.096799</td>
          <td>25.646297</td>
          <td>0.111246</td>
          <td>26.041389</td>
          <td>0.286565</td>
          <td>24.881259</td>
          <td>0.236548</td>
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
          <td>27.783548</td>
          <td>1.018562</td>
          <td>26.483175</td>
          <td>0.159905</td>
          <td>25.436032</td>
          <td>0.057543</td>
          <td>25.204930</td>
          <td>0.077141</td>
          <td>24.856621</td>
          <td>0.107320</td>
          <td>24.667242</td>
          <td>0.201950</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.952369</td>
          <td>1.851153</td>
          <td>26.508126</td>
          <td>0.160950</td>
          <td>26.067728</td>
          <td>0.098812</td>
          <td>25.129340</td>
          <td>0.070901</td>
          <td>24.798946</td>
          <td>0.100346</td>
          <td>24.220662</td>
          <td>0.135752</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.691061</td>
          <td>0.486519</td>
          <td>26.666195</td>
          <td>0.185457</td>
          <td>26.549115</td>
          <td>0.151324</td>
          <td>26.379693</td>
          <td>0.211021</td>
          <td>26.265859</td>
          <td>0.346719</td>
          <td>25.356839</td>
          <td>0.351378</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.196712</td>
          <td>0.337243</td>
          <td>26.217189</td>
          <td>0.128405</td>
          <td>26.137228</td>
          <td>0.107895</td>
          <td>25.937509</td>
          <td>0.147759</td>
          <td>26.017879</td>
          <td>0.289319</td>
          <td>26.680206</td>
          <td>0.912966</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.895302</td>
          <td>0.563716</td>
          <td>27.280260</td>
          <td>0.306952</td>
          <td>26.461131</td>
          <td>0.139923</td>
          <td>26.300155</td>
          <td>0.196873</td>
          <td>25.835639</td>
          <td>0.244452</td>
          <td>25.205805</td>
          <td>0.310909</td>
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

.. parsed-literal::

    




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
          <td>26.991546</td>
          <td>0.545404</td>
          <td>26.677953</td>
          <td>0.161347</td>
          <td>26.111052</td>
          <td>0.086968</td>
          <td>25.117784</td>
          <td>0.058946</td>
          <td>24.614609</td>
          <td>0.072263</td>
          <td>24.218436</td>
          <td>0.114425</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.155382</td>
          <td>0.613292</td>
          <td>27.147090</td>
          <td>0.239463</td>
          <td>26.642928</td>
          <td>0.138426</td>
          <td>26.513982</td>
          <td>0.198784</td>
          <td>25.622023</td>
          <td>0.173776</td>
          <td>27.329358</td>
          <td>1.173723</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.964290</td>
          <td>2.492480</td>
          <td>28.330132</td>
          <td>0.575429</td>
          <td>26.150245</td>
          <td>0.158433</td>
          <td>25.260379</td>
          <td>0.137946</td>
          <td>24.325212</td>
          <td>0.136464</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.281113</td>
          <td>0.675859</td>
          <td>27.365151</td>
          <td>0.312920</td>
          <td>26.147454</td>
          <td>0.182513</td>
          <td>25.858602</td>
          <td>0.261858</td>
          <td>25.001277</td>
          <td>0.277269</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.541365</td>
          <td>0.389643</td>
          <td>26.185808</td>
          <td>0.105569</td>
          <td>25.942269</td>
          <td>0.075034</td>
          <td>25.549403</td>
          <td>0.086469</td>
          <td>25.638656</td>
          <td>0.176328</td>
          <td>25.324761</td>
          <td>0.291313</td>
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
          <td>26.263936</td>
          <td>0.329289</td>
          <td>26.371815</td>
          <td>0.132688</td>
          <td>25.463011</td>
          <td>0.053054</td>
          <td>25.051578</td>
          <td>0.060406</td>
          <td>24.828659</td>
          <td>0.094428</td>
          <td>24.668738</td>
          <td>0.182494</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.247653</td>
          <td>0.659626</td>
          <td>26.709655</td>
          <td>0.168068</td>
          <td>26.033819</td>
          <td>0.082591</td>
          <td>25.170940</td>
          <td>0.062873</td>
          <td>24.951063</td>
          <td>0.098797</td>
          <td>24.007001</td>
          <td>0.096735</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.577559</td>
          <td>0.154350</td>
          <td>26.370193</td>
          <td>0.114576</td>
          <td>26.272312</td>
          <td>0.170064</td>
          <td>25.824571</td>
          <td>0.215776</td>
          <td>25.538588</td>
          <td>0.361057</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.001211</td>
          <td>0.273177</td>
          <td>26.118956</td>
          <td>0.109964</td>
          <td>26.383359</td>
          <td>0.123689</td>
          <td>25.855168</td>
          <td>0.127082</td>
          <td>25.744420</td>
          <td>0.214803</td>
          <td>25.075324</td>
          <td>0.265109</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.599822</td>
          <td>0.417013</td>
          <td>27.029739</td>
          <td>0.224172</td>
          <td>26.532565</td>
          <td>0.130625</td>
          <td>26.737833</td>
          <td>0.248647</td>
          <td>26.124182</td>
          <td>0.273628</td>
          <td>25.864386</td>
          <td>0.459417</td>
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
