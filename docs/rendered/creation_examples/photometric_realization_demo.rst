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

    <pzflow.flow.Flow at 0x7f2c5590cbb0>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.436019</td>
          <td>0.131051</td>
          <td>26.091906</td>
          <td>0.085503</td>
          <td>25.204126</td>
          <td>0.063629</td>
          <td>24.613638</td>
          <td>0.072191</td>
          <td>23.948218</td>
          <td>0.090309</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.841584</td>
          <td>2.499896</td>
          <td>28.039971</td>
          <td>0.483273</td>
          <td>26.823419</td>
          <td>0.161475</td>
          <td>26.270629</td>
          <td>0.161584</td>
          <td>25.714009</td>
          <td>0.187685</td>
          <td>26.078312</td>
          <td>0.520333</td>
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
          <td>28.013042</td>
          <td>0.424994</td>
          <td>26.007614</td>
          <td>0.128864</td>
          <td>25.128206</td>
          <td>0.113467</td>
          <td>24.276740</td>
          <td>0.120363</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.115227</td>
          <td>0.595845</td>
          <td>29.116037</td>
          <td>0.999168</td>
          <td>27.162866</td>
          <td>0.215095</td>
          <td>26.008910</td>
          <td>0.129009</td>
          <td>25.365318</td>
          <td>0.139363</td>
          <td>25.137817</td>
          <td>0.249820</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.995438</td>
          <td>0.251896</td>
          <td>26.252113</td>
          <td>0.111717</td>
          <td>25.979888</td>
          <td>0.077459</td>
          <td>25.624534</td>
          <td>0.092239</td>
          <td>25.456743</td>
          <td>0.150765</td>
          <td>25.367210</td>
          <td>0.301037</td>
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
          <td>29.635732</td>
          <td>2.316493</td>
          <td>26.391285</td>
          <td>0.126075</td>
          <td>25.523648</td>
          <td>0.051696</td>
          <td>25.113719</td>
          <td>0.058725</td>
          <td>24.821212</td>
          <td>0.086706</td>
          <td>24.906797</td>
          <td>0.206232</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.397614</td>
          <td>0.724057</td>
          <td>26.730621</td>
          <td>0.168735</td>
          <td>26.185488</td>
          <td>0.092840</td>
          <td>25.111123</td>
          <td>0.058590</td>
          <td>24.825101</td>
          <td>0.087003</td>
          <td>24.098221</td>
          <td>0.103009</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.283594</td>
          <td>0.670097</td>
          <td>26.579708</td>
          <td>0.148318</td>
          <td>26.173626</td>
          <td>0.091878</td>
          <td>26.325152</td>
          <td>0.169274</td>
          <td>26.177350</td>
          <td>0.275627</td>
          <td>25.376927</td>
          <td>0.303396</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.152602</td>
          <td>0.286265</td>
          <td>26.273517</td>
          <td>0.113819</td>
          <td>26.241802</td>
          <td>0.097545</td>
          <td>26.138521</td>
          <td>0.144282</td>
          <td>25.748167</td>
          <td>0.193171</td>
          <td>25.549904</td>
          <td>0.348140</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.636102</td>
          <td>0.418687</td>
          <td>26.908157</td>
          <td>0.196080</td>
          <td>26.327470</td>
          <td>0.105144</td>
          <td>26.245952</td>
          <td>0.158212</td>
          <td>25.909470</td>
          <td>0.221109</td>
          <td>26.595678</td>
          <td>0.747161</td>
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
          <td>28.013931</td>
          <td>1.152494</td>
          <td>26.854897</td>
          <td>0.214900</td>
          <td>25.995117</td>
          <td>0.092328</td>
          <td>25.203769</td>
          <td>0.075394</td>
          <td>24.608927</td>
          <td>0.084569</td>
          <td>23.855933</td>
          <td>0.098417</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.954180</td>
          <td>0.233414</td>
          <td>26.424336</td>
          <td>0.134248</td>
          <td>26.977603</td>
          <td>0.339293</td>
          <td>25.838124</td>
          <td>0.242707</td>
          <td>25.026243</td>
          <td>0.266427</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.919308</td>
          <td>2.545331</td>
          <td>28.021327</td>
          <td>0.500235</td>
          <td>25.968548</td>
          <td>0.150357</td>
          <td>25.112031</td>
          <td>0.134153</td>
          <td>24.052405</td>
          <td>0.119517</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.502095</td>
          <td>0.383046</td>
          <td>27.496432</td>
          <td>0.348394</td>
          <td>26.259004</td>
          <td>0.201214</td>
          <td>25.846833</td>
          <td>0.260213</td>
          <td>25.092978</td>
          <td>0.299600</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.959673</td>
          <td>0.272911</td>
          <td>25.956546</td>
          <td>0.099534</td>
          <td>25.809207</td>
          <td>0.078411</td>
          <td>25.595612</td>
          <td>0.106431</td>
          <td>25.324041</td>
          <td>0.157550</td>
          <td>25.315971</td>
          <td>0.336345</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.276854</td>
          <td>0.133947</td>
          <td>25.478529</td>
          <td>0.059753</td>
          <td>25.084244</td>
          <td>0.069335</td>
          <td>24.867323</td>
          <td>0.108327</td>
          <td>25.055638</td>
          <td>0.278327</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.722203</td>
          <td>0.495114</td>
          <td>27.325122</td>
          <td>0.316633</td>
          <td>26.091376</td>
          <td>0.100880</td>
          <td>25.149584</td>
          <td>0.072182</td>
          <td>24.820032</td>
          <td>0.102216</td>
          <td>24.346699</td>
          <td>0.151302</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.085957</td>
          <td>0.304825</td>
          <td>26.906888</td>
          <td>0.226863</td>
          <td>26.529192</td>
          <td>0.148759</td>
          <td>26.491373</td>
          <td>0.231571</td>
          <td>25.503496</td>
          <td>0.185746</td>
          <td>25.335668</td>
          <td>0.345571</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.038057</td>
          <td>0.297189</td>
          <td>26.218854</td>
          <td>0.128590</td>
          <td>26.065991</td>
          <td>0.101381</td>
          <td>26.007705</td>
          <td>0.156923</td>
          <td>26.231534</td>
          <td>0.343143</td>
          <td>25.916487</td>
          <td>0.545418</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.105463</td>
          <td>1.219053</td>
          <td>26.779583</td>
          <td>0.203534</td>
          <td>26.630479</td>
          <td>0.161802</td>
          <td>26.138753</td>
          <td>0.171755</td>
          <td>26.238151</td>
          <td>0.338384</td>
          <td>26.714905</td>
          <td>0.918516</td>
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
          <td>29.061255</td>
          <td>1.828308</td>
          <td>26.923190</td>
          <td>0.198595</td>
          <td>26.086719</td>
          <td>0.085125</td>
          <td>25.136609</td>
          <td>0.059939</td>
          <td>24.624397</td>
          <td>0.072891</td>
          <td>23.984129</td>
          <td>0.093217</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.460786</td>
          <td>0.309060</td>
          <td>26.725451</td>
          <td>0.148616</td>
          <td>26.124689</td>
          <td>0.142713</td>
          <td>26.177016</td>
          <td>0.275793</td>
          <td>25.570101</td>
          <td>0.354026</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.057232</td>
          <td>0.599477</td>
          <td>28.906913</td>
          <td>0.924382</td>
          <td>27.438596</td>
          <td>0.291202</td>
          <td>26.001814</td>
          <td>0.139474</td>
          <td>25.028980</td>
          <td>0.112865</td>
          <td>24.405643</td>
          <td>0.146253</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.645085</td>
          <td>0.958274</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.659293</td>
          <td>0.394311</td>
          <td>26.760195</td>
          <td>0.302768</td>
          <td>25.234001</td>
          <td>0.155130</td>
          <td>25.163376</td>
          <td>0.315930</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.877680</td>
          <td>0.228818</td>
          <td>26.056644</td>
          <td>0.094285</td>
          <td>25.953758</td>
          <td>0.075799</td>
          <td>25.921539</td>
          <td>0.119766</td>
          <td>25.415197</td>
          <td>0.145681</td>
          <td>25.075077</td>
          <td>0.237560</td>
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
          <td>26.390742</td>
          <td>0.363848</td>
          <td>26.238319</td>
          <td>0.118198</td>
          <td>25.454635</td>
          <td>0.052661</td>
          <td>25.082059</td>
          <td>0.062061</td>
          <td>24.783784</td>
          <td>0.090778</td>
          <td>24.660929</td>
          <td>0.181291</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.638961</td>
          <td>1.507604</td>
          <td>27.035877</td>
          <td>0.221182</td>
          <td>26.109099</td>
          <td>0.088254</td>
          <td>25.269858</td>
          <td>0.068633</td>
          <td>24.833338</td>
          <td>0.089094</td>
          <td>24.634857</td>
          <td>0.166607</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.316047</td>
          <td>0.336598</td>
          <td>26.667415</td>
          <td>0.166657</td>
          <td>26.381683</td>
          <td>0.115728</td>
          <td>26.185141</td>
          <td>0.157875</td>
          <td>25.827771</td>
          <td>0.216353</td>
          <td>25.352453</td>
          <td>0.311588</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.689788</td>
          <td>0.467922</td>
          <td>25.959304</td>
          <td>0.095644</td>
          <td>26.053653</td>
          <td>0.092742</td>
          <td>25.925106</td>
          <td>0.135008</td>
          <td>25.464969</td>
          <td>0.169721</td>
          <td>25.099863</td>
          <td>0.270467</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.451436</td>
          <td>0.371924</td>
          <td>27.150302</td>
          <td>0.247662</td>
          <td>26.557779</td>
          <td>0.133504</td>
          <td>26.127078</td>
          <td>0.148687</td>
          <td>25.915364</td>
          <td>0.230507</td>
          <td>26.209245</td>
          <td>0.591083</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
