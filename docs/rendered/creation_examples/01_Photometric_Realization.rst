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

    <pzflow.flow.Flow at 0x7f74c0156bc0>



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
    0      23.994413  0.190467  0.178528  
    1      25.391064  0.019113  0.013790  
    2      24.304707  0.020933  0.012914  
    3      25.291103  0.207405  0.114015  
    4      25.096743  0.047605  0.024200  
    ...          ...       ...       ...  
    99995  24.737946  0.232620  0.173696  
    99996  24.224169  0.036883  0.030630  
    99997  25.613836  0.003200  0.002762  
    99998  25.274899  0.064272  0.057713  
    99999  25.699642  0.119118  0.090947  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.664790</td>
          <td>0.159526</td>
          <td>25.981434</td>
          <td>0.077565</td>
          <td>25.260074</td>
          <td>0.066863</td>
          <td>24.734668</td>
          <td>0.080338</td>
          <td>24.092316</td>
          <td>0.102478</td>
          <td>0.190467</td>
          <td>0.178528</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.242564</td>
          <td>0.560430</td>
          <td>26.429803</td>
          <td>0.114967</td>
          <td>26.393553</td>
          <td>0.179400</td>
          <td>25.696771</td>
          <td>0.184972</td>
          <td>26.136300</td>
          <td>0.542771</td>
          <td>0.019113</td>
          <td>0.013790</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.093318</td>
          <td>1.117483</td>
          <td>28.182189</td>
          <td>0.536501</td>
          <td>27.990797</td>
          <td>0.417842</td>
          <td>25.862664</td>
          <td>0.113616</td>
          <td>25.302043</td>
          <td>0.131953</td>
          <td>24.373788</td>
          <td>0.130930</td>
          <td>0.020933</td>
          <td>0.012914</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.476398</td>
          <td>1.378555</td>
          <td>29.114704</td>
          <td>0.998366</td>
          <td>27.244117</td>
          <td>0.230134</td>
          <td>26.459299</td>
          <td>0.189657</td>
          <td>25.878324</td>
          <td>0.215445</td>
          <td>25.258663</td>
          <td>0.275753</td>
          <td>0.207405</td>
          <td>0.114015</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.627600</td>
          <td>0.415977</td>
          <td>26.133778</td>
          <td>0.100752</td>
          <td>25.986163</td>
          <td>0.077889</td>
          <td>25.716899</td>
          <td>0.100026</td>
          <td>25.287302</td>
          <td>0.130281</td>
          <td>24.947442</td>
          <td>0.213363</td>
          <td>0.047605</td>
          <td>0.024200</td>
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
          <td>26.559289</td>
          <td>0.394730</td>
          <td>26.440025</td>
          <td>0.131505</td>
          <td>25.349287</td>
          <td>0.044283</td>
          <td>25.004006</td>
          <td>0.053275</td>
          <td>24.848213</td>
          <td>0.088791</td>
          <td>24.696968</td>
          <td>0.172763</td>
          <td>0.232620</td>
          <td>0.173696</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.558152</td>
          <td>0.145598</td>
          <td>26.079278</td>
          <td>0.084557</td>
          <td>25.145182</td>
          <td>0.060388</td>
          <td>24.795298</td>
          <td>0.084750</td>
          <td>24.099515</td>
          <td>0.103126</td>
          <td>0.036883</td>
          <td>0.030630</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.867416</td>
          <td>0.977841</td>
          <td>26.764642</td>
          <td>0.173686</td>
          <td>26.189012</td>
          <td>0.093128</td>
          <td>26.709614</td>
          <td>0.233801</td>
          <td>25.455434</td>
          <td>0.150595</td>
          <td>25.577089</td>
          <td>0.355661</td>
          <td>0.003200</td>
          <td>0.002762</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.483804</td>
          <td>0.372309</td>
          <td>26.159974</td>
          <td>0.103086</td>
          <td>26.061048</td>
          <td>0.083209</td>
          <td>25.882705</td>
          <td>0.115616</td>
          <td>25.966063</td>
          <td>0.231747</td>
          <td>25.150748</td>
          <td>0.252487</td>
          <td>0.064272</td>
          <td>0.057713</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.120888</td>
          <td>1.135276</td>
          <td>26.838510</td>
          <td>0.184900</td>
          <td>26.676138</td>
          <td>0.142313</td>
          <td>26.244544</td>
          <td>0.158021</td>
          <td>26.209310</td>
          <td>0.282868</td>
          <td>25.442126</td>
          <td>0.319643</td>
          <td>0.119118</td>
          <td>0.090947</td>
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
          <td>27.234442</td>
          <td>0.754780</td>
          <td>26.778600</td>
          <td>0.220595</td>
          <td>26.138031</td>
          <td>0.116031</td>
          <td>25.190083</td>
          <td>0.082972</td>
          <td>24.551427</td>
          <td>0.089223</td>
          <td>23.848793</td>
          <td>0.108756</td>
          <td>0.190467</td>
          <td>0.178528</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.972140</td>
          <td>2.740834</td>
          <td>27.148254</td>
          <td>0.273851</td>
          <td>26.641539</td>
          <td>0.161895</td>
          <td>26.510255</td>
          <td>0.232513</td>
          <td>25.932744</td>
          <td>0.262482</td>
          <td>24.842293</td>
          <td>0.229169</td>
          <td>0.019113</td>
          <td>0.013790</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.361881</td>
          <td>0.294767</td>
          <td>26.167869</td>
          <td>0.174471</td>
          <td>25.128622</td>
          <td>0.133269</td>
          <td>24.188781</td>
          <td>0.131641</td>
          <td>0.020933</td>
          <td>0.012914</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.934831</td>
          <td>0.537787</td>
          <td>27.830211</td>
          <td>0.457589</td>
          <td>26.435220</td>
          <td>0.237135</td>
          <td>25.130755</td>
          <td>0.144997</td>
          <td>25.387489</td>
          <td>0.384365</td>
          <td>0.207405</td>
          <td>0.114015</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.167266</td>
          <td>0.323460</td>
          <td>25.975718</td>
          <td>0.101625</td>
          <td>25.816580</td>
          <td>0.079279</td>
          <td>25.477083</td>
          <td>0.096385</td>
          <td>25.402028</td>
          <td>0.169122</td>
          <td>25.423817</td>
          <td>0.367612</td>
          <td>0.047605</td>
          <td>0.024200</td>
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
          <td>27.211507</td>
          <td>0.752787</td>
          <td>26.245830</td>
          <td>0.143121</td>
          <td>25.635003</td>
          <td>0.076191</td>
          <td>25.022348</td>
          <td>0.073099</td>
          <td>24.870487</td>
          <td>0.120369</td>
          <td>25.265796</td>
          <td>0.362329</td>
          <td>0.232620</td>
          <td>0.173696</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.212344</td>
          <td>0.700753</td>
          <td>26.441132</td>
          <td>0.151949</td>
          <td>25.943478</td>
          <td>0.088573</td>
          <td>25.173230</td>
          <td>0.073683</td>
          <td>24.720784</td>
          <td>0.093672</td>
          <td>24.092949</td>
          <td>0.121505</td>
          <td>0.036883</td>
          <td>0.030630</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.205828</td>
          <td>0.696078</td>
          <td>26.599105</td>
          <td>0.173276</td>
          <td>26.498274</td>
          <td>0.143056</td>
          <td>26.274349</td>
          <td>0.190732</td>
          <td>25.785681</td>
          <td>0.232371</td>
          <td>24.572014</td>
          <td>0.182562</td>
          <td>0.003200</td>
          <td>0.002762</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.064916</td>
          <td>0.299668</td>
          <td>26.098051</td>
          <td>0.113854</td>
          <td>25.990732</td>
          <td>0.093133</td>
          <td>26.045707</td>
          <td>0.159053</td>
          <td>25.711302</td>
          <td>0.221055</td>
          <td>24.968884</td>
          <td>0.257233</td>
          <td>0.064272</td>
          <td>0.057713</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.059672</td>
          <td>0.643588</td>
          <td>26.633667</td>
          <td>0.184232</td>
          <td>26.331840</td>
          <td>0.128463</td>
          <td>26.571317</td>
          <td>0.253153</td>
          <td>25.761614</td>
          <td>0.235793</td>
          <td>24.960708</td>
          <td>0.261492</td>
          <td>0.119118</td>
          <td>0.090947</td>
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
          <td>27.850047</td>
          <td>1.140350</td>
          <td>26.682315</td>
          <td>0.214228</td>
          <td>25.910245</td>
          <td>0.100771</td>
          <td>25.163170</td>
          <td>0.086031</td>
          <td>24.773241</td>
          <td>0.114792</td>
          <td>23.785259</td>
          <td>0.109145</td>
          <td>0.190467</td>
          <td>0.178528</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.012192</td>
          <td>1.067914</td>
          <td>27.261885</td>
          <td>0.263728</td>
          <td>26.479422</td>
          <td>0.120486</td>
          <td>26.493424</td>
          <td>0.195930</td>
          <td>25.795058</td>
          <td>0.201663</td>
          <td>25.669868</td>
          <td>0.383688</td>
          <td>0.019113</td>
          <td>0.013790</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.445001</td>
          <td>1.211399</td>
          <td>28.366788</td>
          <td>0.554442</td>
          <td>26.120104</td>
          <td>0.142612</td>
          <td>25.142645</td>
          <td>0.115370</td>
          <td>24.315906</td>
          <td>0.125050</td>
          <td>0.020933</td>
          <td>0.012914</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.221123</td>
          <td>1.359367</td>
          <td>30.678209</td>
          <td>2.406538</td>
          <td>27.574169</td>
          <td>0.382738</td>
          <td>26.218390</td>
          <td>0.201680</td>
          <td>25.527404</td>
          <td>0.206912</td>
          <td>25.080222</td>
          <td>0.307011</td>
          <td>0.207405</td>
          <td>0.114015</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.111706</td>
          <td>0.280331</td>
          <td>26.124832</td>
          <td>0.101617</td>
          <td>25.868862</td>
          <td>0.071566</td>
          <td>25.679421</td>
          <td>0.098727</td>
          <td>25.550249</td>
          <td>0.166340</td>
          <td>24.993581</td>
          <td>0.225878</td>
          <td>0.047605</td>
          <td>0.024200</td>
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
          <td>27.598040</td>
          <td>1.010370</td>
          <td>26.645249</td>
          <td>0.216607</td>
          <td>25.458093</td>
          <td>0.070968</td>
          <td>24.941452</td>
          <td>0.074263</td>
          <td>24.854263</td>
          <td>0.129023</td>
          <td>25.498895</td>
          <td>0.466882</td>
          <td>0.232620</td>
          <td>0.173696</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.009662</td>
          <td>0.557406</td>
          <td>26.683639</td>
          <td>0.164229</td>
          <td>26.186227</td>
          <td>0.094340</td>
          <td>25.243732</td>
          <td>0.066985</td>
          <td>24.908951</td>
          <td>0.095109</td>
          <td>24.269161</td>
          <td>0.121476</td>
          <td>0.036883</td>
          <td>0.030630</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.913071</td>
          <td>0.515119</td>
          <td>26.637397</td>
          <td>0.155850</td>
          <td>26.526935</td>
          <td>0.125110</td>
          <td>26.321427</td>
          <td>0.168759</td>
          <td>25.335074</td>
          <td>0.135790</td>
          <td>26.338993</td>
          <td>0.627117</td>
          <td>0.003200</td>
          <td>0.002762</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.297390</td>
          <td>0.331489</td>
          <td>25.954963</td>
          <td>0.089832</td>
          <td>26.000079</td>
          <td>0.082766</td>
          <td>25.738887</td>
          <td>0.107231</td>
          <td>25.736530</td>
          <td>0.200307</td>
          <td>24.981930</td>
          <td>0.230215</td>
          <td>0.064272</td>
          <td>0.057713</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.242392</td>
          <td>0.335316</td>
          <td>26.760702</td>
          <td>0.193562</td>
          <td>26.679309</td>
          <td>0.162245</td>
          <td>26.671313</td>
          <td>0.257620</td>
          <td>26.477795</td>
          <td>0.393809</td>
          <td>25.524581</td>
          <td>0.385165</td>
          <td>0.119118</td>
          <td>0.090947</td>
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
