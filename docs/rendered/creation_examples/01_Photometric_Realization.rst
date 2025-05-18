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

    <pzflow.flow.Flow at 0x7f09c4333e50>



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
          <td>28.072635</td>
          <td>1.104239</td>
          <td>26.764256</td>
          <td>0.173629</td>
          <td>26.026000</td>
          <td>0.080677</td>
          <td>25.226960</td>
          <td>0.064930</td>
          <td>24.741685</td>
          <td>0.080837</td>
          <td>23.858085</td>
          <td>0.083420</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.039375</td>
          <td>0.564483</td>
          <td>27.574068</td>
          <td>0.337962</td>
          <td>26.707934</td>
          <td>0.146260</td>
          <td>26.237019</td>
          <td>0.157007</td>
          <td>25.488227</td>
          <td>0.154889</td>
          <td>24.821977</td>
          <td>0.192045</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.032272</td>
          <td>1.078658</td>
          <td>29.551992</td>
          <td>1.281674</td>
          <td>28.452848</td>
          <td>0.587722</td>
          <td>26.055488</td>
          <td>0.134313</td>
          <td>24.977371</td>
          <td>0.099454</td>
          <td>24.231941</td>
          <td>0.115763</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.870793</td>
          <td>0.425520</td>
          <td>27.201471</td>
          <td>0.222126</td>
          <td>26.400660</td>
          <td>0.180484</td>
          <td>25.295030</td>
          <td>0.131155</td>
          <td>24.926274</td>
          <td>0.209622</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.887671</td>
          <td>0.230504</td>
          <td>26.022196</td>
          <td>0.091366</td>
          <td>26.035336</td>
          <td>0.081344</td>
          <td>25.716601</td>
          <td>0.100000</td>
          <td>25.452633</td>
          <td>0.150234</td>
          <td>25.027222</td>
          <td>0.228014</td>
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
          <td>27.902279</td>
          <td>0.998664</td>
          <td>26.492238</td>
          <td>0.137568</td>
          <td>25.454138</td>
          <td>0.048602</td>
          <td>25.078621</td>
          <td>0.056924</td>
          <td>24.993265</td>
          <td>0.100848</td>
          <td>24.717884</td>
          <td>0.175860</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.660117</td>
          <td>0.158891</td>
          <td>25.977005</td>
          <td>0.077262</td>
          <td>25.281229</td>
          <td>0.068128</td>
          <td>24.839271</td>
          <td>0.088095</td>
          <td>24.265086</td>
          <td>0.119150</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.875868</td>
          <td>0.501195</td>
          <td>26.581954</td>
          <td>0.148604</td>
          <td>26.427650</td>
          <td>0.114751</td>
          <td>26.149324</td>
          <td>0.145629</td>
          <td>26.192351</td>
          <td>0.279005</td>
          <td>26.013961</td>
          <td>0.496289</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.626435</td>
          <td>0.185304</td>
          <td>26.072089</td>
          <td>0.095453</td>
          <td>26.125074</td>
          <td>0.088037</td>
          <td>25.890231</td>
          <td>0.116377</td>
          <td>25.920349</td>
          <td>0.223119</td>
          <td>25.170011</td>
          <td>0.256508</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.271131</td>
          <td>0.664380</td>
          <td>26.897844</td>
          <td>0.194386</td>
          <td>26.481357</td>
          <td>0.120241</td>
          <td>26.324668</td>
          <td>0.169204</td>
          <td>25.705332</td>
          <td>0.186315</td>
          <td>26.122256</td>
          <td>0.537269</td>
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
          <td>26.359661</td>
          <td>0.141201</td>
          <td>25.992297</td>
          <td>0.092099</td>
          <td>25.192667</td>
          <td>0.074658</td>
          <td>24.576216</td>
          <td>0.082167</td>
          <td>23.940064</td>
          <td>0.105935</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.590914</td>
          <td>0.895328</td>
          <td>26.804868</td>
          <td>0.206139</td>
          <td>26.552441</td>
          <td>0.149903</td>
          <td>26.225462</td>
          <td>0.183054</td>
          <td>25.798349</td>
          <td>0.234865</td>
          <td>25.127615</td>
          <td>0.289281</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.246515</td>
          <td>0.724908</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.141503</td>
          <td>0.546145</td>
          <td>25.805709</td>
          <td>0.130674</td>
          <td>24.987412</td>
          <td>0.120425</td>
          <td>24.052025</td>
          <td>0.119478</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.615447</td>
          <td>0.845563</td>
          <td>27.278194</td>
          <td>0.292769</td>
          <td>26.488667</td>
          <td>0.243572</td>
          <td>25.521742</td>
          <td>0.198709</td>
          <td>24.876907</td>
          <td>0.251345</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.222147</td>
          <td>0.336802</td>
          <td>26.034921</td>
          <td>0.106589</td>
          <td>26.098791</td>
          <td>0.101148</td>
          <td>25.759005</td>
          <td>0.122707</td>
          <td>25.240809</td>
          <td>0.146700</td>
          <td>25.147117</td>
          <td>0.293902</td>
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
          <td>28.494734</td>
          <td>1.503347</td>
          <td>26.568627</td>
          <td>0.171975</td>
          <td>25.400060</td>
          <td>0.055736</td>
          <td>25.044924</td>
          <td>0.066964</td>
          <td>24.963447</td>
          <td>0.117791</td>
          <td>24.647656</td>
          <td>0.198656</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.638507</td>
          <td>0.465249</td>
          <td>26.816487</td>
          <td>0.208863</td>
          <td>25.967466</td>
          <td>0.090489</td>
          <td>25.146983</td>
          <td>0.072016</td>
          <td>24.858876</td>
          <td>0.105747</td>
          <td>24.330290</td>
          <td>0.149187</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.887265</td>
          <td>0.561435</td>
          <td>27.549352</td>
          <td>0.380408</td>
          <td>26.212687</td>
          <td>0.113122</td>
          <td>26.143584</td>
          <td>0.172936</td>
          <td>25.680239</td>
          <td>0.215462</td>
          <td>24.921370</td>
          <td>0.247457</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.653213</td>
          <td>0.216906</td>
          <td>25.997411</td>
          <td>0.106083</td>
          <td>26.127456</td>
          <td>0.106979</td>
          <td>25.727288</td>
          <td>0.123231</td>
          <td>26.167533</td>
          <td>0.326183</td>
          <td>24.970352</td>
          <td>0.262235</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.791824</td>
          <td>0.205632</td>
          <td>26.423739</td>
          <td>0.135482</td>
          <td>26.867220</td>
          <td>0.313665</td>
          <td>25.988684</td>
          <td>0.277051</td>
          <td>25.632575</td>
          <td>0.433757</td>
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
          <td>26.485463</td>
          <td>0.372820</td>
          <td>26.805009</td>
          <td>0.179753</td>
          <td>26.022364</td>
          <td>0.080429</td>
          <td>25.201106</td>
          <td>0.063468</td>
          <td>24.736027</td>
          <td>0.080445</td>
          <td>24.146140</td>
          <td>0.107431</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.340985</td>
          <td>1.283460</td>
          <td>27.392108</td>
          <td>0.292466</td>
          <td>26.565769</td>
          <td>0.129497</td>
          <td>26.270508</td>
          <td>0.161724</td>
          <td>25.926390</td>
          <td>0.224443</td>
          <td>25.135559</td>
          <td>0.249586</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.074550</td>
          <td>0.526508</td>
          <td>27.734142</td>
          <td>0.368245</td>
          <td>26.173732</td>
          <td>0.161645</td>
          <td>25.031541</td>
          <td>0.113117</td>
          <td>24.162083</td>
          <td>0.118478</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.427137</td>
          <td>1.480872</td>
          <td>28.500791</td>
          <td>0.783153</td>
          <td>27.287040</td>
          <td>0.293900</td>
          <td>26.603043</td>
          <td>0.266601</td>
          <td>25.724006</td>
          <td>0.234420</td>
          <td>24.835089</td>
          <td>0.242013</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.913516</td>
          <td>0.235697</td>
          <td>26.135248</td>
          <td>0.101005</td>
          <td>25.899881</td>
          <td>0.072273</td>
          <td>25.655613</td>
          <td>0.094933</td>
          <td>25.429418</td>
          <td>0.147473</td>
          <td>24.809563</td>
          <td>0.190315</td>
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
          <td>26.465454</td>
          <td>0.143839</td>
          <td>25.432262</td>
          <td>0.051626</td>
          <td>25.094835</td>
          <td>0.062768</td>
          <td>24.942649</td>
          <td>0.104346</td>
          <td>24.716713</td>
          <td>0.190043</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.578481</td>
          <td>0.404648</td>
          <td>26.894479</td>
          <td>0.196514</td>
          <td>25.857100</td>
          <td>0.070653</td>
          <td>25.251369</td>
          <td>0.067518</td>
          <td>24.828563</td>
          <td>0.088721</td>
          <td>24.405267</td>
          <td>0.136825</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.084073</td>
          <td>0.599241</td>
          <td>26.843752</td>
          <td>0.193495</td>
          <td>26.125038</td>
          <td>0.092456</td>
          <td>26.297951</td>
          <td>0.173812</td>
          <td>25.856698</td>
          <td>0.221629</td>
          <td>25.893586</td>
          <td>0.473722</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.355394</td>
          <td>0.362323</td>
          <td>26.192028</td>
          <td>0.117184</td>
          <td>26.106120</td>
          <td>0.097113</td>
          <td>25.857053</td>
          <td>0.127289</td>
          <td>25.920083</td>
          <td>0.248453</td>
          <td>26.421014</td>
          <td>0.727583</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.377189</td>
          <td>0.729219</td>
          <td>26.783539</td>
          <td>0.182360</td>
          <td>26.523277</td>
          <td>0.129579</td>
          <td>26.464551</td>
          <td>0.198092</td>
          <td>25.940854</td>
          <td>0.235423</td>
          <td>25.525823</td>
          <td>0.354175</td>
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
