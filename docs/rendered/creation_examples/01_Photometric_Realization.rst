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

    <pzflow.flow.Flow at 0x7fe39759f250>



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
          <td>26.540702</td>
          <td>0.389107</td>
          <td>26.678112</td>
          <td>0.161351</td>
          <td>26.108288</td>
          <td>0.086746</td>
          <td>25.184707</td>
          <td>0.062543</td>
          <td>24.641425</td>
          <td>0.073987</td>
          <td>24.118822</td>
          <td>0.104883</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.808064</td>
          <td>0.476645</td>
          <td>27.626985</td>
          <td>0.352357</td>
          <td>26.528618</td>
          <td>0.125277</td>
          <td>26.064938</td>
          <td>0.135414</td>
          <td>25.631510</td>
          <td>0.175022</td>
          <td>25.580274</td>
          <td>0.356551</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.567548</td>
          <td>0.397250</td>
          <td>28.485549</td>
          <td>0.664977</td>
          <td>27.843340</td>
          <td>0.372901</td>
          <td>25.917035</td>
          <td>0.119122</td>
          <td>24.935613</td>
          <td>0.095878</td>
          <td>24.265201</td>
          <td>0.119162</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.543681</td>
          <td>1.427342</td>
          <td>28.249192</td>
          <td>0.563106</td>
          <td>27.207785</td>
          <td>0.223296</td>
          <td>26.072118</td>
          <td>0.136256</td>
          <td>25.270197</td>
          <td>0.128366</td>
          <td>24.814433</td>
          <td>0.190828</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>28.202083</td>
          <td>1.188602</td>
          <td>25.987907</td>
          <td>0.088655</td>
          <td>25.908166</td>
          <td>0.072701</td>
          <td>25.634045</td>
          <td>0.093013</td>
          <td>25.247879</td>
          <td>0.125907</td>
          <td>24.775445</td>
          <td>0.184648</td>
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
          <td>28.120718</td>
          <td>1.135166</td>
          <td>26.278479</td>
          <td>0.114312</td>
          <td>25.418671</td>
          <td>0.047096</td>
          <td>25.064909</td>
          <td>0.056236</td>
          <td>24.796653</td>
          <td>0.084851</td>
          <td>24.780499</td>
          <td>0.185438</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.152902</td>
          <td>1.156138</td>
          <td>26.937606</td>
          <td>0.200991</td>
          <td>26.105762</td>
          <td>0.086553</td>
          <td>25.216032</td>
          <td>0.064304</td>
          <td>24.793705</td>
          <td>0.084631</td>
          <td>24.264862</td>
          <td>0.119127</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.224409</td>
          <td>1.203503</td>
          <td>26.547638</td>
          <td>0.144288</td>
          <td>26.292997</td>
          <td>0.102020</td>
          <td>26.103981</td>
          <td>0.140053</td>
          <td>25.938038</td>
          <td>0.226423</td>
          <td>26.076264</td>
          <td>0.519554</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.706355</td>
          <td>0.198187</td>
          <td>26.150073</td>
          <td>0.102198</td>
          <td>26.217509</td>
          <td>0.095488</td>
          <td>25.983250</td>
          <td>0.126172</td>
          <td>25.720892</td>
          <td>0.188779</td>
          <td>25.579269</td>
          <td>0.356270</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.427329</td>
          <td>0.356243</td>
          <td>26.889578</td>
          <td>0.193039</td>
          <td>26.519939</td>
          <td>0.124338</td>
          <td>26.309968</td>
          <td>0.167099</td>
          <td>26.267851</td>
          <td>0.296565</td>
          <td>26.038232</td>
          <td>0.505253</td>
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
          <td>26.557530</td>
          <td>0.436553</td>
          <td>26.649234</td>
          <td>0.180797</td>
          <td>25.873972</td>
          <td>0.082993</td>
          <td>25.405676</td>
          <td>0.090079</td>
          <td>24.573743</td>
          <td>0.081988</td>
          <td>23.825553</td>
          <td>0.095830</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.191350</td>
          <td>1.271836</td>
          <td>27.052424</td>
          <td>0.253089</td>
          <td>26.558776</td>
          <td>0.150720</td>
          <td>26.595049</td>
          <td>0.249190</td>
          <td>26.067315</td>
          <td>0.292597</td>
          <td>25.802290</td>
          <td>0.488425</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>37.320677</td>
          <td>8.837504</td>
          <td>27.238640</td>
          <td>0.272055</td>
          <td>25.890401</td>
          <td>0.140587</td>
          <td>24.949787</td>
          <td>0.116550</td>
          <td>24.117137</td>
          <td>0.126423</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>32.794420</td>
          <td>4.376708</td>
          <td>27.111374</td>
          <td>0.255629</td>
          <td>26.291741</td>
          <td>0.206812</td>
          <td>25.608644</td>
          <td>0.213710</td>
          <td>25.196497</td>
          <td>0.325455</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.966936</td>
          <td>0.274525</td>
          <td>25.976927</td>
          <td>0.101324</td>
          <td>26.095944</td>
          <td>0.100896</td>
          <td>25.538643</td>
          <td>0.101258</td>
          <td>25.482687</td>
          <td>0.180329</td>
          <td>25.412243</td>
          <td>0.362816</td>
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
          <td>26.458232</td>
          <td>0.410423</td>
          <td>26.776679</td>
          <td>0.204968</td>
          <td>25.327637</td>
          <td>0.052268</td>
          <td>25.084564</td>
          <td>0.069355</td>
          <td>24.873637</td>
          <td>0.108926</td>
          <td>25.245675</td>
          <td>0.324254</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.777133</td>
          <td>1.709971</td>
          <td>26.479365</td>
          <td>0.157044</td>
          <td>26.088485</td>
          <td>0.100625</td>
          <td>25.308928</td>
          <td>0.083085</td>
          <td>24.748711</td>
          <td>0.096024</td>
          <td>24.241341</td>
          <td>0.138196</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.576849</td>
          <td>0.893553</td>
          <td>27.015227</td>
          <td>0.248095</td>
          <td>26.232386</td>
          <td>0.115080</td>
          <td>26.392037</td>
          <td>0.213208</td>
          <td>25.700091</td>
          <td>0.219057</td>
          <td>26.026123</td>
          <td>0.580963</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.903276</td>
          <td>0.266487</td>
          <td>26.120774</td>
          <td>0.118111</td>
          <td>26.023996</td>
          <td>0.097719</td>
          <td>25.838392</td>
          <td>0.135671</td>
          <td>25.663434</td>
          <td>0.216250</td>
          <td>25.682319</td>
          <td>0.458919</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.542093</td>
          <td>0.872887</td>
          <td>27.027898</td>
          <td>0.250109</td>
          <td>26.476530</td>
          <td>0.141792</td>
          <td>26.356362</td>
          <td>0.206382</td>
          <td>26.892341</td>
          <td>0.555357</td>
          <td>25.779133</td>
          <td>0.484215</td>
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
          <td>26.713898</td>
          <td>0.166369</td>
          <td>26.184349</td>
          <td>0.092760</td>
          <td>25.236444</td>
          <td>0.065487</td>
          <td>24.644491</td>
          <td>0.074198</td>
          <td>24.068398</td>
          <td>0.100368</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.423644</td>
          <td>0.299987</td>
          <td>26.696265</td>
          <td>0.144934</td>
          <td>26.357220</td>
          <td>0.174121</td>
          <td>26.300683</td>
          <td>0.304759</td>
          <td>25.093316</td>
          <td>0.241053</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.636542</td>
          <td>0.440563</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.005649</td>
          <td>0.453481</td>
          <td>25.905355</td>
          <td>0.128320</td>
          <td>24.916496</td>
          <td>0.102304</td>
          <td>24.311695</td>
          <td>0.134881</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.442392</td>
          <td>2.161098</td>
          <td>27.107724</td>
          <td>0.254017</td>
          <td>26.367552</td>
          <td>0.219557</td>
          <td>25.519975</td>
          <td>0.197738</td>
          <td>25.281947</td>
          <td>0.347081</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.029387</td>
          <td>0.259230</td>
          <td>26.223966</td>
          <td>0.109143</td>
          <td>26.046292</td>
          <td>0.082251</td>
          <td>25.708339</td>
          <td>0.099427</td>
          <td>25.457233</td>
          <td>0.151037</td>
          <td>25.056631</td>
          <td>0.233964</td>
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
          <td>26.218507</td>
          <td>0.116180</td>
          <td>25.479647</td>
          <td>0.053843</td>
          <td>24.990109</td>
          <td>0.057200</td>
          <td>24.642690</td>
          <td>0.080172</td>
          <td>24.710549</td>
          <td>0.189057</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.801988</td>
          <td>0.479129</td>
          <td>26.593105</td>
          <td>0.152145</td>
          <td>26.130519</td>
          <td>0.089933</td>
          <td>25.270806</td>
          <td>0.068691</td>
          <td>24.815044</td>
          <td>0.087672</td>
          <td>24.301717</td>
          <td>0.125100</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.131454</td>
          <td>0.290473</td>
          <td>26.378635</td>
          <td>0.130068</td>
          <td>26.365856</td>
          <td>0.114144</td>
          <td>26.203021</td>
          <td>0.160306</td>
          <td>25.605191</td>
          <td>0.179424</td>
          <td>25.298528</td>
          <td>0.298395</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.388391</td>
          <td>0.371770</td>
          <td>26.444865</td>
          <td>0.145802</td>
          <td>26.119124</td>
          <td>0.098226</td>
          <td>25.879126</td>
          <td>0.129746</td>
          <td>25.732649</td>
          <td>0.212703</td>
          <td>24.895069</td>
          <td>0.228559</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.891185</td>
          <td>0.518540</td>
          <td>26.996850</td>
          <td>0.218123</td>
          <td>26.708524</td>
          <td>0.152007</td>
          <td>26.648599</td>
          <td>0.230989</td>
          <td>25.902302</td>
          <td>0.228023</td>
          <td>26.221187</td>
          <td>0.596111</td>
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
