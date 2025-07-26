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

    <pzflow.flow.Flow at 0x7f6289276650>



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
    0      23.994413  0.103103  0.096770  
    1      25.391064  0.038943  0.030154  
    2      24.304707  0.048246  0.031841  
    3      25.291103  0.031832  0.020284  
    4      25.096743  0.113156  0.112596  
    ...          ...       ...       ...  
    99995  24.737946  0.057232  0.051740  
    99996  24.224169  0.130865  0.067060  
    99997  25.613836  0.041780  0.030115  
    99998  25.274899  0.171379  0.115046  
    99999  25.699642  0.000466  0.000286  
    
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
          <td>27.202091</td>
          <td>0.633350</td>
          <td>26.890741</td>
          <td>0.193228</td>
          <td>26.082717</td>
          <td>0.084814</td>
          <td>25.047942</td>
          <td>0.055395</td>
          <td>24.658070</td>
          <td>0.075084</td>
          <td>24.069153</td>
          <td>0.100421</td>
          <td>0.103103</td>
          <td>0.096770</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.508288</td>
          <td>1.401574</td>
          <td>27.200253</td>
          <td>0.249985</td>
          <td>26.576372</td>
          <td>0.130569</td>
          <td>26.481420</td>
          <td>0.193227</td>
          <td>25.487383</td>
          <td>0.154777</td>
          <td>25.264195</td>
          <td>0.276995</td>
          <td>0.038943</td>
          <td>0.030154</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.712875</td>
          <td>3.309514</td>
          <td>28.586392</td>
          <td>0.712309</td>
          <td>27.360416</td>
          <td>0.253304</td>
          <td>26.264038</td>
          <td>0.160677</td>
          <td>24.963556</td>
          <td>0.098257</td>
          <td>24.431717</td>
          <td>0.137650</td>
          <td>0.048246</td>
          <td>0.031841</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.532378</td>
          <td>0.386611</td>
          <td>28.283110</td>
          <td>0.576952</td>
          <td>27.511589</td>
          <td>0.286510</td>
          <td>26.359842</td>
          <td>0.174341</td>
          <td>25.653552</td>
          <td>0.178325</td>
          <td>25.212258</td>
          <td>0.265526</td>
          <td>0.031832</td>
          <td>0.020284</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.654781</td>
          <td>0.856738</td>
          <td>26.224279</td>
          <td>0.109040</td>
          <td>26.029271</td>
          <td>0.080910</td>
          <td>25.693708</td>
          <td>0.098014</td>
          <td>25.697290</td>
          <td>0.185053</td>
          <td>25.212852</td>
          <td>0.265655</td>
          <td>0.113156</td>
          <td>0.112596</td>
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
          <td>27.263841</td>
          <td>0.661052</td>
          <td>26.271932</td>
          <td>0.113662</td>
          <td>25.424801</td>
          <td>0.047353</td>
          <td>25.020466</td>
          <td>0.054060</td>
          <td>24.977822</td>
          <td>0.099493</td>
          <td>24.667228</td>
          <td>0.168447</td>
          <td>0.057232</td>
          <td>0.051740</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.861633</td>
          <td>1.668571</td>
          <td>26.723837</td>
          <td>0.167764</td>
          <td>25.963123</td>
          <td>0.076320</td>
          <td>25.125476</td>
          <td>0.059341</td>
          <td>24.804043</td>
          <td>0.085405</td>
          <td>24.301407</td>
          <td>0.122969</td>
          <td>0.130865</td>
          <td>0.067060</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.407693</td>
          <td>0.728971</td>
          <td>26.812546</td>
          <td>0.180884</td>
          <td>26.267183</td>
          <td>0.099739</td>
          <td>26.181359</td>
          <td>0.149693</td>
          <td>25.982423</td>
          <td>0.234907</td>
          <td>25.441735</td>
          <td>0.319543</td>
          <td>0.041780</td>
          <td>0.030115</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.786147</td>
          <td>0.468917</td>
          <td>26.181016</td>
          <td>0.104999</td>
          <td>26.029892</td>
          <td>0.080954</td>
          <td>25.838745</td>
          <td>0.111271</td>
          <td>25.715613</td>
          <td>0.187940</td>
          <td>24.949661</td>
          <td>0.213759</td>
          <td>0.171379</td>
          <td>0.115046</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.775430</td>
          <td>0.924200</td>
          <td>26.664715</td>
          <td>0.159516</td>
          <td>26.767615</td>
          <td>0.153948</td>
          <td>26.164239</td>
          <td>0.147508</td>
          <td>25.756858</td>
          <td>0.194590</td>
          <td>26.158959</td>
          <td>0.551738</td>
          <td>0.000466</td>
          <td>0.000286</td>
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
          <td>26.780258</td>
          <td>0.207704</td>
          <td>26.023948</td>
          <td>0.097859</td>
          <td>25.170786</td>
          <td>0.075781</td>
          <td>24.769308</td>
          <td>0.100615</td>
          <td>24.002381</td>
          <td>0.115668</td>
          <td>0.103103</td>
          <td>0.096770</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.632088</td>
          <td>0.920544</td>
          <td>29.517121</td>
          <td>1.374642</td>
          <td>26.884886</td>
          <td>0.199565</td>
          <td>26.215874</td>
          <td>0.182279</td>
          <td>26.301424</td>
          <td>0.353793</td>
          <td>24.830984</td>
          <td>0.227729</td>
          <td>0.038943</td>
          <td>0.030154</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.046823</td>
          <td>0.625905</td>
          <td>28.564262</td>
          <td>0.785047</td>
          <td>29.020227</td>
          <td>0.969704</td>
          <td>25.992220</td>
          <td>0.150879</td>
          <td>25.024038</td>
          <td>0.122284</td>
          <td>24.237095</td>
          <td>0.137893</td>
          <td>0.048246</td>
          <td>0.031841</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.324108</td>
          <td>0.666588</td>
          <td>27.309834</td>
          <td>0.282991</td>
          <td>26.380682</td>
          <td>0.209044</td>
          <td>25.604015</td>
          <td>0.200155</td>
          <td>25.009431</td>
          <td>0.263344</td>
          <td>0.031832</td>
          <td>0.020284</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.131010</td>
          <td>0.322401</td>
          <td>26.211035</td>
          <td>0.128886</td>
          <td>25.943317</td>
          <td>0.091967</td>
          <td>25.591695</td>
          <td>0.110660</td>
          <td>25.246284</td>
          <td>0.153442</td>
          <td>25.413623</td>
          <td>0.377102</td>
          <td>0.113156</td>
          <td>0.112596</td>
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
          <td>28.077680</td>
          <td>1.200376</td>
          <td>26.152321</td>
          <td>0.119085</td>
          <td>25.487518</td>
          <td>0.059565</td>
          <td>25.025205</td>
          <td>0.065053</td>
          <td>24.796543</td>
          <td>0.100720</td>
          <td>24.617249</td>
          <td>0.191556</td>
          <td>0.057232</td>
          <td>0.051740</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.163400</td>
          <td>0.690508</td>
          <td>26.847781</td>
          <td>0.220158</td>
          <td>26.074219</td>
          <td>0.102485</td>
          <td>25.343242</td>
          <td>0.088420</td>
          <td>24.960468</td>
          <td>0.119132</td>
          <td>24.244792</td>
          <td>0.142985</td>
          <td>0.130865</td>
          <td>0.067060</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>30.326047</td>
          <td>3.073144</td>
          <td>26.512670</td>
          <td>0.161617</td>
          <td>26.293227</td>
          <td>0.120337</td>
          <td>26.454080</td>
          <td>0.222695</td>
          <td>25.879375</td>
          <td>0.252087</td>
          <td>30.028793</td>
          <td>3.655167</td>
          <td>0.041780</td>
          <td>0.030115</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.636090</td>
          <td>0.483953</td>
          <td>26.094525</td>
          <td>0.119214</td>
          <td>25.945798</td>
          <td>0.094544</td>
          <td>26.393772</td>
          <td>0.225141</td>
          <td>26.214870</td>
          <td>0.349627</td>
          <td>25.182594</td>
          <td>0.321822</td>
          <td>0.171379</td>
          <td>0.115046</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.334100</td>
          <td>0.367653</td>
          <td>27.360274</td>
          <td>0.324508</td>
          <td>26.580125</td>
          <td>0.153469</td>
          <td>26.705312</td>
          <td>0.272645</td>
          <td>26.092858</td>
          <td>0.298615</td>
          <td>25.983893</td>
          <td>0.557632</td>
          <td>0.000466</td>
          <td>0.000286</td>
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
          <td>26.205402</td>
          <td>0.322969</td>
          <td>26.945506</td>
          <td>0.223610</td>
          <td>26.037369</td>
          <td>0.091875</td>
          <td>25.182756</td>
          <td>0.070828</td>
          <td>24.766925</td>
          <td>0.093173</td>
          <td>24.056906</td>
          <td>0.112373</td>
          <td>0.103103</td>
          <td>0.096770</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.834509</td>
          <td>0.490742</td>
          <td>27.425633</td>
          <td>0.304126</td>
          <td>26.630032</td>
          <td>0.138960</td>
          <td>26.293319</td>
          <td>0.167483</td>
          <td>26.048942</td>
          <td>0.251938</td>
          <td>26.676605</td>
          <td>0.798080</td>
          <td>0.038943</td>
          <td>0.030154</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.073078</td>
          <td>1.852187</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.195212</td>
          <td>0.496541</td>
          <td>26.227685</td>
          <td>0.159317</td>
          <td>25.027960</td>
          <td>0.106270</td>
          <td>24.308671</td>
          <td>0.126586</td>
          <td>0.048246</td>
          <td>0.031841</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.190609</td>
          <td>0.222131</td>
          <td>26.423334</td>
          <td>0.185768</td>
          <td>25.590822</td>
          <td>0.170643</td>
          <td>25.679122</td>
          <td>0.388509</td>
          <td>0.031832</td>
          <td>0.020284</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.154540</td>
          <td>0.116812</td>
          <td>26.067825</td>
          <td>0.097067</td>
          <td>25.612912</td>
          <td>0.106495</td>
          <td>25.520024</td>
          <td>0.183614</td>
          <td>25.539177</td>
          <td>0.395333</td>
          <td>0.113156</td>
          <td>0.112596</td>
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
          <td>27.129727</td>
          <td>0.615251</td>
          <td>26.137164</td>
          <td>0.104516</td>
          <td>25.411253</td>
          <td>0.048664</td>
          <td>25.090614</td>
          <td>0.059953</td>
          <td>24.784090</td>
          <td>0.087247</td>
          <td>24.835680</td>
          <td>0.201911</td>
          <td>0.057232</td>
          <td>0.051740</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.331016</td>
          <td>1.347504</td>
          <td>26.804804</td>
          <td>0.199581</td>
          <td>25.938652</td>
          <td>0.084573</td>
          <td>25.166927</td>
          <td>0.070125</td>
          <td>24.829916</td>
          <td>0.098864</td>
          <td>24.086808</td>
          <td>0.115799</td>
          <td>0.130865</td>
          <td>0.067060</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.766867</td>
          <td>0.176614</td>
          <td>26.466318</td>
          <td>0.120773</td>
          <td>26.947111</td>
          <td>0.288882</td>
          <td>26.240797</td>
          <td>0.294902</td>
          <td>25.007105</td>
          <td>0.228158</td>
          <td>0.041780</td>
          <td>0.030115</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.471214</td>
          <td>0.425810</td>
          <td>26.114193</td>
          <td>0.120536</td>
          <td>26.084912</td>
          <td>0.106040</td>
          <td>25.918318</td>
          <td>0.149559</td>
          <td>25.635450</td>
          <td>0.217048</td>
          <td>25.210934</td>
          <td>0.326975</td>
          <td>0.171379</td>
          <td>0.115046</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.078018</td>
          <td>0.580300</td>
          <td>26.486597</td>
          <td>0.136901</td>
          <td>26.706005</td>
          <td>0.146018</td>
          <td>26.234715</td>
          <td>0.156698</td>
          <td>25.820871</td>
          <td>0.205340</td>
          <td>25.800134</td>
          <td>0.422671</td>
          <td>0.000466</td>
          <td>0.000286</td>
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
