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

    <pzflow.flow.Flow at 0x7f518c1e0430>



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
          <td>26.575847</td>
          <td>0.147827</td>
          <td>26.105414</td>
          <td>0.086526</td>
          <td>25.186369</td>
          <td>0.062635</td>
          <td>24.742496</td>
          <td>0.080895</td>
          <td>24.008931</td>
          <td>0.095256</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.783334</td>
          <td>0.397947</td>
          <td>26.393394</td>
          <td>0.111376</td>
          <td>26.137333</td>
          <td>0.144134</td>
          <td>26.120932</td>
          <td>0.263242</td>
          <td>24.994021</td>
          <td>0.221810</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.793574</td>
          <td>0.471525</td>
          <td>28.704251</td>
          <td>0.770590</td>
          <td>27.837684</td>
          <td>0.371261</td>
          <td>26.127259</td>
          <td>0.142890</td>
          <td>24.905927</td>
          <td>0.093412</td>
          <td>24.601242</td>
          <td>0.159226</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.776652</td>
          <td>0.395903</td>
          <td>27.079118</td>
          <td>0.200532</td>
          <td>26.461941</td>
          <td>0.190080</td>
          <td>25.382026</td>
          <td>0.141385</td>
          <td>24.893837</td>
          <td>0.204004</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.915405</td>
          <td>0.235847</td>
          <td>26.341557</td>
          <td>0.120754</td>
          <td>25.788026</td>
          <td>0.065364</td>
          <td>25.599547</td>
          <td>0.090235</td>
          <td>25.707581</td>
          <td>0.186669</td>
          <td>25.169479</td>
          <td>0.256396</td>
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
          <td>27.360909</td>
          <td>0.706361</td>
          <td>26.330861</td>
          <td>0.119638</td>
          <td>25.475884</td>
          <td>0.049550</td>
          <td>25.043732</td>
          <td>0.055188</td>
          <td>24.971326</td>
          <td>0.098929</td>
          <td>24.996691</td>
          <td>0.222303</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.684748</td>
          <td>1.532262</td>
          <td>26.573082</td>
          <td>0.147477</td>
          <td>26.019312</td>
          <td>0.080202</td>
          <td>25.189527</td>
          <td>0.062810</td>
          <td>24.768270</td>
          <td>0.082755</td>
          <td>24.305661</td>
          <td>0.123424</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.263990</td>
          <td>0.661120</td>
          <td>26.797436</td>
          <td>0.178585</td>
          <td>26.293206</td>
          <td>0.102039</td>
          <td>25.995010</td>
          <td>0.127465</td>
          <td>25.799708</td>
          <td>0.201727</td>
          <td>26.800907</td>
          <td>0.854083</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.336595</td>
          <td>0.331662</td>
          <td>26.319029</td>
          <td>0.118415</td>
          <td>26.060615</td>
          <td>0.083178</td>
          <td>25.893011</td>
          <td>0.116658</td>
          <td>25.704199</td>
          <td>0.186137</td>
          <td>25.546537</td>
          <td>0.347218</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.943927</td>
          <td>1.023891</td>
          <td>26.794207</td>
          <td>0.178097</td>
          <td>26.574289</td>
          <td>0.130334</td>
          <td>26.110853</td>
          <td>0.140885</td>
          <td>26.216539</td>
          <td>0.284529</td>
          <td>25.311359</td>
          <td>0.287786</td>
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
          <td>27.438041</td>
          <td>0.812095</td>
          <td>26.612334</td>
          <td>0.175232</td>
          <td>26.078408</td>
          <td>0.099326</td>
          <td>25.279683</td>
          <td>0.080619</td>
          <td>24.519784</td>
          <td>0.078177</td>
          <td>24.065707</td>
          <td>0.118196</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.631533</td>
          <td>0.401324</td>
          <td>26.805239</td>
          <td>0.185920</td>
          <td>26.158170</td>
          <td>0.172902</td>
          <td>25.912555</td>
          <td>0.258012</td>
          <td>24.695497</td>
          <td>0.202620</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.944576</td>
          <td>0.588383</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.267916</td>
          <td>0.597868</td>
          <td>26.310919</td>
          <td>0.201084</td>
          <td>25.082452</td>
          <td>0.130767</td>
          <td>24.316048</td>
          <td>0.150080</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.534691</td>
          <td>0.448628</td>
          <td>27.625472</td>
          <td>0.421173</td>
          <td>27.626400</td>
          <td>0.385617</td>
          <td>26.129097</td>
          <td>0.180338</td>
          <td>25.376261</td>
          <td>0.175737</td>
          <td>25.091907</td>
          <td>0.299342</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.966406</td>
          <td>0.274407</td>
          <td>26.310920</td>
          <td>0.135433</td>
          <td>25.824401</td>
          <td>0.079469</td>
          <td>25.778634</td>
          <td>0.124815</td>
          <td>25.751494</td>
          <td>0.225946</td>
          <td>25.155961</td>
          <td>0.296004</td>
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
          <td>26.299768</td>
          <td>0.363074</td>
          <td>26.182065</td>
          <td>0.123404</td>
          <td>25.452120</td>
          <td>0.058370</td>
          <td>25.246898</td>
          <td>0.080051</td>
          <td>24.890204</td>
          <td>0.110512</td>
          <td>24.647371</td>
          <td>0.198609</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.580531</td>
          <td>0.171186</td>
          <td>25.880312</td>
          <td>0.083808</td>
          <td>25.293675</td>
          <td>0.081975</td>
          <td>24.928697</td>
          <td>0.112391</td>
          <td>24.457017</td>
          <td>0.166266</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.682550</td>
          <td>0.483457</td>
          <td>26.932133</td>
          <td>0.231659</td>
          <td>26.742640</td>
          <td>0.178477</td>
          <td>26.208054</td>
          <td>0.182654</td>
          <td>26.165718</td>
          <td>0.320269</td>
          <td>25.824915</td>
          <td>0.502089</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.425171</td>
          <td>0.402956</td>
          <td>26.285813</td>
          <td>0.136245</td>
          <td>26.036794</td>
          <td>0.098821</td>
          <td>25.597783</td>
          <td>0.110099</td>
          <td>25.881121</td>
          <td>0.258869</td>
          <td>25.193405</td>
          <td>0.314050</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.092613</td>
          <td>0.647901</td>
          <td>26.838864</td>
          <td>0.213877</td>
          <td>26.575411</td>
          <td>0.154359</td>
          <td>26.335263</td>
          <td>0.202764</td>
          <td>25.872190</td>
          <td>0.251911</td>
          <td>25.183078</td>
          <td>0.305300</td>
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
          <td>26.923155</td>
          <td>0.518938</td>
          <td>26.570133</td>
          <td>0.147120</td>
          <td>26.193902</td>
          <td>0.093541</td>
          <td>25.032616</td>
          <td>0.054654</td>
          <td>24.738312</td>
          <td>0.080607</td>
          <td>24.197523</td>
          <td>0.112358</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.819147</td>
          <td>0.949891</td>
          <td>27.140959</td>
          <td>0.238254</td>
          <td>26.605915</td>
          <td>0.134073</td>
          <td>26.641257</td>
          <td>0.221113</td>
          <td>26.172379</td>
          <td>0.274756</td>
          <td>25.548891</td>
          <td>0.348169</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>33.193711</td>
          <td>5.809231</td>
          <td>29.692000</td>
          <td>1.440008</td>
          <td>28.576299</td>
          <td>0.683500</td>
          <td>26.000715</td>
          <td>0.139342</td>
          <td>25.019885</td>
          <td>0.111973</td>
          <td>24.411338</td>
          <td>0.146970</td>
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
          <td>27.070222</td>
          <td>0.246309</td>
          <td>26.331296</td>
          <td>0.213019</td>
          <td>25.601077</td>
          <td>0.211645</td>
          <td>25.255197</td>
          <td>0.339833</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.019472</td>
          <td>0.257138</td>
          <td>26.049988</td>
          <td>0.093737</td>
          <td>25.969030</td>
          <td>0.076829</td>
          <td>25.609594</td>
          <td>0.091172</td>
          <td>25.389790</td>
          <td>0.142531</td>
          <td>25.018929</td>
          <td>0.226766</td>
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
          <td>27.397809</td>
          <td>0.755512</td>
          <td>26.268054</td>
          <td>0.121289</td>
          <td>25.484834</td>
          <td>0.054092</td>
          <td>25.134161</td>
          <td>0.064994</td>
          <td>24.933245</td>
          <td>0.103491</td>
          <td>25.047393</td>
          <td>0.250309</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.788287</td>
          <td>3.393932</td>
          <td>26.964128</td>
          <td>0.208333</td>
          <td>26.073548</td>
          <td>0.085534</td>
          <td>25.183630</td>
          <td>0.063585</td>
          <td>24.694445</td>
          <td>0.078830</td>
          <td>24.286681</td>
          <td>0.123478</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.047384</td>
          <td>1.113137</td>
          <td>26.758867</td>
          <td>0.180114</td>
          <td>26.672223</td>
          <td>0.148791</td>
          <td>26.174791</td>
          <td>0.156483</td>
          <td>25.533670</td>
          <td>0.168849</td>
          <td>26.165230</td>
          <td>0.577668</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.120811</td>
          <td>0.300871</td>
          <td>26.214216</td>
          <td>0.119465</td>
          <td>26.128302</td>
          <td>0.099019</td>
          <td>25.792953</td>
          <td>0.120403</td>
          <td>25.851929</td>
          <td>0.234873</td>
          <td>25.123623</td>
          <td>0.275746</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.546451</td>
          <td>0.815247</td>
          <td>27.198722</td>
          <td>0.257699</td>
          <td>26.500284</td>
          <td>0.127024</td>
          <td>26.316409</td>
          <td>0.174784</td>
          <td>26.078371</td>
          <td>0.263599</td>
          <td>25.900202</td>
          <td>0.471902</td>
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
