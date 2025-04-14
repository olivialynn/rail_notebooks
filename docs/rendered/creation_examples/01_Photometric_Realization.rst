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

    <pzflow.flow.Flow at 0x7f73c1055d80>



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
          <td>28.256934</td>
          <td>1.225393</td>
          <td>26.712312</td>
          <td>0.166126</td>
          <td>26.020431</td>
          <td>0.080281</td>
          <td>25.143935</td>
          <td>0.060321</td>
          <td>24.724021</td>
          <td>0.079587</td>
          <td>23.955179</td>
          <td>0.090863</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.569444</td>
          <td>0.811025</td>
          <td>26.842482</td>
          <td>0.185521</td>
          <td>26.317012</td>
          <td>0.104187</td>
          <td>26.092647</td>
          <td>0.138691</td>
          <td>26.033354</td>
          <td>0.244993</td>
          <td>25.001256</td>
          <td>0.223149</td>
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
          <td>27.249961</td>
          <td>0.231251</td>
          <td>26.099512</td>
          <td>0.139515</td>
          <td>24.983092</td>
          <td>0.099954</td>
          <td>24.326908</td>
          <td>0.125720</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.524167</td>
          <td>0.682827</td>
          <td>27.146205</td>
          <td>0.212123</td>
          <td>26.068449</td>
          <td>0.135825</td>
          <td>25.515439</td>
          <td>0.158539</td>
          <td>24.948706</td>
          <td>0.213589</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.069186</td>
          <td>0.267539</td>
          <td>26.015576</td>
          <td>0.090836</td>
          <td>25.859196</td>
          <td>0.069618</td>
          <td>25.546046</td>
          <td>0.086085</td>
          <td>25.361570</td>
          <td>0.138914</td>
          <td>24.881792</td>
          <td>0.201954</td>
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
          <td>27.353740</td>
          <td>0.702940</td>
          <td>26.493409</td>
          <td>0.137707</td>
          <td>25.424497</td>
          <td>0.047340</td>
          <td>25.036794</td>
          <td>0.054849</td>
          <td>24.910957</td>
          <td>0.093825</td>
          <td>24.885710</td>
          <td>0.202619</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.434381</td>
          <td>0.742095</td>
          <td>26.425812</td>
          <td>0.129899</td>
          <td>26.177989</td>
          <td>0.092231</td>
          <td>25.275847</td>
          <td>0.067804</td>
          <td>24.959176</td>
          <td>0.097880</td>
          <td>24.045962</td>
          <td>0.098401</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.515472</td>
          <td>0.381582</td>
          <td>27.302362</td>
          <td>0.271752</td>
          <td>26.551585</td>
          <td>0.127797</td>
          <td>26.117654</td>
          <td>0.141713</td>
          <td>26.355978</td>
          <td>0.318270</td>
          <td>25.084468</td>
          <td>0.239078</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.423962</td>
          <td>0.355303</td>
          <td>26.079489</td>
          <td>0.096074</td>
          <td>26.027247</td>
          <td>0.080766</td>
          <td>25.801621</td>
          <td>0.107722</td>
          <td>26.250131</td>
          <td>0.292359</td>
          <td>25.176770</td>
          <td>0.257932</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.954722</td>
          <td>0.530987</td>
          <td>26.571031</td>
          <td>0.147217</td>
          <td>26.288705</td>
          <td>0.101638</td>
          <td>26.739257</td>
          <td>0.239600</td>
          <td>26.369553</td>
          <td>0.321733</td>
          <td>25.921510</td>
          <td>0.463291</td>
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
          <td>26.428880</td>
          <td>0.149851</td>
          <td>26.208239</td>
          <td>0.111263</td>
          <td>25.040173</td>
          <td>0.065235</td>
          <td>24.798178</td>
          <td>0.099863</td>
          <td>23.849665</td>
          <td>0.097878</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.838206</td>
          <td>0.537649</td>
          <td>27.663949</td>
          <td>0.411436</td>
          <td>26.360925</td>
          <td>0.127081</td>
          <td>26.593781</td>
          <td>0.248931</td>
          <td>26.277767</td>
          <td>0.346067</td>
          <td>25.579755</td>
          <td>0.413006</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.422141</td>
          <td>0.722543</td>
          <td>27.482492</td>
          <td>0.330964</td>
          <td>25.985477</td>
          <td>0.152556</td>
          <td>25.039169</td>
          <td>0.125956</td>
          <td>24.136089</td>
          <td>0.128515</td>
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
          <td>27.238320</td>
          <td>0.283485</td>
          <td>26.137331</td>
          <td>0.181599</td>
          <td>25.397998</td>
          <td>0.179007</td>
          <td>25.874910</td>
          <td>0.545447</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.538438</td>
          <td>0.430371</td>
          <td>25.983360</td>
          <td>0.101896</td>
          <td>26.043730</td>
          <td>0.096384</td>
          <td>25.754403</td>
          <td>0.122218</td>
          <td>25.513748</td>
          <td>0.185131</td>
          <td>25.337965</td>
          <td>0.342243</td>
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
          <td>26.697404</td>
          <td>0.491405</td>
          <td>26.558098</td>
          <td>0.170443</td>
          <td>25.403084</td>
          <td>0.055886</td>
          <td>25.085756</td>
          <td>0.069428</td>
          <td>24.795809</td>
          <td>0.101763</td>
          <td>24.738290</td>
          <td>0.214321</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.851741</td>
          <td>0.544298</td>
          <td>26.619204</td>
          <td>0.176898</td>
          <td>26.036480</td>
          <td>0.096142</td>
          <td>25.183195</td>
          <td>0.074359</td>
          <td>25.043909</td>
          <td>0.124233</td>
          <td>24.127441</td>
          <td>0.125234</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.973141</td>
          <td>0.278340</td>
          <td>26.911869</td>
          <td>0.227802</td>
          <td>26.196497</td>
          <td>0.111537</td>
          <td>26.460095</td>
          <td>0.225641</td>
          <td>26.347825</td>
          <td>0.369730</td>
          <td>25.739782</td>
          <td>0.471355</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.501030</td>
          <td>0.427006</td>
          <td>26.350941</td>
          <td>0.144101</td>
          <td>26.210322</td>
          <td>0.114995</td>
          <td>25.591329</td>
          <td>0.109480</td>
          <td>25.826687</td>
          <td>0.247561</td>
          <td>25.044147</td>
          <td>0.278479</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.085815</td>
          <td>0.304206</td>
          <td>26.839500</td>
          <td>0.213991</td>
          <td>26.433426</td>
          <td>0.136620</td>
          <td>26.136851</td>
          <td>0.171477</td>
          <td>25.839553</td>
          <td>0.245241</td>
          <td>26.694792</td>
          <td>0.907082</td>
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
          <td>26.909831</td>
          <td>0.513900</td>
          <td>26.441716</td>
          <td>0.131713</td>
          <td>25.993460</td>
          <td>0.078403</td>
          <td>25.185016</td>
          <td>0.062568</td>
          <td>24.716449</td>
          <td>0.079067</td>
          <td>24.061904</td>
          <td>0.099799</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.327604</td>
          <td>0.690922</td>
          <td>27.258519</td>
          <td>0.262408</td>
          <td>26.987275</td>
          <td>0.185768</td>
          <td>26.203915</td>
          <td>0.152766</td>
          <td>25.931767</td>
          <td>0.225448</td>
          <td>25.949879</td>
          <td>0.473620</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.311179</td>
          <td>0.714442</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.049757</td>
          <td>0.468736</td>
          <td>26.213750</td>
          <td>0.167258</td>
          <td>24.920601</td>
          <td>0.102672</td>
          <td>24.432603</td>
          <td>0.149679</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.221630</td>
          <td>0.730503</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.747457</td>
          <td>0.188185</td>
          <td>25.825680</td>
          <td>0.138637</td>
          <td>25.393216</td>
          <td>0.177670</td>
          <td>25.316960</td>
          <td>0.356766</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.106850</td>
          <td>0.592788</td>
          <td>26.184960</td>
          <td>0.105491</td>
          <td>26.080729</td>
          <td>0.084786</td>
          <td>25.494849</td>
          <td>0.082410</td>
          <td>25.579347</td>
          <td>0.167657</td>
          <td>25.115729</td>
          <td>0.245661</td>
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
          <td>26.622629</td>
          <td>0.164555</td>
          <td>25.457858</td>
          <td>0.052812</td>
          <td>25.027970</td>
          <td>0.059154</td>
          <td>24.805779</td>
          <td>0.092550</td>
          <td>24.965147</td>
          <td>0.233896</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.484026</td>
          <td>0.376177</td>
          <td>26.532809</td>
          <td>0.144472</td>
          <td>26.085355</td>
          <td>0.086428</td>
          <td>25.134484</td>
          <td>0.060873</td>
          <td>24.784308</td>
          <td>0.085331</td>
          <td>24.113181</td>
          <td>0.106160</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.887680</td>
          <td>0.520311</td>
          <td>26.780253</td>
          <td>0.183402</td>
          <td>26.498491</td>
          <td>0.128084</td>
          <td>26.226429</td>
          <td>0.163543</td>
          <td>25.927343</td>
          <td>0.235006</td>
          <td>25.210780</td>
          <td>0.277965</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.747059</td>
          <td>0.488291</td>
          <td>26.142146</td>
          <td>0.112208</td>
          <td>26.037965</td>
          <td>0.091473</td>
          <td>26.054679</td>
          <td>0.150938</td>
          <td>25.542012</td>
          <td>0.181191</td>
          <td>25.253669</td>
          <td>0.306267</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.358604</td>
          <td>1.316867</td>
          <td>27.121634</td>
          <td>0.241885</td>
          <td>26.522135</td>
          <td>0.129451</td>
          <td>26.378448</td>
          <td>0.184220</td>
          <td>26.464603</td>
          <td>0.359157</td>
          <td>25.434009</td>
          <td>0.329406</td>
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
