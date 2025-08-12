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

    <pzflow.flow.Flow at 0x7f0a89689720>



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
    0      23.994413  0.043585  0.041275  
    1      25.391064  0.001956  0.001566  
    2      24.304707  0.057758  0.036974  
    3      25.291103  0.018601  0.010497  
    4      25.096743  0.110029  0.086736  
    ...          ...       ...       ...  
    99995  24.737946  0.027791  0.023456  
    99996  24.224169  0.065943  0.052140  
    99997  25.613836  0.402022  0.309530  
    99998  25.274899  0.020209  0.020036  
    99999  25.699642  0.013596  0.011560  
    
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
          <td>27.758190</td>
          <td>0.914358</td>
          <td>26.769052</td>
          <td>0.174337</td>
          <td>25.942361</td>
          <td>0.074932</td>
          <td>25.119236</td>
          <td>0.059014</td>
          <td>24.561625</td>
          <td>0.068944</td>
          <td>24.094890</td>
          <td>0.102710</td>
          <td>0.043585</td>
          <td>0.041275</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.000692</td>
          <td>0.469333</td>
          <td>26.517479</td>
          <td>0.124073</td>
          <td>26.181916</td>
          <td>0.149764</td>
          <td>26.162383</td>
          <td>0.272293</td>
          <td>25.290759</td>
          <td>0.283028</td>
          <td>0.001956</td>
          <td>0.001566</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.556905</td>
          <td>0.394005</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.519174</td>
          <td>0.288272</td>
          <td>25.890936</td>
          <td>0.116448</td>
          <td>24.757254</td>
          <td>0.081955</td>
          <td>24.347962</td>
          <td>0.128035</td>
          <td>0.057758</td>
          <td>0.036974</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.198316</td>
          <td>0.631684</td>
          <td>28.930509</td>
          <td>0.891419</td>
          <td>27.117397</td>
          <td>0.207073</td>
          <td>26.462355</td>
          <td>0.190147</td>
          <td>25.328986</td>
          <td>0.135062</td>
          <td>25.318885</td>
          <td>0.289541</td>
          <td>0.018601</td>
          <td>0.010497</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.591618</td>
          <td>0.404671</td>
          <td>26.291347</td>
          <td>0.115599</td>
          <td>25.910655</td>
          <td>0.072861</td>
          <td>25.730474</td>
          <td>0.101223</td>
          <td>25.278619</td>
          <td>0.129305</td>
          <td>25.238765</td>
          <td>0.271326</td>
          <td>0.110029</td>
          <td>0.086736</td>
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
          <td>26.330520</td>
          <td>0.119602</td>
          <td>25.456059</td>
          <td>0.048685</td>
          <td>25.121562</td>
          <td>0.059135</td>
          <td>24.851917</td>
          <td>0.089081</td>
          <td>24.955067</td>
          <td>0.214726</td>
          <td>0.027791</td>
          <td>0.023456</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.654335</td>
          <td>0.424545</td>
          <td>26.737868</td>
          <td>0.169779</td>
          <td>25.984954</td>
          <td>0.077806</td>
          <td>25.080533</td>
          <td>0.057021</td>
          <td>25.150753</td>
          <td>0.115718</td>
          <td>24.290417</td>
          <td>0.121801</td>
          <td>0.065943</td>
          <td>0.052140</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.514176</td>
          <td>0.782311</td>
          <td>26.453995</td>
          <td>0.133102</td>
          <td>26.483954</td>
          <td>0.120513</td>
          <td>25.796433</td>
          <td>0.107235</td>
          <td>25.961152</td>
          <td>0.230806</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.402022</td>
          <td>0.309530</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.200708</td>
          <td>0.632739</td>
          <td>26.161606</td>
          <td>0.103234</td>
          <td>26.033554</td>
          <td>0.081216</td>
          <td>25.803183</td>
          <td>0.107870</td>
          <td>26.118259</td>
          <td>0.262668</td>
          <td>25.769748</td>
          <td>0.412969</td>
          <td>0.020209</td>
          <td>0.020036</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.007229</td>
          <td>0.254341</td>
          <td>26.723271</td>
          <td>0.167683</td>
          <td>26.720659</td>
          <td>0.147868</td>
          <td>26.575590</td>
          <td>0.209125</td>
          <td>26.010706</td>
          <td>0.240461</td>
          <td>25.770617</td>
          <td>0.413244</td>
          <td>0.013596</td>
          <td>0.011560</td>
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
          <td>28.933305</td>
          <td>1.836920</td>
          <td>26.649087</td>
          <td>0.181731</td>
          <td>25.970455</td>
          <td>0.090900</td>
          <td>25.217542</td>
          <td>0.076801</td>
          <td>24.774157</td>
          <td>0.098377</td>
          <td>23.913539</td>
          <td>0.104153</td>
          <td>0.043585</td>
          <td>0.041275</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.442963</td>
          <td>0.814678</td>
          <td>27.192541</td>
          <td>0.283646</td>
          <td>26.919765</td>
          <td>0.204688</td>
          <td>26.413790</td>
          <td>0.214398</td>
          <td>25.750873</td>
          <td>0.225755</td>
          <td>25.142219</td>
          <td>0.292648</td>
          <td>0.001956</td>
          <td>0.001566</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.039946</td>
          <td>1.174086</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.789847</td>
          <td>0.415221</td>
          <td>25.979643</td>
          <td>0.149605</td>
          <td>24.946567</td>
          <td>0.114578</td>
          <td>24.179514</td>
          <td>0.131507</td>
          <td>0.057758</td>
          <td>0.036974</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.223029</td>
          <td>1.169607</td>
          <td>26.916911</td>
          <td>0.204348</td>
          <td>26.507097</td>
          <td>0.231868</td>
          <td>25.539628</td>
          <td>0.189298</td>
          <td>24.821380</td>
          <td>0.225189</td>
          <td>0.018601</td>
          <td>0.010497</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.159856</td>
          <td>0.327739</td>
          <td>25.983159</td>
          <td>0.104852</td>
          <td>25.994571</td>
          <td>0.095310</td>
          <td>25.625707</td>
          <td>0.112909</td>
          <td>25.424804</td>
          <td>0.177063</td>
          <td>25.132638</td>
          <td>0.299352</td>
          <td>0.110029</td>
          <td>0.086736</td>
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
          <td>26.556725</td>
          <td>0.436920</td>
          <td>26.283554</td>
          <td>0.132494</td>
          <td>25.460107</td>
          <td>0.057678</td>
          <td>25.281888</td>
          <td>0.080963</td>
          <td>24.816864</td>
          <td>0.101734</td>
          <td>24.246985</td>
          <td>0.138597</td>
          <td>0.027791</td>
          <td>0.023456</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.863464</td>
          <td>1.784338</td>
          <td>27.129593</td>
          <td>0.272204</td>
          <td>25.955622</td>
          <td>0.090244</td>
          <td>25.000389</td>
          <td>0.063762</td>
          <td>24.797944</td>
          <td>0.101029</td>
          <td>24.252813</td>
          <td>0.140655</td>
          <td>0.065943</td>
          <td>0.052140</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.596294</td>
          <td>1.058110</td>
          <td>26.883344</td>
          <td>0.287118</td>
          <td>26.375853</td>
          <td>0.173825</td>
          <td>26.497010</td>
          <td>0.307927</td>
          <td>25.689317</td>
          <td>0.285952</td>
          <td>26.542444</td>
          <td>1.020391</td>
          <td>0.402022</td>
          <td>0.309530</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.194797</td>
          <td>0.329822</td>
          <td>26.372932</td>
          <td>0.142992</td>
          <td>26.094266</td>
          <td>0.100851</td>
          <td>25.928134</td>
          <td>0.142177</td>
          <td>25.789722</td>
          <td>0.233445</td>
          <td>24.912700</td>
          <td>0.243006</td>
          <td>0.020209</td>
          <td>0.020036</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.090200</td>
          <td>0.643127</td>
          <td>26.540009</td>
          <td>0.164856</td>
          <td>26.516514</td>
          <td>0.145391</td>
          <td>26.376392</td>
          <td>0.207910</td>
          <td>26.057344</td>
          <td>0.290333</td>
          <td>25.814999</td>
          <td>0.493174</td>
          <td>0.013596</td>
          <td>0.011560</td>
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
          <td>27.359058</td>
          <td>0.714561</td>
          <td>26.786053</td>
          <td>0.180449</td>
          <td>26.061753</td>
          <td>0.085282</td>
          <td>25.205034</td>
          <td>0.065314</td>
          <td>24.782301</td>
          <td>0.085815</td>
          <td>23.915619</td>
          <td>0.089965</td>
          <td>0.043585</td>
          <td>0.041275</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.363440</td>
          <td>0.285567</td>
          <td>26.662970</td>
          <td>0.140714</td>
          <td>26.131982</td>
          <td>0.143478</td>
          <td>26.407803</td>
          <td>0.331677</td>
          <td>25.150198</td>
          <td>0.252384</td>
          <td>0.001956</td>
          <td>0.001566</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.154588</td>
          <td>1.041208</td>
          <td>28.017012</td>
          <td>0.437766</td>
          <td>26.293329</td>
          <td>0.169982</td>
          <td>25.127039</td>
          <td>0.116861</td>
          <td>24.221387</td>
          <td>0.118395</td>
          <td>0.057758</td>
          <td>0.036974</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.818395</td>
          <td>0.409774</td>
          <td>27.332068</td>
          <td>0.248188</td>
          <td>26.629493</td>
          <td>0.219424</td>
          <td>25.519100</td>
          <td>0.159514</td>
          <td>25.019375</td>
          <td>0.227221</td>
          <td>0.018601</td>
          <td>0.010497</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.897266</td>
          <td>0.251404</td>
          <td>25.925872</td>
          <td>0.093085</td>
          <td>25.814240</td>
          <td>0.075285</td>
          <td>25.500569</td>
          <td>0.093487</td>
          <td>25.407365</td>
          <td>0.161965</td>
          <td>25.250537</td>
          <td>0.306182</td>
          <td>0.110029</td>
          <td>0.086736</td>
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
          <td>26.777601</td>
          <td>0.468387</td>
          <td>26.377447</td>
          <td>0.125527</td>
          <td>25.479579</td>
          <td>0.050163</td>
          <td>25.016445</td>
          <td>0.054380</td>
          <td>25.002331</td>
          <td>0.102558</td>
          <td>24.889406</td>
          <td>0.205050</td>
          <td>0.027791</td>
          <td>0.023456</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.733822</td>
          <td>0.175842</td>
          <td>26.043846</td>
          <td>0.085803</td>
          <td>25.150960</td>
          <td>0.063708</td>
          <td>24.698778</td>
          <td>0.081487</td>
          <td>24.091303</td>
          <td>0.107336</td>
          <td>0.065943</td>
          <td>0.052140</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.555893</td>
          <td>0.647021</td>
          <td>26.863777</td>
          <td>0.353215</td>
          <td>26.614375</td>
          <td>0.270339</td>
          <td>26.507670</td>
          <td>0.391933</td>
          <td>26.403493</td>
          <td>0.613956</td>
          <td>25.508540</td>
          <td>0.631949</td>
          <td>0.402022</td>
          <td>0.309530</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.193977</td>
          <td>0.297002</td>
          <td>26.335852</td>
          <td>0.120723</td>
          <td>26.147567</td>
          <td>0.090290</td>
          <td>25.824525</td>
          <td>0.110528</td>
          <td>25.522161</td>
          <td>0.160303</td>
          <td>25.604992</td>
          <td>0.365366</td>
          <td>0.020209</td>
          <td>0.020036</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.033749</td>
          <td>0.562894</td>
          <td>26.607324</td>
          <td>0.152151</td>
          <td>26.264171</td>
          <td>0.099691</td>
          <td>26.240151</td>
          <td>0.157777</td>
          <td>25.909543</td>
          <td>0.221575</td>
          <td>26.192643</td>
          <td>0.566315</td>
          <td>0.013596</td>
          <td>0.011560</td>
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
