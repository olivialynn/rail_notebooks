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

    <pzflow.flow.Flow at 0x7f303b9d4730>



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
    0      23.994413  0.041970  0.021157  
    1      25.391064  0.056076  0.038854  
    2      24.304707  0.049368  0.034505  
    3      25.291103  0.041840  0.041543  
    4      25.096743  0.067177  0.034010  
    ...          ...       ...       ...  
    99995  24.737946  0.216483  0.138152  
    99996  24.224169  0.060186  0.034984  
    99997  25.613836  0.033168  0.020170  
    99998  25.274899  0.123667  0.062581  
    99999  25.699642  0.062758  0.038068  
    
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
          <td>27.394022</td>
          <td>0.722312</td>
          <td>26.707818</td>
          <td>0.165491</td>
          <td>25.997859</td>
          <td>0.078698</td>
          <td>25.411533</td>
          <td>0.076452</td>
          <td>24.595711</td>
          <td>0.071055</td>
          <td>24.130520</td>
          <td>0.105961</td>
          <td>0.041970</td>
          <td>0.021157</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.663151</td>
          <td>0.362492</td>
          <td>26.698037</td>
          <td>0.145021</td>
          <td>26.297454</td>
          <td>0.165326</td>
          <td>25.991415</td>
          <td>0.236660</td>
          <td>25.730398</td>
          <td>0.400680</td>
          <td>0.056076</td>
          <td>0.038854</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.303771</td>
          <td>0.585513</td>
          <td>28.725748</td>
          <td>0.710201</td>
          <td>25.969551</td>
          <td>0.124682</td>
          <td>25.146181</td>
          <td>0.115258</td>
          <td>24.318966</td>
          <td>0.124857</td>
          <td>0.049368</td>
          <td>0.034505</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.752137</td>
          <td>0.347186</td>
          <td>26.452649</td>
          <td>0.188595</td>
          <td>25.443536</td>
          <td>0.149065</td>
          <td>25.206715</td>
          <td>0.264327</td>
          <td>0.041840</td>
          <td>0.041543</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.379457</td>
          <td>0.343087</td>
          <td>26.239043</td>
          <td>0.110452</td>
          <td>26.143741</td>
          <td>0.089495</td>
          <td>25.851085</td>
          <td>0.112475</td>
          <td>25.567520</td>
          <td>0.165748</td>
          <td>24.778747</td>
          <td>0.185164</td>
          <td>0.067177</td>
          <td>0.034010</td>
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
          <td>27.446477</td>
          <td>0.748096</td>
          <td>26.554869</td>
          <td>0.145188</td>
          <td>25.466355</td>
          <td>0.049132</td>
          <td>25.024503</td>
          <td>0.054254</td>
          <td>24.832969</td>
          <td>0.087608</td>
          <td>24.593760</td>
          <td>0.158210</td>
          <td>0.216483</td>
          <td>0.138152</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.881974</td>
          <td>0.503454</td>
          <td>26.844199</td>
          <td>0.185791</td>
          <td>26.048486</td>
          <td>0.082293</td>
          <td>25.169177</td>
          <td>0.061687</td>
          <td>24.882847</td>
          <td>0.091537</td>
          <td>23.944997</td>
          <td>0.090053</td>
          <td>0.060186</td>
          <td>0.034984</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.990781</td>
          <td>1.052726</td>
          <td>26.584326</td>
          <td>0.148907</td>
          <td>26.364275</td>
          <td>0.108581</td>
          <td>26.437601</td>
          <td>0.186214</td>
          <td>25.949481</td>
          <td>0.228583</td>
          <td>25.298248</td>
          <td>0.284750</td>
          <td>0.033168</td>
          <td>0.020170</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.747206</td>
          <td>0.455436</td>
          <td>26.167461</td>
          <td>0.103763</td>
          <td>25.994665</td>
          <td>0.078476</td>
          <td>25.838245</td>
          <td>0.111222</td>
          <td>25.465221</td>
          <td>0.151865</td>
          <td>24.945997</td>
          <td>0.213106</td>
          <td>0.123667</td>
          <td>0.062581</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.192982</td>
          <td>0.629337</td>
          <td>26.535449</td>
          <td>0.142784</td>
          <td>26.437305</td>
          <td>0.115720</td>
          <td>26.210897</td>
          <td>0.153534</td>
          <td>25.765589</td>
          <td>0.196026</td>
          <td>25.743483</td>
          <td>0.404732</td>
          <td>0.062758</td>
          <td>0.038068</td>
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
          <td>27.356534</td>
          <td>0.771590</td>
          <td>26.650244</td>
          <td>0.181536</td>
          <td>25.891549</td>
          <td>0.084603</td>
          <td>25.334545</td>
          <td>0.084940</td>
          <td>24.637417</td>
          <td>0.087040</td>
          <td>24.015703</td>
          <td>0.113592</td>
          <td>0.041970</td>
          <td>0.021157</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.367901</td>
          <td>0.328579</td>
          <td>26.996974</td>
          <td>0.219960</td>
          <td>26.071352</td>
          <td>0.161811</td>
          <td>25.715655</td>
          <td>0.220875</td>
          <td>26.105351</td>
          <td>0.611988</td>
          <td>0.056076</td>
          <td>0.038854</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.665002</td>
          <td>0.940485</td>
          <td>28.786879</td>
          <td>0.905554</td>
          <td>28.126982</td>
          <td>0.533250</td>
          <td>26.308016</td>
          <td>0.197409</td>
          <td>24.873382</td>
          <td>0.107302</td>
          <td>24.249670</td>
          <td>0.139464</td>
          <td>0.049368</td>
          <td>0.034505</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.754703</td>
          <td>0.507656</td>
          <td>28.392320</td>
          <td>0.700090</td>
          <td>27.397283</td>
          <td>0.304630</td>
          <td>26.141924</td>
          <td>0.171499</td>
          <td>25.468931</td>
          <td>0.179193</td>
          <td>24.889252</td>
          <td>0.239390</td>
          <td>0.041840</td>
          <td>0.041543</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.128228</td>
          <td>0.314583</td>
          <td>26.110910</td>
          <td>0.114824</td>
          <td>26.019567</td>
          <td>0.095232</td>
          <td>25.849744</td>
          <td>0.133988</td>
          <td>25.469160</td>
          <td>0.179858</td>
          <td>24.835801</td>
          <td>0.229845</td>
          <td>0.067177</td>
          <td>0.034010</td>
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
          <td>26.616094</td>
          <td>0.487298</td>
          <td>26.189410</td>
          <td>0.133274</td>
          <td>25.384312</td>
          <td>0.059503</td>
          <td>25.203464</td>
          <td>0.083560</td>
          <td>24.683894</td>
          <td>0.099766</td>
          <td>24.969074</td>
          <td>0.279406</td>
          <td>0.216483</td>
          <td>0.138152</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.673285</td>
          <td>0.478703</td>
          <td>26.630708</td>
          <td>0.179249</td>
          <td>25.872628</td>
          <td>0.083575</td>
          <td>25.096071</td>
          <td>0.069131</td>
          <td>24.886832</td>
          <td>0.108789</td>
          <td>24.319367</td>
          <td>0.148385</td>
          <td>0.060186</td>
          <td>0.034984</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.590851</td>
          <td>0.448419</td>
          <td>27.035305</td>
          <td>0.250046</td>
          <td>26.480958</td>
          <td>0.141289</td>
          <td>26.229788</td>
          <td>0.184151</td>
          <td>25.807896</td>
          <td>0.237244</td>
          <td>26.139405</td>
          <td>0.624076</td>
          <td>0.033168</td>
          <td>0.020170</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>28.677170</td>
          <td>1.649937</td>
          <td>26.328479</td>
          <td>0.141334</td>
          <td>26.284922</td>
          <td>0.122685</td>
          <td>25.728805</td>
          <td>0.123377</td>
          <td>25.699119</td>
          <td>0.222747</td>
          <td>25.543718</td>
          <td>0.413090</td>
          <td>0.123667</td>
          <td>0.062581</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.841925</td>
          <td>1.765043</td>
          <td>26.525180</td>
          <td>0.164007</td>
          <td>26.591525</td>
          <td>0.156352</td>
          <td>26.169120</td>
          <td>0.176068</td>
          <td>25.742731</td>
          <td>0.226162</td>
          <td>25.431261</td>
          <td>0.371188</td>
          <td>0.062758</td>
          <td>0.038068</td>
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
          <td>27.117970</td>
          <td>0.236416</td>
          <td>26.069978</td>
          <td>0.085115</td>
          <td>25.279710</td>
          <td>0.069105</td>
          <td>24.640598</td>
          <td>0.075036</td>
          <td>23.961339</td>
          <td>0.092764</td>
          <td>0.041970</td>
          <td>0.021157</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.319454</td>
          <td>1.284840</td>
          <td>27.634046</td>
          <td>0.362810</td>
          <td>26.505779</td>
          <td>0.126578</td>
          <td>26.285077</td>
          <td>0.168748</td>
          <td>25.895111</td>
          <td>0.224878</td>
          <td>24.929286</td>
          <td>0.216573</td>
          <td>0.056076</td>
          <td>0.038854</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.100532</td>
          <td>0.514289</td>
          <td>28.296916</td>
          <td>0.535795</td>
          <td>26.034743</td>
          <td>0.135231</td>
          <td>25.110966</td>
          <td>0.114464</td>
          <td>24.287535</td>
          <td>0.124524</td>
          <td>0.049368</td>
          <td>0.034505</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.825276</td>
          <td>0.846040</td>
          <td>27.491269</td>
          <td>0.287884</td>
          <td>26.144510</td>
          <td>0.148502</td>
          <td>25.665647</td>
          <td>0.184206</td>
          <td>24.849685</td>
          <td>0.201134</td>
          <td>0.041840</td>
          <td>0.041543</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.884396</td>
          <td>0.515302</td>
          <td>26.037468</td>
          <td>0.095603</td>
          <td>26.033915</td>
          <td>0.084285</td>
          <td>25.677143</td>
          <td>0.100371</td>
          <td>25.527499</td>
          <td>0.165972</td>
          <td>25.602880</td>
          <td>0.375361</td>
          <td>0.067177</td>
          <td>0.034010</td>
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
          <td>26.519284</td>
          <td>0.184068</td>
          <td>25.437881</td>
          <td>0.065305</td>
          <td>25.042038</td>
          <td>0.075912</td>
          <td>24.936828</td>
          <td>0.130027</td>
          <td>25.148542</td>
          <td>0.336410</td>
          <td>0.216483</td>
          <td>0.138152</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.946048</td>
          <td>0.537400</td>
          <td>26.635891</td>
          <td>0.159863</td>
          <td>25.975821</td>
          <td>0.079676</td>
          <td>25.177749</td>
          <td>0.064276</td>
          <td>24.817150</td>
          <td>0.089176</td>
          <td>24.176016</td>
          <td>0.113907</td>
          <td>0.060186</td>
          <td>0.034984</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.925988</td>
          <td>1.017829</td>
          <td>26.543590</td>
          <td>0.145028</td>
          <td>26.344045</td>
          <td>0.107758</td>
          <td>26.530976</td>
          <td>0.203502</td>
          <td>25.605186</td>
          <td>0.172831</td>
          <td>26.721346</td>
          <td>0.817789</td>
          <td>0.033168</td>
          <td>0.020170</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.149348</td>
          <td>0.307393</td>
          <td>26.203147</td>
          <td>0.118041</td>
          <td>26.128225</td>
          <td>0.098705</td>
          <td>25.707299</td>
          <td>0.111376</td>
          <td>25.351651</td>
          <td>0.153603</td>
          <td>25.154690</td>
          <td>0.281916</td>
          <td>0.123667</td>
          <td>0.062581</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.913697</td>
          <td>1.022485</td>
          <td>26.618043</td>
          <td>0.157903</td>
          <td>26.578293</td>
          <td>0.135374</td>
          <td>26.229114</td>
          <td>0.161622</td>
          <td>26.024565</td>
          <td>0.251323</td>
          <td>26.427442</td>
          <td>0.685895</td>
          <td>0.062758</td>
          <td>0.038068</td>
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
