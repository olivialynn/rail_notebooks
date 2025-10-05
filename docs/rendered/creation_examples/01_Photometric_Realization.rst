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

    <pzflow.flow.Flow at 0x7f58834a8c10>



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
    0      23.994413  0.070412  0.043226  
    1      25.391064  0.085761  0.081052  
    2      24.304707  0.263503  0.205795  
    3      25.291103  0.104970  0.062281  
    4      25.096743  0.008421  0.007887  
    ...          ...       ...       ...  
    99995  24.737946  0.137396  0.083114  
    99996  24.224169  0.109307  0.080257  
    99997  25.613836  0.033735  0.020578  
    99998  25.274899  0.148957  0.083126  
    99999  25.699642  0.203309  0.200272  
    
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
          <td>26.765571</td>
          <td>0.461754</td>
          <td>26.584434</td>
          <td>0.148921</td>
          <td>26.222182</td>
          <td>0.095880</td>
          <td>25.315402</td>
          <td>0.070221</td>
          <td>24.616263</td>
          <td>0.072359</td>
          <td>23.933876</td>
          <td>0.089177</td>
          <td>0.070412</td>
          <td>0.043226</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.480306</td>
          <td>0.313689</td>
          <td>26.548659</td>
          <td>0.127473</td>
          <td>26.037634</td>
          <td>0.132256</td>
          <td>26.873506</td>
          <td>0.474773</td>
          <td>26.177947</td>
          <td>0.559340</td>
          <td>0.085761</td>
          <td>0.081052</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.940363</td>
          <td>0.525462</td>
          <td>27.914934</td>
          <td>0.440019</td>
          <td>28.723859</td>
          <td>0.709295</td>
          <td>26.022961</td>
          <td>0.130587</td>
          <td>24.992984</td>
          <td>0.100824</td>
          <td>24.364364</td>
          <td>0.129867</td>
          <td>0.263503</td>
          <td>0.205795</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.382772</td>
          <td>1.312073</td>
          <td>27.841399</td>
          <td>0.416083</td>
          <td>27.558043</td>
          <td>0.297453</td>
          <td>26.645070</td>
          <td>0.221607</td>
          <td>25.551848</td>
          <td>0.163547</td>
          <td>25.762468</td>
          <td>0.410673</td>
          <td>0.104970</td>
          <td>0.062281</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.248065</td>
          <td>0.309095</td>
          <td>26.066014</td>
          <td>0.094946</td>
          <td>25.956039</td>
          <td>0.075844</td>
          <td>25.713633</td>
          <td>0.099741</td>
          <td>25.468100</td>
          <td>0.152240</td>
          <td>25.079755</td>
          <td>0.238149</td>
          <td>0.008421</td>
          <td>0.007887</td>
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
          <td>26.721287</td>
          <td>0.446637</td>
          <td>26.641073</td>
          <td>0.156325</td>
          <td>25.440489</td>
          <td>0.048017</td>
          <td>25.140153</td>
          <td>0.060119</td>
          <td>24.813747</td>
          <td>0.086138</td>
          <td>24.928790</td>
          <td>0.210063</td>
          <td>0.137396</td>
          <td>0.083114</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.591632</td>
          <td>0.822750</td>
          <td>26.793509</td>
          <td>0.177991</td>
          <td>26.186911</td>
          <td>0.092956</td>
          <td>25.161177</td>
          <td>0.061251</td>
          <td>24.706911</td>
          <td>0.078394</td>
          <td>24.123280</td>
          <td>0.105292</td>
          <td>0.109307</td>
          <td>0.080257</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.961062</td>
          <td>0.244886</td>
          <td>26.639535</td>
          <td>0.156119</td>
          <td>26.527955</td>
          <td>0.125205</td>
          <td>26.416978</td>
          <td>0.182994</td>
          <td>25.861066</td>
          <td>0.212363</td>
          <td>25.595037</td>
          <td>0.360701</td>
          <td>0.033735</td>
          <td>0.020578</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.044674</td>
          <td>0.262247</td>
          <td>26.209376</td>
          <td>0.107631</td>
          <td>26.015998</td>
          <td>0.079968</td>
          <td>25.842911</td>
          <td>0.111676</td>
          <td>26.199485</td>
          <td>0.280625</td>
          <td>25.172409</td>
          <td>0.257013</td>
          <td>0.148957</td>
          <td>0.083126</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.845636</td>
          <td>0.490128</td>
          <td>27.126550</td>
          <td>0.235253</td>
          <td>26.534316</td>
          <td>0.125898</td>
          <td>26.443640</td>
          <td>0.187166</td>
          <td>25.890609</td>
          <td>0.217663</td>
          <td>25.833654</td>
          <td>0.433587</td>
          <td>0.203309</td>
          <td>0.200272</td>
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
          <td>26.392713</td>
          <td>0.387755</td>
          <td>26.845277</td>
          <td>0.215283</td>
          <td>25.978250</td>
          <td>0.092017</td>
          <td>25.159593</td>
          <td>0.073378</td>
          <td>24.723464</td>
          <td>0.094605</td>
          <td>24.286075</td>
          <td>0.144679</td>
          <td>0.070412</td>
          <td>0.043226</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.067133</td>
          <td>0.261147</td>
          <td>26.632849</td>
          <td>0.164205</td>
          <td>25.979784</td>
          <td>0.151942</td>
          <td>26.384785</td>
          <td>0.384007</td>
          <td>25.123403</td>
          <td>0.294599</td>
          <td>0.085761</td>
          <td>0.081052</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.077480</td>
          <td>0.189084</td>
          <td>24.979399</td>
          <td>0.136861</td>
          <td>24.185827</td>
          <td>0.153886</td>
          <td>0.263503</td>
          <td>0.205795</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>33.096047</td>
          <td>4.630265</td>
          <td>27.213893</td>
          <td>0.267186</td>
          <td>26.249780</td>
          <td>0.191434</td>
          <td>25.207647</td>
          <td>0.145998</td>
          <td>25.021198</td>
          <td>0.271562</td>
          <td>0.104970</td>
          <td>0.062281</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.397192</td>
          <td>0.386177</td>
          <td>26.159966</td>
          <td>0.118832</td>
          <td>25.823723</td>
          <td>0.079411</td>
          <td>25.937791</td>
          <td>0.143196</td>
          <td>25.158544</td>
          <td>0.136650</td>
          <td>25.119352</td>
          <td>0.287353</td>
          <td>0.008421</td>
          <td>0.007887</td>
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
          <td>26.463156</td>
          <td>0.160089</td>
          <td>25.405473</td>
          <td>0.057188</td>
          <td>25.049954</td>
          <td>0.068727</td>
          <td>24.953564</td>
          <td>0.119194</td>
          <td>24.456246</td>
          <td>0.172465</td>
          <td>0.137396</td>
          <td>0.083114</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.079693</td>
          <td>0.265426</td>
          <td>25.999170</td>
          <td>0.095503</td>
          <td>25.274696</td>
          <td>0.082829</td>
          <td>24.642111</td>
          <td>0.089755</td>
          <td>24.370947</td>
          <td>0.158562</td>
          <td>0.109307</td>
          <td>0.080257</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.667485</td>
          <td>0.474946</td>
          <td>26.890139</td>
          <td>0.221796</td>
          <td>26.504555</td>
          <td>0.144201</td>
          <td>26.586670</td>
          <td>0.248059</td>
          <td>26.396596</td>
          <td>0.380596</td>
          <td>25.351698</td>
          <td>0.346703</td>
          <td>0.033735</td>
          <td>0.020578</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.828823</td>
          <td>0.549944</td>
          <td>26.314118</td>
          <td>0.141513</td>
          <td>25.825005</td>
          <td>0.083296</td>
          <td>25.677436</td>
          <td>0.119849</td>
          <td>26.180922</td>
          <td>0.334206</td>
          <td>25.096683</td>
          <td>0.294747</td>
          <td>0.148957</td>
          <td>0.083126</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.707067</td>
          <td>0.528738</td>
          <td>26.645664</td>
          <td>0.200532</td>
          <td>27.132347</td>
          <td>0.273598</td>
          <td>26.073145</td>
          <td>0.181580</td>
          <td>25.930121</td>
          <td>0.292852</td>
          <td>26.362850</td>
          <td>0.797491</td>
          <td>0.203309</td>
          <td>0.200272</td>
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
          <td>27.537691</td>
          <td>0.812794</td>
          <td>26.868551</td>
          <td>0.196743</td>
          <td>26.101492</td>
          <td>0.090108</td>
          <td>25.155319</td>
          <td>0.063835</td>
          <td>24.577851</td>
          <td>0.073104</td>
          <td>24.060313</td>
          <td>0.104276</td>
          <td>0.070412</td>
          <td>0.043226</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.006495</td>
          <td>0.502452</td>
          <td>26.722032</td>
          <td>0.161045</td>
          <td>26.340420</td>
          <td>0.187093</td>
          <td>26.004056</td>
          <td>0.259234</td>
          <td>24.744341</td>
          <td>0.195992</td>
          <td>0.085761</td>
          <td>0.081052</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.635196</td>
          <td>0.505416</td>
          <td>29.647360</td>
          <td>1.621878</td>
          <td>25.966724</td>
          <td>0.195367</td>
          <td>25.113593</td>
          <td>0.174020</td>
          <td>24.339667</td>
          <td>0.198958</td>
          <td>0.263503</td>
          <td>0.205795</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.759270</td>
          <td>0.846740</td>
          <td>27.341339</td>
          <td>0.271195</td>
          <td>26.537897</td>
          <td>0.221610</td>
          <td>25.676923</td>
          <td>0.198318</td>
          <td>25.609207</td>
          <td>0.396066</td>
          <td>0.104970</td>
          <td>0.062281</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.089537</td>
          <td>0.272163</td>
          <td>26.150439</td>
          <td>0.102311</td>
          <td>26.036766</td>
          <td>0.081521</td>
          <td>25.931335</td>
          <td>0.120726</td>
          <td>25.438780</td>
          <td>0.148589</td>
          <td>24.881116</td>
          <td>0.202020</td>
          <td>0.008421</td>
          <td>0.007887</td>
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
          <td>27.995936</td>
          <td>1.132368</td>
          <td>26.631385</td>
          <td>0.175689</td>
          <td>25.513530</td>
          <td>0.059373</td>
          <td>25.067475</td>
          <td>0.065714</td>
          <td>24.929567</td>
          <td>0.110249</td>
          <td>24.772547</td>
          <td>0.212696</td>
          <td>0.137396</td>
          <td>0.083114</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.053007</td>
          <td>1.895068</td>
          <td>26.597216</td>
          <td>0.165457</td>
          <td>25.974403</td>
          <td>0.086106</td>
          <td>25.177654</td>
          <td>0.069811</td>
          <td>24.713912</td>
          <td>0.088095</td>
          <td>24.321553</td>
          <td>0.139994</td>
          <td>0.109307</td>
          <td>0.080257</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.909750</td>
          <td>0.516957</td>
          <td>26.575115</td>
          <td>0.149054</td>
          <td>26.374447</td>
          <td>0.110696</td>
          <td>26.373576</td>
          <td>0.178268</td>
          <td>25.943975</td>
          <td>0.229799</td>
          <td>25.627158</td>
          <td>0.373457</td>
          <td>0.033735</td>
          <td>0.020578</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.009069</td>
          <td>0.283915</td>
          <td>26.250859</td>
          <td>0.128413</td>
          <td>25.981178</td>
          <td>0.091043</td>
          <td>25.895268</td>
          <td>0.137742</td>
          <td>25.623150</td>
          <td>0.202592</td>
          <td>25.308522</td>
          <td>0.333672</td>
          <td>0.148957</td>
          <td>0.083126</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.481423</td>
          <td>0.188234</td>
          <td>26.652271</td>
          <td>0.199106</td>
          <td>26.122349</td>
          <td>0.205586</td>
          <td>25.907151</td>
          <td>0.310487</td>
          <td>27.806072</td>
          <td>1.847347</td>
          <td>0.203309</td>
          <td>0.200272</td>
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
