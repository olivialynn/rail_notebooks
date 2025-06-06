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

    <pzflow.flow.Flow at 0x7f5f4e45c640>



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
    0      23.994413  0.000806  0.000547  
    1      25.391064  0.036225  0.029633  
    2      24.304707  0.020551  0.015863  
    3      25.291103  0.008141  0.004148  
    4      25.096743  0.036405  0.034505  
    ...          ...       ...       ...  
    99995  24.737946  0.099344  0.079091  
    99996  24.224169  0.050578  0.031394  
    99997  25.613836  0.007647  0.006659  
    99998  25.274899  0.148340  0.098935  
    99999  25.699642  0.097119  0.072287  
    
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
          <td>27.505725</td>
          <td>0.777982</td>
          <td>26.537979</td>
          <td>0.143095</td>
          <td>26.006990</td>
          <td>0.079335</td>
          <td>25.171108</td>
          <td>0.061793</td>
          <td>24.592917</td>
          <td>0.070880</td>
          <td>23.930922</td>
          <td>0.088945</td>
          <td>0.000806</td>
          <td>0.000547</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.518171</td>
          <td>0.323305</td>
          <td>26.591300</td>
          <td>0.132266</td>
          <td>26.540068</td>
          <td>0.202993</td>
          <td>25.931408</td>
          <td>0.225180</td>
          <td>25.181671</td>
          <td>0.258969</td>
          <td>0.036225</td>
          <td>0.029633</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.235955</td>
          <td>0.557771</td>
          <td>29.540577</td>
          <td>1.176636</td>
          <td>25.859730</td>
          <td>0.113325</td>
          <td>25.200238</td>
          <td>0.120807</td>
          <td>24.353619</td>
          <td>0.128664</td>
          <td>0.020551</td>
          <td>0.015863</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.510122</td>
          <td>0.780233</td>
          <td>28.860518</td>
          <td>0.852790</td>
          <td>27.370984</td>
          <td>0.255510</td>
          <td>26.281739</td>
          <td>0.163124</td>
          <td>25.745068</td>
          <td>0.192668</td>
          <td>25.235105</td>
          <td>0.270519</td>
          <td>0.008141</td>
          <td>0.004148</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.851328</td>
          <td>0.223667</td>
          <td>25.963736</td>
          <td>0.086792</td>
          <td>26.038002</td>
          <td>0.081536</td>
          <td>25.822955</td>
          <td>0.109748</td>
          <td>25.561387</td>
          <td>0.164883</td>
          <td>25.405257</td>
          <td>0.310366</td>
          <td>0.036405</td>
          <td>0.034505</td>
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
          <td>27.079419</td>
          <td>0.580880</td>
          <td>26.107390</td>
          <td>0.098452</td>
          <td>25.493136</td>
          <td>0.050315</td>
          <td>25.049434</td>
          <td>0.055468</td>
          <td>25.014708</td>
          <td>0.102760</td>
          <td>24.504476</td>
          <td>0.146551</td>
          <td>0.099344</td>
          <td>0.079091</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.679697</td>
          <td>0.161570</td>
          <td>26.171763</td>
          <td>0.091727</td>
          <td>25.154897</td>
          <td>0.060911</td>
          <td>24.783774</td>
          <td>0.083893</td>
          <td>24.072815</td>
          <td>0.100744</td>
          <td>0.050578</td>
          <td>0.031394</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.779359</td>
          <td>0.466545</td>
          <td>26.791849</td>
          <td>0.177741</td>
          <td>26.365151</td>
          <td>0.108664</td>
          <td>26.086278</td>
          <td>0.137931</td>
          <td>25.447203</td>
          <td>0.149535</td>
          <td>26.472587</td>
          <td>0.687680</td>
          <td>0.007647</td>
          <td>0.006659</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.323942</td>
          <td>0.328352</td>
          <td>26.203039</td>
          <td>0.107038</td>
          <td>26.006467</td>
          <td>0.079298</td>
          <td>25.948553</td>
          <td>0.122430</td>
          <td>25.518474</td>
          <td>0.158950</td>
          <td>25.243979</td>
          <td>0.272480</td>
          <td>0.148340</td>
          <td>0.098935</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.829500</td>
          <td>0.484301</td>
          <td>26.788345</td>
          <td>0.177214</td>
          <td>26.556593</td>
          <td>0.128352</td>
          <td>26.481635</td>
          <td>0.193262</td>
          <td>25.961896</td>
          <td>0.230948</td>
          <td>25.414420</td>
          <td>0.312649</td>
          <td>0.097119</td>
          <td>0.072287</td>
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
          <td>27.846038</td>
          <td>1.045691</td>
          <td>26.900090</td>
          <td>0.223133</td>
          <td>26.085267</td>
          <td>0.099922</td>
          <td>25.133373</td>
          <td>0.070843</td>
          <td>24.638061</td>
          <td>0.086763</td>
          <td>23.860648</td>
          <td>0.098821</td>
          <td>0.000806</td>
          <td>0.000547</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.743496</td>
          <td>0.502794</td>
          <td>27.256461</td>
          <td>0.299573</td>
          <td>26.587886</td>
          <td>0.155054</td>
          <td>26.371487</td>
          <td>0.207707</td>
          <td>26.640538</td>
          <td>0.458970</td>
          <td>24.573124</td>
          <td>0.183402</td>
          <td>0.036225</td>
          <td>0.029633</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.830247</td>
          <td>0.534856</td>
          <td>29.185117</td>
          <td>1.145006</td>
          <td>27.894463</td>
          <td>0.446942</td>
          <td>25.949132</td>
          <td>0.144735</td>
          <td>25.074488</td>
          <td>0.127186</td>
          <td>24.495291</td>
          <td>0.171249</td>
          <td>0.020551</td>
          <td>0.015863</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.982357</td>
          <td>0.522161</td>
          <td>27.525540</td>
          <td>0.335667</td>
          <td>26.437728</td>
          <td>0.218749</td>
          <td>25.359259</td>
          <td>0.162329</td>
          <td>25.388917</td>
          <td>0.356175</td>
          <td>0.008141</td>
          <td>0.004148</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.183395</td>
          <td>0.327502</td>
          <td>26.112196</td>
          <td>0.114412</td>
          <td>26.060464</td>
          <td>0.098191</td>
          <td>25.580289</td>
          <td>0.105440</td>
          <td>25.446133</td>
          <td>0.175487</td>
          <td>25.957978</td>
          <td>0.549279</td>
          <td>0.036405</td>
          <td>0.034505</td>
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
          <td>27.287367</td>
          <td>0.746858</td>
          <td>26.285432</td>
          <td>0.135624</td>
          <td>25.422221</td>
          <td>0.057169</td>
          <td>25.081766</td>
          <td>0.069593</td>
          <td>24.720624</td>
          <td>0.095816</td>
          <td>24.938910</td>
          <td>0.254411</td>
          <td>0.099344</td>
          <td>0.079091</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.673475</td>
          <td>0.185498</td>
          <td>26.030808</td>
          <td>0.095836</td>
          <td>25.125427</td>
          <td>0.070789</td>
          <td>24.947354</td>
          <td>0.114436</td>
          <td>24.168487</td>
          <td>0.130002</td>
          <td>0.050578</td>
          <td>0.031394</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.515816</td>
          <td>0.161438</td>
          <td>26.197102</td>
          <td>0.110203</td>
          <td>25.982478</td>
          <td>0.148796</td>
          <td>26.012371</td>
          <td>0.279862</td>
          <td>25.280468</td>
          <td>0.326953</td>
          <td>0.007647</td>
          <td>0.006659</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.100235</td>
          <td>0.316705</td>
          <td>26.083979</td>
          <td>0.116426</td>
          <td>26.161761</td>
          <td>0.112389</td>
          <td>25.814981</td>
          <td>0.135615</td>
          <td>25.914313</td>
          <td>0.270842</td>
          <td>25.399958</td>
          <td>0.376305</td>
          <td>0.148340</td>
          <td>0.098935</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.102410</td>
          <td>0.311351</td>
          <td>26.752039</td>
          <td>0.201348</td>
          <td>26.652053</td>
          <td>0.167100</td>
          <td>26.154117</td>
          <td>0.176505</td>
          <td>26.089813</td>
          <td>0.304571</td>
          <td>25.368023</td>
          <td>0.358231</td>
          <td>0.097119</td>
          <td>0.072287</td>
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
          <td>26.978038</td>
          <td>0.540058</td>
          <td>26.759325</td>
          <td>0.172904</td>
          <td>26.013516</td>
          <td>0.079793</td>
          <td>25.154561</td>
          <td>0.060893</td>
          <td>24.733253</td>
          <td>0.080238</td>
          <td>23.858402</td>
          <td>0.083444</td>
          <td>0.000806</td>
          <td>0.000547</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.941676</td>
          <td>0.530425</td>
          <td>27.184862</td>
          <td>0.249799</td>
          <td>26.555319</td>
          <td>0.130076</td>
          <td>26.644525</td>
          <td>0.224753</td>
          <td>26.211264</td>
          <td>0.287160</td>
          <td>24.782463</td>
          <td>0.188485</td>
          <td>0.036225</td>
          <td>0.029633</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.363089</td>
          <td>2.083028</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.381132</td>
          <td>0.560413</td>
          <td>26.006152</td>
          <td>0.129310</td>
          <td>24.992023</td>
          <td>0.101197</td>
          <td>24.291971</td>
          <td>0.122539</td>
          <td>0.020551</td>
          <td>0.015863</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.624785</td>
          <td>0.415225</td>
          <td>28.137881</td>
          <td>0.519662</td>
          <td>27.391756</td>
          <td>0.260031</td>
          <td>26.320412</td>
          <td>0.168689</td>
          <td>25.390201</td>
          <td>0.142463</td>
          <td>25.337012</td>
          <td>0.293967</td>
          <td>0.008141</td>
          <td>0.004148</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.191955</td>
          <td>0.298671</td>
          <td>26.119709</td>
          <td>0.100979</td>
          <td>25.897512</td>
          <td>0.073249</td>
          <td>25.463279</td>
          <td>0.081458</td>
          <td>25.460191</td>
          <td>0.153708</td>
          <td>25.074268</td>
          <td>0.240992</td>
          <td>0.036405</td>
          <td>0.034505</td>
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
          <td>26.537675</td>
          <td>0.412788</td>
          <td>26.635922</td>
          <td>0.169252</td>
          <td>25.508272</td>
          <td>0.056333</td>
          <td>25.089630</td>
          <td>0.063777</td>
          <td>24.823339</td>
          <td>0.095841</td>
          <td>24.400788</td>
          <td>0.148088</td>
          <td>0.099344</td>
          <td>0.079091</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.959082</td>
          <td>1.044595</td>
          <td>27.315666</td>
          <td>0.279920</td>
          <td>26.023122</td>
          <td>0.082387</td>
          <td>25.236974</td>
          <td>0.067153</td>
          <td>24.790858</td>
          <td>0.086422</td>
          <td>24.014046</td>
          <td>0.098036</td>
          <td>0.050578</td>
          <td>0.031394</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.768232</td>
          <td>0.462867</td>
          <td>26.550830</td>
          <td>0.144770</td>
          <td>26.412957</td>
          <td>0.113370</td>
          <td>26.045338</td>
          <td>0.133236</td>
          <td>26.034176</td>
          <td>0.245319</td>
          <td>25.836229</td>
          <td>0.434707</td>
          <td>0.007647</td>
          <td>0.006659</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.637468</td>
          <td>0.467892</td>
          <td>26.169624</td>
          <td>0.121277</td>
          <td>26.142142</td>
          <td>0.106422</td>
          <td>25.781355</td>
          <td>0.126763</td>
          <td>25.578570</td>
          <td>0.197973</td>
          <td>25.498005</td>
          <td>0.392393</td>
          <td>0.148340</td>
          <td>0.098935</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.978493</td>
          <td>0.569366</td>
          <td>26.501008</td>
          <td>0.149749</td>
          <td>26.505206</td>
          <td>0.134139</td>
          <td>26.149894</td>
          <td>0.159697</td>
          <td>26.058137</td>
          <td>0.271833</td>
          <td>26.162969</td>
          <td>0.596849</td>
          <td>0.097119</td>
          <td>0.072287</td>
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
