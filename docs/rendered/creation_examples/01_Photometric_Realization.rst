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

    <pzflow.flow.Flow at 0x7fa65d3232e0>



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
    0      23.994413  0.092267  0.050012  
    1      25.391064  0.189328  0.152797  
    2      24.304707  0.035534  0.032951  
    3      25.291103  0.125525  0.097661  
    4      25.096743  0.083933  0.064580  
    ...          ...       ...       ...  
    99995  24.737946  0.090677  0.087549  
    99996  24.224169  0.054407  0.037481  
    99997  25.613836  0.159425  0.097128  
    99998  25.274899  0.016538  0.014078  
    99999  25.699642  0.064247  0.040400  
    
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
          <td>30.235068</td>
          <td>2.859835</td>
          <td>26.576203</td>
          <td>0.147873</td>
          <td>26.029817</td>
          <td>0.080949</td>
          <td>25.245531</td>
          <td>0.066007</td>
          <td>24.571834</td>
          <td>0.069570</td>
          <td>23.940166</td>
          <td>0.089672</td>
          <td>0.092267</td>
          <td>0.050012</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.151965</td>
          <td>2.782919</td>
          <td>27.934084</td>
          <td>0.446433</td>
          <td>26.684715</td>
          <td>0.143368</td>
          <td>26.253937</td>
          <td>0.159296</td>
          <td>25.947980</td>
          <td>0.228299</td>
          <td>26.188968</td>
          <td>0.563789</td>
          <td>0.189328</td>
          <td>0.152797</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.082708</td>
          <td>0.582242</td>
          <td>28.582645</td>
          <td>0.710508</td>
          <td>29.265594</td>
          <td>1.002701</td>
          <td>25.728887</td>
          <td>0.101082</td>
          <td>25.070041</td>
          <td>0.107853</td>
          <td>24.329074</td>
          <td>0.125957</td>
          <td>0.035534</td>
          <td>0.032951</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.586917</td>
          <td>1.459126</td>
          <td>27.355655</td>
          <td>0.283764</td>
          <td>27.418674</td>
          <td>0.265675</td>
          <td>26.360368</td>
          <td>0.174419</td>
          <td>25.353797</td>
          <td>0.137986</td>
          <td>25.846155</td>
          <td>0.437717</td>
          <td>0.125525</td>
          <td>0.097661</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.669300</td>
          <td>0.429404</td>
          <td>26.210098</td>
          <td>0.107699</td>
          <td>26.002363</td>
          <td>0.079011</td>
          <td>25.553370</td>
          <td>0.086642</td>
          <td>25.623940</td>
          <td>0.173900</td>
          <td>25.262310</td>
          <td>0.276571</td>
          <td>0.083933</td>
          <td>0.064580</td>
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
          <td>28.040894</td>
          <td>1.084093</td>
          <td>26.551819</td>
          <td>0.144808</td>
          <td>25.381693</td>
          <td>0.045575</td>
          <td>25.175040</td>
          <td>0.062009</td>
          <td>24.872507</td>
          <td>0.090709</td>
          <td>24.493278</td>
          <td>0.145147</td>
          <td>0.090677</td>
          <td>0.087549</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.519467</td>
          <td>0.785030</td>
          <td>26.849860</td>
          <td>0.186681</td>
          <td>26.071025</td>
          <td>0.083944</td>
          <td>25.163389</td>
          <td>0.061371</td>
          <td>24.724838</td>
          <td>0.079644</td>
          <td>24.251631</td>
          <td>0.117764</td>
          <td>0.054407</td>
          <td>0.037481</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.600198</td>
          <td>0.407344</td>
          <td>26.692222</td>
          <td>0.163305</td>
          <td>26.537109</td>
          <td>0.126203</td>
          <td>26.331007</td>
          <td>0.170119</td>
          <td>25.927421</td>
          <td>0.224435</td>
          <td>25.320779</td>
          <td>0.289984</td>
          <td>0.159425</td>
          <td>0.097128</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.922171</td>
          <td>0.237167</td>
          <td>26.154640</td>
          <td>0.102607</td>
          <td>26.019512</td>
          <td>0.080216</td>
          <td>25.865253</td>
          <td>0.113872</td>
          <td>25.573029</td>
          <td>0.166528</td>
          <td>25.190060</td>
          <td>0.260753</td>
          <td>0.016538</td>
          <td>0.014078</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.070532</td>
          <td>0.267833</td>
          <td>27.136006</td>
          <td>0.237099</td>
          <td>26.630901</td>
          <td>0.136870</td>
          <td>26.577259</td>
          <td>0.209417</td>
          <td>25.744887</td>
          <td>0.192638</td>
          <td>25.483879</td>
          <td>0.330436</td>
          <td>0.064247</td>
          <td>0.040400</td>
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
          <td>26.543253</td>
          <td>0.167906</td>
          <td>26.223428</td>
          <td>0.114810</td>
          <td>25.212430</td>
          <td>0.077434</td>
          <td>24.723287</td>
          <td>0.095240</td>
          <td>24.041275</td>
          <td>0.117884</td>
          <td>0.092267</td>
          <td>0.050012</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.907583</td>
          <td>0.598545</td>
          <td>27.237088</td>
          <td>0.317153</td>
          <td>26.581002</td>
          <td>0.167830</td>
          <td>26.322424</td>
          <td>0.217233</td>
          <td>26.267001</td>
          <td>0.372126</td>
          <td>25.322313</td>
          <td>0.367390</td>
          <td>0.189328</td>
          <td>0.152797</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.733636</td>
          <td>0.396305</td>
          <td>25.975864</td>
          <td>0.148525</td>
          <td>25.052552</td>
          <td>0.125141</td>
          <td>24.286643</td>
          <td>0.143664</td>
          <td>0.035534</td>
          <td>0.032951</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.712285</td>
          <td>1.684233</td>
          <td>28.761269</td>
          <td>0.911490</td>
          <td>27.219523</td>
          <td>0.272571</td>
          <td>26.398137</td>
          <td>0.220337</td>
          <td>25.349142</td>
          <td>0.167470</td>
          <td>25.419988</td>
          <td>0.378837</td>
          <td>0.125525</td>
          <td>0.097661</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.812746</td>
          <td>0.533986</td>
          <td>26.213581</td>
          <td>0.126558</td>
          <td>26.058154</td>
          <td>0.099417</td>
          <td>25.517772</td>
          <td>0.101328</td>
          <td>25.294198</td>
          <td>0.156348</td>
          <td>25.612514</td>
          <td>0.430490</td>
          <td>0.083933</td>
          <td>0.064580</td>
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
          <td>28.823674</td>
          <td>1.762451</td>
          <td>26.280338</td>
          <td>0.134987</td>
          <td>25.520561</td>
          <td>0.062355</td>
          <td>25.071304</td>
          <td>0.068927</td>
          <td>24.809175</td>
          <td>0.103507</td>
          <td>24.778591</td>
          <td>0.222787</td>
          <td>0.090677</td>
          <td>0.087549</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.672424</td>
          <td>0.185556</td>
          <td>26.168589</td>
          <td>0.108267</td>
          <td>25.192885</td>
          <td>0.075247</td>
          <td>24.937603</td>
          <td>0.113623</td>
          <td>24.301966</td>
          <td>0.146067</td>
          <td>0.054407</td>
          <td>0.037481</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.800080</td>
          <td>0.541666</td>
          <td>26.837823</td>
          <td>0.222192</td>
          <td>26.392913</td>
          <td>0.137932</td>
          <td>26.368246</td>
          <td>0.217921</td>
          <td>25.701866</td>
          <td>0.228384</td>
          <td>25.874025</td>
          <td>0.539689</td>
          <td>0.159425</td>
          <td>0.097128</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.654818</td>
          <td>0.469920</td>
          <td>26.281664</td>
          <td>0.132108</td>
          <td>26.047848</td>
          <td>0.096776</td>
          <td>25.907500</td>
          <td>0.139590</td>
          <td>26.365830</td>
          <td>0.370987</td>
          <td>24.881299</td>
          <td>0.236658</td>
          <td>0.016538</td>
          <td>0.014078</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.686926</td>
          <td>0.484055</td>
          <td>26.566447</td>
          <td>0.169963</td>
          <td>26.564525</td>
          <td>0.152871</td>
          <td>26.220569</td>
          <td>0.184025</td>
          <td>25.731244</td>
          <td>0.224146</td>
          <td>25.992345</td>
          <td>0.565620</td>
          <td>0.064247</td>
          <td>0.040400</td>
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
          <td>28.038039</td>
          <td>1.117394</td>
          <td>26.797897</td>
          <td>0.189242</td>
          <td>26.049693</td>
          <td>0.088221</td>
          <td>25.217373</td>
          <td>0.069194</td>
          <td>24.796062</td>
          <td>0.090807</td>
          <td>24.009189</td>
          <td>0.102247</td>
          <td>0.092267</td>
          <td>0.050012</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.223500</td>
          <td>0.760530</td>
          <td>26.992637</td>
          <td>0.268690</td>
          <td>26.401538</td>
          <td>0.149222</td>
          <td>26.459715</td>
          <td>0.252195</td>
          <td>26.371582</td>
          <td>0.416683</td>
          <td>25.357700</td>
          <td>0.390414</td>
          <td>0.189328</td>
          <td>0.152797</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.878796</td>
          <td>2.357003</td>
          <td>28.481399</td>
          <td>0.607356</td>
          <td>26.038314</td>
          <td>0.134497</td>
          <td>24.978078</td>
          <td>0.101085</td>
          <td>24.347944</td>
          <td>0.130109</td>
          <td>0.035534</td>
          <td>0.032951</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.267968</td>
          <td>0.268990</td>
          <td>26.308103</td>
          <td>0.193190</td>
          <td>25.371521</td>
          <td>0.161537</td>
          <td>25.084769</td>
          <td>0.275220</td>
          <td>0.125525</td>
          <td>0.097661</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.898827</td>
          <td>0.243802</td>
          <td>26.130139</td>
          <td>0.106794</td>
          <td>26.035532</td>
          <td>0.087320</td>
          <td>25.585324</td>
          <td>0.095937</td>
          <td>25.627966</td>
          <td>0.186741</td>
          <td>25.127082</td>
          <td>0.265067</td>
          <td>0.083933</td>
          <td>0.064580</td>
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
          <td>29.553721</td>
          <td>2.313751</td>
          <td>26.460636</td>
          <td>0.145578</td>
          <td>25.432042</td>
          <td>0.052599</td>
          <td>25.079195</td>
          <td>0.063131</td>
          <td>24.810214</td>
          <td>0.094658</td>
          <td>24.473895</td>
          <td>0.157522</td>
          <td>0.090677</td>
          <td>0.087549</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.366939</td>
          <td>1.316957</td>
          <td>26.813380</td>
          <td>0.185416</td>
          <td>25.991460</td>
          <td>0.080545</td>
          <td>25.317117</td>
          <td>0.072494</td>
          <td>24.694477</td>
          <td>0.079806</td>
          <td>24.198031</td>
          <td>0.115763</td>
          <td>0.054407</td>
          <td>0.037481</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.143709</td>
          <td>0.677787</td>
          <td>26.407874</td>
          <td>0.150562</td>
          <td>26.362353</td>
          <td>0.130419</td>
          <td>26.335656</td>
          <td>0.205859</td>
          <td>25.492477</td>
          <td>0.186242</td>
          <td>25.405930</td>
          <td>0.369287</td>
          <td>0.159425</td>
          <td>0.097128</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.138674</td>
          <td>0.606877</td>
          <td>26.505860</td>
          <td>0.139571</td>
          <td>26.102956</td>
          <td>0.086617</td>
          <td>25.897754</td>
          <td>0.117531</td>
          <td>25.738401</td>
          <td>0.192176</td>
          <td>25.285150</td>
          <td>0.282609</td>
          <td>0.016538</td>
          <td>0.014078</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.761817</td>
          <td>0.933510</td>
          <td>27.044119</td>
          <td>0.226581</td>
          <td>26.364160</td>
          <td>0.112677</td>
          <td>26.299288</td>
          <td>0.171990</td>
          <td>25.950260</td>
          <td>0.236918</td>
          <td>25.000521</td>
          <td>0.231332</td>
          <td>0.064247</td>
          <td>0.040400</td>
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
