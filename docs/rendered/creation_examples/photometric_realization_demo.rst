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

    <pzflow.flow.Flow at 0x7fcb8ffcba30>



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
    0      1.398945  27.667538  26.723339  26.032640  25.178589  24.695959   
    1      2.285624  28.786999  27.476589  26.640173  26.259747  25.865671   
    2      1.495130  30.011343  29.789326  28.200378  26.014816  25.030161   
    3      0.842595  29.306242  28.721798  27.353014  26.256908  25.529823   
    4      1.588960  26.273870  26.115385  25.950439  25.687403  25.466604   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270809  26.371513  25.436861  25.077417  24.852785   
    99996  1.481047  27.478111  26.735254  26.042774  25.204937  24.825092   
    99997  2.023549  26.990149  26.714739  26.377953  26.250345  25.917372   
    99998  1.548203  26.367432  26.206882  26.087980  25.876928  25.715893   
    99999  1.739491  26.881981  26.773064  26.553120  26.319618  25.955980   
    
                   y     major     minor  
    0      23.994417  0.003319  0.002869  
    1      25.391062  0.008733  0.007945  
    2      24.304695  0.103938  0.052162  
    3      25.291105  0.147522  0.143359  
    4      25.096741  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737953  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613838  0.073146  0.047825  
    99998  25.274897  0.100551  0.094662  
    99999  25.699638  0.059611  0.049181  
    
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
          <td>1.398945</td>
          <td>27.886094</td>
          <td>0.988964</td>
          <td>26.861198</td>
          <td>0.188476</td>
          <td>26.103810</td>
          <td>0.086404</td>
          <td>25.141649</td>
          <td>0.060199</td>
          <td>24.717043</td>
          <td>0.079098</td>
          <td>24.030337</td>
          <td>0.097062</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.893217</td>
          <td>0.507634</td>
          <td>28.022938</td>
          <td>0.477188</td>
          <td>26.823648</td>
          <td>0.161507</td>
          <td>26.542938</td>
          <td>0.203482</td>
          <td>25.845666</td>
          <td>0.209647</td>
          <td>25.243886</td>
          <td>0.272459</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>29.644208</td>
          <td>2.323968</td>
          <td>28.578458</td>
          <td>0.708500</td>
          <td>27.753810</td>
          <td>0.347644</td>
          <td>26.057152</td>
          <td>0.134506</td>
          <td>25.137951</td>
          <td>0.114435</td>
          <td>24.607044</td>
          <td>0.160017</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.180145</td>
          <td>0.623713</td>
          <td>28.789223</td>
          <td>0.814592</td>
          <td>27.880227</td>
          <td>0.383747</td>
          <td>26.061216</td>
          <td>0.134979</td>
          <td>25.446479</td>
          <td>0.149442</td>
          <td>25.367458</td>
          <td>0.301097</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.877184</td>
          <td>0.501681</td>
          <td>26.008598</td>
          <td>0.090282</td>
          <td>25.924839</td>
          <td>0.073781</td>
          <td>25.589816</td>
          <td>0.089466</td>
          <td>25.345882</td>
          <td>0.137047</td>
          <td>25.617935</td>
          <td>0.367218</td>
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
          <td>27.117679</td>
          <td>0.596881</td>
          <td>26.199730</td>
          <td>0.106729</td>
          <td>25.401179</td>
          <td>0.046370</td>
          <td>25.125785</td>
          <td>0.059358</td>
          <td>24.853681</td>
          <td>0.089219</td>
          <td>24.709513</td>
          <td>0.174614</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.004143</td>
          <td>0.212503</td>
          <td>25.946478</td>
          <td>0.075206</td>
          <td>25.227283</td>
          <td>0.064949</td>
          <td>24.791671</td>
          <td>0.084479</td>
          <td>24.240795</td>
          <td>0.116659</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.354812</td>
          <td>0.336477</td>
          <td>26.766715</td>
          <td>0.173992</td>
          <td>26.339557</td>
          <td>0.106261</td>
          <td>26.172853</td>
          <td>0.148603</td>
          <td>25.779571</td>
          <td>0.198344</td>
          <td>25.692750</td>
          <td>0.389206</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.082054</td>
          <td>0.270356</td>
          <td>26.233385</td>
          <td>0.109909</td>
          <td>25.964099</td>
          <td>0.076386</td>
          <td>25.713626</td>
          <td>0.099740</td>
          <td>25.690411</td>
          <td>0.183979</td>
          <td>25.116373</td>
          <td>0.245451</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.712421</td>
          <td>0.443659</td>
          <td>26.802691</td>
          <td>0.179381</td>
          <td>26.424083</td>
          <td>0.114395</td>
          <td>26.152521</td>
          <td>0.146030</td>
          <td>26.234386</td>
          <td>0.288666</td>
          <td>26.033017</td>
          <td>0.503316</td>
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
          <td>1.398945</td>
          <td>26.780476</td>
          <td>0.515426</td>
          <td>26.867827</td>
          <td>0.217229</td>
          <td>26.067684</td>
          <td>0.098397</td>
          <td>25.298150</td>
          <td>0.081943</td>
          <td>24.566369</td>
          <td>0.081456</td>
          <td>23.765979</td>
          <td>0.090947</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.969361</td>
          <td>1.123676</td>
          <td>27.665269</td>
          <td>0.411852</td>
          <td>26.921662</td>
          <td>0.205058</td>
          <td>26.155080</td>
          <td>0.172449</td>
          <td>25.935153</td>
          <td>0.262826</td>
          <td>25.101785</td>
          <td>0.283300</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>29.133577</td>
          <td>2.014865</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.027028</td>
          <td>0.502342</td>
          <td>25.939823</td>
          <td>0.146694</td>
          <td>25.090692</td>
          <td>0.131702</td>
          <td>24.408038</td>
          <td>0.162372</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.275723</td>
          <td>0.758904</td>
          <td>32.424698</td>
          <td>4.014782</td>
          <td>27.337501</td>
          <td>0.307067</td>
          <td>26.515252</td>
          <td>0.248960</td>
          <td>25.270865</td>
          <td>0.160655</td>
          <td>24.798426</td>
          <td>0.235606</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.208593</td>
          <td>0.333212</td>
          <td>26.270316</td>
          <td>0.130768</td>
          <td>25.866762</td>
          <td>0.082494</td>
          <td>25.658088</td>
          <td>0.112395</td>
          <td>26.137002</td>
          <td>0.309485</td>
          <td>24.941992</td>
          <td>0.248692</td>
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
          <td>27.494358</td>
          <td>0.851963</td>
          <td>26.225226</td>
          <td>0.128103</td>
          <td>25.381403</td>
          <td>0.054821</td>
          <td>24.998147</td>
          <td>0.064246</td>
          <td>24.972590</td>
          <td>0.118731</td>
          <td>24.887011</td>
          <td>0.242462</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.409788</td>
          <td>0.390994</td>
          <td>26.843149</td>
          <td>0.213566</td>
          <td>26.132579</td>
          <td>0.104583</td>
          <td>25.325838</td>
          <td>0.084332</td>
          <td>24.860211</td>
          <td>0.105870</td>
          <td>24.148449</td>
          <td>0.127535</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.598912</td>
          <td>0.175191</td>
          <td>26.281529</td>
          <td>0.120105</td>
          <td>26.154936</td>
          <td>0.174612</td>
          <td>25.480039</td>
          <td>0.182097</td>
          <td>26.156002</td>
          <td>0.636659</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.074671</td>
          <td>0.306047</td>
          <td>26.206791</td>
          <td>0.127255</td>
          <td>26.021223</td>
          <td>0.097482</td>
          <td>25.894693</td>
          <td>0.142418</td>
          <td>25.439098</td>
          <td>0.179077</td>
          <td>25.347377</td>
          <td>0.354789</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.253684</td>
          <td>0.347580</td>
          <td>26.592986</td>
          <td>0.173894</td>
          <td>26.753637</td>
          <td>0.179671</td>
          <td>26.350704</td>
          <td>0.205406</td>
          <td>25.485477</td>
          <td>0.182454</td>
          <td>25.317785</td>
          <td>0.339866</td>
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
          <td>1.398945</td>
          <td>27.284703</td>
          <td>0.670655</td>
          <td>26.971017</td>
          <td>0.206721</td>
          <td>26.068628</td>
          <td>0.083778</td>
          <td>25.055980</td>
          <td>0.055799</td>
          <td>24.671639</td>
          <td>0.075999</td>
          <td>23.979543</td>
          <td>0.092842</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.795887</td>
          <td>0.936395</td>
          <td>27.036347</td>
          <td>0.218457</td>
          <td>26.803353</td>
          <td>0.158876</td>
          <td>26.008300</td>
          <td>0.129067</td>
          <td>25.611330</td>
          <td>0.172204</td>
          <td>24.893618</td>
          <td>0.204159</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.972984</td>
          <td>1.810558</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.645654</td>
          <td>0.716443</td>
          <td>25.878499</td>
          <td>0.125368</td>
          <td>25.190657</td>
          <td>0.129881</td>
          <td>24.320228</td>
          <td>0.135878</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.321877</td>
          <td>2.057867</td>
          <td>26.916017</td>
          <td>0.216771</td>
          <td>26.364522</td>
          <td>0.219004</td>
          <td>25.620462</td>
          <td>0.215098</td>
          <td>25.125642</td>
          <td>0.306535</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.888490</td>
          <td>0.230874</td>
          <td>26.084856</td>
          <td>0.096646</td>
          <td>25.983480</td>
          <td>0.077816</td>
          <td>25.836972</td>
          <td>0.111263</td>
          <td>25.266681</td>
          <td>0.128155</td>
          <td>25.131791</td>
          <td>0.248929</td>
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
          <td>26.387799</td>
          <td>0.134532</td>
          <td>25.489386</td>
          <td>0.054311</td>
          <td>25.055014</td>
          <td>0.060590</td>
          <td>24.910628</td>
          <td>0.101463</td>
          <td>24.906253</td>
          <td>0.222743</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.627157</td>
          <td>0.848815</td>
          <td>26.684174</td>
          <td>0.164460</td>
          <td>25.963307</td>
          <td>0.077609</td>
          <td>25.216391</td>
          <td>0.065458</td>
          <td>24.920717</td>
          <td>0.096203</td>
          <td>24.229712</td>
          <td>0.117515</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.683874</td>
          <td>0.894501</td>
          <td>26.966592</td>
          <td>0.214475</td>
          <td>26.351352</td>
          <td>0.112710</td>
          <td>25.731680</td>
          <td>0.106637</td>
          <td>25.412478</td>
          <td>0.152240</td>
          <td>26.363691</td>
          <td>0.663962</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.158544</td>
          <td>0.310102</td>
          <td>26.486110</td>
          <td>0.151054</td>
          <td>26.073995</td>
          <td>0.094414</td>
          <td>25.909808</td>
          <td>0.133235</td>
          <td>25.798408</td>
          <td>0.224679</td>
          <td>25.209165</td>
          <td>0.295506</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.203209</td>
          <td>0.647657</td>
          <td>26.744732</td>
          <td>0.176464</td>
          <td>26.595397</td>
          <td>0.137912</td>
          <td>26.698584</td>
          <td>0.240737</td>
          <td>26.022909</td>
          <td>0.251893</td>
          <td>25.696398</td>
          <td>0.404374</td>
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
