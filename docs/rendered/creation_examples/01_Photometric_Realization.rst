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

    <pzflow.flow.Flow at 0x7fe7732cd270>



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
          <td>28.322204</td>
          <td>1.269963</td>
          <td>26.695589</td>
          <td>0.163775</td>
          <td>26.024380</td>
          <td>0.080562</td>
          <td>25.124368</td>
          <td>0.059283</td>
          <td>24.717071</td>
          <td>0.079100</td>
          <td>23.930713</td>
          <td>0.088929</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.171494</td>
          <td>1.918813</td>
          <td>27.583685</td>
          <td>0.340540</td>
          <td>26.734951</td>
          <td>0.149695</td>
          <td>26.061709</td>
          <td>0.135037</td>
          <td>25.876260</td>
          <td>0.215074</td>
          <td>25.757082</td>
          <td>0.408980</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.661842</td>
          <td>0.426977</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.802015</td>
          <td>0.361057</td>
          <td>25.879954</td>
          <td>0.115340</td>
          <td>24.968728</td>
          <td>0.098703</td>
          <td>24.395589</td>
          <td>0.133422</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.401560</td>
          <td>0.627331</td>
          <td>27.361487</td>
          <td>0.253527</td>
          <td>26.575826</td>
          <td>0.209166</td>
          <td>25.668991</td>
          <td>0.180674</td>
          <td>24.814809</td>
          <td>0.190888</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.073229</td>
          <td>0.095549</td>
          <td>25.889898</td>
          <td>0.071535</td>
          <td>25.545071</td>
          <td>0.086011</td>
          <td>25.435537</td>
          <td>0.148045</td>
          <td>24.755285</td>
          <td>0.181525</td>
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
          <td>26.625254</td>
          <td>0.415232</td>
          <td>26.600992</td>
          <td>0.151051</td>
          <td>25.416503</td>
          <td>0.047005</td>
          <td>25.221366</td>
          <td>0.064609</td>
          <td>24.912946</td>
          <td>0.093989</td>
          <td>24.708468</td>
          <td>0.174459</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.066893</td>
          <td>0.223904</td>
          <td>26.048043</td>
          <td>0.082261</td>
          <td>25.229678</td>
          <td>0.065087</td>
          <td>24.832033</td>
          <td>0.087536</td>
          <td>24.284217</td>
          <td>0.121147</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.176077</td>
          <td>0.621939</td>
          <td>26.702553</td>
          <td>0.164750</td>
          <td>26.351617</td>
          <td>0.107387</td>
          <td>26.101464</td>
          <td>0.139750</td>
          <td>26.277108</td>
          <td>0.298783</td>
          <td>25.840398</td>
          <td>0.435811</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.602117</td>
          <td>0.407943</td>
          <td>26.050891</td>
          <td>0.093696</td>
          <td>26.050249</td>
          <td>0.082421</td>
          <td>25.651384</td>
          <td>0.094440</td>
          <td>25.748241</td>
          <td>0.193183</td>
          <td>25.080422</td>
          <td>0.238281</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.736671</td>
          <td>0.451843</td>
          <td>26.915512</td>
          <td>0.197296</td>
          <td>26.774114</td>
          <td>0.154807</td>
          <td>26.265973</td>
          <td>0.160943</td>
          <td>26.204836</td>
          <td>0.281845</td>
          <td>25.946428</td>
          <td>0.472007</td>
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
          <td>27.317272</td>
          <td>0.750209</td>
          <td>26.423636</td>
          <td>0.149178</td>
          <td>25.946444</td>
          <td>0.088461</td>
          <td>25.269049</td>
          <td>0.079867</td>
          <td>24.774385</td>
          <td>0.097802</td>
          <td>23.966404</td>
          <td>0.108400</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.092979</td>
          <td>0.261633</td>
          <td>26.329484</td>
          <td>0.123664</td>
          <td>26.824127</td>
          <td>0.300215</td>
          <td>25.842089</td>
          <td>0.243502</td>
          <td>26.832640</td>
          <td>0.980424</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.168145</td>
          <td>2.044096</td>
          <td>27.798939</td>
          <td>0.463652</td>
          <td>30.312951</td>
          <td>1.913906</td>
          <td>26.161540</td>
          <td>0.177269</td>
          <td>25.065262</td>
          <td>0.128836</td>
          <td>24.450271</td>
          <td>0.168324</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.771050</td>
          <td>1.035797</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.616619</td>
          <td>0.382705</td>
          <td>26.076501</td>
          <td>0.172467</td>
          <td>25.770950</td>
          <td>0.244500</td>
          <td>25.408476</td>
          <td>0.384401</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.373489</td>
          <td>0.379183</td>
          <td>26.021375</td>
          <td>0.105336</td>
          <td>25.955685</td>
          <td>0.089212</td>
          <td>25.786286</td>
          <td>0.125646</td>
          <td>25.395863</td>
          <td>0.167510</td>
          <td>25.020841</td>
          <td>0.265286</td>
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
          <td>26.543224</td>
          <td>0.437855</td>
          <td>26.240979</td>
          <td>0.129860</td>
          <td>25.514102</td>
          <td>0.061667</td>
          <td>24.960462</td>
          <td>0.062136</td>
          <td>24.793099</td>
          <td>0.101522</td>
          <td>24.315504</td>
          <td>0.149815</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.702653</td>
          <td>0.189827</td>
          <td>26.050890</td>
          <td>0.097364</td>
          <td>25.235076</td>
          <td>0.077845</td>
          <td>24.651276</td>
          <td>0.088148</td>
          <td>24.553557</td>
          <td>0.180476</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.551495</td>
          <td>0.438225</td>
          <td>26.527029</td>
          <td>0.164806</td>
          <td>26.566472</td>
          <td>0.153592</td>
          <td>26.581324</td>
          <td>0.249413</td>
          <td>26.071954</td>
          <td>0.297096</td>
          <td>25.821672</td>
          <td>0.500890</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.916159</td>
          <td>0.579873</td>
          <td>25.913167</td>
          <td>0.098555</td>
          <td>26.146696</td>
          <td>0.108791</td>
          <td>25.931343</td>
          <td>0.146978</td>
          <td>25.244515</td>
          <td>0.151704</td>
          <td>25.317599</td>
          <td>0.346577</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.005526</td>
          <td>0.609662</td>
          <td>26.800330</td>
          <td>0.207101</td>
          <td>26.456909</td>
          <td>0.139415</td>
          <td>26.318721</td>
          <td>0.199969</td>
          <td>25.999702</td>
          <td>0.279540</td>
          <td>25.496503</td>
          <td>0.390823</td>
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
          <td>26.911213</td>
          <td>0.514421</td>
          <td>26.730217</td>
          <td>0.168696</td>
          <td>26.042624</td>
          <td>0.081879</td>
          <td>25.247982</td>
          <td>0.066160</td>
          <td>24.579453</td>
          <td>0.070050</td>
          <td>24.000422</td>
          <td>0.094560</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.511024</td>
          <td>0.321710</td>
          <td>26.643056</td>
          <td>0.138441</td>
          <td>26.570184</td>
          <td>0.208378</td>
          <td>25.865752</td>
          <td>0.213387</td>
          <td>25.152322</td>
          <td>0.253046</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.395961</td>
          <td>0.661417</td>
          <td>27.781037</td>
          <td>0.381933</td>
          <td>26.008508</td>
          <td>0.140281</td>
          <td>25.006303</td>
          <td>0.110655</td>
          <td>24.177230</td>
          <td>0.120048</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.510268</td>
          <td>0.881319</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.997881</td>
          <td>0.232030</td>
          <td>26.693435</td>
          <td>0.286911</td>
          <td>25.445966</td>
          <td>0.185782</td>
          <td>25.678410</td>
          <td>0.470606</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.478501</td>
          <td>0.371098</td>
          <td>26.237673</td>
          <td>0.110455</td>
          <td>26.024295</td>
          <td>0.080671</td>
          <td>25.752912</td>
          <td>0.103385</td>
          <td>25.492750</td>
          <td>0.155705</td>
          <td>24.864464</td>
          <td>0.199317</td>
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
          <td>27.069061</td>
          <td>0.603208</td>
          <td>26.321364</td>
          <td>0.127024</td>
          <td>25.451979</td>
          <td>0.052537</td>
          <td>25.022925</td>
          <td>0.058890</td>
          <td>24.959364</td>
          <td>0.105882</td>
          <td>25.575397</td>
          <td>0.381865</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.676160</td>
          <td>0.435943</td>
          <td>26.784589</td>
          <td>0.179109</td>
          <td>26.172604</td>
          <td>0.093321</td>
          <td>25.126392</td>
          <td>0.060437</td>
          <td>24.835822</td>
          <td>0.089289</td>
          <td>24.240074</td>
          <td>0.118579</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.569320</td>
          <td>0.153265</td>
          <td>26.346292</td>
          <td>0.112214</td>
          <td>26.222427</td>
          <td>0.162985</td>
          <td>26.030611</td>
          <td>0.255865</td>
          <td>25.188258</td>
          <td>0.272924</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.676435</td>
          <td>0.463272</td>
          <td>26.229789</td>
          <td>0.121090</td>
          <td>26.091592</td>
          <td>0.095883</td>
          <td>25.868721</td>
          <td>0.128582</td>
          <td>25.408338</td>
          <td>0.161723</td>
          <td>25.849542</td>
          <td>0.485577</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.754089</td>
          <td>0.929746</td>
          <td>27.026775</td>
          <td>0.223621</td>
          <td>26.647373</td>
          <td>0.144228</td>
          <td>26.705363</td>
          <td>0.242087</td>
          <td>26.153747</td>
          <td>0.280278</td>
          <td>27.173453</td>
          <td>1.101604</td>
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
