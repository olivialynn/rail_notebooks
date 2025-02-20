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

    <pzflow.flow.Flow at 0x7f2bce2f8670>



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
          <td>27.443943</td>
          <td>0.746837</td>
          <td>26.401084</td>
          <td>0.127149</td>
          <td>25.903300</td>
          <td>0.072388</td>
          <td>25.127562</td>
          <td>0.059451</td>
          <td>24.573947</td>
          <td>0.069700</td>
          <td>24.044181</td>
          <td>0.098247</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.982449</td>
          <td>0.541785</td>
          <td>27.809170</td>
          <td>0.405935</td>
          <td>26.917717</td>
          <td>0.174978</td>
          <td>26.304219</td>
          <td>0.166282</td>
          <td>25.977121</td>
          <td>0.233879</td>
          <td>25.243220</td>
          <td>0.272312</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.754544</td>
          <td>1.585437</td>
          <td>29.982764</td>
          <td>1.597305</td>
          <td>27.885602</td>
          <td>0.385349</td>
          <td>26.163541</td>
          <td>0.147419</td>
          <td>24.946503</td>
          <td>0.096799</td>
          <td>24.177981</td>
          <td>0.110445</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.964229</td>
          <td>0.456683</td>
          <td>27.236286</td>
          <td>0.228644</td>
          <td>26.306793</td>
          <td>0.166647</td>
          <td>25.418938</td>
          <td>0.145948</td>
          <td>24.782217</td>
          <td>0.185708</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>28.412331</td>
          <td>1.332883</td>
          <td>25.973527</td>
          <td>0.087542</td>
          <td>25.889322</td>
          <td>0.071499</td>
          <td>25.778529</td>
          <td>0.105570</td>
          <td>25.852401</td>
          <td>0.210831</td>
          <td>25.207220</td>
          <td>0.264436</td>
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
          <td>26.687381</td>
          <td>0.435335</td>
          <td>26.384786</td>
          <td>0.125367</td>
          <td>25.479482</td>
          <td>0.049708</td>
          <td>25.150400</td>
          <td>0.060668</td>
          <td>24.859694</td>
          <td>0.089692</td>
          <td>25.016583</td>
          <td>0.226009</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.615565</td>
          <td>0.412167</td>
          <td>27.267194</td>
          <td>0.264073</td>
          <td>25.931011</td>
          <td>0.074184</td>
          <td>25.254459</td>
          <td>0.066532</td>
          <td>24.776499</td>
          <td>0.083357</td>
          <td>24.178998</td>
          <td>0.110543</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>29.152713</td>
          <td>1.903264</td>
          <td>26.653181</td>
          <td>0.157952</td>
          <td>26.383916</td>
          <td>0.110458</td>
          <td>26.556819</td>
          <td>0.205864</td>
          <td>25.992631</td>
          <td>0.236898</td>
          <td>25.089374</td>
          <td>0.240048</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.837377</td>
          <td>0.221091</td>
          <td>26.188722</td>
          <td>0.105708</td>
          <td>26.175894</td>
          <td>0.092061</td>
          <td>26.015783</td>
          <td>0.129779</td>
          <td>26.016972</td>
          <td>0.241707</td>
          <td>25.193730</td>
          <td>0.261537</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.506106</td>
          <td>0.778177</td>
          <td>26.734383</td>
          <td>0.169276</td>
          <td>26.578477</td>
          <td>0.130807</td>
          <td>26.735933</td>
          <td>0.238944</td>
          <td>26.158807</td>
          <td>0.271501</td>
          <td>25.041664</td>
          <td>0.230761</td>
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
          <td>27.035809</td>
          <td>0.618984</td>
          <td>26.794777</td>
          <td>0.204369</td>
          <td>25.976630</td>
          <td>0.090840</td>
          <td>25.301201</td>
          <td>0.082163</td>
          <td>24.734517</td>
          <td>0.094442</td>
          <td>24.047392</td>
          <td>0.116328</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.204841</td>
          <td>0.286537</td>
          <td>26.827123</td>
          <td>0.189388</td>
          <td>26.096978</td>
          <td>0.164125</td>
          <td>25.949440</td>
          <td>0.265910</td>
          <td>25.445936</td>
          <td>0.372446</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.547096</td>
          <td>1.323915</td>
          <td>26.001697</td>
          <td>0.154691</td>
          <td>25.450227</td>
          <td>0.179193</td>
          <td>24.345497</td>
          <td>0.153917</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.220738</td>
          <td>0.731622</td>
          <td>27.854275</td>
          <td>0.500057</td>
          <td>27.343331</td>
          <td>0.308505</td>
          <td>26.042617</td>
          <td>0.167567</td>
          <td>25.323381</td>
          <td>0.168013</td>
          <td>25.397163</td>
          <td>0.381043</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.827531</td>
          <td>0.244980</td>
          <td>26.158106</td>
          <td>0.118654</td>
          <td>25.920518</td>
          <td>0.086494</td>
          <td>25.720384</td>
          <td>0.118659</td>
          <td>25.552285</td>
          <td>0.191252</td>
          <td>25.471006</td>
          <td>0.379822</td>
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
          <td>27.649564</td>
          <td>0.939040</td>
          <td>26.432667</td>
          <td>0.153147</td>
          <td>25.327509</td>
          <td>0.052262</td>
          <td>25.021415</td>
          <td>0.065584</td>
          <td>24.737181</td>
          <td>0.096669</td>
          <td>24.781979</td>
          <td>0.222265</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.662289</td>
          <td>0.183468</td>
          <td>26.069039</td>
          <td>0.098925</td>
          <td>25.190853</td>
          <td>0.074864</td>
          <td>24.857877</td>
          <td>0.105655</td>
          <td>24.255449</td>
          <td>0.139887</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.680634</td>
          <td>0.187731</td>
          <td>26.278134</td>
          <td>0.119752</td>
          <td>26.160736</td>
          <td>0.175474</td>
          <td>26.091364</td>
          <td>0.301770</td>
          <td>25.920328</td>
          <td>0.538384</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.072051</td>
          <td>0.305405</td>
          <td>26.326841</td>
          <td>0.141146</td>
          <td>26.094420</td>
          <td>0.103934</td>
          <td>25.641113</td>
          <td>0.114336</td>
          <td>26.221415</td>
          <td>0.340412</td>
          <td>24.636957</td>
          <td>0.198902</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.020744</td>
          <td>0.288701</td>
          <td>26.721056</td>
          <td>0.193773</td>
          <td>26.463453</td>
          <td>0.140203</td>
          <td>26.278033</td>
          <td>0.193241</td>
          <td>25.859225</td>
          <td>0.249242</td>
          <td>26.345165</td>
          <td>0.723084</td>
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
          <td>26.743535</td>
          <td>0.454216</td>
          <td>26.773861</td>
          <td>0.175069</td>
          <td>26.070935</td>
          <td>0.083949</td>
          <td>25.136694</td>
          <td>0.059943</td>
          <td>24.618143</td>
          <td>0.072489</td>
          <td>23.933401</td>
          <td>0.089152</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.517666</td>
          <td>2.213754</td>
          <td>27.472781</td>
          <td>0.312041</td>
          <td>26.647645</td>
          <td>0.138990</td>
          <td>26.474788</td>
          <td>0.192334</td>
          <td>25.892000</td>
          <td>0.218111</td>
          <td>25.573511</td>
          <td>0.354975</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.437439</td>
          <td>1.398453</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.843117</td>
          <td>0.816297</td>
          <td>26.161629</td>
          <td>0.159983</td>
          <td>25.072396</td>
          <td>0.117213</td>
          <td>24.339696</td>
          <td>0.138180</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.098389</td>
          <td>0.595029</td>
          <td>27.092274</td>
          <td>0.250816</td>
          <td>26.154477</td>
          <td>0.183601</td>
          <td>25.351107</td>
          <td>0.171430</td>
          <td>25.071120</td>
          <td>0.293389</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.999179</td>
          <td>0.548824</td>
          <td>26.054630</td>
          <td>0.094119</td>
          <td>26.150047</td>
          <td>0.090121</td>
          <td>25.687190</td>
          <td>0.097600</td>
          <td>25.450974</td>
          <td>0.150228</td>
          <td>24.926829</td>
          <td>0.210014</td>
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
          <td>29.434024</td>
          <td>2.195597</td>
          <td>26.152038</td>
          <td>0.109648</td>
          <td>25.446864</td>
          <td>0.052299</td>
          <td>25.061083</td>
          <td>0.060917</td>
          <td>24.878281</td>
          <td>0.098628</td>
          <td>24.649103</td>
          <td>0.179485</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.831808</td>
          <td>0.489850</td>
          <td>26.937605</td>
          <td>0.203758</td>
          <td>25.941041</td>
          <td>0.076097</td>
          <td>25.194109</td>
          <td>0.064178</td>
          <td>24.812477</td>
          <td>0.087474</td>
          <td>24.135374</td>
          <td>0.108238</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.662134</td>
          <td>0.439958</td>
          <td>26.697652</td>
          <td>0.170999</td>
          <td>26.155741</td>
          <td>0.094983</td>
          <td>26.497512</td>
          <td>0.205690</td>
          <td>25.780097</td>
          <td>0.207905</td>
          <td>26.254296</td>
          <td>0.615293</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.589739</td>
          <td>0.433979</td>
          <td>26.376394</td>
          <td>0.137460</td>
          <td>26.095391</td>
          <td>0.096203</td>
          <td>25.979575</td>
          <td>0.141502</td>
          <td>25.726336</td>
          <td>0.211584</td>
          <td>25.002613</td>
          <td>0.249781</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.016167</td>
          <td>0.567653</td>
          <td>26.624626</td>
          <td>0.159316</td>
          <td>26.741536</td>
          <td>0.156367</td>
          <td>26.206792</td>
          <td>0.159198</td>
          <td>25.990374</td>
          <td>0.245243</td>
          <td>25.738979</td>
          <td>0.417787</td>
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
