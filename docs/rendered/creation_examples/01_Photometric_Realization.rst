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

    <pzflow.flow.Flow at 0x7f6584b75e40>



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
    0      23.994413  0.077874  0.057106  
    1      25.391064  0.242260  0.230653  
    2      24.304707  0.192076  0.102933  
    3      25.291103  0.129723  0.099328  
    4      25.096743  0.065136  0.053104  
    ...          ...       ...       ...  
    99995  24.737946  0.101735  0.100613  
    99996  24.224169  0.025490  0.015860  
    99997  25.613836  0.129274  0.109798  
    99998  25.274899  0.040436  0.024702  
    99999  25.699642  0.046379  0.043498  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.566390</td>
          <td>0.146632</td>
          <td>25.953542</td>
          <td>0.075677</td>
          <td>25.096018</td>
          <td>0.057810</td>
          <td>24.730123</td>
          <td>0.080017</td>
          <td>23.966404</td>
          <td>0.091764</td>
          <td>0.077874</td>
          <td>0.057106</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.014501</td>
          <td>1.790310</td>
          <td>27.026858</td>
          <td>0.216568</td>
          <td>26.535499</td>
          <td>0.126027</td>
          <td>26.194744</td>
          <td>0.151422</td>
          <td>25.982543</td>
          <td>0.234930</td>
          <td>25.445879</td>
          <td>0.320601</td>
          <td>0.242260</td>
          <td>0.230653</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.124230</td>
          <td>1.137444</td>
          <td>29.544402</td>
          <td>1.276421</td>
          <td>28.051323</td>
          <td>0.437537</td>
          <td>26.106348</td>
          <td>0.140339</td>
          <td>24.901748</td>
          <td>0.093070</td>
          <td>24.350785</td>
          <td>0.128349</td>
          <td>0.192076</td>
          <td>0.102933</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.491963</td>
          <td>0.770968</td>
          <td>28.975720</td>
          <td>0.916964</td>
          <td>27.233479</td>
          <td>0.228112</td>
          <td>26.633362</td>
          <td>0.219457</td>
          <td>25.560247</td>
          <td>0.164723</td>
          <td>25.275913</td>
          <td>0.279643</td>
          <td>0.129723</td>
          <td>0.099328</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.371254</td>
          <td>0.340875</td>
          <td>26.246428</td>
          <td>0.111165</td>
          <td>25.959600</td>
          <td>0.076083</td>
          <td>25.761218</td>
          <td>0.103984</td>
          <td>25.637662</td>
          <td>0.175938</td>
          <td>25.286034</td>
          <td>0.281947</td>
          <td>0.065136</td>
          <td>0.053104</td>
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
          <td>27.105846</td>
          <td>0.591897</td>
          <td>26.279265</td>
          <td>0.114390</td>
          <td>25.454210</td>
          <td>0.048605</td>
          <td>24.948832</td>
          <td>0.050728</td>
          <td>24.857881</td>
          <td>0.089549</td>
          <td>24.801809</td>
          <td>0.188806</td>
          <td>0.101735</td>
          <td>0.100613</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.582097</td>
          <td>3.185314</td>
          <td>26.986611</td>
          <td>0.209413</td>
          <td>25.867980</td>
          <td>0.070161</td>
          <td>25.260363</td>
          <td>0.066881</td>
          <td>24.957866</td>
          <td>0.097768</td>
          <td>24.096123</td>
          <td>0.102820</td>
          <td>0.025490</td>
          <td>0.015860</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.385853</td>
          <td>0.344821</td>
          <td>26.935339</td>
          <td>0.200609</td>
          <td>26.441648</td>
          <td>0.116159</td>
          <td>26.329296</td>
          <td>0.169872</td>
          <td>25.780338</td>
          <td>0.198472</td>
          <td>25.237849</td>
          <td>0.271124</td>
          <td>0.129274</td>
          <td>0.109798</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.182631</td>
          <td>0.624799</td>
          <td>26.173008</td>
          <td>0.104267</td>
          <td>26.118020</td>
          <td>0.087492</td>
          <td>25.716445</td>
          <td>0.099987</td>
          <td>25.486965</td>
          <td>0.154722</td>
          <td>25.424750</td>
          <td>0.315241</td>
          <td>0.040436</td>
          <td>0.024702</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.321494</td>
          <td>0.327715</td>
          <td>26.552612</td>
          <td>0.144906</td>
          <td>26.911457</td>
          <td>0.174050</td>
          <td>26.630579</td>
          <td>0.218949</td>
          <td>25.998936</td>
          <td>0.238136</td>
          <td>25.198335</td>
          <td>0.262523</td>
          <td>0.046379</td>
          <td>0.043498</td>
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
          <td>28.100992</td>
          <td>1.219288</td>
          <td>26.944766</td>
          <td>0.234650</td>
          <td>26.218598</td>
          <td>0.114022</td>
          <td>25.088552</td>
          <td>0.069208</td>
          <td>24.663200</td>
          <td>0.090101</td>
          <td>24.013438</td>
          <td>0.114748</td>
          <td>0.077874</td>
          <td>0.057106</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.240987</td>
          <td>1.408673</td>
          <td>27.343506</td>
          <td>0.365306</td>
          <td>26.498062</td>
          <td>0.167240</td>
          <td>25.980383</td>
          <td>0.174382</td>
          <td>26.794466</td>
          <td>0.585164</td>
          <td>25.355159</td>
          <td>0.401107</td>
          <td>0.242260</td>
          <td>0.230653</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.110445</td>
          <td>0.556824</td>
          <td>25.834614</td>
          <td>0.141070</td>
          <td>25.058235</td>
          <td>0.134633</td>
          <td>24.164947</td>
          <td>0.138697</td>
          <td>0.192076</td>
          <td>0.102933</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.613898</td>
          <td>3.378363</td>
          <td>28.099621</td>
          <td>0.586747</td>
          <td>27.366191</td>
          <td>0.307456</td>
          <td>26.325031</td>
          <td>0.207734</td>
          <td>25.455121</td>
          <td>0.183613</td>
          <td>25.520219</td>
          <td>0.410103</td>
          <td>0.129723</td>
          <td>0.099328</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.159769</td>
          <td>0.323107</td>
          <td>26.039853</td>
          <td>0.108162</td>
          <td>25.918877</td>
          <td>0.087374</td>
          <td>25.735620</td>
          <td>0.121668</td>
          <td>25.718983</td>
          <td>0.222331</td>
          <td>24.935479</td>
          <td>0.250119</td>
          <td>0.065136</td>
          <td>0.053104</td>
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
          <td>27.349043</td>
          <td>0.781080</td>
          <td>26.272575</td>
          <td>0.134967</td>
          <td>25.494790</td>
          <td>0.061402</td>
          <td>24.975440</td>
          <td>0.063805</td>
          <td>24.638909</td>
          <td>0.089808</td>
          <td>25.243096</td>
          <td>0.327413</td>
          <td>0.101735</td>
          <td>0.100613</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.048883</td>
          <td>0.625245</td>
          <td>26.639084</td>
          <td>0.179484</td>
          <td>26.016475</td>
          <td>0.094217</td>
          <td>25.287125</td>
          <td>0.081277</td>
          <td>24.843997</td>
          <td>0.104104</td>
          <td>24.077568</td>
          <td>0.119604</td>
          <td>0.025490</td>
          <td>0.015860</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.424992</td>
          <td>0.407058</td>
          <td>27.037222</td>
          <td>0.259939</td>
          <td>26.472754</td>
          <td>0.146484</td>
          <td>26.246453</td>
          <td>0.195107</td>
          <td>26.070612</td>
          <td>0.306083</td>
          <td>25.053052</td>
          <td>0.284551</td>
          <td>0.129274</td>
          <td>0.109798</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.914362</td>
          <td>0.263672</td>
          <td>26.283275</td>
          <td>0.132646</td>
          <td>26.216287</td>
          <td>0.112467</td>
          <td>25.941518</td>
          <td>0.144179</td>
          <td>25.606239</td>
          <td>0.200795</td>
          <td>25.444818</td>
          <td>0.373334</td>
          <td>0.040436</td>
          <td>0.024702</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.971021</td>
          <td>0.593807</td>
          <td>26.918369</td>
          <td>0.227869</td>
          <td>26.706344</td>
          <td>0.172065</td>
          <td>26.374318</td>
          <td>0.208842</td>
          <td>25.926828</td>
          <td>0.262651</td>
          <td>25.252678</td>
          <td>0.321793</td>
          <td>0.046379</td>
          <td>0.043498</td>
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
          <td>27.266830</td>
          <td>0.684374</td>
          <td>26.538843</td>
          <td>0.150640</td>
          <td>25.953274</td>
          <td>0.080306</td>
          <td>25.195217</td>
          <td>0.067211</td>
          <td>24.464576</td>
          <td>0.067161</td>
          <td>23.955726</td>
          <td>0.096668</td>
          <td>0.077874</td>
          <td>0.057106</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.475943</td>
          <td>0.450747</td>
          <td>27.022351</td>
          <td>0.293090</td>
          <td>26.809464</td>
          <td>0.388725</td>
          <td>26.087981</td>
          <td>0.387058</td>
          <td>25.507842</td>
          <td>0.505188</td>
          <td>0.242260</td>
          <td>0.230653</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.315742</td>
          <td>0.783014</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.835892</td>
          <td>0.141126</td>
          <td>24.847572</td>
          <td>0.112103</td>
          <td>24.343166</td>
          <td>0.161512</td>
          <td>0.192076</td>
          <td>0.102933</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.623830</td>
          <td>0.396110</td>
          <td>27.013527</td>
          <td>0.219489</td>
          <td>26.654119</td>
          <td>0.259238</td>
          <td>25.474493</td>
          <td>0.177483</td>
          <td>25.042299</td>
          <td>0.267575</td>
          <td>0.129723</td>
          <td>0.099328</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.695232</td>
          <td>0.202458</td>
          <td>26.298255</td>
          <td>0.120974</td>
          <td>25.919173</td>
          <td>0.076868</td>
          <td>25.616853</td>
          <td>0.096111</td>
          <td>25.266900</td>
          <td>0.133883</td>
          <td>25.005528</td>
          <td>0.234183</td>
          <td>0.065136</td>
          <td>0.053104</td>
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
          <td>27.245935</td>
          <td>0.699119</td>
          <td>26.469964</td>
          <td>0.149951</td>
          <td>25.355637</td>
          <td>0.050401</td>
          <td>25.086970</td>
          <td>0.065245</td>
          <td>24.822417</td>
          <td>0.098073</td>
          <td>24.618675</td>
          <td>0.182638</td>
          <td>0.101735</td>
          <td>0.100613</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.281511</td>
          <td>0.671347</td>
          <td>27.033293</td>
          <td>0.218824</td>
          <td>26.006915</td>
          <td>0.079816</td>
          <td>25.193035</td>
          <td>0.063414</td>
          <td>24.750938</td>
          <td>0.081998</td>
          <td>24.118465</td>
          <td>0.105512</td>
          <td>0.025490</td>
          <td>0.015860</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.405858</td>
          <td>0.796153</td>
          <td>26.818450</td>
          <td>0.208796</td>
          <td>26.290621</td>
          <td>0.119753</td>
          <td>26.864448</td>
          <td>0.310555</td>
          <td>25.659571</td>
          <td>0.209583</td>
          <td>25.376337</td>
          <td>0.353228</td>
          <td>0.129274</td>
          <td>0.109798</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.214988</td>
          <td>0.303890</td>
          <td>26.253064</td>
          <td>0.113264</td>
          <td>26.117506</td>
          <td>0.088776</td>
          <td>25.758520</td>
          <td>0.105376</td>
          <td>25.681209</td>
          <td>0.185200</td>
          <td>25.263243</td>
          <td>0.280785</td>
          <td>0.040436</td>
          <td>0.024702</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.185334</td>
          <td>0.635318</td>
          <td>26.652264</td>
          <td>0.161437</td>
          <td>26.443426</td>
          <td>0.119459</td>
          <td>26.508251</td>
          <td>0.203006</td>
          <td>25.596056</td>
          <td>0.174268</td>
          <td>25.237927</td>
          <td>0.278189</td>
          <td>0.046379</td>
          <td>0.043498</td>
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
