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

    <pzflow.flow.Flow at 0x7fa906d6f730>



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
          <td>28.965001</td>
          <td>1.750510</td>
          <td>26.678589</td>
          <td>0.161417</td>
          <td>26.044853</td>
          <td>0.082030</td>
          <td>25.281823</td>
          <td>0.068164</td>
          <td>24.713113</td>
          <td>0.078824</td>
          <td>23.919010</td>
          <td>0.088018</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.355589</td>
          <td>0.703822</td>
          <td>27.129869</td>
          <td>0.235899</td>
          <td>26.680081</td>
          <td>0.142797</td>
          <td>26.185477</td>
          <td>0.150223</td>
          <td>26.017619</td>
          <td>0.241837</td>
          <td>25.304942</td>
          <td>0.286296</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>26.883369</td>
          <td>0.503971</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.109832</td>
          <td>0.140761</td>
          <td>24.936405</td>
          <td>0.095945</td>
          <td>24.649471</td>
          <td>0.165918</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.378960</td>
          <td>0.617474</td>
          <td>26.880216</td>
          <td>0.169488</td>
          <td>26.378540</td>
          <td>0.177131</td>
          <td>25.484817</td>
          <td>0.154437</td>
          <td>25.049704</td>
          <td>0.232303</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.110541</td>
          <td>0.276683</td>
          <td>25.968224</td>
          <td>0.087135</td>
          <td>25.996256</td>
          <td>0.078586</td>
          <td>25.754606</td>
          <td>0.103384</td>
          <td>25.577023</td>
          <td>0.167095</td>
          <td>25.204677</td>
          <td>0.263887</td>
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
          <td>27.897803</td>
          <td>0.995975</td>
          <td>26.596483</td>
          <td>0.150468</td>
          <td>25.402659</td>
          <td>0.046431</td>
          <td>25.077114</td>
          <td>0.056848</td>
          <td>24.880445</td>
          <td>0.091344</td>
          <td>24.878915</td>
          <td>0.201467</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.118161</td>
          <td>0.597084</td>
          <td>26.948795</td>
          <td>0.202886</td>
          <td>26.027222</td>
          <td>0.080764</td>
          <td>25.222988</td>
          <td>0.064702</td>
          <td>24.716405</td>
          <td>0.079054</td>
          <td>24.245581</td>
          <td>0.117145</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.076872</td>
          <td>0.579826</td>
          <td>26.563124</td>
          <td>0.146221</td>
          <td>26.356094</td>
          <td>0.107808</td>
          <td>26.303504</td>
          <td>0.166181</td>
          <td>26.082662</td>
          <td>0.255124</td>
          <td>25.919441</td>
          <td>0.462573</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.419835</td>
          <td>0.354155</td>
          <td>26.137314</td>
          <td>0.101064</td>
          <td>26.126971</td>
          <td>0.088184</td>
          <td>25.988030</td>
          <td>0.126696</td>
          <td>25.629809</td>
          <td>0.174769</td>
          <td>25.550715</td>
          <td>0.348363</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.700209</td>
          <td>0.439583</td>
          <td>26.651959</td>
          <td>0.157787</td>
          <td>26.467257</td>
          <td>0.118776</td>
          <td>26.248721</td>
          <td>0.158587</td>
          <td>25.771020</td>
          <td>0.196923</td>
          <td>25.310146</td>
          <td>0.287504</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.383078</td>
          <td>0.144073</td>
          <td>26.041582</td>
          <td>0.096171</td>
          <td>25.183674</td>
          <td>0.074067</td>
          <td>24.692518</td>
          <td>0.091022</td>
          <td>23.859398</td>
          <td>0.098716</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.511804</td>
          <td>1.502620</td>
          <td>27.099686</td>
          <td>0.263070</td>
          <td>26.659739</td>
          <td>0.164316</td>
          <td>26.123502</td>
          <td>0.167878</td>
          <td>25.750950</td>
          <td>0.225818</td>
          <td>25.029644</td>
          <td>0.267167</td>
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
          <td>27.611055</td>
          <td>0.366218</td>
          <td>25.963459</td>
          <td>0.149702</td>
          <td>25.036517</td>
          <td>0.125667</td>
          <td>24.081929</td>
          <td>0.122621</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>26.777255</td>
          <td>0.536799</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.900437</td>
          <td>0.214699</td>
          <td>26.775636</td>
          <td>0.307573</td>
          <td>25.884137</td>
          <td>0.268261</td>
          <td>25.061956</td>
          <td>0.292208</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.121635</td>
          <td>0.310949</td>
          <td>26.041814</td>
          <td>0.107231</td>
          <td>25.957069</td>
          <td>0.089320</td>
          <td>25.710661</td>
          <td>0.117660</td>
          <td>25.495805</td>
          <td>0.182343</td>
          <td>25.174732</td>
          <td>0.300508</td>
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
          <td>27.216363</td>
          <td>0.709775</td>
          <td>26.287747</td>
          <td>0.135212</td>
          <td>25.446161</td>
          <td>0.058062</td>
          <td>24.981954</td>
          <td>0.063331</td>
          <td>24.789700</td>
          <td>0.101221</td>
          <td>24.648589</td>
          <td>0.198812</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.750800</td>
          <td>0.197676</td>
          <td>25.843352</td>
          <td>0.081122</td>
          <td>25.131989</td>
          <td>0.071067</td>
          <td>24.885970</td>
          <td>0.108279</td>
          <td>24.481528</td>
          <td>0.169772</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.631172</td>
          <td>0.924337</td>
          <td>26.877375</td>
          <td>0.221369</td>
          <td>26.481432</td>
          <td>0.142775</td>
          <td>25.998272</td>
          <td>0.152765</td>
          <td>26.844726</td>
          <td>0.537768</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.744747</td>
          <td>0.233999</td>
          <td>26.119965</td>
          <td>0.118028</td>
          <td>26.043832</td>
          <td>0.099433</td>
          <td>26.318987</td>
          <td>0.204279</td>
          <td>25.700804</td>
          <td>0.223084</td>
          <td>25.445736</td>
          <td>0.383093</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.815717</td>
          <td>0.532219</td>
          <td>26.607598</td>
          <td>0.176063</td>
          <td>26.636975</td>
          <td>0.162701</td>
          <td>26.155404</td>
          <td>0.174202</td>
          <td>25.922020</td>
          <td>0.262406</td>
          <td>25.665069</td>
          <td>0.444560</td>
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
          <td>27.544021</td>
          <td>0.797783</td>
          <td>26.654481</td>
          <td>0.158145</td>
          <td>26.057810</td>
          <td>0.082983</td>
          <td>25.096476</td>
          <td>0.057842</td>
          <td>24.813895</td>
          <td>0.086161</td>
          <td>24.096611</td>
          <td>0.102878</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.974702</td>
          <td>0.207499</td>
          <td>26.675460</td>
          <td>0.142362</td>
          <td>26.769274</td>
          <td>0.245833</td>
          <td>25.668674</td>
          <td>0.180790</td>
          <td>25.665772</td>
          <td>0.381485</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.335161</td>
          <td>1.325380</td>
          <td>29.032827</td>
          <td>0.998309</td>
          <td>28.175740</td>
          <td>0.514569</td>
          <td>26.206220</td>
          <td>0.166188</td>
          <td>24.859152</td>
          <td>0.097292</td>
          <td>24.466143</td>
          <td>0.154046</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.451596</td>
          <td>1.499126</td>
          <td>28.754139</td>
          <td>0.920647</td>
          <td>26.925840</td>
          <td>0.218553</td>
          <td>26.459220</td>
          <td>0.236904</td>
          <td>25.504392</td>
          <td>0.195163</td>
          <td>25.293008</td>
          <td>0.350116</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.384148</td>
          <td>0.344661</td>
          <td>26.089303</td>
          <td>0.097023</td>
          <td>25.878222</td>
          <td>0.070901</td>
          <td>25.721137</td>
          <td>0.100548</td>
          <td>25.412824</td>
          <td>0.145384</td>
          <td>24.957628</td>
          <td>0.215487</td>
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
          <td>26.857525</td>
          <td>0.518109</td>
          <td>26.286516</td>
          <td>0.123247</td>
          <td>25.469007</td>
          <td>0.053337</td>
          <td>25.123937</td>
          <td>0.064408</td>
          <td>24.809684</td>
          <td>0.092868</td>
          <td>24.649577</td>
          <td>0.179557</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.137108</td>
          <td>1.901151</td>
          <td>26.554443</td>
          <td>0.147183</td>
          <td>26.049933</td>
          <td>0.083773</td>
          <td>25.151554</td>
          <td>0.061801</td>
          <td>24.789088</td>
          <td>0.085691</td>
          <td>24.666922</td>
          <td>0.171217</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.943051</td>
          <td>0.210301</td>
          <td>26.430601</td>
          <td>0.120758</td>
          <td>26.078932</td>
          <td>0.144125</td>
          <td>25.650929</td>
          <td>0.186504</td>
          <td>25.628918</td>
          <td>0.387366</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.453379</td>
          <td>0.390982</td>
          <td>26.055543</td>
          <td>0.104045</td>
          <td>26.091447</td>
          <td>0.095871</td>
          <td>25.751560</td>
          <td>0.116146</td>
          <td>25.679953</td>
          <td>0.203526</td>
          <td>25.290413</td>
          <td>0.315406</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.093929</td>
          <td>0.279942</td>
          <td>26.697901</td>
          <td>0.169584</td>
          <td>26.473259</td>
          <td>0.124082</td>
          <td>26.342046</td>
          <td>0.178628</td>
          <td>25.981794</td>
          <td>0.243516</td>
          <td>25.866721</td>
          <td>0.460223</td>
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
