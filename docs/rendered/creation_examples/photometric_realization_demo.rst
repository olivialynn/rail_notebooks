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

    <pzflow.flow.Flow at 0x7f3ea7b842e0>



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
          <td>27.559541</td>
          <td>0.805829</td>
          <td>26.766256</td>
          <td>0.173924</td>
          <td>26.192620</td>
          <td>0.093424</td>
          <td>25.164100</td>
          <td>0.061410</td>
          <td>24.729301</td>
          <td>0.079959</td>
          <td>24.041880</td>
          <td>0.098049</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.262100</td>
          <td>0.262977</td>
          <td>26.491560</td>
          <td>0.121312</td>
          <td>26.250460</td>
          <td>0.158823</td>
          <td>26.616499</td>
          <td>0.390564</td>
          <td>26.096768</td>
          <td>0.527395</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.124264</td>
          <td>0.599668</td>
          <td>29.365045</td>
          <td>1.155662</td>
          <td>28.456151</td>
          <td>0.589103</td>
          <td>26.228793</td>
          <td>0.155906</td>
          <td>25.131069</td>
          <td>0.113751</td>
          <td>24.468973</td>
          <td>0.142142</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.109315</td>
          <td>0.205676</td>
          <td>26.184749</td>
          <td>0.150129</td>
          <td>25.313873</td>
          <td>0.133310</td>
          <td>24.889701</td>
          <td>0.203298</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.925805</td>
          <td>0.237879</td>
          <td>26.127261</td>
          <td>0.100179</td>
          <td>25.879540</td>
          <td>0.070882</td>
          <td>25.845979</td>
          <td>0.111975</td>
          <td>25.436489</td>
          <td>0.148166</td>
          <td>24.725379</td>
          <td>0.176982</td>
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
          <td>26.753652</td>
          <td>0.457646</td>
          <td>26.135769</td>
          <td>0.100927</td>
          <td>25.450446</td>
          <td>0.048443</td>
          <td>25.081338</td>
          <td>0.057062</td>
          <td>24.834206</td>
          <td>0.087703</td>
          <td>24.717423</td>
          <td>0.175791</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.712336</td>
          <td>0.443631</td>
          <td>26.596964</td>
          <td>0.150530</td>
          <td>25.964559</td>
          <td>0.076417</td>
          <td>25.134463</td>
          <td>0.059816</td>
          <td>24.827648</td>
          <td>0.087199</td>
          <td>24.207309</td>
          <td>0.113306</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.923607</td>
          <td>0.198643</td>
          <td>26.368031</td>
          <td>0.108937</td>
          <td>26.219233</td>
          <td>0.154634</td>
          <td>25.871679</td>
          <td>0.214253</td>
          <td>25.558846</td>
          <td>0.350599</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>27.220850</td>
          <td>0.641673</td>
          <td>26.201517</td>
          <td>0.106896</td>
          <td>26.147209</td>
          <td>0.089768</td>
          <td>25.692369</td>
          <td>0.097899</td>
          <td>25.704056</td>
          <td>0.186114</td>
          <td>25.103219</td>
          <td>0.242805</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.465646</td>
          <td>0.367078</td>
          <td>26.823181</td>
          <td>0.182519</td>
          <td>26.760507</td>
          <td>0.153013</td>
          <td>26.519592</td>
          <td>0.199533</td>
          <td>25.982750</td>
          <td>0.234971</td>
          <td>25.239408</td>
          <td>0.271468</td>
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
          <td>28.244134</td>
          <td>1.308398</td>
          <td>26.575045</td>
          <td>0.169770</td>
          <td>26.052636</td>
          <td>0.097108</td>
          <td>25.094213</td>
          <td>0.068433</td>
          <td>24.675579</td>
          <td>0.089677</td>
          <td>23.889477</td>
          <td>0.101351</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.693698</td>
          <td>0.483543</td>
          <td>27.345758</td>
          <td>0.320843</td>
          <td>26.698639</td>
          <td>0.169852</td>
          <td>26.544398</td>
          <td>0.239006</td>
          <td>25.665239</td>
          <td>0.210253</td>
          <td>26.112903</td>
          <td>0.611404</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>32.509454</td>
          <td>5.214968</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.483079</td>
          <td>0.331118</td>
          <td>26.074020</td>
          <td>0.164554</td>
          <td>24.998529</td>
          <td>0.121593</td>
          <td>24.547002</td>
          <td>0.182726</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.827737</td>
          <td>1.794754</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.397681</td>
          <td>0.322191</td>
          <td>26.253104</td>
          <td>0.200220</td>
          <td>25.141855</td>
          <td>0.143836</td>
          <td>25.572642</td>
          <td>0.435961</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.822804</td>
          <td>0.244030</td>
          <td>26.127962</td>
          <td>0.115586</td>
          <td>25.906922</td>
          <td>0.085465</td>
          <td>25.739237</td>
          <td>0.120619</td>
          <td>25.394436</td>
          <td>0.167306</td>
          <td>24.807010</td>
          <td>0.222425</td>
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
          <td>27.633604</td>
          <td>0.929834</td>
          <td>26.599210</td>
          <td>0.176496</td>
          <td>25.446170</td>
          <td>0.058063</td>
          <td>24.987498</td>
          <td>0.063643</td>
          <td>24.816987</td>
          <td>0.103666</td>
          <td>24.582039</td>
          <td>0.187976</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.365318</td>
          <td>1.397246</td>
          <td>26.389112</td>
          <td>0.145356</td>
          <td>25.891350</td>
          <td>0.084627</td>
          <td>25.176664</td>
          <td>0.073931</td>
          <td>24.736368</td>
          <td>0.094990</td>
          <td>23.811957</td>
          <td>0.095101</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.568271</td>
          <td>0.443817</td>
          <td>26.730515</td>
          <td>0.195786</td>
          <td>26.235055</td>
          <td>0.115347</td>
          <td>26.177534</td>
          <td>0.177992</td>
          <td>25.984018</td>
          <td>0.276704</td>
          <td>25.606415</td>
          <td>0.426256</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.064700</td>
          <td>0.303612</td>
          <td>26.153724</td>
          <td>0.121537</td>
          <td>26.116410</td>
          <td>0.105951</td>
          <td>25.643976</td>
          <td>0.114622</td>
          <td>26.033655</td>
          <td>0.293026</td>
          <td>24.802716</td>
          <td>0.228424</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.651716</td>
          <td>0.934774</td>
          <td>26.457126</td>
          <td>0.154883</td>
          <td>26.624594</td>
          <td>0.160991</td>
          <td>26.205269</td>
          <td>0.181725</td>
          <td>25.656274</td>
          <td>0.210643</td>
          <td>26.003180</td>
          <td>0.570207</td>
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
          <td>27.940459</td>
          <td>1.021839</td>
          <td>26.674633</td>
          <td>0.160891</td>
          <td>26.141677</td>
          <td>0.089344</td>
          <td>25.185580</td>
          <td>0.062600</td>
          <td>24.646156</td>
          <td>0.074307</td>
          <td>24.098322</td>
          <td>0.103032</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.066519</td>
          <td>1.100816</td>
          <td>27.419971</td>
          <td>0.299103</td>
          <td>26.706400</td>
          <td>0.146203</td>
          <td>26.361791</td>
          <td>0.174798</td>
          <td>25.843861</td>
          <td>0.209519</td>
          <td>25.383740</td>
          <td>0.305334</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.207280</td>
          <td>0.665626</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.025368</td>
          <td>0.916393</td>
          <td>26.037240</td>
          <td>0.143795</td>
          <td>24.879934</td>
          <td>0.099080</td>
          <td>24.479390</td>
          <td>0.155804</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.489797</td>
          <td>1.527841</td>
          <td>27.559542</td>
          <td>0.399286</td>
          <td>27.319833</td>
          <td>0.301759</td>
          <td>26.599019</td>
          <td>0.265728</td>
          <td>25.577954</td>
          <td>0.207592</td>
          <td>25.520970</td>
          <td>0.417824</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.612936</td>
          <td>0.411691</td>
          <td>26.110209</td>
          <td>0.098816</td>
          <td>25.990263</td>
          <td>0.078283</td>
          <td>25.719103</td>
          <td>0.100369</td>
          <td>25.300935</td>
          <td>0.132011</td>
          <td>25.293645</td>
          <td>0.284077</td>
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
          <td>27.729395</td>
          <td>0.934244</td>
          <td>26.444207</td>
          <td>0.141234</td>
          <td>25.357403</td>
          <td>0.048307</td>
          <td>25.104268</td>
          <td>0.063295</td>
          <td>24.980928</td>
          <td>0.107895</td>
          <td>24.454667</td>
          <td>0.152068</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.495801</td>
          <td>2.205442</td>
          <td>26.871070</td>
          <td>0.192681</td>
          <td>25.976969</td>
          <td>0.078551</td>
          <td>25.231171</td>
          <td>0.066321</td>
          <td>24.699537</td>
          <td>0.079186</td>
          <td>24.248549</td>
          <td>0.119455</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.350419</td>
          <td>0.345849</td>
          <td>26.505132</td>
          <td>0.145053</td>
          <td>26.400777</td>
          <td>0.117667</td>
          <td>25.857683</td>
          <td>0.119018</td>
          <td>25.724648</td>
          <td>0.198457</td>
          <td>25.772217</td>
          <td>0.432360</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.249001</td>
          <td>0.333236</td>
          <td>26.091392</td>
          <td>0.107353</td>
          <td>26.072718</td>
          <td>0.094308</td>
          <td>26.134697</td>
          <td>0.161638</td>
          <td>25.555239</td>
          <td>0.183231</td>
          <td>25.284974</td>
          <td>0.314039</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.581708</td>
          <td>0.153574</td>
          <td>26.492304</td>
          <td>0.126148</td>
          <td>26.223572</td>
          <td>0.161497</td>
          <td>25.879287</td>
          <td>0.223707</td>
          <td>25.662597</td>
          <td>0.393982</td>
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
