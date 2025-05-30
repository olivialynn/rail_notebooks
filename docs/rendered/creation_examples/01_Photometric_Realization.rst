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

    <pzflow.flow.Flow at 0x7f499f3d69b0>



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
    0      23.994413  0.014384  0.010763  
    1      25.391064  0.037802  0.030091  
    2      24.304707  0.188568  0.171464  
    3      25.291103  0.125312  0.087238  
    4      25.096743  0.015132  0.011637  
    ...          ...       ...       ...  
    99995  24.737946  0.005822  0.005495  
    99996  24.224169  0.061637  0.061282  
    99997  25.613836  0.072844  0.038162  
    99998  25.274899  0.008937  0.006722  
    99999  25.699642  0.022677  0.013466  
    
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
          <td>26.399067</td>
          <td>0.348425</td>
          <td>26.680189</td>
          <td>0.161638</td>
          <td>25.981637</td>
          <td>0.077578</td>
          <td>25.121613</td>
          <td>0.059138</td>
          <td>24.686950</td>
          <td>0.077024</td>
          <td>23.903646</td>
          <td>0.086836</td>
          <td>0.014384</td>
          <td>0.010763</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.316823</td>
          <td>1.266256</td>
          <td>27.091794</td>
          <td>0.228580</td>
          <td>26.500882</td>
          <td>0.122298</td>
          <td>26.358140</td>
          <td>0.174089</td>
          <td>25.781791</td>
          <td>0.198715</td>
          <td>25.334427</td>
          <td>0.293196</td>
          <td>0.037802</td>
          <td>0.030091</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.220809</td>
          <td>0.551715</td>
          <td>28.126160</td>
          <td>0.462923</td>
          <td>25.925325</td>
          <td>0.119984</td>
          <td>25.039160</td>
          <td>0.104981</td>
          <td>24.221767</td>
          <td>0.114742</td>
          <td>0.188568</td>
          <td>0.171464</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.391098</td>
          <td>0.720893</td>
          <td>27.957806</td>
          <td>0.454484</td>
          <td>27.353760</td>
          <td>0.251924</td>
          <td>26.195146</td>
          <td>0.151474</td>
          <td>25.465448</td>
          <td>0.151895</td>
          <td>25.185355</td>
          <td>0.259752</td>
          <td>0.125312</td>
          <td>0.087238</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.827603</td>
          <td>0.219303</td>
          <td>26.086293</td>
          <td>0.096649</td>
          <td>25.839588</td>
          <td>0.068419</td>
          <td>25.796438</td>
          <td>0.107236</td>
          <td>25.357820</td>
          <td>0.138465</td>
          <td>25.025528</td>
          <td>0.227693</td>
          <td>0.015132</td>
          <td>0.011637</td>
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
          <td>27.564339</td>
          <td>0.808343</td>
          <td>26.388280</td>
          <td>0.125747</td>
          <td>25.553260</td>
          <td>0.053074</td>
          <td>25.003144</td>
          <td>0.053235</td>
          <td>25.001349</td>
          <td>0.101565</td>
          <td>24.960815</td>
          <td>0.215758</td>
          <td>0.005822</td>
          <td>0.005495</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.065325</td>
          <td>1.099581</td>
          <td>26.742462</td>
          <td>0.170443</td>
          <td>26.025824</td>
          <td>0.080664</td>
          <td>25.209905</td>
          <td>0.063956</td>
          <td>24.966822</td>
          <td>0.098539</td>
          <td>24.254936</td>
          <td>0.118103</td>
          <td>0.061637</td>
          <td>0.061282</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.583247</td>
          <td>0.402077</td>
          <td>26.663407</td>
          <td>0.159338</td>
          <td>26.542676</td>
          <td>0.126814</td>
          <td>26.114253</td>
          <td>0.141298</td>
          <td>26.310833</td>
          <td>0.306985</td>
          <td>26.031394</td>
          <td>0.502714</td>
          <td>0.072844</td>
          <td>0.038162</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.901539</td>
          <td>0.233162</td>
          <td>26.207013</td>
          <td>0.107410</td>
          <td>26.150777</td>
          <td>0.090050</td>
          <td>25.887693</td>
          <td>0.116120</td>
          <td>26.088145</td>
          <td>0.256274</td>
          <td>25.732917</td>
          <td>0.401457</td>
          <td>0.008937</td>
          <td>0.006722</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.884848</td>
          <td>0.192271</td>
          <td>26.532626</td>
          <td>0.125714</td>
          <td>26.550086</td>
          <td>0.204706</td>
          <td>25.838653</td>
          <td>0.208420</td>
          <td>25.515203</td>
          <td>0.338737</td>
          <td>0.022677</td>
          <td>0.013466</td>
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
          <td>27.168292</td>
          <td>0.678698</td>
          <td>26.743573</td>
          <td>0.195860</td>
          <td>26.104178</td>
          <td>0.101646</td>
          <td>25.244377</td>
          <td>0.078189</td>
          <td>24.662759</td>
          <td>0.088718</td>
          <td>23.904286</td>
          <td>0.102727</td>
          <td>0.014384</td>
          <td>0.010763</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.039030</td>
          <td>0.545679</td>
          <td>26.616511</td>
          <td>0.158936</td>
          <td>26.146816</td>
          <td>0.171882</td>
          <td>25.487793</td>
          <td>0.181741</td>
          <td>25.267151</td>
          <td>0.324650</td>
          <td>0.037802</td>
          <td>0.030091</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.770567</td>
          <td>2.636019</td>
          <td>28.991805</td>
          <td>1.086249</td>
          <td>27.874734</td>
          <td>0.478898</td>
          <td>26.031535</td>
          <td>0.171435</td>
          <td>24.764616</td>
          <td>0.107072</td>
          <td>24.344927</td>
          <td>0.166142</td>
          <td>0.188568</td>
          <td>0.171464</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.839122</td>
          <td>0.954471</td>
          <td>27.326041</td>
          <td>0.296249</td>
          <td>26.535031</td>
          <td>0.246015</td>
          <td>25.834554</td>
          <td>0.250698</td>
          <td>25.399087</td>
          <td>0.371650</td>
          <td>0.125312</td>
          <td>0.087238</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.260168</td>
          <td>0.347114</td>
          <td>26.370766</td>
          <td>0.142630</td>
          <td>25.843400</td>
          <td>0.080834</td>
          <td>25.626909</td>
          <td>0.109410</td>
          <td>25.633646</td>
          <td>0.204841</td>
          <td>24.820770</td>
          <td>0.225041</td>
          <td>0.015132</td>
          <td>0.011637</td>
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
          <td>27.202702</td>
          <td>0.694634</td>
          <td>26.641763</td>
          <td>0.179669</td>
          <td>25.485751</td>
          <td>0.058877</td>
          <td>25.072911</td>
          <td>0.067159</td>
          <td>24.802509</td>
          <td>0.100250</td>
          <td>24.624082</td>
          <td>0.190785</td>
          <td>0.005822</td>
          <td>0.005495</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.799484</td>
          <td>0.207425</td>
          <td>26.066286</td>
          <td>0.099528</td>
          <td>25.252369</td>
          <td>0.079745</td>
          <td>24.943766</td>
          <td>0.114840</td>
          <td>24.285020</td>
          <td>0.144728</td>
          <td>0.061637</td>
          <td>0.061282</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.317900</td>
          <td>0.755422</td>
          <td>26.365109</td>
          <td>0.143291</td>
          <td>26.306702</td>
          <td>0.122582</td>
          <td>26.301527</td>
          <td>0.197349</td>
          <td>26.234234</td>
          <td>0.337725</td>
          <td>25.145828</td>
          <td>0.296651</td>
          <td>0.072844</td>
          <td>0.038162</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.179720</td>
          <td>0.683884</td>
          <td>26.163627</td>
          <td>0.119209</td>
          <td>25.933613</td>
          <td>0.087484</td>
          <td>25.993769</td>
          <td>0.150251</td>
          <td>25.418968</td>
          <td>0.170813</td>
          <td>25.212020</td>
          <td>0.309592</td>
          <td>0.008937</td>
          <td>0.006722</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.317605</td>
          <td>0.363240</td>
          <td>26.781198</td>
          <td>0.202257</td>
          <td>26.755546</td>
          <td>0.178426</td>
          <td>26.027912</td>
          <td>0.154865</td>
          <td>25.624888</td>
          <td>0.203451</td>
          <td>27.229983</td>
          <td>1.234457</td>
          <td>0.022677</td>
          <td>0.013466</td>
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
          <td>29.654288</td>
          <td>2.334388</td>
          <td>26.788667</td>
          <td>0.177587</td>
          <td>26.001968</td>
          <td>0.079158</td>
          <td>25.251969</td>
          <td>0.066539</td>
          <td>24.718875</td>
          <td>0.079401</td>
          <td>23.887458</td>
          <td>0.085803</td>
          <td>0.014384</td>
          <td>0.010763</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.912155</td>
          <td>2.574781</td>
          <td>27.077227</td>
          <td>0.228742</td>
          <td>26.745905</td>
          <td>0.153427</td>
          <td>26.566589</td>
          <td>0.210815</td>
          <td>25.687960</td>
          <td>0.186371</td>
          <td>25.771268</td>
          <td>0.419317</td>
          <td>0.037802</td>
          <td>0.030091</td>
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
          <td>25.848697</td>
          <td>0.154349</td>
          <td>25.134090</td>
          <td>0.155018</td>
          <td>24.049835</td>
          <td>0.135744</td>
          <td>0.188568</td>
          <td>0.171464</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.351135</td>
          <td>0.284934</td>
          <td>26.460356</td>
          <td>0.217147</td>
          <td>25.768080</td>
          <td>0.223352</td>
          <td>24.849755</td>
          <td>0.224524</td>
          <td>0.125312</td>
          <td>0.087238</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.132640</td>
          <td>0.282126</td>
          <td>26.109936</td>
          <td>0.098882</td>
          <td>25.938593</td>
          <td>0.074869</td>
          <td>25.670467</td>
          <td>0.096285</td>
          <td>25.451573</td>
          <td>0.150460</td>
          <td>25.049673</td>
          <td>0.232860</td>
          <td>0.015132</td>
          <td>0.011637</td>
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
          <td>27.787472</td>
          <td>0.931312</td>
          <td>26.387567</td>
          <td>0.125717</td>
          <td>25.510058</td>
          <td>0.051099</td>
          <td>25.026400</td>
          <td>0.054371</td>
          <td>24.774602</td>
          <td>0.083254</td>
          <td>24.761291</td>
          <td>0.182530</td>
          <td>0.005822</td>
          <td>0.005495</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.468804</td>
          <td>1.401195</td>
          <td>26.706938</td>
          <td>0.172327</td>
          <td>25.935716</td>
          <td>0.078244</td>
          <td>25.188808</td>
          <td>0.066098</td>
          <td>24.806016</td>
          <td>0.089837</td>
          <td>24.392091</td>
          <td>0.139802</td>
          <td>0.061637</td>
          <td>0.061282</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.293906</td>
          <td>0.329573</td>
          <td>26.629846</td>
          <td>0.160636</td>
          <td>26.366654</td>
          <td>0.113588</td>
          <td>26.239255</td>
          <td>0.164399</td>
          <td>26.510125</td>
          <td>0.373610</td>
          <td>25.374583</td>
          <td>0.315471</td>
          <td>0.072844</td>
          <td>0.038162</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.772245</td>
          <td>0.464300</td>
          <td>26.161468</td>
          <td>0.103297</td>
          <td>25.981091</td>
          <td>0.077607</td>
          <td>25.795908</td>
          <td>0.107281</td>
          <td>26.037310</td>
          <td>0.245988</td>
          <td>25.182545</td>
          <td>0.259368</td>
          <td>0.008937</td>
          <td>0.006722</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.542855</td>
          <td>0.390873</td>
          <td>26.454396</td>
          <td>0.133684</td>
          <td>26.543978</td>
          <td>0.127547</td>
          <td>26.395718</td>
          <td>0.180587</td>
          <td>25.826911</td>
          <td>0.207306</td>
          <td>25.430761</td>
          <td>0.318166</td>
          <td>0.022677</td>
          <td>0.013466</td>
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
