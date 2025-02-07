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

    <pzflow.flow.Flow at 0x7fa9db863df0>



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
          <td>27.185247</td>
          <td>0.625944</td>
          <td>27.352873</td>
          <td>0.283126</td>
          <td>25.853769</td>
          <td>0.069284</td>
          <td>25.064297</td>
          <td>0.056205</td>
          <td>24.763360</td>
          <td>0.082397</td>
          <td>23.960560</td>
          <td>0.091294</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.471803</td>
          <td>0.311564</td>
          <td>26.591081</td>
          <td>0.132241</td>
          <td>26.169586</td>
          <td>0.148187</td>
          <td>25.657649</td>
          <td>0.178946</td>
          <td>25.462876</td>
          <td>0.324968</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.951332</td>
          <td>1.739583</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.367716</td>
          <td>0.552954</td>
          <td>26.034476</td>
          <td>0.131895</td>
          <td>24.959962</td>
          <td>0.097948</td>
          <td>24.408040</td>
          <td>0.134865</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>29.445392</td>
          <td>2.150528</td>
          <td>28.066790</td>
          <td>0.492978</td>
          <td>27.463907</td>
          <td>0.275645</td>
          <td>26.171618</td>
          <td>0.148446</td>
          <td>25.707814</td>
          <td>0.186706</td>
          <td>25.069471</td>
          <td>0.236134</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.444233</td>
          <td>0.360989</td>
          <td>26.213051</td>
          <td>0.107977</td>
          <td>25.925998</td>
          <td>0.073856</td>
          <td>25.593337</td>
          <td>0.089744</td>
          <td>25.148646</td>
          <td>0.115506</td>
          <td>24.944887</td>
          <td>0.212908</td>
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
          <td>26.237606</td>
          <td>0.110314</td>
          <td>25.419217</td>
          <td>0.047118</td>
          <td>25.012748</td>
          <td>0.053691</td>
          <td>24.747865</td>
          <td>0.081279</td>
          <td>25.036837</td>
          <td>0.229839</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.174833</td>
          <td>0.621397</td>
          <td>26.561096</td>
          <td>0.145967</td>
          <td>25.913222</td>
          <td>0.073026</td>
          <td>25.380250</td>
          <td>0.074367</td>
          <td>24.905439</td>
          <td>0.093372</td>
          <td>24.162785</td>
          <td>0.108990</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.085928</td>
          <td>0.271209</td>
          <td>26.580070</td>
          <td>0.148364</td>
          <td>26.431150</td>
          <td>0.115102</td>
          <td>26.193275</td>
          <td>0.151231</td>
          <td>26.370250</td>
          <td>0.321912</td>
          <td>26.153335</td>
          <td>0.549502</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.584059</td>
          <td>0.402328</td>
          <td>26.313027</td>
          <td>0.117799</td>
          <td>26.291209</td>
          <td>0.101861</td>
          <td>25.675523</td>
          <td>0.096463</td>
          <td>25.537665</td>
          <td>0.161579</td>
          <td>25.563154</td>
          <td>0.351789</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.347897</td>
          <td>0.334643</td>
          <td>26.571879</td>
          <td>0.147325</td>
          <td>26.452421</td>
          <td>0.117253</td>
          <td>26.319276</td>
          <td>0.168429</td>
          <td>25.700735</td>
          <td>0.185593</td>
          <td>26.690808</td>
          <td>0.795519</td>
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
          <td>27.976863</td>
          <td>1.128403</td>
          <td>26.827728</td>
          <td>0.210082</td>
          <td>26.015649</td>
          <td>0.094007</td>
          <td>25.184771</td>
          <td>0.074139</td>
          <td>24.681183</td>
          <td>0.090120</td>
          <td>24.184270</td>
          <td>0.130996</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.014646</td>
          <td>0.245354</td>
          <td>26.709298</td>
          <td>0.171399</td>
          <td>26.102971</td>
          <td>0.164966</td>
          <td>25.621844</td>
          <td>0.202750</td>
          <td>27.469162</td>
          <td>1.401673</td>
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
          <td>28.768689</td>
          <td>0.838451</td>
          <td>25.766670</td>
          <td>0.126331</td>
          <td>25.050748</td>
          <td>0.127227</td>
          <td>24.159864</td>
          <td>0.131187</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>26.883286</td>
          <td>0.579346</td>
          <td>28.604734</td>
          <td>0.839782</td>
          <td>27.578535</td>
          <td>0.371538</td>
          <td>26.199653</td>
          <td>0.191415</td>
          <td>25.793220</td>
          <td>0.249021</td>
          <td>25.263849</td>
          <td>0.343288</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.850426</td>
          <td>0.542472</td>
          <td>26.162368</td>
          <td>0.119094</td>
          <td>25.943765</td>
          <td>0.088282</td>
          <td>25.508335</td>
          <td>0.098606</td>
          <td>25.688352</td>
          <td>0.214376</td>
          <td>25.198357</td>
          <td>0.306262</td>
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
          <td>29.463901</td>
          <td>2.298661</td>
          <td>26.383878</td>
          <td>0.146873</td>
          <td>25.618441</td>
          <td>0.067638</td>
          <td>24.969522</td>
          <td>0.062637</td>
          <td>25.036788</td>
          <td>0.125535</td>
          <td>24.929563</td>
          <td>0.251101</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.880633</td>
          <td>0.220341</td>
          <td>25.949817</td>
          <td>0.089096</td>
          <td>25.237267</td>
          <td>0.077996</td>
          <td>24.820433</td>
          <td>0.102251</td>
          <td>24.112082</td>
          <td>0.123577</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.461985</td>
          <td>0.830667</td>
          <td>26.801581</td>
          <td>0.207809</td>
          <td>26.480438</td>
          <td>0.142653</td>
          <td>26.069360</td>
          <td>0.162343</td>
          <td>25.595949</td>
          <td>0.200788</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>27.007539</td>
          <td>0.618580</td>
          <td>26.195703</td>
          <td>0.126040</td>
          <td>26.033389</td>
          <td>0.098527</td>
          <td>25.954641</td>
          <td>0.149948</td>
          <td>25.823569</td>
          <td>0.246927</td>
          <td>25.570455</td>
          <td>0.421666</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.039098</td>
          <td>0.293003</td>
          <td>26.795998</td>
          <td>0.206352</td>
          <td>26.683933</td>
          <td>0.169345</td>
          <td>26.368204</td>
          <td>0.208438</td>
          <td>25.931037</td>
          <td>0.264346</td>
          <td>25.554458</td>
          <td>0.408660</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>27.248508</td>
          <td>0.260099</td>
          <td>26.105602</td>
          <td>0.086552</td>
          <td>25.095319</td>
          <td>0.057782</td>
          <td>24.758773</td>
          <td>0.082075</td>
          <td>23.811542</td>
          <td>0.080076</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.578994</td>
          <td>0.339531</td>
          <td>26.501296</td>
          <td>0.122456</td>
          <td>26.062211</td>
          <td>0.135227</td>
          <td>25.423313</td>
          <td>0.146634</td>
          <td>24.895926</td>
          <td>0.204554</td>
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
          <td>28.480574</td>
          <td>0.639841</td>
          <td>25.983097</td>
          <td>0.137241</td>
          <td>24.940091</td>
          <td>0.104438</td>
          <td>24.113605</td>
          <td>0.113583</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.011870</td>
          <td>0.258296</td>
          <td>27.848335</td>
          <td>0.455404</td>
          <td>26.795851</td>
          <td>0.311545</td>
          <td>25.374839</td>
          <td>0.174921</td>
          <td>24.752954</td>
          <td>0.226113</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.351214</td>
          <td>0.335819</td>
          <td>26.165274</td>
          <td>0.103692</td>
          <td>25.825332</td>
          <td>0.067658</td>
          <td>25.770357</td>
          <td>0.104974</td>
          <td>25.857002</td>
          <td>0.211928</td>
          <td>24.993887</td>
          <td>0.222095</td>
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
          <td>29.514883</td>
          <td>2.266085</td>
          <td>26.394621</td>
          <td>0.135327</td>
          <td>25.469887</td>
          <td>0.053379</td>
          <td>25.045395</td>
          <td>0.060075</td>
          <td>24.820713</td>
          <td>0.093771</td>
          <td>24.869266</td>
          <td>0.215987</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.829592</td>
          <td>0.489047</td>
          <td>26.664947</td>
          <td>0.161785</td>
          <td>25.982303</td>
          <td>0.078922</td>
          <td>25.146796</td>
          <td>0.061541</td>
          <td>24.858675</td>
          <td>0.091101</td>
          <td>24.173970</td>
          <td>0.111946</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>28.072087</td>
          <td>1.129038</td>
          <td>26.683839</td>
          <td>0.169003</td>
          <td>26.398365</td>
          <td>0.117421</td>
          <td>26.152367</td>
          <td>0.153506</td>
          <td>25.491433</td>
          <td>0.162878</td>
          <td>26.908430</td>
          <td>0.947146</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.313780</td>
          <td>0.350699</td>
          <td>26.047838</td>
          <td>0.103347</td>
          <td>26.193377</td>
          <td>0.104823</td>
          <td>25.726883</td>
          <td>0.113677</td>
          <td>25.614597</td>
          <td>0.192647</td>
          <td>24.779479</td>
          <td>0.207571</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.336672</td>
          <td>0.709604</td>
          <td>27.007868</td>
          <td>0.220133</td>
          <td>26.518304</td>
          <td>0.129022</td>
          <td>26.406646</td>
          <td>0.188661</td>
          <td>25.829170</td>
          <td>0.214560</td>
          <td>25.781068</td>
          <td>0.431401</td>
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
