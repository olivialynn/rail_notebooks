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

    <pzflow.flow.Flow at 0x7fe301255e40>



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
          <td>27.939640</td>
          <td>1.021277</td>
          <td>26.891613</td>
          <td>0.193370</td>
          <td>26.006050</td>
          <td>0.079269</td>
          <td>25.180242</td>
          <td>0.062295</td>
          <td>24.642945</td>
          <td>0.074087</td>
          <td>24.043102</td>
          <td>0.098155</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.192187</td>
          <td>0.248334</td>
          <td>26.697924</td>
          <td>0.145007</td>
          <td>26.262176</td>
          <td>0.160421</td>
          <td>25.880862</td>
          <td>0.215901</td>
          <td>25.600877</td>
          <td>0.362354</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>26.496116</td>
          <td>0.375891</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.121915</td>
          <td>0.461452</td>
          <td>26.089449</td>
          <td>0.138309</td>
          <td>25.054488</td>
          <td>0.106397</td>
          <td>24.332941</td>
          <td>0.126380</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.710331</td>
          <td>0.773684</td>
          <td>27.371566</td>
          <td>0.255632</td>
          <td>26.249235</td>
          <td>0.158657</td>
          <td>25.395664</td>
          <td>0.143055</td>
          <td>25.276908</td>
          <td>0.279868</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.791145</td>
          <td>0.212748</td>
          <td>26.133025</td>
          <td>0.100685</td>
          <td>25.973485</td>
          <td>0.077022</td>
          <td>25.708921</td>
          <td>0.099330</td>
          <td>25.658327</td>
          <td>0.179049</td>
          <td>25.335387</td>
          <td>0.293423</td>
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
          <td>26.663979</td>
          <td>0.427671</td>
          <td>26.401351</td>
          <td>0.127179</td>
          <td>25.443783</td>
          <td>0.048157</td>
          <td>25.036092</td>
          <td>0.054815</td>
          <td>24.874747</td>
          <td>0.090887</td>
          <td>24.765714</td>
          <td>0.183134</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.813377</td>
          <td>0.946099</td>
          <td>26.621403</td>
          <td>0.153716</td>
          <td>26.068775</td>
          <td>0.083778</td>
          <td>25.251054</td>
          <td>0.066331</td>
          <td>24.727976</td>
          <td>0.079865</td>
          <td>24.047616</td>
          <td>0.098544</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.370887</td>
          <td>0.340776</td>
          <td>26.552451</td>
          <td>0.144886</td>
          <td>26.291049</td>
          <td>0.101846</td>
          <td>26.364726</td>
          <td>0.175066</td>
          <td>25.935599</td>
          <td>0.225965</td>
          <td>25.336683</td>
          <td>0.293730</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.300356</td>
          <td>0.322259</td>
          <td>26.118840</td>
          <td>0.099443</td>
          <td>26.208827</td>
          <td>0.094763</td>
          <td>26.187244</td>
          <td>0.150451</td>
          <td>25.651570</td>
          <td>0.178026</td>
          <td>25.033821</td>
          <td>0.229265</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.140227</td>
          <td>1.147853</td>
          <td>26.955921</td>
          <td>0.204102</td>
          <td>26.672871</td>
          <td>0.141913</td>
          <td>26.343754</td>
          <td>0.171974</td>
          <td>26.436943</td>
          <td>0.339405</td>
          <td>25.956738</td>
          <td>0.475652</td>
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
          <td>27.673431</td>
          <td>0.942310</td>
          <td>27.130339</td>
          <td>0.269683</td>
          <td>26.035371</td>
          <td>0.095649</td>
          <td>25.168812</td>
          <td>0.073101</td>
          <td>24.653316</td>
          <td>0.087938</td>
          <td>24.180311</td>
          <td>0.130548</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>32.247133</td>
          <td>4.936078</td>
          <td>27.803406</td>
          <td>0.457347</td>
          <td>26.558520</td>
          <td>0.150687</td>
          <td>26.284867</td>
          <td>0.192470</td>
          <td>25.512130</td>
          <td>0.184857</td>
          <td>27.142255</td>
          <td>1.174926</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.000066</td>
          <td>0.537801</td>
          <td>27.988226</td>
          <td>0.488138</td>
          <td>26.155936</td>
          <td>0.176429</td>
          <td>24.995739</td>
          <td>0.121299</td>
          <td>24.217912</td>
          <td>0.137931</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.205107</td>
          <td>0.723993</td>
          <td>27.923810</td>
          <td>0.526223</td>
          <td>27.618151</td>
          <td>0.383160</td>
          <td>26.512906</td>
          <td>0.248480</td>
          <td>25.376741</td>
          <td>0.175809</td>
          <td>25.177068</td>
          <td>0.320461</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.174633</td>
          <td>0.324359</td>
          <td>26.104356</td>
          <td>0.113237</td>
          <td>25.793514</td>
          <td>0.077332</td>
          <td>25.683584</td>
          <td>0.114920</td>
          <td>25.618740</td>
          <td>0.202245</td>
          <td>25.183685</td>
          <td>0.302677</td>
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
          <td>27.527476</td>
          <td>0.870084</td>
          <td>26.454947</td>
          <td>0.156094</td>
          <td>25.502873</td>
          <td>0.061057</td>
          <td>25.161228</td>
          <td>0.074220</td>
          <td>24.858915</td>
          <td>0.107535</td>
          <td>24.620457</td>
          <td>0.194164</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.003493</td>
          <td>1.148031</td>
          <td>26.632725</td>
          <td>0.178936</td>
          <td>25.849599</td>
          <td>0.081570</td>
          <td>25.074741</td>
          <td>0.067557</td>
          <td>24.755313</td>
          <td>0.096582</td>
          <td>24.184765</td>
          <td>0.131607</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.238139</td>
          <td>0.297419</td>
          <td>26.291106</td>
          <td>0.121109</td>
          <td>26.836633</td>
          <td>0.306871</td>
          <td>26.546747</td>
          <td>0.430935</td>
          <td>25.091094</td>
          <td>0.284222</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.692957</td>
          <td>0.493104</td>
          <td>26.068651</td>
          <td>0.112878</td>
          <td>25.926081</td>
          <td>0.089671</td>
          <td>25.741673</td>
          <td>0.124779</td>
          <td>25.682212</td>
          <td>0.219660</td>
          <td>25.966630</td>
          <td>0.565488</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.064476</td>
          <td>0.299044</td>
          <td>26.766899</td>
          <td>0.201382</td>
          <td>26.775309</td>
          <td>0.182998</td>
          <td>26.139049</td>
          <td>0.171798</td>
          <td>26.274730</td>
          <td>0.348292</td>
          <td>26.099684</td>
          <td>0.610660</td>
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
          <td>26.474855</td>
          <td>0.135536</td>
          <td>25.997518</td>
          <td>0.078684</td>
          <td>25.152580</td>
          <td>0.060794</td>
          <td>24.677539</td>
          <td>0.076397</td>
          <td>24.129201</td>
          <td>0.105853</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.400514</td>
          <td>0.294455</td>
          <td>26.568836</td>
          <td>0.129841</td>
          <td>26.106203</td>
          <td>0.140459</td>
          <td>25.603278</td>
          <td>0.171029</td>
          <td>24.896329</td>
          <td>0.204623</td>
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
          <td>28.455419</td>
          <td>0.628715</td>
          <td>25.858243</td>
          <td>0.123184</td>
          <td>24.904641</td>
          <td>0.101248</td>
          <td>24.297954</td>
          <td>0.133289</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>34.042486</td>
          <td>5.607837</td>
          <td>27.090549</td>
          <td>0.250461</td>
          <td>26.359142</td>
          <td>0.218025</td>
          <td>25.400399</td>
          <td>0.178755</td>
          <td>24.762179</td>
          <td>0.227851</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.973458</td>
          <td>0.247621</td>
          <td>26.403333</td>
          <td>0.127552</td>
          <td>25.883974</td>
          <td>0.071263</td>
          <td>25.715987</td>
          <td>0.100095</td>
          <td>25.359743</td>
          <td>0.138888</td>
          <td>25.076157</td>
          <td>0.237772</td>
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
          <td>26.061155</td>
          <td>0.101283</td>
          <td>25.458302</td>
          <td>0.052833</td>
          <td>25.040278</td>
          <td>0.059803</td>
          <td>24.648195</td>
          <td>0.080562</td>
          <td>25.127613</td>
          <td>0.267299</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.424322</td>
          <td>0.743578</td>
          <td>27.034477</td>
          <td>0.220925</td>
          <td>26.234362</td>
          <td>0.098518</td>
          <td>25.159822</td>
          <td>0.062256</td>
          <td>24.724806</td>
          <td>0.080971</td>
          <td>24.226248</td>
          <td>0.117161</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.394086</td>
          <td>0.357913</td>
          <td>26.826689</td>
          <td>0.190734</td>
          <td>26.352551</td>
          <td>0.112828</td>
          <td>26.258122</td>
          <td>0.168022</td>
          <td>26.159258</td>
          <td>0.284138</td>
          <td>25.922969</td>
          <td>0.484197</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.295454</td>
          <td>0.345681</td>
          <td>26.582682</td>
          <td>0.164051</td>
          <td>26.164422</td>
          <td>0.102202</td>
          <td>25.750387</td>
          <td>0.116027</td>
          <td>26.028802</td>
          <td>0.271566</td>
          <td>25.491814</td>
          <td>0.369766</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.912270</td>
          <td>0.526585</td>
          <td>26.908200</td>
          <td>0.202547</td>
          <td>26.594323</td>
          <td>0.137784</td>
          <td>26.351918</td>
          <td>0.180129</td>
          <td>25.578236</td>
          <td>0.173683</td>
          <td>25.597718</td>
          <td>0.374656</td>
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
