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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fe45076b520>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>27.994889</td>
          <td>1.055277</td>
          <td>26.712337</td>
          <td>0.166129</td>
          <td>25.984105</td>
          <td>0.077748</td>
          <td>25.294888</td>
          <td>0.068957</td>
          <td>25.031232</td>
          <td>0.104256</td>
          <td>25.144428</td>
          <td>0.251180</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.820855</td>
          <td>1.636699</td>
          <td>28.641101</td>
          <td>0.738965</td>
          <td>27.279263</td>
          <td>0.236927</td>
          <td>27.289573</td>
          <td>0.372851</td>
          <td>27.453797</td>
          <td>0.717188</td>
          <td>26.050360</td>
          <td>0.509779</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.245686</td>
          <td>0.652817</td>
          <td>25.952784</td>
          <td>0.085960</td>
          <td>24.793651</td>
          <td>0.027127</td>
          <td>23.888058</td>
          <td>0.020027</td>
          <td>23.148455</td>
          <td>0.019925</td>
          <td>22.837475</td>
          <td>0.033781</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.687173</td>
          <td>1.534095</td>
          <td>27.624503</td>
          <td>0.351670</td>
          <td>27.481774</td>
          <td>0.279673</td>
          <td>26.185928</td>
          <td>0.150281</td>
          <td>26.553921</td>
          <td>0.372045</td>
          <td>25.037213</td>
          <td>0.229911</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.212603</td>
          <td>0.300435</td>
          <td>25.843927</td>
          <td>0.078103</td>
          <td>25.471767</td>
          <td>0.049369</td>
          <td>24.780455</td>
          <td>0.043685</td>
          <td>24.347502</td>
          <td>0.057023</td>
          <td>23.541658</td>
          <td>0.063060</td>
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
          <td>2.147172</td>
          <td>26.920484</td>
          <td>0.517886</td>
          <td>26.170336</td>
          <td>0.104024</td>
          <td>26.162952</td>
          <td>0.091020</td>
          <td>25.969299</td>
          <td>0.124654</td>
          <td>25.746257</td>
          <td>0.192861</td>
          <td>26.555254</td>
          <td>0.727243</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.861565</td>
          <td>0.495935</td>
          <td>26.852054</td>
          <td>0.187027</td>
          <td>26.785675</td>
          <td>0.156347</td>
          <td>26.283557</td>
          <td>0.163378</td>
          <td>25.783236</td>
          <td>0.198956</td>
          <td>25.540546</td>
          <td>0.345583</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.951807</td>
          <td>0.529862</td>
          <td>27.642489</td>
          <td>0.356672</td>
          <td>26.738515</td>
          <td>0.150153</td>
          <td>26.497679</td>
          <td>0.195891</td>
          <td>26.231483</td>
          <td>0.287989</td>
          <td>25.453141</td>
          <td>0.322460</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.446636</td>
          <td>0.748176</td>
          <td>27.578204</td>
          <td>0.339069</td>
          <td>26.530306</td>
          <td>0.125461</td>
          <td>25.811192</td>
          <td>0.108627</td>
          <td>25.420853</td>
          <td>0.146188</td>
          <td>25.632284</td>
          <td>0.371353</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.829844</td>
          <td>0.484424</td>
          <td>26.742502</td>
          <td>0.170449</td>
          <td>26.057097</td>
          <td>0.082920</td>
          <td>25.791473</td>
          <td>0.106772</td>
          <td>25.197142</td>
          <td>0.120482</td>
          <td>24.768345</td>
          <td>0.183542</td>
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
          <td>0.890625</td>
          <td>30.564097</td>
          <td>3.294831</td>
          <td>26.547035</td>
          <td>0.165772</td>
          <td>25.987125</td>
          <td>0.091682</td>
          <td>25.295100</td>
          <td>0.081723</td>
          <td>25.016884</td>
          <td>0.120854</td>
          <td>24.886465</td>
          <td>0.237493</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.933855</td>
          <td>2.537960</td>
          <td>27.587322</td>
          <td>0.352460</td>
          <td>27.886449</td>
          <td>0.666114</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.171612</td>
          <td>0.328536</td>
          <td>25.870084</td>
          <td>0.094127</td>
          <td>24.830254</td>
          <td>0.033695</td>
          <td>23.842718</td>
          <td>0.023250</td>
          <td>23.160294</td>
          <td>0.024111</td>
          <td>22.808796</td>
          <td>0.039904</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.987654</td>
          <td>0.623677</td>
          <td>29.131889</td>
          <td>1.154722</td>
          <td>27.146326</td>
          <td>0.263046</td>
          <td>26.474390</td>
          <td>0.240722</td>
          <td>27.032524</td>
          <td>0.642136</td>
          <td>25.280634</td>
          <td>0.347860</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>28.333316</td>
          <td>1.371796</td>
          <td>25.814252</td>
          <td>0.087862</td>
          <td>25.439607</td>
          <td>0.056530</td>
          <td>24.784251</td>
          <td>0.052007</td>
          <td>24.412492</td>
          <td>0.071131</td>
          <td>23.663849</td>
          <td>0.083159</td>
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
          <td>2.147172</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.363281</td>
          <td>0.144298</td>
          <td>25.893662</td>
          <td>0.086244</td>
          <td>26.131657</td>
          <td>0.172605</td>
          <td>26.174162</td>
          <td>0.324834</td>
          <td>25.202428</td>
          <td>0.313262</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.843872</td>
          <td>0.541206</td>
          <td>26.854530</td>
          <td>0.215603</td>
          <td>26.396941</td>
          <td>0.131619</td>
          <td>26.213274</td>
          <td>0.181894</td>
          <td>25.650456</td>
          <td>0.208454</td>
          <td>24.995306</td>
          <td>0.260765</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.956691</td>
          <td>0.236412</td>
          <td>26.621306</td>
          <td>0.160969</td>
          <td>26.135116</td>
          <td>0.171696</td>
          <td>25.614402</td>
          <td>0.203920</td>
          <td>25.881231</td>
          <td>0.523272</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.063583</td>
          <td>0.262215</td>
          <td>26.391312</td>
          <td>0.134540</td>
          <td>25.918603</td>
          <td>0.145378</td>
          <td>25.717031</td>
          <td>0.226112</td>
          <td>25.192691</td>
          <td>0.313871</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.348572</td>
          <td>0.770328</td>
          <td>26.401256</td>
          <td>0.147642</td>
          <td>26.179366</td>
          <td>0.109583</td>
          <td>25.754005</td>
          <td>0.123396</td>
          <td>25.245670</td>
          <td>0.148718</td>
          <td>24.799640</td>
          <td>0.223164</td>
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
          <td>0.890625</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.584800</td>
          <td>0.148984</td>
          <td>25.989350</td>
          <td>0.078119</td>
          <td>25.346264</td>
          <td>0.072175</td>
          <td>25.036644</td>
          <td>0.104764</td>
          <td>24.993711</td>
          <td>0.221781</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.993815</td>
          <td>1.055073</td>
          <td>27.451461</td>
          <td>0.306760</td>
          <td>27.315339</td>
          <td>0.244304</td>
          <td>27.395106</td>
          <td>0.404930</td>
          <td>26.107903</td>
          <td>0.260683</td>
          <td>26.112026</td>
          <td>0.533723</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.213958</td>
          <td>0.317232</td>
          <td>26.024220</td>
          <td>0.098354</td>
          <td>24.781654</td>
          <td>0.029137</td>
          <td>23.893787</td>
          <td>0.021882</td>
          <td>23.139266</td>
          <td>0.021414</td>
          <td>22.868471</td>
          <td>0.037822</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.217972</td>
          <td>0.728715</td>
          <td>28.514574</td>
          <td>0.790255</td>
          <td>27.256031</td>
          <td>0.286633</td>
          <td>27.173083</td>
          <td>0.418516</td>
          <td>25.913118</td>
          <td>0.273760</td>
          <td>25.116171</td>
          <td>0.304215</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.367115</td>
          <td>0.340064</td>
          <td>25.808977</td>
          <td>0.075826</td>
          <td>25.482101</td>
          <td>0.049896</td>
          <td>24.792404</td>
          <td>0.044217</td>
          <td>24.319372</td>
          <td>0.055697</td>
          <td>23.700222</td>
          <td>0.072674</td>
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
          <td>2.147172</td>
          <td>25.995711</td>
          <td>0.265403</td>
          <td>26.330458</td>
          <td>0.128028</td>
          <td>26.118217</td>
          <td>0.094685</td>
          <td>26.170366</td>
          <td>0.160685</td>
          <td>25.921411</td>
          <td>0.240494</td>
          <td>25.244628</td>
          <td>0.293905</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.511569</td>
          <td>0.384301</td>
          <td>26.545844</td>
          <td>0.146100</td>
          <td>27.113531</td>
          <td>0.209667</td>
          <td>26.308693</td>
          <td>0.169737</td>
          <td>25.645204</td>
          <td>0.179901</td>
          <td>26.227760</td>
          <td>0.587752</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.690528</td>
          <td>0.449487</td>
          <td>27.123032</td>
          <td>0.244172</td>
          <td>26.557847</td>
          <td>0.134833</td>
          <td>26.713621</td>
          <td>0.246138</td>
          <td>25.649510</td>
          <td>0.186281</td>
          <td>25.918967</td>
          <td>0.482760</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.256065</td>
          <td>0.700938</td>
          <td>27.339341</td>
          <td>0.307183</td>
          <td>26.722484</td>
          <td>0.165601</td>
          <td>26.194903</td>
          <td>0.170149</td>
          <td>26.085020</td>
          <td>0.284247</td>
          <td>25.212504</td>
          <td>0.296302</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.333260</td>
          <td>0.707970</td>
          <td>26.686380</td>
          <td>0.167929</td>
          <td>26.017646</td>
          <td>0.083278</td>
          <td>25.742519</td>
          <td>0.106539</td>
          <td>25.390685</td>
          <td>0.147968</td>
          <td>25.073604</td>
          <td>0.246095</td>
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
