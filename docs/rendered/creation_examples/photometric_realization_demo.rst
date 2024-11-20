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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f1e9c70b4f0>



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
          <td>26.684333</td>
          <td>0.434331</td>
          <td>26.525268</td>
          <td>0.141538</td>
          <td>25.891294</td>
          <td>0.071624</td>
          <td>25.458514</td>
          <td>0.079691</td>
          <td>25.021518</td>
          <td>0.103374</td>
          <td>24.540635</td>
          <td>0.151172</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.588951</td>
          <td>0.403843</td>
          <td>27.674316</td>
          <td>0.365669</td>
          <td>28.112549</td>
          <td>0.458219</td>
          <td>27.628687</td>
          <td>0.482727</td>
          <td>26.165529</td>
          <td>0.272991</td>
          <td>28.429961</td>
          <td>2.008919</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.758049</td>
          <td>0.914278</td>
          <td>25.930116</td>
          <td>0.084263</td>
          <td>24.814800</td>
          <td>0.027633</td>
          <td>23.886707</td>
          <td>0.020004</td>
          <td>23.158821</td>
          <td>0.020101</td>
          <td>22.828035</td>
          <td>0.033501</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.392251</td>
          <td>0.623257</td>
          <td>27.276403</td>
          <td>0.236368</td>
          <td>26.790715</td>
          <td>0.249975</td>
          <td>26.091599</td>
          <td>0.257000</td>
          <td>25.578466</td>
          <td>0.356045</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.290505</td>
          <td>0.319742</td>
          <td>25.952323</td>
          <td>0.085925</td>
          <td>25.496121</td>
          <td>0.050448</td>
          <td>24.737524</td>
          <td>0.042052</td>
          <td>24.408981</td>
          <td>0.060220</td>
          <td>23.619712</td>
          <td>0.067575</td>
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
          <td>25.911614</td>
          <td>0.235110</td>
          <td>26.377157</td>
          <td>0.124541</td>
          <td>26.112598</td>
          <td>0.087076</td>
          <td>26.026376</td>
          <td>0.130974</td>
          <td>25.842586</td>
          <td>0.209107</td>
          <td>25.312907</td>
          <td>0.288146</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.621724</td>
          <td>0.838832</td>
          <td>27.357968</td>
          <td>0.284296</td>
          <td>26.921516</td>
          <td>0.175543</td>
          <td>26.712304</td>
          <td>0.234322</td>
          <td>25.738637</td>
          <td>0.191626</td>
          <td>25.241286</td>
          <td>0.271884</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.418281</td>
          <td>0.734158</td>
          <td>27.420675</td>
          <td>0.299047</td>
          <td>26.672138</td>
          <td>0.141824</td>
          <td>26.250204</td>
          <td>0.158788</td>
          <td>25.861538</td>
          <td>0.212447</td>
          <td>25.809022</td>
          <td>0.425543</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.250621</td>
          <td>0.260521</td>
          <td>26.596843</td>
          <td>0.132902</td>
          <td>25.910377</td>
          <td>0.118435</td>
          <td>25.615363</td>
          <td>0.172637</td>
          <td>25.654125</td>
          <td>0.377721</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.004601</td>
          <td>1.061322</td>
          <td>26.500007</td>
          <td>0.138492</td>
          <td>26.132104</td>
          <td>0.088583</td>
          <td>25.627631</td>
          <td>0.092491</td>
          <td>25.143913</td>
          <td>0.115031</td>
          <td>24.605166</td>
          <td>0.159761</td>
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
          <td>28.410095</td>
          <td>1.427196</td>
          <td>26.850526</td>
          <td>0.214119</td>
          <td>25.934488</td>
          <td>0.087536</td>
          <td>25.330555</td>
          <td>0.084316</td>
          <td>24.797429</td>
          <td>0.099797</td>
          <td>24.738737</td>
          <td>0.210050</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.832068</td>
          <td>0.535259</td>
          <td>27.928545</td>
          <td>0.501978</td>
          <td>27.696357</td>
          <td>0.383776</td>
          <td>27.048001</td>
          <td>0.358624</td>
          <td>26.018130</td>
          <td>0.281188</td>
          <td>25.605810</td>
          <td>0.421313</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.903012</td>
          <td>0.264709</td>
          <td>26.030622</td>
          <td>0.108309</td>
          <td>24.754265</td>
          <td>0.031516</td>
          <td>23.900316</td>
          <td>0.024434</td>
          <td>23.153536</td>
          <td>0.023970</td>
          <td>22.827143</td>
          <td>0.040557</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.906598</td>
          <td>1.121076</td>
          <td>30.138061</td>
          <td>1.906812</td>
          <td>27.374415</td>
          <td>0.316269</td>
          <td>26.190639</td>
          <td>0.189966</td>
          <td>26.233770</td>
          <td>0.354911</td>
          <td>25.821497</td>
          <td>0.524676</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.183075</td>
          <td>0.326541</td>
          <td>25.715100</td>
          <td>0.080525</td>
          <td>25.506291</td>
          <td>0.059974</td>
          <td>24.895477</td>
          <td>0.057401</td>
          <td>24.385873</td>
          <td>0.069475</td>
          <td>23.784231</td>
          <td>0.092448</td>
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
          <td>26.738605</td>
          <td>0.506564</td>
          <td>26.442144</td>
          <td>0.154394</td>
          <td>26.052873</td>
          <td>0.099186</td>
          <td>26.043403</td>
          <td>0.160100</td>
          <td>26.206176</td>
          <td>0.333196</td>
          <td>25.943499</td>
          <td>0.551305</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.717124</td>
          <td>0.493259</td>
          <td>27.245016</td>
          <td>0.296947</td>
          <td>26.598525</td>
          <td>0.156546</td>
          <td>26.611307</td>
          <td>0.253510</td>
          <td>25.801426</td>
          <td>0.236343</td>
          <td>25.132917</td>
          <td>0.291611</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.686766</td>
          <td>0.188705</td>
          <td>27.072593</td>
          <td>0.235303</td>
          <td>26.875316</td>
          <td>0.316516</td>
          <td>25.920192</td>
          <td>0.262684</td>
          <td>26.698567</td>
          <td>0.911042</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.959430</td>
          <td>1.134803</td>
          <td>28.072357</td>
          <td>0.570417</td>
          <td>26.611321</td>
          <td>0.162519</td>
          <td>25.979934</td>
          <td>0.153236</td>
          <td>25.535953</td>
          <td>0.194343</td>
          <td>25.357062</td>
          <td>0.357495</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.344569</td>
          <td>0.140622</td>
          <td>26.202270</td>
          <td>0.111794</td>
          <td>25.689886</td>
          <td>0.116709</td>
          <td>25.678758</td>
          <td>0.214636</td>
          <td>25.186736</td>
          <td>0.306197</td>
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
          <td>26.259293</td>
          <td>0.311907</td>
          <td>26.794644</td>
          <td>0.178182</td>
          <td>26.001141</td>
          <td>0.078936</td>
          <td>25.207328</td>
          <td>0.063819</td>
          <td>25.047697</td>
          <td>0.105782</td>
          <td>24.925503</td>
          <td>0.209514</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.161671</td>
          <td>1.027414</td>
          <td>27.256636</td>
          <td>0.232741</td>
          <td>26.990203</td>
          <td>0.294327</td>
          <td>27.345031</td>
          <td>0.666470</td>
          <td>26.154813</td>
          <td>0.550534</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.359634</td>
          <td>0.355947</td>
          <td>25.978710</td>
          <td>0.094510</td>
          <td>24.758670</td>
          <td>0.028557</td>
          <td>23.914356</td>
          <td>0.022271</td>
          <td>23.135377</td>
          <td>0.021343</td>
          <td>22.824416</td>
          <td>0.036377</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.698936</td>
          <td>2.385953</td>
          <td>27.820200</td>
          <td>0.445853</td>
          <td>26.412683</td>
          <td>0.227949</td>
          <td>26.133068</td>
          <td>0.326724</td>
          <td>26.206092</td>
          <td>0.686509</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.370905</td>
          <td>0.341082</td>
          <td>25.801257</td>
          <td>0.075311</td>
          <td>25.493870</td>
          <td>0.050420</td>
          <td>24.847419</td>
          <td>0.046430</td>
          <td>24.444754</td>
          <td>0.062251</td>
          <td>23.701324</td>
          <td>0.072745</td>
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
          <td>26.036397</td>
          <td>0.274332</td>
          <td>26.132556</td>
          <td>0.107801</td>
          <td>26.042755</td>
          <td>0.088610</td>
          <td>26.174372</td>
          <td>0.161236</td>
          <td>26.185397</td>
          <td>0.298238</td>
          <td>26.563797</td>
          <td>0.777863</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.623736</td>
          <td>0.418910</td>
          <td>27.100781</td>
          <td>0.233418</td>
          <td>27.396015</td>
          <td>0.264826</td>
          <td>26.787123</td>
          <td>0.253296</td>
          <td>26.400985</td>
          <td>0.334806</td>
          <td>24.981901</td>
          <td>0.223170</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.564506</td>
          <td>0.829162</td>
          <td>27.077009</td>
          <td>0.235073</td>
          <td>26.731118</td>
          <td>0.156497</td>
          <td>26.251644</td>
          <td>0.167097</td>
          <td>25.589781</td>
          <td>0.177095</td>
          <td>25.633996</td>
          <td>0.388891</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.377023</td>
          <td>0.760108</td>
          <td>27.081915</td>
          <td>0.249260</td>
          <td>26.760594</td>
          <td>0.171062</td>
          <td>25.785906</td>
          <td>0.119668</td>
          <td>25.683968</td>
          <td>0.204212</td>
          <td>25.084347</td>
          <td>0.267068</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.848998</td>
          <td>0.502731</td>
          <td>26.621113</td>
          <td>0.158839</td>
          <td>26.102886</td>
          <td>0.089769</td>
          <td>25.698136</td>
          <td>0.102482</td>
          <td>25.564189</td>
          <td>0.171622</td>
          <td>24.681030</td>
          <td>0.177224</td>
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
