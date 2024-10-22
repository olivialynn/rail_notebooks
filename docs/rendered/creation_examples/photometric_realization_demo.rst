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

    <pzflow.flow.Flow at 0x7f27b8ca61d0>



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
          <td>27.451628</td>
          <td>0.750663</td>
          <td>26.530190</td>
          <td>0.142139</td>
          <td>26.058164</td>
          <td>0.082998</td>
          <td>25.230696</td>
          <td>0.065145</td>
          <td>25.019420</td>
          <td>0.103184</td>
          <td>24.723003</td>
          <td>0.176625</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.745353</td>
          <td>1.578388</td>
          <td>28.329204</td>
          <td>0.596182</td>
          <td>27.547526</td>
          <td>0.294944</td>
          <td>28.839576</td>
          <td>1.081388</td>
          <td>26.509209</td>
          <td>0.359270</td>
          <td>25.559211</td>
          <td>0.350700</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.319673</td>
          <td>0.327242</td>
          <td>25.842743</td>
          <td>0.078021</td>
          <td>24.801399</td>
          <td>0.027311</td>
          <td>23.849958</td>
          <td>0.019392</td>
          <td>23.148815</td>
          <td>0.019931</td>
          <td>22.807490</td>
          <td>0.032900</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.171258</td>
          <td>0.619842</td>
          <td>28.202145</td>
          <td>0.544322</td>
          <td>27.538277</td>
          <td>0.292753</td>
          <td>26.870365</td>
          <td>0.266822</td>
          <td>25.591368</td>
          <td>0.169149</td>
          <td>25.291754</td>
          <td>0.283256</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.924029</td>
          <td>0.237531</td>
          <td>25.711603</td>
          <td>0.069492</td>
          <td>25.366614</td>
          <td>0.044969</td>
          <td>24.825407</td>
          <td>0.045463</td>
          <td>24.363906</td>
          <td>0.057860</td>
          <td>23.683055</td>
          <td>0.071472</td>
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
          <td>26.869888</td>
          <td>0.498990</td>
          <td>26.192475</td>
          <td>0.106055</td>
          <td>26.094099</td>
          <td>0.085669</td>
          <td>26.090518</td>
          <td>0.138437</td>
          <td>26.048885</td>
          <td>0.248145</td>
          <td>26.226184</td>
          <td>0.579012</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.031503</td>
          <td>1.078174</td>
          <td>26.582245</td>
          <td>0.148641</td>
          <td>26.987984</td>
          <td>0.185711</td>
          <td>26.570762</td>
          <td>0.208282</td>
          <td>26.061456</td>
          <td>0.250722</td>
          <td>25.791077</td>
          <td>0.419759</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.334770</td>
          <td>0.693947</td>
          <td>27.173248</td>
          <td>0.244494</td>
          <td>26.539270</td>
          <td>0.126440</td>
          <td>26.191975</td>
          <td>0.151063</td>
          <td>25.952003</td>
          <td>0.229062</td>
          <td>25.438233</td>
          <td>0.318652</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.237063</td>
          <td>0.257647</td>
          <td>26.625766</td>
          <td>0.136264</td>
          <td>25.795148</td>
          <td>0.107115</td>
          <td>25.571735</td>
          <td>0.166344</td>
          <td>25.256282</td>
          <td>0.275220</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.733917</td>
          <td>0.450907</td>
          <td>26.259259</td>
          <td>0.112415</td>
          <td>25.999408</td>
          <td>0.078805</td>
          <td>25.541057</td>
          <td>0.085707</td>
          <td>25.128094</td>
          <td>0.113456</td>
          <td>25.210072</td>
          <td>0.265052</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>27.154919</td>
          <td>0.275129</td>
          <td>25.967080</td>
          <td>0.090081</td>
          <td>25.391800</td>
          <td>0.088986</td>
          <td>25.104962</td>
          <td>0.130443</td>
          <td>24.784203</td>
          <td>0.218174</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.973708</td>
          <td>1.717502</td>
          <td>28.493440</td>
          <td>0.687245</td>
          <td>27.019705</td>
          <td>0.350745</td>
          <td>26.922479</td>
          <td>0.562988</td>
          <td>25.553780</td>
          <td>0.404859</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.001054</td>
          <td>0.286620</td>
          <td>26.108674</td>
          <td>0.115925</td>
          <td>24.738108</td>
          <td>0.031072</td>
          <td>23.854487</td>
          <td>0.023487</td>
          <td>23.125312</td>
          <td>0.023395</td>
          <td>22.822067</td>
          <td>0.040375</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.298641</td>
          <td>0.374458</td>
          <td>28.147808</td>
          <td>0.617748</td>
          <td>27.417831</td>
          <td>0.327396</td>
          <td>27.026789</td>
          <td>0.375078</td>
          <td>26.442288</td>
          <td>0.417143</td>
          <td>25.990922</td>
          <td>0.592732</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.830095</td>
          <td>0.245497</td>
          <td>25.959626</td>
          <td>0.099803</td>
          <td>25.441239</td>
          <td>0.056612</td>
          <td>24.828348</td>
          <td>0.054083</td>
          <td>24.407284</td>
          <td>0.070804</td>
          <td>23.795827</td>
          <td>0.093394</td>
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
          <td>26.482985</td>
          <td>0.418262</td>
          <td>26.130016</td>
          <td>0.117955</td>
          <td>26.155833</td>
          <td>0.108532</td>
          <td>26.058110</td>
          <td>0.162124</td>
          <td>25.511608</td>
          <td>0.188503</td>
          <td>25.631469</td>
          <td>0.437620</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.735249</td>
          <td>2.527501</td>
          <td>26.998348</td>
          <td>0.242897</td>
          <td>26.774872</td>
          <td>0.181900</td>
          <td>26.150192</td>
          <td>0.172417</td>
          <td>26.478062</td>
          <td>0.405860</td>
          <td>25.741024</td>
          <td>0.468258</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.906145</td>
          <td>0.569089</td>
          <td>29.089176</td>
          <td>1.091043</td>
          <td>26.998860</td>
          <td>0.221342</td>
          <td>26.590822</td>
          <td>0.251367</td>
          <td>25.935314</td>
          <td>0.265948</td>
          <td>25.089293</td>
          <td>0.283808</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.802668</td>
          <td>0.534392</td>
          <td>27.189368</td>
          <td>0.290413</td>
          <td>26.910913</td>
          <td>0.209354</td>
          <td>25.918555</td>
          <td>0.145372</td>
          <td>25.404950</td>
          <td>0.173964</td>
          <td>25.938729</td>
          <td>0.554252</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.529756</td>
          <td>0.430283</td>
          <td>26.121326</td>
          <td>0.115924</td>
          <td>26.144952</td>
          <td>0.106339</td>
          <td>25.670100</td>
          <td>0.114716</td>
          <td>25.479429</td>
          <td>0.181523</td>
          <td>25.135189</td>
          <td>0.293766</td>
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
          <td>27.075035</td>
          <td>0.579109</td>
          <td>26.786569</td>
          <td>0.176967</td>
          <td>25.880759</td>
          <td>0.070968</td>
          <td>25.271284</td>
          <td>0.067540</td>
          <td>24.999950</td>
          <td>0.101454</td>
          <td>24.798093</td>
          <td>0.188240</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>30.742297</td>
          <td>3.338267</td>
          <td>28.794292</td>
          <td>0.817763</td>
          <td>27.709879</td>
          <td>0.336077</td>
          <td>27.095428</td>
          <td>0.320230</td>
          <td>26.327585</td>
          <td>0.311400</td>
          <td>25.802562</td>
          <td>0.423814</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.253661</td>
          <td>0.687112</td>
          <td>25.970082</td>
          <td>0.093798</td>
          <td>24.757828</td>
          <td>0.028536</td>
          <td>23.893047</td>
          <td>0.021868</td>
          <td>23.132631</td>
          <td>0.021293</td>
          <td>22.823140</td>
          <td>0.036336</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.706421</td>
          <td>0.893632</td>
          <td>28.244288</td>
          <td>0.607791</td>
          <td>26.251411</td>
          <td>0.199232</td>
          <td>26.535699</td>
          <td>0.446435</td>
          <td>25.349951</td>
          <td>0.366099</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.931712</td>
          <td>0.239261</td>
          <td>25.708680</td>
          <td>0.069399</td>
          <td>25.489410</td>
          <td>0.050221</td>
          <td>24.755470</td>
          <td>0.042792</td>
          <td>24.319085</td>
          <td>0.055683</td>
          <td>23.770845</td>
          <td>0.077355</td>
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
          <td>26.373631</td>
          <td>0.359012</td>
          <td>26.326832</td>
          <td>0.127627</td>
          <td>26.085018</td>
          <td>0.091964</td>
          <td>26.136766</td>
          <td>0.156134</td>
          <td>25.990951</td>
          <td>0.254653</td>
          <td>25.248770</td>
          <td>0.294888</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.828534</td>
          <td>0.185891</td>
          <td>27.017841</td>
          <td>0.193485</td>
          <td>26.724732</td>
          <td>0.240619</td>
          <td>25.990729</td>
          <td>0.240212</td>
          <td>26.615164</td>
          <td>0.766704</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.560903</td>
          <td>0.827241</td>
          <td>27.315964</td>
          <td>0.285817</td>
          <td>26.747638</td>
          <td>0.158724</td>
          <td>26.417737</td>
          <td>0.192353</td>
          <td>25.884880</td>
          <td>0.226881</td>
          <td>24.759791</td>
          <td>0.191327</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.327726</td>
          <td>1.339123</td>
          <td>27.277423</td>
          <td>0.292268</td>
          <td>26.408610</td>
          <td>0.126428</td>
          <td>25.953584</td>
          <td>0.138367</td>
          <td>25.153865</td>
          <td>0.129941</td>
          <td>25.805658</td>
          <td>0.469963</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.678324</td>
          <td>0.166782</td>
          <td>26.052137</td>
          <td>0.085848</td>
          <td>25.463373</td>
          <td>0.083383</td>
          <td>25.082609</td>
          <td>0.113336</td>
          <td>24.805112</td>
          <td>0.196808</td>
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
