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

    <pzflow.flow.Flow at 0x7f9443dd6800>



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
          <td>29.799559</td>
          <td>2.462151</td>
          <td>26.649769</td>
          <td>0.157492</td>
          <td>26.119047</td>
          <td>0.087571</td>
          <td>25.354785</td>
          <td>0.072711</td>
          <td>24.832590</td>
          <td>0.087579</td>
          <td>25.032969</td>
          <td>0.229103</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.496386</td>
          <td>0.669952</td>
          <td>27.471078</td>
          <td>0.277255</td>
          <td>27.379209</td>
          <td>0.399660</td>
          <td>27.127082</td>
          <td>0.571469</td>
          <td>26.510950</td>
          <td>0.705844</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.427197</td>
          <td>0.356206</td>
          <td>25.902185</td>
          <td>0.082216</td>
          <td>24.801279</td>
          <td>0.027308</td>
          <td>23.834899</td>
          <td>0.019147</td>
          <td>23.145038</td>
          <td>0.019867</td>
          <td>22.803837</td>
          <td>0.032794</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.543377</td>
          <td>1.427120</td>
          <td>27.542842</td>
          <td>0.329705</td>
          <td>27.471446</td>
          <td>0.277338</td>
          <td>26.802752</td>
          <td>0.252459</td>
          <td>25.775989</td>
          <td>0.197748</td>
          <td>25.182547</td>
          <td>0.259155</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.263681</td>
          <td>0.312976</td>
          <td>25.816422</td>
          <td>0.076231</td>
          <td>25.397930</td>
          <td>0.046236</td>
          <td>24.824301</td>
          <td>0.045418</td>
          <td>24.397402</td>
          <td>0.059605</td>
          <td>23.814110</td>
          <td>0.080247</td>
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
          <td>26.289624</td>
          <td>0.319518</td>
          <td>26.277337</td>
          <td>0.114198</td>
          <td>26.282270</td>
          <td>0.101066</td>
          <td>26.061040</td>
          <td>0.134958</td>
          <td>25.906343</td>
          <td>0.220534</td>
          <td>25.505467</td>
          <td>0.336138</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.915577</td>
          <td>0.197307</td>
          <td>26.788569</td>
          <td>0.156735</td>
          <td>26.289119</td>
          <td>0.164155</td>
          <td>25.993772</td>
          <td>0.237122</td>
          <td>25.827825</td>
          <td>0.431672</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>31.407601</td>
          <td>3.979599</td>
          <td>27.289985</td>
          <td>0.269027</td>
          <td>26.807018</td>
          <td>0.159228</td>
          <td>26.325399</td>
          <td>0.169309</td>
          <td>25.827976</td>
          <td>0.206565</td>
          <td>25.462031</td>
          <td>0.324750</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.869888</td>
          <td>2.525401</td>
          <td>27.865839</td>
          <td>0.423918</td>
          <td>26.472202</td>
          <td>0.119288</td>
          <td>26.000818</td>
          <td>0.128108</td>
          <td>25.506348</td>
          <td>0.157310</td>
          <td>26.113689</td>
          <td>0.533934</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.024627</td>
          <td>0.558533</td>
          <td>26.503268</td>
          <td>0.138882</td>
          <td>26.098480</td>
          <td>0.086000</td>
          <td>25.585619</td>
          <td>0.089136</td>
          <td>25.294557</td>
          <td>0.131101</td>
          <td>24.965774</td>
          <td>0.216652</td>
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
          <td>26.406272</td>
          <td>0.388848</td>
          <td>27.041309</td>
          <td>0.250748</td>
          <td>26.094662</td>
          <td>0.100750</td>
          <td>25.194003</td>
          <td>0.074746</td>
          <td>25.192856</td>
          <td>0.140726</td>
          <td>24.932275</td>
          <td>0.246635</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.946161</td>
          <td>1.108780</td>
          <td>28.926278</td>
          <td>0.982969</td>
          <td>27.443325</td>
          <td>0.314447</td>
          <td>27.782383</td>
          <td>0.619642</td>
          <td>26.277928</td>
          <td>0.346111</td>
          <td>26.336346</td>
          <td>0.713260</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.839050</td>
          <td>0.545522</td>
          <td>25.737637</td>
          <td>0.083794</td>
          <td>24.777283</td>
          <td>0.032160</td>
          <td>23.870938</td>
          <td>0.023822</td>
          <td>23.167353</td>
          <td>0.024258</td>
          <td>22.832940</td>
          <td>0.040766</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>30.931470</td>
          <td>3.703983</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.177456</td>
          <td>0.269812</td>
          <td>27.915585</td>
          <td>0.716843</td>
          <td>26.129206</td>
          <td>0.326776</td>
          <td>25.283107</td>
          <td>0.348538</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.594376</td>
          <td>0.448966</td>
          <td>25.764754</td>
          <td>0.084121</td>
          <td>25.405223</td>
          <td>0.054832</td>
          <td>24.858800</td>
          <td>0.055564</td>
          <td>24.323775</td>
          <td>0.065760</td>
          <td>23.620689</td>
          <td>0.080054</td>
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
          <td>26.701398</td>
          <td>0.492859</td>
          <td>26.471535</td>
          <td>0.158323</td>
          <td>26.311430</td>
          <td>0.124269</td>
          <td>26.296688</td>
          <td>0.198440</td>
          <td>25.809887</td>
          <td>0.241789</td>
          <td>25.930512</td>
          <td>0.546155</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.317629</td>
          <td>1.362945</td>
          <td>27.028689</td>
          <td>0.249036</td>
          <td>26.841721</td>
          <td>0.192464</td>
          <td>26.382711</td>
          <td>0.209767</td>
          <td>26.584738</td>
          <td>0.440249</td>
          <td>25.311074</td>
          <td>0.336239</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.150269</td>
          <td>0.675272</td>
          <td>27.377857</td>
          <td>0.332525</td>
          <td>26.994303</td>
          <td>0.220504</td>
          <td>26.157558</td>
          <td>0.175001</td>
          <td>25.944717</td>
          <td>0.267995</td>
          <td>25.014241</td>
          <td>0.267016</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.298691</td>
          <td>0.754581</td>
          <td>27.237893</td>
          <td>0.301981</td>
          <td>26.402743</td>
          <td>0.135874</td>
          <td>25.885931</td>
          <td>0.141347</td>
          <td>25.409999</td>
          <td>0.174711</td>
          <td>26.946820</td>
          <td>1.072471</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.507898</td>
          <td>0.161749</td>
          <td>26.111475</td>
          <td>0.103272</td>
          <td>25.586375</td>
          <td>0.106638</td>
          <td>25.108179</td>
          <td>0.132104</td>
          <td>24.650064</td>
          <td>0.196927</td>
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
          <td>30.475844</td>
          <td>3.085084</td>
          <td>26.730719</td>
          <td>0.168768</td>
          <td>26.214100</td>
          <td>0.095215</td>
          <td>25.244675</td>
          <td>0.065967</td>
          <td>25.129381</td>
          <td>0.113598</td>
          <td>24.971747</td>
          <td>0.217762</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.695369</td>
          <td>0.438224</td>
          <td>28.865360</td>
          <td>0.855935</td>
          <td>27.597148</td>
          <td>0.307211</td>
          <td>27.265392</td>
          <td>0.366206</td>
          <td>26.421743</td>
          <td>0.335634</td>
          <td>25.857398</td>
          <td>0.441832</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.952511</td>
          <td>0.256845</td>
          <td>25.885594</td>
          <td>0.087093</td>
          <td>24.783290</td>
          <td>0.029179</td>
          <td>23.857257</td>
          <td>0.021209</td>
          <td>23.143208</td>
          <td>0.021486</td>
          <td>22.870421</td>
          <td>0.037887</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.174437</td>
          <td>0.627748</td>
          <td>27.937063</td>
          <td>0.486608</td>
          <td>27.593897</td>
          <td>0.571527</td>
          <td>26.134532</td>
          <td>0.327104</td>
          <td>25.004531</td>
          <td>0.278002</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.343406</td>
          <td>0.333751</td>
          <td>25.846083</td>
          <td>0.078348</td>
          <td>25.416210</td>
          <td>0.047060</td>
          <td>24.737919</td>
          <td>0.042131</td>
          <td>24.475159</td>
          <td>0.063952</td>
          <td>23.667769</td>
          <td>0.070617</td>
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
          <td>26.516652</td>
          <td>0.401144</td>
          <td>26.312376</td>
          <td>0.126040</td>
          <td>26.198301</td>
          <td>0.101571</td>
          <td>26.173824</td>
          <td>0.161161</td>
          <td>25.710504</td>
          <td>0.201773</td>
          <td>25.518630</td>
          <td>0.365351</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.209650</td>
          <td>0.302885</td>
          <td>26.672019</td>
          <td>0.162764</td>
          <td>26.899844</td>
          <td>0.175108</td>
          <td>26.616986</td>
          <td>0.220059</td>
          <td>26.498031</td>
          <td>0.361397</td>
          <td>25.307587</td>
          <td>0.291456</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.810196</td>
          <td>0.188100</td>
          <td>26.841239</td>
          <td>0.171911</td>
          <td>26.485629</td>
          <td>0.203651</td>
          <td>25.846677</td>
          <td>0.219788</td>
          <td>24.997156</td>
          <td>0.233302</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.358650</td>
          <td>0.659410</td>
          <td>26.886092</td>
          <td>0.190247</td>
          <td>25.953441</td>
          <td>0.138350</td>
          <td>25.629942</td>
          <td>0.195153</td>
          <td>25.980875</td>
          <td>0.534765</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.057729</td>
          <td>0.584751</td>
          <td>26.265626</td>
          <td>0.116906</td>
          <td>26.131816</td>
          <td>0.092081</td>
          <td>25.687031</td>
          <td>0.101491</td>
          <td>25.065836</td>
          <td>0.111691</td>
          <td>25.200880</td>
          <td>0.273112</td>
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
