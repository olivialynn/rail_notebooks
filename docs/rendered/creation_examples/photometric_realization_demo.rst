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

    <pzflow.flow.Flow at 0x7fb7163eadd0>



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
          <td>26.401673</td>
          <td>0.349140</td>
          <td>26.959761</td>
          <td>0.204760</td>
          <td>25.887779</td>
          <td>0.071401</td>
          <td>25.259373</td>
          <td>0.066822</td>
          <td>24.999826</td>
          <td>0.101430</td>
          <td>24.745512</td>
          <td>0.180028</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.147166</td>
          <td>0.239294</td>
          <td>28.070346</td>
          <td>0.443880</td>
          <td>27.048287</td>
          <td>0.308112</td>
          <td>27.499527</td>
          <td>0.739531</td>
          <td>25.848780</td>
          <td>0.438588</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.321033</td>
          <td>0.327595</td>
          <td>25.991969</td>
          <td>0.088972</td>
          <td>24.803161</td>
          <td>0.027353</td>
          <td>23.883541</td>
          <td>0.019951</td>
          <td>23.177533</td>
          <td>0.020422</td>
          <td>22.864484</td>
          <td>0.034596</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.271031</td>
          <td>0.515408</td>
          <td>26.557871</td>
          <td>0.206045</td>
          <td>25.750577</td>
          <td>0.193564</td>
          <td>24.783029</td>
          <td>0.185835</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.962066</td>
          <td>0.245089</td>
          <td>25.899280</td>
          <td>0.082006</td>
          <td>25.519883</td>
          <td>0.051524</td>
          <td>24.846642</td>
          <td>0.046328</td>
          <td>24.376916</td>
          <td>0.058531</td>
          <td>23.736054</td>
          <td>0.074902</td>
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
          <td>27.953025</td>
          <td>1.029453</td>
          <td>26.307804</td>
          <td>0.117265</td>
          <td>26.068014</td>
          <td>0.083722</td>
          <td>26.037547</td>
          <td>0.132246</td>
          <td>25.959753</td>
          <td>0.230539</td>
          <td>25.214754</td>
          <td>0.266067</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.414833</td>
          <td>0.732466</td>
          <td>27.010893</td>
          <td>0.213703</td>
          <td>26.842508</td>
          <td>0.164128</td>
          <td>26.712509</td>
          <td>0.234361</td>
          <td>26.119148</td>
          <td>0.262859</td>
          <td>25.856279</td>
          <td>0.441085</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.243961</td>
          <td>1.979232</td>
          <td>26.819848</td>
          <td>0.182006</td>
          <td>26.553051</td>
          <td>0.127959</td>
          <td>26.803340</td>
          <td>0.252581</td>
          <td>25.832414</td>
          <td>0.207335</td>
          <td>24.845108</td>
          <td>0.195823</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.660274</td>
          <td>0.361676</td>
          <td>26.670125</td>
          <td>0.141578</td>
          <td>25.873728</td>
          <td>0.114716</td>
          <td>25.897162</td>
          <td>0.218855</td>
          <td>27.641087</td>
          <td>1.388573</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.352284</td>
          <td>0.121884</td>
          <td>26.189064</td>
          <td>0.093132</td>
          <td>25.698073</td>
          <td>0.098390</td>
          <td>25.198574</td>
          <td>0.120632</td>
          <td>24.651394</td>
          <td>0.166190</td>
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
          <td>28.218806</td>
          <td>1.290727</td>
          <td>26.688140</td>
          <td>0.186841</td>
          <td>26.082621</td>
          <td>0.099694</td>
          <td>25.376153</td>
          <td>0.087770</td>
          <td>24.925259</td>
          <td>0.111592</td>
          <td>24.902105</td>
          <td>0.240579</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.874811</td>
          <td>1.063690</td>
          <td>27.962225</td>
          <td>0.514560</td>
          <td>27.252099</td>
          <td>0.269480</td>
          <td>27.294696</td>
          <td>0.433816</td>
          <td>26.297694</td>
          <td>0.351538</td>
          <td>26.005435</td>
          <td>0.566445</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.516658</td>
          <td>0.429506</td>
          <td>25.899441</td>
          <td>0.096579</td>
          <td>24.795956</td>
          <td>0.032693</td>
          <td>23.854416</td>
          <td>0.023485</td>
          <td>23.177443</td>
          <td>0.024470</td>
          <td>22.784074</td>
          <td>0.039041</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.282150</td>
          <td>0.678058</td>
          <td>27.916956</td>
          <td>0.480843</td>
          <td>27.004727</td>
          <td>0.368684</td>
          <td>26.248456</td>
          <td>0.359023</td>
          <td>25.598288</td>
          <td>0.444505</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.310035</td>
          <td>0.360895</td>
          <td>25.853365</td>
          <td>0.090931</td>
          <td>25.455526</td>
          <td>0.057334</td>
          <td>24.881491</td>
          <td>0.056694</td>
          <td>24.389458</td>
          <td>0.069696</td>
          <td>23.802504</td>
          <td>0.093943</td>
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
          <td>26.272831</td>
          <td>0.355500</td>
          <td>26.300261</td>
          <td>0.136679</td>
          <td>26.145347</td>
          <td>0.107543</td>
          <td>26.219591</td>
          <td>0.185958</td>
          <td>25.975922</td>
          <td>0.276983</td>
          <td>25.332182</td>
          <td>0.347238</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.998998</td>
          <td>1.145101</td>
          <td>26.644315</td>
          <td>0.180700</td>
          <td>26.796346</td>
          <td>0.185234</td>
          <td>26.315600</td>
          <td>0.198291</td>
          <td>25.840412</td>
          <td>0.244070</td>
          <td>26.257110</td>
          <td>0.677985</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.109626</td>
          <td>0.656656</td>
          <td>27.820625</td>
          <td>0.467793</td>
          <td>27.281931</td>
          <td>0.279327</td>
          <td>26.125935</td>
          <td>0.170361</td>
          <td>26.532100</td>
          <td>0.426161</td>
          <td>25.844797</td>
          <td>0.509490</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.673146</td>
          <td>0.958073</td>
          <td>27.170215</td>
          <td>0.285954</td>
          <td>26.490388</td>
          <td>0.146528</td>
          <td>26.059293</td>
          <td>0.163994</td>
          <td>25.733653</td>
          <td>0.229252</td>
          <td>25.243011</td>
          <td>0.326713</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>30.476818</td>
          <td>3.220150</td>
          <td>26.459943</td>
          <td>0.155256</td>
          <td>26.020939</td>
          <td>0.095398</td>
          <td>25.737343</td>
          <td>0.121624</td>
          <td>25.027591</td>
          <td>0.123198</td>
          <td>24.888951</td>
          <td>0.240298</td>
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
          <td>26.416404</td>
          <td>0.128861</td>
          <td>25.999351</td>
          <td>0.078812</td>
          <td>25.304345</td>
          <td>0.069547</td>
          <td>24.972298</td>
          <td>0.099026</td>
          <td>24.807092</td>
          <td>0.189675</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.356633</td>
          <td>2.075098</td>
          <td>29.652652</td>
          <td>1.353053</td>
          <td>27.947174</td>
          <td>0.404438</td>
          <td>26.789861</td>
          <td>0.250032</td>
          <td>26.399935</td>
          <td>0.329882</td>
          <td>25.824219</td>
          <td>0.430857</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.810147</td>
          <td>0.081500</td>
          <td>24.779986</td>
          <td>0.029095</td>
          <td>23.859730</td>
          <td>0.021254</td>
          <td>23.128762</td>
          <td>0.021223</td>
          <td>22.864335</td>
          <td>0.037684</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.527554</td>
          <td>0.890957</td>
          <td>27.729756</td>
          <td>0.454510</td>
          <td>28.079576</td>
          <td>0.540244</td>
          <td>26.634898</td>
          <td>0.273609</td>
          <td>26.332585</td>
          <td>0.382153</td>
          <td>25.591799</td>
          <td>0.440940</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.664936</td>
          <td>0.191591</td>
          <td>25.756490</td>
          <td>0.072393</td>
          <td>25.484958</td>
          <td>0.050022</td>
          <td>24.838375</td>
          <td>0.046059</td>
          <td>24.421270</td>
          <td>0.060968</td>
          <td>23.636429</td>
          <td>0.068685</td>
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
          <td>26.647804</td>
          <td>0.443306</td>
          <td>26.248217</td>
          <td>0.119218</td>
          <td>26.078891</td>
          <td>0.091470</td>
          <td>25.976826</td>
          <td>0.136075</td>
          <td>25.510284</td>
          <td>0.170364</td>
          <td>25.804697</td>
          <td>0.455003</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.093365</td>
          <td>0.592139</td>
          <td>27.591971</td>
          <td>0.347205</td>
          <td>26.547413</td>
          <td>0.129421</td>
          <td>26.455320</td>
          <td>0.192183</td>
          <td>26.557635</td>
          <td>0.378596</td>
          <td>25.878145</td>
          <td>0.455050</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.807622</td>
          <td>2.504567</td>
          <td>27.169517</td>
          <td>0.253680</td>
          <td>27.023526</td>
          <td>0.200543</td>
          <td>26.432838</td>
          <td>0.194816</td>
          <td>26.007324</td>
          <td>0.251022</td>
          <td>25.430046</td>
          <td>0.331455</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.704755</td>
          <td>1.619083</td>
          <td>28.637115</td>
          <td>0.795012</td>
          <td>26.374235</td>
          <td>0.122714</td>
          <td>25.677112</td>
          <td>0.108848</td>
          <td>25.762794</td>
          <td>0.218119</td>
          <td>25.448314</td>
          <td>0.357397</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.536514</td>
          <td>0.397251</td>
          <td>26.453793</td>
          <td>0.137590</td>
          <td>26.053808</td>
          <td>0.085974</td>
          <td>25.568249</td>
          <td>0.091447</td>
          <td>25.235411</td>
          <td>0.129423</td>
          <td>25.004593</td>
          <td>0.232464</td>
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
