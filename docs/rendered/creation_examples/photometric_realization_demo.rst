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

    <pzflow.flow.Flow at 0x7f75049bd060>



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
          <td>27.478467</td>
          <td>0.764133</td>
          <td>26.758952</td>
          <td>0.172849</td>
          <td>26.002158</td>
          <td>0.078997</td>
          <td>25.385334</td>
          <td>0.074702</td>
          <td>25.110346</td>
          <td>0.111714</td>
          <td>25.503758</td>
          <td>0.335684</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.105865</td>
          <td>1.864684</td>
          <td>28.642475</td>
          <td>0.739643</td>
          <td>28.157353</td>
          <td>0.473848</td>
          <td>26.930211</td>
          <td>0.280133</td>
          <td>27.615479</td>
          <td>0.798341</td>
          <td>26.001260</td>
          <td>0.491648</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.582935</td>
          <td>0.401981</td>
          <td>25.877814</td>
          <td>0.080470</td>
          <td>24.853174</td>
          <td>0.028576</td>
          <td>23.860142</td>
          <td>0.019559</td>
          <td>23.111507</td>
          <td>0.019313</td>
          <td>22.851689</td>
          <td>0.034207</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.454809</td>
          <td>0.752250</td>
          <td>28.820730</td>
          <td>0.831329</td>
          <td>27.309679</td>
          <td>0.242951</td>
          <td>26.671718</td>
          <td>0.226570</td>
          <td>25.961768</td>
          <td>0.230924</td>
          <td>25.302374</td>
          <td>0.285702</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.285129</td>
          <td>0.318376</td>
          <td>25.831905</td>
          <td>0.077279</td>
          <td>25.349077</td>
          <td>0.044274</td>
          <td>24.924876</td>
          <td>0.049660</td>
          <td>24.327667</td>
          <td>0.056028</td>
          <td>23.685578</td>
          <td>0.071632</td>
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
          <td>26.317201</td>
          <td>0.326601</td>
          <td>26.240612</td>
          <td>0.110603</td>
          <td>26.252674</td>
          <td>0.098479</td>
          <td>26.008130</td>
          <td>0.128922</td>
          <td>26.294356</td>
          <td>0.302953</td>
          <td>26.171586</td>
          <td>0.556785</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.417642</td>
          <td>0.733844</td>
          <td>27.269772</td>
          <td>0.264629</td>
          <td>26.822360</td>
          <td>0.161329</td>
          <td>26.302580</td>
          <td>0.166050</td>
          <td>25.477275</td>
          <td>0.153443</td>
          <td>25.444455</td>
          <td>0.320237</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.903065</td>
          <td>0.511318</td>
          <td>27.404619</td>
          <td>0.295208</td>
          <td>27.074669</td>
          <td>0.199784</td>
          <td>26.740278</td>
          <td>0.239803</td>
          <td>26.014132</td>
          <td>0.241142</td>
          <td>25.875572</td>
          <td>0.447561</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.101818</td>
          <td>0.590207</td>
          <td>27.164822</td>
          <td>0.242803</td>
          <td>26.618005</td>
          <td>0.135354</td>
          <td>25.923363</td>
          <td>0.119780</td>
          <td>25.408780</td>
          <td>0.144679</td>
          <td>26.036278</td>
          <td>0.504526</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.143658</td>
          <td>0.607933</td>
          <td>26.907726</td>
          <td>0.196009</td>
          <td>25.955582</td>
          <td>0.075813</td>
          <td>25.546998</td>
          <td>0.086157</td>
          <td>25.561878</td>
          <td>0.164952</td>
          <td>24.851527</td>
          <td>0.196883</td>
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
          <td>29.038081</td>
          <td>1.918779</td>
          <td>26.503144</td>
          <td>0.159682</td>
          <td>26.054446</td>
          <td>0.097262</td>
          <td>25.183740</td>
          <td>0.074071</td>
          <td>25.199591</td>
          <td>0.141544</td>
          <td>24.935523</td>
          <td>0.247295</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.960510</td>
          <td>0.513914</td>
          <td>28.624691</td>
          <td>0.750762</td>
          <td>26.610791</td>
          <td>0.252434</td>
          <td>26.757108</td>
          <td>0.499106</td>
          <td>25.936751</td>
          <td>0.539058</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.335137</td>
          <td>0.373565</td>
          <td>25.806986</td>
          <td>0.089059</td>
          <td>24.809996</td>
          <td>0.033100</td>
          <td>23.902620</td>
          <td>0.024483</td>
          <td>23.145521</td>
          <td>0.023805</td>
          <td>22.808926</td>
          <td>0.039908</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.467208</td>
          <td>0.859343</td>
          <td>28.783809</td>
          <td>0.939848</td>
          <td>27.094759</td>
          <td>0.252168</td>
          <td>26.416425</td>
          <td>0.229454</td>
          <td>25.814083</td>
          <td>0.253325</td>
          <td>24.966937</td>
          <td>0.270548</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.742751</td>
          <td>0.228435</td>
          <td>25.721744</td>
          <td>0.080997</td>
          <td>25.486795</td>
          <td>0.058946</td>
          <td>24.861291</td>
          <td>0.055687</td>
          <td>24.421236</td>
          <td>0.071683</td>
          <td>23.776334</td>
          <td>0.091809</td>
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
          <td>26.727754</td>
          <td>0.502537</td>
          <td>26.350470</td>
          <td>0.142717</td>
          <td>26.136429</td>
          <td>0.106708</td>
          <td>26.407547</td>
          <td>0.217733</td>
          <td>25.863094</td>
          <td>0.252608</td>
          <td>25.450054</td>
          <td>0.380768</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.621983</td>
          <td>0.914825</td>
          <td>26.747879</td>
          <td>0.197191</td>
          <td>26.764437</td>
          <td>0.180300</td>
          <td>26.618770</td>
          <td>0.255067</td>
          <td>26.280142</td>
          <td>0.347948</td>
          <td>25.810425</td>
          <td>0.493066</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.322648</td>
          <td>0.758443</td>
          <td>26.943101</td>
          <td>0.233771</td>
          <td>26.829940</td>
          <td>0.192144</td>
          <td>27.113805</td>
          <td>0.381895</td>
          <td>25.687305</td>
          <td>0.216735</td>
          <td>25.563236</td>
          <td>0.412432</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.225226</td>
          <td>0.298924</td>
          <td>26.415899</td>
          <td>0.137425</td>
          <td>25.658230</td>
          <td>0.116053</td>
          <td>25.476893</td>
          <td>0.184898</td>
          <td>25.166894</td>
          <td>0.307458</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.142911</td>
          <td>0.318403</td>
          <td>26.238578</td>
          <td>0.128331</td>
          <td>26.165737</td>
          <td>0.108287</td>
          <td>25.533371</td>
          <td>0.101808</td>
          <td>25.297173</td>
          <td>0.155432</td>
          <td>24.966627</td>
          <td>0.256148</td>
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
          <td>26.830552</td>
          <td>0.183681</td>
          <td>25.905002</td>
          <td>0.072507</td>
          <td>25.366627</td>
          <td>0.073487</td>
          <td>25.092866</td>
          <td>0.110038</td>
          <td>25.257881</td>
          <td>0.275613</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.421406</td>
          <td>0.736060</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.407350</td>
          <td>0.263461</td>
          <td>27.151116</td>
          <td>0.334718</td>
          <td>27.023211</td>
          <td>0.530584</td>
          <td>26.308067</td>
          <td>0.614081</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.822572</td>
          <td>0.230811</td>
          <td>25.926859</td>
          <td>0.090308</td>
          <td>24.772383</td>
          <td>0.028902</td>
          <td>23.905711</td>
          <td>0.022107</td>
          <td>23.115900</td>
          <td>0.020992</td>
          <td>22.817737</td>
          <td>0.036163</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.206223</td>
          <td>0.722991</td>
          <td>27.571784</td>
          <td>0.403064</td>
          <td>27.108689</td>
          <td>0.254218</td>
          <td>26.751398</td>
          <td>0.300636</td>
          <td>26.343912</td>
          <td>0.385523</td>
          <td>25.362719</td>
          <td>0.369766</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.999765</td>
          <td>0.253023</td>
          <td>25.811039</td>
          <td>0.075964</td>
          <td>25.518928</td>
          <td>0.051554</td>
          <td>24.832373</td>
          <td>0.045814</td>
          <td>24.443489</td>
          <td>0.062181</td>
          <td>23.722970</td>
          <td>0.074151</td>
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
          <td>26.638286</td>
          <td>0.440129</td>
          <td>26.373854</td>
          <td>0.132922</td>
          <td>26.071606</td>
          <td>0.090887</td>
          <td>26.062341</td>
          <td>0.146478</td>
          <td>25.804353</td>
          <td>0.218247</td>
          <td>25.131965</td>
          <td>0.268249</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.578023</td>
          <td>0.404506</td>
          <td>27.413375</td>
          <td>0.301217</td>
          <td>27.355294</td>
          <td>0.256149</td>
          <td>26.385010</td>
          <td>0.181099</td>
          <td>25.854635</td>
          <td>0.214554</td>
          <td>24.778046</td>
          <td>0.188124</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.332925</td>
          <td>0.711724</td>
          <td>27.004099</td>
          <td>0.221281</td>
          <td>26.890750</td>
          <td>0.179291</td>
          <td>26.180609</td>
          <td>0.157264</td>
          <td>26.332094</td>
          <td>0.326405</td>
          <td>25.347744</td>
          <td>0.310416</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.915002</td>
          <td>0.552081</td>
          <td>27.151438</td>
          <td>0.263865</td>
          <td>26.521749</td>
          <td>0.139417</td>
          <td>25.615471</td>
          <td>0.103139</td>
          <td>25.481702</td>
          <td>0.172154</td>
          <td>25.115868</td>
          <td>0.274013</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.223998</td>
          <td>0.310826</td>
          <td>26.499135</td>
          <td>0.143068</td>
          <td>26.057190</td>
          <td>0.086231</td>
          <td>25.639843</td>
          <td>0.097380</td>
          <td>25.021035</td>
          <td>0.107408</td>
          <td>24.854619</td>
          <td>0.205160</td>
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
