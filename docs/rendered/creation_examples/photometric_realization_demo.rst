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

    <pzflow.flow.Flow at 0x7f913d155f60>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.546445</td>
          <td>0.144140</td>
          <td>26.132347</td>
          <td>0.088602</td>
          <td>25.311672</td>
          <td>0.069990</td>
          <td>25.034551</td>
          <td>0.104559</td>
          <td>24.717966</td>
          <td>0.175872</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>30.159271</td>
          <td>2.789663</td>
          <td>28.480388</td>
          <td>0.662617</td>
          <td>27.005469</td>
          <td>0.188474</td>
          <td>27.228192</td>
          <td>0.355376</td>
          <td>27.087955</td>
          <td>0.555632</td>
          <td>26.425185</td>
          <td>0.665702</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.059712</td>
          <td>0.265483</td>
          <td>25.885617</td>
          <td>0.081026</td>
          <td>24.779640</td>
          <td>0.026798</td>
          <td>23.888916</td>
          <td>0.020042</td>
          <td>23.146588</td>
          <td>0.019893</td>
          <td>22.863186</td>
          <td>0.034556</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.797420</td>
          <td>0.402286</td>
          <td>27.264998</td>
          <td>0.234148</td>
          <td>26.225881</td>
          <td>0.155517</td>
          <td>26.478386</td>
          <td>0.350680</td>
          <td>24.995021</td>
          <td>0.221995</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.887017</td>
          <td>0.505326</td>
          <td>25.609271</td>
          <td>0.063480</td>
          <td>25.254161</td>
          <td>0.040700</td>
          <td>24.798125</td>
          <td>0.044375</td>
          <td>24.345508</td>
          <td>0.056923</td>
          <td>23.730813</td>
          <td>0.074556</td>
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
          <td>25.953827</td>
          <td>0.243434</td>
          <td>26.196676</td>
          <td>0.106445</td>
          <td>26.243981</td>
          <td>0.097731</td>
          <td>26.228733</td>
          <td>0.155898</td>
          <td>26.073740</td>
          <td>0.253264</td>
          <td>25.047234</td>
          <td>0.231828</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.918880</td>
          <td>0.517279</td>
          <td>26.664655</td>
          <td>0.159508</td>
          <td>26.882360</td>
          <td>0.169797</td>
          <td>26.414424</td>
          <td>0.182599</td>
          <td>26.034457</td>
          <td>0.245216</td>
          <td>26.018556</td>
          <td>0.497976</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.979188</td>
          <td>0.540506</td>
          <td>27.056503</td>
          <td>0.221979</td>
          <td>26.883353</td>
          <td>0.169941</td>
          <td>26.521962</td>
          <td>0.199931</td>
          <td>25.516916</td>
          <td>0.158739</td>
          <td>25.420244</td>
          <td>0.314108</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.609132</td>
          <td>0.410142</td>
          <td>27.601496</td>
          <td>0.345359</td>
          <td>26.667479</td>
          <td>0.141256</td>
          <td>25.832944</td>
          <td>0.110709</td>
          <td>25.438676</td>
          <td>0.148444</td>
          <td>25.342031</td>
          <td>0.294999</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.726725</td>
          <td>1.564146</td>
          <td>26.644636</td>
          <td>0.156802</td>
          <td>25.989255</td>
          <td>0.078102</td>
          <td>25.649918</td>
          <td>0.094319</td>
          <td>24.987607</td>
          <td>0.100350</td>
          <td>24.758464</td>
          <td>0.182014</td>
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
          <td>29.077126</td>
          <td>1.951249</td>
          <td>26.548445</td>
          <td>0.165971</td>
          <td>26.061957</td>
          <td>0.097905</td>
          <td>25.187629</td>
          <td>0.074326</td>
          <td>25.114573</td>
          <td>0.131531</td>
          <td>25.306966</td>
          <td>0.333855</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.306218</td>
          <td>1.224792</td>
          <td>27.578281</td>
          <td>0.349964</td>
          <td>26.730332</td>
          <td>0.278308</td>
          <td>26.031753</td>
          <td>0.284308</td>
          <td>25.612270</td>
          <td>0.423393</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.838134</td>
          <td>0.251038</td>
          <td>25.914079</td>
          <td>0.097825</td>
          <td>24.750285</td>
          <td>0.031406</td>
          <td>23.861809</td>
          <td>0.023635</td>
          <td>23.133773</td>
          <td>0.023566</td>
          <td>22.799841</td>
          <td>0.039589</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>31.375111</td>
          <td>4.135533</td>
          <td>27.075310</td>
          <td>0.272838</td>
          <td>26.660962</td>
          <td>0.175503</td>
          <td>26.567403</td>
          <td>0.259839</td>
          <td>26.814284</td>
          <td>0.550106</td>
          <td>25.063530</td>
          <td>0.292579</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.059757</td>
          <td>0.295902</td>
          <td>25.749641</td>
          <td>0.083010</td>
          <td>25.438316</td>
          <td>0.056466</td>
          <td>24.694894</td>
          <td>0.048043</td>
          <td>24.241253</td>
          <td>0.061124</td>
          <td>23.843611</td>
          <td>0.097392</td>
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
          <td>26.749534</td>
          <td>0.510647</td>
          <td>26.714875</td>
          <td>0.194607</td>
          <td>26.573421</td>
          <td>0.155751</td>
          <td>26.077362</td>
          <td>0.164808</td>
          <td>25.871659</td>
          <td>0.254389</td>
          <td>24.794185</td>
          <td>0.224532</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.822840</td>
          <td>0.465478</td>
          <td>26.804592</td>
          <td>0.186529</td>
          <td>26.295416</td>
          <td>0.194953</td>
          <td>26.024942</td>
          <td>0.283779</td>
          <td>25.241606</td>
          <td>0.318183</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.767258</td>
          <td>0.201921</td>
          <td>26.998556</td>
          <td>0.221286</td>
          <td>26.712362</td>
          <td>0.277596</td>
          <td>25.518059</td>
          <td>0.188044</td>
          <td>24.987506</td>
          <td>0.261249</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.996300</td>
          <td>1.158841</td>
          <td>26.992219</td>
          <td>0.247318</td>
          <td>26.575338</td>
          <td>0.157598</td>
          <td>25.962387</td>
          <td>0.150948</td>
          <td>25.341426</td>
          <td>0.164810</td>
          <td>25.706212</td>
          <td>0.467210</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.960383</td>
          <td>0.590516</td>
          <td>26.568874</td>
          <td>0.170369</td>
          <td>26.230343</td>
          <td>0.114563</td>
          <td>25.570509</td>
          <td>0.105169</td>
          <td>24.945030</td>
          <td>0.114666</td>
          <td>25.017052</td>
          <td>0.266929</td>
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
          <td>26.612210</td>
          <td>0.411142</td>
          <td>26.771834</td>
          <td>0.174769</td>
          <td>26.096266</td>
          <td>0.085843</td>
          <td>25.320747</td>
          <td>0.070564</td>
          <td>25.005960</td>
          <td>0.101989</td>
          <td>24.835320</td>
          <td>0.194242</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.226652</td>
          <td>0.644598</td>
          <td>28.469620</td>
          <td>0.658138</td>
          <td>27.251044</td>
          <td>0.231665</td>
          <td>27.303171</td>
          <td>0.377150</td>
          <td>26.574118</td>
          <td>0.378256</td>
          <td>25.262633</td>
          <td>0.276896</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.909284</td>
          <td>0.088925</td>
          <td>24.821480</td>
          <td>0.030172</td>
          <td>23.880299</td>
          <td>0.021631</td>
          <td>23.136475</td>
          <td>0.021363</td>
          <td>22.878329</td>
          <td>0.038153</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.197908</td>
          <td>0.718960</td>
          <td>29.618248</td>
          <td>1.493705</td>
          <td>28.438441</td>
          <td>0.695273</td>
          <td>26.756354</td>
          <td>0.301835</td>
          <td>26.522462</td>
          <td>0.441993</td>
          <td>25.575975</td>
          <td>0.435688</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.447851</td>
          <td>0.362329</td>
          <td>25.759987</td>
          <td>0.072617</td>
          <td>25.528270</td>
          <td>0.051984</td>
          <td>24.732878</td>
          <td>0.041943</td>
          <td>24.415195</td>
          <td>0.060640</td>
          <td>23.642563</td>
          <td>0.069059</td>
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
          <td>26.315201</td>
          <td>0.342907</td>
          <td>26.438373</td>
          <td>0.140527</td>
          <td>26.069006</td>
          <td>0.090679</td>
          <td>25.914601</td>
          <td>0.128948</td>
          <td>26.171566</td>
          <td>0.294935</td>
          <td>25.432351</td>
          <td>0.341406</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.896510</td>
          <td>0.513764</td>
          <td>27.240887</td>
          <td>0.261922</td>
          <td>26.802771</td>
          <td>0.161214</td>
          <td>26.525316</td>
          <td>0.203832</td>
          <td>26.434687</td>
          <td>0.343846</td>
          <td>25.394800</td>
          <td>0.312610</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.212764</td>
          <td>1.222008</td>
          <td>27.272996</td>
          <td>0.276035</td>
          <td>27.053982</td>
          <td>0.205732</td>
          <td>26.260077</td>
          <td>0.168302</td>
          <td>26.282513</td>
          <td>0.313757</td>
          <td>25.532567</td>
          <td>0.359358</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.537156</td>
          <td>0.359315</td>
          <td>26.621665</td>
          <td>0.151924</td>
          <td>25.889077</td>
          <td>0.130868</td>
          <td>25.669799</td>
          <td>0.201800</td>
          <td>26.220870</td>
          <td>0.634463</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.140500</td>
          <td>0.290677</td>
          <td>26.544822</td>
          <td>0.148795</td>
          <td>26.200711</td>
          <td>0.097822</td>
          <td>25.791001</td>
          <td>0.111145</td>
          <td>25.074050</td>
          <td>0.112493</td>
          <td>25.168028</td>
          <td>0.265898</td>
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
