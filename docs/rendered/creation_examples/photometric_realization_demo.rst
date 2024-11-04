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

    <pzflow.flow.Flow at 0x7f914ee1f4c0>



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
          <td>26.641158</td>
          <td>0.156336</td>
          <td>26.128692</td>
          <td>0.088318</td>
          <td>25.343723</td>
          <td>0.072003</td>
          <td>24.973261</td>
          <td>0.099096</td>
          <td>24.935399</td>
          <td>0.211227</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.617491</td>
          <td>1.327466</td>
          <td>27.742619</td>
          <td>0.344591</td>
          <td>27.459918</td>
          <td>0.425147</td>
          <td>26.379895</td>
          <td>0.324393</td>
          <td>25.931870</td>
          <td>0.466899</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.002148</td>
          <td>0.253285</td>
          <td>25.842939</td>
          <td>0.078035</td>
          <td>24.764202</td>
          <td>0.026439</td>
          <td>23.885387</td>
          <td>0.019982</td>
          <td>23.125514</td>
          <td>0.019542</td>
          <td>22.786944</td>
          <td>0.032310</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.497488</td>
          <td>0.773779</td>
          <td>28.704183</td>
          <td>0.770556</td>
          <td>28.327220</td>
          <td>0.536979</td>
          <td>26.677786</td>
          <td>0.227714</td>
          <td>26.607210</td>
          <td>0.387767</td>
          <td>25.053289</td>
          <td>0.232994</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.997454</td>
          <td>0.252312</td>
          <td>25.697157</td>
          <td>0.068611</td>
          <td>25.491136</td>
          <td>0.050225</td>
          <td>24.881123</td>
          <td>0.047768</td>
          <td>24.500485</td>
          <td>0.065309</td>
          <td>23.803151</td>
          <td>0.079474</td>
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
          <td>26.558532</td>
          <td>0.394500</td>
          <td>26.533256</td>
          <td>0.142515</td>
          <td>26.324427</td>
          <td>0.104865</td>
          <td>26.289285</td>
          <td>0.164178</td>
          <td>25.455703</td>
          <td>0.150630</td>
          <td>25.969241</td>
          <td>0.480101</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.445049</td>
          <td>0.361219</td>
          <td>26.970316</td>
          <td>0.206577</td>
          <td>26.763122</td>
          <td>0.153356</td>
          <td>26.552974</td>
          <td>0.205202</td>
          <td>26.069274</td>
          <td>0.252337</td>
          <td>25.703496</td>
          <td>0.392453</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.183110</td>
          <td>0.246487</td>
          <td>26.783768</td>
          <td>0.156092</td>
          <td>26.688574</td>
          <td>0.229761</td>
          <td>25.780069</td>
          <td>0.198427</td>
          <td>25.622581</td>
          <td>0.368553</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.927991</td>
          <td>0.520737</td>
          <td>27.493270</td>
          <td>0.316953</td>
          <td>26.681621</td>
          <td>0.142987</td>
          <td>25.874911</td>
          <td>0.114834</td>
          <td>25.617614</td>
          <td>0.172968</td>
          <td>25.355831</td>
          <td>0.298295</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.453817</td>
          <td>0.133082</td>
          <td>26.100189</td>
          <td>0.086129</td>
          <td>25.577947</td>
          <td>0.088537</td>
          <td>25.378136</td>
          <td>0.140912</td>
          <td>25.093007</td>
          <td>0.240769</td>
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
          <td>26.722845</td>
          <td>0.494031</td>
          <td>26.845687</td>
          <td>0.213256</td>
          <td>26.179868</td>
          <td>0.108543</td>
          <td>25.326307</td>
          <td>0.084001</td>
          <td>24.745880</td>
          <td>0.095388</td>
          <td>24.988112</td>
          <td>0.258202</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.494624</td>
          <td>0.842338</td>
          <td>27.719326</td>
          <td>0.429194</td>
          <td>27.347292</td>
          <td>0.291108</td>
          <td>27.746100</td>
          <td>0.604017</td>
          <td>26.957066</td>
          <td>0.577110</td>
          <td>25.828210</td>
          <td>0.497882</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.283692</td>
          <td>0.358867</td>
          <td>25.882171</td>
          <td>0.095129</td>
          <td>24.763068</td>
          <td>0.031761</td>
          <td>23.908831</td>
          <td>0.024615</td>
          <td>23.170299</td>
          <td>0.024320</td>
          <td>22.846244</td>
          <td>0.041249</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.858533</td>
          <td>1.090385</td>
          <td>28.438281</td>
          <td>0.753343</td>
          <td>27.311431</td>
          <td>0.300709</td>
          <td>26.481982</td>
          <td>0.242234</td>
          <td>25.810763</td>
          <td>0.252635</td>
          <td>24.979073</td>
          <td>0.273234</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.282403</td>
          <td>0.353168</td>
          <td>25.732552</td>
          <td>0.081771</td>
          <td>25.439364</td>
          <td>0.056518</td>
          <td>24.798401</td>
          <td>0.052664</td>
          <td>24.416167</td>
          <td>0.071362</td>
          <td>23.754619</td>
          <td>0.090074</td>
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
          <td>27.009751</td>
          <td>0.615592</td>
          <td>26.480765</td>
          <td>0.159576</td>
          <td>26.255046</td>
          <td>0.118330</td>
          <td>26.019762</td>
          <td>0.156896</td>
          <td>25.894671</td>
          <td>0.259230</td>
          <td>26.556386</td>
          <td>0.837754</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.872788</td>
          <td>0.552636</td>
          <td>26.967102</td>
          <td>0.236716</td>
          <td>26.734512</td>
          <td>0.175783</td>
          <td>26.568091</td>
          <td>0.244662</td>
          <td>25.857643</td>
          <td>0.247557</td>
          <td>25.064707</td>
          <td>0.275943</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.961316</td>
          <td>0.519104</td>
          <td>26.752479</td>
          <td>0.179971</td>
          <td>26.278154</td>
          <td>0.193788</td>
          <td>26.384896</td>
          <td>0.380552</td>
          <td>25.152540</td>
          <td>0.298671</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.842455</td>
          <td>0.550012</td>
          <td>28.670415</td>
          <td>0.855338</td>
          <td>27.098230</td>
          <td>0.244576</td>
          <td>25.886303</td>
          <td>0.141393</td>
          <td>25.789056</td>
          <td>0.240003</td>
          <td>27.078487</td>
          <td>1.156939</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.331060</td>
          <td>0.761472</td>
          <td>26.826090</td>
          <td>0.211609</td>
          <td>26.002163</td>
          <td>0.093839</td>
          <td>25.671918</td>
          <td>0.114898</td>
          <td>25.184998</td>
          <td>0.141158</td>
          <td>24.889482</td>
          <td>0.240404</td>
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
          <td>26.607874</td>
          <td>0.151961</td>
          <td>25.980046</td>
          <td>0.077480</td>
          <td>25.324819</td>
          <td>0.070819</td>
          <td>24.958585</td>
          <td>0.097842</td>
          <td>24.827665</td>
          <td>0.192993</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.060729</td>
          <td>0.491110</td>
          <td>27.342170</td>
          <td>0.249758</td>
          <td>26.922272</td>
          <td>0.278590</td>
          <td>25.980240</td>
          <td>0.234692</td>
          <td>28.006091</td>
          <td>1.664533</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.519003</td>
          <td>0.402813</td>
          <td>26.091136</td>
          <td>0.104281</td>
          <td>24.838564</td>
          <td>0.030628</td>
          <td>23.866192</td>
          <td>0.021372</td>
          <td>23.174258</td>
          <td>0.022064</td>
          <td>22.832207</td>
          <td>0.036629</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.642096</td>
          <td>0.425339</td>
          <td>28.291643</td>
          <td>0.628338</td>
          <td>26.615043</td>
          <td>0.269223</td>
          <td>26.395942</td>
          <td>0.401326</td>
          <td>25.175205</td>
          <td>0.318925</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.569932</td>
          <td>0.398323</td>
          <td>25.738828</td>
          <td>0.071273</td>
          <td>25.441236</td>
          <td>0.048118</td>
          <td>24.788719</td>
          <td>0.044073</td>
          <td>24.375443</td>
          <td>0.058539</td>
          <td>23.719302</td>
          <td>0.073911</td>
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
          <td>26.420919</td>
          <td>0.372511</td>
          <td>26.537845</td>
          <td>0.153057</td>
          <td>26.016740</td>
          <td>0.086604</td>
          <td>26.044728</td>
          <td>0.144276</td>
          <td>25.645607</td>
          <td>0.191054</td>
          <td>25.531831</td>
          <td>0.369137</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.535813</td>
          <td>0.800252</td>
          <td>26.887850</td>
          <td>0.195421</td>
          <td>27.307607</td>
          <td>0.246310</td>
          <td>26.476071</td>
          <td>0.195571</td>
          <td>25.654131</td>
          <td>0.181267</td>
          <td>25.481845</td>
          <td>0.335033</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.054177</td>
          <td>0.586673</td>
          <td>27.201215</td>
          <td>0.260351</td>
          <td>26.820865</td>
          <td>0.168957</td>
          <td>26.492780</td>
          <td>0.204876</td>
          <td>26.188668</td>
          <td>0.290974</td>
          <td>25.996154</td>
          <td>0.511089</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.707772</td>
          <td>0.474244</td>
          <td>27.513261</td>
          <td>0.352642</td>
          <td>26.436248</td>
          <td>0.129491</td>
          <td>25.778397</td>
          <td>0.118889</td>
          <td>25.571018</td>
          <td>0.185692</td>
          <td>25.972510</td>
          <td>0.531521</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.157799</td>
          <td>1.179784</td>
          <td>26.604633</td>
          <td>0.156617</td>
          <td>26.136554</td>
          <td>0.092465</td>
          <td>25.823654</td>
          <td>0.114354</td>
          <td>25.342820</td>
          <td>0.142000</td>
          <td>24.922807</td>
          <td>0.217192</td>
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
