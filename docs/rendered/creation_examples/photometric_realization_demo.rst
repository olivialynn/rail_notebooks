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

    <pzflow.flow.Flow at 0x7f1372cc7970>



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
          <td>28.325350</td>
          <td>1.272132</td>
          <td>26.848922</td>
          <td>0.186533</td>
          <td>26.043958</td>
          <td>0.081965</td>
          <td>25.244336</td>
          <td>0.065938</td>
          <td>25.122366</td>
          <td>0.112891</td>
          <td>24.543003</td>
          <td>0.151480</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.285068</td>
          <td>1.244500</td>
          <td>27.967140</td>
          <td>0.457683</td>
          <td>27.255923</td>
          <td>0.232396</td>
          <td>27.436071</td>
          <td>0.417481</td>
          <td>25.872723</td>
          <td>0.214440</td>
          <td>26.288268</td>
          <td>0.605097</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.226353</td>
          <td>0.303768</td>
          <td>26.003205</td>
          <td>0.089855</td>
          <td>24.786439</td>
          <td>0.026957</td>
          <td>23.863018</td>
          <td>0.019607</td>
          <td>23.113086</td>
          <td>0.019338</td>
          <td>22.809056</td>
          <td>0.032945</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.266295</td>
          <td>0.570056</td>
          <td>27.216580</td>
          <td>0.224934</td>
          <td>26.543205</td>
          <td>0.203528</td>
          <td>25.914248</td>
          <td>0.221990</td>
          <td>25.060473</td>
          <td>0.234383</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.704534</td>
          <td>0.441023</td>
          <td>25.689997</td>
          <td>0.068178</td>
          <td>25.440670</td>
          <td>0.048024</td>
          <td>24.841088</td>
          <td>0.046100</td>
          <td>24.337813</td>
          <td>0.056535</td>
          <td>23.812079</td>
          <td>0.080103</td>
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
          <td>26.573374</td>
          <td>0.399036</td>
          <td>26.316980</td>
          <td>0.118204</td>
          <td>25.994600</td>
          <td>0.078472</td>
          <td>26.330141</td>
          <td>0.169994</td>
          <td>26.318181</td>
          <td>0.308798</td>
          <td>25.375790</td>
          <td>0.303120</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.990881</td>
          <td>1.771274</td>
          <td>26.882924</td>
          <td>0.191960</td>
          <td>26.854061</td>
          <td>0.165753</td>
          <td>26.692145</td>
          <td>0.230442</td>
          <td>26.021426</td>
          <td>0.242597</td>
          <td>25.582115</td>
          <td>0.357066</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.759578</td>
          <td>0.459685</td>
          <td>27.518446</td>
          <td>0.323376</td>
          <td>26.916826</td>
          <td>0.174846</td>
          <td>26.290219</td>
          <td>0.164309</td>
          <td>26.033008</td>
          <td>0.244923</td>
          <td>25.776281</td>
          <td>0.415039</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.643566</td>
          <td>0.850635</td>
          <td>29.227444</td>
          <td>1.067535</td>
          <td>26.714745</td>
          <td>0.147119</td>
          <td>25.921742</td>
          <td>0.119611</td>
          <td>25.999337</td>
          <td>0.238215</td>
          <td>27.476591</td>
          <td>1.272431</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.543822</td>
          <td>0.390047</td>
          <td>26.550222</td>
          <td>0.144609</td>
          <td>26.042552</td>
          <td>0.081863</td>
          <td>25.739760</td>
          <td>0.102050</td>
          <td>25.443696</td>
          <td>0.149086</td>
          <td>25.102270</td>
          <td>0.242616</td>
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
          <td>27.384532</td>
          <td>0.784261</td>
          <td>26.644161</td>
          <td>0.180023</td>
          <td>25.953155</td>
          <td>0.088985</td>
          <td>25.342330</td>
          <td>0.085195</td>
          <td>25.274806</td>
          <td>0.150996</td>
          <td>25.152045</td>
          <td>0.294982</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>30.589239</td>
          <td>3.318944</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.273837</td>
          <td>0.274290</td>
          <td>27.272246</td>
          <td>0.426477</td>
          <td>26.230241</td>
          <td>0.333310</td>
          <td>25.677801</td>
          <td>0.444977</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.714196</td>
          <td>0.497970</td>
          <td>25.874522</td>
          <td>0.094494</td>
          <td>24.804518</td>
          <td>0.032940</td>
          <td>23.869038</td>
          <td>0.023783</td>
          <td>23.140987</td>
          <td>0.023712</td>
          <td>22.772866</td>
          <td>0.038656</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.744548</td>
          <td>1.019588</td>
          <td>28.041942</td>
          <td>0.573100</td>
          <td>28.688018</td>
          <td>0.822583</td>
          <td>27.930348</td>
          <td>0.724000</td>
          <td>25.988988</td>
          <td>0.292070</td>
          <td>25.181698</td>
          <td>0.321645</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.687041</td>
          <td>0.481196</td>
          <td>25.769641</td>
          <td>0.084483</td>
          <td>25.360922</td>
          <td>0.052718</td>
          <td>24.746307</td>
          <td>0.050285</td>
          <td>24.426714</td>
          <td>0.072031</td>
          <td>23.636371</td>
          <td>0.081169</td>
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
          <td>26.877040</td>
          <td>0.560196</td>
          <td>26.336078</td>
          <td>0.140961</td>
          <td>26.075756</td>
          <td>0.101194</td>
          <td>25.843901</td>
          <td>0.134882</td>
          <td>26.043988</td>
          <td>0.292669</td>
          <td>25.261226</td>
          <td>0.328287</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.626201</td>
          <td>0.917228</td>
          <td>27.477296</td>
          <td>0.357139</td>
          <td>26.623653</td>
          <td>0.159946</td>
          <td>26.575617</td>
          <td>0.246183</td>
          <td>26.377851</td>
          <td>0.375605</td>
          <td>25.182369</td>
          <td>0.303452</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.233247</td>
          <td>0.714455</td>
          <td>29.281895</td>
          <td>1.217087</td>
          <td>26.848186</td>
          <td>0.195119</td>
          <td>25.993862</td>
          <td>0.152189</td>
          <td>25.757443</td>
          <td>0.229748</td>
          <td>24.789080</td>
          <td>0.221804</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.790143</td>
          <td>0.463853</td>
          <td>26.737975</td>
          <td>0.180995</td>
          <td>25.642208</td>
          <td>0.114446</td>
          <td>25.782518</td>
          <td>0.238711</td>
          <td>25.530084</td>
          <td>0.408844</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.926348</td>
          <td>1.101564</td>
          <td>26.405398</td>
          <td>0.148168</td>
          <td>26.133532</td>
          <td>0.105283</td>
          <td>25.550869</td>
          <td>0.103379</td>
          <td>25.179260</td>
          <td>0.140462</td>
          <td>25.035882</td>
          <td>0.271056</td>
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
          <td>25.876514</td>
          <td>0.228405</td>
          <td>27.124144</td>
          <td>0.234811</td>
          <td>26.062229</td>
          <td>0.083307</td>
          <td>25.411279</td>
          <td>0.076446</td>
          <td>24.992714</td>
          <td>0.100813</td>
          <td>25.012296</td>
          <td>0.225235</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.510702</td>
          <td>0.676997</td>
          <td>27.998579</td>
          <td>0.420678</td>
          <td>26.895940</td>
          <td>0.272692</td>
          <td>25.860131</td>
          <td>0.212388</td>
          <td>25.934655</td>
          <td>0.468264</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.346963</td>
          <td>0.352427</td>
          <td>25.915468</td>
          <td>0.089409</td>
          <td>24.781937</td>
          <td>0.029144</td>
          <td>23.871086</td>
          <td>0.021461</td>
          <td>23.121035</td>
          <td>0.021084</td>
          <td>22.779911</td>
          <td>0.034976</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.904007</td>
          <td>1.008817</td>
          <td>27.699039</td>
          <td>0.406562</td>
          <td>27.098450</td>
          <td>0.395211</td>
          <td>25.573103</td>
          <td>0.206751</td>
          <td>25.452479</td>
          <td>0.396425</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.271008</td>
          <td>0.315093</td>
          <td>25.798824</td>
          <td>0.075150</td>
          <td>25.474823</td>
          <td>0.049574</td>
          <td>24.834349</td>
          <td>0.045894</td>
          <td>24.416560</td>
          <td>0.060714</td>
          <td>23.780039</td>
          <td>0.077985</td>
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
          <td>26.418482</td>
          <td>0.371805</td>
          <td>26.638496</td>
          <td>0.166794</td>
          <td>25.990990</td>
          <td>0.084662</td>
          <td>26.129677</td>
          <td>0.155189</td>
          <td>25.750647</td>
          <td>0.208676</td>
          <td>26.883711</td>
          <td>0.953175</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.324988</td>
          <td>1.280902</td>
          <td>27.035749</td>
          <td>0.221158</td>
          <td>26.882487</td>
          <td>0.172545</td>
          <td>26.527415</td>
          <td>0.204191</td>
          <td>26.155723</td>
          <td>0.274977</td>
          <td>25.290556</td>
          <td>0.287474</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.831890</td>
          <td>0.499430</td>
          <td>27.391725</td>
          <td>0.303805</td>
          <td>27.094798</td>
          <td>0.212878</td>
          <td>26.207431</td>
          <td>0.160912</td>
          <td>25.993498</td>
          <td>0.248186</td>
          <td>26.001554</td>
          <td>0.513119</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.351002</td>
          <td>0.655937</td>
          <td>26.712381</td>
          <td>0.164180</td>
          <td>25.732731</td>
          <td>0.114257</td>
          <td>25.841120</td>
          <td>0.232781</td>
          <td>25.563917</td>
          <td>0.391059</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.547121</td>
          <td>0.815601</td>
          <td>26.465012</td>
          <td>0.138927</td>
          <td>26.280511</td>
          <td>0.104902</td>
          <td>25.524307</td>
          <td>0.087980</td>
          <td>25.173356</td>
          <td>0.122644</td>
          <td>25.278786</td>
          <td>0.290910</td>
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
