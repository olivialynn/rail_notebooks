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

    <pzflow.flow.Flow at 0x7fc2c5f83a90>



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
          <td>27.988386</td>
          <td>1.051240</td>
          <td>26.896050</td>
          <td>0.194093</td>
          <td>26.065806</td>
          <td>0.083559</td>
          <td>25.371825</td>
          <td>0.073816</td>
          <td>25.295575</td>
          <td>0.131217</td>
          <td>24.853934</td>
          <td>0.197282</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.567245</td>
          <td>2.256350</td>
          <td>28.678359</td>
          <td>0.757512</td>
          <td>28.050547</td>
          <td>0.437279</td>
          <td>27.328876</td>
          <td>0.384416</td>
          <td>26.170191</td>
          <td>0.274028</td>
          <td>27.598826</td>
          <td>1.358243</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.500153</td>
          <td>0.377073</td>
          <td>25.867468</td>
          <td>0.079740</td>
          <td>24.740643</td>
          <td>0.025903</td>
          <td>23.889881</td>
          <td>0.020058</td>
          <td>23.131959</td>
          <td>0.019649</td>
          <td>22.818533</td>
          <td>0.033222</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.966232</td>
          <td>0.457371</td>
          <td>27.934946</td>
          <td>0.400320</td>
          <td>26.391081</td>
          <td>0.179025</td>
          <td>26.000244</td>
          <td>0.238393</td>
          <td>25.638731</td>
          <td>0.373223</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.063839</td>
          <td>0.266377</td>
          <td>25.801814</td>
          <td>0.075255</td>
          <td>25.403024</td>
          <td>0.046446</td>
          <td>24.835025</td>
          <td>0.045852</td>
          <td>24.390862</td>
          <td>0.059260</td>
          <td>23.704258</td>
          <td>0.072826</td>
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
          <td>27.201886</td>
          <td>0.633259</td>
          <td>26.514879</td>
          <td>0.140278</td>
          <td>26.081582</td>
          <td>0.084729</td>
          <td>26.384398</td>
          <td>0.178013</td>
          <td>25.734983</td>
          <td>0.191037</td>
          <td>25.842387</td>
          <td>0.436469</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.515714</td>
          <td>0.381654</td>
          <td>26.809337</td>
          <td>0.180394</td>
          <td>26.878351</td>
          <td>0.169219</td>
          <td>26.442188</td>
          <td>0.186937</td>
          <td>26.155039</td>
          <td>0.270669</td>
          <td>26.053994</td>
          <td>0.511142</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.416746</td>
          <td>0.733404</td>
          <td>26.900818</td>
          <td>0.194874</td>
          <td>27.089956</td>
          <td>0.202364</td>
          <td>26.420146</td>
          <td>0.183486</td>
          <td>25.902466</td>
          <td>0.219824</td>
          <td>25.424645</td>
          <td>0.315215</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.511196</td>
          <td>0.321515</td>
          <td>26.277763</td>
          <td>0.100668</td>
          <td>25.802790</td>
          <td>0.107832</td>
          <td>25.492242</td>
          <td>0.155423</td>
          <td>25.342152</td>
          <td>0.295028</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.595895</td>
          <td>0.406001</td>
          <td>26.406091</td>
          <td>0.127702</td>
          <td>26.031236</td>
          <td>0.081050</td>
          <td>25.656416</td>
          <td>0.094859</td>
          <td>25.239523</td>
          <td>0.124998</td>
          <td>24.900145</td>
          <td>0.205086</td>
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
          <td>28.591617</td>
          <td>1.562828</td>
          <td>26.989096</td>
          <td>0.240202</td>
          <td>26.174002</td>
          <td>0.107988</td>
          <td>25.383115</td>
          <td>0.088309</td>
          <td>25.050806</td>
          <td>0.124465</td>
          <td>25.080306</td>
          <td>0.278355</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.460204</td>
          <td>0.351255</td>
          <td>26.838140</td>
          <td>0.191156</td>
          <td>27.187153</td>
          <td>0.399575</td>
          <td>28.090291</td>
          <td>1.189830</td>
          <td>28.883403</td>
          <td>2.573240</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.449326</td>
          <td>0.407999</td>
          <td>26.065592</td>
          <td>0.111660</td>
          <td>24.781285</td>
          <td>0.032274</td>
          <td>23.881880</td>
          <td>0.024048</td>
          <td>23.155755</td>
          <td>0.024016</td>
          <td>22.864261</td>
          <td>0.041912</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.989711</td>
          <td>0.624575</td>
          <td>27.399810</td>
          <td>0.353663</td>
          <td>28.763206</td>
          <td>0.863163</td>
          <td>26.461842</td>
          <td>0.238241</td>
          <td>25.983507</td>
          <td>0.290781</td>
          <td>25.624731</td>
          <td>0.453455</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.061499</td>
          <td>0.296317</td>
          <td>25.818033</td>
          <td>0.088154</td>
          <td>25.468468</td>
          <td>0.057996</td>
          <td>24.786725</td>
          <td>0.052122</td>
          <td>24.392049</td>
          <td>0.069856</td>
          <td>23.700167</td>
          <td>0.085861</td>
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
          <td>27.966542</td>
          <td>1.133512</td>
          <td>26.406527</td>
          <td>0.149755</td>
          <td>26.328881</td>
          <td>0.126163</td>
          <td>25.966441</td>
          <td>0.149889</td>
          <td>25.535048</td>
          <td>0.192266</td>
          <td>26.203363</td>
          <td>0.662288</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.380766</td>
          <td>0.784178</td>
          <td>26.940509</td>
          <td>0.231567</td>
          <td>26.797026</td>
          <td>0.185340</td>
          <td>26.151650</td>
          <td>0.172631</td>
          <td>26.280461</td>
          <td>0.348036</td>
          <td>25.283380</td>
          <td>0.328938</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.227588</td>
          <td>0.711733</td>
          <td>27.269166</td>
          <td>0.304924</td>
          <td>26.670581</td>
          <td>0.167877</td>
          <td>26.120199</td>
          <td>0.169531</td>
          <td>25.791345</td>
          <td>0.236287</td>
          <td>25.128081</td>
          <td>0.292845</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.327366</td>
          <td>0.769029</td>
          <td>26.879707</td>
          <td>0.225364</td>
          <td>26.380022</td>
          <td>0.133234</td>
          <td>25.776181</td>
          <td>0.128566</td>
          <td>25.679599</td>
          <td>0.219183</td>
          <td>25.512299</td>
          <td>0.403299</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.190252</td>
          <td>0.330604</td>
          <td>26.555480</td>
          <td>0.168440</td>
          <td>25.959948</td>
          <td>0.090423</td>
          <td>25.544911</td>
          <td>0.102841</td>
          <td>25.331148</td>
          <td>0.160014</td>
          <td>24.784076</td>
          <td>0.220293</td>
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
          <td>27.788631</td>
          <td>0.931841</td>
          <td>26.658281</td>
          <td>0.158659</td>
          <td>25.982740</td>
          <td>0.077664</td>
          <td>25.371161</td>
          <td>0.073782</td>
          <td>24.965113</td>
          <td>0.098404</td>
          <td>24.790056</td>
          <td>0.186966</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.468541</td>
          <td>0.657647</td>
          <td>27.926287</td>
          <td>0.397989</td>
          <td>28.084303</td>
          <td>0.669442</td>
          <td>27.443973</td>
          <td>0.712968</td>
          <td>25.533311</td>
          <td>0.343920</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.239447</td>
          <td>0.323733</td>
          <td>25.915569</td>
          <td>0.089417</td>
          <td>24.765960</td>
          <td>0.028740</td>
          <td>23.860436</td>
          <td>0.021267</td>
          <td>23.120490</td>
          <td>0.021074</td>
          <td>22.771716</td>
          <td>0.034724</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.332200</td>
          <td>0.383390</td>
          <td>27.429346</td>
          <td>0.360897</td>
          <td>27.114097</td>
          <td>0.255348</td>
          <td>26.556625</td>
          <td>0.256674</td>
          <td>25.925906</td>
          <td>0.276620</td>
          <td>25.114878</td>
          <td>0.303900</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.389520</td>
          <td>0.346122</td>
          <td>25.758190</td>
          <td>0.072502</td>
          <td>25.497983</td>
          <td>0.050604</td>
          <td>24.878436</td>
          <td>0.047726</td>
          <td>24.472526</td>
          <td>0.063802</td>
          <td>23.672405</td>
          <td>0.070908</td>
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
          <td>26.234526</td>
          <td>0.321687</td>
          <td>26.208269</td>
          <td>0.115150</td>
          <td>26.026240</td>
          <td>0.087331</td>
          <td>26.199964</td>
          <td>0.164797</td>
          <td>25.652549</td>
          <td>0.192175</td>
          <td>25.727868</td>
          <td>0.429321</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.547224</td>
          <td>0.806215</td>
          <td>27.303556</td>
          <td>0.275644</td>
          <td>26.734236</td>
          <td>0.152029</td>
          <td>26.693527</td>
          <td>0.234494</td>
          <td>26.144082</td>
          <td>0.272385</td>
          <td>24.996313</td>
          <td>0.225859</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.090595</td>
          <td>0.602009</td>
          <td>27.004648</td>
          <td>0.221383</td>
          <td>26.645101</td>
          <td>0.145364</td>
          <td>26.160715</td>
          <td>0.154608</td>
          <td>25.968422</td>
          <td>0.243115</td>
          <td>25.244623</td>
          <td>0.285695</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.161467</td>
          <td>0.266033</td>
          <td>26.610124</td>
          <td>0.150427</td>
          <td>25.806877</td>
          <td>0.121868</td>
          <td>25.523817</td>
          <td>0.178420</td>
          <td>25.906442</td>
          <td>0.506431</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.613682</td>
          <td>0.421446</td>
          <td>26.443845</td>
          <td>0.136415</td>
          <td>26.055544</td>
          <td>0.086106</td>
          <td>25.643059</td>
          <td>0.097655</td>
          <td>25.425757</td>
          <td>0.152489</td>
          <td>24.822901</td>
          <td>0.199772</td>
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
