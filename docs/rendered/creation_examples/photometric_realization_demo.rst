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

    <pzflow.flow.Flow at 0x7f83d7a0ec20>



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
          <td>30.448421</td>
          <td>3.059184</td>
          <td>26.976544</td>
          <td>0.207657</td>
          <td>25.914774</td>
          <td>0.073127</td>
          <td>25.404409</td>
          <td>0.075973</td>
          <td>25.059303</td>
          <td>0.106846</td>
          <td>25.086660</td>
          <td>0.239511</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.302440</td>
          <td>1.256377</td>
          <td>30.139526</td>
          <td>1.720004</td>
          <td>28.559815</td>
          <td>0.633710</td>
          <td>26.890226</td>
          <td>0.271177</td>
          <td>27.673376</td>
          <td>0.828863</td>
          <td>26.567682</td>
          <td>0.733326</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.400944</td>
          <td>0.725678</td>
          <td>25.928793</td>
          <td>0.084165</td>
          <td>24.782350</td>
          <td>0.026861</td>
          <td>23.838347</td>
          <td>0.019203</td>
          <td>23.148638</td>
          <td>0.019928</td>
          <td>22.837389</td>
          <td>0.033778</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.651083</td>
          <td>1.506908</td>
          <td>28.373151</td>
          <td>0.614960</td>
          <td>28.277874</td>
          <td>0.517998</td>
          <td>26.664741</td>
          <td>0.225261</td>
          <td>25.928247</td>
          <td>0.224589</td>
          <td>26.104612</td>
          <td>0.530419</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.817710</td>
          <td>0.480078</td>
          <td>25.797047</td>
          <td>0.074939</td>
          <td>25.339973</td>
          <td>0.043918</td>
          <td>24.802542</td>
          <td>0.044549</td>
          <td>24.485418</td>
          <td>0.064443</td>
          <td>23.641017</td>
          <td>0.068862</td>
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
          <td>28.576955</td>
          <td>1.451773</td>
          <td>26.306336</td>
          <td>0.117116</td>
          <td>25.958410</td>
          <td>0.076003</td>
          <td>25.980736</td>
          <td>0.125897</td>
          <td>25.962345</td>
          <td>0.231034</td>
          <td>26.369057</td>
          <td>0.640342</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.862035</td>
          <td>0.974651</td>
          <td>27.316827</td>
          <td>0.274967</td>
          <td>26.675893</td>
          <td>0.142283</td>
          <td>26.563404</td>
          <td>0.207002</td>
          <td>26.275354</td>
          <td>0.298361</td>
          <td>25.797378</td>
          <td>0.421783</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.658141</td>
          <td>1.512207</td>
          <td>26.741388</td>
          <td>0.170288</td>
          <td>26.978911</td>
          <td>0.184292</td>
          <td>26.689050</td>
          <td>0.229852</td>
          <td>26.003174</td>
          <td>0.238971</td>
          <td>25.885390</td>
          <td>0.450886</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.946348</td>
          <td>1.025369</td>
          <td>27.529012</td>
          <td>0.326104</td>
          <td>26.741307</td>
          <td>0.150513</td>
          <td>25.879912</td>
          <td>0.115336</td>
          <td>25.506199</td>
          <td>0.157291</td>
          <td>25.803190</td>
          <td>0.423656</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.545098</td>
          <td>0.798290</td>
          <td>26.239159</td>
          <td>0.110463</td>
          <td>26.029914</td>
          <td>0.080956</td>
          <td>25.575679</td>
          <td>0.088360</td>
          <td>25.534330</td>
          <td>0.161119</td>
          <td>24.840680</td>
          <td>0.195095</td>
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
          <td>27.342509</td>
          <td>0.762864</td>
          <td>26.552671</td>
          <td>0.166569</td>
          <td>25.919498</td>
          <td>0.086388</td>
          <td>25.318324</td>
          <td>0.083413</td>
          <td>25.198904</td>
          <td>0.141461</td>
          <td>25.161310</td>
          <td>0.297191</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.352059</td>
          <td>0.678455</td>
          <td>27.375612</td>
          <td>0.297829</td>
          <td>27.544721</td>
          <td>0.522642</td>
          <td>26.230574</td>
          <td>0.333398</td>
          <td>26.608464</td>
          <td>0.852784</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.208832</td>
          <td>0.706731</td>
          <td>25.958026</td>
          <td>0.101658</td>
          <td>24.819341</td>
          <td>0.033373</td>
          <td>23.873284</td>
          <td>0.023871</td>
          <td>23.143983</td>
          <td>0.023774</td>
          <td>22.905133</td>
          <td>0.043457</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.608437</td>
          <td>0.841777</td>
          <td>27.556483</td>
          <td>0.365198</td>
          <td>26.337950</td>
          <td>0.214956</td>
          <td>26.477036</td>
          <td>0.428343</td>
          <td>25.056240</td>
          <td>0.290863</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.343401</td>
          <td>0.370417</td>
          <td>25.762318</td>
          <td>0.083941</td>
          <td>25.469317</td>
          <td>0.058040</td>
          <td>24.716506</td>
          <td>0.048973</td>
          <td>24.324671</td>
          <td>0.065812</td>
          <td>23.550622</td>
          <td>0.075254</td>
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
          <td>26.513473</td>
          <td>0.428087</td>
          <td>26.314174</td>
          <td>0.138328</td>
          <td>26.100848</td>
          <td>0.103441</td>
          <td>26.187263</td>
          <td>0.180942</td>
          <td>25.519089</td>
          <td>0.189697</td>
          <td>25.545808</td>
          <td>0.409962</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.064264</td>
          <td>0.633018</td>
          <td>27.202217</td>
          <td>0.286872</td>
          <td>26.777827</td>
          <td>0.182355</td>
          <td>26.509295</td>
          <td>0.233068</td>
          <td>26.288719</td>
          <td>0.350305</td>
          <td>25.561241</td>
          <td>0.408637</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.833579</td>
          <td>1.760949</td>
          <td>26.929713</td>
          <td>0.231195</td>
          <td>27.004357</td>
          <td>0.222356</td>
          <td>26.790909</td>
          <td>0.295801</td>
          <td>25.980240</td>
          <td>0.275857</td>
          <td>25.614682</td>
          <td>0.428946</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.617756</td>
          <td>0.407013</td>
          <td>26.589745</td>
          <td>0.159551</td>
          <td>25.907873</td>
          <td>0.144042</td>
          <td>25.733556</td>
          <td>0.229234</td>
          <td>25.396145</td>
          <td>0.368594</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.782676</td>
          <td>1.718400</td>
          <td>26.588546</td>
          <td>0.173240</td>
          <td>26.152318</td>
          <td>0.107026</td>
          <td>25.551267</td>
          <td>0.103415</td>
          <td>24.958893</td>
          <td>0.116058</td>
          <td>25.235621</td>
          <td>0.318404</td>
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
          <td>26.621895</td>
          <td>0.414200</td>
          <td>26.703585</td>
          <td>0.164913</td>
          <td>26.156345</td>
          <td>0.090504</td>
          <td>25.295265</td>
          <td>0.068990</td>
          <td>25.073723</td>
          <td>0.108214</td>
          <td>25.092221</td>
          <td>0.240644</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.689268</td>
          <td>0.436204</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.859166</td>
          <td>0.377839</td>
          <td>27.339243</td>
          <td>0.387855</td>
          <td>27.811143</td>
          <td>0.905182</td>
          <td>26.019338</td>
          <td>0.498675</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.543229</td>
          <td>0.410367</td>
          <td>25.943931</td>
          <td>0.091671</td>
          <td>24.756623</td>
          <td>0.028506</td>
          <td>23.870691</td>
          <td>0.021454</td>
          <td>23.131637</td>
          <td>0.021275</td>
          <td>22.823568</td>
          <td>0.036350</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.023903</td>
          <td>1.953325</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.613292</td>
          <td>0.380516</td>
          <td>26.753797</td>
          <td>0.301216</td>
          <td>25.651580</td>
          <td>0.220748</td>
          <td>25.934178</td>
          <td>0.567534</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.066455</td>
          <td>0.267189</td>
          <td>25.682219</td>
          <td>0.067795</td>
          <td>25.445982</td>
          <td>0.048321</td>
          <td>24.774865</td>
          <td>0.043534</td>
          <td>24.373662</td>
          <td>0.058447</td>
          <td>23.787196</td>
          <td>0.078480</td>
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
          <td>25.938511</td>
          <td>0.253285</td>
          <td>26.427977</td>
          <td>0.139274</td>
          <td>26.090173</td>
          <td>0.092382</td>
          <td>26.053462</td>
          <td>0.145364</td>
          <td>26.091951</td>
          <td>0.276536</td>
          <td>25.805498</td>
          <td>0.455277</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.448063</td>
          <td>0.365787</td>
          <td>27.152561</td>
          <td>0.243612</td>
          <td>26.971761</td>
          <td>0.186107</td>
          <td>26.263055</td>
          <td>0.163262</td>
          <td>26.629350</td>
          <td>0.400192</td>
          <td>25.149168</td>
          <td>0.256219</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.429358</td>
          <td>0.759127</td>
          <td>27.590341</td>
          <td>0.355678</td>
          <td>26.894182</td>
          <td>0.179813</td>
          <td>26.402989</td>
          <td>0.189976</td>
          <td>25.938626</td>
          <td>0.237209</td>
          <td>25.347256</td>
          <td>0.310295</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.612056</td>
          <td>0.884697</td>
          <td>26.962637</td>
          <td>0.225877</td>
          <td>26.617768</td>
          <td>0.151417</td>
          <td>25.810829</td>
          <td>0.122287</td>
          <td>25.596328</td>
          <td>0.189703</td>
          <td>25.554147</td>
          <td>0.388115</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>30.891900</td>
          <td>3.510426</td>
          <td>26.664965</td>
          <td>0.164895</td>
          <td>26.170892</td>
          <td>0.095296</td>
          <td>25.405411</td>
          <td>0.079227</td>
          <td>25.466647</td>
          <td>0.157923</td>
          <td>24.819697</td>
          <td>0.199235</td>
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
