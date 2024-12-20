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

    <pzflow.flow.Flow at 0x7fb28b19d240>



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
          <td>26.725284</td>
          <td>0.167971</td>
          <td>26.066926</td>
          <td>0.083642</td>
          <td>25.333525</td>
          <td>0.071357</td>
          <td>25.021009</td>
          <td>0.103328</td>
          <td>25.072676</td>
          <td>0.236760</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.290770</td>
          <td>2.018609</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.767050</td>
          <td>0.351285</td>
          <td>26.870634</td>
          <td>0.266881</td>
          <td>26.364759</td>
          <td>0.320506</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.052041</td>
          <td>0.263828</td>
          <td>25.861665</td>
          <td>0.079333</td>
          <td>24.745155</td>
          <td>0.026005</td>
          <td>23.881835</td>
          <td>0.019922</td>
          <td>23.109728</td>
          <td>0.019284</td>
          <td>22.833423</td>
          <td>0.033661</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.547647</td>
          <td>0.799617</td>
          <td>28.208030</td>
          <td>0.546645</td>
          <td>28.171970</td>
          <td>0.479038</td>
          <td>26.757700</td>
          <td>0.243274</td>
          <td>26.346443</td>
          <td>0.315857</td>
          <td>25.770106</td>
          <td>0.413082</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.347013</td>
          <td>0.334409</td>
          <td>25.733237</td>
          <td>0.070834</td>
          <td>25.439281</td>
          <td>0.047965</td>
          <td>24.797484</td>
          <td>0.044350</td>
          <td>24.374095</td>
          <td>0.058385</td>
          <td>23.746042</td>
          <td>0.075566</td>
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
          <td>26.506088</td>
          <td>0.378814</td>
          <td>26.187212</td>
          <td>0.105569</td>
          <td>25.988895</td>
          <td>0.078077</td>
          <td>25.883873</td>
          <td>0.115734</td>
          <td>25.798202</td>
          <td>0.201472</td>
          <td>25.920378</td>
          <td>0.462898</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.114747</td>
          <td>0.232968</td>
          <td>26.917842</td>
          <td>0.174997</td>
          <td>26.626349</td>
          <td>0.218179</td>
          <td>25.738290</td>
          <td>0.191570</td>
          <td>25.562510</td>
          <td>0.351611</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.723288</td>
          <td>0.447311</td>
          <td>27.377164</td>
          <td>0.288743</td>
          <td>26.823210</td>
          <td>0.161446</td>
          <td>26.354789</td>
          <td>0.173595</td>
          <td>25.801909</td>
          <td>0.202100</td>
          <td>25.126083</td>
          <td>0.247421</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.141696</td>
          <td>1.148812</td>
          <td>28.007967</td>
          <td>0.471891</td>
          <td>26.411896</td>
          <td>0.113187</td>
          <td>25.982771</td>
          <td>0.126119</td>
          <td>25.472633</td>
          <td>0.152833</td>
          <td>26.256853</td>
          <td>0.591790</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.325152</td>
          <td>0.689419</td>
          <td>26.737749</td>
          <td>0.169762</td>
          <td>26.126740</td>
          <td>0.088166</td>
          <td>25.605979</td>
          <td>0.090747</td>
          <td>25.281618</td>
          <td>0.129642</td>
          <td>24.600346</td>
          <td>0.159104</td>
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
          <td>27.091594</td>
          <td>0.643553</td>
          <td>26.716817</td>
          <td>0.191414</td>
          <td>26.065830</td>
          <td>0.098238</td>
          <td>25.301657</td>
          <td>0.082196</td>
          <td>24.809985</td>
          <td>0.100900</td>
          <td>25.056849</td>
          <td>0.273100</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.800280</td>
          <td>0.456275</td>
          <td>27.543515</td>
          <td>0.340502</td>
          <td>28.127173</td>
          <td>0.783120</td>
          <td>26.517984</td>
          <td>0.417023</td>
          <td>26.299632</td>
          <td>0.695740</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.936165</td>
          <td>0.271947</td>
          <td>25.988869</td>
          <td>0.104434</td>
          <td>24.760322</td>
          <td>0.031685</td>
          <td>23.880932</td>
          <td>0.024029</td>
          <td>23.125907</td>
          <td>0.023407</td>
          <td>22.874574</td>
          <td>0.042296</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.613752</td>
          <td>0.844647</td>
          <td>29.133222</td>
          <td>1.081418</td>
          <td>26.420611</td>
          <td>0.230251</td>
          <td>25.859542</td>
          <td>0.262931</td>
          <td>25.308811</td>
          <td>0.355651</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.145116</td>
          <td>0.316830</td>
          <td>25.689279</td>
          <td>0.078714</td>
          <td>25.430096</td>
          <td>0.056055</td>
          <td>24.787930</td>
          <td>0.052177</td>
          <td>24.318659</td>
          <td>0.065463</td>
          <td>23.755404</td>
          <td>0.090136</td>
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
          <td>27.445363</td>
          <td>0.825614</td>
          <td>26.243362</td>
          <td>0.130128</td>
          <td>26.138084</td>
          <td>0.106863</td>
          <td>26.013297</td>
          <td>0.156031</td>
          <td>25.945178</td>
          <td>0.270143</td>
          <td>25.297011</td>
          <td>0.337731</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.278074</td>
          <td>0.352891</td>
          <td>27.052936</td>
          <td>0.254042</td>
          <td>26.997804</td>
          <td>0.219350</td>
          <td>26.640855</td>
          <td>0.259723</td>
          <td>27.745535</td>
          <td>0.975576</td>
          <td>25.084629</td>
          <td>0.280441</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.590771</td>
          <td>0.901378</td>
          <td>27.418051</td>
          <td>0.343262</td>
          <td>26.709887</td>
          <td>0.173584</td>
          <td>26.222734</td>
          <td>0.184935</td>
          <td>26.060907</td>
          <td>0.294464</td>
          <td>25.612526</td>
          <td>0.428243</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.383170</td>
          <td>0.339034</td>
          <td>26.655585</td>
          <td>0.168767</td>
          <td>26.157549</td>
          <td>0.178285</td>
          <td>25.562254</td>
          <td>0.198689</td>
          <td>24.924143</td>
          <td>0.252496</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.627876</td>
          <td>0.179115</td>
          <td>26.198221</td>
          <td>0.111400</td>
          <td>25.575090</td>
          <td>0.105591</td>
          <td>25.252986</td>
          <td>0.149655</td>
          <td>25.279328</td>
          <td>0.329671</td>
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
          <td>26.413827</td>
          <td>0.352518</td>
          <td>26.737423</td>
          <td>0.169733</td>
          <td>26.084796</td>
          <td>0.084980</td>
          <td>25.405166</td>
          <td>0.076034</td>
          <td>25.041214</td>
          <td>0.105184</td>
          <td>24.971150</td>
          <td>0.217653</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.438822</td>
          <td>0.644252</td>
          <td>27.889863</td>
          <td>0.386946</td>
          <td>26.766284</td>
          <td>0.245229</td>
          <td>27.491044</td>
          <td>0.735879</td>
          <td>26.126030</td>
          <td>0.539181</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>28.094418</td>
          <td>1.161426</td>
          <td>25.831213</td>
          <td>0.083025</td>
          <td>24.815959</td>
          <td>0.030027</td>
          <td>23.862919</td>
          <td>0.021312</td>
          <td>23.085941</td>
          <td>0.020464</td>
          <td>22.812889</td>
          <td>0.036009</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.830935</td>
          <td>1.070994</td>
          <td>28.930020</td>
          <td>1.024628</td>
          <td>28.036071</td>
          <td>0.523404</td>
          <td>26.864599</td>
          <td>0.329090</td>
          <td>25.865439</td>
          <td>0.263325</td>
          <td>24.927988</td>
          <td>0.261196</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.663092</td>
          <td>0.427747</td>
          <td>25.848944</td>
          <td>0.078546</td>
          <td>25.407975</td>
          <td>0.046718</td>
          <td>24.760762</td>
          <td>0.042993</td>
          <td>24.404696</td>
          <td>0.060078</td>
          <td>23.787997</td>
          <td>0.078535</td>
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
          <td>27.038753</td>
          <td>0.590405</td>
          <td>26.564840</td>
          <td>0.156634</td>
          <td>26.033208</td>
          <td>0.087869</td>
          <td>26.009316</td>
          <td>0.139943</td>
          <td>26.275895</td>
          <td>0.320653</td>
          <td>26.519964</td>
          <td>0.755668</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.654759</td>
          <td>0.428923</td>
          <td>26.931084</td>
          <td>0.202647</td>
          <td>26.821679</td>
          <td>0.163837</td>
          <td>26.459452</td>
          <td>0.192853</td>
          <td>25.661664</td>
          <td>0.182426</td>
          <td>25.304208</td>
          <td>0.290663</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.385440</td>
          <td>0.302276</td>
          <td>27.083616</td>
          <td>0.210898</td>
          <td>26.259823</td>
          <td>0.168265</td>
          <td>26.333972</td>
          <td>0.326892</td>
          <td>25.149169</td>
          <td>0.264365</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.865069</td>
          <td>0.532474</td>
          <td>27.195779</td>
          <td>0.273571</td>
          <td>26.791119</td>
          <td>0.175557</td>
          <td>25.795357</td>
          <td>0.120655</td>
          <td>25.502998</td>
          <td>0.175296</td>
          <td>25.871644</td>
          <td>0.493595</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.147928</td>
          <td>0.292422</td>
          <td>26.583275</td>
          <td>0.153780</td>
          <td>26.091783</td>
          <td>0.088897</td>
          <td>25.507853</td>
          <td>0.086715</td>
          <td>25.207218</td>
          <td>0.126300</td>
          <td>25.079835</td>
          <td>0.247360</td>
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
