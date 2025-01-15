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

    <pzflow.flow.Flow at 0x7f40638468c0>



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
          <td>28.943152</td>
          <td>1.733056</td>
          <td>26.742077</td>
          <td>0.170388</td>
          <td>26.072132</td>
          <td>0.084026</td>
          <td>25.284889</td>
          <td>0.068349</td>
          <td>24.929863</td>
          <td>0.095396</td>
          <td>24.785359</td>
          <td>0.186202</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.343242</td>
          <td>0.697953</td>
          <td>28.133178</td>
          <td>0.517664</td>
          <td>27.335491</td>
          <td>0.248170</td>
          <td>27.326930</td>
          <td>0.383836</td>
          <td>26.198244</td>
          <td>0.280342</td>
          <td>27.340881</td>
          <td>1.180608</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.319453</td>
          <td>0.327185</td>
          <td>25.883292</td>
          <td>0.080860</td>
          <td>24.802590</td>
          <td>0.027340</td>
          <td>23.874881</td>
          <td>0.019805</td>
          <td>23.110158</td>
          <td>0.019291</td>
          <td>22.813167</td>
          <td>0.033065</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.345466</td>
          <td>2.765206</td>
          <td>27.547287</td>
          <td>0.294887</td>
          <td>26.600029</td>
          <td>0.213440</td>
          <td>26.519421</td>
          <td>0.362154</td>
          <td>25.259480</td>
          <td>0.275936</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.229947</td>
          <td>0.304644</td>
          <td>25.803204</td>
          <td>0.075347</td>
          <td>25.485729</td>
          <td>0.049985</td>
          <td>24.752970</td>
          <td>0.042632</td>
          <td>24.356613</td>
          <td>0.057486</td>
          <td>23.738231</td>
          <td>0.075046</td>
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
          <td>27.232555</td>
          <td>0.646908</td>
          <td>26.261458</td>
          <td>0.112630</td>
          <td>26.135436</td>
          <td>0.088843</td>
          <td>25.919840</td>
          <td>0.119413</td>
          <td>25.955145</td>
          <td>0.229660</td>
          <td>25.544591</td>
          <td>0.346686</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.956671</td>
          <td>1.031687</td>
          <td>27.275870</td>
          <td>0.265949</td>
          <td>26.650061</td>
          <td>0.139151</td>
          <td>26.333825</td>
          <td>0.170528</td>
          <td>25.928965</td>
          <td>0.224723</td>
          <td>25.500770</td>
          <td>0.334890</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>30.993619</td>
          <td>3.578448</td>
          <td>27.084047</td>
          <td>0.227116</td>
          <td>26.825668</td>
          <td>0.161786</td>
          <td>26.113547</td>
          <td>0.141212</td>
          <td>25.779362</td>
          <td>0.198309</td>
          <td>25.681319</td>
          <td>0.385777</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.447221</td>
          <td>0.361834</td>
          <td>27.193285</td>
          <td>0.248558</td>
          <td>26.527307</td>
          <td>0.125135</td>
          <td>26.010516</td>
          <td>0.129188</td>
          <td>25.657826</td>
          <td>0.178973</td>
          <td>25.170531</td>
          <td>0.256617</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.916967</td>
          <td>0.516555</td>
          <td>26.378882</td>
          <td>0.124727</td>
          <td>26.101038</td>
          <td>0.086194</td>
          <td>25.722122</td>
          <td>0.100485</td>
          <td>25.119002</td>
          <td>0.112561</td>
          <td>25.008523</td>
          <td>0.224501</td>
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
          <td>26.363373</td>
          <td>0.376135</td>
          <td>26.868209</td>
          <td>0.217298</td>
          <td>25.937689</td>
          <td>0.087782</td>
          <td>25.187469</td>
          <td>0.074316</td>
          <td>24.918761</td>
          <td>0.110961</td>
          <td>24.837165</td>
          <td>0.227995</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.843614</td>
          <td>0.934508</td>
          <td>27.872426</td>
          <td>0.439216</td>
          <td>27.479459</td>
          <td>0.498178</td>
          <td>26.481943</td>
          <td>0.405665</td>
          <td>26.488473</td>
          <td>0.789162</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.736420</td>
          <td>0.506187</td>
          <td>26.012101</td>
          <td>0.106574</td>
          <td>24.789203</td>
          <td>0.032499</td>
          <td>23.849357</td>
          <td>0.023383</td>
          <td>23.122375</td>
          <td>0.023335</td>
          <td>22.806678</td>
          <td>0.039829</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.631359</td>
          <td>0.482268</td>
          <td>27.685850</td>
          <td>0.440937</td>
          <td>27.901701</td>
          <td>0.475412</td>
          <td>26.697910</td>
          <td>0.288927</td>
          <td>26.335578</td>
          <td>0.384249</td>
          <td>25.369434</td>
          <td>0.372915</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.898332</td>
          <td>0.259612</td>
          <td>25.763219</td>
          <td>0.084008</td>
          <td>25.446646</td>
          <td>0.056884</td>
          <td>24.816870</td>
          <td>0.053535</td>
          <td>24.428704</td>
          <td>0.072158</td>
          <td>23.744603</td>
          <td>0.089284</td>
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
          <td>27.360582</td>
          <td>0.781321</td>
          <td>26.535510</td>
          <td>0.167200</td>
          <td>26.020561</td>
          <td>0.096417</td>
          <td>26.090246</td>
          <td>0.166628</td>
          <td>25.756073</td>
          <td>0.231270</td>
          <td>25.493099</td>
          <td>0.393671</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.086857</td>
          <td>0.261195</td>
          <td>26.828616</td>
          <td>0.190350</td>
          <td>26.406794</td>
          <td>0.214031</td>
          <td>25.506397</td>
          <td>0.184666</td>
          <td>25.234183</td>
          <td>0.316303</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.767644</td>
          <td>0.201986</td>
          <td>26.945153</td>
          <td>0.211649</td>
          <td>26.982972</td>
          <td>0.344741</td>
          <td>26.397719</td>
          <td>0.384357</td>
          <td>25.672641</td>
          <td>0.448193</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>30.859307</td>
          <td>3.603842</td>
          <td>27.484050</td>
          <td>0.366992</td>
          <td>26.718726</td>
          <td>0.178067</td>
          <td>25.627411</td>
          <td>0.112980</td>
          <td>25.561940</td>
          <td>0.198637</td>
          <td>25.490744</td>
          <td>0.396661</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.338901</td>
          <td>0.139938</td>
          <td>26.276699</td>
          <td>0.119277</td>
          <td>25.614026</td>
          <td>0.109244</td>
          <td>25.127283</td>
          <td>0.134303</td>
          <td>24.687403</td>
          <td>0.203199</td>
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
          <td>28.117847</td>
          <td>1.133373</td>
          <td>26.384827</td>
          <td>0.125386</td>
          <td>26.109884</td>
          <td>0.086879</td>
          <td>25.411428</td>
          <td>0.076456</td>
          <td>25.216283</td>
          <td>0.122518</td>
          <td>25.623310</td>
          <td>0.368807</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.304697</td>
          <td>0.586288</td>
          <td>27.391396</td>
          <td>0.260046</td>
          <td>26.988101</td>
          <td>0.293828</td>
          <td>26.674334</td>
          <td>0.408691</td>
          <td>27.272897</td>
          <td>1.136752</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.734989</td>
          <td>0.474353</td>
          <td>26.013805</td>
          <td>0.097461</td>
          <td>24.790653</td>
          <td>0.029368</td>
          <td>23.888555</td>
          <td>0.021784</td>
          <td>23.131786</td>
          <td>0.021278</td>
          <td>22.769179</td>
          <td>0.034646</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.731504</td>
          <td>1.579432</td>
          <td>27.388528</td>
          <td>0.318815</td>
          <td>26.524777</td>
          <td>0.250054</td>
          <td>26.022386</td>
          <td>0.299054</td>
          <td>25.450677</td>
          <td>0.395874</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.606140</td>
          <td>0.409554</td>
          <td>25.684255</td>
          <td>0.067917</td>
          <td>25.395877</td>
          <td>0.046219</td>
          <td>24.822280</td>
          <td>0.045405</td>
          <td>24.336372</td>
          <td>0.056544</td>
          <td>23.663217</td>
          <td>0.070333</td>
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
          <td>26.275890</td>
          <td>0.332422</td>
          <td>26.299404</td>
          <td>0.124631</td>
          <td>26.137232</td>
          <td>0.096278</td>
          <td>26.067414</td>
          <td>0.147118</td>
          <td>25.648108</td>
          <td>0.191457</td>
          <td>25.605430</td>
          <td>0.390852</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.603194</td>
          <td>0.835890</td>
          <td>27.049032</td>
          <td>0.223614</td>
          <td>26.618521</td>
          <td>0.137625</td>
          <td>26.502052</td>
          <td>0.199890</td>
          <td>26.466654</td>
          <td>0.352610</td>
          <td>25.170066</td>
          <td>0.260640</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.511362</td>
          <td>0.392084</td>
          <td>27.021540</td>
          <td>0.224512</td>
          <td>26.664356</td>
          <td>0.147789</td>
          <td>26.978807</td>
          <td>0.305358</td>
          <td>26.087257</td>
          <td>0.267993</td>
          <td>25.314599</td>
          <td>0.302275</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.523947</td>
          <td>0.412771</td>
          <td>28.933401</td>
          <td>0.958808</td>
          <td>26.681378</td>
          <td>0.159891</td>
          <td>25.916234</td>
          <td>0.133977</td>
          <td>25.398626</td>
          <td>0.160387</td>
          <td>25.051712</td>
          <td>0.260043</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.332995</td>
          <td>0.707843</td>
          <td>26.527474</td>
          <td>0.146595</td>
          <td>26.146932</td>
          <td>0.093312</td>
          <td>25.699291</td>
          <td>0.102586</td>
          <td>25.018263</td>
          <td>0.107148</td>
          <td>25.006735</td>
          <td>0.232877</td>
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
