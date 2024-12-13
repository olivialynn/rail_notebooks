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

    <pzflow.flow.Flow at 0x7f3384135f90>



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
          <td>26.292503</td>
          <td>0.320251</td>
          <td>26.831086</td>
          <td>0.183744</td>
          <td>26.104449</td>
          <td>0.086453</td>
          <td>25.331404</td>
          <td>0.071223</td>
          <td>24.900335</td>
          <td>0.092954</td>
          <td>25.082210</td>
          <td>0.238633</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.798669</td>
          <td>0.937572</td>
          <td>27.427592</td>
          <td>0.300715</td>
          <td>28.300628</td>
          <td>0.526685</td>
          <td>27.535585</td>
          <td>0.450234</td>
          <td>25.814695</td>
          <td>0.204279</td>
          <td>29.186655</td>
          <td>2.676368</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.100622</td>
          <td>0.274465</td>
          <td>26.010716</td>
          <td>0.090450</td>
          <td>24.770271</td>
          <td>0.026580</td>
          <td>23.848355</td>
          <td>0.019366</td>
          <td>23.113458</td>
          <td>0.019344</td>
          <td>22.829366</td>
          <td>0.033540</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.845977</td>
          <td>1.656304</td>
          <td>28.069598</td>
          <td>0.494003</td>
          <td>27.203662</td>
          <td>0.222531</td>
          <td>26.970724</td>
          <td>0.289472</td>
          <td>25.873378</td>
          <td>0.214557</td>
          <td>25.252968</td>
          <td>0.274480</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.151128</td>
          <td>0.285924</td>
          <td>25.766535</td>
          <td>0.072948</td>
          <td>25.416911</td>
          <td>0.047022</td>
          <td>24.764706</td>
          <td>0.043079</td>
          <td>24.328406</td>
          <td>0.056065</td>
          <td>23.784561</td>
          <td>0.078181</td>
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
          <td>26.421636</td>
          <td>0.354656</td>
          <td>26.175249</td>
          <td>0.104472</td>
          <td>26.177935</td>
          <td>0.092226</td>
          <td>25.890940</td>
          <td>0.116448</td>
          <td>26.050315</td>
          <td>0.248437</td>
          <td>25.864977</td>
          <td>0.443995</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.374516</td>
          <td>0.712885</td>
          <td>27.059119</td>
          <td>0.222462</td>
          <td>27.311728</td>
          <td>0.243362</td>
          <td>26.343883</td>
          <td>0.171993</td>
          <td>26.320552</td>
          <td>0.309385</td>
          <td>25.561655</td>
          <td>0.351374</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.469827</td>
          <td>0.311072</td>
          <td>26.674946</td>
          <td>0.142167</td>
          <td>26.749019</td>
          <td>0.241539</td>
          <td>25.582054</td>
          <td>0.167813</td>
          <td>25.500254</td>
          <td>0.334754</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.419198</td>
          <td>0.734608</td>
          <td>27.528276</td>
          <td>0.325913</td>
          <td>26.460131</td>
          <td>0.118042</td>
          <td>25.987701</td>
          <td>0.126660</td>
          <td>25.837350</td>
          <td>0.208193</td>
          <td>25.009883</td>
          <td>0.224755</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.666107</td>
          <td>0.428364</td>
          <td>26.557047</td>
          <td>0.145460</td>
          <td>26.014401</td>
          <td>0.079855</td>
          <td>25.676333</td>
          <td>0.096531</td>
          <td>25.213384</td>
          <td>0.122194</td>
          <td>24.803610</td>
          <td>0.189094</td>
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
          <td>26.361999</td>
          <td>0.375733</td>
          <td>26.569221</td>
          <td>0.168931</td>
          <td>25.923369</td>
          <td>0.086683</td>
          <td>25.434671</td>
          <td>0.092403</td>
          <td>25.242700</td>
          <td>0.146892</td>
          <td>24.700978</td>
          <td>0.203513</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.491552</td>
          <td>0.745480</td>
          <td>27.458549</td>
          <td>0.318292</td>
          <td>27.362706</td>
          <td>0.456680</td>
          <td>26.237668</td>
          <td>0.335277</td>
          <td>25.913719</td>
          <td>0.530106</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.048628</td>
          <td>0.110022</td>
          <td>24.838950</td>
          <td>0.033955</td>
          <td>23.876483</td>
          <td>0.023937</td>
          <td>23.099154</td>
          <td>0.022874</td>
          <td>22.875258</td>
          <td>0.042322</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.853549</td>
          <td>0.499789</td>
          <td>26.958294</td>
          <td>0.225294</td>
          <td>26.400030</td>
          <td>0.226354</td>
          <td>26.402810</td>
          <td>0.404713</td>
          <td>27.502221</td>
          <td>1.483354</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.478796</td>
          <td>0.411242</td>
          <td>25.874252</td>
          <td>0.092612</td>
          <td>25.421118</td>
          <td>0.055611</td>
          <td>24.724747</td>
          <td>0.049333</td>
          <td>24.337375</td>
          <td>0.066556</td>
          <td>23.779735</td>
          <td>0.092084</td>
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
          <td>26.848297</td>
          <td>0.548715</td>
          <td>26.510274</td>
          <td>0.163644</td>
          <td>26.185587</td>
          <td>0.111386</td>
          <td>26.203795</td>
          <td>0.183491</td>
          <td>25.873621</td>
          <td>0.254799</td>
          <td>26.159023</td>
          <td>0.642270</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.502496</td>
          <td>0.419822</td>
          <td>27.383218</td>
          <td>0.331607</td>
          <td>26.838314</td>
          <td>0.191913</td>
          <td>26.714778</td>
          <td>0.275860</td>
          <td>26.114983</td>
          <td>0.305139</td>
          <td>25.472603</td>
          <td>0.381624</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.575956</td>
          <td>0.446398</td>
          <td>27.808806</td>
          <td>0.463674</td>
          <td>26.534479</td>
          <td>0.149436</td>
          <td>26.797085</td>
          <td>0.297276</td>
          <td>25.781446</td>
          <td>0.234361</td>
          <td>25.348680</td>
          <td>0.349131</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.737816</td>
          <td>0.509672</td>
          <td>27.914463</td>
          <td>0.508676</td>
          <td>26.484066</td>
          <td>0.145734</td>
          <td>25.906356</td>
          <td>0.143854</td>
          <td>25.725641</td>
          <td>0.227734</td>
          <td>25.823027</td>
          <td>0.509482</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.604551</td>
          <td>0.455288</td>
          <td>26.393455</td>
          <td>0.146657</td>
          <td>26.097703</td>
          <td>0.102035</td>
          <td>25.674310</td>
          <td>0.115138</td>
          <td>25.104833</td>
          <td>0.131723</td>
          <td>24.719479</td>
          <td>0.208733</td>
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
          <td>26.915840</td>
          <td>0.516167</td>
          <td>26.360091</td>
          <td>0.122726</td>
          <td>26.201889</td>
          <td>0.094200</td>
          <td>25.381843</td>
          <td>0.074483</td>
          <td>24.937249</td>
          <td>0.096028</td>
          <td>24.768463</td>
          <td>0.183585</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.099543</td>
          <td>0.505386</td>
          <td>27.624710</td>
          <td>0.314063</td>
          <td>27.609196</td>
          <td>0.476175</td>
          <td>27.819019</td>
          <td>0.909643</td>
          <td>26.328174</td>
          <td>0.622810</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.154842</td>
          <td>0.302589</td>
          <td>25.877182</td>
          <td>0.086451</td>
          <td>24.759159</td>
          <td>0.028569</td>
          <td>23.888009</td>
          <td>0.021774</td>
          <td>23.138549</td>
          <td>0.021401</td>
          <td>22.857350</td>
          <td>0.037452</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.230191</td>
          <td>0.734700</td>
          <td>27.369192</td>
          <td>0.344240</td>
          <td>27.154033</td>
          <td>0.263830</td>
          <td>26.191726</td>
          <td>0.189469</td>
          <td>26.282748</td>
          <td>0.367614</td>
          <td>26.204167</td>
          <td>0.685607</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.052181</td>
          <td>0.264099</td>
          <td>25.812389</td>
          <td>0.076054</td>
          <td>25.351898</td>
          <td>0.044449</td>
          <td>24.684649</td>
          <td>0.040187</td>
          <td>24.335757</td>
          <td>0.056513</td>
          <td>23.747089</td>
          <td>0.075748</td>
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
          <td>26.603259</td>
          <td>0.428597</td>
          <td>26.497647</td>
          <td>0.147872</td>
          <td>26.190013</td>
          <td>0.100837</td>
          <td>26.141132</td>
          <td>0.156718</td>
          <td>25.921385</td>
          <td>0.240489</td>
          <td>25.314181</td>
          <td>0.310792</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.629064</td>
          <td>0.420616</td>
          <td>27.180218</td>
          <td>0.249219</td>
          <td>26.944991</td>
          <td>0.181941</td>
          <td>26.403095</td>
          <td>0.183893</td>
          <td>25.868078</td>
          <td>0.216973</td>
          <td>25.710421</td>
          <td>0.400520</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.004274</td>
          <td>0.221314</td>
          <td>26.799764</td>
          <td>0.165947</td>
          <td>26.208817</td>
          <td>0.161102</td>
          <td>25.984446</td>
          <td>0.246344</td>
          <td>24.952999</td>
          <td>0.224913</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.509491</td>
          <td>2.287601</td>
          <td>27.979465</td>
          <td>0.503050</td>
          <td>26.584727</td>
          <td>0.147183</td>
          <td>25.922634</td>
          <td>0.134720</td>
          <td>25.633393</td>
          <td>0.195720</td>
          <td>25.358719</td>
          <td>0.333021</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.007012</td>
          <td>0.563939</td>
          <td>26.299341</td>
          <td>0.120380</td>
          <td>26.167820</td>
          <td>0.095039</td>
          <td>25.681767</td>
          <td>0.101024</td>
          <td>25.096660</td>
          <td>0.114731</td>
          <td>24.957898</td>
          <td>0.223630</td>
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
