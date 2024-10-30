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

    <pzflow.flow.Flow at 0x7f9078056e90>



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
          <td>27.179176</td>
          <td>0.623290</td>
          <td>26.754296</td>
          <td>0.172166</td>
          <td>26.112076</td>
          <td>0.087036</td>
          <td>25.417584</td>
          <td>0.076862</td>
          <td>24.965084</td>
          <td>0.098389</td>
          <td>25.099749</td>
          <td>0.242112</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>30.593381</td>
          <td>3.196001</td>
          <td>30.154403</td>
          <td>1.731845</td>
          <td>28.209003</td>
          <td>0.492389</td>
          <td>26.939810</td>
          <td>0.282322</td>
          <td>26.962124</td>
          <td>0.506983</td>
          <td>25.749328</td>
          <td>0.406554</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.698198</td>
          <td>0.438915</td>
          <td>26.146714</td>
          <td>0.101898</td>
          <td>24.740384</td>
          <td>0.025897</td>
          <td>23.897855</td>
          <td>0.020194</td>
          <td>23.132885</td>
          <td>0.019664</td>
          <td>22.804422</td>
          <td>0.032811</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.526959</td>
          <td>0.384993</td>
          <td>28.196831</td>
          <td>0.542231</td>
          <td>27.761633</td>
          <td>0.349791</td>
          <td>26.492751</td>
          <td>0.195080</td>
          <td>26.404988</td>
          <td>0.330925</td>
          <td>25.302940</td>
          <td>0.285833</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.137156</td>
          <td>0.282713</td>
          <td>25.734198</td>
          <td>0.070894</td>
          <td>25.389317</td>
          <td>0.045884</td>
          <td>24.824724</td>
          <td>0.045435</td>
          <td>24.438823</td>
          <td>0.061835</td>
          <td>23.744224</td>
          <td>0.075444</td>
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
          <td>26.540940</td>
          <td>0.389179</td>
          <td>26.239896</td>
          <td>0.110534</td>
          <td>26.165122</td>
          <td>0.091193</td>
          <td>26.271473</td>
          <td>0.161701</td>
          <td>25.810235</td>
          <td>0.203517</td>
          <td>25.400979</td>
          <td>0.309305</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.372866</td>
          <td>0.341309</td>
          <td>27.739003</td>
          <td>0.384546</td>
          <td>26.957411</td>
          <td>0.180969</td>
          <td>26.374548</td>
          <td>0.176532</td>
          <td>25.684516</td>
          <td>0.183064</td>
          <td>26.195028</td>
          <td>0.566247</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.844774</td>
          <td>0.964462</td>
          <td>27.448026</td>
          <td>0.305688</td>
          <td>26.812591</td>
          <td>0.159988</td>
          <td>26.925840</td>
          <td>0.279142</td>
          <td>25.668918</td>
          <td>0.180663</td>
          <td>25.544633</td>
          <td>0.346698</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.439488</td>
          <td>0.303601</td>
          <td>26.565466</td>
          <td>0.129342</td>
          <td>25.934033</td>
          <td>0.120895</td>
          <td>25.974180</td>
          <td>0.233310</td>
          <td>25.405657</td>
          <td>0.310465</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.305312</td>
          <td>0.680145</td>
          <td>26.618946</td>
          <td>0.153392</td>
          <td>26.148498</td>
          <td>0.089870</td>
          <td>25.520405</td>
          <td>0.084162</td>
          <td>25.306733</td>
          <td>0.132489</td>
          <td>25.779068</td>
          <td>0.415925</td>
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
          <td>28.547371</td>
          <td>1.529243</td>
          <td>26.658101</td>
          <td>0.182158</td>
          <td>25.977367</td>
          <td>0.090899</td>
          <td>25.328633</td>
          <td>0.084174</td>
          <td>24.983292</td>
          <td>0.117376</td>
          <td>24.618525</td>
          <td>0.189879</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.306137</td>
          <td>0.744759</td>
          <td>27.396227</td>
          <td>0.333964</td>
          <td>27.785134</td>
          <td>0.410960</td>
          <td>27.356960</td>
          <td>0.454711</td>
          <td>26.441557</td>
          <td>0.393242</td>
          <td>27.346806</td>
          <td>1.314370</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.350191</td>
          <td>0.377960</td>
          <td>25.947874</td>
          <td>0.100760</td>
          <td>24.762764</td>
          <td>0.031753</td>
          <td>23.879216</td>
          <td>0.023993</td>
          <td>23.091644</td>
          <td>0.022727</td>
          <td>22.821360</td>
          <td>0.040350</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.104482</td>
          <td>0.676227</td>
          <td>27.902694</td>
          <td>0.518167</td>
          <td>27.187261</td>
          <td>0.271974</td>
          <td>26.768754</td>
          <td>0.305881</td>
          <td>26.202124</td>
          <td>0.346186</td>
          <td>25.114130</td>
          <td>0.304733</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.151284</td>
          <td>0.318391</td>
          <td>25.692448</td>
          <td>0.078934</td>
          <td>25.565588</td>
          <td>0.063211</td>
          <td>24.881768</td>
          <td>0.056708</td>
          <td>24.383978</td>
          <td>0.069359</td>
          <td>23.465431</td>
          <td>0.069795</td>
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
          <td>26.704735</td>
          <td>0.494076</td>
          <td>26.552518</td>
          <td>0.169637</td>
          <td>25.979159</td>
          <td>0.092976</td>
          <td>25.764159</td>
          <td>0.125889</td>
          <td>25.843357</td>
          <td>0.248545</td>
          <td>25.823162</td>
          <td>0.504996</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>30.439670</td>
          <td>3.180140</td>
          <td>26.722906</td>
          <td>0.193093</td>
          <td>26.984207</td>
          <td>0.216879</td>
          <td>26.199657</td>
          <td>0.179809</td>
          <td>26.296674</td>
          <td>0.352503</td>
          <td>26.226075</td>
          <td>0.663685</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.563517</td>
          <td>2.380403</td>
          <td>27.256015</td>
          <td>0.301723</td>
          <td>26.833876</td>
          <td>0.192782</td>
          <td>25.969905</td>
          <td>0.149093</td>
          <td>25.694524</td>
          <td>0.218043</td>
          <td>25.225872</td>
          <td>0.316747</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.826178</td>
          <td>0.543580</td>
          <td>27.534955</td>
          <td>0.381823</td>
          <td>26.595042</td>
          <td>0.160275</td>
          <td>26.069322</td>
          <td>0.165402</td>
          <td>25.475210</td>
          <td>0.184635</td>
          <td>25.071714</td>
          <td>0.284770</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.728504</td>
          <td>0.499292</td>
          <td>26.684678</td>
          <td>0.187924</td>
          <td>26.164133</td>
          <td>0.108136</td>
          <td>25.478955</td>
          <td>0.097068</td>
          <td>25.102563</td>
          <td>0.131465</td>
          <td>24.767884</td>
          <td>0.217342</td>
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
          <td>26.988954</td>
          <td>0.544383</td>
          <td>27.037108</td>
          <td>0.218449</td>
          <td>26.152546</td>
          <td>0.090202</td>
          <td>25.390592</td>
          <td>0.075061</td>
          <td>24.793225</td>
          <td>0.084606</td>
          <td>25.170321</td>
          <td>0.256606</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.222525</td>
          <td>0.642753</td>
          <td>27.893554</td>
          <td>0.433254</td>
          <td>27.775516</td>
          <td>0.353929</td>
          <td>26.993692</td>
          <td>0.295156</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.825677</td>
          <td>0.431335</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.783303</td>
          <td>0.491678</td>
          <td>25.900731</td>
          <td>0.088259</td>
          <td>24.791375</td>
          <td>0.029386</td>
          <td>23.855939</td>
          <td>0.021186</td>
          <td>23.130005</td>
          <td>0.021246</td>
          <td>22.829798</td>
          <td>0.036551</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.437116</td>
          <td>0.750910</td>
          <td>28.086566</td>
          <td>0.542988</td>
          <td>26.637516</td>
          <td>0.274193</td>
          <td>25.668354</td>
          <td>0.223848</td>
          <td>24.930284</td>
          <td>0.261687</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.477781</td>
          <td>0.163532</td>
          <td>25.899767</td>
          <td>0.082143</td>
          <td>25.446284</td>
          <td>0.048334</td>
          <td>24.816062</td>
          <td>0.045155</td>
          <td>24.266604</td>
          <td>0.053149</td>
          <td>23.640342</td>
          <td>0.068924</td>
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
          <td>26.394445</td>
          <td>0.364902</td>
          <td>26.206577</td>
          <td>0.114981</td>
          <td>26.127964</td>
          <td>0.095499</td>
          <td>26.106069</td>
          <td>0.152081</td>
          <td>25.654535</td>
          <td>0.192497</td>
          <td>25.661180</td>
          <td>0.408000</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.964198</td>
          <td>0.208345</td>
          <td>26.646693</td>
          <td>0.141009</td>
          <td>26.269694</td>
          <td>0.164189</td>
          <td>25.831473</td>
          <td>0.210442</td>
          <td>25.436055</td>
          <td>0.323071</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.759470</td>
          <td>0.473319</td>
          <td>27.694652</td>
          <td>0.385810</td>
          <td>27.092953</td>
          <td>0.212550</td>
          <td>26.365534</td>
          <td>0.184060</td>
          <td>25.858408</td>
          <td>0.221945</td>
          <td>25.066094</td>
          <td>0.246960</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.390083</td>
          <td>1.383669</td>
          <td>26.882581</td>
          <td>0.211311</td>
          <td>27.115155</td>
          <td>0.230418</td>
          <td>25.775104</td>
          <td>0.118549</td>
          <td>25.397712</td>
          <td>0.160262</td>
          <td>25.375608</td>
          <td>0.337505</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.040100</td>
          <td>0.577452</td>
          <td>26.327412</td>
          <td>0.123348</td>
          <td>26.144550</td>
          <td>0.093117</td>
          <td>25.567348</td>
          <td>0.091374</td>
          <td>25.306667</td>
          <td>0.137643</td>
          <td>25.213455</td>
          <td>0.275918</td>
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
