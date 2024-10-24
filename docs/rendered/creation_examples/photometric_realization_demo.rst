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

    <pzflow.flow.Flow at 0x7ff9e229f310>



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
          <td>28.205097</td>
          <td>1.190608</td>
          <td>26.349479</td>
          <td>0.121587</td>
          <td>25.977285</td>
          <td>0.077281</td>
          <td>25.416277</td>
          <td>0.076773</td>
          <td>25.060351</td>
          <td>0.106944</td>
          <td>25.282704</td>
          <td>0.281187</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.571816</td>
          <td>0.398558</td>
          <td>27.284070</td>
          <td>0.267733</td>
          <td>27.432454</td>
          <td>0.268678</td>
          <td>26.725818</td>
          <td>0.236955</td>
          <td>26.490447</td>
          <td>0.354020</td>
          <td>26.281079</td>
          <td>0.602032</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.339552</td>
          <td>0.332440</td>
          <td>26.090062</td>
          <td>0.096969</td>
          <td>24.769932</td>
          <td>0.026572</td>
          <td>23.895469</td>
          <td>0.020153</td>
          <td>23.165512</td>
          <td>0.020215</td>
          <td>22.788761</td>
          <td>0.032362</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.575424</td>
          <td>0.814174</td>
          <td>28.490918</td>
          <td>0.667438</td>
          <td>27.473721</td>
          <td>0.277851</td>
          <td>26.998535</td>
          <td>0.296040</td>
          <td>26.214880</td>
          <td>0.284147</td>
          <td>25.701510</td>
          <td>0.391851</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.129006</td>
          <td>0.280854</td>
          <td>25.821735</td>
          <td>0.076589</td>
          <td>25.506502</td>
          <td>0.050915</td>
          <td>24.844759</td>
          <td>0.046250</td>
          <td>24.402145</td>
          <td>0.059856</td>
          <td>23.679632</td>
          <td>0.071256</td>
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
          <td>26.541371</td>
          <td>0.389309</td>
          <td>26.281058</td>
          <td>0.114569</td>
          <td>26.200726</td>
          <td>0.094091</td>
          <td>26.086565</td>
          <td>0.137965</td>
          <td>25.855889</td>
          <td>0.211446</td>
          <td>25.453616</td>
          <td>0.322583</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.093031</td>
          <td>0.272779</td>
          <td>26.785644</td>
          <td>0.176809</td>
          <td>26.659570</td>
          <td>0.140296</td>
          <td>26.439965</td>
          <td>0.186586</td>
          <td>26.070029</td>
          <td>0.252494</td>
          <td>25.404787</td>
          <td>0.310249</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.209363</td>
          <td>0.636567</td>
          <td>27.112027</td>
          <td>0.232444</td>
          <td>26.823725</td>
          <td>0.161517</td>
          <td>26.549323</td>
          <td>0.204575</td>
          <td>25.984593</td>
          <td>0.235329</td>
          <td>25.046896</td>
          <td>0.231763</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.380514</td>
          <td>0.289525</td>
          <td>26.544688</td>
          <td>0.127035</td>
          <td>25.870947</td>
          <td>0.114439</td>
          <td>25.830864</td>
          <td>0.207066</td>
          <td>25.674229</td>
          <td>0.383663</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.258222</td>
          <td>0.311615</td>
          <td>26.409237</td>
          <td>0.128050</td>
          <td>26.164732</td>
          <td>0.091162</td>
          <td>25.646871</td>
          <td>0.094067</td>
          <td>25.067752</td>
          <td>0.107638</td>
          <td>24.652984</td>
          <td>0.166415</td>
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
          <td>26.997897</td>
          <td>0.602688</td>
          <td>27.208412</td>
          <td>0.287316</td>
          <td>26.063400</td>
          <td>0.098029</td>
          <td>25.278957</td>
          <td>0.080568</td>
          <td>25.031621</td>
          <td>0.122410</td>
          <td>25.738365</td>
          <td>0.465626</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.042102</td>
          <td>0.545384</td>
          <td>27.468833</td>
          <td>0.320913</td>
          <td>26.794889</td>
          <td>0.293230</td>
          <td>26.608862</td>
          <td>0.446824</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.857676</td>
          <td>0.255088</td>
          <td>25.884876</td>
          <td>0.095355</td>
          <td>24.915170</td>
          <td>0.036316</td>
          <td>23.862551</td>
          <td>0.023651</td>
          <td>23.143791</td>
          <td>0.023770</td>
          <td>22.836095</td>
          <td>0.040880</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>30.987581</td>
          <td>3.758287</td>
          <td>28.547552</td>
          <td>0.809370</td>
          <td>27.154556</td>
          <td>0.264820</td>
          <td>26.468654</td>
          <td>0.239585</td>
          <td>25.495823</td>
          <td>0.194425</td>
          <td>25.363887</td>
          <td>0.371307</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.524668</td>
          <td>0.425891</td>
          <td>25.722495</td>
          <td>0.081051</td>
          <td>25.434163</td>
          <td>0.056258</td>
          <td>24.791337</td>
          <td>0.052335</td>
          <td>24.289210</td>
          <td>0.063777</td>
          <td>23.646027</td>
          <td>0.081863</td>
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
          <td>26.621297</td>
          <td>0.464351</td>
          <td>26.616921</td>
          <td>0.179165</td>
          <td>26.068822</td>
          <td>0.100582</td>
          <td>25.933550</td>
          <td>0.145714</td>
          <td>25.994462</td>
          <td>0.281181</td>
          <td>26.325282</td>
          <td>0.719646</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.302386</td>
          <td>0.359678</td>
          <td>27.038437</td>
          <td>0.251038</td>
          <td>27.117302</td>
          <td>0.242184</td>
          <td>26.433045</td>
          <td>0.218767</td>
          <td>25.742890</td>
          <td>0.225153</td>
          <td>26.066735</td>
          <td>0.593722</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.874961</td>
          <td>0.556490</td>
          <td>27.351954</td>
          <td>0.325759</td>
          <td>26.616323</td>
          <td>0.160285</td>
          <td>26.545753</td>
          <td>0.242216</td>
          <td>26.618823</td>
          <td>0.455064</td>
          <td>25.104213</td>
          <td>0.287255</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.713155</td>
          <td>1.678173</td>
          <td>27.716916</td>
          <td>0.438969</td>
          <td>26.342784</td>
          <td>0.129011</td>
          <td>25.839652</td>
          <td>0.135818</td>
          <td>25.338319</td>
          <td>0.164374</td>
          <td>25.182282</td>
          <td>0.311269</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.521968</td>
          <td>0.427745</td>
          <td>26.378761</td>
          <td>0.144818</td>
          <td>26.050974</td>
          <td>0.097944</td>
          <td>25.695938</td>
          <td>0.117325</td>
          <td>25.028145</td>
          <td>0.123257</td>
          <td>25.207211</td>
          <td>0.311259</td>
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
          <td>26.561135</td>
          <td>0.145988</td>
          <td>26.022043</td>
          <td>0.080406</td>
          <td>25.394492</td>
          <td>0.075320</td>
          <td>24.938163</td>
          <td>0.096105</td>
          <td>25.051792</td>
          <td>0.232735</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.213017</td>
          <td>1.196384</td>
          <td>28.765067</td>
          <td>0.802401</td>
          <td>27.085484</td>
          <td>0.201789</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.496739</td>
          <td>0.356073</td>
          <td>27.571367</td>
          <td>1.339510</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.285475</td>
          <td>0.335762</td>
          <td>26.104738</td>
          <td>0.105527</td>
          <td>24.792806</td>
          <td>0.029423</td>
          <td>23.844348</td>
          <td>0.020977</td>
          <td>23.179298</td>
          <td>0.022160</td>
          <td>22.872466</td>
          <td>0.037956</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.310236</td>
          <td>0.774764</td>
          <td>27.977318</td>
          <td>0.545611</td>
          <td>26.747031</td>
          <td>0.188118</td>
          <td>26.535602</td>
          <td>0.252287</td>
          <td>27.739245</td>
          <td>1.012260</td>
          <td>25.122109</td>
          <td>0.305668</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.356087</td>
          <td>0.704592</td>
          <td>25.625828</td>
          <td>0.064497</td>
          <td>25.464518</td>
          <td>0.049123</td>
          <td>24.813554</td>
          <td>0.045055</td>
          <td>24.516827</td>
          <td>0.066357</td>
          <td>23.636177</td>
          <td>0.068670</td>
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
          <td>27.800412</td>
          <td>0.975767</td>
          <td>26.284755</td>
          <td>0.123059</td>
          <td>26.148320</td>
          <td>0.097219</td>
          <td>25.920287</td>
          <td>0.129584</td>
          <td>25.875032</td>
          <td>0.231446</td>
          <td>25.652445</td>
          <td>0.405272</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>31.484709</td>
          <td>4.067536</td>
          <td>27.131013</td>
          <td>0.239322</td>
          <td>26.759315</td>
          <td>0.155332</td>
          <td>26.519469</td>
          <td>0.202834</td>
          <td>25.919456</td>
          <td>0.226449</td>
          <td>25.427294</td>
          <td>0.320825</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.803776</td>
          <td>0.489161</td>
          <td>27.382423</td>
          <td>0.301545</td>
          <td>26.900567</td>
          <td>0.180788</td>
          <td>26.343021</td>
          <td>0.180586</td>
          <td>25.606355</td>
          <td>0.179601</td>
          <td>25.553354</td>
          <td>0.365252</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.061994</td>
          <td>0.613011</td>
          <td>26.978327</td>
          <td>0.228836</td>
          <td>26.818598</td>
          <td>0.179696</td>
          <td>25.885956</td>
          <td>0.130515</td>
          <td>25.476524</td>
          <td>0.171398</td>
          <td>25.258719</td>
          <td>0.307510</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.540571</td>
          <td>0.148253</td>
          <td>26.203449</td>
          <td>0.098057</td>
          <td>25.643124</td>
          <td>0.097660</td>
          <td>25.300402</td>
          <td>0.136901</td>
          <td>24.948469</td>
          <td>0.221884</td>
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
