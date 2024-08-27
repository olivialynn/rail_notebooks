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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fe1033b6f20>



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
          <td>28.797701</td>
          <td>1.618719</td>
          <td>26.664019</td>
          <td>0.159421</td>
          <td>26.188776</td>
          <td>0.093109</td>
          <td>25.270688</td>
          <td>0.067495</td>
          <td>25.147068</td>
          <td>0.115347</td>
          <td>25.571451</td>
          <td>0.354090</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.673125</td>
          <td>1.523486</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.436383</td>
          <td>0.269539</td>
          <td>27.857737</td>
          <td>0.570567</td>
          <td>27.071307</td>
          <td>0.548996</td>
          <td>26.566228</td>
          <td>0.732613</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.380486</td>
          <td>0.343366</td>
          <td>25.891321</td>
          <td>0.081434</td>
          <td>24.804990</td>
          <td>0.027397</td>
          <td>23.893255</td>
          <td>0.020116</td>
          <td>23.190408</td>
          <td>0.020647</td>
          <td>22.842397</td>
          <td>0.033928</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.756951</td>
          <td>0.913653</td>
          <td>29.338400</td>
          <td>1.138284</td>
          <td>27.475355</td>
          <td>0.278220</td>
          <td>26.956801</td>
          <td>0.286233</td>
          <td>26.506453</td>
          <td>0.358494</td>
          <td>25.109092</td>
          <td>0.243984</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.871404</td>
          <td>0.499548</td>
          <td>25.766359</td>
          <td>0.072936</td>
          <td>25.368138</td>
          <td>0.045030</td>
          <td>24.843480</td>
          <td>0.046198</td>
          <td>24.407360</td>
          <td>0.060134</td>
          <td>23.692773</td>
          <td>0.072090</td>
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
          <td>26.913588</td>
          <td>0.515278</td>
          <td>26.331850</td>
          <td>0.119741</td>
          <td>26.348427</td>
          <td>0.107088</td>
          <td>25.937782</td>
          <td>0.121290</td>
          <td>25.820935</td>
          <td>0.205351</td>
          <td>25.666408</td>
          <td>0.381342</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.753769</td>
          <td>0.911845</td>
          <td>27.098047</td>
          <td>0.229768</td>
          <td>26.732810</td>
          <td>0.149420</td>
          <td>26.399751</td>
          <td>0.180345</td>
          <td>25.865293</td>
          <td>0.213114</td>
          <td>25.638840</td>
          <td>0.373255</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.206730</td>
          <td>0.635401</td>
          <td>27.257504</td>
          <td>0.261991</td>
          <td>26.885657</td>
          <td>0.170274</td>
          <td>26.362097</td>
          <td>0.174676</td>
          <td>26.372452</td>
          <td>0.322477</td>
          <td>25.159250</td>
          <td>0.254255</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.090432</td>
          <td>0.585452</td>
          <td>27.314252</td>
          <td>0.274392</td>
          <td>26.498690</td>
          <td>0.122065</td>
          <td>25.688382</td>
          <td>0.097557</td>
          <td>25.561913</td>
          <td>0.164957</td>
          <td>25.108422</td>
          <td>0.243849</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.670960</td>
          <td>0.429946</td>
          <td>26.898295</td>
          <td>0.194460</td>
          <td>26.141603</td>
          <td>0.089327</td>
          <td>25.712184</td>
          <td>0.099614</td>
          <td>25.378149</td>
          <td>0.140913</td>
          <td>24.897653</td>
          <td>0.204658</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.670993</td>
          <td>0.184155</td>
          <td>26.010463</td>
          <td>0.093580</td>
          <td>25.256730</td>
          <td>0.079003</td>
          <td>24.991769</td>
          <td>0.118245</td>
          <td>24.501059</td>
          <td>0.171900</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.824902</td>
          <td>0.923751</td>
          <td>27.429971</td>
          <td>0.311108</td>
          <td>26.756281</td>
          <td>0.284225</td>
          <td>26.082150</td>
          <td>0.296116</td>
          <td>26.916051</td>
          <td>1.030785</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.031450</td>
          <td>0.293730</td>
          <td>25.931438</td>
          <td>0.099322</td>
          <td>24.792388</td>
          <td>0.032591</td>
          <td>23.871445</td>
          <td>0.023833</td>
          <td>23.211756</td>
          <td>0.025208</td>
          <td>22.766043</td>
          <td>0.038423</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.188878</td>
          <td>0.299079</td>
          <td>27.513413</td>
          <td>0.353078</td>
          <td>26.229898</td>
          <td>0.196353</td>
          <td>26.176788</td>
          <td>0.339333</td>
          <td>25.477999</td>
          <td>0.405585</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.922379</td>
          <td>0.264755</td>
          <td>25.786209</td>
          <td>0.085723</td>
          <td>25.416443</td>
          <td>0.055380</td>
          <td>24.720550</td>
          <td>0.049149</td>
          <td>24.339287</td>
          <td>0.066669</td>
          <td>23.551258</td>
          <td>0.075296</td>
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
          <td>28.254667</td>
          <td>1.328606</td>
          <td>26.294021</td>
          <td>0.135946</td>
          <td>26.025689</td>
          <td>0.096851</td>
          <td>25.648865</td>
          <td>0.113888</td>
          <td>26.077039</td>
          <td>0.300562</td>
          <td>25.181267</td>
          <td>0.308002</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.798453</td>
          <td>0.523624</td>
          <td>27.417320</td>
          <td>0.340676</td>
          <td>26.578756</td>
          <td>0.153919</td>
          <td>26.630006</td>
          <td>0.257426</td>
          <td>25.980873</td>
          <td>0.273811</td>
          <td>25.409115</td>
          <td>0.363205</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.240689</td>
          <td>0.344692</td>
          <td>26.932738</td>
          <td>0.231775</td>
          <td>27.205653</td>
          <td>0.262506</td>
          <td>26.056381</td>
          <td>0.160553</td>
          <td>25.618247</td>
          <td>0.204578</td>
          <td>27.786798</td>
          <td>1.652255</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.104142</td>
          <td>0.661554</td>
          <td>27.274847</td>
          <td>0.311058</td>
          <td>26.432439</td>
          <td>0.139400</td>
          <td>25.933406</td>
          <td>0.147239</td>
          <td>25.705934</td>
          <td>0.224038</td>
          <td>25.784887</td>
          <td>0.495361</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.532623</td>
          <td>0.867668</td>
          <td>26.453326</td>
          <td>0.154380</td>
          <td>26.062800</td>
          <td>0.098964</td>
          <td>25.748844</td>
          <td>0.122845</td>
          <td>25.112932</td>
          <td>0.132648</td>
          <td>24.663099</td>
          <td>0.199096</td>
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
          <td>26.456776</td>
          <td>0.133437</td>
          <td>26.163716</td>
          <td>0.091093</td>
          <td>25.350664</td>
          <td>0.072457</td>
          <td>25.157198</td>
          <td>0.116384</td>
          <td>24.931593</td>
          <td>0.210584</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.509826</td>
          <td>1.403230</td>
          <td>28.052116</td>
          <td>0.487986</td>
          <td>27.904493</td>
          <td>0.391350</td>
          <td>27.352136</td>
          <td>0.391741</td>
          <td>27.071994</td>
          <td>0.549696</td>
          <td>25.924738</td>
          <td>0.464802</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.476818</td>
          <td>0.797397</td>
          <td>25.682409</td>
          <td>0.072820</td>
          <td>24.705897</td>
          <td>0.027270</td>
          <td>23.836665</td>
          <td>0.020840</td>
          <td>23.128288</td>
          <td>0.021214</td>
          <td>22.881868</td>
          <td>0.038273</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.838514</td>
          <td>1.800903</td>
          <td>27.267903</td>
          <td>0.317672</td>
          <td>26.986106</td>
          <td>0.229777</td>
          <td>26.974756</td>
          <td>0.358966</td>
          <td>26.728696</td>
          <td>0.515367</td>
          <td>25.243622</td>
          <td>0.336737</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.285699</td>
          <td>0.318805</td>
          <td>25.766565</td>
          <td>0.073040</td>
          <td>25.400968</td>
          <td>0.046428</td>
          <td>24.783807</td>
          <td>0.043881</td>
          <td>24.357458</td>
          <td>0.057612</td>
          <td>23.686526</td>
          <td>0.071799</td>
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
          <td>26.420156</td>
          <td>0.372290</td>
          <td>26.608408</td>
          <td>0.162572</td>
          <td>26.223000</td>
          <td>0.103791</td>
          <td>26.276340</td>
          <td>0.175861</td>
          <td>25.820425</td>
          <td>0.221187</td>
          <td>25.748812</td>
          <td>0.436202</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.031688</td>
          <td>1.086522</td>
          <td>26.887921</td>
          <td>0.195433</td>
          <td>26.664717</td>
          <td>0.143214</td>
          <td>26.489314</td>
          <td>0.197762</td>
          <td>26.240439</td>
          <td>0.294495</td>
          <td>25.102289</td>
          <td>0.246541</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.283007</td>
          <td>0.688027</td>
          <td>27.181227</td>
          <td>0.256126</td>
          <td>26.721259</td>
          <td>0.155181</td>
          <td>26.323879</td>
          <td>0.177680</td>
          <td>25.703686</td>
          <td>0.194989</td>
          <td>26.170852</td>
          <td>0.579990</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.110917</td>
          <td>0.634367</td>
          <td>27.556036</td>
          <td>0.364662</td>
          <td>26.631079</td>
          <td>0.153155</td>
          <td>25.683560</td>
          <td>0.109462</td>
          <td>25.751799</td>
          <td>0.216129</td>
          <td>25.748712</td>
          <td>0.450304</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.602546</td>
          <td>0.845211</td>
          <td>26.416873</td>
          <td>0.133276</td>
          <td>26.136346</td>
          <td>0.092449</td>
          <td>25.672433</td>
          <td>0.100201</td>
          <td>25.117347</td>
          <td>0.116816</td>
          <td>25.305888</td>
          <td>0.297336</td>
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
