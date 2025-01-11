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

    <pzflow.flow.Flow at 0x7f38d109bdc0>



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
          <td>26.345080</td>
          <td>0.333898</td>
          <td>26.620426</td>
          <td>0.153587</td>
          <td>26.003527</td>
          <td>0.079092</td>
          <td>25.368088</td>
          <td>0.073572</td>
          <td>25.145489</td>
          <td>0.115189</td>
          <td>25.091143</td>
          <td>0.240399</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.005415</td>
          <td>1.614793</td>
          <td>27.833193</td>
          <td>0.369963</td>
          <td>26.640744</td>
          <td>0.220810</td>
          <td>26.654568</td>
          <td>0.402200</td>
          <td>25.969709</td>
          <td>0.480268</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.265152</td>
          <td>0.313344</td>
          <td>26.012524</td>
          <td>0.090593</td>
          <td>24.778336</td>
          <td>0.026767</td>
          <td>23.869061</td>
          <td>0.019708</td>
          <td>23.131880</td>
          <td>0.019647</td>
          <td>22.861670</td>
          <td>0.034510</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.218253</td>
          <td>0.301801</td>
          <td>28.661053</td>
          <td>0.748857</td>
          <td>27.486646</td>
          <td>0.280780</td>
          <td>26.913110</td>
          <td>0.276271</td>
          <td>26.118122</td>
          <td>0.262639</td>
          <td>25.587435</td>
          <td>0.358559</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.909088</td>
          <td>0.234620</td>
          <td>25.753806</td>
          <td>0.072132</td>
          <td>25.401685</td>
          <td>0.046391</td>
          <td>24.809042</td>
          <td>0.044807</td>
          <td>24.441496</td>
          <td>0.061982</td>
          <td>23.669542</td>
          <td>0.070623</td>
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
          <td>25.842078</td>
          <td>0.221956</td>
          <td>26.277360</td>
          <td>0.114201</td>
          <td>26.301672</td>
          <td>0.102798</td>
          <td>26.099389</td>
          <td>0.139500</td>
          <td>25.484150</td>
          <td>0.154349</td>
          <td>25.123708</td>
          <td>0.246938</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.446385</td>
          <td>0.305286</td>
          <td>26.766187</td>
          <td>0.153759</td>
          <td>26.464126</td>
          <td>0.190431</td>
          <td>26.638498</td>
          <td>0.397254</td>
          <td>25.265402</td>
          <td>0.277267</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.652769</td>
          <td>0.855641</td>
          <td>26.817575</td>
          <td>0.181656</td>
          <td>26.955899</td>
          <td>0.180737</td>
          <td>26.684488</td>
          <td>0.228984</td>
          <td>26.302523</td>
          <td>0.304946</td>
          <td>25.167952</td>
          <td>0.256075</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.318992</td>
          <td>0.686530</td>
          <td>26.995788</td>
          <td>0.211025</td>
          <td>26.620458</td>
          <td>0.135641</td>
          <td>26.006351</td>
          <td>0.128723</td>
          <td>25.464991</td>
          <td>0.151835</td>
          <td>25.381808</td>
          <td>0.304587</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.492065</td>
          <td>1.389841</td>
          <td>26.697113</td>
          <td>0.163988</td>
          <td>26.051271</td>
          <td>0.082495</td>
          <td>25.807904</td>
          <td>0.108315</td>
          <td>25.065226</td>
          <td>0.107400</td>
          <td>25.302662</td>
          <td>0.285769</td>
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
          <td>28.076223</td>
          <td>1.193623</td>
          <td>26.848155</td>
          <td>0.213695</td>
          <td>26.069864</td>
          <td>0.098586</td>
          <td>25.285608</td>
          <td>0.081042</td>
          <td>24.992614</td>
          <td>0.118331</td>
          <td>24.852243</td>
          <td>0.230863</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.655105</td>
          <td>0.931823</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.232768</td>
          <td>0.265265</td>
          <td>27.106295</td>
          <td>0.375330</td>
          <td>26.898560</td>
          <td>0.553378</td>
          <td>26.817275</td>
          <td>0.971316</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.154567</td>
          <td>0.324122</td>
          <td>25.790449</td>
          <td>0.087775</td>
          <td>24.828389</td>
          <td>0.033640</td>
          <td>23.865946</td>
          <td>0.023720</td>
          <td>23.130212</td>
          <td>0.023493</td>
          <td>22.798852</td>
          <td>0.039554</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.289605</td>
          <td>0.765903</td>
          <td>27.741182</td>
          <td>0.459705</td>
          <td>27.584867</td>
          <td>0.373376</td>
          <td>27.418367</td>
          <td>0.504663</td>
          <td>26.173218</td>
          <td>0.338377</td>
          <td>25.791039</td>
          <td>0.513109</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.134319</td>
          <td>0.314114</td>
          <td>25.720943</td>
          <td>0.080940</td>
          <td>25.508107</td>
          <td>0.060071</td>
          <td>24.850069</td>
          <td>0.055135</td>
          <td>24.438146</td>
          <td>0.072762</td>
          <td>23.859999</td>
          <td>0.098801</td>
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
          <td>26.021738</td>
          <td>0.291166</td>
          <td>26.305937</td>
          <td>0.137350</td>
          <td>25.959290</td>
          <td>0.091368</td>
          <td>26.260871</td>
          <td>0.192549</td>
          <td>25.810697</td>
          <td>0.241951</td>
          <td>25.075099</td>
          <td>0.282753</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.211730</td>
          <td>0.700586</td>
          <td>26.574276</td>
          <td>0.170278</td>
          <td>26.608187</td>
          <td>0.157846</td>
          <td>26.550647</td>
          <td>0.241170</td>
          <td>25.895639</td>
          <td>0.255403</td>
          <td>25.028131</td>
          <td>0.267849</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.964259</td>
          <td>0.593147</td>
          <td>27.272861</td>
          <td>0.305829</td>
          <td>26.758962</td>
          <td>0.180962</td>
          <td>26.340843</td>
          <td>0.204269</td>
          <td>25.763632</td>
          <td>0.230930</td>
          <td>25.138245</td>
          <td>0.295254</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.416078</td>
          <td>0.347949</td>
          <td>26.312805</td>
          <td>0.125704</td>
          <td>26.063682</td>
          <td>0.164609</td>
          <td>25.521202</td>
          <td>0.191943</td>
          <td>25.850814</td>
          <td>0.519968</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.638266</td>
          <td>0.466937</td>
          <td>26.445901</td>
          <td>0.153402</td>
          <td>26.034725</td>
          <td>0.096558</td>
          <td>25.592013</td>
          <td>0.107164</td>
          <td>25.035687</td>
          <td>0.124066</td>
          <td>25.467863</td>
          <td>0.382250</td>
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
          <td>27.726851</td>
          <td>0.896697</td>
          <td>26.516578</td>
          <td>0.140499</td>
          <td>26.022624</td>
          <td>0.080447</td>
          <td>25.266505</td>
          <td>0.067255</td>
          <td>25.177485</td>
          <td>0.118456</td>
          <td>24.953790</td>
          <td>0.214525</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.109951</td>
          <td>0.593936</td>
          <td>27.971809</td>
          <td>0.459612</td>
          <td>27.288329</td>
          <td>0.238921</td>
          <td>27.763133</td>
          <td>0.533339</td>
          <td>26.131455</td>
          <td>0.265748</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.833345</td>
          <td>0.232876</td>
          <td>25.952219</td>
          <td>0.092340</td>
          <td>24.733969</td>
          <td>0.027947</td>
          <td>23.898582</td>
          <td>0.021972</td>
          <td>23.146493</td>
          <td>0.021547</td>
          <td>22.776284</td>
          <td>0.034864</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.077454</td>
          <td>0.586250</td>
          <td>27.151305</td>
          <td>0.263243</td>
          <td>26.829352</td>
          <td>0.319992</td>
          <td>25.680057</td>
          <td>0.226035</td>
          <td>25.163335</td>
          <td>0.315919</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.197898</td>
          <td>0.297174</td>
          <td>25.745560</td>
          <td>0.071698</td>
          <td>25.477739</td>
          <td>0.049703</td>
          <td>24.820388</td>
          <td>0.045329</td>
          <td>24.290015</td>
          <td>0.054265</td>
          <td>23.748465</td>
          <td>0.075840</td>
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
          <td>27.004490</td>
          <td>0.576179</td>
          <td>26.269019</td>
          <td>0.121391</td>
          <td>26.082330</td>
          <td>0.091747</td>
          <td>25.919474</td>
          <td>0.129493</td>
          <td>26.085973</td>
          <td>0.275196</td>
          <td>25.758833</td>
          <td>0.439527</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.192100</td>
          <td>1.190643</td>
          <td>27.317185</td>
          <td>0.278710</td>
          <td>26.638903</td>
          <td>0.140065</td>
          <td>26.293227</td>
          <td>0.167517</td>
          <td>25.850526</td>
          <td>0.213819</td>
          <td>25.141544</td>
          <td>0.254622</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.830032</td>
          <td>1.674505</td>
          <td>27.060538</td>
          <td>0.231892</td>
          <td>26.661339</td>
          <td>0.147407</td>
          <td>26.243562</td>
          <td>0.165950</td>
          <td>25.982272</td>
          <td>0.245904</td>
          <td>26.177202</td>
          <td>0.582622</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.807964</td>
          <td>0.510719</td>
          <td>27.353127</td>
          <td>0.310592</td>
          <td>26.343208</td>
          <td>0.119451</td>
          <td>25.955336</td>
          <td>0.138576</td>
          <td>25.985515</td>
          <td>0.262145</td>
          <td>25.705378</td>
          <td>0.435793</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.124808</td>
          <td>0.287020</td>
          <td>26.477438</td>
          <td>0.140422</td>
          <td>26.176880</td>
          <td>0.095798</td>
          <td>25.641303</td>
          <td>0.097504</td>
          <td>25.103968</td>
          <td>0.115464</td>
          <td>24.812974</td>
          <td>0.198113</td>
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
