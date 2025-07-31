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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fa2d4c3cfd0>



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
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.158542  0.097810  
    1      25.391064  0.133933  0.076818  
    2      24.304707  0.104589  0.097386  
    3      25.291103  0.089897  0.067126  
    4      25.096743  0.015312  0.008070  
    ...          ...       ...       ...  
    99995  24.737946  0.007479  0.006676  
    99996  24.224169  0.013544  0.010045  
    99997  25.613836  0.040592  0.038150  
    99998  25.274899  0.089550  0.084183  
    99999  25.699642  0.140711  0.075776  
    
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
          <td>1.398944</td>
          <td>27.084800</td>
          <td>0.583110</td>
          <td>26.315728</td>
          <td>0.118075</td>
          <td>26.028668</td>
          <td>0.080867</td>
          <td>25.165621</td>
          <td>0.061493</td>
          <td>24.807872</td>
          <td>0.085693</td>
          <td>23.879925</td>
          <td>0.085041</td>
          <td>0.158542</td>
          <td>0.097810</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.707992</td>
          <td>0.375397</td>
          <td>26.818772</td>
          <td>0.160836</td>
          <td>26.379185</td>
          <td>0.177228</td>
          <td>25.660131</td>
          <td>0.179323</td>
          <td>25.798783</td>
          <td>0.422235</td>
          <td>0.133933</td>
          <td>0.076818</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.956314</td>
          <td>0.406951</td>
          <td>26.048828</td>
          <td>0.133542</td>
          <td>25.092365</td>
          <td>0.109976</td>
          <td>24.446507</td>
          <td>0.139417</td>
          <td>0.104589</td>
          <td>0.097386</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.011619</td>
          <td>2.654065</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.989636</td>
          <td>0.185970</td>
          <td>25.990758</td>
          <td>0.126996</td>
          <td>25.200372</td>
          <td>0.120821</td>
          <td>25.448215</td>
          <td>0.321198</td>
          <td>0.089897</td>
          <td>0.067126</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.962819</td>
          <td>0.245241</td>
          <td>26.254189</td>
          <td>0.111920</td>
          <td>25.919021</td>
          <td>0.073402</td>
          <td>25.690247</td>
          <td>0.097717</td>
          <td>25.360331</td>
          <td>0.138766</td>
          <td>25.222642</td>
          <td>0.267785</td>
          <td>0.015312</td>
          <td>0.008070</td>
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
          <td>0.389450</td>
          <td>26.791224</td>
          <td>0.470698</td>
          <td>26.340950</td>
          <td>0.120691</td>
          <td>25.477273</td>
          <td>0.049611</td>
          <td>25.107782</td>
          <td>0.058417</td>
          <td>24.672646</td>
          <td>0.076057</td>
          <td>24.533631</td>
          <td>0.150267</td>
          <td>0.007479</td>
          <td>0.006676</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.817704</td>
          <td>0.181676</td>
          <td>26.058581</td>
          <td>0.083029</td>
          <td>25.092270</td>
          <td>0.057618</td>
          <td>24.875030</td>
          <td>0.090910</td>
          <td>24.192381</td>
          <td>0.111841</td>
          <td>0.013544</td>
          <td>0.010045</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.070395</td>
          <td>0.577153</td>
          <td>26.975478</td>
          <td>0.207472</td>
          <td>26.319945</td>
          <td>0.104455</td>
          <td>26.119518</td>
          <td>0.141941</td>
          <td>25.562105</td>
          <td>0.164984</td>
          <td>25.115891</td>
          <td>0.245354</td>
          <td>0.040592</td>
          <td>0.038150</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.015503</td>
          <td>0.256069</td>
          <td>26.023135</td>
          <td>0.091441</td>
          <td>26.193590</td>
          <td>0.093503</td>
          <td>25.856424</td>
          <td>0.112999</td>
          <td>25.602340</td>
          <td>0.170736</td>
          <td>25.172528</td>
          <td>0.257038</td>
          <td>0.089550</td>
          <td>0.084183</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.569625</td>
          <td>0.397886</td>
          <td>26.673710</td>
          <td>0.160746</td>
          <td>26.491480</td>
          <td>0.121303</td>
          <td>26.283365</td>
          <td>0.163351</td>
          <td>25.754027</td>
          <td>0.194127</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.140711</td>
          <td>0.075776</td>
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
          <td>1.398944</td>
          <td>28.016822</td>
          <td>1.186273</td>
          <td>26.320093</td>
          <td>0.143335</td>
          <td>25.945658</td>
          <td>0.093420</td>
          <td>25.237347</td>
          <td>0.082243</td>
          <td>24.718546</td>
          <td>0.098401</td>
          <td>24.075820</td>
          <td>0.126100</td>
          <td>0.158542</td>
          <td>0.097810</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.218646</td>
          <td>2.099281</td>
          <td>26.783782</td>
          <td>0.209353</td>
          <td>26.770144</td>
          <td>0.187276</td>
          <td>26.451225</td>
          <td>0.229696</td>
          <td>25.822883</td>
          <td>0.248450</td>
          <td>25.972309</td>
          <td>0.571209</td>
          <td>0.133933</td>
          <td>0.076818</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.345208</td>
          <td>1.401223</td>
          <td>27.666738</td>
          <td>0.423266</td>
          <td>27.155232</td>
          <td>0.256816</td>
          <td>25.975875</td>
          <td>0.153035</td>
          <td>25.197687</td>
          <td>0.146033</td>
          <td>24.101488</td>
          <td>0.126152</td>
          <td>0.104589</td>
          <td>0.097386</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.497712</td>
          <td>0.853753</td>
          <td>28.175805</td>
          <td>0.609213</td>
          <td>27.288430</td>
          <td>0.282917</td>
          <td>26.049658</td>
          <td>0.160945</td>
          <td>25.276276</td>
          <td>0.154302</td>
          <td>26.120294</td>
          <td>0.625114</td>
          <td>0.089897</td>
          <td>0.067126</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.742744</td>
          <td>0.501490</td>
          <td>26.348331</td>
          <td>0.139889</td>
          <td>26.048589</td>
          <td>0.096810</td>
          <td>25.914559</td>
          <td>0.140400</td>
          <td>25.399075</td>
          <td>0.167994</td>
          <td>25.135312</td>
          <td>0.291159</td>
          <td>0.015312</td>
          <td>0.008070</td>
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
          <td>0.389450</td>
          <td>27.514655</td>
          <td>0.853156</td>
          <td>26.168288</td>
          <td>0.119687</td>
          <td>25.485989</td>
          <td>0.058893</td>
          <td>25.106100</td>
          <td>0.069166</td>
          <td>24.701014</td>
          <td>0.091717</td>
          <td>24.732852</td>
          <td>0.209047</td>
          <td>0.007479</td>
          <td>0.006676</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.235162</td>
          <td>0.710237</td>
          <td>26.865450</td>
          <td>0.216883</td>
          <td>25.958154</td>
          <td>0.089417</td>
          <td>25.152978</td>
          <td>0.072118</td>
          <td>24.938801</td>
          <td>0.112967</td>
          <td>24.339429</td>
          <td>0.149801</td>
          <td>0.013544</td>
          <td>0.010045</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.575007</td>
          <td>0.888852</td>
          <td>26.848491</td>
          <td>0.214714</td>
          <td>26.328124</td>
          <td>0.124132</td>
          <td>26.273158</td>
          <td>0.191533</td>
          <td>26.009422</td>
          <td>0.280506</td>
          <td>26.237192</td>
          <td>0.669350</td>
          <td>0.040592</td>
          <td>0.038150</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.657110</td>
          <td>0.478185</td>
          <td>26.077890</td>
          <td>0.113139</td>
          <td>26.208385</td>
          <td>0.114078</td>
          <td>25.973571</td>
          <td>0.151426</td>
          <td>25.577549</td>
          <td>0.200037</td>
          <td>25.528768</td>
          <td>0.406108</td>
          <td>0.089550</td>
          <td>0.084183</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.787664</td>
          <td>0.531925</td>
          <td>26.695852</td>
          <td>0.194904</td>
          <td>26.479979</td>
          <td>0.146610</td>
          <td>26.283324</td>
          <td>0.200163</td>
          <td>25.538733</td>
          <td>0.196617</td>
          <td>26.386991</td>
          <td>0.761750</td>
          <td>0.140711</td>
          <td>0.075776</td>
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
          <td>1.398944</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.600465</td>
          <td>0.177357</td>
          <td>26.057268</td>
          <td>0.099952</td>
          <td>25.191312</td>
          <td>0.076510</td>
          <td>24.770592</td>
          <td>0.099925</td>
          <td>23.920125</td>
          <td>0.106769</td>
          <td>0.158542</td>
          <td>0.097810</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.387436</td>
          <td>0.324784</td>
          <td>26.774664</td>
          <td>0.176658</td>
          <td>26.519272</td>
          <td>0.228151</td>
          <td>25.780700</td>
          <td>0.225819</td>
          <td>25.873611</td>
          <td>0.503293</td>
          <td>0.133933</td>
          <td>0.076818</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.113525</td>
          <td>1.072559</td>
          <td>28.182209</td>
          <td>0.534448</td>
          <td>26.146474</td>
          <td>0.164453</td>
          <td>25.278378</td>
          <td>0.145693</td>
          <td>24.210343</td>
          <td>0.128692</td>
          <td>0.104589</td>
          <td>0.097386</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.335053</td>
          <td>0.632382</td>
          <td>27.650897</td>
          <td>0.343856</td>
          <td>26.196492</td>
          <td>0.164232</td>
          <td>25.377589</td>
          <td>0.152043</td>
          <td>25.235383</td>
          <td>0.291594</td>
          <td>0.089897</td>
          <td>0.067126</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.249404</td>
          <td>0.309823</td>
          <td>26.077896</td>
          <td>0.096109</td>
          <td>25.847069</td>
          <td>0.069015</td>
          <td>25.716137</td>
          <td>0.100173</td>
          <td>25.353328</td>
          <td>0.138205</td>
          <td>24.928631</td>
          <td>0.210458</td>
          <td>0.015312</td>
          <td>0.008070</td>
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
          <td>0.389450</td>
          <td>27.396922</td>
          <td>0.723982</td>
          <td>26.210978</td>
          <td>0.107845</td>
          <td>25.410319</td>
          <td>0.046780</td>
          <td>25.158164</td>
          <td>0.061132</td>
          <td>24.804369</td>
          <td>0.085488</td>
          <td>24.856570</td>
          <td>0.197854</td>
          <td>0.007479</td>
          <td>0.006676</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.466357</td>
          <td>0.758797</td>
          <td>26.695096</td>
          <td>0.163972</td>
          <td>26.277787</td>
          <td>0.100864</td>
          <td>25.212175</td>
          <td>0.064216</td>
          <td>24.934430</td>
          <td>0.095963</td>
          <td>24.294077</td>
          <td>0.122432</td>
          <td>0.013544</td>
          <td>0.010045</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.317313</td>
          <td>0.693425</td>
          <td>26.647326</td>
          <td>0.159931</td>
          <td>26.136979</td>
          <td>0.090823</td>
          <td>25.966057</td>
          <td>0.126991</td>
          <td>25.955989</td>
          <td>0.234329</td>
          <td>25.602725</td>
          <td>0.369857</td>
          <td>0.040592</td>
          <td>0.038150</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.496020</td>
          <td>0.398428</td>
          <td>26.314044</td>
          <td>0.127804</td>
          <td>26.137146</td>
          <td>0.097653</td>
          <td>25.952271</td>
          <td>0.135182</td>
          <td>25.318941</td>
          <td>0.146670</td>
          <td>26.243379</td>
          <td>0.633174</td>
          <td>0.089550</td>
          <td>0.084183</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.702993</td>
          <td>0.481461</td>
          <td>26.468735</td>
          <td>0.152609</td>
          <td>26.539947</td>
          <td>0.145593</td>
          <td>26.261756</td>
          <td>0.185207</td>
          <td>25.927040</td>
          <td>0.256548</td>
          <td>27.603982</td>
          <td>1.484439</td>
          <td>0.140711</td>
          <td>0.075776</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
