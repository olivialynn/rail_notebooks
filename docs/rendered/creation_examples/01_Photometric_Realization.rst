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

    <pzflow.flow.Flow at 0x7f0134e87010>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>26.905156</td>
          <td>0.512103</td>
          <td>26.594762</td>
          <td>0.150246</td>
          <td>26.088337</td>
          <td>0.085235</td>
          <td>25.163223</td>
          <td>0.061362</td>
          <td>24.651892</td>
          <td>0.074675</td>
          <td>23.862937</td>
          <td>0.083777</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.391065</td>
          <td>0.292000</td>
          <td>26.809128</td>
          <td>0.159515</td>
          <td>26.261517</td>
          <td>0.160331</td>
          <td>25.609452</td>
          <td>0.171772</td>
          <td>25.306474</td>
          <td>0.286651</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.801657</td>
          <td>0.360956</td>
          <td>26.029461</td>
          <td>0.131324</td>
          <td>25.140781</td>
          <td>0.114717</td>
          <td>24.224325</td>
          <td>0.114998</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>32.356316</td>
          <td>3.723252</td>
          <td>27.709360</td>
          <td>0.335652</td>
          <td>26.323640</td>
          <td>0.169056</td>
          <td>25.686447</td>
          <td>0.183364</td>
          <td>25.663345</td>
          <td>0.380437</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.398875</td>
          <td>0.348373</td>
          <td>26.129932</td>
          <td>0.100413</td>
          <td>26.050292</td>
          <td>0.082424</td>
          <td>25.790626</td>
          <td>0.106693</td>
          <td>25.431251</td>
          <td>0.147501</td>
          <td>24.846172</td>
          <td>0.195998</td>
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
          <td>0.389450</td>
          <td>26.570554</td>
          <td>0.398171</td>
          <td>26.420934</td>
          <td>0.129353</td>
          <td>25.373613</td>
          <td>0.045249</td>
          <td>25.069235</td>
          <td>0.056452</td>
          <td>24.914317</td>
          <td>0.094103</td>
          <td>24.647297</td>
          <td>0.165611</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.961360</td>
          <td>0.205034</td>
          <td>26.235758</td>
          <td>0.097029</td>
          <td>25.138662</td>
          <td>0.060040</td>
          <td>24.846314</td>
          <td>0.088643</td>
          <td>24.101505</td>
          <td>0.103306</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.728029</td>
          <td>0.448912</td>
          <td>26.415559</td>
          <td>0.128753</td>
          <td>26.305471</td>
          <td>0.103140</td>
          <td>26.265006</td>
          <td>0.160810</td>
          <td>25.578870</td>
          <td>0.167358</td>
          <td>25.423351</td>
          <td>0.314889</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.008343</td>
          <td>0.254573</td>
          <td>26.484100</td>
          <td>0.136606</td>
          <td>26.030747</td>
          <td>0.081015</td>
          <td>26.006204</td>
          <td>0.128707</td>
          <td>25.875821</td>
          <td>0.214995</td>
          <td>25.163240</td>
          <td>0.255088</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.696462</td>
          <td>0.438339</td>
          <td>26.909730</td>
          <td>0.196340</td>
          <td>26.486631</td>
          <td>0.120793</td>
          <td>26.208998</td>
          <td>0.153284</td>
          <td>26.021856</td>
          <td>0.242683</td>
          <td>25.823148</td>
          <td>0.430141</td>
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
          <td>1.398944</td>
          <td>26.965139</td>
          <td>0.588869</td>
          <td>26.838101</td>
          <td>0.211910</td>
          <td>26.040666</td>
          <td>0.096094</td>
          <td>25.204641</td>
          <td>0.075452</td>
          <td>24.662017</td>
          <td>0.088614</td>
          <td>24.029238</td>
          <td>0.114504</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.234478</td>
          <td>0.340070</td>
          <td>27.207514</td>
          <td>0.287156</td>
          <td>26.608076</td>
          <td>0.157222</td>
          <td>26.223902</td>
          <td>0.182813</td>
          <td>26.003684</td>
          <td>0.277913</td>
          <td>25.186583</td>
          <td>0.303349</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.353738</td>
          <td>0.778417</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.412162</td>
          <td>0.312936</td>
          <td>25.842269</td>
          <td>0.134870</td>
          <td>25.348223</td>
          <td>0.164309</td>
          <td>24.350173</td>
          <td>0.154535</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.655332</td>
          <td>3.438178</td>
          <td>29.366871</td>
          <td>1.314163</td>
          <td>28.110822</td>
          <td>0.554216</td>
          <td>26.350245</td>
          <td>0.217171</td>
          <td>25.846469</td>
          <td>0.260136</td>
          <td>24.990710</td>
          <td>0.275831</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.646696</td>
          <td>0.466943</td>
          <td>26.242798</td>
          <td>0.127693</td>
          <td>25.895619</td>
          <td>0.084618</td>
          <td>25.570289</td>
          <td>0.104101</td>
          <td>25.197988</td>
          <td>0.141394</td>
          <td>25.044686</td>
          <td>0.270493</td>
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
          <td>0.389450</td>
          <td>26.735914</td>
          <td>0.505563</td>
          <td>26.290264</td>
          <td>0.135506</td>
          <td>25.479932</td>
          <td>0.059827</td>
          <td>25.134122</td>
          <td>0.072463</td>
          <td>24.758871</td>
          <td>0.098525</td>
          <td>24.645622</td>
          <td>0.198317</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.933229</td>
          <td>0.577123</td>
          <td>27.195699</td>
          <td>0.285364</td>
          <td>25.957323</td>
          <td>0.089686</td>
          <td>25.350708</td>
          <td>0.086199</td>
          <td>24.845715</td>
          <td>0.104538</td>
          <td>24.152961</td>
          <td>0.128034</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.374825</td>
          <td>0.784966</td>
          <td>26.899636</td>
          <td>0.225501</td>
          <td>26.197842</td>
          <td>0.111668</td>
          <td>26.791179</td>
          <td>0.295866</td>
          <td>26.268796</td>
          <td>0.347522</td>
          <td>25.776671</td>
          <td>0.484484</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.293588</td>
          <td>0.752030</td>
          <td>26.197309</td>
          <td>0.126215</td>
          <td>26.217437</td>
          <td>0.115710</td>
          <td>25.993715</td>
          <td>0.155055</td>
          <td>25.486172</td>
          <td>0.186353</td>
          <td>25.741839</td>
          <td>0.479796</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.907163</td>
          <td>0.568530</td>
          <td>27.237440</td>
          <td>0.296576</td>
          <td>26.486892</td>
          <td>0.143062</td>
          <td>26.290051</td>
          <td>0.195206</td>
          <td>26.182721</td>
          <td>0.323828</td>
          <td>26.757565</td>
          <td>0.943069</td>
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
          <td>1.398944</td>
          <td>29.020327</td>
          <td>1.795101</td>
          <td>26.478151</td>
          <td>0.135922</td>
          <td>25.963908</td>
          <td>0.076383</td>
          <td>25.133800</td>
          <td>0.059789</td>
          <td>24.777710</td>
          <td>0.083457</td>
          <td>23.993903</td>
          <td>0.094020</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.443756</td>
          <td>0.304871</td>
          <td>26.614476</td>
          <td>0.135068</td>
          <td>26.547960</td>
          <td>0.204535</td>
          <td>25.642899</td>
          <td>0.176883</td>
          <td>25.158299</td>
          <td>0.254290</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.716932</td>
          <td>0.819613</td>
          <td>27.641864</td>
          <td>0.342512</td>
          <td>25.938045</td>
          <td>0.132002</td>
          <td>24.999371</td>
          <td>0.109988</td>
          <td>24.204244</td>
          <td>0.122898</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.353879</td>
          <td>0.797230</td>
          <td>28.009543</td>
          <td>0.558446</td>
          <td>27.310395</td>
          <td>0.299478</td>
          <td>25.877890</td>
          <td>0.145012</td>
          <td>25.681705</td>
          <td>0.226345</td>
          <td>25.518108</td>
          <td>0.416911</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.244818</td>
          <td>0.308569</td>
          <td>26.214820</td>
          <td>0.108276</td>
          <td>25.829130</td>
          <td>0.067886</td>
          <td>25.689332</td>
          <td>0.097784</td>
          <td>25.578234</td>
          <td>0.167498</td>
          <td>24.689769</td>
          <td>0.171954</td>
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
          <td>0.389450</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.359004</td>
          <td>0.131228</td>
          <td>25.451333</td>
          <td>0.052507</td>
          <td>25.077309</td>
          <td>0.061800</td>
          <td>24.892322</td>
          <td>0.099849</td>
          <td>25.242573</td>
          <td>0.293418</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.040030</td>
          <td>0.264057</td>
          <td>26.422014</td>
          <td>0.131313</td>
          <td>26.124814</td>
          <td>0.089482</td>
          <td>25.156754</td>
          <td>0.062087</td>
          <td>24.760834</td>
          <td>0.083584</td>
          <td>24.170513</td>
          <td>0.111609</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.692173</td>
          <td>0.450044</td>
          <td>26.629905</td>
          <td>0.161412</td>
          <td>26.427298</td>
          <td>0.120412</td>
          <td>25.946952</td>
          <td>0.128604</td>
          <td>25.894258</td>
          <td>0.228654</td>
          <td>25.806904</td>
          <td>0.443871</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.198850</td>
          <td>0.320233</td>
          <td>25.995603</td>
          <td>0.098733</td>
          <td>25.957765</td>
          <td>0.085241</td>
          <td>26.036224</td>
          <td>0.148566</td>
          <td>25.736059</td>
          <td>0.213309</td>
          <td>26.055969</td>
          <td>0.564579</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.389004</td>
          <td>0.354214</td>
          <td>26.598728</td>
          <td>0.155828</td>
          <td>26.810350</td>
          <td>0.165834</td>
          <td>26.324402</td>
          <td>0.175974</td>
          <td>26.210500</td>
          <td>0.293442</td>
          <td>25.875203</td>
          <td>0.463159</td>
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
