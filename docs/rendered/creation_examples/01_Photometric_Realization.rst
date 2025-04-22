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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f663352bb50>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.727249</td>
          <td>0.168252</td>
          <td>25.982555</td>
          <td>0.077641</td>
          <td>25.155725</td>
          <td>0.060955</td>
          <td>24.718734</td>
          <td>0.079216</td>
          <td>23.946887</td>
          <td>0.090203</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.649489</td>
          <td>0.358634</td>
          <td>26.665563</td>
          <td>0.141023</td>
          <td>26.416441</td>
          <td>0.182911</td>
          <td>26.175080</td>
          <td>0.275119</td>
          <td>25.724846</td>
          <td>0.398970</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.669286</td>
          <td>3.268034</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.821468</td>
          <td>0.366592</td>
          <td>26.056633</td>
          <td>0.134446</td>
          <td>25.105876</td>
          <td>0.111280</td>
          <td>24.486715</td>
          <td>0.144329</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.706300</td>
          <td>0.374903</td>
          <td>27.175153</td>
          <td>0.217311</td>
          <td>26.048559</td>
          <td>0.133511</td>
          <td>25.597657</td>
          <td>0.170057</td>
          <td>25.615456</td>
          <td>0.366508</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.160176</td>
          <td>0.288021</td>
          <td>26.107351</td>
          <td>0.098448</td>
          <td>25.834555</td>
          <td>0.068115</td>
          <td>25.526574</td>
          <td>0.084621</td>
          <td>25.347254</td>
          <td>0.137209</td>
          <td>24.968312</td>
          <td>0.217111</td>
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
          <td>26.887046</td>
          <td>0.505336</td>
          <td>26.564378</td>
          <td>0.146379</td>
          <td>25.528373</td>
          <td>0.051914</td>
          <td>25.132223</td>
          <td>0.059698</td>
          <td>24.740350</td>
          <td>0.080742</td>
          <td>24.655323</td>
          <td>0.166747</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.335450</td>
          <td>0.694268</td>
          <td>26.976010</td>
          <td>0.207564</td>
          <td>26.074670</td>
          <td>0.084215</td>
          <td>25.175137</td>
          <td>0.062014</td>
          <td>24.771182</td>
          <td>0.082967</td>
          <td>24.125444</td>
          <td>0.105492</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.842019</td>
          <td>0.185449</td>
          <td>26.227574</td>
          <td>0.096335</td>
          <td>26.296424</td>
          <td>0.165181</td>
          <td>25.524074</td>
          <td>0.159713</td>
          <td>27.260624</td>
          <td>1.128080</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.800482</td>
          <td>0.214409</td>
          <td>26.232304</td>
          <td>0.109805</td>
          <td>26.169100</td>
          <td>0.091513</td>
          <td>25.952542</td>
          <td>0.122855</td>
          <td>25.599268</td>
          <td>0.170290</td>
          <td>25.374516</td>
          <td>0.302810</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.115347</td>
          <td>0.595896</td>
          <td>26.836992</td>
          <td>0.184663</td>
          <td>26.519788</td>
          <td>0.124321</td>
          <td>26.334927</td>
          <td>0.170688</td>
          <td>25.864375</td>
          <td>0.212951</td>
          <td>25.075770</td>
          <td>0.237367</td>
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
          <td>28.412170</td>
          <td>1.428714</td>
          <td>26.458465</td>
          <td>0.153697</td>
          <td>25.863743</td>
          <td>0.082248</td>
          <td>25.196080</td>
          <td>0.074884</td>
          <td>24.729679</td>
          <td>0.094042</td>
          <td>24.030105</td>
          <td>0.114591</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.191485</td>
          <td>1.271930</td>
          <td>28.049754</td>
          <td>0.548410</td>
          <td>26.817095</td>
          <td>0.187792</td>
          <td>26.306452</td>
          <td>0.196000</td>
          <td>26.298862</td>
          <td>0.351861</td>
          <td>24.923397</td>
          <td>0.244887</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.552728</td>
          <td>0.884722</td>
          <td>30.847487</td>
          <td>2.480611</td>
          <td>27.375651</td>
          <td>0.303916</td>
          <td>26.012088</td>
          <td>0.156073</td>
          <td>25.369406</td>
          <td>0.167302</td>
          <td>24.332768</td>
          <td>0.152248</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.154440</td>
          <td>0.699651</td>
          <td>28.314582</td>
          <td>0.693235</td>
          <td>27.592258</td>
          <td>0.375530</td>
          <td>26.227163</td>
          <td>0.195901</td>
          <td>25.677043</td>
          <td>0.226232</td>
          <td>24.774198</td>
          <td>0.230929</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.017140</td>
          <td>0.285912</td>
          <td>26.157593</td>
          <td>0.118601</td>
          <td>25.945860</td>
          <td>0.088444</td>
          <td>25.729881</td>
          <td>0.119642</td>
          <td>25.779832</td>
          <td>0.231319</td>
          <td>25.063637</td>
          <td>0.274696</td>
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
          <td>27.915540</td>
          <td>1.100752</td>
          <td>26.256596</td>
          <td>0.131625</td>
          <td>25.526137</td>
          <td>0.062329</td>
          <td>25.059193</td>
          <td>0.067815</td>
          <td>24.641748</td>
          <td>0.088899</td>
          <td>24.550639</td>
          <td>0.183054</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.786856</td>
          <td>1.011697</td>
          <td>26.552045</td>
          <td>0.167089</td>
          <td>26.039076</td>
          <td>0.096361</td>
          <td>25.193431</td>
          <td>0.075034</td>
          <td>24.986624</td>
          <td>0.118204</td>
          <td>24.207667</td>
          <td>0.134238</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.491650</td>
          <td>0.418745</td>
          <td>26.507407</td>
          <td>0.162072</td>
          <td>26.395794</td>
          <td>0.132609</td>
          <td>26.540179</td>
          <td>0.241105</td>
          <td>25.679449</td>
          <td>0.215320</td>
          <td>25.333267</td>
          <td>0.344917</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.367745</td>
          <td>0.789694</td>
          <td>26.193391</td>
          <td>0.125788</td>
          <td>26.294972</td>
          <td>0.123775</td>
          <td>25.974307</td>
          <td>0.152499</td>
          <td>25.625223</td>
          <td>0.209459</td>
          <td>25.187916</td>
          <td>0.312675</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.230025</td>
          <td>0.341163</td>
          <td>26.771079</td>
          <td>0.202089</td>
          <td>26.509514</td>
          <td>0.145873</td>
          <td>26.521002</td>
          <td>0.236684</td>
          <td>25.668150</td>
          <td>0.212743</td>
          <td>26.191262</td>
          <td>0.650992</td>
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
          <td>28.868613</td>
          <td>1.674134</td>
          <td>26.605287</td>
          <td>0.151625</td>
          <td>26.032446</td>
          <td>0.081148</td>
          <td>25.130647</td>
          <td>0.059622</td>
          <td>24.686408</td>
          <td>0.076997</td>
          <td>23.912299</td>
          <td>0.087512</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.562830</td>
          <td>1.441925</td>
          <td>27.643615</td>
          <td>0.357248</td>
          <td>26.463000</td>
          <td>0.118448</td>
          <td>26.577606</td>
          <td>0.209676</td>
          <td>25.971474</td>
          <td>0.232995</td>
          <td>26.479704</td>
          <td>0.691551</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.648440</td>
          <td>0.889996</td>
          <td>28.448693</td>
          <td>0.685782</td>
          <td>29.308601</td>
          <td>1.086738</td>
          <td>26.120380</td>
          <td>0.154435</td>
          <td>25.200370</td>
          <td>0.130977</td>
          <td>24.383748</td>
          <td>0.143525</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.285670</td>
          <td>0.293575</td>
          <td>26.556034</td>
          <td>0.256550</td>
          <td>25.604548</td>
          <td>0.212259</td>
          <td>25.077800</td>
          <td>0.294972</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.229510</td>
          <td>0.304811</td>
          <td>26.138059</td>
          <td>0.101254</td>
          <td>26.000587</td>
          <td>0.079000</td>
          <td>25.750286</td>
          <td>0.103147</td>
          <td>25.310130</td>
          <td>0.133065</td>
          <td>26.078316</td>
          <td>0.520974</td>
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
          <td>26.525630</td>
          <td>0.403921</td>
          <td>26.441258</td>
          <td>0.140876</td>
          <td>25.534525</td>
          <td>0.056531</td>
          <td>25.177094</td>
          <td>0.067513</td>
          <td>25.013646</td>
          <td>0.111020</td>
          <td>24.785229</td>
          <td>0.201321</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.963094</td>
          <td>1.043668</td>
          <td>26.790883</td>
          <td>0.180066</td>
          <td>26.202465</td>
          <td>0.095800</td>
          <td>25.274996</td>
          <td>0.068946</td>
          <td>25.004022</td>
          <td>0.103487</td>
          <td>24.195683</td>
          <td>0.114084</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.780759</td>
          <td>0.949911</td>
          <td>26.523640</td>
          <td>0.147378</td>
          <td>26.315397</td>
          <td>0.109230</td>
          <td>26.374317</td>
          <td>0.185432</td>
          <td>27.134607</td>
          <td>0.597722</td>
          <td>25.440669</td>
          <td>0.334258</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.755710</td>
          <td>0.491429</td>
          <td>26.086182</td>
          <td>0.106866</td>
          <td>26.115011</td>
          <td>0.097873</td>
          <td>25.724046</td>
          <td>0.113396</td>
          <td>25.730946</td>
          <td>0.212400</td>
          <td>25.574697</td>
          <td>0.394328</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.896190</td>
          <td>0.520441</td>
          <td>26.772994</td>
          <td>0.180740</td>
          <td>26.555591</td>
          <td>0.133252</td>
          <td>25.930219</td>
          <td>0.125453</td>
          <td>25.956313</td>
          <td>0.238450</td>
          <td>25.308809</td>
          <td>0.298036</td>
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
