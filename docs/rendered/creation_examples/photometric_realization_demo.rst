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

    <pzflow.flow.Flow at 0x7f9c5033ed70>



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
          <td>26.517093</td>
          <td>0.382062</td>
          <td>26.506764</td>
          <td>0.139301</td>
          <td>25.971200</td>
          <td>0.076867</td>
          <td>25.301078</td>
          <td>0.069336</td>
          <td>25.094280</td>
          <td>0.110160</td>
          <td>25.294227</td>
          <td>0.283824</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.311404</td>
          <td>1.120830</td>
          <td>27.353558</td>
          <td>0.251882</td>
          <td>28.261248</td>
          <td>0.753881</td>
          <td>27.827956</td>
          <td>0.914109</td>
          <td>25.897855</td>
          <td>0.455136</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.468242</td>
          <td>0.367822</td>
          <td>26.073246</td>
          <td>0.095550</td>
          <td>24.796891</td>
          <td>0.027204</td>
          <td>23.868356</td>
          <td>0.019696</td>
          <td>23.125313</td>
          <td>0.019539</td>
          <td>22.870301</td>
          <td>0.034774</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.102931</td>
          <td>1.123669</td>
          <td>27.825097</td>
          <td>0.410924</td>
          <td>27.052479</td>
          <td>0.196091</td>
          <td>26.491993</td>
          <td>0.194955</td>
          <td>26.686757</td>
          <td>0.412262</td>
          <td>24.879753</td>
          <td>0.201608</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.024661</td>
          <td>0.257995</td>
          <td>25.743544</td>
          <td>0.071481</td>
          <td>25.550661</td>
          <td>0.052951</td>
          <td>24.755798</td>
          <td>0.042739</td>
          <td>24.317645</td>
          <td>0.055532</td>
          <td>23.674580</td>
          <td>0.070939</td>
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
          <td>26.220452</td>
          <td>0.302334</td>
          <td>26.441269</td>
          <td>0.131647</td>
          <td>26.229264</td>
          <td>0.096478</td>
          <td>25.931512</td>
          <td>0.120631</td>
          <td>26.279451</td>
          <td>0.299346</td>
          <td>25.355087</td>
          <td>0.298117</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.868376</td>
          <td>0.189621</td>
          <td>26.548444</td>
          <td>0.127449</td>
          <td>26.303370</td>
          <td>0.166162</td>
          <td>25.604981</td>
          <td>0.171120</td>
          <td>25.542817</td>
          <td>0.346202</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.635614</td>
          <td>0.846325</td>
          <td>27.444106</td>
          <td>0.304728</td>
          <td>26.634082</td>
          <td>0.137246</td>
          <td>26.572526</td>
          <td>0.208589</td>
          <td>26.491725</td>
          <td>0.354375</td>
          <td>25.644406</td>
          <td>0.374876</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.934605</td>
          <td>1.726248</td>
          <td>27.666593</td>
          <td>0.363469</td>
          <td>26.650446</td>
          <td>0.139197</td>
          <td>25.862089</td>
          <td>0.113559</td>
          <td>25.441458</td>
          <td>0.148800</td>
          <td>25.401319</td>
          <td>0.309389</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.633244</td>
          <td>0.845044</td>
          <td>26.716161</td>
          <td>0.166671</td>
          <td>26.095204</td>
          <td>0.085752</td>
          <td>25.611171</td>
          <td>0.091162</td>
          <td>25.305283</td>
          <td>0.132323</td>
          <td>25.154943</td>
          <td>0.253358</td>
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
          <td>27.460632</td>
          <td>0.824043</td>
          <td>26.688047</td>
          <td>0.186826</td>
          <td>25.778144</td>
          <td>0.076265</td>
          <td>25.219886</td>
          <td>0.076475</td>
          <td>25.238454</td>
          <td>0.146357</td>
          <td>24.889093</td>
          <td>0.238009</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.473677</td>
          <td>1.474181</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.100113</td>
          <td>0.237885</td>
          <td>27.175934</td>
          <td>0.396135</td>
          <td>26.284589</td>
          <td>0.347932</td>
          <td>26.918622</td>
          <td>1.032361</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.428285</td>
          <td>0.401463</td>
          <td>25.909340</td>
          <td>0.097420</td>
          <td>24.779758</td>
          <td>0.032231</td>
          <td>23.845591</td>
          <td>0.023308</td>
          <td>23.149898</td>
          <td>0.023895</td>
          <td>22.845501</td>
          <td>0.041221</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.428065</td>
          <td>0.748246</td>
          <td>27.196160</td>
          <td>0.273950</td>
          <td>26.645449</td>
          <td>0.276907</td>
          <td>26.071949</td>
          <td>0.312195</td>
          <td>25.759288</td>
          <td>0.501266</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.114008</td>
          <td>0.653751</td>
          <td>25.688469</td>
          <td>0.078658</td>
          <td>25.430310</td>
          <td>0.056066</td>
          <td>24.810502</td>
          <td>0.053233</td>
          <td>24.402259</td>
          <td>0.070490</td>
          <td>23.761379</td>
          <td>0.090611</td>
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
          <td>27.063658</td>
          <td>0.639227</td>
          <td>26.144142</td>
          <td>0.119411</td>
          <td>26.135105</td>
          <td>0.106585</td>
          <td>26.349608</td>
          <td>0.207447</td>
          <td>26.006225</td>
          <td>0.283873</td>
          <td>25.829351</td>
          <td>0.507301</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.382084</td>
          <td>0.382705</td>
          <td>26.660623</td>
          <td>0.183210</td>
          <td>26.706057</td>
          <td>0.171585</td>
          <td>26.674860</td>
          <td>0.267039</td>
          <td>26.968129</td>
          <td>0.583554</td>
          <td>25.691326</td>
          <td>0.451114</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.422798</td>
          <td>0.344549</td>
          <td>26.845235</td>
          <td>0.194635</td>
          <td>26.459371</td>
          <td>0.225505</td>
          <td>25.994651</td>
          <td>0.279103</td>
          <td>24.976792</td>
          <td>0.258970</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.052059</td>
          <td>0.259757</td>
          <td>26.516627</td>
          <td>0.149867</td>
          <td>25.934650</td>
          <td>0.147397</td>
          <td>25.270923</td>
          <td>0.155175</td>
          <td>25.160249</td>
          <td>0.305824</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.505016</td>
          <td>1.503908</td>
          <td>26.584181</td>
          <td>0.172599</td>
          <td>26.175548</td>
          <td>0.109219</td>
          <td>25.759037</td>
          <td>0.123936</td>
          <td>25.169684</td>
          <td>0.139308</td>
          <td>24.930422</td>
          <td>0.248648</td>
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
          <td>27.095119</td>
          <td>0.587449</td>
          <td>26.730131</td>
          <td>0.168684</td>
          <td>25.988969</td>
          <td>0.078093</td>
          <td>25.347053</td>
          <td>0.072226</td>
          <td>24.953526</td>
          <td>0.097409</td>
          <td>25.398178</td>
          <td>0.308650</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.063777</td>
          <td>0.968588</td>
          <td>27.257161</td>
          <td>0.232842</td>
          <td>26.731663</td>
          <td>0.238325</td>
          <td>26.190088</td>
          <td>0.278737</td>
          <td>25.362429</td>
          <td>0.300153</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.349971</td>
          <td>0.353260</td>
          <td>25.970603</td>
          <td>0.093841</td>
          <td>24.798168</td>
          <td>0.029562</td>
          <td>23.860263</td>
          <td>0.021264</td>
          <td>23.193011</td>
          <td>0.022422</td>
          <td>22.824649</td>
          <td>0.036385</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.727244</td>
          <td>0.453651</td>
          <td>27.782331</td>
          <td>0.433253</td>
          <td>26.647299</td>
          <td>0.276381</td>
          <td>26.216328</td>
          <td>0.348963</td>
          <td>25.331293</td>
          <td>0.360796</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.473803</td>
          <td>0.369743</td>
          <td>25.686931</td>
          <td>0.068078</td>
          <td>25.424500</td>
          <td>0.047408</td>
          <td>24.791009</td>
          <td>0.044162</td>
          <td>24.348682</td>
          <td>0.057165</td>
          <td>23.569746</td>
          <td>0.064746</td>
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
          <td>25.933947</td>
          <td>0.252340</td>
          <td>26.163174</td>
          <td>0.110717</td>
          <td>26.190013</td>
          <td>0.100837</td>
          <td>26.435822</td>
          <td>0.201206</td>
          <td>25.992573</td>
          <td>0.254992</td>
          <td>25.449744</td>
          <td>0.346123</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.883645</td>
          <td>0.508937</td>
          <td>26.737230</td>
          <td>0.172055</td>
          <td>26.812510</td>
          <td>0.162560</td>
          <td>26.403719</td>
          <td>0.183990</td>
          <td>26.143803</td>
          <td>0.272323</td>
          <td>25.684185</td>
          <td>0.392498</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.308221</td>
          <td>1.287380</td>
          <td>27.166196</td>
          <td>0.252990</td>
          <td>26.851179</td>
          <td>0.173370</td>
          <td>26.340419</td>
          <td>0.180188</td>
          <td>25.830990</td>
          <td>0.216934</td>
          <td>25.728250</td>
          <td>0.418120</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.425072</td>
          <td>0.382514</td>
          <td>27.012956</td>
          <td>0.235490</td>
          <td>26.762146</td>
          <td>0.171288</td>
          <td>25.914498</td>
          <td>0.133776</td>
          <td>25.532506</td>
          <td>0.179738</td>
          <td>25.348101</td>
          <td>0.330229</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.759428</td>
          <td>0.470429</td>
          <td>26.549621</td>
          <td>0.149409</td>
          <td>26.120889</td>
          <td>0.091201</td>
          <td>25.617925</td>
          <td>0.095525</td>
          <td>25.197553</td>
          <td>0.125246</td>
          <td>25.142144</td>
          <td>0.260333</td>
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
