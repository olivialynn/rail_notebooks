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

    <pzflow.flow.Flow at 0x7f905859a5c0>



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
          <td>25.841005</td>
          <td>0.221759</td>
          <td>26.889613</td>
          <td>0.193044</td>
          <td>26.132166</td>
          <td>0.088588</td>
          <td>25.170118</td>
          <td>0.061739</td>
          <td>24.669851</td>
          <td>0.075869</td>
          <td>24.003087</td>
          <td>0.094769</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.461103</td>
          <td>0.308908</td>
          <td>26.450319</td>
          <td>0.117039</td>
          <td>26.359583</td>
          <td>0.174303</td>
          <td>25.862262</td>
          <td>0.212575</td>
          <td>25.746242</td>
          <td>0.405591</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.601768</td>
          <td>0.719731</td>
          <td>27.973420</td>
          <td>0.412324</td>
          <td>25.685540</td>
          <td>0.097314</td>
          <td>25.281825</td>
          <td>0.129665</td>
          <td>24.498720</td>
          <td>0.145827</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.677754</td>
          <td>0.757209</td>
          <td>27.671935</td>
          <td>0.325832</td>
          <td>26.191399</td>
          <td>0.150988</td>
          <td>25.397994</td>
          <td>0.143342</td>
          <td>25.522489</td>
          <td>0.340693</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.332705</td>
          <td>0.330642</td>
          <td>26.208192</td>
          <td>0.107520</td>
          <td>25.977878</td>
          <td>0.077321</td>
          <td>25.709768</td>
          <td>0.099403</td>
          <td>25.654347</td>
          <td>0.178446</td>
          <td>24.678596</td>
          <td>0.170085</td>
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
          <td>26.250589</td>
          <td>0.111569</td>
          <td>25.496921</td>
          <td>0.050484</td>
          <td>25.143092</td>
          <td>0.060276</td>
          <td>24.729202</td>
          <td>0.079952</td>
          <td>24.855642</td>
          <td>0.197566</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.429758</td>
          <td>1.345229</td>
          <td>26.580127</td>
          <td>0.148372</td>
          <td>25.969098</td>
          <td>0.076724</td>
          <td>25.379971</td>
          <td>0.074349</td>
          <td>24.885043</td>
          <td>0.091714</td>
          <td>24.244589</td>
          <td>0.117044</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.346413</td>
          <td>1.286708</td>
          <td>26.658006</td>
          <td>0.158605</td>
          <td>26.323130</td>
          <td>0.104746</td>
          <td>25.946075</td>
          <td>0.122167</td>
          <td>26.456100</td>
          <td>0.344578</td>
          <td>26.026330</td>
          <td>0.500841</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.642844</td>
          <td>0.420845</td>
          <td>26.176721</td>
          <td>0.104606</td>
          <td>26.087607</td>
          <td>0.085180</td>
          <td>25.829528</td>
          <td>0.110379</td>
          <td>25.808910</td>
          <td>0.203291</td>
          <td>25.982809</td>
          <td>0.484968</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.681613</td>
          <td>0.433436</td>
          <td>26.920701</td>
          <td>0.198158</td>
          <td>26.499952</td>
          <td>0.122199</td>
          <td>26.186040</td>
          <td>0.150295</td>
          <td>26.096558</td>
          <td>0.258046</td>
          <td>25.201419</td>
          <td>0.263186</td>
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
          <td>29.040952</td>
          <td>1.921160</td>
          <td>26.647915</td>
          <td>0.180595</td>
          <td>25.902909</td>
          <td>0.085136</td>
          <td>25.104877</td>
          <td>0.069081</td>
          <td>24.796930</td>
          <td>0.099753</td>
          <td>24.044241</td>
          <td>0.116010</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.682289</td>
          <td>0.185953</td>
          <td>26.271921</td>
          <td>0.117632</td>
          <td>26.241514</td>
          <td>0.185556</td>
          <td>25.912139</td>
          <td>0.257924</td>
          <td>26.739962</td>
          <td>0.926281</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.102390</td>
          <td>1.988617</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.999500</td>
          <td>0.154400</td>
          <td>24.951834</td>
          <td>0.116757</td>
          <td>24.553256</td>
          <td>0.183695</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.511155</td>
          <td>0.790403</td>
          <td>27.402543</td>
          <td>0.323440</td>
          <td>26.244064</td>
          <td>0.198705</td>
          <td>25.623563</td>
          <td>0.216386</td>
          <td>24.814269</td>
          <td>0.238710</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.979435</td>
          <td>0.594988</td>
          <td>26.355667</td>
          <td>0.140756</td>
          <td>25.875843</td>
          <td>0.083157</td>
          <td>25.586303</td>
          <td>0.105569</td>
          <td>25.388383</td>
          <td>0.166446</td>
          <td>25.004509</td>
          <td>0.261770</td>
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
          <td>26.642739</td>
          <td>0.471849</td>
          <td>26.316239</td>
          <td>0.138574</td>
          <td>25.409262</td>
          <td>0.056193</td>
          <td>25.166782</td>
          <td>0.074585</td>
          <td>24.979130</td>
          <td>0.119407</td>
          <td>24.833106</td>
          <td>0.231899</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.919858</td>
          <td>0.227638</td>
          <td>25.745699</td>
          <td>0.074422</td>
          <td>25.239151</td>
          <td>0.078126</td>
          <td>24.700545</td>
          <td>0.092050</td>
          <td>24.303720</td>
          <td>0.145821</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.007202</td>
          <td>0.286116</td>
          <td>26.716820</td>
          <td>0.193544</td>
          <td>26.426729</td>
          <td>0.136200</td>
          <td>26.619979</td>
          <td>0.257450</td>
          <td>26.001399</td>
          <td>0.280635</td>
          <td>25.491559</td>
          <td>0.390299</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.733207</td>
          <td>0.231779</td>
          <td>26.262608</td>
          <td>0.133545</td>
          <td>26.097744</td>
          <td>0.104237</td>
          <td>25.828768</td>
          <td>0.134548</td>
          <td>25.676233</td>
          <td>0.218569</td>
          <td>25.031855</td>
          <td>0.275713</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.932255</td>
          <td>0.578817</td>
          <td>26.881251</td>
          <td>0.221562</td>
          <td>26.487144</td>
          <td>0.143093</td>
          <td>26.207261</td>
          <td>0.182032</td>
          <td>26.126002</td>
          <td>0.309495</td>
          <td>26.248761</td>
          <td>0.677290</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.958580</td>
          <td>0.204579</td>
          <td>25.906747</td>
          <td>0.072619</td>
          <td>25.171368</td>
          <td>0.061816</td>
          <td>24.720912</td>
          <td>0.079379</td>
          <td>24.047953</td>
          <td>0.098586</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.288067</td>
          <td>0.268811</td>
          <td>26.888832</td>
          <td>0.170891</td>
          <td>26.185686</td>
          <td>0.150396</td>
          <td>25.576280</td>
          <td>0.167143</td>
          <td>24.864812</td>
          <td>0.199282</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.237846</td>
          <td>1.257713</td>
          <td>31.797788</td>
          <td>3.265727</td>
          <td>27.624817</td>
          <td>0.337930</td>
          <td>25.923297</td>
          <td>0.130329</td>
          <td>25.150057</td>
          <td>0.125392</td>
          <td>24.467243</td>
          <td>0.154191</td>
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
          <td>27.818329</td>
          <td>0.445223</td>
          <td>26.485168</td>
          <td>0.242033</td>
          <td>25.612447</td>
          <td>0.213664</td>
          <td>25.629860</td>
          <td>0.453785</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.048897</td>
          <td>0.263392</td>
          <td>26.120943</td>
          <td>0.099749</td>
          <td>25.890980</td>
          <td>0.071706</td>
          <td>25.721413</td>
          <td>0.100572</td>
          <td>25.375009</td>
          <td>0.140728</td>
          <td>25.246444</td>
          <td>0.273400</td>
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
          <td>26.511856</td>
          <td>0.149686</td>
          <td>25.414241</td>
          <td>0.050806</td>
          <td>24.987732</td>
          <td>0.057079</td>
          <td>24.855787</td>
          <td>0.096702</td>
          <td>24.503822</td>
          <td>0.158605</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.316240</td>
          <td>0.329753</td>
          <td>26.810934</td>
          <td>0.183147</td>
          <td>26.097709</td>
          <td>0.087374</td>
          <td>25.149036</td>
          <td>0.061664</td>
          <td>24.882291</td>
          <td>0.093012</td>
          <td>24.222635</td>
          <td>0.116793</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.318847</td>
          <td>0.704983</td>
          <td>26.774555</td>
          <td>0.182521</td>
          <td>26.314137</td>
          <td>0.109110</td>
          <td>26.126978</td>
          <td>0.150200</td>
          <td>26.107848</td>
          <td>0.272525</td>
          <td>26.621302</td>
          <td>0.789400</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.024730</td>
          <td>0.278439</td>
          <td>26.358766</td>
          <td>0.135386</td>
          <td>26.129262</td>
          <td>0.099103</td>
          <td>26.098547</td>
          <td>0.156719</td>
          <td>25.524380</td>
          <td>0.178505</td>
          <td>24.809940</td>
          <td>0.212925</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.232565</td>
          <td>0.312960</td>
          <td>27.064612</td>
          <td>0.230752</td>
          <td>26.663866</td>
          <td>0.146289</td>
          <td>26.664442</td>
          <td>0.234040</td>
          <td>25.838503</td>
          <td>0.216238</td>
          <td>25.401699</td>
          <td>0.321050</td>
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
