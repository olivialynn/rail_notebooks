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

    <pzflow.flow.Flow at 0x7f83f4a21240>



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
    0      23.994413  0.257724  0.236919  
    1      25.391064  0.117174  0.077782  
    2      24.304707  0.096522  0.050911  
    3      25.291103  0.048625  0.040513  
    4      25.096743  0.021488  0.011992  
    ...          ...       ...       ...  
    99995  24.737946  0.136046  0.119798  
    99996  24.224169  0.007405  0.004264  
    99997  25.613836  0.280931  0.215921  
    99998  25.274899  0.071013  0.058087  
    99999  25.699642  0.072582  0.070343  
    
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
          <td>26.904325</td>
          <td>0.511791</td>
          <td>26.667372</td>
          <td>0.159879</td>
          <td>25.920097</td>
          <td>0.073472</td>
          <td>25.208308</td>
          <td>0.063865</td>
          <td>24.896557</td>
          <td>0.092646</td>
          <td>23.907898</td>
          <td>0.087162</td>
          <td>0.257724</td>
          <td>0.236919</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.154122</td>
          <td>0.240671</td>
          <td>26.722734</td>
          <td>0.148132</td>
          <td>26.266421</td>
          <td>0.161004</td>
          <td>26.013712</td>
          <td>0.241058</td>
          <td>25.793898</td>
          <td>0.420664</td>
          <td>0.117174</td>
          <td>0.077782</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.468006</td>
          <td>1.372528</td>
          <td>28.491071</td>
          <td>0.667508</td>
          <td>27.555986</td>
          <td>0.296961</td>
          <td>26.127086</td>
          <td>0.142869</td>
          <td>25.036915</td>
          <td>0.104775</td>
          <td>24.141204</td>
          <td>0.106955</td>
          <td>0.096522</td>
          <td>0.050911</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.034145</td>
          <td>1.806203</td>
          <td>29.053122</td>
          <td>0.961765</td>
          <td>27.495814</td>
          <td>0.282874</td>
          <td>26.390682</td>
          <td>0.178964</td>
          <td>25.954949</td>
          <td>0.229623</td>
          <td>24.590092</td>
          <td>0.157715</td>
          <td>0.048625</td>
          <td>0.040513</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.782226</td>
          <td>0.467546</td>
          <td>26.003095</td>
          <td>0.089846</td>
          <td>26.012896</td>
          <td>0.079749</td>
          <td>25.745453</td>
          <td>0.102559</td>
          <td>25.372637</td>
          <td>0.140245</td>
          <td>25.756291</td>
          <td>0.408732</td>
          <td>0.021488</td>
          <td>0.011992</td>
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
          <td>28.316255</td>
          <td>1.265865</td>
          <td>26.404990</td>
          <td>0.127580</td>
          <td>25.417975</td>
          <td>0.047066</td>
          <td>25.059635</td>
          <td>0.055973</td>
          <td>24.661515</td>
          <td>0.075313</td>
          <td>24.546545</td>
          <td>0.151941</td>
          <td>0.136046</td>
          <td>0.119798</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.947001</td>
          <td>0.528011</td>
          <td>26.573466</td>
          <td>0.147526</td>
          <td>26.065209</td>
          <td>0.083515</td>
          <td>25.305721</td>
          <td>0.069622</td>
          <td>24.838307</td>
          <td>0.088021</td>
          <td>24.312602</td>
          <td>0.124170</td>
          <td>0.007405</td>
          <td>0.004264</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.585204</td>
          <td>0.149019</td>
          <td>26.318981</td>
          <td>0.104367</td>
          <td>26.219915</td>
          <td>0.154725</td>
          <td>25.663097</td>
          <td>0.179774</td>
          <td>26.194592</td>
          <td>0.566070</td>
          <td>0.280931</td>
          <td>0.215921</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.396358</td>
          <td>0.347684</td>
          <td>26.192842</td>
          <td>0.106089</td>
          <td>25.991096</td>
          <td>0.078229</td>
          <td>25.817902</td>
          <td>0.109265</td>
          <td>25.713505</td>
          <td>0.187606</td>
          <td>25.457809</td>
          <td>0.323661</td>
          <td>0.071013</td>
          <td>0.058087</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.920454</td>
          <td>1.009626</td>
          <td>26.684744</td>
          <td>0.162267</td>
          <td>26.570659</td>
          <td>0.129925</td>
          <td>26.188614</td>
          <td>0.150628</td>
          <td>25.990434</td>
          <td>0.236468</td>
          <td>25.590438</td>
          <td>0.359404</td>
          <td>0.072582</td>
          <td>0.070343</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.680556</td>
          <td>0.216012</td>
          <td>25.984455</td>
          <td>0.108716</td>
          <td>25.210920</td>
          <td>0.090747</td>
          <td>24.796046</td>
          <td>0.118384</td>
          <td>23.921180</td>
          <td>0.124223</td>
          <td>0.257724</td>
          <td>0.236919</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.001220</td>
          <td>0.249364</td>
          <td>26.475879</td>
          <td>0.144850</td>
          <td>25.943238</td>
          <td>0.148633</td>
          <td>26.345847</td>
          <td>0.375625</td>
          <td>24.960363</td>
          <td>0.260344</td>
          <td>0.117174</td>
          <td>0.077782</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.429610</td>
          <td>2.901005</td>
          <td>26.238771</td>
          <td>0.188754</td>
          <td>25.022994</td>
          <td>0.123886</td>
          <td>24.534846</td>
          <td>0.180395</td>
          <td>0.096522</td>
          <td>0.050911</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.988212</td>
          <td>0.601051</td>
          <td>28.863308</td>
          <td>0.949819</td>
          <td>27.559314</td>
          <td>0.346802</td>
          <td>26.046829</td>
          <td>0.158283</td>
          <td>25.271851</td>
          <td>0.151609</td>
          <td>25.790036</td>
          <td>0.486758</td>
          <td>0.048625</td>
          <td>0.040513</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.823461</td>
          <td>0.244279</td>
          <td>26.053786</td>
          <td>0.108422</td>
          <td>26.031549</td>
          <td>0.095424</td>
          <td>25.458642</td>
          <td>0.094466</td>
          <td>25.256114</td>
          <td>0.148739</td>
          <td>25.382209</td>
          <td>0.354594</td>
          <td>0.021488</td>
          <td>0.011992</td>
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
          <td>26.704218</td>
          <td>0.504212</td>
          <td>26.259119</td>
          <td>0.135724</td>
          <td>25.386589</td>
          <td>0.056882</td>
          <td>25.056986</td>
          <td>0.069971</td>
          <td>24.796360</td>
          <td>0.105098</td>
          <td>24.640844</td>
          <td>0.203794</td>
          <td>0.136046</td>
          <td>0.119798</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.382014</td>
          <td>0.783008</td>
          <td>26.783931</td>
          <td>0.202536</td>
          <td>25.948732</td>
          <td>0.088647</td>
          <td>25.314946</td>
          <td>0.083173</td>
          <td>24.741851</td>
          <td>0.095060</td>
          <td>24.144779</td>
          <td>0.126605</td>
          <td>0.007405</td>
          <td>0.004264</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.933212</td>
          <td>0.303428</td>
          <td>26.508280</td>
          <td>0.187334</td>
          <td>26.413262</td>
          <td>0.157830</td>
          <td>26.037658</td>
          <td>0.185753</td>
          <td>26.589593</td>
          <td>0.510796</td>
          <td>25.888015</td>
          <td>0.602328</td>
          <td>0.280931</td>
          <td>0.215921</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.231796</td>
          <td>0.342592</td>
          <td>26.200278</td>
          <td>0.124596</td>
          <td>25.975894</td>
          <td>0.092072</td>
          <td>26.103874</td>
          <td>0.167411</td>
          <td>25.696104</td>
          <td>0.218601</td>
          <td>25.311199</td>
          <td>0.339371</td>
          <td>0.071013</td>
          <td>0.058087</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.493114</td>
          <td>0.849429</td>
          <td>26.520872</td>
          <td>0.164561</td>
          <td>26.843078</td>
          <td>0.195081</td>
          <td>26.508025</td>
          <td>0.235767</td>
          <td>26.261943</td>
          <td>0.346980</td>
          <td>25.409566</td>
          <td>0.367639</td>
          <td>0.072582</td>
          <td>0.070343</td>
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
          <td>28.261366</td>
          <td>1.536673</td>
          <td>27.347459</td>
          <td>0.417912</td>
          <td>25.873824</td>
          <td>0.114221</td>
          <td>25.066919</td>
          <td>0.092917</td>
          <td>24.782117</td>
          <td>0.135189</td>
          <td>24.259440</td>
          <td>0.191882</td>
          <td>0.257724</td>
          <td>0.236919</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.223944</td>
          <td>0.601152</td>
          <td>26.418334</td>
          <td>0.127758</td>
          <td>26.208806</td>
          <td>0.172520</td>
          <td>25.612634</td>
          <td>0.192710</td>
          <td>25.707915</td>
          <td>0.437414</td>
          <td>0.117174</td>
          <td>0.077782</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.510645</td>
          <td>2.086620</td>
          <td>28.208308</td>
          <td>0.523193</td>
          <td>26.160548</td>
          <td>0.158439</td>
          <td>25.003330</td>
          <td>0.109422</td>
          <td>24.401026</td>
          <td>0.144404</td>
          <td>0.096522</td>
          <td>0.050911</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.404346</td>
          <td>2.133263</td>
          <td>28.595422</td>
          <td>0.729380</td>
          <td>27.414988</td>
          <td>0.271457</td>
          <td>26.450457</td>
          <td>0.193328</td>
          <td>25.500819</td>
          <td>0.160648</td>
          <td>25.417546</td>
          <td>0.321347</td>
          <td>0.048625</td>
          <td>0.040513</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.139335</td>
          <td>0.283952</td>
          <td>26.034191</td>
          <td>0.092662</td>
          <td>25.980419</td>
          <td>0.077815</td>
          <td>25.851906</td>
          <td>0.113036</td>
          <td>25.535081</td>
          <td>0.161864</td>
          <td>25.016608</td>
          <td>0.226924</td>
          <td>0.021488</td>
          <td>0.011992</td>
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
          <td>26.962485</td>
          <td>0.595558</td>
          <td>26.523514</td>
          <td>0.165552</td>
          <td>25.375569</td>
          <td>0.054556</td>
          <td>25.080137</td>
          <td>0.069106</td>
          <td>24.797104</td>
          <td>0.101911</td>
          <td>24.922520</td>
          <td>0.249726</td>
          <td>0.136046</td>
          <td>0.119798</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.964090</td>
          <td>1.750093</td>
          <td>26.538788</td>
          <td>0.143255</td>
          <td>25.944487</td>
          <td>0.075111</td>
          <td>25.166139</td>
          <td>0.061554</td>
          <td>24.838913</td>
          <td>0.088111</td>
          <td>24.118370</td>
          <td>0.104895</td>
          <td>0.007405</td>
          <td>0.004264</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.116378</td>
          <td>1.430379</td>
          <td>26.854973</td>
          <td>0.284024</td>
          <td>26.574383</td>
          <td>0.208215</td>
          <td>26.689437</td>
          <td>0.363133</td>
          <td>25.946946</td>
          <td>0.355526</td>
          <td>24.990386</td>
          <td>0.349022</td>
          <td>0.280931</td>
          <td>0.215921</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.917825</td>
          <td>0.244894</td>
          <td>26.184506</td>
          <td>0.110370</td>
          <td>26.007208</td>
          <td>0.083770</td>
          <td>25.963435</td>
          <td>0.131130</td>
          <td>25.920544</td>
          <td>0.234793</td>
          <td>25.236476</td>
          <td>0.285180</td>
          <td>0.071013</td>
          <td>0.058087</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.750065</td>
          <td>0.474553</td>
          <td>26.579346</td>
          <td>0.156704</td>
          <td>26.489606</td>
          <td>0.129114</td>
          <td>26.498040</td>
          <td>0.209070</td>
          <td>26.064493</td>
          <td>0.266974</td>
          <td>25.114345</td>
          <td>0.260917</td>
          <td>0.072582</td>
          <td>0.070343</td>
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
