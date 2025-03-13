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

    <pzflow.flow.Flow at 0x7f7991c43fa0>



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
          <td>27.621668</td>
          <td>0.838802</td>
          <td>26.666887</td>
          <td>0.159812</td>
          <td>25.998060</td>
          <td>0.078712</td>
          <td>25.247268</td>
          <td>0.066109</td>
          <td>24.720793</td>
          <td>0.079360</td>
          <td>23.987105</td>
          <td>0.093448</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.431536</td>
          <td>0.301669</td>
          <td>26.671532</td>
          <td>0.141750</td>
          <td>26.317832</td>
          <td>0.168222</td>
          <td>25.893505</td>
          <td>0.218189</td>
          <td>24.995731</td>
          <td>0.222126</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.049730</td>
          <td>2.688919</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.871013</td>
          <td>0.381014</td>
          <td>26.153686</td>
          <td>0.146176</td>
          <td>25.054430</td>
          <td>0.106392</td>
          <td>24.233460</td>
          <td>0.115916</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.860277</td>
          <td>1.667507</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.159369</td>
          <td>0.214468</td>
          <td>26.336425</td>
          <td>0.170905</td>
          <td>25.688752</td>
          <td>0.183721</td>
          <td>25.985220</td>
          <td>0.485836</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.896000</td>
          <td>0.232097</td>
          <td>26.084884</td>
          <td>0.096530</td>
          <td>26.050341</td>
          <td>0.082428</td>
          <td>25.654182</td>
          <td>0.094673</td>
          <td>25.399153</td>
          <td>0.143485</td>
          <td>25.113483</td>
          <td>0.244868</td>
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
          <td>28.504553</td>
          <td>1.398869</td>
          <td>26.362335</td>
          <td>0.122951</td>
          <td>25.431541</td>
          <td>0.047637</td>
          <td>25.093305</td>
          <td>0.057671</td>
          <td>24.843477</td>
          <td>0.088422</td>
          <td>24.593603</td>
          <td>0.158189</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.453764</td>
          <td>1.362331</td>
          <td>26.793081</td>
          <td>0.177927</td>
          <td>26.218512</td>
          <td>0.095572</td>
          <td>25.168575</td>
          <td>0.061654</td>
          <td>24.840745</td>
          <td>0.088210</td>
          <td>24.311044</td>
          <td>0.124002</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.462373</td>
          <td>0.756035</td>
          <td>26.677679</td>
          <td>0.161292</td>
          <td>26.315184</td>
          <td>0.104020</td>
          <td>26.088592</td>
          <td>0.138207</td>
          <td>25.861480</td>
          <td>0.212436</td>
          <td>25.727469</td>
          <td>0.399777</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.688061</td>
          <td>0.435560</td>
          <td>26.187325</td>
          <td>0.105579</td>
          <td>25.993397</td>
          <td>0.078388</td>
          <td>25.920399</td>
          <td>0.119471</td>
          <td>25.614056</td>
          <td>0.172445</td>
          <td>24.929931</td>
          <td>0.210264</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.808940</td>
          <td>0.476956</td>
          <td>26.626489</td>
          <td>0.154386</td>
          <td>26.464243</td>
          <td>0.118465</td>
          <td>26.211335</td>
          <td>0.153592</td>
          <td>26.103133</td>
          <td>0.259439</td>
          <td>26.383469</td>
          <td>0.646785</td>
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
          <td>27.636272</td>
          <td>0.920918</td>
          <td>26.946626</td>
          <td>0.231919</td>
          <td>25.947391</td>
          <td>0.088535</td>
          <td>25.188061</td>
          <td>0.074355</td>
          <td>24.662349</td>
          <td>0.088640</td>
          <td>24.261112</td>
          <td>0.139981</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.304099</td>
          <td>0.743749</td>
          <td>27.131655</td>
          <td>0.270018</td>
          <td>26.516884</td>
          <td>0.145394</td>
          <td>26.305210</td>
          <td>0.195795</td>
          <td>25.955621</td>
          <td>0.267254</td>
          <td>26.040338</td>
          <td>0.580763</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.603407</td>
          <td>0.913239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.133762</td>
          <td>0.543093</td>
          <td>25.939774</td>
          <td>0.146688</td>
          <td>25.142956</td>
          <td>0.137782</td>
          <td>24.125245</td>
          <td>0.127314</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.980422</td>
          <td>0.548304</td>
          <td>28.014809</td>
          <td>0.516860</td>
          <td>26.209635</td>
          <td>0.193032</td>
          <td>25.177274</td>
          <td>0.148282</td>
          <td>25.805782</td>
          <td>0.518683</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.663011</td>
          <td>0.213800</td>
          <td>26.088322</td>
          <td>0.111667</td>
          <td>25.945643</td>
          <td>0.088427</td>
          <td>25.569319</td>
          <td>0.104013</td>
          <td>25.341266</td>
          <td>0.159887</td>
          <td>24.968363</td>
          <td>0.254136</td>
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
          <td>27.345924</td>
          <td>0.773831</td>
          <td>26.209578</td>
          <td>0.126380</td>
          <td>25.410103</td>
          <td>0.056235</td>
          <td>25.017403</td>
          <td>0.065352</td>
          <td>24.780114</td>
          <td>0.100375</td>
          <td>24.849002</td>
          <td>0.234970</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.476653</td>
          <td>0.411615</td>
          <td>26.963898</td>
          <td>0.236090</td>
          <td>26.092434</td>
          <td>0.100973</td>
          <td>25.302526</td>
          <td>0.082617</td>
          <td>24.949125</td>
          <td>0.114409</td>
          <td>24.165566</td>
          <td>0.129439</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.480242</td>
          <td>0.158357</td>
          <td>26.348454</td>
          <td>0.127286</td>
          <td>26.172210</td>
          <td>0.177191</td>
          <td>26.142161</td>
          <td>0.314305</td>
          <td>25.178965</td>
          <td>0.305079</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.323964</td>
          <td>0.372628</td>
          <td>26.435394</td>
          <td>0.154924</td>
          <td>26.227009</td>
          <td>0.116678</td>
          <td>26.056447</td>
          <td>0.163596</td>
          <td>25.666064</td>
          <td>0.216725</td>
          <td>26.033117</td>
          <td>0.592963</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.244102</td>
          <td>0.718541</td>
          <td>26.797845</td>
          <td>0.206671</td>
          <td>26.406324</td>
          <td>0.133459</td>
          <td>27.063545</td>
          <td>0.366316</td>
          <td>26.420448</td>
          <td>0.390235</td>
          <td>27.600937</td>
          <td>1.507374</td>
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
          <td>27.230271</td>
          <td>0.645930</td>
          <td>26.626908</td>
          <td>0.154459</td>
          <td>26.091598</td>
          <td>0.085491</td>
          <td>25.280251</td>
          <td>0.068079</td>
          <td>24.728030</td>
          <td>0.079879</td>
          <td>23.956899</td>
          <td>0.091013</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.549808</td>
          <td>0.331778</td>
          <td>27.066383</td>
          <td>0.198577</td>
          <td>26.200608</td>
          <td>0.152333</td>
          <td>25.675082</td>
          <td>0.181774</td>
          <td>25.597768</td>
          <td>0.361790</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.157981</td>
          <td>0.643329</td>
          <td>29.772436</td>
          <td>1.499658</td>
          <td>28.205551</td>
          <td>0.525913</td>
          <td>26.010029</td>
          <td>0.140465</td>
          <td>25.097070</td>
          <td>0.119755</td>
          <td>24.303450</td>
          <td>0.133923</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.821673</td>
          <td>1.787291</td>
          <td>30.610579</td>
          <td>2.307777</td>
          <td>27.194831</td>
          <td>0.272752</td>
          <td>26.090875</td>
          <td>0.173966</td>
          <td>25.668354</td>
          <td>0.223848</td>
          <td>25.195078</td>
          <td>0.324014</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.708822</td>
          <td>0.198784</td>
          <td>26.019534</td>
          <td>0.091265</td>
          <td>25.834467</td>
          <td>0.068208</td>
          <td>25.614310</td>
          <td>0.091551</td>
          <td>25.625656</td>
          <td>0.174392</td>
          <td>24.866675</td>
          <td>0.199688</td>
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
          <td>26.864698</td>
          <td>0.520833</td>
          <td>26.149952</td>
          <td>0.109448</td>
          <td>25.406584</td>
          <td>0.050462</td>
          <td>25.135372</td>
          <td>0.065064</td>
          <td>24.988337</td>
          <td>0.108595</td>
          <td>24.679898</td>
          <td>0.184225</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.669111</td>
          <td>0.162361</td>
          <td>25.985519</td>
          <td>0.079146</td>
          <td>25.101385</td>
          <td>0.059111</td>
          <td>24.606250</td>
          <td>0.072921</td>
          <td>24.003400</td>
          <td>0.096430</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.761831</td>
          <td>0.474153</td>
          <td>27.096906</td>
          <td>0.238969</td>
          <td>26.353140</td>
          <td>0.112886</td>
          <td>26.403642</td>
          <td>0.190081</td>
          <td>25.873135</td>
          <td>0.224679</td>
          <td>25.553639</td>
          <td>0.365333</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.813875</td>
          <td>0.512938</td>
          <td>26.209600</td>
          <td>0.118987</td>
          <td>26.072278</td>
          <td>0.094271</td>
          <td>25.727373</td>
          <td>0.113725</td>
          <td>25.463952</td>
          <td>0.169574</td>
          <td>24.910672</td>
          <td>0.231534</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.769902</td>
          <td>0.474118</td>
          <td>26.904697</td>
          <td>0.201953</td>
          <td>26.570581</td>
          <td>0.134989</td>
          <td>26.491763</td>
          <td>0.202671</td>
          <td>25.796829</td>
          <td>0.208840</td>
          <td>25.359304</td>
          <td>0.310363</td>
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
