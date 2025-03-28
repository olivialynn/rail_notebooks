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

    <pzflow.flow.Flow at 0x7f1a3ab5e500>



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
          <td>29.458902</td>
          <td>2.162183</td>
          <td>26.925308</td>
          <td>0.198927</td>
          <td>26.024332</td>
          <td>0.080558</td>
          <td>25.180489</td>
          <td>0.062309</td>
          <td>24.584787</td>
          <td>0.070372</td>
          <td>24.143654</td>
          <td>0.107184</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.429428</td>
          <td>1.344995</td>
          <td>27.466186</td>
          <td>0.310167</td>
          <td>26.561305</td>
          <td>0.128877</td>
          <td>25.948019</td>
          <td>0.122373</td>
          <td>26.498568</td>
          <td>0.356284</td>
          <td>25.490076</td>
          <td>0.332065</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.106776</td>
          <td>0.993607</td>
          <td>27.981237</td>
          <td>0.414799</td>
          <td>26.104182</td>
          <td>0.140077</td>
          <td>25.082072</td>
          <td>0.108992</td>
          <td>24.289920</td>
          <td>0.121749</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.687915</td>
          <td>2.362622</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.561246</td>
          <td>0.298221</td>
          <td>26.248933</td>
          <td>0.158616</td>
          <td>25.598633</td>
          <td>0.170198</td>
          <td>25.239082</td>
          <td>0.271396</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.522969</td>
          <td>0.383805</td>
          <td>26.037162</td>
          <td>0.092574</td>
          <td>26.056274</td>
          <td>0.082860</td>
          <td>25.826712</td>
          <td>0.110108</td>
          <td>25.302348</td>
          <td>0.131988</td>
          <td>25.100701</td>
          <td>0.242302</td>
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
          <td>28.174551</td>
          <td>1.170367</td>
          <td>26.523088</td>
          <td>0.141273</td>
          <td>25.419107</td>
          <td>0.047114</td>
          <td>25.051737</td>
          <td>0.055582</td>
          <td>24.961079</td>
          <td>0.098044</td>
          <td>24.759708</td>
          <td>0.182206</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.360105</td>
          <td>0.705977</td>
          <td>26.612668</td>
          <td>0.152570</td>
          <td>26.129257</td>
          <td>0.088362</td>
          <td>25.220571</td>
          <td>0.064563</td>
          <td>24.869869</td>
          <td>0.090498</td>
          <td>24.202456</td>
          <td>0.112827</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.903886</td>
          <td>0.999630</td>
          <td>26.649644</td>
          <td>0.157475</td>
          <td>26.393879</td>
          <td>0.111423</td>
          <td>26.072029</td>
          <td>0.136245</td>
          <td>25.682274</td>
          <td>0.182717</td>
          <td>25.497656</td>
          <td>0.334065</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.851026</td>
          <td>0.223611</td>
          <td>26.272455</td>
          <td>0.113714</td>
          <td>26.020117</td>
          <td>0.080259</td>
          <td>25.831043</td>
          <td>0.110525</td>
          <td>26.086093</td>
          <td>0.255843</td>
          <td>25.115551</td>
          <td>0.245285</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.391601</td>
          <td>0.346385</td>
          <td>26.730063</td>
          <td>0.168655</td>
          <td>26.714903</td>
          <td>0.147139</td>
          <td>26.305593</td>
          <td>0.166477</td>
          <td>25.639825</td>
          <td>0.176261</td>
          <td>26.128354</td>
          <td>0.539653</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>27.061814</td>
          <td>0.255001</td>
          <td>26.111985</td>
          <td>0.102290</td>
          <td>25.298348</td>
          <td>0.081957</td>
          <td>24.762486</td>
          <td>0.096787</td>
          <td>23.987278</td>
          <td>0.110393</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.933785</td>
          <td>0.575939</td>
          <td>27.426502</td>
          <td>0.342054</td>
          <td>27.025689</td>
          <td>0.223657</td>
          <td>26.227509</td>
          <td>0.183372</td>
          <td>25.899914</td>
          <td>0.255354</td>
          <td>26.820593</td>
          <td>0.973278</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.467698</td>
          <td>0.838187</td>
          <td>27.484191</td>
          <td>0.364354</td>
          <td>29.398633</td>
          <td>1.221472</td>
          <td>25.951368</td>
          <td>0.148156</td>
          <td>24.929527</td>
          <td>0.114512</td>
          <td>24.241046</td>
          <td>0.140709</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.000425</td>
          <td>0.556274</td>
          <td>27.787551</td>
          <td>0.436314</td>
          <td>26.161885</td>
          <td>0.185410</td>
          <td>25.627434</td>
          <td>0.217086</td>
          <td>25.058806</td>
          <td>0.291466</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.857849</td>
          <td>0.545394</td>
          <td>25.887828</td>
          <td>0.093721</td>
          <td>26.079760</td>
          <td>0.099476</td>
          <td>25.772981</td>
          <td>0.124205</td>
          <td>25.811316</td>
          <td>0.237423</td>
          <td>25.565186</td>
          <td>0.408464</td>
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
          <td>27.136025</td>
          <td>0.671994</td>
          <td>26.495139</td>
          <td>0.161546</td>
          <td>25.445521</td>
          <td>0.058030</td>
          <td>25.024824</td>
          <td>0.065783</td>
          <td>24.956889</td>
          <td>0.117121</td>
          <td>24.979988</td>
          <td>0.261693</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.633615</td>
          <td>0.179071</td>
          <td>26.098659</td>
          <td>0.101525</td>
          <td>25.376302</td>
          <td>0.088162</td>
          <td>24.871557</td>
          <td>0.106925</td>
          <td>24.050224</td>
          <td>0.117112</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.087010</td>
          <td>0.263124</td>
          <td>26.109395</td>
          <td>0.103367</td>
          <td>26.048371</td>
          <td>0.159458</td>
          <td>26.235721</td>
          <td>0.338569</td>
          <td>26.033688</td>
          <td>0.584104</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.344392</td>
          <td>0.777697</td>
          <td>26.349608</td>
          <td>0.143936</td>
          <td>26.032081</td>
          <td>0.098414</td>
          <td>25.768422</td>
          <td>0.127705</td>
          <td>25.900047</td>
          <td>0.262906</td>
          <td>25.344869</td>
          <td>0.354091</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.913221</td>
          <td>0.571001</td>
          <td>26.869865</td>
          <td>0.219474</td>
          <td>26.603219</td>
          <td>0.158077</td>
          <td>26.189715</td>
          <td>0.179347</td>
          <td>25.721582</td>
          <td>0.222431</td>
          <td>25.464967</td>
          <td>0.381393</td>
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
          <td>26.964836</td>
          <td>0.534946</td>
          <td>26.627344</td>
          <td>0.154516</td>
          <td>25.905876</td>
          <td>0.072563</td>
          <td>25.219615</td>
          <td>0.064518</td>
          <td>24.768891</td>
          <td>0.082811</td>
          <td>24.056249</td>
          <td>0.099306</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.195283</td>
          <td>0.630677</td>
          <td>27.023094</td>
          <td>0.216058</td>
          <td>26.534863</td>
          <td>0.126075</td>
          <td>26.011438</td>
          <td>0.129418</td>
          <td>26.434449</td>
          <td>0.339025</td>
          <td>24.831881</td>
          <td>0.193837</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.763566</td>
          <td>0.484540</td>
          <td>29.063603</td>
          <td>1.016915</td>
          <td>28.594756</td>
          <td>0.692159</td>
          <td>26.260515</td>
          <td>0.174046</td>
          <td>25.071039</td>
          <td>0.117075</td>
          <td>24.146236</td>
          <td>0.116856</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.066211</td>
          <td>0.581576</td>
          <td>27.063040</td>
          <td>0.244857</td>
          <td>26.519729</td>
          <td>0.249019</td>
          <td>25.384607</td>
          <td>0.176377</td>
          <td>26.696260</td>
          <td>0.943725</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.921681</td>
          <td>0.237291</td>
          <td>26.030907</td>
          <td>0.092181</td>
          <td>25.814407</td>
          <td>0.067006</td>
          <td>25.637174</td>
          <td>0.093408</td>
          <td>25.646404</td>
          <td>0.177491</td>
          <td>25.169659</td>
          <td>0.256787</td>
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
          <td>27.021144</td>
          <td>0.583061</td>
          <td>26.181562</td>
          <td>0.112504</td>
          <td>25.390444</td>
          <td>0.049744</td>
          <td>25.085318</td>
          <td>0.062240</td>
          <td>25.199815</td>
          <td>0.130512</td>
          <td>24.691922</td>
          <td>0.186107</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.847915</td>
          <td>0.974019</td>
          <td>26.907971</td>
          <td>0.198754</td>
          <td>26.122891</td>
          <td>0.089331</td>
          <td>25.254743</td>
          <td>0.067720</td>
          <td>24.875850</td>
          <td>0.092487</td>
          <td>24.146576</td>
          <td>0.109302</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.122642</td>
          <td>0.615752</td>
          <td>26.562239</td>
          <td>0.152338</td>
          <td>26.317451</td>
          <td>0.109426</td>
          <td>26.258320</td>
          <td>0.168050</td>
          <td>25.316412</td>
          <td>0.140173</td>
          <td>25.200280</td>
          <td>0.275605</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.558236</td>
          <td>0.423716</td>
          <td>26.417639</td>
          <td>0.142429</td>
          <td>26.009444</td>
          <td>0.089208</td>
          <td>26.193082</td>
          <td>0.169886</td>
          <td>25.477487</td>
          <td>0.171538</td>
          <td>25.230836</td>
          <td>0.300705</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.411923</td>
          <td>0.360631</td>
          <td>26.879420</td>
          <td>0.197713</td>
          <td>26.713253</td>
          <td>0.152624</td>
          <td>26.368212</td>
          <td>0.182631</td>
          <td>26.635315</td>
          <td>0.409982</td>
          <td>26.160053</td>
          <td>0.570709</td>
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
