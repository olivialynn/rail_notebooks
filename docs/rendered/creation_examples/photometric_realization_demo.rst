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

    <pzflow.flow.Flow at 0x7fb4e63dc3a0>



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
          <td>26.097517</td>
          <td>0.273774</td>
          <td>27.172954</td>
          <td>0.244435</td>
          <td>25.916448</td>
          <td>0.073235</td>
          <td>25.395109</td>
          <td>0.075351</td>
          <td>25.061218</td>
          <td>0.107025</td>
          <td>25.022233</td>
          <td>0.227072</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.660309</td>
          <td>0.361686</td>
          <td>27.553440</td>
          <td>0.296353</td>
          <td>26.907308</td>
          <td>0.274972</td>
          <td>26.565041</td>
          <td>0.375281</td>
          <td>26.883400</td>
          <td>0.899779</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.076642</td>
          <td>0.269168</td>
          <td>25.916680</td>
          <td>0.083272</td>
          <td>24.777521</td>
          <td>0.026748</td>
          <td>23.875140</td>
          <td>0.019809</td>
          <td>23.126496</td>
          <td>0.019558</td>
          <td>22.877858</td>
          <td>0.035007</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.069049</td>
          <td>1.101953</td>
          <td>27.906084</td>
          <td>0.437081</td>
          <td>27.132110</td>
          <td>0.209639</td>
          <td>26.669030</td>
          <td>0.226065</td>
          <td>27.300948</td>
          <td>0.645993</td>
          <td>25.234422</td>
          <td>0.270368</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.070954</td>
          <td>0.267925</td>
          <td>25.736253</td>
          <td>0.071023</td>
          <td>25.488045</td>
          <td>0.050088</td>
          <td>24.842309</td>
          <td>0.046150</td>
          <td>24.348446</td>
          <td>0.057071</td>
          <td>23.646990</td>
          <td>0.069227</td>
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
          <td>26.742948</td>
          <td>0.453981</td>
          <td>26.351466</td>
          <td>0.121797</td>
          <td>25.973068</td>
          <td>0.076993</td>
          <td>25.980636</td>
          <td>0.125886</td>
          <td>26.382575</td>
          <td>0.325085</td>
          <td>25.331674</td>
          <td>0.292546</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.308361</td>
          <td>0.324316</td>
          <td>26.980693</td>
          <td>0.208379</td>
          <td>26.670185</td>
          <td>0.141585</td>
          <td>26.057086</td>
          <td>0.134498</td>
          <td>26.289225</td>
          <td>0.301707</td>
          <td>25.068390</td>
          <td>0.235923</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.329895</td>
          <td>1.275270</td>
          <td>27.250130</td>
          <td>0.260416</td>
          <td>26.663767</td>
          <td>0.140805</td>
          <td>26.441932</td>
          <td>0.186896</td>
          <td>26.704927</td>
          <td>0.418033</td>
          <td>25.522479</td>
          <td>0.340691</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.174621</td>
          <td>0.621305</td>
          <td>28.326123</td>
          <td>0.594882</td>
          <td>26.603650</td>
          <td>0.133686</td>
          <td>25.961564</td>
          <td>0.123821</td>
          <td>25.573976</td>
          <td>0.166662</td>
          <td>26.439872</td>
          <td>0.672457</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.573662</td>
          <td>0.399124</td>
          <td>26.648957</td>
          <td>0.157382</td>
          <td>26.286103</td>
          <td>0.101406</td>
          <td>25.720469</td>
          <td>0.100340</td>
          <td>25.277994</td>
          <td>0.129235</td>
          <td>25.363330</td>
          <td>0.300100</td>
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
          <td>26.791766</td>
          <td>0.470889</td>
          <td>26.724870</td>
          <td>0.167912</td>
          <td>25.982612</td>
          <td>0.077645</td>
          <td>25.439534</td>
          <td>0.078367</td>
          <td>25.183987</td>
          <td>0.119113</td>
          <td>25.223442</td>
          <td>0.267960</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.222030</td>
          <td>0.642200</td>
          <td>28.182126</td>
          <td>0.536477</td>
          <td>27.238427</td>
          <td>0.229050</td>
          <td>26.783442</td>
          <td>0.248485</td>
          <td>27.243022</td>
          <td>0.620406</td>
          <td>26.847470</td>
          <td>0.879686</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.771687</td>
          <td>0.209323</td>
          <td>25.826765</td>
          <td>0.076930</td>
          <td>24.785623</td>
          <td>0.026938</td>
          <td>23.873458</td>
          <td>0.019781</td>
          <td>23.146977</td>
          <td>0.019900</td>
          <td>22.831213</td>
          <td>0.033595</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>29.155774</td>
          <td>1.905795</td>
          <td>32.218023</td>
          <td>3.589731</td>
          <td>27.735703</td>
          <td>0.342716</td>
          <td>26.494523</td>
          <td>0.195371</td>
          <td>26.069394</td>
          <td>0.252362</td>
          <td>25.892454</td>
          <td>0.453291</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.481071</td>
          <td>0.371518</td>
          <td>25.694557</td>
          <td>0.068453</td>
          <td>25.409870</td>
          <td>0.046729</td>
          <td>24.745377</td>
          <td>0.042346</td>
          <td>24.399794</td>
          <td>0.059732</td>
          <td>23.630990</td>
          <td>0.068254</td>
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
          <td>25.717725</td>
          <td>0.200085</td>
          <td>26.116311</td>
          <td>0.099223</td>
          <td>26.232881</td>
          <td>0.096784</td>
          <td>26.292643</td>
          <td>0.164649</td>
          <td>25.913278</td>
          <td>0.221811</td>
          <td>25.485146</td>
          <td>0.330769</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.962773</td>
          <td>0.534105</td>
          <td>26.775884</td>
          <td>0.175351</td>
          <td>27.042924</td>
          <td>0.194520</td>
          <td>26.391408</td>
          <td>0.179074</td>
          <td>26.106967</td>
          <td>0.260254</td>
          <td>25.843697</td>
          <td>0.436902</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.784078</td>
          <td>0.468193</td>
          <td>27.305917</td>
          <td>0.272539</td>
          <td>27.081556</td>
          <td>0.200942</td>
          <td>26.590810</td>
          <td>0.211803</td>
          <td>26.296044</td>
          <td>0.303364</td>
          <td>25.669239</td>
          <td>0.382181</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.722797</td>
          <td>0.447146</td>
          <td>27.183113</td>
          <td>0.246488</td>
          <td>26.579209</td>
          <td>0.130890</td>
          <td>26.329022</td>
          <td>0.169832</td>
          <td>25.546348</td>
          <td>0.162781</td>
          <td>25.035770</td>
          <td>0.229636</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.435540</td>
          <td>0.130997</td>
          <td>26.049271</td>
          <td>0.082350</td>
          <td>25.598016</td>
          <td>0.090114</td>
          <td>25.250255</td>
          <td>0.126167</td>
          <td>24.921591</td>
          <td>0.208802</td>
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
          <td>28.093526</td>
          <td>1.117617</td>
          <td>26.730316</td>
          <td>0.168692</td>
          <td>26.010993</td>
          <td>0.079615</td>
          <td>25.430761</td>
          <td>0.077762</td>
          <td>25.064717</td>
          <td>0.107353</td>
          <td>24.673862</td>
          <td>0.169401</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.668747</td>
          <td>0.364081</td>
          <td>27.645203</td>
          <td>0.318969</td>
          <td>26.641944</td>
          <td>0.221031</td>
          <td>28.115285</td>
          <td>1.086794</td>
          <td>28.758628</td>
          <td>2.292058</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.349035</td>
          <td>0.334944</td>
          <td>26.067850</td>
          <td>0.095099</td>
          <td>24.737476</td>
          <td>0.025831</td>
          <td>23.883098</td>
          <td>0.019943</td>
          <td>23.145816</td>
          <td>0.019880</td>
          <td>22.860288</td>
          <td>0.034468</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.570722</td>
          <td>0.398222</td>
          <td>27.716293</td>
          <td>0.377828</td>
          <td>27.214611</td>
          <td>0.224566</td>
          <td>26.518050</td>
          <td>0.199275</td>
          <td>25.777494</td>
          <td>0.197998</td>
          <td>25.238668</td>
          <td>0.271305</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.173445</td>
          <td>0.291119</td>
          <td>25.752650</td>
          <td>0.072059</td>
          <td>25.444816</td>
          <td>0.048202</td>
          <td>24.779281</td>
          <td>0.043639</td>
          <td>24.415086</td>
          <td>0.060547</td>
          <td>23.667867</td>
          <td>0.070518</td>
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
          <td>26.320407</td>
          <td>0.327433</td>
          <td>26.205101</td>
          <td>0.107231</td>
          <td>26.192796</td>
          <td>0.093438</td>
          <td>26.405627</td>
          <td>0.181244</td>
          <td>25.434342</td>
          <td>0.147893</td>
          <td>25.175449</td>
          <td>0.257653</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.539563</td>
          <td>0.388765</td>
          <td>27.023583</td>
          <td>0.215977</td>
          <td>26.670453</td>
          <td>0.141618</td>
          <td>26.200432</td>
          <td>0.152163</td>
          <td>26.485903</td>
          <td>0.352758</td>
          <td>25.637337</td>
          <td>0.372818</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.163321</td>
          <td>0.288752</td>
          <td>27.228236</td>
          <td>0.255791</td>
          <td>26.844024</td>
          <td>0.164340</td>
          <td>26.193216</td>
          <td>0.151224</td>
          <td>26.010545</td>
          <td>0.240429</td>
          <td>25.968847</td>
          <td>0.479960</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.958823</td>
          <td>1.745568</td>
          <td>27.112876</td>
          <td>0.232607</td>
          <td>26.525908</td>
          <td>0.124983</td>
          <td>25.805775</td>
          <td>0.108114</td>
          <td>25.579188</td>
          <td>0.167404</td>
          <td>25.534770</td>
          <td>0.344012</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>25.853982</td>
          <td>0.224160</td>
          <td>26.312133</td>
          <td>0.117707</td>
          <td>25.976530</td>
          <td>0.077229</td>
          <td>25.604567</td>
          <td>0.090634</td>
          <td>25.382980</td>
          <td>0.141501</td>
          <td>24.971039</td>
          <td>0.217605</td>
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
