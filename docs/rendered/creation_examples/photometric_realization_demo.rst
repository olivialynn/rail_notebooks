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

    <pzflow.flow.Flow at 0x7fb91eb87460>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.702927</td>
          <td>0.164803</td>
          <td>25.927113</td>
          <td>0.073929</td>
          <td>25.225828</td>
          <td>0.064865</td>
          <td>24.883840</td>
          <td>0.091617</td>
          <td>25.047112</td>
          <td>0.231805</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>26.770534</td>
          <td>0.463474</td>
          <td>28.307628</td>
          <td>0.587122</td>
          <td>28.085003</td>
          <td>0.448818</td>
          <td>27.066072</td>
          <td>0.312530</td>
          <td>27.267802</td>
          <td>0.631258</td>
          <td>25.513329</td>
          <td>0.338236</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.584113</td>
          <td>0.402345</td>
          <td>25.944268</td>
          <td>0.085319</td>
          <td>24.787990</td>
          <td>0.026993</td>
          <td>23.863124</td>
          <td>0.019609</td>
          <td>23.143462</td>
          <td>0.019841</td>
          <td>22.821030</td>
          <td>0.033295</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.842392</td>
          <td>1.653500</td>
          <td>28.464735</td>
          <td>0.655498</td>
          <td>27.325827</td>
          <td>0.246204</td>
          <td>26.439777</td>
          <td>0.186556</td>
          <td>25.962621</td>
          <td>0.231087</td>
          <td>25.528589</td>
          <td>0.342339</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.118921</td>
          <td>0.278569</td>
          <td>25.679982</td>
          <td>0.067577</td>
          <td>25.377309</td>
          <td>0.045398</td>
          <td>24.795518</td>
          <td>0.044273</td>
          <td>24.412094</td>
          <td>0.060387</td>
          <td>23.749268</td>
          <td>0.075781</td>
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
          <td>26.073001</td>
          <td>0.268372</td>
          <td>26.217001</td>
          <td>0.108350</td>
          <td>26.125258</td>
          <td>0.088051</td>
          <td>26.049865</td>
          <td>0.133662</td>
          <td>25.731973</td>
          <td>0.190552</td>
          <td>26.394408</td>
          <td>0.651707</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.317003</td>
          <td>0.326549</td>
          <td>26.817654</td>
          <td>0.181668</td>
          <td>27.263267</td>
          <td>0.233813</td>
          <td>26.351365</td>
          <td>0.173090</td>
          <td>26.300176</td>
          <td>0.304372</td>
          <td>24.887705</td>
          <td>0.202958</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>29.376805</td>
          <td>2.091672</td>
          <td>26.956303</td>
          <td>0.204167</td>
          <td>26.784146</td>
          <td>0.156143</td>
          <td>26.305056</td>
          <td>0.166401</td>
          <td>26.010196</td>
          <td>0.240360</td>
          <td>25.860600</td>
          <td>0.442528</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.249309</td>
          <td>0.260242</td>
          <td>26.645061</td>
          <td>0.138552</td>
          <td>25.961670</td>
          <td>0.123832</td>
          <td>26.046066</td>
          <td>0.247570</td>
          <td>25.441224</td>
          <td>0.319413</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.683157</td>
          <td>0.433944</td>
          <td>26.684901</td>
          <td>0.162289</td>
          <td>26.175956</td>
          <td>0.092066</td>
          <td>25.854709</td>
          <td>0.112830</td>
          <td>25.331830</td>
          <td>0.135394</td>
          <td>25.063886</td>
          <td>0.235046</td>
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
          <td>27.779105</td>
          <td>1.004831</td>
          <td>26.417035</td>
          <td>0.148336</td>
          <td>25.931976</td>
          <td>0.087342</td>
          <td>25.390120</td>
          <td>0.088855</td>
          <td>25.253041</td>
          <td>0.148202</td>
          <td>25.213023</td>
          <td>0.309788</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.264653</td>
          <td>0.638731</td>
          <td>27.368651</td>
          <td>0.296164</td>
          <td>27.235200</td>
          <td>0.414588</td>
          <td>26.595861</td>
          <td>0.442458</td>
          <td>26.148259</td>
          <td>0.626764</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.024616</td>
          <td>0.292118</td>
          <td>26.092583</td>
          <td>0.114314</td>
          <td>24.790341</td>
          <td>0.032532</td>
          <td>23.896561</td>
          <td>0.024355</td>
          <td>23.179911</td>
          <td>0.024523</td>
          <td>22.874364</td>
          <td>0.042289</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.256301</td>
          <td>0.287639</td>
          <td>26.539786</td>
          <td>0.254026</td>
          <td>26.375877</td>
          <td>0.396409</td>
          <td>25.164851</td>
          <td>0.317355</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.952445</td>
          <td>0.271313</td>
          <td>25.949576</td>
          <td>0.098929</td>
          <td>25.465277</td>
          <td>0.057832</td>
          <td>24.839017</td>
          <td>0.054597</td>
          <td>24.347171</td>
          <td>0.067136</td>
          <td>23.646037</td>
          <td>0.081864</td>
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
          <td>26.130807</td>
          <td>0.317756</td>
          <td>26.369096</td>
          <td>0.145020</td>
          <td>26.013836</td>
          <td>0.095850</td>
          <td>25.984503</td>
          <td>0.152229</td>
          <td>25.908214</td>
          <td>0.262118</td>
          <td>26.241948</td>
          <td>0.680071</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.893704</td>
          <td>0.561017</td>
          <td>27.380503</td>
          <td>0.330894</td>
          <td>27.055218</td>
          <td>0.230067</td>
          <td>26.721893</td>
          <td>0.277459</td>
          <td>26.027775</td>
          <td>0.284431</td>
          <td>26.957562</td>
          <td>1.059301</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.605483</td>
          <td>0.456426</td>
          <td>27.645542</td>
          <td>0.409708</td>
          <td>27.228794</td>
          <td>0.267512</td>
          <td>26.397615</td>
          <td>0.214203</td>
          <td>25.908715</td>
          <td>0.260231</td>
          <td>24.884076</td>
          <td>0.239968</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.923469</td>
          <td>0.582900</td>
          <td>26.942961</td>
          <td>0.237481</td>
          <td>26.523568</td>
          <td>0.150762</td>
          <td>25.827263</td>
          <td>0.134373</td>
          <td>25.894701</td>
          <td>0.261760</td>
          <td>25.215357</td>
          <td>0.319601</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.477569</td>
          <td>0.413509</td>
          <td>26.782721</td>
          <td>0.204070</td>
          <td>26.051961</td>
          <td>0.098028</td>
          <td>25.801589</td>
          <td>0.128592</td>
          <td>25.170704</td>
          <td>0.139430</td>
          <td>24.878294</td>
          <td>0.238194</td>
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
          <td>27.825587</td>
          <td>0.953274</td>
          <td>26.735766</td>
          <td>0.169494</td>
          <td>26.177196</td>
          <td>0.092178</td>
          <td>25.327337</td>
          <td>0.070977</td>
          <td>24.914026</td>
          <td>0.094091</td>
          <td>25.053618</td>
          <td>0.233087</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.452670</td>
          <td>0.650469</td>
          <td>28.271644</td>
          <td>0.516047</td>
          <td>27.667805</td>
          <td>0.497340</td>
          <td>27.572789</td>
          <td>0.776878</td>
          <td>25.833512</td>
          <td>0.433909</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.918805</td>
          <td>0.089671</td>
          <td>24.793395</td>
          <td>0.029438</td>
          <td>23.910575</td>
          <td>0.022199</td>
          <td>23.121875</td>
          <td>0.021099</td>
          <td>22.865687</td>
          <td>0.037729</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.727639</td>
          <td>1.712046</td>
          <td>26.847797</td>
          <td>0.225625</td>
          <td>27.166242</td>
          <td>0.266472</td>
          <td>26.404419</td>
          <td>0.226391</td>
          <td>25.615835</td>
          <td>0.214269</td>
          <td>25.727518</td>
          <td>0.488124</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.261991</td>
          <td>0.312834</td>
          <td>25.923181</td>
          <td>0.083854</td>
          <td>25.368716</td>
          <td>0.045118</td>
          <td>24.827954</td>
          <td>0.045634</td>
          <td>24.436622</td>
          <td>0.061804</td>
          <td>23.633719</td>
          <td>0.068521</td>
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
          <td>26.156507</td>
          <td>0.302252</td>
          <td>26.245564</td>
          <td>0.118944</td>
          <td>26.086763</td>
          <td>0.092105</td>
          <td>26.136767</td>
          <td>0.156134</td>
          <td>25.850658</td>
          <td>0.226815</td>
          <td>26.623545</td>
          <td>0.808829</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.980500</td>
          <td>0.211204</td>
          <td>26.845896</td>
          <td>0.167255</td>
          <td>26.265458</td>
          <td>0.163597</td>
          <td>26.006758</td>
          <td>0.243409</td>
          <td>25.685788</td>
          <td>0.392984</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>31.043005</td>
          <td>3.663829</td>
          <td>27.128058</td>
          <td>0.245184</td>
          <td>26.937521</td>
          <td>0.186529</td>
          <td>26.301034</td>
          <td>0.174268</td>
          <td>25.789478</td>
          <td>0.209543</td>
          <td>24.926004</td>
          <td>0.219919</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.643920</td>
          <td>0.452104</td>
          <td>27.574510</td>
          <td>0.369959</td>
          <td>26.966880</td>
          <td>0.203623</td>
          <td>25.853800</td>
          <td>0.126931</td>
          <td>25.672512</td>
          <td>0.202260</td>
          <td>26.798593</td>
          <td>0.928297</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.527749</td>
          <td>0.805418</td>
          <td>26.726274</td>
          <td>0.173722</td>
          <td>26.102274</td>
          <td>0.089721</td>
          <td>25.640334</td>
          <td>0.097422</td>
          <td>25.323343</td>
          <td>0.139637</td>
          <td>24.874291</td>
          <td>0.208567</td>
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
