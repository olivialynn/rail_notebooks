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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fdc80f00940>



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
          <td>26.555044</td>
          <td>0.393440</td>
          <td>26.988130</td>
          <td>0.209679</td>
          <td>25.956610</td>
          <td>0.075882</td>
          <td>25.315158</td>
          <td>0.070206</td>
          <td>24.822677</td>
          <td>0.086818</td>
          <td>24.899364</td>
          <td>0.204952</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.150122</td>
          <td>0.524117</td>
          <td>28.273592</td>
          <td>0.516376</td>
          <td>26.798943</td>
          <td>0.251670</td>
          <td>26.136024</td>
          <td>0.266506</td>
          <td>25.820244</td>
          <td>0.429192</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.337073</td>
          <td>0.331788</td>
          <td>25.930119</td>
          <td>0.084263</td>
          <td>24.814062</td>
          <td>0.027615</td>
          <td>23.867957</td>
          <td>0.019689</td>
          <td>23.153212</td>
          <td>0.020005</td>
          <td>22.815015</td>
          <td>0.033119</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.859778</td>
          <td>1.667116</td>
          <td>28.142901</td>
          <td>0.521360</td>
          <td>27.288916</td>
          <td>0.238824</td>
          <td>26.304458</td>
          <td>0.166316</td>
          <td>25.768004</td>
          <td>0.196424</td>
          <td>25.338871</td>
          <td>0.294249</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.447768</td>
          <td>0.361988</td>
          <td>25.853985</td>
          <td>0.078798</td>
          <td>25.506019</td>
          <td>0.050893</td>
          <td>24.782629</td>
          <td>0.043769</td>
          <td>24.369003</td>
          <td>0.058122</td>
          <td>23.732749</td>
          <td>0.074683</td>
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
          <td>27.064918</td>
          <td>0.574900</td>
          <td>26.193914</td>
          <td>0.106189</td>
          <td>26.166682</td>
          <td>0.091319</td>
          <td>26.035108</td>
          <td>0.131967</td>
          <td>26.463910</td>
          <td>0.346706</td>
          <td>25.688595</td>
          <td>0.387957</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.413407</td>
          <td>0.352373</td>
          <td>27.345615</td>
          <td>0.281466</td>
          <td>26.884883</td>
          <td>0.170162</td>
          <td>26.245506</td>
          <td>0.158151</td>
          <td>25.857406</td>
          <td>0.211714</td>
          <td>25.686545</td>
          <td>0.387342</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.016313</td>
          <td>0.214672</td>
          <td>26.939341</td>
          <td>0.178218</td>
          <td>26.960520</td>
          <td>0.287095</td>
          <td>25.731456</td>
          <td>0.190469</td>
          <td>24.802849</td>
          <td>0.188972</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.262374</td>
          <td>0.660384</td>
          <td>27.290449</td>
          <td>0.269129</td>
          <td>26.547539</td>
          <td>0.127349</td>
          <td>25.938028</td>
          <td>0.121316</td>
          <td>25.648162</td>
          <td>0.177512</td>
          <td>27.340644</td>
          <td>1.180452</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.695425</td>
          <td>0.879095</td>
          <td>26.750708</td>
          <td>0.171642</td>
          <td>26.115489</td>
          <td>0.087297</td>
          <td>25.760324</td>
          <td>0.103903</td>
          <td>25.084349</td>
          <td>0.109209</td>
          <td>24.850263</td>
          <td>0.196674</td>
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
          <td>27.690411</td>
          <td>0.952188</td>
          <td>26.877762</td>
          <td>0.219033</td>
          <td>25.964753</td>
          <td>0.089897</td>
          <td>25.233383</td>
          <td>0.077392</td>
          <td>25.266213</td>
          <td>0.149887</td>
          <td>24.579551</td>
          <td>0.183730</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.465530</td>
          <td>0.320069</td>
          <td>28.091846</td>
          <td>0.765114</td>
          <td>28.149430</td>
          <td>1.229554</td>
          <td>27.654674</td>
          <td>1.539234</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.437351</td>
          <td>0.821979</td>
          <td>25.893982</td>
          <td>0.096118</td>
          <td>24.811111</td>
          <td>0.033132</td>
          <td>23.859958</td>
          <td>0.023598</td>
          <td>23.169759</td>
          <td>0.024308</td>
          <td>22.886342</td>
          <td>0.042740</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.979289</td>
          <td>0.252257</td>
          <td>27.354087</td>
          <td>0.311173</td>
          <td>26.764552</td>
          <td>0.304852</td>
          <td>25.741173</td>
          <td>0.238569</td>
          <td>25.045605</td>
          <td>0.288376</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.362540</td>
          <td>0.375973</td>
          <td>25.699381</td>
          <td>0.079418</td>
          <td>25.425708</td>
          <td>0.055838</td>
          <td>24.782398</td>
          <td>0.051922</td>
          <td>24.287465</td>
          <td>0.063679</td>
          <td>23.595906</td>
          <td>0.078323</td>
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
          <td>26.476839</td>
          <td>0.416305</td>
          <td>26.160781</td>
          <td>0.121147</td>
          <td>26.080352</td>
          <td>0.101602</td>
          <td>26.183589</td>
          <td>0.180380</td>
          <td>26.441189</td>
          <td>0.400358</td>
          <td>25.847839</td>
          <td>0.514236</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.487573</td>
          <td>0.840386</td>
          <td>27.302242</td>
          <td>0.310899</td>
          <td>26.768052</td>
          <td>0.180852</td>
          <td>26.621404</td>
          <td>0.255618</td>
          <td>25.531126</td>
          <td>0.188564</td>
          <td>25.658858</td>
          <td>0.440190</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.987836</td>
          <td>1.142697</td>
          <td>26.936907</td>
          <td>0.232576</td>
          <td>27.054593</td>
          <td>0.231823</td>
          <td>26.675978</td>
          <td>0.269501</td>
          <td>25.810083</td>
          <td>0.239973</td>
          <td>25.314754</td>
          <td>0.339914</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.136246</td>
          <td>0.596967</td>
          <td>26.631819</td>
          <td>0.165385</td>
          <td>26.006408</td>
          <td>0.156749</td>
          <td>25.488296</td>
          <td>0.186688</td>
          <td>26.338197</td>
          <td>0.731866</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.112275</td>
          <td>0.656773</td>
          <td>26.282686</td>
          <td>0.133317</td>
          <td>25.962624</td>
          <td>0.090636</td>
          <td>25.308314</td>
          <td>0.083548</td>
          <td>25.329940</td>
          <td>0.159849</td>
          <td>24.795284</td>
          <td>0.222357</td>
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
          <td>26.394164</td>
          <td>0.347112</td>
          <td>26.886845</td>
          <td>0.192616</td>
          <td>26.103089</td>
          <td>0.086361</td>
          <td>25.192981</td>
          <td>0.063012</td>
          <td>25.079865</td>
          <td>0.108796</td>
          <td>25.084499</td>
          <td>0.239115</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.056895</td>
          <td>0.964533</td>
          <td>27.816993</td>
          <td>0.365620</td>
          <td>27.482334</td>
          <td>0.432829</td>
          <td>26.524095</td>
          <td>0.363787</td>
          <td>26.511117</td>
          <td>0.706459</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.990887</td>
          <td>0.095524</td>
          <td>24.802249</td>
          <td>0.029668</td>
          <td>23.870817</td>
          <td>0.021456</td>
          <td>23.125598</td>
          <td>0.021166</td>
          <td>22.887918</td>
          <td>0.038478</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.581219</td>
          <td>0.921307</td>
          <td>27.107971</td>
          <td>0.279329</td>
          <td>27.285715</td>
          <td>0.293586</td>
          <td>26.259028</td>
          <td>0.200511</td>
          <td>26.357676</td>
          <td>0.389653</td>
          <td>25.145609</td>
          <td>0.311475</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.234028</td>
          <td>0.305916</td>
          <td>25.758486</td>
          <td>0.072521</td>
          <td>25.417814</td>
          <td>0.047127</td>
          <td>24.796139</td>
          <td>0.044364</td>
          <td>24.337464</td>
          <td>0.056599</td>
          <td>23.787107</td>
          <td>0.078473</td>
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
          <td>26.283454</td>
          <td>0.334418</td>
          <td>26.463006</td>
          <td>0.143537</td>
          <td>25.997628</td>
          <td>0.085159</td>
          <td>25.863873</td>
          <td>0.123400</td>
          <td>25.495409</td>
          <td>0.168221</td>
          <td>26.457546</td>
          <td>0.724826</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.667041</td>
          <td>0.432940</td>
          <td>26.963572</td>
          <td>0.208236</td>
          <td>26.647778</td>
          <td>0.141140</td>
          <td>26.337471</td>
          <td>0.173942</td>
          <td>26.349827</td>
          <td>0.321474</td>
          <td>25.251501</td>
          <td>0.278522</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.071330</td>
          <td>0.233972</td>
          <td>26.619624</td>
          <td>0.142211</td>
          <td>26.179992</td>
          <td>0.157181</td>
          <td>26.306707</td>
          <td>0.319875</td>
          <td>25.406330</td>
          <td>0.325270</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>29.962645</td>
          <td>2.694707</td>
          <td>26.750150</td>
          <td>0.189082</td>
          <td>26.507929</td>
          <td>0.137766</td>
          <td>25.767780</td>
          <td>0.117797</td>
          <td>25.833966</td>
          <td>0.231406</td>
          <td>25.228015</td>
          <td>0.300023</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.684052</td>
          <td>0.167597</td>
          <td>26.044198</td>
          <td>0.085250</td>
          <td>25.616454</td>
          <td>0.095402</td>
          <td>25.375078</td>
          <td>0.145996</td>
          <td>24.969049</td>
          <td>0.225712</td>
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
