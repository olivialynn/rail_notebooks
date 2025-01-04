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

    <pzflow.flow.Flow at 0x7fdfe1d27eb0>



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
          <td>26.996810</td>
          <td>0.547443</td>
          <td>26.501036</td>
          <td>0.138615</td>
          <td>26.089152</td>
          <td>0.085296</td>
          <td>25.392317</td>
          <td>0.075165</td>
          <td>25.036351</td>
          <td>0.104724</td>
          <td>25.221745</td>
          <td>0.267589</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.042429</td>
          <td>1.812921</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.728756</td>
          <td>0.340841</td>
          <td>27.291356</td>
          <td>0.373369</td>
          <td>26.673687</td>
          <td>0.408152</td>
          <td>25.791477</td>
          <td>0.419887</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.892709</td>
          <td>0.231466</td>
          <td>26.052472</td>
          <td>0.093826</td>
          <td>24.759632</td>
          <td>0.026334</td>
          <td>23.900468</td>
          <td>0.020239</td>
          <td>23.146388</td>
          <td>0.019890</td>
          <td>22.827456</td>
          <td>0.033484</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>30.235672</td>
          <td>2.860396</td>
          <td>27.721647</td>
          <td>0.379403</td>
          <td>27.321659</td>
          <td>0.245361</td>
          <td>26.596057</td>
          <td>0.212734</td>
          <td>25.633309</td>
          <td>0.175289</td>
          <td>25.754750</td>
          <td>0.408249</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.861047</td>
          <td>0.225477</td>
          <td>25.842379</td>
          <td>0.077996</td>
          <td>25.371275</td>
          <td>0.045155</td>
          <td>24.824352</td>
          <td>0.045420</td>
          <td>24.406885</td>
          <td>0.060109</td>
          <td>23.785613</td>
          <td>0.078254</td>
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
          <td>26.693384</td>
          <td>0.437319</td>
          <td>26.257103</td>
          <td>0.112204</td>
          <td>26.203411</td>
          <td>0.094313</td>
          <td>26.099871</td>
          <td>0.139558</td>
          <td>25.898767</td>
          <td>0.219148</td>
          <td>26.218436</td>
          <td>0.575818</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.330043</td>
          <td>0.329945</td>
          <td>26.743845</td>
          <td>0.170644</td>
          <td>26.526004</td>
          <td>0.124994</td>
          <td>26.115503</td>
          <td>0.141451</td>
          <td>26.008224</td>
          <td>0.239969</td>
          <td>25.154341</td>
          <td>0.253233</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.697528</td>
          <td>0.880262</td>
          <td>26.823593</td>
          <td>0.182583</td>
          <td>26.868358</td>
          <td>0.167785</td>
          <td>26.430561</td>
          <td>0.185109</td>
          <td>25.999675</td>
          <td>0.238281</td>
          <td>25.138510</td>
          <td>0.249962</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.840554</td>
          <td>0.488286</td>
          <td>27.341705</td>
          <td>0.280575</td>
          <td>26.765194</td>
          <td>0.153628</td>
          <td>26.011376</td>
          <td>0.129284</td>
          <td>25.470146</td>
          <td>0.152508</td>
          <td>25.161259</td>
          <td>0.254674</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.291645</td>
          <td>0.673809</td>
          <td>26.593044</td>
          <td>0.150025</td>
          <td>26.146723</td>
          <td>0.089730</td>
          <td>25.786642</td>
          <td>0.106322</td>
          <td>24.987125</td>
          <td>0.100308</td>
          <td>24.930797</td>
          <td>0.210416</td>
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
          <td>27.214179</td>
          <td>0.700037</td>
          <td>26.543782</td>
          <td>0.165313</td>
          <td>26.039503</td>
          <td>0.095996</td>
          <td>25.249674</td>
          <td>0.078513</td>
          <td>24.923420</td>
          <td>0.111413</td>
          <td>24.998613</td>
          <td>0.260430</td>
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
          <td>27.834778</td>
          <td>0.426841</td>
          <td>27.365361</td>
          <td>0.457591</td>
          <td>25.878626</td>
          <td>0.250932</td>
          <td>25.388944</td>
          <td>0.356214</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.218863</td>
          <td>0.341045</td>
          <td>25.824278</td>
          <td>0.090421</td>
          <td>24.807183</td>
          <td>0.033018</td>
          <td>23.884416</td>
          <td>0.024101</td>
          <td>23.183199</td>
          <td>0.024592</td>
          <td>22.825336</td>
          <td>0.040492</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.692481</td>
          <td>1.552465</td>
          <td>27.593061</td>
          <td>0.375765</td>
          <td>26.153859</td>
          <td>0.184156</td>
          <td>26.207985</td>
          <td>0.347788</td>
          <td>25.099494</td>
          <td>0.301173</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.080528</td>
          <td>0.300881</td>
          <td>25.935623</td>
          <td>0.097729</td>
          <td>25.512625</td>
          <td>0.060312</td>
          <td>24.841581</td>
          <td>0.054721</td>
          <td>24.492497</td>
          <td>0.076341</td>
          <td>23.806286</td>
          <td>0.094255</td>
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
          <td>26.730832</td>
          <td>0.503677</td>
          <td>26.389387</td>
          <td>0.147569</td>
          <td>26.041745</td>
          <td>0.098224</td>
          <td>26.122340</td>
          <td>0.171243</td>
          <td>26.182062</td>
          <td>0.326881</td>
          <td>25.190147</td>
          <td>0.310200</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.167611</td>
          <td>0.679850</td>
          <td>26.913254</td>
          <td>0.226395</td>
          <td>27.253911</td>
          <td>0.270872</td>
          <td>26.690015</td>
          <td>0.270358</td>
          <td>26.432940</td>
          <td>0.391996</td>
          <td>26.008569</td>
          <td>0.569609</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.444600</td>
          <td>0.821413</td>
          <td>27.994622</td>
          <td>0.531874</td>
          <td>26.662341</td>
          <td>0.166703</td>
          <td>26.331346</td>
          <td>0.202648</td>
          <td>26.073548</td>
          <td>0.297478</td>
          <td>25.023520</td>
          <td>0.269043</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.293419</td>
          <td>0.315708</td>
          <td>26.817167</td>
          <td>0.193513</td>
          <td>26.054864</td>
          <td>0.163375</td>
          <td>25.910431</td>
          <td>0.265145</td>
          <td>25.059540</td>
          <td>0.281977</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.514363</td>
          <td>0.162643</td>
          <td>25.894151</td>
          <td>0.085337</td>
          <td>25.325401</td>
          <td>0.084815</td>
          <td>25.497880</td>
          <td>0.184378</td>
          <td>25.063681</td>
          <td>0.277253</td>
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
          <td>27.787921</td>
          <td>0.931432</td>
          <td>26.880156</td>
          <td>0.191534</td>
          <td>26.058922</td>
          <td>0.083065</td>
          <td>25.376523</td>
          <td>0.074133</td>
          <td>25.225793</td>
          <td>0.123534</td>
          <td>25.051234</td>
          <td>0.232627</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.803180</td>
          <td>0.404361</td>
          <td>27.851940</td>
          <td>0.375721</td>
          <td>27.499997</td>
          <td>0.438665</td>
          <td>26.554624</td>
          <td>0.372561</td>
          <td>26.812528</td>
          <td>0.861039</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.280126</td>
          <td>0.699592</td>
          <td>25.908307</td>
          <td>0.088848</td>
          <td>24.814705</td>
          <td>0.029994</td>
          <td>23.872807</td>
          <td>0.021493</td>
          <td>23.196353</td>
          <td>0.022486</td>
          <td>22.867114</td>
          <td>0.037777</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.095804</td>
          <td>0.670752</td>
          <td>28.771630</td>
          <td>0.930679</td>
          <td>28.260456</td>
          <td>0.614749</td>
          <td>26.432862</td>
          <td>0.231794</td>
          <td>26.443283</td>
          <td>0.416167</td>
          <td>25.133776</td>
          <td>0.308539</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.695693</td>
          <td>0.438455</td>
          <td>25.804937</td>
          <td>0.075556</td>
          <td>25.453476</td>
          <td>0.048643</td>
          <td>24.793867</td>
          <td>0.044275</td>
          <td>24.415210</td>
          <td>0.060641</td>
          <td>23.753684</td>
          <td>0.076191</td>
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
          <td>26.459662</td>
          <td>0.383886</td>
          <td>26.428671</td>
          <td>0.139358</td>
          <td>26.117361</td>
          <td>0.094614</td>
          <td>25.977925</td>
          <td>0.136204</td>
          <td>25.599444</td>
          <td>0.183749</td>
          <td>26.042926</td>
          <td>0.542529</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.446045</td>
          <td>0.309213</td>
          <td>26.667439</td>
          <td>0.143550</td>
          <td>26.272749</td>
          <td>0.164618</td>
          <td>26.070583</td>
          <td>0.256517</td>
          <td>25.264441</td>
          <td>0.281461</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.392825</td>
          <td>0.740917</td>
          <td>27.220734</td>
          <td>0.264536</td>
          <td>26.872903</td>
          <td>0.176598</td>
          <td>26.544464</td>
          <td>0.213928</td>
          <td>26.069076</td>
          <td>0.264046</td>
          <td>25.697315</td>
          <td>0.408334</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.951161</td>
          <td>0.566622</td>
          <td>27.162081</td>
          <td>0.266166</td>
          <td>26.853604</td>
          <td>0.185100</td>
          <td>25.856182</td>
          <td>0.127193</td>
          <td>25.686951</td>
          <td>0.204724</td>
          <td>25.418949</td>
          <td>0.349246</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.675811</td>
          <td>0.885429</td>
          <td>26.720482</td>
          <td>0.172869</td>
          <td>26.165505</td>
          <td>0.094847</td>
          <td>25.621331</td>
          <td>0.095811</td>
          <td>25.099041</td>
          <td>0.114970</td>
          <td>25.024775</td>
          <td>0.236379</td>
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
