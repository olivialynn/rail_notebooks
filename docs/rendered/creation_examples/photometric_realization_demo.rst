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

    <pzflow.flow.Flow at 0x7f195e25ac50>



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
          <td>27.449961</td>
          <td>0.749832</td>
          <td>26.937504</td>
          <td>0.200974</td>
          <td>26.155130</td>
          <td>0.090396</td>
          <td>25.201683</td>
          <td>0.063491</td>
          <td>25.125469</td>
          <td>0.113197</td>
          <td>24.678424</td>
          <td>0.170060</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>30.525978</td>
          <td>3.132256</td>
          <td>27.468236</td>
          <td>0.310676</td>
          <td>27.792241</td>
          <td>0.358303</td>
          <td>27.715452</td>
          <td>0.514660</td>
          <td>26.429553</td>
          <td>0.337428</td>
          <td>25.804925</td>
          <td>0.424217</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.671717</td>
          <td>0.430193</td>
          <td>25.962155</td>
          <td>0.086671</td>
          <td>24.788093</td>
          <td>0.026996</td>
          <td>23.838219</td>
          <td>0.019201</td>
          <td>23.120655</td>
          <td>0.019462</td>
          <td>22.840494</td>
          <td>0.033871</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.859950</td>
          <td>0.495343</td>
          <td>27.468557</td>
          <td>0.310756</td>
          <td>27.457428</td>
          <td>0.274197</td>
          <td>26.360608</td>
          <td>0.174455</td>
          <td>26.453253</td>
          <td>0.343805</td>
          <td>25.051679</td>
          <td>0.232683</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.849675</td>
          <td>0.967348</td>
          <td>25.767991</td>
          <td>0.073041</td>
          <td>25.445542</td>
          <td>0.048233</td>
          <td>24.831486</td>
          <td>0.045709</td>
          <td>24.427094</td>
          <td>0.061196</td>
          <td>23.600294</td>
          <td>0.066423</td>
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
          <td>26.578952</td>
          <td>0.400752</td>
          <td>26.233429</td>
          <td>0.109913</td>
          <td>26.069077</td>
          <td>0.083800</td>
          <td>26.208985</td>
          <td>0.153282</td>
          <td>26.221135</td>
          <td>0.285589</td>
          <td>25.247832</td>
          <td>0.273336</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.883776</td>
          <td>0.987579</td>
          <td>27.185596</td>
          <td>0.246992</td>
          <td>26.787539</td>
          <td>0.156597</td>
          <td>26.134838</td>
          <td>0.143825</td>
          <td>26.449757</td>
          <td>0.342858</td>
          <td>25.700055</td>
          <td>0.391411</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.466319</td>
          <td>0.758015</td>
          <td>26.877830</td>
          <td>0.191138</td>
          <td>27.001904</td>
          <td>0.187907</td>
          <td>26.552747</td>
          <td>0.205163</td>
          <td>26.468020</td>
          <td>0.347830</td>
          <td>25.884666</td>
          <td>0.450640</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.106236</td>
          <td>0.592061</td>
          <td>27.513541</td>
          <td>0.322115</td>
          <td>26.586636</td>
          <td>0.131734</td>
          <td>25.563919</td>
          <td>0.087450</td>
          <td>26.097928</td>
          <td>0.258336</td>
          <td>25.537072</td>
          <td>0.344638</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.281455</td>
          <td>0.669113</td>
          <td>26.377937</td>
          <td>0.124625</td>
          <td>25.990654</td>
          <td>0.078199</td>
          <td>25.731846</td>
          <td>0.101345</td>
          <td>25.240319</td>
          <td>0.125084</td>
          <td>24.709816</td>
          <td>0.174659</td>
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
          <td>26.953381</td>
          <td>0.583967</td>
          <td>26.720057</td>
          <td>0.191937</td>
          <td>26.142307</td>
          <td>0.105039</td>
          <td>25.223287</td>
          <td>0.076705</td>
          <td>25.027308</td>
          <td>0.121953</td>
          <td>24.839102</td>
          <td>0.228361</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.034270</td>
          <td>0.542301</td>
          <td>27.402233</td>
          <td>0.304269</td>
          <td>27.226783</td>
          <td>0.411925</td>
          <td>26.271409</td>
          <td>0.344336</td>
          <td>25.796660</td>
          <td>0.486389</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.707655</td>
          <td>0.495572</td>
          <td>25.886659</td>
          <td>0.095504</td>
          <td>24.806435</td>
          <td>0.032996</td>
          <td>23.839814</td>
          <td>0.023192</td>
          <td>23.147615</td>
          <td>0.023848</td>
          <td>22.711930</td>
          <td>0.036629</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.027930</td>
          <td>0.641442</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.183002</td>
          <td>0.271033</td>
          <td>26.959272</td>
          <td>0.355800</td>
          <td>25.695528</td>
          <td>0.229728</td>
          <td>25.361896</td>
          <td>0.370731</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.421027</td>
          <td>0.393387</td>
          <td>25.943606</td>
          <td>0.098414</td>
          <td>25.360781</td>
          <td>0.052712</td>
          <td>24.780918</td>
          <td>0.051854</td>
          <td>24.264569</td>
          <td>0.062400</td>
          <td>23.645140</td>
          <td>0.081799</td>
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
          <td>26.463127</td>
          <td>0.411963</td>
          <td>26.307433</td>
          <td>0.137527</td>
          <td>26.153092</td>
          <td>0.108272</td>
          <td>26.155201</td>
          <td>0.176090</td>
          <td>26.075863</td>
          <td>0.300278</td>
          <td>25.823311</td>
          <td>0.505051</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.262548</td>
          <td>0.725026</td>
          <td>27.363404</td>
          <td>0.326433</td>
          <td>26.928801</td>
          <td>0.207069</td>
          <td>26.463026</td>
          <td>0.224293</td>
          <td>25.603851</td>
          <td>0.200469</td>
          <td>25.766393</td>
          <td>0.477209</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.686312</td>
          <td>0.188632</td>
          <td>26.536773</td>
          <td>0.149730</td>
          <td>26.017500</td>
          <td>0.155302</td>
          <td>26.523411</td>
          <td>0.423349</td>
          <td>25.091524</td>
          <td>0.284321</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.280661</td>
          <td>0.312508</td>
          <td>26.273500</td>
          <td>0.121489</td>
          <td>25.620109</td>
          <td>0.112263</td>
          <td>25.517185</td>
          <td>0.191294</td>
          <td>25.045076</td>
          <td>0.278689</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.582237</td>
          <td>1.562297</td>
          <td>26.648900</td>
          <td>0.182330</td>
          <td>26.276262</td>
          <td>0.119232</td>
          <td>25.525990</td>
          <td>0.101152</td>
          <td>25.174568</td>
          <td>0.139896</td>
          <td>24.974115</td>
          <td>0.257724</td>
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
          <td>29.460519</td>
          <td>2.163668</td>
          <td>26.638093</td>
          <td>0.155944</td>
          <td>25.945477</td>
          <td>0.075149</td>
          <td>25.313457</td>
          <td>0.070110</td>
          <td>25.065312</td>
          <td>0.107422</td>
          <td>24.988415</td>
          <td>0.220806</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.483926</td>
          <td>0.767270</td>
          <td>27.888717</td>
          <td>0.431666</td>
          <td>28.251981</td>
          <td>0.508653</td>
          <td>27.127663</td>
          <td>0.328550</td>
          <td>26.062067</td>
          <td>0.251070</td>
          <td>26.182963</td>
          <td>0.561813</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.304349</td>
          <td>0.340805</td>
          <td>25.956297</td>
          <td>0.092671</td>
          <td>24.755770</td>
          <td>0.028484</td>
          <td>23.850842</td>
          <td>0.021094</td>
          <td>23.154693</td>
          <td>0.021698</td>
          <td>22.864796</td>
          <td>0.037699</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.277776</td>
          <td>0.758340</td>
          <td>28.807850</td>
          <td>0.951670</td>
          <td>27.335311</td>
          <td>0.305531</td>
          <td>26.289990</td>
          <td>0.205785</td>
          <td>26.348478</td>
          <td>0.386889</td>
          <td>25.458853</td>
          <td>0.398377</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.229507</td>
          <td>0.304810</td>
          <td>25.735195</td>
          <td>0.071044</td>
          <td>25.355608</td>
          <td>0.044596</td>
          <td>24.777407</td>
          <td>0.043633</td>
          <td>24.319343</td>
          <td>0.055696</td>
          <td>23.587466</td>
          <td>0.065771</td>
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
          <td>25.943661</td>
          <td>0.254356</td>
          <td>26.387066</td>
          <td>0.134447</td>
          <td>26.350264</td>
          <td>0.115982</td>
          <td>26.073168</td>
          <td>0.147847</td>
          <td>26.111937</td>
          <td>0.281057</td>
          <td>25.978645</td>
          <td>0.517711</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.676808</td>
          <td>0.436157</td>
          <td>26.785957</td>
          <td>0.179316</td>
          <td>26.876239</td>
          <td>0.171631</td>
          <td>26.452549</td>
          <td>0.191735</td>
          <td>25.865100</td>
          <td>0.216435</td>
          <td>25.114068</td>
          <td>0.248942</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.639659</td>
          <td>0.432532</td>
          <td>27.580485</td>
          <td>0.352936</td>
          <td>26.966422</td>
          <td>0.191135</td>
          <td>26.739340</td>
          <td>0.251399</td>
          <td>26.067173</td>
          <td>0.263636</td>
          <td>25.532147</td>
          <td>0.359240</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.415060</td>
          <td>0.779411</td>
          <td>27.297917</td>
          <td>0.297133</td>
          <td>26.937524</td>
          <td>0.198666</td>
          <td>25.974010</td>
          <td>0.140825</td>
          <td>25.400307</td>
          <td>0.160618</td>
          <td>25.161754</td>
          <td>0.284406</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.531336</td>
          <td>0.395669</td>
          <td>26.535793</td>
          <td>0.147646</td>
          <td>26.157282</td>
          <td>0.094164</td>
          <td>25.745666</td>
          <td>0.106832</td>
          <td>25.199009</td>
          <td>0.125404</td>
          <td>25.099896</td>
          <td>0.251472</td>
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
