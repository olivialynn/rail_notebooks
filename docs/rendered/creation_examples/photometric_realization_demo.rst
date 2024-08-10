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

    <pzflow.flow.Flow at 0x7fb65236cf40>



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
          <td>26.454735</td>
          <td>0.363965</td>
          <td>26.672052</td>
          <td>0.160519</td>
          <td>25.949984</td>
          <td>0.075439</td>
          <td>25.350811</td>
          <td>0.072456</td>
          <td>24.884138</td>
          <td>0.091641</td>
          <td>25.419107</td>
          <td>0.313823</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.130567</td>
          <td>1.007930</td>
          <td>27.747468</td>
          <td>0.345911</td>
          <td>26.752044</td>
          <td>0.242142</td>
          <td>25.770486</td>
          <td>0.196835</td>
          <td>26.706923</td>
          <td>0.803918</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.273990</td>
          <td>0.315562</td>
          <td>25.986899</td>
          <td>0.088577</td>
          <td>24.759059</td>
          <td>0.026321</td>
          <td>23.886687</td>
          <td>0.020004</td>
          <td>23.161714</td>
          <td>0.020150</td>
          <td>22.880889</td>
          <td>0.035100</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.080555</td>
          <td>0.977966</td>
          <td>27.821702</td>
          <td>0.366659</td>
          <td>26.493943</td>
          <td>0.195276</td>
          <td>26.055940</td>
          <td>0.249588</td>
          <td>25.310083</td>
          <td>0.287489</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.446869</td>
          <td>0.361734</td>
          <td>25.774844</td>
          <td>0.073485</td>
          <td>25.474248</td>
          <td>0.049478</td>
          <td>24.891156</td>
          <td>0.048195</td>
          <td>24.357880</td>
          <td>0.057551</td>
          <td>23.775887</td>
          <td>0.077584</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.385368</td>
          <td>0.125430</td>
          <td>26.161085</td>
          <td>0.090870</td>
          <td>26.108296</td>
          <td>0.140575</td>
          <td>25.793541</td>
          <td>0.200686</td>
          <td>25.542345</td>
          <td>0.346073</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.886297</td>
          <td>0.505058</td>
          <td>27.467591</td>
          <td>0.310516</td>
          <td>26.766020</td>
          <td>0.153737</td>
          <td>26.323574</td>
          <td>0.169047</td>
          <td>25.870448</td>
          <td>0.214033</td>
          <td>25.340916</td>
          <td>0.294734</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.154990</td>
          <td>1.157506</td>
          <td>26.941712</td>
          <td>0.201685</td>
          <td>26.635524</td>
          <td>0.137417</td>
          <td>26.440169</td>
          <td>0.186618</td>
          <td>25.825469</td>
          <td>0.206132</td>
          <td>26.239290</td>
          <td>0.584447</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.744833</td>
          <td>1.577990</td>
          <td>27.293949</td>
          <td>0.269897</td>
          <td>26.541739</td>
          <td>0.126711</td>
          <td>25.969916</td>
          <td>0.124721</td>
          <td>25.711303</td>
          <td>0.187257</td>
          <td>25.955362</td>
          <td>0.475164</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.627580</td>
          <td>1.489323</td>
          <td>26.573790</td>
          <td>0.147567</td>
          <td>26.212410</td>
          <td>0.095061</td>
          <td>25.554638</td>
          <td>0.086739</td>
          <td>25.220280</td>
          <td>0.122928</td>
          <td>25.072926</td>
          <td>0.236809</td>
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
          <td>26.997741</td>
          <td>0.547812</td>
          <td>27.221063</td>
          <td>0.254291</td>
          <td>25.961381</td>
          <td>0.076203</td>
          <td>25.299862</td>
          <td>0.069262</td>
          <td>24.878947</td>
          <td>0.091224</td>
          <td>25.021718</td>
          <td>0.226974</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>32.281217</td>
          <td>3.650671</td>
          <td>27.385761</td>
          <td>0.258622</td>
          <td>27.015580</td>
          <td>0.300128</td>
          <td>26.301854</td>
          <td>0.304782</td>
          <td>25.284022</td>
          <td>0.281487</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.489093</td>
          <td>0.373845</td>
          <td>25.919212</td>
          <td>0.083458</td>
          <td>24.765270</td>
          <td>0.026464</td>
          <td>23.863821</td>
          <td>0.019620</td>
          <td>23.138011</td>
          <td>0.019750</td>
          <td>22.839192</td>
          <td>0.033832</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.515829</td>
          <td>0.678944</td>
          <td>27.144800</td>
          <td>0.211874</td>
          <td>26.534950</td>
          <td>0.202123</td>
          <td>26.695045</td>
          <td>0.414886</td>
          <td>25.675152</td>
          <td>0.383938</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.918729</td>
          <td>0.236495</td>
          <td>25.669148</td>
          <td>0.066933</td>
          <td>25.414769</td>
          <td>0.046933</td>
          <td>24.801737</td>
          <td>0.044518</td>
          <td>24.459137</td>
          <td>0.062959</td>
          <td>23.758083</td>
          <td>0.076374</td>
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
          <td>26.362900</td>
          <td>0.338635</td>
          <td>26.308195</td>
          <td>0.117305</td>
          <td>26.110571</td>
          <td>0.086920</td>
          <td>25.962085</td>
          <td>0.123877</td>
          <td>26.070630</td>
          <td>0.252618</td>
          <td>25.785126</td>
          <td>0.417856</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.806714</td>
          <td>0.476166</td>
          <td>27.090560</td>
          <td>0.228346</td>
          <td>26.547362</td>
          <td>0.127330</td>
          <td>26.434007</td>
          <td>0.185649</td>
          <td>25.961313</td>
          <td>0.230837</td>
          <td>25.220858</td>
          <td>0.267395</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.153959</td>
          <td>0.612358</td>
          <td>27.174660</td>
          <td>0.244779</td>
          <td>27.051371</td>
          <td>0.195908</td>
          <td>26.604616</td>
          <td>0.214260</td>
          <td>26.064463</td>
          <td>0.251342</td>
          <td>25.649612</td>
          <td>0.376398</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.866212</td>
          <td>0.424038</td>
          <td>26.562196</td>
          <td>0.128977</td>
          <td>25.890556</td>
          <td>0.116409</td>
          <td>25.554301</td>
          <td>0.163889</td>
          <td>25.274206</td>
          <td>0.279256</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.734307</td>
          <td>0.451040</td>
          <td>26.370667</td>
          <td>0.123842</td>
          <td>26.167930</td>
          <td>0.091419</td>
          <td>25.609859</td>
          <td>0.091057</td>
          <td>25.117545</td>
          <td>0.112418</td>
          <td>24.889605</td>
          <td>0.203282</td>
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
          <td>28.982982</td>
          <td>1.764926</td>
          <td>26.954133</td>
          <td>0.203796</td>
          <td>26.044923</td>
          <td>0.082035</td>
          <td>25.317045</td>
          <td>0.070323</td>
          <td>24.945156</td>
          <td>0.096684</td>
          <td>25.121223</td>
          <td>0.246433</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>29.439824</td>
          <td>2.145731</td>
          <td>28.923180</td>
          <td>0.887322</td>
          <td>27.916547</td>
          <td>0.394682</td>
          <td>27.374389</td>
          <td>0.398179</td>
          <td>26.594443</td>
          <td>0.383951</td>
          <td>26.166926</td>
          <td>0.554918</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.067032</td>
          <td>0.575769</td>
          <td>26.218438</td>
          <td>0.108486</td>
          <td>24.789677</td>
          <td>0.027033</td>
          <td>23.832121</td>
          <td>0.019103</td>
          <td>23.093215</td>
          <td>0.019017</td>
          <td>22.790200</td>
          <td>0.032403</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.503391</td>
          <td>0.776790</td>
          <td>30.535874</td>
          <td>2.045989</td>
          <td>26.767666</td>
          <td>0.153954</td>
          <td>26.901566</td>
          <td>0.273691</td>
          <td>25.937801</td>
          <td>0.226378</td>
          <td>25.458048</td>
          <td>0.323722</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.902830</td>
          <td>0.233411</td>
          <td>25.715852</td>
          <td>0.069754</td>
          <td>25.397171</td>
          <td>0.046205</td>
          <td>24.689266</td>
          <td>0.040291</td>
          <td>24.345865</td>
          <td>0.056941</td>
          <td>23.778632</td>
          <td>0.077773</td>
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
          <td>26.087166</td>
          <td>0.271482</td>
          <td>26.289348</td>
          <td>0.115398</td>
          <td>26.080721</td>
          <td>0.084665</td>
          <td>26.018678</td>
          <td>0.130104</td>
          <td>25.673058</td>
          <td>0.181297</td>
          <td>25.476360</td>
          <td>0.328470</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.766424</td>
          <td>0.462050</td>
          <td>27.213641</td>
          <td>0.252748</td>
          <td>26.576143</td>
          <td>0.130543</td>
          <td>26.454769</td>
          <td>0.188933</td>
          <td>25.924552</td>
          <td>0.223900</td>
          <td>25.115022</td>
          <td>0.245178</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.308013</td>
          <td>0.681403</td>
          <td>27.213282</td>
          <td>0.252674</td>
          <td>26.995602</td>
          <td>0.186910</td>
          <td>26.737519</td>
          <td>0.239257</td>
          <td>25.862035</td>
          <td>0.212535</td>
          <td>25.863587</td>
          <td>0.443529</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.429372</td>
          <td>0.739619</td>
          <td>27.011090</td>
          <td>0.213739</td>
          <td>26.471846</td>
          <td>0.119251</td>
          <td>25.902604</td>
          <td>0.117636</td>
          <td>25.511216</td>
          <td>0.157967</td>
          <td>25.774849</td>
          <td>0.414585</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.065219</td>
          <td>1.099513</td>
          <td>26.506343</td>
          <td>0.139250</td>
          <td>26.150025</td>
          <td>0.089991</td>
          <td>25.487785</td>
          <td>0.081776</td>
          <td>25.536396</td>
          <td>0.161404</td>
          <td>24.660779</td>
          <td>0.167524</td>
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
