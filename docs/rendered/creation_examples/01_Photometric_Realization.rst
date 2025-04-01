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

    <pzflow.flow.Flow at 0x7f14a8075b10>



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
          <td>27.099824</td>
          <td>0.589372</td>
          <td>27.168498</td>
          <td>0.243540</td>
          <td>25.906212</td>
          <td>0.072575</td>
          <td>25.215361</td>
          <td>0.064266</td>
          <td>24.776306</td>
          <td>0.083343</td>
          <td>24.023654</td>
          <td>0.096495</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.154340</td>
          <td>0.286667</td>
          <td>27.255567</td>
          <td>0.261576</td>
          <td>26.533222</td>
          <td>0.125779</td>
          <td>26.252277</td>
          <td>0.159070</td>
          <td>26.734404</td>
          <td>0.427535</td>
          <td>24.906264</td>
          <td>0.206140</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.174007</td>
          <td>1.170008</td>
          <td>27.912733</td>
          <td>0.439287</td>
          <td>28.682804</td>
          <td>0.689800</td>
          <td>26.025406</td>
          <td>0.130864</td>
          <td>25.059984</td>
          <td>0.106910</td>
          <td>24.338537</td>
          <td>0.126994</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.242626</td>
          <td>1.215737</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.726530</td>
          <td>0.340242</td>
          <td>26.414266</td>
          <td>0.182575</td>
          <td>25.470994</td>
          <td>0.152619</td>
          <td>24.830050</td>
          <td>0.193356</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.630830</td>
          <td>0.417005</td>
          <td>26.076117</td>
          <td>0.095791</td>
          <td>25.926681</td>
          <td>0.073901</td>
          <td>25.697550</td>
          <td>0.098345</td>
          <td>25.353519</td>
          <td>0.137953</td>
          <td>26.053662</td>
          <td>0.511017</td>
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
          <td>27.928181</td>
          <td>1.014309</td>
          <td>26.337266</td>
          <td>0.120305</td>
          <td>25.406796</td>
          <td>0.046602</td>
          <td>25.099600</td>
          <td>0.057994</td>
          <td>24.952030</td>
          <td>0.097269</td>
          <td>24.663683</td>
          <td>0.167939</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.835738</td>
          <td>0.486547</td>
          <td>26.888244</td>
          <td>0.192822</td>
          <td>25.914504</td>
          <td>0.073109</td>
          <td>25.143017</td>
          <td>0.060272</td>
          <td>25.082498</td>
          <td>0.109033</td>
          <td>24.300299</td>
          <td>0.122851</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.615472</td>
          <td>0.412138</td>
          <td>26.659589</td>
          <td>0.158819</td>
          <td>26.402246</td>
          <td>0.112239</td>
          <td>26.341781</td>
          <td>0.171686</td>
          <td>25.978759</td>
          <td>0.234196</td>
          <td>25.596503</td>
          <td>0.361115</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.338506</td>
          <td>0.332165</td>
          <td>26.116360</td>
          <td>0.099228</td>
          <td>26.169388</td>
          <td>0.091536</td>
          <td>25.816074</td>
          <td>0.109091</td>
          <td>25.664260</td>
          <td>0.179951</td>
          <td>25.056579</td>
          <td>0.233629</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.478924</td>
          <td>0.764363</td>
          <td>26.950564</td>
          <td>0.203188</td>
          <td>26.442123</td>
          <td>0.116207</td>
          <td>26.224031</td>
          <td>0.155271</td>
          <td>25.889138</td>
          <td>0.217396</td>
          <td>25.452155</td>
          <td>0.322207</td>
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
          <td>28.220734</td>
          <td>1.292068</td>
          <td>26.770426</td>
          <td>0.200239</td>
          <td>26.069238</td>
          <td>0.098532</td>
          <td>25.251500</td>
          <td>0.078639</td>
          <td>24.741211</td>
          <td>0.094998</td>
          <td>24.011803</td>
          <td>0.112779</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.439111</td>
          <td>0.345472</td>
          <td>26.625649</td>
          <td>0.159602</td>
          <td>26.607320</td>
          <td>0.251715</td>
          <td>25.709714</td>
          <td>0.218204</td>
          <td>25.380885</td>
          <td>0.353968</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.904931</td>
          <td>0.501646</td>
          <td>27.421995</td>
          <td>0.315405</td>
          <td>25.779856</td>
          <td>0.127783</td>
          <td>25.043578</td>
          <td>0.126439</td>
          <td>24.459208</td>
          <td>0.169609</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.524036</td>
          <td>1.556207</td>
          <td>30.497569</td>
          <td>2.212117</td>
          <td>27.084976</td>
          <td>0.250151</td>
          <td>25.997735</td>
          <td>0.161275</td>
          <td>25.841612</td>
          <td>0.259104</td>
          <td>26.011639</td>
          <td>0.601492</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.321210</td>
          <td>0.364061</td>
          <td>26.260257</td>
          <td>0.129635</td>
          <td>26.018525</td>
          <td>0.094276</td>
          <td>25.585155</td>
          <td>0.105463</td>
          <td>25.100022</td>
          <td>0.129928</td>
          <td>24.588390</td>
          <td>0.185168</td>
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
          <td>26.499540</td>
          <td>0.423574</td>
          <td>26.417178</td>
          <td>0.151128</td>
          <td>25.381658</td>
          <td>0.054834</td>
          <td>24.984037</td>
          <td>0.063448</td>
          <td>24.799166</td>
          <td>0.102063</td>
          <td>24.752351</td>
          <td>0.216849</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>32.689253</td>
          <td>5.377738</td>
          <td>26.540522</td>
          <td>0.165457</td>
          <td>26.060287</td>
          <td>0.098170</td>
          <td>25.231015</td>
          <td>0.077567</td>
          <td>24.904107</td>
          <td>0.110007</td>
          <td>24.352156</td>
          <td>0.152012</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.164133</td>
          <td>0.324440</td>
          <td>26.652804</td>
          <td>0.183371</td>
          <td>26.601976</td>
          <td>0.158332</td>
          <td>26.631833</td>
          <td>0.259961</td>
          <td>25.834426</td>
          <td>0.244837</td>
          <td>24.893218</td>
          <td>0.241785</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.234468</td>
          <td>0.347430</td>
          <td>26.355459</td>
          <td>0.144662</td>
          <td>26.240619</td>
          <td>0.118067</td>
          <td>25.709086</td>
          <td>0.121299</td>
          <td>25.698980</td>
          <td>0.222746</td>
          <td>25.583342</td>
          <td>0.425828</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.213890</td>
          <td>0.336845</td>
          <td>26.806007</td>
          <td>0.208087</td>
          <td>26.321636</td>
          <td>0.124024</td>
          <td>26.195030</td>
          <td>0.180157</td>
          <td>26.178884</td>
          <td>0.322841</td>
          <td>25.119329</td>
          <td>0.290031</td>
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
          <td>26.619369</td>
          <td>0.413401</td>
          <td>26.720506</td>
          <td>0.167307</td>
          <td>25.956130</td>
          <td>0.075860</td>
          <td>25.172352</td>
          <td>0.061870</td>
          <td>24.648954</td>
          <td>0.074491</td>
          <td>24.035888</td>
          <td>0.097549</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.522544</td>
          <td>0.324672</td>
          <td>26.720227</td>
          <td>0.147950</td>
          <td>26.302575</td>
          <td>0.166209</td>
          <td>25.639593</td>
          <td>0.176387</td>
          <td>25.607386</td>
          <td>0.364522</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.765734</td>
          <td>0.418045</td>
          <td>28.220648</td>
          <td>0.531732</td>
          <td>25.984366</td>
          <td>0.137391</td>
          <td>25.079561</td>
          <td>0.117946</td>
          <td>24.176629</td>
          <td>0.119986</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.500332</td>
          <td>0.875810</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.805888</td>
          <td>0.441056</td>
          <td>26.235053</td>
          <td>0.196512</td>
          <td>25.573851</td>
          <td>0.206880</td>
          <td>25.340447</td>
          <td>0.363390</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.703572</td>
          <td>0.197911</td>
          <td>26.134396</td>
          <td>0.100930</td>
          <td>26.019664</td>
          <td>0.080342</td>
          <td>25.771764</td>
          <td>0.105104</td>
          <td>25.594639</td>
          <td>0.169854</td>
          <td>24.800232</td>
          <td>0.188823</td>
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
          <td>26.359169</td>
          <td>0.354968</td>
          <td>26.570878</td>
          <td>0.157445</td>
          <td>25.490621</td>
          <td>0.054370</td>
          <td>25.110170</td>
          <td>0.063627</td>
          <td>24.845811</td>
          <td>0.095860</td>
          <td>24.720809</td>
          <td>0.190700</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.184103</td>
          <td>0.631175</td>
          <td>27.206283</td>
          <td>0.254608</td>
          <td>26.063828</td>
          <td>0.084805</td>
          <td>25.189382</td>
          <td>0.063910</td>
          <td>24.730213</td>
          <td>0.081358</td>
          <td>24.126277</td>
          <td>0.107382</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.761361</td>
          <td>0.938648</td>
          <td>26.626858</td>
          <td>0.160993</td>
          <td>26.592797</td>
          <td>0.138962</td>
          <td>26.221847</td>
          <td>0.162905</td>
          <td>26.157417</td>
          <td>0.283715</td>
          <td>25.413592</td>
          <td>0.327153</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.020754</td>
          <td>0.595428</td>
          <td>26.327450</td>
          <td>0.131775</td>
          <td>26.022787</td>
          <td>0.090261</td>
          <td>26.155706</td>
          <td>0.164562</td>
          <td>25.790798</td>
          <td>0.223263</td>
          <td>24.885178</td>
          <td>0.226690</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.916500</td>
          <td>0.528211</td>
          <td>26.861375</td>
          <td>0.194736</td>
          <td>26.439609</td>
          <td>0.120509</td>
          <td>26.361320</td>
          <td>0.181569</td>
          <td>25.811639</td>
          <td>0.211442</td>
          <td>26.085862</td>
          <td>0.540998</td>
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
