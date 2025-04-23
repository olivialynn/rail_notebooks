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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.17/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f846ca72020>



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
          <td>27.365247</td>
          <td>0.708436</td>
          <td>26.712812</td>
          <td>0.166197</td>
          <td>26.081585</td>
          <td>0.084729</td>
          <td>25.114037</td>
          <td>0.058742</td>
          <td>24.750414</td>
          <td>0.081462</td>
          <td>23.797953</td>
          <td>0.079111</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.520503</td>
          <td>1.410442</td>
          <td>27.798225</td>
          <td>0.402535</td>
          <td>26.658147</td>
          <td>0.140124</td>
          <td>26.165215</td>
          <td>0.147632</td>
          <td>25.876287</td>
          <td>0.215079</td>
          <td>25.290348</td>
          <td>0.282934</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.349932</td>
          <td>1.289152</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.982989</td>
          <td>1.488547</td>
          <td>25.924668</td>
          <td>0.119916</td>
          <td>24.952133</td>
          <td>0.097278</td>
          <td>24.222663</td>
          <td>0.114831</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.251599</td>
          <td>0.231565</td>
          <td>26.023608</td>
          <td>0.130661</td>
          <td>25.590927</td>
          <td>0.169086</td>
          <td>25.229335</td>
          <td>0.269250</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.191783</td>
          <td>0.628810</td>
          <td>26.346735</td>
          <td>0.121298</td>
          <td>25.841537</td>
          <td>0.068538</td>
          <td>25.724179</td>
          <td>0.100666</td>
          <td>25.565145</td>
          <td>0.165412</td>
          <td>25.871668</td>
          <td>0.446244</td>
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
          <td>26.604568</td>
          <td>0.408710</td>
          <td>26.332363</td>
          <td>0.119794</td>
          <td>25.433648</td>
          <td>0.047726</td>
          <td>25.138108</td>
          <td>0.060010</td>
          <td>24.874255</td>
          <td>0.090848</td>
          <td>24.722529</td>
          <td>0.176554</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.780948</td>
          <td>0.467099</td>
          <td>27.018547</td>
          <td>0.215072</td>
          <td>25.951735</td>
          <td>0.075556</td>
          <td>25.144751</td>
          <td>0.060365</td>
          <td>24.840233</td>
          <td>0.088170</td>
          <td>24.345520</td>
          <td>0.127765</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.965540</td>
          <td>0.535180</td>
          <td>26.575051</td>
          <td>0.147726</td>
          <td>26.633143</td>
          <td>0.137135</td>
          <td>26.166158</td>
          <td>0.147751</td>
          <td>26.069292</td>
          <td>0.252341</td>
          <td>25.388147</td>
          <td>0.306140</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.468478</td>
          <td>0.367890</td>
          <td>26.173834</td>
          <td>0.104343</td>
          <td>26.086822</td>
          <td>0.085121</td>
          <td>25.884754</td>
          <td>0.115823</td>
          <td>25.680431</td>
          <td>0.182433</td>
          <td>25.259239</td>
          <td>0.275882</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.167674</td>
          <td>1.165837</td>
          <td>26.453587</td>
          <td>0.133055</td>
          <td>26.833852</td>
          <td>0.162920</td>
          <td>26.091783</td>
          <td>0.138588</td>
          <td>25.904941</td>
          <td>0.220277</td>
          <td>26.347749</td>
          <td>0.630903</td>
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
          <td>26.839678</td>
          <td>0.538154</td>
          <td>26.576849</td>
          <td>0.170031</td>
          <td>26.131266</td>
          <td>0.104030</td>
          <td>25.255477</td>
          <td>0.078916</td>
          <td>24.844866</td>
          <td>0.104027</td>
          <td>23.841761</td>
          <td>0.097202</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.159011</td>
          <td>0.674273</td>
          <td>27.501647</td>
          <td>0.362857</td>
          <td>26.748546</td>
          <td>0.177208</td>
          <td>26.375094</td>
          <td>0.207621</td>
          <td>25.776888</td>
          <td>0.230729</td>
          <td>25.337554</td>
          <td>0.342094</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.915011</td>
          <td>1.101161</td>
          <td>27.680911</td>
          <td>0.424098</td>
          <td>28.004535</td>
          <td>0.494069</td>
          <td>26.136708</td>
          <td>0.173572</td>
          <td>25.031185</td>
          <td>0.125088</td>
          <td>24.364977</td>
          <td>0.156506</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.563103</td>
          <td>0.912793</td>
          <td>30.537383</td>
          <td>2.246820</td>
          <td>27.090863</td>
          <td>0.251363</td>
          <td>26.017706</td>
          <td>0.164047</td>
          <td>25.806321</td>
          <td>0.251716</td>
          <td>25.347039</td>
          <td>0.366457</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.547193</td>
          <td>0.433239</td>
          <td>26.147054</td>
          <td>0.117520</td>
          <td>26.004959</td>
          <td>0.093159</td>
          <td>25.702094</td>
          <td>0.116786</td>
          <td>25.273008</td>
          <td>0.150811</td>
          <td>24.508454</td>
          <td>0.173040</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.591670</td>
          <td>0.175371</td>
          <td>25.362576</td>
          <td>0.053913</td>
          <td>25.104953</td>
          <td>0.070617</td>
          <td>24.931510</td>
          <td>0.114562</td>
          <td>24.508992</td>
          <td>0.176709</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.957011</td>
          <td>0.586981</td>
          <td>26.862511</td>
          <td>0.217042</td>
          <td>26.104879</td>
          <td>0.102079</td>
          <td>25.118346</td>
          <td>0.070215</td>
          <td>24.826932</td>
          <td>0.102834</td>
          <td>24.473720</td>
          <td>0.168648</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.340611</td>
          <td>0.372737</td>
          <td>26.815605</td>
          <td>0.210260</td>
          <td>26.502221</td>
          <td>0.145352</td>
          <td>26.252072</td>
          <td>0.189575</td>
          <td>25.753867</td>
          <td>0.229068</td>
          <td>25.937520</td>
          <td>0.545135</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.760530</td>
          <td>0.237067</td>
          <td>26.014331</td>
          <td>0.107660</td>
          <td>26.089031</td>
          <td>0.103445</td>
          <td>25.856769</td>
          <td>0.137839</td>
          <td>25.850534</td>
          <td>0.252459</td>
          <td>25.361831</td>
          <td>0.358834</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.122046</td>
          <td>0.661215</td>
          <td>27.073805</td>
          <td>0.259699</td>
          <td>26.482847</td>
          <td>0.142565</td>
          <td>26.509558</td>
          <td>0.234455</td>
          <td>25.619315</td>
          <td>0.204226</td>
          <td>25.602107</td>
          <td>0.423822</td>
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
          <td>27.415238</td>
          <td>0.732715</td>
          <td>26.993519</td>
          <td>0.210648</td>
          <td>26.184010</td>
          <td>0.092732</td>
          <td>25.118797</td>
          <td>0.058999</td>
          <td>24.721826</td>
          <td>0.079443</td>
          <td>24.013868</td>
          <td>0.095683</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.111826</td>
          <td>0.594726</td>
          <td>27.318998</td>
          <td>0.275662</td>
          <td>26.728527</td>
          <td>0.149009</td>
          <td>26.169699</td>
          <td>0.148346</td>
          <td>25.972535</td>
          <td>0.233200</td>
          <td>25.671876</td>
          <td>0.383296</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.439721</td>
          <td>0.778269</td>
          <td>30.313224</td>
          <td>1.927732</td>
          <td>27.489288</td>
          <td>0.303331</td>
          <td>26.103879</td>
          <td>0.152266</td>
          <td>24.915860</td>
          <td>0.102247</td>
          <td>24.144106</td>
          <td>0.116640</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.517601</td>
          <td>0.885399</td>
          <td>28.484204</td>
          <td>0.774664</td>
          <td>27.322613</td>
          <td>0.302433</td>
          <td>26.493596</td>
          <td>0.243720</td>
          <td>25.463427</td>
          <td>0.188541</td>
          <td>24.812053</td>
          <td>0.237456</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.549110</td>
          <td>0.391981</td>
          <td>26.051368</td>
          <td>0.093850</td>
          <td>25.837183</td>
          <td>0.068372</td>
          <td>25.714487</td>
          <td>0.099964</td>
          <td>25.796421</td>
          <td>0.201444</td>
          <td>25.325681</td>
          <td>0.291529</td>
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
          <td>27.039634</td>
          <td>0.590774</td>
          <td>26.351761</td>
          <td>0.130409</td>
          <td>25.492440</td>
          <td>0.054458</td>
          <td>25.094511</td>
          <td>0.062750</td>
          <td>24.872376</td>
          <td>0.098119</td>
          <td>24.838218</td>
          <td>0.210459</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.843593</td>
          <td>0.494139</td>
          <td>26.481389</td>
          <td>0.138218</td>
          <td>26.061379</td>
          <td>0.084622</td>
          <td>25.209228</td>
          <td>0.065044</td>
          <td>24.748420</td>
          <td>0.082675</td>
          <td>24.272867</td>
          <td>0.122006</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.737611</td>
          <td>0.176899</td>
          <td>26.394289</td>
          <td>0.117005</td>
          <td>26.190818</td>
          <td>0.158643</td>
          <td>26.240288</td>
          <td>0.303321</td>
          <td>25.531810</td>
          <td>0.359145</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.811465</td>
          <td>0.233853</td>
          <td>26.179138</td>
          <td>0.115879</td>
          <td>25.993832</td>
          <td>0.087991</td>
          <td>25.845893</td>
          <td>0.126064</td>
          <td>25.850532</td>
          <td>0.234602</td>
          <td>25.104815</td>
          <td>0.271560</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.187511</td>
          <td>0.640638</td>
          <td>26.943615</td>
          <td>0.208644</td>
          <td>26.715413</td>
          <td>0.152907</td>
          <td>26.070387</td>
          <td>0.141611</td>
          <td>25.999464</td>
          <td>0.247085</td>
          <td>26.127614</td>
          <td>0.557569</td>
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
