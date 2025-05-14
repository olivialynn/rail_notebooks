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

    <pzflow.flow.Flow at 0x7fcda04dd960>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.432779</td>
          <td>0.130684</td>
          <td>26.142433</td>
          <td>0.089392</td>
          <td>25.241022</td>
          <td>0.065744</td>
          <td>24.639325</td>
          <td>0.073850</td>
          <td>24.026247</td>
          <td>0.096714</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.703432</td>
          <td>0.883543</td>
          <td>27.332177</td>
          <td>0.278416</td>
          <td>26.712293</td>
          <td>0.146809</td>
          <td>26.445797</td>
          <td>0.187508</td>
          <td>25.918220</td>
          <td>0.222725</td>
          <td>24.910457</td>
          <td>0.206865</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.936046</td>
          <td>1.019089</td>
          <td>28.416923</td>
          <td>0.634097</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.960319</td>
          <td>0.123687</td>
          <td>25.164614</td>
          <td>0.117122</td>
          <td>24.445507</td>
          <td>0.139297</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.550223</td>
          <td>0.800959</td>
          <td>28.993365</td>
          <td>0.927059</td>
          <td>27.810336</td>
          <td>0.363416</td>
          <td>26.183541</td>
          <td>0.149973</td>
          <td>25.754716</td>
          <td>0.194240</td>
          <td>25.372251</td>
          <td>0.302259</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.087587</td>
          <td>0.271575</td>
          <td>26.254763</td>
          <td>0.111976</td>
          <td>25.988815</td>
          <td>0.078072</td>
          <td>25.756680</td>
          <td>0.103572</td>
          <td>25.370334</td>
          <td>0.139967</td>
          <td>25.129025</td>
          <td>0.248020</td>
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
          <td>26.647742</td>
          <td>0.422419</td>
          <td>26.192408</td>
          <td>0.106049</td>
          <td>25.453638</td>
          <td>0.048581</td>
          <td>25.040476</td>
          <td>0.055029</td>
          <td>24.757210</td>
          <td>0.081952</td>
          <td>24.670561</td>
          <td>0.168926</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.148645</td>
          <td>2.779855</td>
          <td>26.808459</td>
          <td>0.180260</td>
          <td>25.985409</td>
          <td>0.077837</td>
          <td>25.173361</td>
          <td>0.061916</td>
          <td>24.722124</td>
          <td>0.079454</td>
          <td>24.053900</td>
          <td>0.099088</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.403557</td>
          <td>1.326688</td>
          <td>26.540431</td>
          <td>0.143397</td>
          <td>26.386988</td>
          <td>0.110755</td>
          <td>25.973008</td>
          <td>0.125056</td>
          <td>26.347825</td>
          <td>0.316206</td>
          <td>26.010443</td>
          <td>0.495000</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.171465</td>
          <td>0.104127</td>
          <td>26.118117</td>
          <td>0.087500</td>
          <td>25.885384</td>
          <td>0.115887</td>
          <td>26.041374</td>
          <td>0.246616</td>
          <td>25.095123</td>
          <td>0.241190</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.034797</td>
          <td>0.562631</td>
          <td>26.878115</td>
          <td>0.191184</td>
          <td>26.728720</td>
          <td>0.148896</td>
          <td>26.425362</td>
          <td>0.184297</td>
          <td>26.174861</td>
          <td>0.275070</td>
          <td>25.640335</td>
          <td>0.373690</td>
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
          <td>27.113685</td>
          <td>0.653478</td>
          <td>26.817391</td>
          <td>0.208274</td>
          <td>25.978853</td>
          <td>0.091018</td>
          <td>25.373224</td>
          <td>0.087544</td>
          <td>25.003607</td>
          <td>0.119468</td>
          <td>24.062616</td>
          <td>0.117879</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.224118</td>
          <td>0.291032</td>
          <td>26.549100</td>
          <td>0.149474</td>
          <td>25.942291</td>
          <td>0.143753</td>
          <td>25.623791</td>
          <td>0.203081</td>
          <td>25.768660</td>
          <td>0.476367</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.062563</td>
          <td>1.080417</td>
          <td>28.344074</td>
          <td>0.630759</td>
          <td>26.036251</td>
          <td>0.159332</td>
          <td>24.924555</td>
          <td>0.114018</td>
          <td>24.244554</td>
          <td>0.141134</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.580510</td>
          <td>0.922719</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.580186</td>
          <td>0.372016</td>
          <td>26.451916</td>
          <td>0.236295</td>
          <td>25.577207</td>
          <td>0.208168</td>
          <td>25.462269</td>
          <td>0.400709</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.523464</td>
          <td>0.425501</td>
          <td>26.039634</td>
          <td>0.107028</td>
          <td>25.950450</td>
          <td>0.088802</td>
          <td>25.726699</td>
          <td>0.119312</td>
          <td>25.178551</td>
          <td>0.139046</td>
          <td>25.075798</td>
          <td>0.277424</td>
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
          <td>27.269939</td>
          <td>0.735795</td>
          <td>26.594911</td>
          <td>0.175854</td>
          <td>25.456328</td>
          <td>0.058588</td>
          <td>25.208223</td>
          <td>0.077365</td>
          <td>24.736685</td>
          <td>0.096627</td>
          <td>24.842885</td>
          <td>0.233784</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.687642</td>
          <td>0.187438</td>
          <td>26.016133</td>
          <td>0.094441</td>
          <td>25.225425</td>
          <td>0.077185</td>
          <td>24.734892</td>
          <td>0.094867</td>
          <td>24.603011</td>
          <td>0.188183</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.556383</td>
          <td>0.439849</td>
          <td>26.973876</td>
          <td>0.239789</td>
          <td>26.506582</td>
          <td>0.145898</td>
          <td>26.192198</td>
          <td>0.180218</td>
          <td>26.526463</td>
          <td>0.424335</td>
          <td>28.107440</td>
          <td>1.910509</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.994946</td>
          <td>0.613134</td>
          <td>26.204243</td>
          <td>0.126975</td>
          <td>26.268525</td>
          <td>0.120966</td>
          <td>25.862928</td>
          <td>0.138573</td>
          <td>25.559191</td>
          <td>0.198178</td>
          <td>25.259612</td>
          <td>0.331047</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.515784</td>
          <td>0.425739</td>
          <td>26.759078</td>
          <td>0.200064</td>
          <td>26.738851</td>
          <td>0.177433</td>
          <td>26.633977</td>
          <td>0.259729</td>
          <td>26.095559</td>
          <td>0.302030</td>
          <td>26.061004</td>
          <td>0.594195</td>
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
          <td>27.689147</td>
          <td>0.875674</td>
          <td>26.817809</td>
          <td>0.181712</td>
          <td>25.983936</td>
          <td>0.077746</td>
          <td>25.171186</td>
          <td>0.061806</td>
          <td>24.588222</td>
          <td>0.070595</td>
          <td>24.005258</td>
          <td>0.094962</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.650422</td>
          <td>0.854768</td>
          <td>26.821093</td>
          <td>0.182341</td>
          <td>26.642922</td>
          <td>0.138425</td>
          <td>25.975054</td>
          <td>0.125401</td>
          <td>26.087620</td>
          <td>0.256389</td>
          <td>25.162429</td>
          <td>0.255153</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.295784</td>
          <td>0.338508</td>
          <td>29.940143</td>
          <td>1.627586</td>
          <td>28.327561</td>
          <td>0.574372</td>
          <td>25.847298</td>
          <td>0.122019</td>
          <td>24.950961</td>
          <td>0.105435</td>
          <td>24.731758</td>
          <td>0.193053</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.815834</td>
          <td>0.956337</td>
          <td>27.679144</td>
          <td>0.400391</td>
          <td>26.498239</td>
          <td>0.244654</td>
          <td>25.799675</td>
          <td>0.249512</td>
          <td>25.252714</td>
          <td>0.339167</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.061396</td>
          <td>0.266090</td>
          <td>26.169241</td>
          <td>0.104052</td>
          <td>26.070515</td>
          <td>0.084026</td>
          <td>25.669162</td>
          <td>0.096069</td>
          <td>25.393052</td>
          <td>0.142932</td>
          <td>25.650064</td>
          <td>0.377022</td>
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
          <td>26.540967</td>
          <td>0.408701</td>
          <td>26.350561</td>
          <td>0.130274</td>
          <td>25.466169</td>
          <td>0.053203</td>
          <td>25.040618</td>
          <td>0.059821</td>
          <td>24.892317</td>
          <td>0.099849</td>
          <td>25.232114</td>
          <td>0.290953</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.038310</td>
          <td>0.569356</td>
          <td>26.920919</td>
          <td>0.200926</td>
          <td>26.060099</td>
          <td>0.084527</td>
          <td>25.101852</td>
          <td>0.059136</td>
          <td>24.859761</td>
          <td>0.091188</td>
          <td>23.901825</td>
          <td>0.088197</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.468735</td>
          <td>0.779099</td>
          <td>26.793517</td>
          <td>0.185470</td>
          <td>26.332326</td>
          <td>0.110856</td>
          <td>26.363935</td>
          <td>0.183811</td>
          <td>25.946411</td>
          <td>0.238739</td>
          <td>25.471962</td>
          <td>0.342632</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.883254</td>
          <td>0.539551</td>
          <td>26.091869</td>
          <td>0.107397</td>
          <td>26.113655</td>
          <td>0.097756</td>
          <td>26.037770</td>
          <td>0.148763</td>
          <td>25.663909</td>
          <td>0.200805</td>
          <td>25.036762</td>
          <td>0.256879</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.919598</td>
          <td>0.529404</td>
          <td>26.890583</td>
          <td>0.199575</td>
          <td>26.598003</td>
          <td>0.138223</td>
          <td>26.467629</td>
          <td>0.198605</td>
          <td>26.317769</td>
          <td>0.319797</td>
          <td>25.771237</td>
          <td>0.428189</td>
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
