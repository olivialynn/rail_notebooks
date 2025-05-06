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

    <pzflow.flow.Flow at 0x7f73360ad240>



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
          <td>27.540337</td>
          <td>0.795815</td>
          <td>27.066350</td>
          <td>0.223803</td>
          <td>26.084985</td>
          <td>0.084983</td>
          <td>25.237458</td>
          <td>0.065537</td>
          <td>24.689997</td>
          <td>0.077232</td>
          <td>24.011874</td>
          <td>0.095502</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.726869</td>
          <td>0.896649</td>
          <td>27.691622</td>
          <td>0.370642</td>
          <td>26.509866</td>
          <td>0.123256</td>
          <td>26.213880</td>
          <td>0.153927</td>
          <td>26.078485</td>
          <td>0.254252</td>
          <td>25.143894</td>
          <td>0.251070</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.548843</td>
          <td>1.431119</td>
          <td>32.353813</td>
          <td>3.720830</td>
          <td>28.209289</td>
          <td>0.492493</td>
          <td>25.991299</td>
          <td>0.127055</td>
          <td>24.998865</td>
          <td>0.101344</td>
          <td>24.291802</td>
          <td>0.121948</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.736221</td>
          <td>0.451690</td>
          <td>28.479818</td>
          <td>0.662357</td>
          <td>27.276172</td>
          <td>0.236323</td>
          <td>26.263308</td>
          <td>0.160577</td>
          <td>25.262522</td>
          <td>0.127515</td>
          <td>25.200637</td>
          <td>0.263018</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.608499</td>
          <td>0.409943</td>
          <td>26.152402</td>
          <td>0.102406</td>
          <td>26.025793</td>
          <td>0.080662</td>
          <td>25.846828</td>
          <td>0.112058</td>
          <td>25.590200</td>
          <td>0.168981</td>
          <td>24.935223</td>
          <td>0.211196</td>
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
          <td>29.138972</td>
          <td>1.891918</td>
          <td>26.276003</td>
          <td>0.114066</td>
          <td>25.385792</td>
          <td>0.045741</td>
          <td>24.997851</td>
          <td>0.052985</td>
          <td>24.860116</td>
          <td>0.089726</td>
          <td>24.817052</td>
          <td>0.191250</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.730498</td>
          <td>0.168718</td>
          <td>26.120625</td>
          <td>0.087693</td>
          <td>25.159225</td>
          <td>0.061145</td>
          <td>24.787222</td>
          <td>0.084149</td>
          <td>24.216584</td>
          <td>0.114225</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.708350</td>
          <td>0.442297</td>
          <td>26.948155</td>
          <td>0.202778</td>
          <td>26.429628</td>
          <td>0.114949</td>
          <td>26.208741</td>
          <td>0.153250</td>
          <td>26.040187</td>
          <td>0.246375</td>
          <td>25.574071</td>
          <td>0.354819</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.900902</td>
          <td>0.510507</td>
          <td>26.273870</td>
          <td>0.113854</td>
          <td>26.073637</td>
          <td>0.084138</td>
          <td>25.913324</td>
          <td>0.118739</td>
          <td>25.585624</td>
          <td>0.168324</td>
          <td>25.391143</td>
          <td>0.306877</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.673448</td>
          <td>0.430759</td>
          <td>27.061337</td>
          <td>0.222873</td>
          <td>26.539138</td>
          <td>0.126425</td>
          <td>26.381879</td>
          <td>0.177633</td>
          <td>25.576899</td>
          <td>0.167078</td>
          <td>26.151692</td>
          <td>0.548850</td>
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
          <td>26.433134</td>
          <td>0.396989</td>
          <td>26.685059</td>
          <td>0.186355</td>
          <td>26.025420</td>
          <td>0.094817</td>
          <td>25.107090</td>
          <td>0.069217</td>
          <td>24.683744</td>
          <td>0.090323</td>
          <td>24.010148</td>
          <td>0.112616</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.779550</td>
          <td>1.709170</td>
          <td>27.452843</td>
          <td>0.349228</td>
          <td>26.626349</td>
          <td>0.159698</td>
          <td>26.021302</td>
          <td>0.153844</td>
          <td>25.918965</td>
          <td>0.259370</td>
          <td>25.273795</td>
          <td>0.325242</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.910504</td>
          <td>0.460645</td>
          <td>25.962756</td>
          <td>0.149612</td>
          <td>25.184899</td>
          <td>0.142852</td>
          <td>24.326093</td>
          <td>0.151379</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.246424</td>
          <td>2.144743</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.365097</td>
          <td>0.313924</td>
          <td>26.374904</td>
          <td>0.221676</td>
          <td>25.700422</td>
          <td>0.230661</td>
          <td>28.882079</td>
          <td>2.643037</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.991853</td>
          <td>0.280126</td>
          <td>26.007694</td>
          <td>0.104085</td>
          <td>26.124381</td>
          <td>0.103439</td>
          <td>25.590400</td>
          <td>0.105948</td>
          <td>25.791349</td>
          <td>0.233535</td>
          <td>25.059261</td>
          <td>0.273721</td>
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
          <td>26.444002</td>
          <td>0.405971</td>
          <td>26.469081</td>
          <td>0.157991</td>
          <td>25.440675</td>
          <td>0.057781</td>
          <td>25.058724</td>
          <td>0.067787</td>
          <td>24.844400</td>
          <td>0.106180</td>
          <td>24.303319</td>
          <td>0.148256</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.915396</td>
          <td>1.091376</td>
          <td>26.712621</td>
          <td>0.191428</td>
          <td>26.051700</td>
          <td>0.097433</td>
          <td>25.121807</td>
          <td>0.070430</td>
          <td>24.824577</td>
          <td>0.102623</td>
          <td>24.347442</td>
          <td>0.151399</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.458798</td>
          <td>0.408357</td>
          <td>27.039745</td>
          <td>0.253140</td>
          <td>26.478577</td>
          <td>0.142425</td>
          <td>26.240674</td>
          <td>0.187760</td>
          <td>26.077747</td>
          <td>0.298485</td>
          <td>25.575851</td>
          <td>0.416432</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.170493</td>
          <td>0.330319</td>
          <td>26.202842</td>
          <td>0.126821</td>
          <td>26.151670</td>
          <td>0.109264</td>
          <td>26.019395</td>
          <td>0.158500</td>
          <td>25.640973</td>
          <td>0.212234</td>
          <td>25.145132</td>
          <td>0.302136</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.757988</td>
          <td>0.510239</td>
          <td>26.614882</td>
          <td>0.177154</td>
          <td>26.290819</td>
          <td>0.120750</td>
          <td>26.461708</td>
          <td>0.225337</td>
          <td>25.686810</td>
          <td>0.216082</td>
          <td>26.378439</td>
          <td>0.739386</td>
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

.. parsed-literal::

    




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
          <td>26.574621</td>
          <td>0.147688</td>
          <td>26.001940</td>
          <td>0.078992</td>
          <td>25.071010</td>
          <td>0.056549</td>
          <td>24.744955</td>
          <td>0.081081</td>
          <td>24.067009</td>
          <td>0.100246</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.258112</td>
          <td>0.658784</td>
          <td>27.102138</td>
          <td>0.230727</td>
          <td>26.672580</td>
          <td>0.142009</td>
          <td>26.192074</td>
          <td>0.151222</td>
          <td>25.766093</td>
          <td>0.196286</td>
          <td>25.347073</td>
          <td>0.296467</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.083993</td>
          <td>0.530144</td>
          <td>27.687679</td>
          <td>0.355091</td>
          <td>26.007429</td>
          <td>0.140151</td>
          <td>25.119415</td>
          <td>0.122102</td>
          <td>24.277465</td>
          <td>0.130948</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.842633</td>
          <td>0.972104</td>
          <td>27.590901</td>
          <td>0.373949</td>
          <td>26.309624</td>
          <td>0.209195</td>
          <td>25.828893</td>
          <td>0.255567</td>
          <td>25.221619</td>
          <td>0.330918</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.051506</td>
          <td>0.569868</td>
          <td>26.104681</td>
          <td>0.098339</td>
          <td>25.951744</td>
          <td>0.075665</td>
          <td>25.713150</td>
          <td>0.099847</td>
          <td>25.671514</td>
          <td>0.181308</td>
          <td>25.155451</td>
          <td>0.253813</td>
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
          <td>26.813435</td>
          <td>0.501607</td>
          <td>26.320818</td>
          <td>0.126964</td>
          <td>25.533427</td>
          <td>0.056476</td>
          <td>25.099504</td>
          <td>0.063028</td>
          <td>24.996766</td>
          <td>0.109397</td>
          <td>24.906937</td>
          <td>0.222870</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.664170</td>
          <td>0.431999</td>
          <td>26.654759</td>
          <td>0.160384</td>
          <td>25.909529</td>
          <td>0.074007</td>
          <td>25.136106</td>
          <td>0.060960</td>
          <td>24.925301</td>
          <td>0.096590</td>
          <td>24.176634</td>
          <td>0.112206</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.125000</td>
          <td>0.616773</td>
          <td>26.630375</td>
          <td>0.161477</td>
          <td>26.413369</td>
          <td>0.118963</td>
          <td>26.661274</td>
          <td>0.235734</td>
          <td>25.804864</td>
          <td>0.212256</td>
          <td>25.503983</td>
          <td>0.351385</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.611071</td>
          <td>0.197926</td>
          <td>26.278429</td>
          <td>0.126305</td>
          <td>26.019386</td>
          <td>0.089991</td>
          <td>25.977604</td>
          <td>0.141262</td>
          <td>25.542073</td>
          <td>0.181201</td>
          <td>25.301272</td>
          <td>0.318152</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.374479</td>
          <td>0.727895</td>
          <td>26.688309</td>
          <td>0.168205</td>
          <td>26.588604</td>
          <td>0.137106</td>
          <td>26.198838</td>
          <td>0.158119</td>
          <td>25.518706</td>
          <td>0.165102</td>
          <td>25.741798</td>
          <td>0.418687</td>
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
