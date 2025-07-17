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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fdea0b5fe20>



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
    0      23.994413  0.034641  0.020171  
    1      25.391064  0.072681  0.064590  
    2      24.304707  0.089804  0.088103  
    3      25.291103  0.003236  0.002889  
    4      25.096743  0.075753  0.049809  
    ...          ...       ...       ...  
    99995  24.737946  0.104475  0.096834  
    99996  24.224169  0.059748  0.054021  
    99997  25.613836  0.018018  0.017712  
    99998  25.274899  0.060642  0.044472  
    99999  25.699642  0.040560  0.026814  
    
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
          <td>26.754117</td>
          <td>0.457805</td>
          <td>26.564694</td>
          <td>0.146419</td>
          <td>26.116080</td>
          <td>0.087343</td>
          <td>25.172568</td>
          <td>0.061873</td>
          <td>24.730158</td>
          <td>0.080019</td>
          <td>24.019423</td>
          <td>0.096137</td>
          <td>0.034641</td>
          <td>0.020171</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.550235</td>
          <td>0.800965</td>
          <td>27.111500</td>
          <td>0.232342</td>
          <td>26.687274</td>
          <td>0.143684</td>
          <td>26.132341</td>
          <td>0.143516</td>
          <td>25.844814</td>
          <td>0.209497</td>
          <td>26.019916</td>
          <td>0.498476</td>
          <td>0.072681</td>
          <td>0.064590</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.777932</td>
          <td>0.354302</td>
          <td>26.042979</td>
          <td>0.132868</td>
          <td>25.027450</td>
          <td>0.103912</td>
          <td>24.350907</td>
          <td>0.128362</td>
          <td>0.089804</td>
          <td>0.088103</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.312931</td>
          <td>1.263579</td>
          <td>28.661790</td>
          <td>0.749225</td>
          <td>27.265373</td>
          <td>0.234221</td>
          <td>26.228058</td>
          <td>0.155808</td>
          <td>25.360120</td>
          <td>0.138740</td>
          <td>25.542478</td>
          <td>0.346109</td>
          <td>0.003236</td>
          <td>0.002889</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.125113</td>
          <td>0.600028</td>
          <td>25.965586</td>
          <td>0.086933</td>
          <td>25.958716</td>
          <td>0.076023</td>
          <td>25.640404</td>
          <td>0.093534</td>
          <td>25.645066</td>
          <td>0.177047</td>
          <td>24.975645</td>
          <td>0.218442</td>
          <td>0.075753</td>
          <td>0.049809</td>
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
          <td>28.718788</td>
          <td>1.558094</td>
          <td>26.466882</td>
          <td>0.134592</td>
          <td>25.474361</td>
          <td>0.049483</td>
          <td>25.079630</td>
          <td>0.056975</td>
          <td>24.732703</td>
          <td>0.080199</td>
          <td>24.605165</td>
          <td>0.159760</td>
          <td>0.104475</td>
          <td>0.096834</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.528954</td>
          <td>0.789920</td>
          <td>27.006874</td>
          <td>0.212988</td>
          <td>26.136050</td>
          <td>0.088891</td>
          <td>25.373671</td>
          <td>0.073936</td>
          <td>24.948441</td>
          <td>0.096963</td>
          <td>24.333649</td>
          <td>0.126457</td>
          <td>0.059748</td>
          <td>0.054021</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.850865</td>
          <td>0.492028</td>
          <td>27.144336</td>
          <td>0.238735</td>
          <td>26.160450</td>
          <td>0.090820</td>
          <td>26.299155</td>
          <td>0.165566</td>
          <td>25.593159</td>
          <td>0.169407</td>
          <td>25.377264</td>
          <td>0.303479</td>
          <td>0.018018</td>
          <td>0.017712</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.020940</td>
          <td>0.257211</td>
          <td>26.276634</td>
          <td>0.114129</td>
          <td>26.052363</td>
          <td>0.082575</td>
          <td>25.924783</td>
          <td>0.119928</td>
          <td>25.133621</td>
          <td>0.114004</td>
          <td>25.817083</td>
          <td>0.428162</td>
          <td>0.060642</td>
          <td>0.044472</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.229984</td>
          <td>0.304653</td>
          <td>26.945827</td>
          <td>0.202382</td>
          <td>26.442185</td>
          <td>0.116213</td>
          <td>26.330402</td>
          <td>0.170032</td>
          <td>25.789534</td>
          <td>0.200011</td>
          <td>25.902927</td>
          <td>0.456875</td>
          <td>0.040560</td>
          <td>0.026814</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.638468</td>
          <td>0.179577</td>
          <td>26.093230</td>
          <td>0.100895</td>
          <td>25.118139</td>
          <td>0.070094</td>
          <td>24.777427</td>
          <td>0.098328</td>
          <td>24.170398</td>
          <td>0.129788</td>
          <td>0.034641</td>
          <td>0.020171</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.380172</td>
          <td>0.333958</td>
          <td>26.703235</td>
          <td>0.173113</td>
          <td>26.366385</td>
          <td>0.209309</td>
          <td>26.106331</td>
          <td>0.306262</td>
          <td>25.669477</td>
          <td>0.448320</td>
          <td>0.072681</td>
          <td>0.064590</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.849038</td>
          <td>3.589560</td>
          <td>inf</td>
          <td>inf</td>
          <td>33.607573</td>
          <td>5.019285</td>
          <td>25.998466</td>
          <td>0.154874</td>
          <td>25.045277</td>
          <td>0.127114</td>
          <td>24.280687</td>
          <td>0.146165</td>
          <td>0.089804</td>
          <td>0.088103</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.724965</td>
          <td>0.173662</td>
          <td>26.045345</td>
          <td>0.157011</td>
          <td>25.795930</td>
          <td>0.234351</td>
          <td>25.260127</td>
          <td>0.321662</td>
          <td>0.003236</td>
          <td>0.002889</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.382960</td>
          <td>0.385439</td>
          <td>26.385977</td>
          <td>0.146197</td>
          <td>25.958456</td>
          <td>0.090640</td>
          <td>25.819295</td>
          <td>0.131073</td>
          <td>25.421538</td>
          <td>0.173446</td>
          <td>25.013894</td>
          <td>0.267185</td>
          <td>0.075753</td>
          <td>0.049809</td>
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
          <td>27.052892</td>
          <td>0.639248</td>
          <td>26.532328</td>
          <td>0.168568</td>
          <td>25.348578</td>
          <td>0.053919</td>
          <td>25.028756</td>
          <td>0.066869</td>
          <td>24.780181</td>
          <td>0.101625</td>
          <td>25.593285</td>
          <td>0.429802</td>
          <td>0.104475</td>
          <td>0.096834</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.684715</td>
          <td>0.483644</td>
          <td>26.668605</td>
          <td>0.185530</td>
          <td>26.140701</td>
          <td>0.106036</td>
          <td>25.177822</td>
          <td>0.074528</td>
          <td>24.910632</td>
          <td>0.111376</td>
          <td>24.319691</td>
          <td>0.148838</td>
          <td>0.059748</td>
          <td>0.054021</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.879439</td>
          <td>0.219537</td>
          <td>26.142409</td>
          <td>0.105159</td>
          <td>26.259498</td>
          <td>0.188555</td>
          <td>26.726234</td>
          <td>0.488184</td>
          <td>25.512270</td>
          <td>0.392417</td>
          <td>0.018018</td>
          <td>0.017712</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.821699</td>
          <td>0.245409</td>
          <td>26.164014</td>
          <td>0.120245</td>
          <td>26.093203</td>
          <td>0.101580</td>
          <td>25.865721</td>
          <td>0.135850</td>
          <td>25.844814</td>
          <td>0.246183</td>
          <td>25.646753</td>
          <td>0.438237</td>
          <td>0.060642</td>
          <td>0.044472</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.931412</td>
          <td>0.267398</td>
          <td>26.544473</td>
          <td>0.165987</td>
          <td>26.456120</td>
          <td>0.138494</td>
          <td>26.387144</td>
          <td>0.210506</td>
          <td>25.942100</td>
          <td>0.265249</td>
          <td>25.762784</td>
          <td>0.475847</td>
          <td>0.040560</td>
          <td>0.026814</td>
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
          <td>27.041340</td>
          <td>0.568739</td>
          <td>26.959738</td>
          <td>0.206589</td>
          <td>26.045612</td>
          <td>0.082978</td>
          <td>25.029956</td>
          <td>0.055145</td>
          <td>24.673103</td>
          <td>0.076917</td>
          <td>24.058412</td>
          <td>0.100598</td>
          <td>0.034641</td>
          <td>0.020171</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.506513</td>
          <td>0.335920</td>
          <td>26.705966</td>
          <td>0.154864</td>
          <td>26.103597</td>
          <td>0.148931</td>
          <td>25.769869</td>
          <td>0.208367</td>
          <td>25.425359</td>
          <td>0.333741</td>
          <td>0.072681</td>
          <td>0.064590</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.199223</td>
          <td>0.316572</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.395736</td>
          <td>1.900734</td>
          <td>25.835311</td>
          <td>0.122670</td>
          <td>25.155293</td>
          <td>0.127862</td>
          <td>24.367638</td>
          <td>0.143750</td>
          <td>0.089804</td>
          <td>0.088103</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.409027</td>
          <td>0.630668</td>
          <td>27.560427</td>
          <td>0.298059</td>
          <td>26.473382</td>
          <td>0.191948</td>
          <td>25.534807</td>
          <td>0.161205</td>
          <td>26.252989</td>
          <td>0.590232</td>
          <td>0.003236</td>
          <td>0.002889</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.888003</td>
          <td>0.521524</td>
          <td>26.070783</td>
          <td>0.099808</td>
          <td>25.961502</td>
          <td>0.080339</td>
          <td>25.514652</td>
          <td>0.088479</td>
          <td>25.493984</td>
          <td>0.163790</td>
          <td>24.754473</td>
          <td>0.191137</td>
          <td>0.075753</td>
          <td>0.049809</td>
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
          <td>26.757316</td>
          <td>0.494000</td>
          <td>26.281598</td>
          <td>0.127374</td>
          <td>25.404005</td>
          <td>0.052557</td>
          <td>25.157863</td>
          <td>0.069397</td>
          <td>25.060786</td>
          <td>0.120631</td>
          <td>25.695961</td>
          <td>0.435180</td>
          <td>0.104475</td>
          <td>0.096834</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.081431</td>
          <td>0.595779</td>
          <td>26.888661</td>
          <td>0.199797</td>
          <td>26.080619</td>
          <td>0.088316</td>
          <td>25.155696</td>
          <td>0.063740</td>
          <td>24.936281</td>
          <td>0.100056</td>
          <td>24.164789</td>
          <td>0.114031</td>
          <td>0.059748</td>
          <td>0.054021</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.345687</td>
          <td>0.700729</td>
          <td>26.657419</td>
          <td>0.159104</td>
          <td>26.491530</td>
          <td>0.121828</td>
          <td>26.203416</td>
          <td>0.153229</td>
          <td>25.939192</td>
          <td>0.227565</td>
          <td>25.150677</td>
          <td>0.253533</td>
          <td>0.018018</td>
          <td>0.017712</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.538982</td>
          <td>0.397503</td>
          <td>26.271311</td>
          <td>0.117276</td>
          <td>26.100071</td>
          <td>0.089363</td>
          <td>26.013937</td>
          <td>0.134595</td>
          <td>25.719283</td>
          <td>0.195299</td>
          <td>25.497369</td>
          <td>0.345673</td>
          <td>0.060642</td>
          <td>0.044472</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.627250</td>
          <td>0.419849</td>
          <td>26.438259</td>
          <td>0.133088</td>
          <td>26.382605</td>
          <td>0.112074</td>
          <td>26.298307</td>
          <td>0.168123</td>
          <td>25.622100</td>
          <td>0.176291</td>
          <td>25.292575</td>
          <td>0.287750</td>
          <td>0.040560</td>
          <td>0.026814</td>
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
