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

    <pzflow.flow.Flow at 0x7f6386010610>



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
    0      23.994413  0.259549  0.245296  
    1      25.391064  0.112804  0.098399  
    2      24.304707  0.086828  0.071187  
    3      25.291103  0.087355  0.046337  
    4      25.096743  0.289005  0.212223  
    ...          ...       ...       ...  
    99995  24.737946  0.067563  0.046811  
    99996  24.224169  0.083093  0.051163  
    99997  25.613836  0.042037  0.041702  
    99998  25.274899  0.271646  0.174759  
    99999  25.699642  0.080286  0.052690  
    
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
          <td>28.284859</td>
          <td>1.244358</td>
          <td>26.683755</td>
          <td>0.162130</td>
          <td>26.034003</td>
          <td>0.081248</td>
          <td>25.271224</td>
          <td>0.067527</td>
          <td>24.652644</td>
          <td>0.074724</td>
          <td>23.915184</td>
          <td>0.087722</td>
          <td>0.259549</td>
          <td>0.245296</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.516023</td>
          <td>1.407186</td>
          <td>27.562549</td>
          <td>0.334896</td>
          <td>26.642610</td>
          <td>0.138260</td>
          <td>26.736487</td>
          <td>0.239053</td>
          <td>25.790218</td>
          <td>0.200126</td>
          <td>25.189715</td>
          <td>0.260680</td>
          <td>0.112804</td>
          <td>0.098399</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.953285</td>
          <td>0.530432</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.553494</td>
          <td>0.296366</td>
          <td>26.067851</td>
          <td>0.135755</td>
          <td>24.993137</td>
          <td>0.100837</td>
          <td>24.564996</td>
          <td>0.154363</td>
          <td>0.086828</td>
          <td>0.071187</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.998510</td>
          <td>1.057528</td>
          <td>27.754762</td>
          <td>0.389266</td>
          <td>26.766881</td>
          <td>0.153851</td>
          <td>26.364014</td>
          <td>0.174960</td>
          <td>25.511836</td>
          <td>0.158051</td>
          <td>25.474553</td>
          <td>0.327999</td>
          <td>0.087355</td>
          <td>0.046337</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.156823</td>
          <td>0.287242</td>
          <td>26.128387</td>
          <td>0.100278</td>
          <td>25.945014</td>
          <td>0.075108</td>
          <td>25.693930</td>
          <td>0.098033</td>
          <td>25.591375</td>
          <td>0.169150</td>
          <td>24.819934</td>
          <td>0.191715</td>
          <td>0.289005</td>
          <td>0.212223</td>
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
          <td>27.060851</td>
          <td>0.573232</td>
          <td>26.385374</td>
          <td>0.125431</td>
          <td>25.458440</td>
          <td>0.048788</td>
          <td>25.023292</td>
          <td>0.054196</td>
          <td>24.929626</td>
          <td>0.095376</td>
          <td>24.386189</td>
          <td>0.132342</td>
          <td>0.067563</td>
          <td>0.046811</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.691700</td>
          <td>0.877030</td>
          <td>26.648215</td>
          <td>0.157283</td>
          <td>26.006410</td>
          <td>0.079294</td>
          <td>25.186118</td>
          <td>0.062621</td>
          <td>24.775436</td>
          <td>0.083279</td>
          <td>24.573846</td>
          <td>0.155537</td>
          <td>0.083093</td>
          <td>0.051163</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.408539</td>
          <td>0.729384</td>
          <td>26.569837</td>
          <td>0.147067</td>
          <td>26.422514</td>
          <td>0.114239</td>
          <td>26.207475</td>
          <td>0.153084</td>
          <td>25.809353</td>
          <td>0.203366</td>
          <td>25.719565</td>
          <td>0.397350</td>
          <td>0.042037</td>
          <td>0.041702</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.014511</td>
          <td>0.554481</td>
          <td>26.311821</td>
          <td>0.117675</td>
          <td>26.155193</td>
          <td>0.090401</td>
          <td>25.906784</td>
          <td>0.118065</td>
          <td>25.881716</td>
          <td>0.216055</td>
          <td>25.029775</td>
          <td>0.228497</td>
          <td>0.271646</td>
          <td>0.174759</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.902627</td>
          <td>0.998874</td>
          <td>26.765607</td>
          <td>0.173828</td>
          <td>26.878347</td>
          <td>0.169218</td>
          <td>26.742904</td>
          <td>0.240323</td>
          <td>26.057347</td>
          <td>0.249877</td>
          <td>26.242766</td>
          <td>0.585895</td>
          <td>0.080286</td>
          <td>0.052690</td>
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
          <td>29.582337</td>
          <td>2.531565</td>
          <td>26.615533</td>
          <td>0.205674</td>
          <td>25.991496</td>
          <td>0.110035</td>
          <td>25.216494</td>
          <td>0.091751</td>
          <td>24.704866</td>
          <td>0.109998</td>
          <td>23.902868</td>
          <td>0.123002</td>
          <td>0.259549</td>
          <td>0.245296</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.064430</td>
          <td>0.645704</td>
          <td>27.658730</td>
          <td>0.421726</td>
          <td>26.503529</td>
          <td>0.148958</td>
          <td>26.339528</td>
          <td>0.208905</td>
          <td>25.484085</td>
          <td>0.186970</td>
          <td>25.213787</td>
          <td>0.320770</td>
          <td>0.112804</td>
          <td>0.098399</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.033703</td>
          <td>0.988072</td>
          <td>25.964361</td>
          <td>0.149618</td>
          <td>24.990568</td>
          <td>0.120597</td>
          <td>24.394506</td>
          <td>0.160295</td>
          <td>0.086828</td>
          <td>0.071187</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.119284</td>
          <td>0.662431</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.333090</td>
          <td>0.292065</td>
          <td>26.414929</td>
          <td>0.218061</td>
          <td>25.526835</td>
          <td>0.190076</td>
          <td>25.171208</td>
          <td>0.304176</td>
          <td>0.087355</td>
          <td>0.046337</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.899041</td>
          <td>0.629444</td>
          <td>26.208319</td>
          <td>0.145503</td>
          <td>25.858994</td>
          <td>0.097917</td>
          <td>25.637635</td>
          <td>0.132327</td>
          <td>25.218120</td>
          <td>0.171059</td>
          <td>25.027327</td>
          <td>0.315210</td>
          <td>0.289005</td>
          <td>0.212223</td>
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
          <td>26.581744</td>
          <td>0.172437</td>
          <td>25.476916</td>
          <td>0.059087</td>
          <td>25.063802</td>
          <td>0.067408</td>
          <td>24.848713</td>
          <td>0.105561</td>
          <td>24.435290</td>
          <td>0.164384</td>
          <td>0.067563</td>
          <td>0.046811</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.952637</td>
          <td>1.121658</td>
          <td>26.806098</td>
          <td>0.209152</td>
          <td>25.896603</td>
          <td>0.086024</td>
          <td>25.237579</td>
          <td>0.078974</td>
          <td>24.808830</td>
          <td>0.102406</td>
          <td>24.440421</td>
          <td>0.165854</td>
          <td>0.083093</td>
          <td>0.051163</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.899599</td>
          <td>0.564005</td>
          <td>26.831540</td>
          <td>0.211822</td>
          <td>26.523784</td>
          <td>0.147077</td>
          <td>26.319173</td>
          <td>0.199229</td>
          <td>26.255630</td>
          <td>0.341819</td>
          <td>25.329463</td>
          <td>0.341710</td>
          <td>0.042037</td>
          <td>0.041702</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.508730</td>
          <td>0.465118</td>
          <td>26.290623</td>
          <td>0.151949</td>
          <td>26.002997</td>
          <td>0.107776</td>
          <td>25.870026</td>
          <td>0.156787</td>
          <td>25.814827</td>
          <td>0.273278</td>
          <td>25.974002</td>
          <td>0.625834</td>
          <td>0.271646</td>
          <td>0.174759</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.243685</td>
          <td>0.346125</td>
          <td>27.074629</td>
          <td>0.261048</td>
          <td>26.410212</td>
          <td>0.134616</td>
          <td>26.126433</td>
          <td>0.170879</td>
          <td>25.623510</td>
          <td>0.205994</td>
          <td>26.320351</td>
          <td>0.714104</td>
          <td>0.080286</td>
          <td>0.052690</td>
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
          <td>28.082563</td>
          <td>1.413017</td>
          <td>26.768135</td>
          <td>0.267199</td>
          <td>25.943098</td>
          <td>0.122824</td>
          <td>25.213826</td>
          <td>0.107020</td>
          <td>24.554608</td>
          <td>0.112374</td>
          <td>23.867946</td>
          <td>0.139151</td>
          <td>0.259549</td>
          <td>0.245296</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.269097</td>
          <td>0.627260</td>
          <td>26.491960</td>
          <td>0.138208</td>
          <td>26.127950</td>
          <td>0.163537</td>
          <td>25.722234</td>
          <td>0.214350</td>
          <td>24.852963</td>
          <td>0.224451</td>
          <td>0.112804</td>
          <td>0.098399</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.678925</td>
          <td>0.730835</td>
          <td>26.026315</td>
          <td>0.141992</td>
          <td>25.203188</td>
          <td>0.130879</td>
          <td>24.277017</td>
          <td>0.130476</td>
          <td>0.086828</td>
          <td>0.071187</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.880858</td>
          <td>0.449014</td>
          <td>27.453139</td>
          <td>0.289083</td>
          <td>26.012272</td>
          <td>0.137787</td>
          <td>25.705399</td>
          <td>0.197560</td>
          <td>25.594198</td>
          <td>0.381289</td>
          <td>0.087355</td>
          <td>0.046337</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.437438</td>
          <td>0.500644</td>
          <td>26.203086</td>
          <td>0.165902</td>
          <td>25.692004</td>
          <td>0.098061</td>
          <td>25.439621</td>
          <td>0.129372</td>
          <td>25.534670</td>
          <td>0.256490</td>
          <td>25.261679</td>
          <td>0.432317</td>
          <td>0.289005</td>
          <td>0.212223</td>
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
          <td>28.017748</td>
          <td>1.091494</td>
          <td>26.351396</td>
          <td>0.126455</td>
          <td>25.494330</td>
          <td>0.052644</td>
          <td>25.016617</td>
          <td>0.056428</td>
          <td>24.905881</td>
          <td>0.097567</td>
          <td>24.448254</td>
          <td>0.145960</td>
          <td>0.067563</td>
          <td>0.046811</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.676232</td>
          <td>0.447734</td>
          <td>26.453819</td>
          <td>0.140128</td>
          <td>25.959383</td>
          <td>0.080803</td>
          <td>25.213824</td>
          <td>0.068383</td>
          <td>24.598261</td>
          <td>0.075652</td>
          <td>24.331375</td>
          <td>0.134187</td>
          <td>0.083093</td>
          <td>0.051163</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.555112</td>
          <td>0.399094</td>
          <td>27.018607</td>
          <td>0.219248</td>
          <td>26.336413</td>
          <td>0.108450</td>
          <td>26.306852</td>
          <td>0.170643</td>
          <td>25.791383</td>
          <td>0.204815</td>
          <td>25.341117</td>
          <td>0.301373</td>
          <td>0.042037</td>
          <td>0.041702</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.931090</td>
          <td>0.323016</td>
          <td>26.069643</td>
          <td>0.139056</td>
          <td>26.262795</td>
          <td>0.150415</td>
          <td>25.745599</td>
          <td>0.157240</td>
          <td>25.639410</td>
          <td>0.262285</td>
          <td>25.533317</td>
          <td>0.499487</td>
          <td>0.271646</td>
          <td>0.174759</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.155587</td>
          <td>0.298024</td>
          <td>26.744265</td>
          <td>0.179338</td>
          <td>26.606678</td>
          <td>0.141947</td>
          <td>26.335644</td>
          <td>0.181181</td>
          <td>25.966252</td>
          <td>0.244844</td>
          <td>24.937319</td>
          <td>0.224069</td>
          <td>0.080286</td>
          <td>0.052690</td>
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
