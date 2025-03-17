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

    <pzflow.flow.Flow at 0x7f87900e7910>



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
          <td>27.065601</td>
          <td>0.575181</td>
          <td>26.797328</td>
          <td>0.178568</td>
          <td>25.995232</td>
          <td>0.078515</td>
          <td>25.255743</td>
          <td>0.066607</td>
          <td>24.750057</td>
          <td>0.081436</td>
          <td>23.894468</td>
          <td>0.086137</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.280514</td>
          <td>0.668681</td>
          <td>27.227407</td>
          <td>0.255617</td>
          <td>26.926086</td>
          <td>0.176226</td>
          <td>26.451809</td>
          <td>0.188462</td>
          <td>25.471114</td>
          <td>0.152634</td>
          <td>25.132315</td>
          <td>0.248692</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.063354</td>
          <td>0.574258</td>
          <td>28.404588</td>
          <td>0.628660</td>
          <td>27.947502</td>
          <td>0.404206</td>
          <td>26.103483</td>
          <td>0.139993</td>
          <td>25.220971</td>
          <td>0.123002</td>
          <td>24.293407</td>
          <td>0.122118</td>
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
          <td>26.992404</td>
          <td>0.186406</td>
          <td>26.499908</td>
          <td>0.196258</td>
          <td>25.473147</td>
          <td>0.152901</td>
          <td>25.539728</td>
          <td>0.345360</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.352820</td>
          <td>0.335948</td>
          <td>26.004634</td>
          <td>0.089968</td>
          <td>25.778855</td>
          <td>0.064835</td>
          <td>25.713326</td>
          <td>0.099714</td>
          <td>25.404093</td>
          <td>0.144096</td>
          <td>25.182287</td>
          <td>0.259100</td>
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
          <td>30.242993</td>
          <td>2.867193</td>
          <td>26.338708</td>
          <td>0.120456</td>
          <td>25.438838</td>
          <td>0.047946</td>
          <td>25.034059</td>
          <td>0.054716</td>
          <td>25.034572</td>
          <td>0.104561</td>
          <td>24.395147</td>
          <td>0.133371</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.283568</td>
          <td>0.670085</td>
          <td>26.606968</td>
          <td>0.151827</td>
          <td>26.099176</td>
          <td>0.086052</td>
          <td>25.386056</td>
          <td>0.074750</td>
          <td>24.751546</td>
          <td>0.081543</td>
          <td>24.476030</td>
          <td>0.143008</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.294898</td>
          <td>0.320862</td>
          <td>26.639026</td>
          <td>0.156052</td>
          <td>26.283342</td>
          <td>0.101161</td>
          <td>26.087811</td>
          <td>0.138114</td>
          <td>25.778003</td>
          <td>0.198083</td>
          <td>25.395811</td>
          <td>0.308027</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.902060</td>
          <td>0.510942</td>
          <td>26.135819</td>
          <td>0.100932</td>
          <td>26.162699</td>
          <td>0.090999</td>
          <td>26.061837</td>
          <td>0.135051</td>
          <td>25.547713</td>
          <td>0.162971</td>
          <td>25.695625</td>
          <td>0.390073</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.312623</td>
          <td>0.683552</td>
          <td>26.555068</td>
          <td>0.145213</td>
          <td>26.525757</td>
          <td>0.124967</td>
          <td>26.125185</td>
          <td>0.142635</td>
          <td>25.883344</td>
          <td>0.216349</td>
          <td>26.272180</td>
          <td>0.598255</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.877521</td>
          <td>0.218990</td>
          <td>26.085996</td>
          <td>0.099989</td>
          <td>25.200503</td>
          <td>0.075177</td>
          <td>24.727724</td>
          <td>0.093880</td>
          <td>24.134948</td>
          <td>0.125520</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.975448</td>
          <td>0.593266</td>
          <td>27.452661</td>
          <td>0.349178</td>
          <td>26.365154</td>
          <td>0.127548</td>
          <td>26.471169</td>
          <td>0.224941</td>
          <td>25.847130</td>
          <td>0.244515</td>
          <td>25.676822</td>
          <td>0.444648</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.748303</td>
          <td>1.699639</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.025216</td>
          <td>0.501672</td>
          <td>26.191588</td>
          <td>0.181841</td>
          <td>24.790239</td>
          <td>0.101400</td>
          <td>24.260611</td>
          <td>0.143099</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.739658</td>
          <td>0.914502</td>
          <td>27.474582</td>
          <td>0.342445</td>
          <td>26.820649</td>
          <td>0.318842</td>
          <td>25.877424</td>
          <td>0.266797</td>
          <td>24.964678</td>
          <td>0.270051</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.124249</td>
          <td>0.311599</td>
          <td>26.297979</td>
          <td>0.133929</td>
          <td>25.922105</td>
          <td>0.086615</td>
          <td>25.697821</td>
          <td>0.116353</td>
          <td>25.403335</td>
          <td>0.168579</td>
          <td>24.695033</td>
          <td>0.202565</td>
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
          <td>26.310618</td>
          <td>0.137905</td>
          <td>25.419730</td>
          <td>0.056717</td>
          <td>25.022183</td>
          <td>0.065629</td>
          <td>24.759268</td>
          <td>0.098559</td>
          <td>24.691836</td>
          <td>0.206157</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.921480</td>
          <td>0.227945</td>
          <td>26.126108</td>
          <td>0.103993</td>
          <td>25.326413</td>
          <td>0.084374</td>
          <td>24.856741</td>
          <td>0.105550</td>
          <td>24.304825</td>
          <td>0.145959</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.754515</td>
          <td>0.232741</td>
          <td>26.539203</td>
          <td>0.166523</td>
          <td>26.292695</td>
          <td>0.121276</td>
          <td>26.217864</td>
          <td>0.184176</td>
          <td>25.494462</td>
          <td>0.184333</td>
          <td>25.572621</td>
          <td>0.415404</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.382388</td>
          <td>0.389894</td>
          <td>26.205444</td>
          <td>0.127107</td>
          <td>25.962852</td>
          <td>0.092615</td>
          <td>25.863376</td>
          <td>0.138627</td>
          <td>25.551776</td>
          <td>0.196947</td>
          <td>25.312206</td>
          <td>0.345106</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.563162</td>
          <td>0.441310</td>
          <td>26.545506</td>
          <td>0.167016</td>
          <td>26.521706</td>
          <td>0.147409</td>
          <td>26.122603</td>
          <td>0.169412</td>
          <td>26.211042</td>
          <td>0.331197</td>
          <td>26.039835</td>
          <td>0.585326</td>
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
          <td>27.194679</td>
          <td>0.630128</td>
          <td>26.867747</td>
          <td>0.189541</td>
          <td>25.977352</td>
          <td>0.077296</td>
          <td>25.048543</td>
          <td>0.055432</td>
          <td>24.639350</td>
          <td>0.073861</td>
          <td>23.852330</td>
          <td>0.083009</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.439079</td>
          <td>0.303729</td>
          <td>26.602300</td>
          <td>0.133654</td>
          <td>26.171354</td>
          <td>0.148557</td>
          <td>25.723714</td>
          <td>0.189401</td>
          <td>25.012535</td>
          <td>0.225460</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.573140</td>
          <td>0.745813</td>
          <td>27.742718</td>
          <td>0.370717</td>
          <td>26.001251</td>
          <td>0.139406</td>
          <td>24.829804</td>
          <td>0.094819</td>
          <td>24.746204</td>
          <td>0.195415</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.813047</td>
          <td>1.059816</td>
          <td>28.470442</td>
          <td>0.767668</td>
          <td>27.181869</td>
          <td>0.269889</td>
          <td>26.824348</td>
          <td>0.318718</td>
          <td>25.206242</td>
          <td>0.151484</td>
          <td>24.980991</td>
          <td>0.272735</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.041333</td>
          <td>0.261772</td>
          <td>26.080140</td>
          <td>0.096248</td>
          <td>25.932864</td>
          <td>0.074412</td>
          <td>25.670520</td>
          <td>0.096184</td>
          <td>25.390691</td>
          <td>0.142642</td>
          <td>25.018980</td>
          <td>0.226775</td>
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
          <td>28.737765</td>
          <td>1.621474</td>
          <td>26.234890</td>
          <td>0.117846</td>
          <td>25.428889</td>
          <td>0.051471</td>
          <td>25.132400</td>
          <td>0.064893</td>
          <td>24.838598</td>
          <td>0.095255</td>
          <td>25.133715</td>
          <td>0.268632</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.871235</td>
          <td>0.987884</td>
          <td>26.950467</td>
          <td>0.205965</td>
          <td>26.056567</td>
          <td>0.084264</td>
          <td>25.251196</td>
          <td>0.067508</td>
          <td>24.879795</td>
          <td>0.092808</td>
          <td>24.390554</td>
          <td>0.135098</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.738807</td>
          <td>0.466072</td>
          <td>26.565760</td>
          <td>0.152798</td>
          <td>26.291400</td>
          <td>0.106965</td>
          <td>25.847922</td>
          <td>0.118012</td>
          <td>25.915991</td>
          <td>0.232809</td>
          <td>25.605053</td>
          <td>0.380264</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.651649</td>
          <td>0.454739</td>
          <td>26.352875</td>
          <td>0.134700</td>
          <td>25.995636</td>
          <td>0.088131</td>
          <td>26.072590</td>
          <td>0.153274</td>
          <td>25.394181</td>
          <td>0.159779</td>
          <td>25.149677</td>
          <td>0.281638</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.032760</td>
          <td>0.266374</td>
          <td>27.010707</td>
          <td>0.220654</td>
          <td>26.561107</td>
          <td>0.133889</td>
          <td>26.080974</td>
          <td>0.142908</td>
          <td>25.911158</td>
          <td>0.229705</td>
          <td>25.826080</td>
          <td>0.446359</td>
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
