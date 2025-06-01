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

    <pzflow.flow.Flow at 0x7f9a4f17c040>



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
    0      23.994413  0.074952  0.066295  
    1      25.391064  0.062846  0.050250  
    2      24.304707  0.124766  0.065394  
    3      25.291103  0.081101  0.065171  
    4      25.096743  0.082377  0.048406  
    ...          ...       ...       ...  
    99995  24.737946  0.121609  0.089435  
    99996  24.224169  0.132202  0.121266  
    99997  25.613836  0.057870  0.040697  
    99998  25.274899  0.022943  0.012316  
    99999  25.699642  0.076977  0.051186  
    
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
          <td>27.098504</td>
          <td>0.588820</td>
          <td>26.790317</td>
          <td>0.177510</td>
          <td>26.015766</td>
          <td>0.079952</td>
          <td>25.173537</td>
          <td>0.061926</td>
          <td>24.714826</td>
          <td>0.078944</td>
          <td>23.948982</td>
          <td>0.090369</td>
          <td>0.074952</td>
          <td>0.066295</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.071017</td>
          <td>0.224673</td>
          <td>26.549021</td>
          <td>0.127513</td>
          <td>26.402857</td>
          <td>0.180820</td>
          <td>26.027205</td>
          <td>0.243755</td>
          <td>25.129059</td>
          <td>0.248027</td>
          <td>0.062846</td>
          <td>0.050250</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.770041</td>
          <td>0.921116</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.882609</td>
          <td>1.414545</td>
          <td>25.992253</td>
          <td>0.127160</td>
          <td>24.806703</td>
          <td>0.085605</td>
          <td>24.212181</td>
          <td>0.113788</td>
          <td>0.124766</td>
          <td>0.065394</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.016038</td>
          <td>0.555091</td>
          <td>29.153144</td>
          <td>1.021638</td>
          <td>27.828434</td>
          <td>0.368592</td>
          <td>26.196429</td>
          <td>0.151641</td>
          <td>25.759819</td>
          <td>0.195076</td>
          <td>25.465062</td>
          <td>0.325534</td>
          <td>0.081101</td>
          <td>0.065171</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.347873</td>
          <td>0.334636</td>
          <td>26.112122</td>
          <td>0.098860</td>
          <td>25.925699</td>
          <td>0.073837</td>
          <td>25.801565</td>
          <td>0.107717</td>
          <td>25.293866</td>
          <td>0.131023</td>
          <td>25.096179</td>
          <td>0.241400</td>
          <td>0.082377</td>
          <td>0.048406</td>
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
          <td>26.499620</td>
          <td>0.138446</td>
          <td>25.345255</td>
          <td>0.044124</td>
          <td>25.047398</td>
          <td>0.055368</td>
          <td>24.756387</td>
          <td>0.081892</td>
          <td>24.640175</td>
          <td>0.164608</td>
          <td>0.121609</td>
          <td>0.089435</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.264905</td>
          <td>0.661537</td>
          <td>26.676266</td>
          <td>0.161097</td>
          <td>25.977211</td>
          <td>0.077276</td>
          <td>25.165105</td>
          <td>0.061465</td>
          <td>24.890923</td>
          <td>0.092189</td>
          <td>24.396665</td>
          <td>0.133546</td>
          <td>0.132202</td>
          <td>0.121266</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.307815</td>
          <td>0.324175</td>
          <td>27.007298</td>
          <td>0.213063</td>
          <td>26.507742</td>
          <td>0.123028</td>
          <td>26.044002</td>
          <td>0.132986</td>
          <td>25.866455</td>
          <td>0.213321</td>
          <td>25.878338</td>
          <td>0.448496</td>
          <td>0.057870</td>
          <td>0.040697</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.014950</td>
          <td>0.255954</td>
          <td>26.292190</td>
          <td>0.115684</td>
          <td>26.186009</td>
          <td>0.092883</td>
          <td>26.075560</td>
          <td>0.136661</td>
          <td>25.573851</td>
          <td>0.166644</td>
          <td>25.164936</td>
          <td>0.255443</td>
          <td>0.022943</td>
          <td>0.012316</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.828185</td>
          <td>0.483828</td>
          <td>26.628033</td>
          <td>0.154590</td>
          <td>26.505009</td>
          <td>0.122737</td>
          <td>26.441746</td>
          <td>0.186867</td>
          <td>26.101585</td>
          <td>0.259110</td>
          <td>26.019492</td>
          <td>0.498320</td>
          <td>0.076977</td>
          <td>0.051186</td>
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
          <td>26.683187</td>
          <td>0.188782</td>
          <td>25.999145</td>
          <td>0.094219</td>
          <td>25.147882</td>
          <td>0.073024</td>
          <td>24.839139</td>
          <td>0.105246</td>
          <td>24.026882</td>
          <td>0.116235</td>
          <td>0.074952</td>
          <td>0.066295</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.527884</td>
          <td>0.865463</td>
          <td>27.448315</td>
          <td>0.350996</td>
          <td>26.656034</td>
          <td>0.165502</td>
          <td>26.161733</td>
          <td>0.175287</td>
          <td>25.870119</td>
          <td>0.251679</td>
          <td>26.754687</td>
          <td>0.941966</td>
          <td>0.062846</td>
          <td>0.050250</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.765153</td>
          <td>1.013437</td>
          <td>31.736890</td>
          <td>3.316187</td>
          <td>28.424835</td>
          <td>0.672313</td>
          <td>26.020433</td>
          <td>0.158782</td>
          <td>25.134003</td>
          <td>0.138072</td>
          <td>24.223741</td>
          <td>0.140030</td>
          <td>0.124766</td>
          <td>0.065394</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.584452</td>
          <td>0.392531</td>
          <td>27.371564</td>
          <td>0.301748</td>
          <td>26.217758</td>
          <td>0.185128</td>
          <td>25.505795</td>
          <td>0.187052</td>
          <td>25.146768</td>
          <td>0.298764</td>
          <td>0.081101</td>
          <td>0.065171</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.202317</td>
          <td>0.334974</td>
          <td>26.189890</td>
          <td>0.123605</td>
          <td>26.015206</td>
          <td>0.095413</td>
          <td>25.657105</td>
          <td>0.114031</td>
          <td>25.735892</td>
          <td>0.226197</td>
          <td>25.312107</td>
          <td>0.339968</td>
          <td>0.082377</td>
          <td>0.048406</td>
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
          <td>26.720448</td>
          <td>0.505097</td>
          <td>26.342094</td>
          <td>0.143747</td>
          <td>25.389056</td>
          <td>0.056108</td>
          <td>24.905389</td>
          <td>0.060186</td>
          <td>24.844668</td>
          <td>0.107929</td>
          <td>24.288177</td>
          <td>0.148738</td>
          <td>0.121609</td>
          <td>0.089435</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.801203</td>
          <td>0.540857</td>
          <td>26.500563</td>
          <td>0.166772</td>
          <td>26.306206</td>
          <td>0.127537</td>
          <td>25.295094</td>
          <td>0.086243</td>
          <td>24.886259</td>
          <td>0.113559</td>
          <td>24.298505</td>
          <td>0.152286</td>
          <td>0.132202</td>
          <td>0.121266</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.840572</td>
          <td>1.046829</td>
          <td>26.496062</td>
          <td>0.159895</td>
          <td>26.639675</td>
          <td>0.162820</td>
          <td>26.236977</td>
          <td>0.186363</td>
          <td>26.101576</td>
          <td>0.303044</td>
          <td>26.256897</td>
          <td>0.680191</td>
          <td>0.057870</td>
          <td>0.040697</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.827901</td>
          <td>0.245194</td>
          <td>26.024894</td>
          <td>0.105736</td>
          <td>26.084090</td>
          <td>0.099934</td>
          <td>25.987090</td>
          <td>0.149534</td>
          <td>25.435666</td>
          <td>0.173412</td>
          <td>25.462824</td>
          <td>0.377688</td>
          <td>0.022943</td>
          <td>0.012316</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.779332</td>
          <td>0.519685</td>
          <td>26.364445</td>
          <td>0.143586</td>
          <td>26.806599</td>
          <td>0.188670</td>
          <td>26.602294</td>
          <td>0.254113</td>
          <td>26.446924</td>
          <td>0.399775</td>
          <td>25.987950</td>
          <td>0.566025</td>
          <td>0.076977</td>
          <td>0.051186</td>
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
          <td>30.515012</td>
          <td>3.170105</td>
          <td>26.343734</td>
          <td>0.127810</td>
          <td>25.971671</td>
          <td>0.081977</td>
          <td>25.166169</td>
          <td>0.065805</td>
          <td>24.725151</td>
          <td>0.084916</td>
          <td>23.953919</td>
          <td>0.096950</td>
          <td>0.074952</td>
          <td>0.066295</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.037702</td>
          <td>0.577500</td>
          <td>27.314900</td>
          <td>0.283977</td>
          <td>26.619586</td>
          <td>0.141248</td>
          <td>26.297962</td>
          <td>0.172608</td>
          <td>25.945625</td>
          <td>0.237044</td>
          <td>25.696352</td>
          <td>0.405498</td>
          <td>0.062846</td>
          <td>0.050250</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.813333</td>
          <td>1.002046</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.726861</td>
          <td>0.377087</td>
          <td>26.330511</td>
          <td>0.190903</td>
          <td>25.247511</td>
          <td>0.140936</td>
          <td>24.192473</td>
          <td>0.125829</td>
          <td>0.124766</td>
          <td>0.065394</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.828053</td>
          <td>2.537165</td>
          <td>28.698003</td>
          <td>0.802342</td>
          <td>27.100497</td>
          <td>0.217852</td>
          <td>26.433283</td>
          <td>0.198691</td>
          <td>25.312116</td>
          <td>0.142324</td>
          <td>25.293905</td>
          <td>0.302775</td>
          <td>0.081101</td>
          <td>0.065171</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.653837</td>
          <td>0.439577</td>
          <td>26.120392</td>
          <td>0.104708</td>
          <td>25.907426</td>
          <td>0.076994</td>
          <td>25.798364</td>
          <td>0.114047</td>
          <td>25.376536</td>
          <td>0.148872</td>
          <td>25.150951</td>
          <td>0.266964</td>
          <td>0.082377</td>
          <td>0.048406</td>
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
          <td>26.496919</td>
          <td>0.409210</td>
          <td>26.410103</td>
          <td>0.143803</td>
          <td>25.465709</td>
          <td>0.056181</td>
          <td>25.079707</td>
          <td>0.065558</td>
          <td>24.782746</td>
          <td>0.095743</td>
          <td>24.668894</td>
          <td>0.192581</td>
          <td>0.121609</td>
          <td>0.089435</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.550870</td>
          <td>0.881785</td>
          <td>26.649601</td>
          <td>0.183769</td>
          <td>25.909156</td>
          <td>0.087198</td>
          <td>25.117658</td>
          <td>0.071220</td>
          <td>24.994392</td>
          <td>0.120688</td>
          <td>24.054850</td>
          <td>0.119295</td>
          <td>0.132202</td>
          <td>0.121266</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.395515</td>
          <td>0.354648</td>
          <td>26.852974</td>
          <td>0.192368</td>
          <td>26.427016</td>
          <td>0.118471</td>
          <td>26.245180</td>
          <td>0.163475</td>
          <td>25.828038</td>
          <td>0.213107</td>
          <td>25.440155</td>
          <td>0.329112</td>
          <td>0.057870</td>
          <td>0.040697</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.733115</td>
          <td>0.451862</td>
          <td>26.195739</td>
          <td>0.106779</td>
          <td>26.163433</td>
          <td>0.091477</td>
          <td>26.031886</td>
          <td>0.132225</td>
          <td>25.724453</td>
          <td>0.190181</td>
          <td>25.299815</td>
          <td>0.286363</td>
          <td>0.022943</td>
          <td>0.012316</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.965600</td>
          <td>0.215282</td>
          <td>26.505359</td>
          <td>0.129547</td>
          <td>25.985433</td>
          <td>0.133710</td>
          <td>25.527080</td>
          <td>0.168798</td>
          <td>25.642169</td>
          <td>0.393313</td>
          <td>0.076977</td>
          <td>0.051186</td>
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
