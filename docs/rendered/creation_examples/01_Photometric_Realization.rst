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

    <pzflow.flow.Flow at 0x7fe874625120>



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
    0      23.994413  0.053723  0.049548  
    1      25.391064  0.162956  0.110203  
    2      24.304707  0.167919  0.134395  
    3      25.291103  0.184293  0.159354  
    4      25.096743  0.049334  0.029090  
    ...          ...       ...       ...  
    99995  24.737946  0.067553  0.044864  
    99996  24.224169  0.144343  0.115506  
    99997  25.613836  0.084149  0.073328  
    99998  25.274899  0.165270  0.117947  
    99999  25.699642  0.090366  0.051940  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>27.112137</td>
          <td>0.594543</td>
          <td>26.474896</td>
          <td>0.135526</td>
          <td>25.962846</td>
          <td>0.076301</td>
          <td>25.205004</td>
          <td>0.063678</td>
          <td>24.639527</td>
          <td>0.073863</td>
          <td>23.962197</td>
          <td>0.091425</td>
          <td>0.053723</td>
          <td>0.049548</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.279060</td>
          <td>0.266642</td>
          <td>26.615553</td>
          <td>0.135068</td>
          <td>26.298611</td>
          <td>0.165489</td>
          <td>25.910655</td>
          <td>0.221327</td>
          <td>24.843353</td>
          <td>0.195534</td>
          <td>0.162956</td>
          <td>0.110203</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.147534</td>
          <td>1.152625</td>
          <td>28.958072</td>
          <td>0.906937</td>
          <td>27.854021</td>
          <td>0.376015</td>
          <td>26.302873</td>
          <td>0.166091</td>
          <td>25.170535</td>
          <td>0.117727</td>
          <td>24.461492</td>
          <td>0.141229</td>
          <td>0.167919</td>
          <td>0.134395</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.340675</td>
          <td>0.601041</td>
          <td>27.687632</td>
          <td>0.329921</td>
          <td>25.984832</td>
          <td>0.126345</td>
          <td>25.339796</td>
          <td>0.136329</td>
          <td>25.258519</td>
          <td>0.275721</td>
          <td>0.184293</td>
          <td>0.159354</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.410117</td>
          <td>0.351464</td>
          <td>26.107032</td>
          <td>0.098421</td>
          <td>25.961829</td>
          <td>0.076233</td>
          <td>25.645754</td>
          <td>0.093975</td>
          <td>25.292912</td>
          <td>0.130915</td>
          <td>25.338137</td>
          <td>0.294074</td>
          <td>0.049334</td>
          <td>0.029090</td>
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
          <td>26.584278</td>
          <td>0.402396</td>
          <td>26.419437</td>
          <td>0.129185</td>
          <td>25.442081</td>
          <td>0.048085</td>
          <td>25.036302</td>
          <td>0.054825</td>
          <td>24.976985</td>
          <td>0.099420</td>
          <td>24.870165</td>
          <td>0.199992</td>
          <td>0.067553</td>
          <td>0.044864</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.361067</td>
          <td>1.296900</td>
          <td>26.818120</td>
          <td>0.181740</td>
          <td>26.127330</td>
          <td>0.088212</td>
          <td>25.100097</td>
          <td>0.058020</td>
          <td>24.944778</td>
          <td>0.096652</td>
          <td>24.269882</td>
          <td>0.119648</td>
          <td>0.144343</td>
          <td>0.115506</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.900680</td>
          <td>0.997703</td>
          <td>26.687230</td>
          <td>0.162612</td>
          <td>26.322847</td>
          <td>0.104720</td>
          <td>26.383780</td>
          <td>0.177920</td>
          <td>26.024186</td>
          <td>0.243149</td>
          <td>25.439982</td>
          <td>0.319097</td>
          <td>0.084149</td>
          <td>0.073328</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.943734</td>
          <td>0.241420</td>
          <td>26.250185</td>
          <td>0.111530</td>
          <td>26.208769</td>
          <td>0.094758</td>
          <td>25.723380</td>
          <td>0.100596</td>
          <td>25.737490</td>
          <td>0.191441</td>
          <td>25.238392</td>
          <td>0.271244</td>
          <td>0.165270</td>
          <td>0.117947</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.247421</td>
          <td>0.308936</td>
          <td>26.892624</td>
          <td>0.193534</td>
          <td>26.842636</td>
          <td>0.164146</td>
          <td>26.134072</td>
          <td>0.143730</td>
          <td>25.788682</td>
          <td>0.199868</td>
          <td>25.479152</td>
          <td>0.329199</td>
          <td>0.090366</td>
          <td>0.051940</td>
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
          <td>26.299018</td>
          <td>0.359905</td>
          <td>27.001707</td>
          <td>0.244563</td>
          <td>26.145688</td>
          <td>0.106297</td>
          <td>25.023793</td>
          <td>0.064903</td>
          <td>24.580284</td>
          <td>0.083209</td>
          <td>24.059947</td>
          <td>0.118686</td>
          <td>0.053723</td>
          <td>0.049548</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.387458</td>
          <td>0.348584</td>
          <td>27.278033</td>
          <td>0.291158</td>
          <td>26.352495</td>
          <td>0.216330</td>
          <td>25.819974</td>
          <td>0.253162</td>
          <td>24.566780</td>
          <td>0.193075</td>
          <td>0.162956</td>
          <td>0.110203</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.598145</td>
          <td>0.935741</td>
          <td>28.122916</td>
          <td>0.609517</td>
          <td>27.668774</td>
          <td>0.400402</td>
          <td>26.117888</td>
          <td>0.179625</td>
          <td>25.382559</td>
          <td>0.177632</td>
          <td>24.450120</td>
          <td>0.176932</td>
          <td>0.167919</td>
          <td>0.134395</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.734386</td>
          <td>1.737387</td>
          <td>30.075495</td>
          <td>1.876192</td>
          <td>27.001090</td>
          <td>0.238861</td>
          <td>26.222906</td>
          <td>0.199959</td>
          <td>25.445508</td>
          <td>0.190773</td>
          <td>25.343859</td>
          <td>0.373753</td>
          <td>0.184293</td>
          <td>0.159354</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.151641</td>
          <td>0.319624</td>
          <td>26.214447</td>
          <td>0.125174</td>
          <td>26.039183</td>
          <td>0.096500</td>
          <td>25.903227</td>
          <td>0.139750</td>
          <td>25.243778</td>
          <td>0.147823</td>
          <td>24.765697</td>
          <td>0.215993</td>
          <td>0.049334</td>
          <td>0.029090</td>
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
          <td>26.455925</td>
          <td>0.406982</td>
          <td>26.407150</td>
          <td>0.148521</td>
          <td>25.499258</td>
          <td>0.060251</td>
          <td>25.233687</td>
          <td>0.078307</td>
          <td>24.713140</td>
          <td>0.093714</td>
          <td>24.293728</td>
          <td>0.145574</td>
          <td>0.067553</td>
          <td>0.044864</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.736461</td>
          <td>1.712887</td>
          <td>26.919253</td>
          <td>0.237579</td>
          <td>25.841003</td>
          <td>0.085178</td>
          <td>25.347006</td>
          <td>0.090545</td>
          <td>24.812780</td>
          <td>0.106819</td>
          <td>24.255318</td>
          <td>0.147175</td>
          <td>0.144343</td>
          <td>0.115506</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.042077</td>
          <td>0.629585</td>
          <td>27.157636</td>
          <td>0.280546</td>
          <td>26.332368</td>
          <td>0.126513</td>
          <td>26.341111</td>
          <td>0.205923</td>
          <td>26.304026</td>
          <td>0.359819</td>
          <td>25.923547</td>
          <td>0.543288</td>
          <td>0.084149</td>
          <td>0.073328</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.602366</td>
          <td>0.471365</td>
          <td>26.389407</td>
          <td>0.153470</td>
          <td>26.068452</td>
          <td>0.105069</td>
          <td>25.788571</td>
          <td>0.134477</td>
          <td>25.630681</td>
          <td>0.217256</td>
          <td>24.877932</td>
          <td>0.251067</td>
          <td>0.165270</td>
          <td>0.117947</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.382599</td>
          <td>0.791344</td>
          <td>26.474764</td>
          <td>0.158350</td>
          <td>26.268477</td>
          <td>0.119371</td>
          <td>26.273858</td>
          <td>0.194099</td>
          <td>26.343945</td>
          <td>0.370338</td>
          <td>27.292208</td>
          <td>1.290788</td>
          <td>0.090366</td>
          <td>0.051940</td>
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
          <td>32.685260</td>
          <td>5.265450</td>
          <td>26.694979</td>
          <td>0.168600</td>
          <td>26.025194</td>
          <td>0.083504</td>
          <td>25.167828</td>
          <td>0.063942</td>
          <td>24.619539</td>
          <td>0.075170</td>
          <td>23.993028</td>
          <td>0.097409</td>
          <td>0.053723</td>
          <td>0.049548</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.687288</td>
          <td>0.433532</td>
          <td>26.846381</td>
          <td>0.200689</td>
          <td>25.779206</td>
          <td>0.130519</td>
          <td>26.081202</td>
          <td>0.307906</td>
          <td>25.612019</td>
          <td>0.439955</td>
          <td>0.162956</td>
          <td>0.110203</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.131463</td>
          <td>0.693632</td>
          <td>27.555702</td>
          <td>0.402851</td>
          <td>27.865884</td>
          <td>0.467244</td>
          <td>26.103974</td>
          <td>0.178486</td>
          <td>25.022353</td>
          <td>0.131169</td>
          <td>24.337910</td>
          <td>0.161695</td>
          <td>0.167919</td>
          <td>0.134395</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.801740</td>
          <td>1.090436</td>
          <td>30.127798</td>
          <td>1.952750</td>
          <td>27.620949</td>
          <td>0.405728</td>
          <td>26.113508</td>
          <td>0.189373</td>
          <td>25.347289</td>
          <td>0.182166</td>
          <td>25.299921</td>
          <td>0.373914</td>
          <td>0.184293</td>
          <td>0.159354</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.749312</td>
          <td>0.208428</td>
          <td>26.134969</td>
          <td>0.102771</td>
          <td>25.998682</td>
          <td>0.080491</td>
          <td>25.634602</td>
          <td>0.095200</td>
          <td>25.667058</td>
          <td>0.184181</td>
          <td>24.943834</td>
          <td>0.217318</td>
          <td>0.049334</td>
          <td>0.029090</td>
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
          <td>27.340401</td>
          <td>0.712798</td>
          <td>26.345914</td>
          <td>0.125731</td>
          <td>25.488870</td>
          <td>0.052329</td>
          <td>25.030197</td>
          <td>0.057043</td>
          <td>25.020686</td>
          <td>0.107755</td>
          <td>24.813847</td>
          <td>0.198957</td>
          <td>0.067553</td>
          <td>0.044864</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.431960</td>
          <td>2.275066</td>
          <td>26.678311</td>
          <td>0.189633</td>
          <td>25.849778</td>
          <td>0.083436</td>
          <td>25.277529</td>
          <td>0.082713</td>
          <td>25.015809</td>
          <td>0.123947</td>
          <td>24.073209</td>
          <td>0.122217</td>
          <td>0.144343</td>
          <td>0.115506</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.896035</td>
          <td>0.532585</td>
          <td>27.110647</td>
          <td>0.247532</td>
          <td>26.325885</td>
          <td>0.113449</td>
          <td>26.293932</td>
          <td>0.178352</td>
          <td>26.308183</td>
          <td>0.328741</td>
          <td>25.165789</td>
          <td>0.275512</td>
          <td>0.084149</td>
          <td>0.073328</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.373184</td>
          <td>0.393739</td>
          <td>26.424061</td>
          <td>0.156778</td>
          <td>26.096696</td>
          <td>0.106667</td>
          <td>26.087231</td>
          <td>0.172016</td>
          <td>26.221744</td>
          <td>0.347927</td>
          <td>26.026372</td>
          <td>0.601937</td>
          <td>0.165270</td>
          <td>0.117947</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.158970</td>
          <td>0.255397</td>
          <td>26.818155</td>
          <td>0.171642</td>
          <td>27.017283</td>
          <td>0.320521</td>
          <td>26.945365</td>
          <td>0.529751</td>
          <td>25.253656</td>
          <td>0.292934</td>
          <td>0.090366</td>
          <td>0.051940</td>
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
