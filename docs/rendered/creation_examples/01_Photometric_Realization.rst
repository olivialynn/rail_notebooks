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

    <pzflow.flow.Flow at 0x7f2e7c5d1b10>



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
    0      23.994413  0.052955  0.047572  
    1      25.391064  0.118226  0.108790  
    2      24.304707  0.041227  0.030405  
    3      25.291103  0.083379  0.055774  
    4      25.096743  0.025766  0.016235  
    ...          ...       ...       ...  
    99995  24.737946  0.067476  0.048814  
    99996  24.224169  0.136565  0.122281  
    99997  25.613836  0.030968  0.027463  
    99998  25.274899  0.112797  0.076562  
    99999  25.699642  0.073621  0.052796  
    
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
          <td>26.487409</td>
          <td>0.373355</td>
          <td>26.553714</td>
          <td>0.145044</td>
          <td>26.034758</td>
          <td>0.081303</td>
          <td>25.146494</td>
          <td>0.060458</td>
          <td>24.583325</td>
          <td>0.070281</td>
          <td>23.936671</td>
          <td>0.089396</td>
          <td>0.052955</td>
          <td>0.047572</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.473522</td>
          <td>0.761638</td>
          <td>27.832751</td>
          <td>0.413340</td>
          <td>26.860750</td>
          <td>0.166701</td>
          <td>26.072392</td>
          <td>0.136288</td>
          <td>25.974376</td>
          <td>0.233348</td>
          <td>25.250361</td>
          <td>0.273898</td>
          <td>0.118226</td>
          <td>0.108790</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.410210</td>
          <td>0.570107</td>
          <td>25.988328</td>
          <td>0.126729</td>
          <td>25.044903</td>
          <td>0.105510</td>
          <td>24.324504</td>
          <td>0.125458</td>
          <td>0.041227</td>
          <td>0.030405</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.071848</td>
          <td>1.836858</td>
          <td>27.899504</td>
          <td>0.434906</td>
          <td>27.513618</td>
          <td>0.286980</td>
          <td>26.526257</td>
          <td>0.200654</td>
          <td>25.434097</td>
          <td>0.147862</td>
          <td>25.425183</td>
          <td>0.315350</td>
          <td>0.083379</td>
          <td>0.055774</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.268530</td>
          <td>0.314190</td>
          <td>26.130847</td>
          <td>0.100494</td>
          <td>26.023455</td>
          <td>0.080496</td>
          <td>25.688295</td>
          <td>0.097550</td>
          <td>25.358696</td>
          <td>0.138570</td>
          <td>25.501865</td>
          <td>0.335181</td>
          <td>0.025766</td>
          <td>0.016235</td>
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
          <td>29.043676</td>
          <td>1.813933</td>
          <td>26.404988</td>
          <td>0.127580</td>
          <td>25.401189</td>
          <td>0.046370</td>
          <td>25.104994</td>
          <td>0.058272</td>
          <td>24.715285</td>
          <td>0.078976</td>
          <td>25.273574</td>
          <td>0.279112</td>
          <td>0.067476</td>
          <td>0.048814</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.785590</td>
          <td>0.468722</td>
          <td>26.479927</td>
          <td>0.136115</td>
          <td>25.994787</td>
          <td>0.078485</td>
          <td>25.324993</td>
          <td>0.070820</td>
          <td>24.703466</td>
          <td>0.078156</td>
          <td>24.387795</td>
          <td>0.132526</td>
          <td>0.136565</td>
          <td>0.122281</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.837757</td>
          <td>0.184782</td>
          <td>26.297828</td>
          <td>0.102453</td>
          <td>26.659851</td>
          <td>0.224347</td>
          <td>25.883504</td>
          <td>0.216378</td>
          <td>26.396501</td>
          <td>0.652652</td>
          <td>0.030968</td>
          <td>0.027463</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.967324</td>
          <td>0.246150</td>
          <td>26.166633</td>
          <td>0.103688</td>
          <td>26.231684</td>
          <td>0.096683</td>
          <td>25.900612</td>
          <td>0.117433</td>
          <td>25.854022</td>
          <td>0.211116</td>
          <td>24.841167</td>
          <td>0.195175</td>
          <td>0.112797</td>
          <td>0.076562</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>29.564674</td>
          <td>2.254101</td>
          <td>26.734995</td>
          <td>0.169365</td>
          <td>26.543037</td>
          <td>0.126853</td>
          <td>26.289401</td>
          <td>0.164194</td>
          <td>25.865276</td>
          <td>0.213111</td>
          <td>25.355788</td>
          <td>0.298285</td>
          <td>0.073621</td>
          <td>0.052796</td>
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
          <td>26.486185</td>
          <td>0.158568</td>
          <td>25.955700</td>
          <td>0.089949</td>
          <td>25.072115</td>
          <td>0.067708</td>
          <td>24.752288</td>
          <td>0.096746</td>
          <td>24.144381</td>
          <td>0.127649</td>
          <td>0.052955</td>
          <td>0.047572</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.184654</td>
          <td>0.291886</td>
          <td>26.943636</td>
          <td>0.217337</td>
          <td>26.844274</td>
          <td>0.317365</td>
          <td>25.570974</td>
          <td>0.202191</td>
          <td>25.605386</td>
          <td>0.437117</td>
          <td>0.118226</td>
          <td>0.108790</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.723980</td>
          <td>0.495842</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.200735</td>
          <td>0.180017</td>
          <td>25.029070</td>
          <td>0.122673</td>
          <td>24.427442</td>
          <td>0.162164</td>
          <td>0.041227</td>
          <td>0.030405</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.882018</td>
          <td>0.966828</td>
          <td>27.651853</td>
          <td>0.376206</td>
          <td>26.235189</td>
          <td>0.187635</td>
          <td>25.950306</td>
          <td>0.270216</td>
          <td>25.553245</td>
          <td>0.410752</td>
          <td>0.083379</td>
          <td>0.055774</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.172574</td>
          <td>0.324101</td>
          <td>26.158153</td>
          <td>0.118788</td>
          <td>26.236961</td>
          <td>0.114258</td>
          <td>25.660608</td>
          <td>0.112784</td>
          <td>25.857802</td>
          <td>0.246982</td>
          <td>25.336787</td>
          <td>0.342310</td>
          <td>0.025766</td>
          <td>0.016235</td>
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
          <td>27.376941</td>
          <td>0.785538</td>
          <td>26.300819</td>
          <td>0.135612</td>
          <td>25.407019</td>
          <td>0.055552</td>
          <td>25.126765</td>
          <td>0.071293</td>
          <td>24.711432</td>
          <td>0.093630</td>
          <td>24.829470</td>
          <td>0.229115</td>
          <td>0.067476</td>
          <td>0.048814</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.325272</td>
          <td>1.399855</td>
          <td>26.412174</td>
          <td>0.154951</td>
          <td>26.025201</td>
          <td>0.100056</td>
          <td>25.195060</td>
          <td>0.079143</td>
          <td>24.803269</td>
          <td>0.105854</td>
          <td>24.545714</td>
          <td>0.188333</td>
          <td>0.136565</td>
          <td>0.122281</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.911805</td>
          <td>0.262956</td>
          <td>26.775745</td>
          <td>0.201633</td>
          <td>26.323844</td>
          <td>0.123386</td>
          <td>26.126815</td>
          <td>0.168804</td>
          <td>25.706024</td>
          <td>0.218086</td>
          <td>25.701332</td>
          <td>0.454009</td>
          <td>0.030968</td>
          <td>0.027463</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.102381</td>
          <td>0.312711</td>
          <td>26.068717</td>
          <td>0.112794</td>
          <td>25.960128</td>
          <td>0.092311</td>
          <td>25.717011</td>
          <td>0.122024</td>
          <td>25.798099</td>
          <td>0.241596</td>
          <td>25.107211</td>
          <td>0.292806</td>
          <td>0.112797</td>
          <td>0.076562</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.993471</td>
          <td>0.283148</td>
          <td>26.823167</td>
          <td>0.211765</td>
          <td>26.604171</td>
          <td>0.158776</td>
          <td>26.391794</td>
          <td>0.213364</td>
          <td>25.945346</td>
          <td>0.268370</td>
          <td>25.391210</td>
          <td>0.361298</td>
          <td>0.073621</td>
          <td>0.052796</td>
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
          <td>28.147130</td>
          <td>1.169716</td>
          <td>26.661647</td>
          <td>0.163639</td>
          <td>26.110460</td>
          <td>0.089858</td>
          <td>25.203423</td>
          <td>0.065870</td>
          <td>24.701188</td>
          <td>0.080649</td>
          <td>24.062436</td>
          <td>0.103328</td>
          <td>0.052955</td>
          <td>0.047572</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.067552</td>
          <td>0.628257</td>
          <td>27.598314</td>
          <td>0.387349</td>
          <td>26.577094</td>
          <td>0.151260</td>
          <td>26.245938</td>
          <td>0.183957</td>
          <td>26.333463</td>
          <td>0.357562</td>
          <td>25.681356</td>
          <td>0.441298</td>
          <td>0.118226</td>
          <td>0.108790</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.125859</td>
          <td>1.015545</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.068613</td>
          <td>0.138315</td>
          <td>25.038735</td>
          <td>0.106783</td>
          <td>24.105793</td>
          <td>0.105587</td>
          <td>0.041227</td>
          <td>0.030405</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.560712</td>
          <td>0.351476</td>
          <td>27.745500</td>
          <td>0.365482</td>
          <td>26.257868</td>
          <td>0.170461</td>
          <td>26.070918</td>
          <td>0.268014</td>
          <td>25.653846</td>
          <td>0.400117</td>
          <td>0.083379</td>
          <td>0.055774</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.963295</td>
          <td>0.536227</td>
          <td>26.202701</td>
          <td>0.107585</td>
          <td>25.888809</td>
          <td>0.071919</td>
          <td>25.644536</td>
          <td>0.094493</td>
          <td>25.487865</td>
          <td>0.155787</td>
          <td>25.281734</td>
          <td>0.282655</td>
          <td>0.025766</td>
          <td>0.016235</td>
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
          <td>27.531293</td>
          <td>0.809776</td>
          <td>26.418084</td>
          <td>0.134094</td>
          <td>25.361291</td>
          <td>0.046835</td>
          <td>25.156651</td>
          <td>0.063970</td>
          <td>24.857284</td>
          <td>0.093602</td>
          <td>25.052016</td>
          <td>0.243136</td>
          <td>0.067476</td>
          <td>0.048814</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.338299</td>
          <td>0.771682</td>
          <td>26.726257</td>
          <td>0.197136</td>
          <td>26.042119</td>
          <td>0.098633</td>
          <td>25.144449</td>
          <td>0.073419</td>
          <td>24.840723</td>
          <td>0.106244</td>
          <td>24.054632</td>
          <td>0.120056</td>
          <td>0.136565</td>
          <td>0.122281</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.394634</td>
          <td>0.349705</td>
          <td>26.867963</td>
          <td>0.191380</td>
          <td>26.234573</td>
          <td>0.098049</td>
          <td>26.422264</td>
          <td>0.185963</td>
          <td>26.004247</td>
          <td>0.241777</td>
          <td>25.463791</td>
          <td>0.328738</td>
          <td>0.030968</td>
          <td>0.027463</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.766323</td>
          <td>0.494181</td>
          <td>26.017501</td>
          <td>0.100299</td>
          <td>26.054102</td>
          <td>0.092408</td>
          <td>25.823086</td>
          <td>0.123082</td>
          <td>25.837892</td>
          <td>0.231288</td>
          <td>26.611063</td>
          <td>0.822011</td>
          <td>0.112797</td>
          <td>0.076562</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.016672</td>
          <td>0.572369</td>
          <td>26.657140</td>
          <td>0.165708</td>
          <td>26.473288</td>
          <td>0.125761</td>
          <td>26.440064</td>
          <td>0.196709</td>
          <td>25.525629</td>
          <td>0.168279</td>
          <td>26.287127</td>
          <td>0.631473</td>
          <td>0.073621</td>
          <td>0.052796</td>
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
