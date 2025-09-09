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

    <pzflow.flow.Flow at 0x7ff4279bb550>



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
    0      23.994413  0.040242  0.020371  
    1      25.391064  0.033643  0.025900  
    2      24.304707  0.028321  0.016145  
    3      25.291103  0.029031  0.027229  
    4      25.096743  0.037371  0.021402  
    ...          ...       ...       ...  
    99995  24.737946  0.125270  0.096233  
    99996  24.224169  0.096638  0.088528  
    99997  25.613836  0.157348  0.125100  
    99998  25.274899  0.074241  0.067441  
    99999  25.699642  0.053791  0.029141  
    
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
          <td>26.984305</td>
          <td>0.542513</td>
          <td>26.763898</td>
          <td>0.173576</td>
          <td>26.292420</td>
          <td>0.101969</td>
          <td>25.162565</td>
          <td>0.061326</td>
          <td>24.739479</td>
          <td>0.080680</td>
          <td>23.890663</td>
          <td>0.085849</td>
          <td>0.040242</td>
          <td>0.020371</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.188193</td>
          <td>0.247520</td>
          <td>26.612964</td>
          <td>0.134766</td>
          <td>26.502951</td>
          <td>0.196762</td>
          <td>26.093353</td>
          <td>0.257369</td>
          <td>25.801319</td>
          <td>0.423052</td>
          <td>0.033643</td>
          <td>0.025900</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.124945</td>
          <td>0.462501</td>
          <td>25.840783</td>
          <td>0.111469</td>
          <td>24.997341</td>
          <td>0.101209</td>
          <td>24.328750</td>
          <td>0.125921</td>
          <td>0.028321</td>
          <td>0.016145</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.106624</td>
          <td>0.456184</td>
          <td>26.362564</td>
          <td>0.174745</td>
          <td>25.726963</td>
          <td>0.189749</td>
          <td>25.569922</td>
          <td>0.353665</td>
          <td>0.029031</td>
          <td>0.027229</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.477889</td>
          <td>0.370598</td>
          <td>25.919823</td>
          <td>0.083503</td>
          <td>25.939755</td>
          <td>0.074760</td>
          <td>25.651852</td>
          <td>0.094479</td>
          <td>25.793977</td>
          <td>0.200759</td>
          <td>25.316189</td>
          <td>0.288911</td>
          <td>0.037371</td>
          <td>0.021402</td>
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
          <td>26.178050</td>
          <td>0.292202</td>
          <td>26.475622</td>
          <td>0.135611</td>
          <td>25.417630</td>
          <td>0.047052</td>
          <td>25.110858</td>
          <td>0.058576</td>
          <td>24.778184</td>
          <td>0.083481</td>
          <td>25.146254</td>
          <td>0.251557</td>
          <td>0.125270</td>
          <td>0.096233</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.665447</td>
          <td>0.159616</td>
          <td>26.050213</td>
          <td>0.082418</td>
          <td>25.178448</td>
          <td>0.062196</td>
          <td>24.885670</td>
          <td>0.091764</td>
          <td>24.127809</td>
          <td>0.105710</td>
          <td>0.096638</td>
          <td>0.088528</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.802147</td>
          <td>0.474549</td>
          <td>26.581071</td>
          <td>0.148492</td>
          <td>26.439519</td>
          <td>0.115943</td>
          <td>26.163468</td>
          <td>0.147410</td>
          <td>26.414204</td>
          <td>0.333351</td>
          <td>25.873632</td>
          <td>0.446906</td>
          <td>0.157348</td>
          <td>0.125100</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.736660</td>
          <td>0.451839</td>
          <td>26.398232</td>
          <td>0.126836</td>
          <td>25.928272</td>
          <td>0.074005</td>
          <td>25.995921</td>
          <td>0.127565</td>
          <td>25.847377</td>
          <td>0.209947</td>
          <td>25.146406</td>
          <td>0.251589</td>
          <td>0.074241</td>
          <td>0.067441</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.113046</td>
          <td>0.277246</td>
          <td>26.962580</td>
          <td>0.205244</td>
          <td>26.982032</td>
          <td>0.184779</td>
          <td>26.657292</td>
          <td>0.223871</td>
          <td>25.951264</td>
          <td>0.228922</td>
          <td>26.324870</td>
          <td>0.620882</td>
          <td>0.053791</td>
          <td>0.029141</td>
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
          <td>26.762965</td>
          <td>0.509955</td>
          <td>26.956814</td>
          <td>0.234562</td>
          <td>26.072501</td>
          <td>0.099151</td>
          <td>25.192513</td>
          <td>0.074915</td>
          <td>24.775023</td>
          <td>0.098191</td>
          <td>24.051733</td>
          <td>0.117176</td>
          <td>0.040242</td>
          <td>0.020371</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.014326</td>
          <td>0.535634</td>
          <td>26.491982</td>
          <td>0.142707</td>
          <td>26.013566</td>
          <td>0.153264</td>
          <td>25.975542</td>
          <td>0.272338</td>
          <td>25.860126</td>
          <td>0.510963</td>
          <td>0.033643</td>
          <td>0.025900</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.231895</td>
          <td>1.176159</td>
          <td>27.203602</td>
          <td>0.259395</td>
          <td>25.870774</td>
          <td>0.135375</td>
          <td>25.034333</td>
          <td>0.122914</td>
          <td>24.171711</td>
          <td>0.129814</td>
          <td>0.028321</td>
          <td>0.016145</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.227534</td>
          <td>0.707497</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.277432</td>
          <td>0.275718</td>
          <td>26.264700</td>
          <td>0.189687</td>
          <td>25.368833</td>
          <td>0.164066</td>
          <td>25.042151</td>
          <td>0.270537</td>
          <td>0.029031</td>
          <td>0.027229</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.572367</td>
          <td>0.442379</td>
          <td>25.907772</td>
          <td>0.095614</td>
          <td>25.860564</td>
          <td>0.082275</td>
          <td>25.671168</td>
          <td>0.114010</td>
          <td>25.434790</td>
          <td>0.173617</td>
          <td>25.301681</td>
          <td>0.333421</td>
          <td>0.037371</td>
          <td>0.021402</td>
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
          <td>27.734128</td>
          <td>0.999077</td>
          <td>26.296179</td>
          <td>0.138600</td>
          <td>25.439169</td>
          <td>0.058861</td>
          <td>25.233659</td>
          <td>0.080750</td>
          <td>24.916331</td>
          <td>0.115277</td>
          <td>24.700567</td>
          <td>0.211685</td>
          <td>0.125270</td>
          <td>0.096233</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.703492</td>
          <td>3.451529</td>
          <td>26.830544</td>
          <td>0.215771</td>
          <td>26.126719</td>
          <td>0.106587</td>
          <td>25.111160</td>
          <td>0.071555</td>
          <td>24.843650</td>
          <td>0.106892</td>
          <td>24.032368</td>
          <td>0.118182</td>
          <td>0.096638</td>
          <td>0.088528</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.779220</td>
          <td>0.536529</td>
          <td>26.913136</td>
          <td>0.238279</td>
          <td>26.561082</td>
          <td>0.160724</td>
          <td>26.693859</td>
          <td>0.287158</td>
          <td>25.659986</td>
          <td>0.222413</td>
          <td>25.888329</td>
          <td>0.549363</td>
          <td>0.157348</td>
          <td>0.125100</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.530867</td>
          <td>0.869942</td>
          <td>26.198482</td>
          <td>0.124705</td>
          <td>26.037367</td>
          <td>0.097440</td>
          <td>25.693727</td>
          <td>0.117916</td>
          <td>25.550841</td>
          <td>0.194062</td>
          <td>24.832509</td>
          <td>0.230848</td>
          <td>0.074241</td>
          <td>0.067441</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.116991</td>
          <td>0.657440</td>
          <td>26.618926</td>
          <td>0.177183</td>
          <td>26.437910</td>
          <td>0.136646</td>
          <td>26.250640</td>
          <td>0.188128</td>
          <td>25.737453</td>
          <td>0.224588</td>
          <td>26.142632</td>
          <td>0.627416</td>
          <td>0.053791</td>
          <td>0.029141</td>
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
          <td>27.429891</td>
          <td>0.745190</td>
          <td>27.114403</td>
          <td>0.235500</td>
          <td>26.010686</td>
          <td>0.080687</td>
          <td>25.090360</td>
          <td>0.058356</td>
          <td>24.699751</td>
          <td>0.078969</td>
          <td>23.989382</td>
          <td>0.094964</td>
          <td>0.040242</td>
          <td>0.020371</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.516243</td>
          <td>0.788289</td>
          <td>27.729132</td>
          <td>0.385163</td>
          <td>26.624190</td>
          <td>0.137705</td>
          <td>26.360069</td>
          <td>0.176526</td>
          <td>25.731321</td>
          <td>0.192667</td>
          <td>25.987957</td>
          <td>0.492005</td>
          <td>0.033643</td>
          <td>0.025900</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.614191</td>
          <td>0.729214</td>
          <td>27.745768</td>
          <td>0.347670</td>
          <td>25.969586</td>
          <td>0.125614</td>
          <td>24.927700</td>
          <td>0.095898</td>
          <td>24.284227</td>
          <td>0.122043</td>
          <td>0.028321</td>
          <td>0.016145</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.092506</td>
          <td>1.860621</td>
          <td>27.830605</td>
          <td>0.415994</td>
          <td>27.834051</td>
          <td>0.373722</td>
          <td>26.013905</td>
          <td>0.131003</td>
          <td>25.398200</td>
          <td>0.144869</td>
          <td>25.311600</td>
          <td>0.290784</td>
          <td>0.029031</td>
          <td>0.027229</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.483445</td>
          <td>0.375058</td>
          <td>26.365740</td>
          <td>0.124634</td>
          <td>25.985611</td>
          <td>0.078829</td>
          <td>25.654571</td>
          <td>0.095946</td>
          <td>25.354187</td>
          <td>0.139721</td>
          <td>24.969525</td>
          <td>0.220002</td>
          <td>0.037371</td>
          <td>0.021402</td>
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
          <td>28.076038</td>
          <td>1.183659</td>
          <td>26.314502</td>
          <td>0.133741</td>
          <td>25.400136</td>
          <td>0.053609</td>
          <td>25.065951</td>
          <td>0.065528</td>
          <td>24.799563</td>
          <td>0.098255</td>
          <td>24.321886</td>
          <td>0.144917</td>
          <td>0.125270</td>
          <td>0.096233</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.805509</td>
          <td>0.506579</td>
          <td>26.616704</td>
          <td>0.167382</td>
          <td>25.997201</td>
          <td>0.087348</td>
          <td>25.238495</td>
          <td>0.073232</td>
          <td>24.787623</td>
          <td>0.093453</td>
          <td>24.282115</td>
          <td>0.134532</td>
          <td>0.096638</td>
          <td>0.088528</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.353478</td>
          <td>1.420446</td>
          <td>26.910429</td>
          <td>0.235682</td>
          <td>26.625396</td>
          <td>0.168095</td>
          <td>26.388502</td>
          <td>0.221290</td>
          <td>25.747074</td>
          <td>0.236753</td>
          <td>24.795353</td>
          <td>0.231997</td>
          <td>0.157348</td>
          <td>0.125100</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.828630</td>
          <td>0.229050</td>
          <td>26.040856</td>
          <td>0.098206</td>
          <td>25.991036</td>
          <td>0.083414</td>
          <td>25.858568</td>
          <td>0.120968</td>
          <td>25.236229</td>
          <td>0.132735</td>
          <td>25.154683</td>
          <td>0.269404</td>
          <td>0.074241</td>
          <td>0.067441</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.902598</td>
          <td>0.518522</td>
          <td>26.721664</td>
          <td>0.170964</td>
          <td>26.386403</td>
          <td>0.113436</td>
          <td>26.296781</td>
          <td>0.169417</td>
          <td>26.271312</td>
          <td>0.304151</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.053791</td>
          <td>0.029141</td>
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
