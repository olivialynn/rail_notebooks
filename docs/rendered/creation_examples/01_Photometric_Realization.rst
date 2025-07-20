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

    <pzflow.flow.Flow at 0x7fdccc85b100>



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
    0      23.994413  0.011232  0.010905  
    1      25.391064  0.059307  0.033372  
    2      24.304707  0.036791  0.036785  
    3      25.291103  0.126672  0.079539  
    4      25.096743  0.164548  0.124041  
    ...          ...       ...       ...  
    99995  24.737946  0.148200  0.113880  
    99996  24.224169  0.169202  0.098963  
    99997  25.613836  0.068693  0.040611  
    99998  25.274899  0.020049  0.014356  
    99999  25.699642  0.065389  0.064069  
    
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
          <td>26.940925</td>
          <td>0.525677</td>
          <td>26.720963</td>
          <td>0.167354</td>
          <td>25.961321</td>
          <td>0.076199</td>
          <td>25.176076</td>
          <td>0.062066</td>
          <td>24.681665</td>
          <td>0.076665</td>
          <td>23.808533</td>
          <td>0.079853</td>
          <td>0.011232</td>
          <td>0.010905</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.051008</td>
          <td>0.569209</td>
          <td>27.228559</td>
          <td>0.255858</td>
          <td>26.585519</td>
          <td>0.131607</td>
          <td>26.408745</td>
          <td>0.181724</td>
          <td>25.937836</td>
          <td>0.226385</td>
          <td>26.909084</td>
          <td>0.914321</td>
          <td>0.059307</td>
          <td>0.033372</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.881346</td>
          <td>2.535743</td>
          <td>28.917331</td>
          <td>0.884061</td>
          <td>28.306111</td>
          <td>0.528795</td>
          <td>26.350953</td>
          <td>0.173030</td>
          <td>24.870792</td>
          <td>0.090572</td>
          <td>24.344232</td>
          <td>0.127622</td>
          <td>0.036791</td>
          <td>0.036785</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.644992</td>
          <td>0.740887</td>
          <td>27.347480</td>
          <td>0.250628</td>
          <td>26.073293</td>
          <td>0.136394</td>
          <td>25.415518</td>
          <td>0.145519</td>
          <td>24.802157</td>
          <td>0.188862</td>
          <td>0.126672</td>
          <td>0.079539</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.949744</td>
          <td>0.529067</td>
          <td>26.489314</td>
          <td>0.137222</td>
          <td>25.894373</td>
          <td>0.071819</td>
          <td>25.530049</td>
          <td>0.084880</td>
          <td>25.505925</td>
          <td>0.157254</td>
          <td>24.923211</td>
          <td>0.209086</td>
          <td>0.164548</td>
          <td>0.124041</td>
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
          <td>26.255436</td>
          <td>0.112041</td>
          <td>25.467860</td>
          <td>0.049198</td>
          <td>25.162723</td>
          <td>0.061335</td>
          <td>24.794479</td>
          <td>0.084688</td>
          <td>24.833243</td>
          <td>0.193877</td>
          <td>0.148200</td>
          <td>0.113880</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.965017</td>
          <td>0.205663</td>
          <td>25.924259</td>
          <td>0.073743</td>
          <td>25.170106</td>
          <td>0.061738</td>
          <td>24.778007</td>
          <td>0.083468</td>
          <td>24.077955</td>
          <td>0.101198</td>
          <td>0.169202</td>
          <td>0.098963</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.612506</td>
          <td>0.411203</td>
          <td>26.546285</td>
          <td>0.144120</td>
          <td>26.566853</td>
          <td>0.129498</td>
          <td>26.658322</td>
          <td>0.224062</td>
          <td>25.984419</td>
          <td>0.235295</td>
          <td>25.937028</td>
          <td>0.468703</td>
          <td>0.068693</td>
          <td>0.040611</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.860706</td>
          <td>0.495620</td>
          <td>26.245958</td>
          <td>0.111120</td>
          <td>25.978656</td>
          <td>0.077374</td>
          <td>25.781853</td>
          <td>0.105878</td>
          <td>25.740729</td>
          <td>0.191964</td>
          <td>25.332648</td>
          <td>0.292776</td>
          <td>0.020049</td>
          <td>0.014356</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.272850</td>
          <td>0.315275</td>
          <td>26.793414</td>
          <td>0.177977</td>
          <td>26.362848</td>
          <td>0.108446</td>
          <td>25.979029</td>
          <td>0.125711</td>
          <td>26.172930</td>
          <td>0.274639</td>
          <td>25.759419</td>
          <td>0.409714</td>
          <td>0.065389</td>
          <td>0.064069</td>
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
          <td>26.869523</td>
          <td>0.217608</td>
          <td>26.067525</td>
          <td>0.098422</td>
          <td>25.133898</td>
          <td>0.070907</td>
          <td>24.578987</td>
          <td>0.082399</td>
          <td>23.974602</td>
          <td>0.109222</td>
          <td>0.011232</td>
          <td>0.010905</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.978037</td>
          <td>0.239594</td>
          <td>26.818590</td>
          <td>0.189402</td>
          <td>25.760744</td>
          <td>0.123834</td>
          <td>26.028969</td>
          <td>0.285652</td>
          <td>25.720689</td>
          <td>0.462642</td>
          <td>0.059307</td>
          <td>0.033372</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.714566</td>
          <td>2.345841</td>
          <td>27.321064</td>
          <td>0.286147</td>
          <td>25.639897</td>
          <td>0.111109</td>
          <td>25.124689</td>
          <td>0.133283</td>
          <td>24.191729</td>
          <td>0.132454</td>
          <td>0.036791</td>
          <td>0.036785</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.755666</td>
          <td>0.518064</td>
          <td>28.144206</td>
          <td>0.602540</td>
          <td>27.478637</td>
          <td>0.334179</td>
          <td>25.939075</td>
          <td>0.148705</td>
          <td>25.684282</td>
          <td>0.221082</td>
          <td>25.286162</td>
          <td>0.339642</td>
          <td>0.126672</td>
          <td>0.079539</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.560091</td>
          <td>0.206072</td>
          <td>26.164076</td>
          <td>0.126627</td>
          <td>25.886081</td>
          <td>0.089719</td>
          <td>25.798398</td>
          <td>0.135889</td>
          <td>25.216115</td>
          <td>0.153291</td>
          <td>24.870015</td>
          <td>0.249903</td>
          <td>0.164548</td>
          <td>0.124041</td>
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
          <td>29.237268</td>
          <td>2.128300</td>
          <td>26.285272</td>
          <td>0.139192</td>
          <td>25.405850</td>
          <td>0.058038</td>
          <td>25.095958</td>
          <td>0.072650</td>
          <td>24.794752</td>
          <td>0.105264</td>
          <td>24.718469</td>
          <td>0.218089</td>
          <td>0.148200</td>
          <td>0.113880</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.145686</td>
          <td>0.692796</td>
          <td>26.981665</td>
          <td>0.251341</td>
          <td>25.954447</td>
          <td>0.094650</td>
          <td>25.251690</td>
          <td>0.083752</td>
          <td>24.765080</td>
          <td>0.103041</td>
          <td>24.150198</td>
          <td>0.135208</td>
          <td>0.169202</td>
          <td>0.098963</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.262058</td>
          <td>0.350029</td>
          <td>26.373836</td>
          <td>0.144288</td>
          <td>26.632603</td>
          <td>0.162196</td>
          <td>26.554485</td>
          <td>0.243466</td>
          <td>25.865326</td>
          <td>0.250644</td>
          <td>26.085723</td>
          <td>0.604995</td>
          <td>0.068693</td>
          <td>0.040611</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.424473</td>
          <td>0.394612</td>
          <td>26.281368</td>
          <td>0.132102</td>
          <td>26.008548</td>
          <td>0.093517</td>
          <td>25.933223</td>
          <td>0.142751</td>
          <td>25.629327</td>
          <td>0.204181</td>
          <td>25.029187</td>
          <td>0.267271</td>
          <td>0.020049</td>
          <td>0.014356</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.189166</td>
          <td>1.278607</td>
          <td>26.779509</td>
          <td>0.204226</td>
          <td>26.418887</td>
          <td>0.135448</td>
          <td>26.529737</td>
          <td>0.239329</td>
          <td>26.252388</td>
          <td>0.343443</td>
          <td>28.200939</td>
          <td>1.989710</td>
          <td>0.065389</td>
          <td>0.064069</td>
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
          <td>27.281346</td>
          <td>0.669665</td>
          <td>26.704330</td>
          <td>0.165231</td>
          <td>26.130993</td>
          <td>0.088644</td>
          <td>25.171159</td>
          <td>0.061905</td>
          <td>24.651224</td>
          <td>0.074756</td>
          <td>24.047935</td>
          <td>0.098742</td>
          <td>0.011232</td>
          <td>0.010905</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.427164</td>
          <td>0.307928</td>
          <td>26.745191</td>
          <td>0.155537</td>
          <td>26.166994</td>
          <td>0.152508</td>
          <td>25.942888</td>
          <td>0.233905</td>
          <td>24.986021</td>
          <td>0.226978</td>
          <td>0.059307</td>
          <td>0.033372</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.344596</td>
          <td>0.705394</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.011243</td>
          <td>0.431095</td>
          <td>25.953517</td>
          <td>0.125282</td>
          <td>25.100386</td>
          <td>0.112749</td>
          <td>24.082354</td>
          <td>0.103506</td>
          <td>0.036791</td>
          <td>0.036785</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.657953</td>
          <td>0.399362</td>
          <td>27.099941</td>
          <td>0.230712</td>
          <td>26.258174</td>
          <td>0.182188</td>
          <td>25.381014</td>
          <td>0.160269</td>
          <td>25.277771</td>
          <td>0.316548</td>
          <td>0.126672</td>
          <td>0.079539</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.952442</td>
          <td>0.606563</td>
          <td>26.257362</td>
          <td>0.136612</td>
          <td>25.850621</td>
          <td>0.086494</td>
          <td>25.692399</td>
          <td>0.123289</td>
          <td>25.409705</td>
          <td>0.179835</td>
          <td>24.824228</td>
          <td>0.239396</td>
          <td>0.164548</td>
          <td>0.124041</td>
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
          <td>26.523368</td>
          <td>0.433856</td>
          <td>26.360486</td>
          <td>0.145056</td>
          <td>25.357309</td>
          <td>0.054123</td>
          <td>25.085971</td>
          <td>0.070052</td>
          <td>24.924781</td>
          <td>0.114851</td>
          <td>24.848785</td>
          <td>0.236843</td>
          <td>0.148200</td>
          <td>0.113880</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.353096</td>
          <td>0.383697</td>
          <td>26.760633</td>
          <td>0.205615</td>
          <td>26.163724</td>
          <td>0.111311</td>
          <td>25.259038</td>
          <td>0.082463</td>
          <td>24.986528</td>
          <td>0.122398</td>
          <td>24.281665</td>
          <td>0.148226</td>
          <td>0.169202</td>
          <td>0.098963</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.290146</td>
          <td>0.688392</td>
          <td>26.464723</td>
          <td>0.139153</td>
          <td>26.272037</td>
          <td>0.104350</td>
          <td>26.236561</td>
          <td>0.163651</td>
          <td>25.955121</td>
          <td>0.238695</td>
          <td>25.616250</td>
          <td>0.380835</td>
          <td>0.068693</td>
          <td>0.040611</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.222074</td>
          <td>0.303516</td>
          <td>26.085393</td>
          <td>0.096918</td>
          <td>26.096898</td>
          <td>0.086235</td>
          <td>25.720513</td>
          <td>0.100777</td>
          <td>25.464568</td>
          <td>0.152390</td>
          <td>25.070079</td>
          <td>0.237205</td>
          <td>0.020049</td>
          <td>0.014356</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.342227</td>
          <td>0.344683</td>
          <td>26.708076</td>
          <td>0.173234</td>
          <td>26.701422</td>
          <td>0.153298</td>
          <td>26.120875</td>
          <td>0.150167</td>
          <td>25.840271</td>
          <td>0.219636</td>
          <td>25.604939</td>
          <td>0.381943</td>
          <td>0.065389</td>
          <td>0.064069</td>
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
