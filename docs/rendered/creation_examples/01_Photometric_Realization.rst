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

    <pzflow.flow.Flow at 0x7f0424c322f0>



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
    0      23.994413  0.176397  0.159581  
    1      25.391064  0.091655  0.079539  
    2      24.304707  0.000286  0.000211  
    3      25.291103  0.046318  0.044088  
    4      25.096743  0.149654  0.112881  
    ...          ...       ...       ...  
    99995  24.737946  0.122311  0.111573  
    99996  24.224169  0.032730  0.023992  
    99997  25.613836  0.083760  0.043054  
    99998  25.274899  0.043889  0.026492  
    99999  25.699642  0.098630  0.079883  
    
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
          <td>26.863971</td>
          <td>0.496816</td>
          <td>26.804587</td>
          <td>0.179670</td>
          <td>25.966174</td>
          <td>0.076526</td>
          <td>25.154332</td>
          <td>0.060880</td>
          <td>24.758511</td>
          <td>0.082046</td>
          <td>24.238547</td>
          <td>0.116430</td>
          <td>0.176397</td>
          <td>0.159581</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.204813</td>
          <td>1.190419</td>
          <td>27.829831</td>
          <td>0.412417</td>
          <td>26.453664</td>
          <td>0.117380</td>
          <td>26.264639</td>
          <td>0.160760</td>
          <td>26.503826</td>
          <td>0.357757</td>
          <td>25.324233</td>
          <td>0.290794</td>
          <td>0.091655</td>
          <td>0.079539</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.047990</td>
          <td>0.567980</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.448489</td>
          <td>0.585902</td>
          <td>26.137330</td>
          <td>0.144134</td>
          <td>24.886017</td>
          <td>0.091792</td>
          <td>24.441623</td>
          <td>0.138831</td>
          <td>0.000286</td>
          <td>0.000211</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.622915</td>
          <td>0.839473</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.557288</td>
          <td>0.297273</td>
          <td>26.090557</td>
          <td>0.138441</td>
          <td>25.451298</td>
          <td>0.150062</td>
          <td>25.547974</td>
          <td>0.347611</td>
          <td>0.046318</td>
          <td>0.044088</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.021404</td>
          <td>0.557239</td>
          <td>25.897138</td>
          <td>0.081852</td>
          <td>25.925305</td>
          <td>0.073811</td>
          <td>25.554051</td>
          <td>0.086694</td>
          <td>25.317620</td>
          <td>0.133742</td>
          <td>25.009268</td>
          <td>0.224640</td>
          <td>0.149654</td>
          <td>0.112881</td>
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
          <td>26.469124</td>
          <td>0.368075</td>
          <td>26.121012</td>
          <td>0.099633</td>
          <td>25.429555</td>
          <td>0.047553</td>
          <td>25.081831</td>
          <td>0.057087</td>
          <td>24.774283</td>
          <td>0.083195</td>
          <td>24.535370</td>
          <td>0.150491</td>
          <td>0.122311</td>
          <td>0.111573</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.594282</td>
          <td>0.150184</td>
          <td>26.260942</td>
          <td>0.099195</td>
          <td>25.177732</td>
          <td>0.062157</td>
          <td>24.962766</td>
          <td>0.098189</td>
          <td>24.241434</td>
          <td>0.116723</td>
          <td>0.032730</td>
          <td>0.023992</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.987121</td>
          <td>0.543620</td>
          <td>26.628387</td>
          <td>0.154637</td>
          <td>26.216511</td>
          <td>0.095404</td>
          <td>26.053143</td>
          <td>0.134041</td>
          <td>25.964351</td>
          <td>0.231419</td>
          <td>25.778671</td>
          <td>0.415799</td>
          <td>0.083760</td>
          <td>0.043054</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.949246</td>
          <td>0.528875</td>
          <td>26.136592</td>
          <td>0.101000</td>
          <td>26.054361</td>
          <td>0.082720</td>
          <td>25.707998</td>
          <td>0.099249</td>
          <td>25.649396</td>
          <td>0.177698</td>
          <td>25.196339</td>
          <td>0.262095</td>
          <td>0.043889</td>
          <td>0.026492</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.339003</td>
          <td>0.332295</td>
          <td>26.523041</td>
          <td>0.141267</td>
          <td>26.709068</td>
          <td>0.146403</td>
          <td>26.261282</td>
          <td>0.160299</td>
          <td>26.497925</td>
          <td>0.356104</td>
          <td>26.596660</td>
          <td>0.747650</td>
          <td>0.098630</td>
          <td>0.079883</td>
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
          <td>29.323647</td>
          <td>2.227638</td>
          <td>26.694356</td>
          <td>0.202734</td>
          <td>26.063519</td>
          <td>0.107002</td>
          <td>25.161325</td>
          <td>0.079555</td>
          <td>24.755402</td>
          <td>0.104982</td>
          <td>23.782321</td>
          <td>0.100943</td>
          <td>0.176397</td>
          <td>0.159581</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.777853</td>
          <td>1.016916</td>
          <td>27.678565</td>
          <td>0.424041</td>
          <td>26.693354</td>
          <td>0.173100</td>
          <td>26.171904</td>
          <td>0.179214</td>
          <td>25.864602</td>
          <td>0.253730</td>
          <td>25.769958</td>
          <td>0.486963</td>
          <td>0.091655</td>
          <td>0.079539</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.505921</td>
          <td>2.319603</td>
          <td>28.821084</td>
          <td>0.921427</td>
          <td>27.889896</td>
          <td>0.444965</td>
          <td>26.170952</td>
          <td>0.174748</td>
          <td>24.959914</td>
          <td>0.115009</td>
          <td>24.525065</td>
          <td>0.175436</td>
          <td>0.000286</td>
          <td>0.000211</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>31.780454</td>
          <td>4.481250</td>
          <td>30.313224</td>
          <td>2.001098</td>
          <td>27.140356</td>
          <td>0.247455</td>
          <td>26.216239</td>
          <td>0.182845</td>
          <td>25.466681</td>
          <td>0.179029</td>
          <td>25.104130</td>
          <td>0.285639</td>
          <td>0.046318</td>
          <td>0.044088</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>31.877706</td>
          <td>4.619597</td>
          <td>26.149359</td>
          <td>0.123804</td>
          <td>25.874456</td>
          <td>0.087844</td>
          <td>25.716704</td>
          <td>0.125226</td>
          <td>25.323193</td>
          <td>0.166211</td>
          <td>25.207846</td>
          <td>0.325083</td>
          <td>0.149654</td>
          <td>0.112881</td>
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
          <td>26.354465</td>
          <td>0.384988</td>
          <td>26.304724</td>
          <td>0.140111</td>
          <td>25.443436</td>
          <td>0.059318</td>
          <td>24.952675</td>
          <td>0.063245</td>
          <td>24.826221</td>
          <td>0.106982</td>
          <td>24.873062</td>
          <td>0.245178</td>
          <td>0.122311</td>
          <td>0.111573</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.867386</td>
          <td>0.217663</td>
          <td>25.876281</td>
          <td>0.083393</td>
          <td>25.140086</td>
          <td>0.071473</td>
          <td>24.804391</td>
          <td>0.100685</td>
          <td>24.144238</td>
          <td>0.126890</td>
          <td>0.032730</td>
          <td>0.023992</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.882473</td>
          <td>0.560206</td>
          <td>26.967417</td>
          <td>0.238919</td>
          <td>26.435692</td>
          <td>0.137526</td>
          <td>26.552786</td>
          <td>0.244094</td>
          <td>26.040145</td>
          <td>0.290103</td>
          <td>24.979586</td>
          <td>0.260057</td>
          <td>0.083760</td>
          <td>0.043054</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.272329</td>
          <td>0.351363</td>
          <td>26.014523</td>
          <td>0.105094</td>
          <td>26.039670</td>
          <td>0.096435</td>
          <td>25.655650</td>
          <td>0.112630</td>
          <td>25.518611</td>
          <td>0.186627</td>
          <td>24.851482</td>
          <td>0.231708</td>
          <td>0.043889</td>
          <td>0.026492</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.623699</td>
          <td>0.181085</td>
          <td>26.507355</td>
          <td>0.147986</td>
          <td>26.467003</td>
          <td>0.230027</td>
          <td>25.938282</td>
          <td>0.270037</td>
          <td>25.559746</td>
          <td>0.416454</td>
          <td>0.098630</td>
          <td>0.079883</td>
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
          <td>26.389188</td>
          <td>0.416619</td>
          <td>27.095271</td>
          <td>0.290000</td>
          <td>25.986744</td>
          <td>0.103335</td>
          <td>25.247737</td>
          <td>0.088761</td>
          <td>24.672472</td>
          <td>0.100836</td>
          <td>23.891785</td>
          <td>0.114779</td>
          <td>0.176397</td>
          <td>0.159581</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.222842</td>
          <td>0.589523</td>
          <td>26.721280</td>
          <td>0.161633</td>
          <td>26.264374</td>
          <td>0.176204</td>
          <td>25.797122</td>
          <td>0.219421</td>
          <td>25.266809</td>
          <td>0.302620</td>
          <td>0.091655</td>
          <td>0.079539</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.705573</td>
          <td>1.548043</td>
          <td>29.869855</td>
          <td>1.511393</td>
          <td>27.536013</td>
          <td>0.292219</td>
          <td>25.898770</td>
          <td>0.117245</td>
          <td>24.996948</td>
          <td>0.101174</td>
          <td>24.347746</td>
          <td>0.128011</td>
          <td>0.000286</td>
          <td>0.000211</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.047280</td>
          <td>1.834299</td>
          <td>30.074181</td>
          <td>1.689413</td>
          <td>27.087604</td>
          <td>0.207237</td>
          <td>26.205383</td>
          <td>0.157090</td>
          <td>25.411847</td>
          <td>0.148941</td>
          <td>25.198310</td>
          <td>0.269445</td>
          <td>0.046318</td>
          <td>0.044088</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.418347</td>
          <td>0.400624</td>
          <td>26.043724</td>
          <td>0.110345</td>
          <td>26.070545</td>
          <td>0.101655</td>
          <td>25.640552</td>
          <td>0.114122</td>
          <td>25.389050</td>
          <td>0.171416</td>
          <td>24.787306</td>
          <td>0.225227</td>
          <td>0.149654</td>
          <td>0.112881</td>
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
          <td>27.190615</td>
          <td>0.687149</td>
          <td>26.647841</td>
          <td>0.179899</td>
          <td>25.418991</td>
          <td>0.055241</td>
          <td>25.022545</td>
          <td>0.063929</td>
          <td>24.875586</td>
          <td>0.106392</td>
          <td>24.650095</td>
          <td>0.194154</td>
          <td>0.122311</td>
          <td>0.111573</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.715160</td>
          <td>0.894977</td>
          <td>26.618261</td>
          <td>0.154744</td>
          <td>26.161596</td>
          <td>0.091923</td>
          <td>25.227956</td>
          <td>0.065754</td>
          <td>24.796257</td>
          <td>0.085766</td>
          <td>24.124840</td>
          <td>0.106648</td>
          <td>0.032730</td>
          <td>0.023992</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.391675</td>
          <td>1.350053</td>
          <td>26.774005</td>
          <td>0.183508</td>
          <td>26.338600</td>
          <td>0.112224</td>
          <td>26.308290</td>
          <td>0.176555</td>
          <td>26.116044</td>
          <td>0.276088</td>
          <td>24.987861</td>
          <td>0.233058</td>
          <td>0.083760</td>
          <td>0.043054</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.465511</td>
          <td>0.371024</td>
          <td>26.069450</td>
          <td>0.096689</td>
          <td>26.083391</td>
          <td>0.086365</td>
          <td>26.074932</td>
          <td>0.139069</td>
          <td>25.322481</td>
          <td>0.136629</td>
          <td>25.257473</td>
          <td>0.280143</td>
          <td>0.043889</td>
          <td>0.026492</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.866929</td>
          <td>1.025565</td>
          <td>26.562784</td>
          <td>0.159018</td>
          <td>26.472943</td>
          <td>0.131523</td>
          <td>26.031974</td>
          <td>0.145570</td>
          <td>26.010679</td>
          <td>0.263540</td>
          <td>25.349424</td>
          <td>0.325551</td>
          <td>0.098630</td>
          <td>0.079883</td>
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
