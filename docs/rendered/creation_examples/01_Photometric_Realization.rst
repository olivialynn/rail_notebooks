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

    <pzflow.flow.Flow at 0x7f292a3389d0>



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
    0      23.994413  0.021864  0.018594  
    1      25.391064  0.169711  0.164853  
    2      24.304707  0.034628  0.020355  
    3      25.291103  0.115413  0.069506  
    4      25.096743  0.113845  0.082413  
    ...          ...       ...       ...  
    99995  24.737946  0.234176  0.201287  
    99996  24.224169  0.077765  0.048685  
    99997  25.613836  0.005522  0.004535  
    99998  25.274899  0.042616  0.025420  
    99999  25.699642  0.023696  0.015643  
    
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
          <td>27.972984</td>
          <td>1.041717</td>
          <td>26.609467</td>
          <td>0.152152</td>
          <td>26.096039</td>
          <td>0.085815</td>
          <td>25.184701</td>
          <td>0.062542</td>
          <td>24.740740</td>
          <td>0.080770</td>
          <td>24.184947</td>
          <td>0.111118</td>
          <td>0.021864</td>
          <td>0.018594</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.360441</td>
          <td>2.077709</td>
          <td>27.304069</td>
          <td>0.272130</td>
          <td>26.535641</td>
          <td>0.126043</td>
          <td>26.363343</td>
          <td>0.174860</td>
          <td>25.724259</td>
          <td>0.189317</td>
          <td>25.453339</td>
          <td>0.322511</td>
          <td>0.169711</td>
          <td>0.164853</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.976912</td>
          <td>1.760055</td>
          <td>28.605390</td>
          <td>0.721487</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.223973</td>
          <td>0.155264</td>
          <td>25.144019</td>
          <td>0.115041</td>
          <td>24.331863</td>
          <td>0.126261</td>
          <td>0.034628</td>
          <td>0.020355</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.189100</td>
          <td>0.539200</td>
          <td>27.317592</td>
          <td>0.244541</td>
          <td>26.340008</td>
          <td>0.171427</td>
          <td>25.322593</td>
          <td>0.134318</td>
          <td>25.354904</td>
          <td>0.298073</td>
          <td>0.115413</td>
          <td>0.069506</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.838418</td>
          <td>0.221283</td>
          <td>26.019538</td>
          <td>0.091153</td>
          <td>25.955629</td>
          <td>0.075816</td>
          <td>25.678037</td>
          <td>0.096676</td>
          <td>25.682042</td>
          <td>0.182681</td>
          <td>25.364984</td>
          <td>0.300499</td>
          <td>0.113845</td>
          <td>0.082413</td>
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
          <td>27.338028</td>
          <td>0.695486</td>
          <td>26.418791</td>
          <td>0.129113</td>
          <td>25.394827</td>
          <td>0.046109</td>
          <td>25.134149</td>
          <td>0.059800</td>
          <td>24.847055</td>
          <td>0.088701</td>
          <td>24.808968</td>
          <td>0.189950</td>
          <td>0.234176</td>
          <td>0.201287</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.561201</td>
          <td>0.145980</td>
          <td>25.949942</td>
          <td>0.075436</td>
          <td>25.179243</td>
          <td>0.062240</td>
          <td>24.875818</td>
          <td>0.090973</td>
          <td>24.449449</td>
          <td>0.139771</td>
          <td>0.077765</td>
          <td>0.048685</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.990726</td>
          <td>0.545041</td>
          <td>26.642226</td>
          <td>0.156479</td>
          <td>26.559019</td>
          <td>0.128622</td>
          <td>26.322304</td>
          <td>0.168864</td>
          <td>25.656780</td>
          <td>0.178814</td>
          <td>25.180378</td>
          <td>0.258695</td>
          <td>0.005522</td>
          <td>0.004535</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.143029</td>
          <td>0.284059</td>
          <td>26.368142</td>
          <td>0.123572</td>
          <td>26.076798</td>
          <td>0.084373</td>
          <td>26.261682</td>
          <td>0.160354</td>
          <td>25.902146</td>
          <td>0.219765</td>
          <td>25.511895</td>
          <td>0.337852</td>
          <td>0.042616</td>
          <td>0.025420</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.983714</td>
          <td>0.249485</td>
          <td>26.628625</td>
          <td>0.154669</td>
          <td>26.731600</td>
          <td>0.149265</td>
          <td>26.227228</td>
          <td>0.155697</td>
          <td>26.181251</td>
          <td>0.276502</td>
          <td>25.667282</td>
          <td>0.381601</td>
          <td>0.023696</td>
          <td>0.015643</td>
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
          <td>26.665009</td>
          <td>0.183443</td>
          <td>25.910571</td>
          <td>0.085830</td>
          <td>25.249473</td>
          <td>0.078611</td>
          <td>24.657547</td>
          <td>0.088387</td>
          <td>24.051526</td>
          <td>0.116910</td>
          <td>0.021864</td>
          <td>0.018594</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.907328</td>
          <td>0.241832</td>
          <td>26.566522</td>
          <td>0.165094</td>
          <td>26.665452</td>
          <td>0.286778</td>
          <td>25.667497</td>
          <td>0.228683</td>
          <td>25.865380</td>
          <td>0.550811</td>
          <td>0.169711</td>
          <td>0.164853</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.078694</td>
          <td>0.638848</td>
          <td>28.498734</td>
          <td>0.750331</td>
          <td>28.146616</td>
          <td>0.539393</td>
          <td>26.192854</td>
          <td>0.178514</td>
          <td>25.000181</td>
          <td>0.119432</td>
          <td>24.228437</td>
          <td>0.136465</td>
          <td>0.034628</td>
          <td>0.020355</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.637371</td>
          <td>0.472670</td>
          <td>28.422418</td>
          <td>0.726415</td>
          <td>28.376217</td>
          <td>0.648838</td>
          <td>25.806401</td>
          <td>0.131748</td>
          <td>25.356012</td>
          <td>0.166603</td>
          <td>26.606557</td>
          <td>0.870634</td>
          <td>0.115413</td>
          <td>0.069506</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.314424</td>
          <td>0.370126</td>
          <td>26.101952</td>
          <td>0.116299</td>
          <td>25.814383</td>
          <td>0.081353</td>
          <td>25.680636</td>
          <td>0.118456</td>
          <td>25.656927</td>
          <td>0.215282</td>
          <td>24.990146</td>
          <td>0.266758</td>
          <td>0.113845</td>
          <td>0.082413</td>
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
          <td>27.011950</td>
          <td>0.663848</td>
          <td>26.674074</td>
          <td>0.208586</td>
          <td>25.449805</td>
          <td>0.065669</td>
          <td>25.080087</td>
          <td>0.078124</td>
          <td>24.883948</td>
          <td>0.123599</td>
          <td>24.800858</td>
          <td>0.253032</td>
          <td>0.234176</td>
          <td>0.201287</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.754918</td>
          <td>0.997609</td>
          <td>26.909433</td>
          <td>0.227591</td>
          <td>26.141806</td>
          <td>0.106474</td>
          <td>25.153798</td>
          <td>0.073203</td>
          <td>24.935888</td>
          <td>0.114212</td>
          <td>24.650123</td>
          <td>0.197718</td>
          <td>0.077765</td>
          <td>0.048685</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.919173</td>
          <td>1.091515</td>
          <td>27.014074</td>
          <td>0.245208</td>
          <td>26.666932</td>
          <td>0.165302</td>
          <td>26.238223</td>
          <td>0.185013</td>
          <td>25.499227</td>
          <td>0.182825</td>
          <td>25.042693</td>
          <td>0.269985</td>
          <td>0.005522</td>
          <td>0.004535</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.171556</td>
          <td>0.324418</td>
          <td>26.120204</td>
          <td>0.115202</td>
          <td>26.113993</td>
          <td>0.102894</td>
          <td>25.662620</td>
          <td>0.113284</td>
          <td>25.772572</td>
          <td>0.230760</td>
          <td>24.887029</td>
          <td>0.238556</td>
          <td>0.042616</td>
          <td>0.025420</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.479408</td>
          <td>0.411711</td>
          <td>26.924583</td>
          <td>0.227982</td>
          <td>26.609459</td>
          <td>0.157582</td>
          <td>26.494904</td>
          <td>0.229670</td>
          <td>25.641104</td>
          <td>0.206272</td>
          <td>25.524952</td>
          <td>0.396379</td>
          <td>0.023696</td>
          <td>0.015643</td>
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
          <td>27.853964</td>
          <td>0.972471</td>
          <td>26.456470</td>
          <td>0.134022</td>
          <td>26.124747</td>
          <td>0.088504</td>
          <td>25.160593</td>
          <td>0.061582</td>
          <td>24.772177</td>
          <td>0.083505</td>
          <td>23.996181</td>
          <td>0.094743</td>
          <td>0.021864</td>
          <td>0.018594</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.689722</td>
          <td>0.460354</td>
          <td>26.503710</td>
          <td>0.161312</td>
          <td>26.459073</td>
          <td>0.249715</td>
          <td>25.709824</td>
          <td>0.243899</td>
          <td>25.901296</td>
          <td>0.580153</td>
          <td>0.169711</td>
          <td>0.164853</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.218025</td>
          <td>0.554866</td>
          <td>29.058829</td>
          <td>0.889691</td>
          <td>25.899917</td>
          <td>0.118689</td>
          <td>24.862379</td>
          <td>0.090883</td>
          <td>24.339516</td>
          <td>0.128519</td>
          <td>0.034628</td>
          <td>0.020355</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.953155</td>
          <td>1.084214</td>
          <td>30.566253</td>
          <td>2.164177</td>
          <td>27.264242</td>
          <td>0.258796</td>
          <td>26.516547</td>
          <td>0.221464</td>
          <td>25.594511</td>
          <td>0.188123</td>
          <td>25.147251</td>
          <td>0.279170</td>
          <td>0.115413</td>
          <td>0.069506</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.796336</td>
          <td>0.231404</td>
          <td>26.236490</td>
          <td>0.122083</td>
          <td>26.006272</td>
          <td>0.089186</td>
          <td>25.599663</td>
          <td>0.101986</td>
          <td>25.768419</td>
          <td>0.219672</td>
          <td>25.006764</td>
          <td>0.251242</td>
          <td>0.113845</td>
          <td>0.082413</td>
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
          <td>26.246979</td>
          <td>0.160006</td>
          <td>25.553196</td>
          <td>0.080117</td>
          <td>24.955215</td>
          <td>0.078099</td>
          <td>24.801276</td>
          <td>0.127824</td>
          <td>25.020222</td>
          <td>0.333684</td>
          <td>0.234176</td>
          <td>0.201287</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.310683</td>
          <td>0.336178</td>
          <td>26.776788</td>
          <td>0.183582</td>
          <td>26.080173</td>
          <td>0.089291</td>
          <td>25.196531</td>
          <td>0.066885</td>
          <td>24.869765</td>
          <td>0.095464</td>
          <td>24.607352</td>
          <td>0.168920</td>
          <td>0.077765</td>
          <td>0.048685</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.482941</td>
          <td>0.766532</td>
          <td>26.739398</td>
          <td>0.170049</td>
          <td>26.336322</td>
          <td>0.105998</td>
          <td>26.104123</td>
          <td>0.140120</td>
          <td>25.584569</td>
          <td>0.168229</td>
          <td>25.567681</td>
          <td>0.353157</td>
          <td>0.005522</td>
          <td>0.004535</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.298567</td>
          <td>0.325141</td>
          <td>26.106405</td>
          <td>0.099775</td>
          <td>26.329159</td>
          <td>0.107032</td>
          <td>25.993454</td>
          <td>0.129470</td>
          <td>25.587778</td>
          <td>0.171328</td>
          <td>24.987676</td>
          <td>0.224225</td>
          <td>0.042616</td>
          <td>0.025420</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.495717</td>
          <td>0.775064</td>
          <td>26.848699</td>
          <td>0.187347</td>
          <td>26.690156</td>
          <td>0.144811</td>
          <td>26.488624</td>
          <td>0.195471</td>
          <td>25.837222</td>
          <td>0.209251</td>
          <td>26.458089</td>
          <td>0.683904</td>
          <td>0.023696</td>
          <td>0.015643</td>
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
