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

    <pzflow.flow.Flow at 0x7ff4c8f81c00>



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
    0      23.994413  0.103593  0.102018  
    1      25.391064  0.019637  0.015700  
    2      24.304707  0.114550  0.111233  
    3      25.291103  0.027067  0.014640  
    4      25.096743  0.034144  0.020326  
    ...          ...       ...       ...  
    99995  24.737946  0.050456  0.030899  
    99996  24.224169  0.015897  0.013271  
    99997  25.613836  0.018268  0.012769  
    99998  25.274899  0.003071  0.001829  
    99999  25.699642  0.003950  0.002138  
    
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
          <td>27.139835</td>
          <td>0.606297</td>
          <td>26.403897</td>
          <td>0.127460</td>
          <td>26.076514</td>
          <td>0.084352</td>
          <td>25.179680</td>
          <td>0.062264</td>
          <td>24.856754</td>
          <td>0.089461</td>
          <td>24.052409</td>
          <td>0.098959</td>
          <td>0.103593</td>
          <td>0.102018</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.866565</td>
          <td>0.856084</td>
          <td>26.834776</td>
          <td>0.163049</td>
          <td>26.558347</td>
          <td>0.206128</td>
          <td>25.305230</td>
          <td>0.132317</td>
          <td>25.882820</td>
          <td>0.450014</td>
          <td>0.019637</td>
          <td>0.015700</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.216287</td>
          <td>0.639641</td>
          <td>29.308346</td>
          <td>1.118863</td>
          <td>27.634696</td>
          <td>0.316306</td>
          <td>26.098806</td>
          <td>0.139430</td>
          <td>25.154115</td>
          <td>0.116057</td>
          <td>24.362466</td>
          <td>0.129654</td>
          <td>0.114550</td>
          <td>0.111233</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.145485</td>
          <td>1.151286</td>
          <td>30.495701</td>
          <td>2.012017</td>
          <td>27.435287</td>
          <td>0.269299</td>
          <td>26.361430</td>
          <td>0.174577</td>
          <td>25.834908</td>
          <td>0.207768</td>
          <td>25.715763</td>
          <td>0.396187</td>
          <td>0.027067</td>
          <td>0.014640</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.322448</td>
          <td>0.327963</td>
          <td>26.335824</td>
          <td>0.120155</td>
          <td>26.023430</td>
          <td>0.080494</td>
          <td>25.772507</td>
          <td>0.105016</td>
          <td>25.384817</td>
          <td>0.141725</td>
          <td>24.880108</td>
          <td>0.201668</td>
          <td>0.034144</td>
          <td>0.020326</td>
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
          <td>26.454973</td>
          <td>0.133215</td>
          <td>25.482353</td>
          <td>0.049835</td>
          <td>25.033889</td>
          <td>0.054708</td>
          <td>24.802123</td>
          <td>0.085261</td>
          <td>24.512734</td>
          <td>0.147594</td>
          <td>0.050456</td>
          <td>0.030899</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.815631</td>
          <td>0.947410</td>
          <td>27.494674</td>
          <td>0.317308</td>
          <td>26.213133</td>
          <td>0.095122</td>
          <td>25.219859</td>
          <td>0.064523</td>
          <td>24.651036</td>
          <td>0.074618</td>
          <td>24.207429</td>
          <td>0.113317</td>
          <td>0.015897</td>
          <td>0.013271</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.206685</td>
          <td>0.635381</td>
          <td>26.542552</td>
          <td>0.143659</td>
          <td>26.225241</td>
          <td>0.096138</td>
          <td>26.174929</td>
          <td>0.148869</td>
          <td>25.886086</td>
          <td>0.216844</td>
          <td>26.184349</td>
          <td>0.561921</td>
          <td>0.018268</td>
          <td>0.012769</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.870510</td>
          <td>0.227252</td>
          <td>26.220281</td>
          <td>0.108660</td>
          <td>26.020744</td>
          <td>0.080303</td>
          <td>25.930795</td>
          <td>0.120556</td>
          <td>25.646016</td>
          <td>0.177189</td>
          <td>25.076259</td>
          <td>0.237463</td>
          <td>0.003071</td>
          <td>0.001829</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.927581</td>
          <td>0.199307</td>
          <td>26.517383</td>
          <td>0.124062</td>
          <td>26.275648</td>
          <td>0.162278</td>
          <td>25.992567</td>
          <td>0.236886</td>
          <td>26.614771</td>
          <td>0.756700</td>
          <td>0.003950</td>
          <td>0.002138</td>
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
          <td>26.462550</td>
          <td>0.415579</td>
          <td>26.890993</td>
          <td>0.228127</td>
          <td>26.015606</td>
          <td>0.097323</td>
          <td>25.065460</td>
          <td>0.069175</td>
          <td>24.724133</td>
          <td>0.096887</td>
          <td>23.914866</td>
          <td>0.107371</td>
          <td>0.103593</td>
          <td>0.102018</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.916731</td>
          <td>3.633745</td>
          <td>27.775119</td>
          <td>0.448004</td>
          <td>26.723715</td>
          <td>0.173653</td>
          <td>26.212862</td>
          <td>0.181264</td>
          <td>25.601051</td>
          <td>0.199401</td>
          <td>24.997838</td>
          <td>0.260525</td>
          <td>0.019637</td>
          <td>0.015700</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.460525</td>
          <td>2.311545</td>
          <td>27.997255</td>
          <td>0.544484</td>
          <td>27.674804</td>
          <td>0.391422</td>
          <td>26.179752</td>
          <td>0.183512</td>
          <td>25.124714</td>
          <td>0.138219</td>
          <td>24.426090</td>
          <td>0.168084</td>
          <td>0.114550</td>
          <td>0.111233</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.201766</td>
          <td>0.331700</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.481354</td>
          <td>0.324531</td>
          <td>25.932374</td>
          <td>0.142730</td>
          <td>25.468030</td>
          <td>0.178319</td>
          <td>25.393626</td>
          <td>0.357975</td>
          <td>0.027067</td>
          <td>0.014640</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.851531</td>
          <td>0.250268</td>
          <td>26.195148</td>
          <td>0.122784</td>
          <td>25.842270</td>
          <td>0.080921</td>
          <td>25.589903</td>
          <td>0.106156</td>
          <td>25.530681</td>
          <td>0.188218</td>
          <td>25.463821</td>
          <td>0.378507</td>
          <td>0.034144</td>
          <td>0.020326</td>
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
          <td>27.202988</td>
          <td>0.697146</td>
          <td>26.628369</td>
          <td>0.178544</td>
          <td>25.520315</td>
          <td>0.061068</td>
          <td>25.094049</td>
          <td>0.068845</td>
          <td>24.779042</td>
          <td>0.098781</td>
          <td>24.706710</td>
          <td>0.205677</td>
          <td>0.050456</td>
          <td>0.030899</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.282929</td>
          <td>0.733519</td>
          <td>26.716188</td>
          <td>0.191428</td>
          <td>25.988535</td>
          <td>0.091860</td>
          <td>25.194704</td>
          <td>0.074847</td>
          <td>24.837867</td>
          <td>0.103464</td>
          <td>24.123397</td>
          <td>0.124357</td>
          <td>0.015897</td>
          <td>0.013271</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.874519</td>
          <td>0.552168</td>
          <td>27.077756</td>
          <td>0.258528</td>
          <td>26.416943</td>
          <td>0.133473</td>
          <td>26.190211</td>
          <td>0.177778</td>
          <td>25.645265</td>
          <td>0.206888</td>
          <td>25.720759</td>
          <td>0.459850</td>
          <td>0.018268</td>
          <td>0.012769</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.637430</td>
          <td>0.463617</td>
          <td>26.038697</td>
          <td>0.106908</td>
          <td>26.092449</td>
          <td>0.100554</td>
          <td>26.133654</td>
          <td>0.169298</td>
          <td>25.557211</td>
          <td>0.191987</td>
          <td>25.170891</td>
          <td>0.299487</td>
          <td>0.003071</td>
          <td>0.001829</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.284708</td>
          <td>0.353729</td>
          <td>26.406961</td>
          <td>0.147059</td>
          <td>26.616385</td>
          <td>0.158312</td>
          <td>26.289400</td>
          <td>0.193167</td>
          <td>26.060137</td>
          <td>0.290853</td>
          <td>25.098353</td>
          <td>0.282459</td>
          <td>0.003950</td>
          <td>0.002138</td>
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
          <td>27.370752</td>
          <td>0.761611</td>
          <td>26.950917</td>
          <td>0.225769</td>
          <td>25.977997</td>
          <td>0.087733</td>
          <td>25.180094</td>
          <td>0.071111</td>
          <td>24.739726</td>
          <td>0.091527</td>
          <td>23.918853</td>
          <td>0.100226</td>
          <td>0.103593</td>
          <td>0.102018</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.596145</td>
          <td>0.826942</td>
          <td>28.206705</td>
          <td>0.547789</td>
          <td>26.659104</td>
          <td>0.140830</td>
          <td>26.156156</td>
          <td>0.147132</td>
          <td>25.561273</td>
          <td>0.165552</td>
          <td>25.174328</td>
          <td>0.258485</td>
          <td>0.019637</td>
          <td>0.015700</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.565418</td>
          <td>1.533978</td>
          <td>28.289940</td>
          <td>0.643658</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.993418</td>
          <td>0.148108</td>
          <td>24.915280</td>
          <td>0.109129</td>
          <td>24.326400</td>
          <td>0.146040</td>
          <td>0.114550</td>
          <td>0.111233</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.206838</td>
          <td>0.548666</td>
          <td>27.274438</td>
          <td>0.237404</td>
          <td>26.259369</td>
          <td>0.161083</td>
          <td>25.763106</td>
          <td>0.196814</td>
          <td>25.048829</td>
          <td>0.233589</td>
          <td>0.027067</td>
          <td>0.014640</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.090510</td>
          <td>0.274080</td>
          <td>25.998249</td>
          <td>0.090292</td>
          <td>25.932486</td>
          <td>0.075078</td>
          <td>25.638053</td>
          <td>0.094385</td>
          <td>25.426451</td>
          <td>0.148421</td>
          <td>25.134543</td>
          <td>0.251724</td>
          <td>0.034144</td>
          <td>0.020326</td>
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
          <td>27.147724</td>
          <td>0.617592</td>
          <td>26.177752</td>
          <td>0.106817</td>
          <td>25.434922</td>
          <td>0.048912</td>
          <td>25.051937</td>
          <td>0.056973</td>
          <td>24.850877</td>
          <td>0.091082</td>
          <td>24.508844</td>
          <td>0.150593</td>
          <td>0.050456</td>
          <td>0.030899</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.463217</td>
          <td>0.367039</td>
          <td>26.596542</td>
          <td>0.150846</td>
          <td>26.224986</td>
          <td>0.096396</td>
          <td>25.116561</td>
          <td>0.059056</td>
          <td>24.799489</td>
          <td>0.085311</td>
          <td>24.180565</td>
          <td>0.111027</td>
          <td>0.015897</td>
          <td>0.013271</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.672683</td>
          <td>0.161061</td>
          <td>26.390136</td>
          <td>0.111431</td>
          <td>26.211291</td>
          <td>0.154116</td>
          <td>25.908215</td>
          <td>0.221582</td>
          <td>25.459121</td>
          <td>0.325026</td>
          <td>0.018268</td>
          <td>0.012769</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.074745</td>
          <td>0.268768</td>
          <td>26.282720</td>
          <td>0.114743</td>
          <td>25.975946</td>
          <td>0.077196</td>
          <td>25.816431</td>
          <td>0.109135</td>
          <td>25.624308</td>
          <td>0.173969</td>
          <td>24.913118</td>
          <td>0.207345</td>
          <td>0.003071</td>
          <td>0.001829</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.701546</td>
          <td>0.882554</td>
          <td>27.045599</td>
          <td>0.219999</td>
          <td>26.590442</td>
          <td>0.132186</td>
          <td>26.138336</td>
          <td>0.144279</td>
          <td>26.247068</td>
          <td>0.291674</td>
          <td>25.938567</td>
          <td>0.469300</td>
          <td>0.003950</td>
          <td>0.002138</td>
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
