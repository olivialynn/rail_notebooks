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

    <pzflow.flow.Flow at 0x7f5b8715a650>



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
    0      23.994413  0.045763  0.039796  
    1      25.391064  0.089193  0.071707  
    2      24.304707  0.185288  0.159553  
    3      25.291103  0.179918  0.163631  
    4      25.096743  0.114747  0.060878  
    ...          ...       ...       ...  
    99995  24.737946  0.069893  0.045563  
    99996  24.224169  0.142645  0.121135  
    99997  25.613836  0.032137  0.017109  
    99998  25.274899  0.192965  0.155809  
    99999  25.699642  0.054116  0.052222  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>27.043429</td>
          <td>0.219578</td>
          <td>25.958217</td>
          <td>0.075990</td>
          <td>25.104220</td>
          <td>0.058232</td>
          <td>24.774689</td>
          <td>0.083224</td>
          <td>24.166149</td>
          <td>0.109310</td>
          <td>0.045763</td>
          <td>0.039796</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.582578</td>
          <td>0.817952</td>
          <td>26.737258</td>
          <td>0.169691</td>
          <td>26.408718</td>
          <td>0.112874</td>
          <td>26.069681</td>
          <td>0.135969</td>
          <td>25.729070</td>
          <td>0.190086</td>
          <td>25.552594</td>
          <td>0.348878</td>
          <td>0.089193</td>
          <td>0.071707</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.027816</td>
          <td>0.865685</td>
          <td>25.942734</td>
          <td>0.121813</td>
          <td>25.034866</td>
          <td>0.104588</td>
          <td>24.372979</td>
          <td>0.130839</td>
          <td>0.185288</td>
          <td>0.159553</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.827712</td>
          <td>0.411748</td>
          <td>27.663215</td>
          <td>0.323580</td>
          <td>25.917043</td>
          <td>0.119123</td>
          <td>25.836584</td>
          <td>0.208060</td>
          <td>25.619690</td>
          <td>0.367722</td>
          <td>0.179918</td>
          <td>0.163631</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.438894</td>
          <td>0.359484</td>
          <td>25.944144</td>
          <td>0.085309</td>
          <td>25.963155</td>
          <td>0.076322</td>
          <td>25.720408</td>
          <td>0.100335</td>
          <td>25.219155</td>
          <td>0.122808</td>
          <td>24.978317</td>
          <td>0.218929</td>
          <td>0.114747</td>
          <td>0.060878</td>
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
          <td>26.909019</td>
          <td>0.513556</td>
          <td>26.270956</td>
          <td>0.113566</td>
          <td>25.402186</td>
          <td>0.046411</td>
          <td>24.944001</td>
          <td>0.050511</td>
          <td>24.857918</td>
          <td>0.089552</td>
          <td>24.676994</td>
          <td>0.169853</td>
          <td>0.069893</td>
          <td>0.045563</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.860534</td>
          <td>0.188370</td>
          <td>26.082911</td>
          <td>0.084828</td>
          <td>25.281888</td>
          <td>0.068168</td>
          <td>24.766918</td>
          <td>0.082656</td>
          <td>24.203288</td>
          <td>0.112909</td>
          <td>0.142645</td>
          <td>0.121135</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.339852</td>
          <td>0.696348</td>
          <td>26.901345</td>
          <td>0.194960</td>
          <td>26.295644</td>
          <td>0.102257</td>
          <td>25.981644</td>
          <td>0.125996</td>
          <td>26.304461</td>
          <td>0.305421</td>
          <td>25.456584</td>
          <td>0.323346</td>
          <td>0.032137</td>
          <td>0.017109</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.174493</td>
          <td>0.291365</td>
          <td>26.108749</td>
          <td>0.098569</td>
          <td>25.985652</td>
          <td>0.077854</td>
          <td>25.855252</td>
          <td>0.112884</td>
          <td>25.854959</td>
          <td>0.211282</td>
          <td>24.794197</td>
          <td>0.187597</td>
          <td>0.192965</td>
          <td>0.155809</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.855176</td>
          <td>0.493599</td>
          <td>26.846002</td>
          <td>0.186074</td>
          <td>26.561033</td>
          <td>0.128847</td>
          <td>26.536144</td>
          <td>0.202326</td>
          <td>26.492973</td>
          <td>0.354723</td>
          <td>25.434292</td>
          <td>0.317652</td>
          <td>0.054116</td>
          <td>0.052222</td>
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
          <td>28.473761</td>
          <td>1.478090</td>
          <td>26.574520</td>
          <td>0.170615</td>
          <td>25.999436</td>
          <td>0.093255</td>
          <td>25.190047</td>
          <td>0.074968</td>
          <td>24.589910</td>
          <td>0.083683</td>
          <td>24.036661</td>
          <td>0.115977</td>
          <td>0.045763</td>
          <td>0.039796</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.679442</td>
          <td>0.189011</td>
          <td>26.787295</td>
          <td>0.186959</td>
          <td>25.869301</td>
          <td>0.137979</td>
          <td>26.324499</td>
          <td>0.365963</td>
          <td>26.122226</td>
          <td>0.626392</td>
          <td>0.089193</td>
          <td>0.071707</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.749066</td>
          <td>0.534582</td>
          <td>29.306262</td>
          <td>1.290035</td>
          <td>27.797458</td>
          <td>0.449161</td>
          <td>25.936310</td>
          <td>0.156914</td>
          <td>25.016357</td>
          <td>0.132295</td>
          <td>24.473284</td>
          <td>0.183935</td>
          <td>0.185288</td>
          <td>0.159553</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.817103</td>
          <td>0.495667</td>
          <td>26.821951</td>
          <td>0.205754</td>
          <td>26.479208</td>
          <td>0.247400</td>
          <td>25.426457</td>
          <td>0.187694</td>
          <td>25.144812</td>
          <td>0.319431</td>
          <td>0.179918</td>
          <td>0.163631</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.581813</td>
          <td>0.203812</td>
          <td>26.072656</td>
          <td>0.112892</td>
          <td>26.055623</td>
          <td>0.100088</td>
          <td>25.604379</td>
          <td>0.110306</td>
          <td>25.161282</td>
          <td>0.140713</td>
          <td>24.765199</td>
          <td>0.220606</td>
          <td>0.114747</td>
          <td>0.060878</td>
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
          <td>29.115854</td>
          <td>1.992126</td>
          <td>26.185008</td>
          <td>0.122692</td>
          <td>25.449733</td>
          <td>0.057699</td>
          <td>25.085135</td>
          <td>0.068717</td>
          <td>24.850467</td>
          <td>0.105758</td>
          <td>25.287966</td>
          <td>0.332439</td>
          <td>0.069893</td>
          <td>0.045563</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.923521</td>
          <td>1.125689</td>
          <td>26.676218</td>
          <td>0.194230</td>
          <td>26.044332</td>
          <td>0.101961</td>
          <td>25.121848</td>
          <td>0.074351</td>
          <td>24.658661</td>
          <td>0.093459</td>
          <td>24.081438</td>
          <td>0.126844</td>
          <td>0.142645</td>
          <td>0.121135</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.763407</td>
          <td>0.509732</td>
          <td>27.022321</td>
          <td>0.247329</td>
          <td>26.167534</td>
          <td>0.107617</td>
          <td>25.916140</td>
          <td>0.140841</td>
          <td>25.707466</td>
          <td>0.218212</td>
          <td>25.099031</td>
          <td>0.283209</td>
          <td>0.032137</td>
          <td>0.017109</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.898573</td>
          <td>0.595972</td>
          <td>26.089606</td>
          <td>0.121690</td>
          <td>26.059763</td>
          <td>0.107361</td>
          <td>25.718263</td>
          <td>0.130364</td>
          <td>25.810385</td>
          <td>0.259059</td>
          <td>25.401124</td>
          <td>0.391726</td>
          <td>0.192965</td>
          <td>0.155809</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.357430</td>
          <td>0.774602</td>
          <td>26.799233</td>
          <td>0.206818</td>
          <td>26.417754</td>
          <td>0.134714</td>
          <td>26.448923</td>
          <td>0.222841</td>
          <td>25.720721</td>
          <td>0.222161</td>
          <td>27.808601</td>
          <td>1.666334</td>
          <td>0.054116</td>
          <td>0.052222</td>
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
          <td>26.911615</td>
          <td>0.521861</td>
          <td>26.659952</td>
          <td>0.162175</td>
          <td>26.061193</td>
          <td>0.085281</td>
          <td>25.174042</td>
          <td>0.063576</td>
          <td>24.637043</td>
          <td>0.075530</td>
          <td>24.119035</td>
          <td>0.107577</td>
          <td>0.045763</td>
          <td>0.039796</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.425721</td>
          <td>0.770707</td>
          <td>27.361438</td>
          <td>0.304354</td>
          <td>26.530224</td>
          <td>0.135922</td>
          <td>26.044353</td>
          <td>0.144620</td>
          <td>25.764418</td>
          <td>0.211625</td>
          <td>25.123681</td>
          <td>0.267120</td>
          <td>0.089193</td>
          <td>0.071707</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.361034</td>
          <td>0.833437</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.833132</td>
          <td>1.628058</td>
          <td>26.301850</td>
          <td>0.222069</td>
          <td>25.116159</td>
          <td>0.149817</td>
          <td>24.446635</td>
          <td>0.186904</td>
          <td>0.185288</td>
          <td>0.159553</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.311666</td>
          <td>0.806401</td>
          <td>28.531527</td>
          <td>0.835548</td>
          <td>28.241502</td>
          <td>0.639412</td>
          <td>25.962826</td>
          <td>0.166635</td>
          <td>25.276540</td>
          <td>0.171527</td>
          <td>24.966061</td>
          <td>0.286801</td>
          <td>0.179918</td>
          <td>0.163631</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.828371</td>
          <td>0.513986</td>
          <td>26.184438</td>
          <td>0.114945</td>
          <td>25.907633</td>
          <td>0.080362</td>
          <td>25.709777</td>
          <td>0.110294</td>
          <td>25.508223</td>
          <td>0.173596</td>
          <td>26.049190</td>
          <td>0.554869</td>
          <td>0.114747</td>
          <td>0.060878</td>
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
          <td>29.624702</td>
          <td>2.338366</td>
          <td>26.516952</td>
          <td>0.146016</td>
          <td>25.412673</td>
          <td>0.049022</td>
          <td>25.104068</td>
          <td>0.061057</td>
          <td>24.876249</td>
          <td>0.095176</td>
          <td>24.432630</td>
          <td>0.144184</td>
          <td>0.069893</td>
          <td>0.045563</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.366383</td>
          <td>0.788835</td>
          <td>26.450926</td>
          <td>0.156934</td>
          <td>26.069721</td>
          <td>0.101661</td>
          <td>25.179657</td>
          <td>0.076219</td>
          <td>24.776566</td>
          <td>0.101058</td>
          <td>24.216155</td>
          <td>0.138921</td>
          <td>0.142645</td>
          <td>0.121135</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.338695</td>
          <td>0.334070</td>
          <td>26.609512</td>
          <td>0.153309</td>
          <td>26.306322</td>
          <td>0.104137</td>
          <td>26.546642</td>
          <td>0.205942</td>
          <td>25.962494</td>
          <td>0.233007</td>
          <td>25.170104</td>
          <td>0.258748</td>
          <td>0.032137</td>
          <td>0.017109</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.879698</td>
          <td>0.282374</td>
          <td>26.145723</td>
          <td>0.132736</td>
          <td>25.931829</td>
          <td>0.100108</td>
          <td>26.080502</td>
          <td>0.185365</td>
          <td>25.392243</td>
          <td>0.190419</td>
          <td>25.213360</td>
          <td>0.351523</td>
          <td>0.192965</td>
          <td>0.155809</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.067533</td>
          <td>0.588183</td>
          <td>26.666093</td>
          <td>0.164781</td>
          <td>26.721151</td>
          <td>0.153371</td>
          <td>26.580277</td>
          <td>0.217822</td>
          <td>25.645799</td>
          <td>0.183578</td>
          <td>25.172809</td>
          <td>0.266441</td>
          <td>0.054116</td>
          <td>0.052222</td>
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
