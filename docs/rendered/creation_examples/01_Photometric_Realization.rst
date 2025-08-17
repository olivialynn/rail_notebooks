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

    <pzflow.flow.Flow at 0x7ff3aa4825f0>



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
    0      23.994413  0.130987  0.081615  
    1      25.391064  0.151887  0.104292  
    2      24.304707  0.011786  0.006380  
    3      25.291103  0.151532  0.128378  
    4      25.096743  0.145059  0.144207  
    ...          ...       ...       ...  
    99995  24.737946  0.057960  0.036025  
    99996  24.224169  0.062214  0.045211  
    99997  25.613836  0.154014  0.141337  
    99998  25.274899  0.040389  0.022025  
    99999  25.699642  0.263392  0.142433  
    
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
          <td>27.629969</td>
          <td>0.843275</td>
          <td>26.817114</td>
          <td>0.181585</td>
          <td>26.230389</td>
          <td>0.096573</td>
          <td>25.105423</td>
          <td>0.058295</td>
          <td>24.646292</td>
          <td>0.074306</td>
          <td>23.992703</td>
          <td>0.093909</td>
          <td>0.130987</td>
          <td>0.081615</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.571602</td>
          <td>2.260163</td>
          <td>27.590198</td>
          <td>0.342296</td>
          <td>26.414005</td>
          <td>0.113395</td>
          <td>26.261433</td>
          <td>0.160320</td>
          <td>25.869454</td>
          <td>0.213856</td>
          <td>25.165263</td>
          <td>0.255512</td>
          <td>0.151887</td>
          <td>0.104292</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.892345</td>
          <td>0.507309</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.990562</td>
          <td>0.417767</td>
          <td>26.121811</td>
          <td>0.142221</td>
          <td>25.062343</td>
          <td>0.107130</td>
          <td>24.476885</td>
          <td>0.143114</td>
          <td>0.011786</td>
          <td>0.006380</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.565405</td>
          <td>0.702266</td>
          <td>27.834052</td>
          <td>0.370211</td>
          <td>26.284589</td>
          <td>0.163521</td>
          <td>25.652605</td>
          <td>0.178182</td>
          <td>25.101532</td>
          <td>0.242468</td>
          <td>0.151532</td>
          <td>0.128378</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.964693</td>
          <td>0.534851</td>
          <td>25.879705</td>
          <td>0.080605</td>
          <td>25.940863</td>
          <td>0.074833</td>
          <td>25.765478</td>
          <td>0.104372</td>
          <td>25.431382</td>
          <td>0.147517</td>
          <td>25.048420</td>
          <td>0.232056</td>
          <td>0.145059</td>
          <td>0.144207</td>
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
          <td>27.382015</td>
          <td>0.716499</td>
          <td>26.445390</td>
          <td>0.132117</td>
          <td>25.453936</td>
          <td>0.048593</td>
          <td>24.945798</td>
          <td>0.050592</td>
          <td>24.887516</td>
          <td>0.091913</td>
          <td>24.662550</td>
          <td>0.167777</td>
          <td>0.057960</td>
          <td>0.036025</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.420021</td>
          <td>1.338323</td>
          <td>27.050271</td>
          <td>0.220831</td>
          <td>26.095240</td>
          <td>0.085755</td>
          <td>25.207676</td>
          <td>0.063829</td>
          <td>25.029520</td>
          <td>0.104100</td>
          <td>24.036224</td>
          <td>0.097564</td>
          <td>0.062214</td>
          <td>0.045211</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.781723</td>
          <td>0.467370</td>
          <td>26.998109</td>
          <td>0.211435</td>
          <td>26.359221</td>
          <td>0.108103</td>
          <td>26.369709</td>
          <td>0.175808</td>
          <td>25.694839</td>
          <td>0.184670</td>
          <td>25.102502</td>
          <td>0.242662</td>
          <td>0.154014</td>
          <td>0.141337</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.127292</td>
          <td>0.280465</td>
          <td>26.211113</td>
          <td>0.107795</td>
          <td>25.987746</td>
          <td>0.077998</td>
          <td>25.943341</td>
          <td>0.121877</td>
          <td>25.681424</td>
          <td>0.182586</td>
          <td>25.895888</td>
          <td>0.454463</td>
          <td>0.040389</td>
          <td>0.022025</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.373609</td>
          <td>0.341509</td>
          <td>26.880530</td>
          <td>0.191573</td>
          <td>26.661461</td>
          <td>0.140525</td>
          <td>26.326069</td>
          <td>0.169406</td>
          <td>26.103912</td>
          <td>0.259604</td>
          <td>25.557427</td>
          <td>0.350208</td>
          <td>0.263392</td>
          <td>0.142433</td>
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
          <td>27.661340</td>
          <td>0.954914</td>
          <td>26.823551</td>
          <td>0.216427</td>
          <td>25.902539</td>
          <td>0.088473</td>
          <td>25.153021</td>
          <td>0.075049</td>
          <td>24.693066</td>
          <td>0.094655</td>
          <td>23.852873</td>
          <td>0.102118</td>
          <td>0.130987</td>
          <td>0.081615</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.775370</td>
          <td>0.466933</td>
          <td>26.415488</td>
          <td>0.140488</td>
          <td>26.178803</td>
          <td>0.185684</td>
          <td>25.722962</td>
          <td>0.232167</td>
          <td>25.450877</td>
          <td>0.392620</td>
          <td>0.151887</td>
          <td>0.104292</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.894635</td>
          <td>0.560035</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.940973</td>
          <td>0.920124</td>
          <td>25.965919</td>
          <td>0.146714</td>
          <td>25.120117</td>
          <td>0.132199</td>
          <td>24.493300</td>
          <td>0.170817</td>
          <td>0.011786</td>
          <td>0.006380</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.267772</td>
          <td>0.669324</td>
          <td>27.400066</td>
          <td>0.321529</td>
          <td>26.645211</td>
          <td>0.275703</td>
          <td>25.682264</td>
          <td>0.226280</td>
          <td>26.146301</td>
          <td>0.658485</td>
          <td>0.151532</td>
          <td>0.128378</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.700915</td>
          <td>0.231539</td>
          <td>26.524243</td>
          <td>0.172365</td>
          <td>25.858331</td>
          <td>0.087503</td>
          <td>25.385260</td>
          <td>0.094770</td>
          <td>25.650199</td>
          <td>0.221093</td>
          <td>25.650202</td>
          <td>0.461930</td>
          <td>0.145059</td>
          <td>0.144207</td>
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
          <td>27.738133</td>
          <td>0.984326</td>
          <td>26.307873</td>
          <td>0.135979</td>
          <td>25.351038</td>
          <td>0.052657</td>
          <td>25.132611</td>
          <td>0.071379</td>
          <td>24.752750</td>
          <td>0.096718</td>
          <td>24.715950</td>
          <td>0.207668</td>
          <td>0.057960</td>
          <td>0.036025</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.871758</td>
          <td>0.554217</td>
          <td>26.605467</td>
          <td>0.175730</td>
          <td>25.945205</td>
          <td>0.089249</td>
          <td>25.324064</td>
          <td>0.084706</td>
          <td>24.755200</td>
          <td>0.097130</td>
          <td>24.260545</td>
          <td>0.141321</td>
          <td>0.062214</td>
          <td>0.045211</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>32.111585</td>
          <td>4.861855</td>
          <td>26.946466</td>
          <td>0.245969</td>
          <td>26.374515</td>
          <td>0.137614</td>
          <td>26.337687</td>
          <td>0.215332</td>
          <td>26.366510</td>
          <td>0.394250</td>
          <td>26.798334</td>
          <td>1.008080</td>
          <td>0.154014</td>
          <td>0.141337</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.878560</td>
          <td>1.067844</td>
          <td>26.436937</td>
          <td>0.151362</td>
          <td>26.086011</td>
          <td>0.100346</td>
          <td>25.808549</td>
          <td>0.128517</td>
          <td>25.812645</td>
          <td>0.238407</td>
          <td>25.389874</td>
          <td>0.357572</td>
          <td>0.040389</td>
          <td>0.022025</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.827616</td>
          <td>0.579509</td>
          <td>26.985850</td>
          <td>0.267301</td>
          <td>26.854718</td>
          <td>0.219022</td>
          <td>26.098071</td>
          <td>0.186620</td>
          <td>26.133586</td>
          <td>0.346465</td>
          <td>24.960130</td>
          <td>0.285040</td>
          <td>0.263392</td>
          <td>0.142433</td>
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
          <td>27.112455</td>
          <td>0.643533</td>
          <td>26.797410</td>
          <td>0.200508</td>
          <td>26.037088</td>
          <td>0.093417</td>
          <td>25.172714</td>
          <td>0.071444</td>
          <td>24.755338</td>
          <td>0.093799</td>
          <td>24.160605</td>
          <td>0.125099</td>
          <td>0.130987</td>
          <td>0.081615</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.457796</td>
          <td>0.833735</td>
          <td>27.585037</td>
          <td>0.394481</td>
          <td>26.565814</td>
          <td>0.155105</td>
          <td>26.194910</td>
          <td>0.182536</td>
          <td>25.925296</td>
          <td>0.266397</td>
          <td>25.008779</td>
          <td>0.268512</td>
          <td>0.151887</td>
          <td>0.104292</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.195419</td>
          <td>1.184811</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.135592</td>
          <td>0.466688</td>
          <td>26.040130</td>
          <td>0.132709</td>
          <td>25.038759</td>
          <td>0.105072</td>
          <td>24.203504</td>
          <td>0.113073</td>
          <td>0.011786</td>
          <td>0.006380</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.056954</td>
          <td>0.647854</td>
          <td>28.342080</td>
          <td>0.698079</td>
          <td>27.108742</td>
          <td>0.251127</td>
          <td>26.304944</td>
          <td>0.205698</td>
          <td>25.683123</td>
          <td>0.223815</td>
          <td>25.471886</td>
          <td>0.397667</td>
          <td>0.151532</td>
          <td>0.128378</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.670473</td>
          <td>0.972167</td>
          <td>26.080918</td>
          <td>0.117233</td>
          <td>25.806638</td>
          <td>0.083202</td>
          <td>25.593587</td>
          <td>0.113140</td>
          <td>25.245052</td>
          <td>0.156299</td>
          <td>25.039337</td>
          <td>0.285415</td>
          <td>0.145059</td>
          <td>0.144207</td>
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
          <td>26.838092</td>
          <td>0.496225</td>
          <td>26.366042</td>
          <td>0.126622</td>
          <td>25.378868</td>
          <td>0.046886</td>
          <td>25.100077</td>
          <td>0.059925</td>
          <td>24.921597</td>
          <td>0.097632</td>
          <td>25.020436</td>
          <td>0.233599</td>
          <td>0.057960</td>
          <td>0.036025</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.667475</td>
          <td>0.165186</td>
          <td>26.153019</td>
          <td>0.093771</td>
          <td>25.153072</td>
          <td>0.063343</td>
          <td>24.839091</td>
          <td>0.091538</td>
          <td>24.393110</td>
          <td>0.138465</td>
          <td>0.062214</td>
          <td>0.045211</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.723322</td>
          <td>0.204374</td>
          <td>26.395480</td>
          <td>0.140126</td>
          <td>26.494660</td>
          <td>0.245254</td>
          <td>26.105476</td>
          <td>0.321252</td>
          <td>26.067802</td>
          <td>0.626757</td>
          <td>0.154014</td>
          <td>0.141337</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.350026</td>
          <td>0.338183</td>
          <td>26.163304</td>
          <td>0.104658</td>
          <td>26.030575</td>
          <td>0.082160</td>
          <td>25.738382</td>
          <td>0.103443</td>
          <td>25.888954</td>
          <td>0.220287</td>
          <td>25.389471</td>
          <td>0.310587</td>
          <td>0.040389</td>
          <td>0.022025</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.708541</td>
          <td>0.562275</td>
          <td>26.699308</td>
          <td>0.226507</td>
          <td>26.459054</td>
          <td>0.169356</td>
          <td>26.864313</td>
          <td>0.375346</td>
          <td>26.076653</td>
          <td>0.355640</td>
          <td>25.988177</td>
          <td>0.662839</td>
          <td>0.263392</td>
          <td>0.142433</td>
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
