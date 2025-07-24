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

    <pzflow.flow.Flow at 0x7fd239faa650>



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
    0      23.994413  0.148268  0.140706  
    1      25.391064  0.030008  0.026064  
    2      24.304707  0.100440  0.052563  
    3      25.291103  0.113755  0.082919  
    4      25.096743  0.067862  0.060625  
    ...          ...       ...       ...  
    99995  24.737946  0.067718  0.065623  
    99996  24.224169  0.123325  0.091497  
    99997  25.613836  0.013513  0.007564  
    99998  25.274899  0.205261  0.180425  
    99999  25.699642  0.044621  0.043705  
    
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
          <td>27.887241</td>
          <td>0.989649</td>
          <td>26.768955</td>
          <td>0.174323</td>
          <td>25.839634</td>
          <td>0.068422</td>
          <td>25.104872</td>
          <td>0.058266</td>
          <td>24.677575</td>
          <td>0.076389</td>
          <td>23.811767</td>
          <td>0.080081</td>
          <td>0.148268</td>
          <td>0.140706</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.704649</td>
          <td>0.441062</td>
          <td>27.466031</td>
          <td>0.310129</td>
          <td>26.432685</td>
          <td>0.115256</td>
          <td>26.077449</td>
          <td>0.136884</td>
          <td>25.675298</td>
          <td>0.181642</td>
          <td>25.984296</td>
          <td>0.485503</td>
          <td>0.030008</td>
          <td>0.026064</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.248033</td>
          <td>1.219381</td>
          <td>30.185590</td>
          <td>1.756775</td>
          <td>29.313868</td>
          <td>1.032057</td>
          <td>25.964512</td>
          <td>0.124138</td>
          <td>24.925410</td>
          <td>0.095024</td>
          <td>24.316091</td>
          <td>0.124546</td>
          <td>0.100440</td>
          <td>0.052563</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.749647</td>
          <td>0.456272</td>
          <td>27.904428</td>
          <td>0.436533</td>
          <td>27.422971</td>
          <td>0.266608</td>
          <td>26.149186</td>
          <td>0.145611</td>
          <td>25.426334</td>
          <td>0.146879</td>
          <td>25.206058</td>
          <td>0.264185</td>
          <td>0.113755</td>
          <td>0.082919</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.102462</td>
          <td>0.274876</td>
          <td>26.032668</td>
          <td>0.092209</td>
          <td>25.924766</td>
          <td>0.073776</td>
          <td>25.820514</td>
          <td>0.109514</td>
          <td>25.654911</td>
          <td>0.178531</td>
          <td>24.920991</td>
          <td>0.208698</td>
          <td>0.067862</td>
          <td>0.060625</td>
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
          <td>27.512016</td>
          <td>0.781203</td>
          <td>26.411913</td>
          <td>0.128347</td>
          <td>25.486427</td>
          <td>0.050016</td>
          <td>25.084462</td>
          <td>0.057220</td>
          <td>24.800818</td>
          <td>0.085163</td>
          <td>24.807462</td>
          <td>0.189709</td>
          <td>0.067718</td>
          <td>0.065623</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.870634</td>
          <td>0.189982</td>
          <td>26.130115</td>
          <td>0.088428</td>
          <td>25.220164</td>
          <td>0.064540</td>
          <td>24.731196</td>
          <td>0.080092</td>
          <td>24.100958</td>
          <td>0.103256</td>
          <td>0.123325</td>
          <td>0.091497</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.732925</td>
          <td>0.450570</td>
          <td>26.654667</td>
          <td>0.158153</td>
          <td>26.182850</td>
          <td>0.092625</td>
          <td>26.387574</td>
          <td>0.178493</td>
          <td>25.907814</td>
          <td>0.220805</td>
          <td>25.311185</td>
          <td>0.287745</td>
          <td>0.013513</td>
          <td>0.007564</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.739411</td>
          <td>0.452775</td>
          <td>26.237832</td>
          <td>0.110336</td>
          <td>26.031595</td>
          <td>0.081076</td>
          <td>25.711333</td>
          <td>0.099540</td>
          <td>25.709775</td>
          <td>0.187016</td>
          <td>25.705069</td>
          <td>0.392930</td>
          <td>0.205261</td>
          <td>0.180425</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.728391</td>
          <td>0.449035</td>
          <td>27.128712</td>
          <td>0.235674</td>
          <td>26.509630</td>
          <td>0.123230</td>
          <td>26.527916</td>
          <td>0.200933</td>
          <td>25.892352</td>
          <td>0.217979</td>
          <td>26.444667</td>
          <td>0.674672</td>
          <td>0.044621</td>
          <td>0.043705</td>
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
          <td>27.060568</td>
          <td>0.655787</td>
          <td>26.884297</td>
          <td>0.233109</td>
          <td>26.136229</td>
          <td>0.111610</td>
          <td>25.140106</td>
          <td>0.076365</td>
          <td>24.669482</td>
          <td>0.095318</td>
          <td>23.889272</td>
          <td>0.108441</td>
          <td>0.148268</td>
          <td>0.140706</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.677250</td>
          <td>0.478421</td>
          <td>27.278257</td>
          <td>0.304602</td>
          <td>26.678643</td>
          <td>0.167382</td>
          <td>26.405851</td>
          <td>0.213543</td>
          <td>26.196976</td>
          <td>0.325342</td>
          <td>26.334830</td>
          <td>0.713908</td>
          <td>0.030008</td>
          <td>0.026064</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.617953</td>
          <td>1.460983</td>
          <td>29.715915</td>
          <td>1.444633</td>
          <td>25.867290</td>
          <td>0.137665</td>
          <td>24.922914</td>
          <td>0.113735</td>
          <td>24.269300</td>
          <td>0.144019</td>
          <td>0.100440</td>
          <td>0.052563</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.595741</td>
          <td>1.453148</td>
          <td>27.323590</td>
          <td>0.294187</td>
          <td>26.592437</td>
          <td>0.256535</td>
          <td>25.296768</td>
          <td>0.158813</td>
          <td>25.214404</td>
          <td>0.319682</td>
          <td>0.113755</td>
          <td>0.082919</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.573945</td>
          <td>0.446038</td>
          <td>26.346482</td>
          <td>0.141333</td>
          <td>25.961867</td>
          <td>0.090924</td>
          <td>25.719137</td>
          <td>0.120187</td>
          <td>25.327803</td>
          <td>0.160154</td>
          <td>24.837786</td>
          <td>0.231201</td>
          <td>0.067862</td>
          <td>0.060625</td>
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
          <td>26.694864</td>
          <td>0.488552</td>
          <td>26.374100</td>
          <td>0.144860</td>
          <td>25.315528</td>
          <td>0.051390</td>
          <td>25.150012</td>
          <td>0.073021</td>
          <td>24.803661</td>
          <td>0.101843</td>
          <td>24.902167</td>
          <td>0.244069</td>
          <td>0.067718</td>
          <td>0.065623</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.796705</td>
          <td>0.534482</td>
          <td>26.686695</td>
          <td>0.192950</td>
          <td>26.141657</td>
          <td>0.109073</td>
          <td>25.163702</td>
          <td>0.075745</td>
          <td>24.985889</td>
          <td>0.122200</td>
          <td>24.312455</td>
          <td>0.152058</td>
          <td>0.123325</td>
          <td>0.091497</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.816689</td>
          <td>1.027862</td>
          <td>26.819041</td>
          <td>0.208629</td>
          <td>26.534318</td>
          <td>0.147614</td>
          <td>26.095488</td>
          <td>0.163945</td>
          <td>26.320890</td>
          <td>0.358053</td>
          <td>27.089568</td>
          <td>1.140515</td>
          <td>0.013513</td>
          <td>0.007564</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.485217</td>
          <td>0.445788</td>
          <td>26.372586</td>
          <td>0.157837</td>
          <td>25.922447</td>
          <td>0.096955</td>
          <td>25.955183</td>
          <td>0.162764</td>
          <td>25.266914</td>
          <td>0.167344</td>
          <td>25.350013</td>
          <td>0.382763</td>
          <td>0.205261</td>
          <td>0.180425</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.557714</td>
          <td>0.879870</td>
          <td>26.742075</td>
          <td>0.196639</td>
          <td>26.745767</td>
          <td>0.177881</td>
          <td>25.998495</td>
          <td>0.151839</td>
          <td>25.810651</td>
          <td>0.238689</td>
          <td>25.898579</td>
          <td>0.527140</td>
          <td>0.044621</td>
          <td>0.043705</td>
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
          <td>26.310234</td>
          <td>0.376311</td>
          <td>26.642035</td>
          <td>0.189561</td>
          <td>26.177351</td>
          <td>0.115096</td>
          <td>25.156951</td>
          <td>0.077099</td>
          <td>24.593741</td>
          <td>0.088729</td>
          <td>24.037727</td>
          <td>0.122758</td>
          <td>0.148268</td>
          <td>0.140706</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.470442</td>
          <td>0.764314</td>
          <td>27.268088</td>
          <td>0.266536</td>
          <td>26.550885</td>
          <td>0.129063</td>
          <td>26.199782</td>
          <td>0.153741</td>
          <td>25.753395</td>
          <td>0.196005</td>
          <td>25.481240</td>
          <td>0.333043</td>
          <td>0.030008</td>
          <td>0.026064</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.517901</td>
          <td>0.817385</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.634632</td>
          <td>0.339498</td>
          <td>25.956516</td>
          <td>0.133651</td>
          <td>25.004568</td>
          <td>0.110106</td>
          <td>24.232237</td>
          <td>0.125474</td>
          <td>0.100440</td>
          <td>0.052563</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.351752</td>
          <td>0.748805</td>
          <td>29.584961</td>
          <td>1.388269</td>
          <td>27.887692</td>
          <td>0.427709</td>
          <td>26.498252</td>
          <td>0.220293</td>
          <td>25.433828</td>
          <td>0.165747</td>
          <td>25.284033</td>
          <td>0.314652</td>
          <td>0.113755</td>
          <td>0.082919</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.433055</td>
          <td>0.762692</td>
          <td>26.106517</td>
          <td>0.103039</td>
          <td>25.971417</td>
          <td>0.081106</td>
          <td>25.720760</td>
          <td>0.106096</td>
          <td>25.443394</td>
          <td>0.156976</td>
          <td>25.168806</td>
          <td>0.269739</td>
          <td>0.067862</td>
          <td>0.060625</td>
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
          <td>26.880693</td>
          <td>0.520095</td>
          <td>26.326734</td>
          <td>0.125214</td>
          <td>25.490038</td>
          <td>0.053155</td>
          <td>25.058106</td>
          <td>0.059374</td>
          <td>24.930728</td>
          <td>0.101045</td>
          <td>25.020623</td>
          <td>0.239734</td>
          <td>0.067718</td>
          <td>0.065623</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.175080</td>
          <td>0.671990</td>
          <td>27.341825</td>
          <td>0.313462</td>
          <td>26.242666</td>
          <td>0.111886</td>
          <td>25.228848</td>
          <td>0.075124</td>
          <td>24.783866</td>
          <td>0.096226</td>
          <td>24.427799</td>
          <td>0.157565</td>
          <td>0.123325</td>
          <td>0.091497</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.156392</td>
          <td>0.287439</td>
          <td>26.805242</td>
          <td>0.180014</td>
          <td>26.412631</td>
          <td>0.113442</td>
          <td>26.419637</td>
          <td>0.183709</td>
          <td>26.008095</td>
          <td>0.240310</td>
          <td>25.086066</td>
          <td>0.239774</td>
          <td>0.013513</td>
          <td>0.007564</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.469633</td>
          <td>0.462765</td>
          <td>25.926304</td>
          <td>0.114554</td>
          <td>26.129605</td>
          <td>0.124593</td>
          <td>26.068242</td>
          <td>0.192113</td>
          <td>25.518991</td>
          <td>0.221399</td>
          <td>25.186607</td>
          <td>0.359207</td>
          <td>0.205261</td>
          <td>0.180425</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.314737</td>
          <td>0.331305</td>
          <td>26.491450</td>
          <td>0.140537</td>
          <td>26.690200</td>
          <td>0.147731</td>
          <td>26.390148</td>
          <td>0.183611</td>
          <td>25.509811</td>
          <td>0.161784</td>
          <td>25.758207</td>
          <td>0.418994</td>
          <td>0.044621</td>
          <td>0.043705</td>
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
