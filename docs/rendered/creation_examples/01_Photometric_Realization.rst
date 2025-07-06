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

    <pzflow.flow.Flow at 0x7f9d69dbc490>



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
    0      23.994413  0.009572  0.008314  
    1      25.391064  0.052462  0.046031  
    2      24.304707  0.058261  0.057810  
    3      25.291103  0.005592  0.004065  
    4      25.096743  0.055375  0.032077  
    ...          ...       ...       ...  
    99995  24.737946  0.100595  0.098322  
    99996  24.224169  0.133435  0.081786  
    99997  25.613836  0.093308  0.068540  
    99998  25.274899  0.088600  0.087507  
    99999  25.699642  0.039714  0.034617  
    
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
          <td>27.632444</td>
          <td>0.844612</td>
          <td>27.011207</td>
          <td>0.213759</td>
          <td>26.165545</td>
          <td>0.091227</td>
          <td>25.115000</td>
          <td>0.058792</td>
          <td>24.716941</td>
          <td>0.079091</td>
          <td>24.154822</td>
          <td>0.108234</td>
          <td>0.009572</td>
          <td>0.008314</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.001746</td>
          <td>0.549398</td>
          <td>27.772721</td>
          <td>0.394704</td>
          <td>26.563420</td>
          <td>0.129113</td>
          <td>26.316926</td>
          <td>0.168092</td>
          <td>25.997769</td>
          <td>0.237906</td>
          <td>26.320877</td>
          <td>0.619145</td>
          <td>0.052462</td>
          <td>0.046031</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.675520</td>
          <td>1.525293</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.726527</td>
          <td>0.340241</td>
          <td>26.168562</td>
          <td>0.148057</td>
          <td>24.908272</td>
          <td>0.093604</td>
          <td>24.662712</td>
          <td>0.167800</td>
          <td>0.058261</td>
          <td>0.057810</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.107815</td>
          <td>0.592724</td>
          <td>28.617712</td>
          <td>0.727485</td>
          <td>27.083416</td>
          <td>0.201257</td>
          <td>26.257806</td>
          <td>0.159824</td>
          <td>25.506756</td>
          <td>0.157365</td>
          <td>25.227058</td>
          <td>0.268751</td>
          <td>0.005592</td>
          <td>0.004065</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.083644</td>
          <td>0.270705</td>
          <td>26.128810</td>
          <td>0.100315</td>
          <td>25.833035</td>
          <td>0.068023</td>
          <td>25.808028</td>
          <td>0.108327</td>
          <td>25.474821</td>
          <td>0.153120</td>
          <td>24.933157</td>
          <td>0.210832</td>
          <td>0.055375</td>
          <td>0.032077</td>
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
          <td>30.926721</td>
          <td>3.514105</td>
          <td>26.299440</td>
          <td>0.116415</td>
          <td>25.531787</td>
          <td>0.052071</td>
          <td>25.211823</td>
          <td>0.064065</td>
          <td>24.926423</td>
          <td>0.095108</td>
          <td>25.056103</td>
          <td>0.233537</td>
          <td>0.100595</td>
          <td>0.098322</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.891810</td>
          <td>0.992383</td>
          <td>26.603658</td>
          <td>0.151396</td>
          <td>26.065845</td>
          <td>0.083562</td>
          <td>25.097394</td>
          <td>0.057881</td>
          <td>24.940383</td>
          <td>0.096280</td>
          <td>24.165099</td>
          <td>0.109210</td>
          <td>0.133435</td>
          <td>0.081786</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.102551</td>
          <td>1.123424</td>
          <td>26.754766</td>
          <td>0.172235</td>
          <td>26.366419</td>
          <td>0.108784</td>
          <td>26.505601</td>
          <td>0.197201</td>
          <td>26.119504</td>
          <td>0.262936</td>
          <td>26.508195</td>
          <td>0.704528</td>
          <td>0.093308</td>
          <td>0.068540</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.318895</td>
          <td>0.327040</td>
          <td>26.360018</td>
          <td>0.122704</td>
          <td>26.038776</td>
          <td>0.081591</td>
          <td>25.863719</td>
          <td>0.113720</td>
          <td>25.926303</td>
          <td>0.224226</td>
          <td>25.416937</td>
          <td>0.313279</td>
          <td>0.088600</td>
          <td>0.087507</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.993962</td>
          <td>0.546317</td>
          <td>26.954841</td>
          <td>0.203917</td>
          <td>26.511885</td>
          <td>0.123472</td>
          <td>26.445358</td>
          <td>0.187438</td>
          <td>25.951851</td>
          <td>0.229033</td>
          <td>25.383849</td>
          <td>0.305087</td>
          <td>0.039714</td>
          <td>0.034617</td>
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
          <td>26.967407</td>
          <td>0.589905</td>
          <td>26.577172</td>
          <td>0.170113</td>
          <td>26.014164</td>
          <td>0.093908</td>
          <td>25.220747</td>
          <td>0.076552</td>
          <td>24.623835</td>
          <td>0.085707</td>
          <td>24.016205</td>
          <td>0.113240</td>
          <td>0.009572</td>
          <td>0.008314</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.434699</td>
          <td>0.346505</td>
          <td>26.549457</td>
          <td>0.150692</td>
          <td>25.992919</td>
          <td>0.151363</td>
          <td>25.561418</td>
          <td>0.194189</td>
          <td>24.813872</td>
          <td>0.225416</td>
          <td>0.052462</td>
          <td>0.046031</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.905699</td>
          <td>1.089153</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.663715</td>
          <td>0.377862</td>
          <td>26.048468</td>
          <td>0.159236</td>
          <td>25.005027</td>
          <td>0.120961</td>
          <td>24.354185</td>
          <td>0.153371</td>
          <td>0.058261</td>
          <td>0.057810</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.398867</td>
          <td>1.419029</td>
          <td>28.048400</td>
          <td>0.547811</td>
          <td>27.451682</td>
          <td>0.316508</td>
          <td>26.580933</td>
          <td>0.246277</td>
          <td>25.664053</td>
          <td>0.210013</td>
          <td>26.408172</td>
          <td>0.748342</td>
          <td>0.005592</td>
          <td>0.004065</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.991969</td>
          <td>0.281445</td>
          <td>25.910762</td>
          <td>0.096191</td>
          <td>25.985888</td>
          <td>0.092216</td>
          <td>25.765705</td>
          <td>0.124254</td>
          <td>25.560586</td>
          <td>0.193809</td>
          <td>25.182484</td>
          <td>0.304254</td>
          <td>0.055375</td>
          <td>0.032077</td>
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
          <td>27.779961</td>
          <td>1.022533</td>
          <td>26.297321</td>
          <td>0.137743</td>
          <td>25.353123</td>
          <td>0.054093</td>
          <td>25.073414</td>
          <td>0.069507</td>
          <td>24.854707</td>
          <td>0.108379</td>
          <td>24.990235</td>
          <td>0.266818</td>
          <td>0.100595</td>
          <td>0.098322</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.312700</td>
          <td>0.765310</td>
          <td>26.996284</td>
          <td>0.249898</td>
          <td>25.960719</td>
          <td>0.093209</td>
          <td>25.127904</td>
          <td>0.073481</td>
          <td>24.819263</td>
          <td>0.105822</td>
          <td>24.331343</td>
          <td>0.154729</td>
          <td>0.133435</td>
          <td>0.081786</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.543148</td>
          <td>0.879353</td>
          <td>26.680329</td>
          <td>0.189231</td>
          <td>26.367794</td>
          <td>0.130647</td>
          <td>26.537012</td>
          <td>0.242693</td>
          <td>25.623642</td>
          <td>0.207364</td>
          <td>25.286163</td>
          <td>0.335230</td>
          <td>0.093308</td>
          <td>0.068540</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.852923</td>
          <td>0.254709</td>
          <td>26.114526</td>
          <td>0.116870</td>
          <td>26.048101</td>
          <td>0.099237</td>
          <td>26.092490</td>
          <td>0.167735</td>
          <td>25.686183</td>
          <td>0.219198</td>
          <td>24.539239</td>
          <td>0.182141</td>
          <td>0.088600</td>
          <td>0.087507</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.308537</td>
          <td>0.361535</td>
          <td>26.831885</td>
          <td>0.211659</td>
          <td>26.508147</td>
          <td>0.144939</td>
          <td>26.535848</td>
          <td>0.238358</td>
          <td>25.838557</td>
          <td>0.243813</td>
          <td>26.026407</td>
          <td>0.577166</td>
          <td>0.039714</td>
          <td>0.034617</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.527748</td>
          <td>0.141972</td>
          <td>26.037336</td>
          <td>0.081577</td>
          <td>25.186202</td>
          <td>0.062698</td>
          <td>24.626764</td>
          <td>0.073114</td>
          <td>24.010262</td>
          <td>0.095476</td>
          <td>0.009572</td>
          <td>0.008314</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.259055</td>
          <td>0.670513</td>
          <td>27.394416</td>
          <td>0.300356</td>
          <td>26.910456</td>
          <td>0.179359</td>
          <td>26.428202</td>
          <td>0.190794</td>
          <td>25.766268</td>
          <td>0.202224</td>
          <td>25.721511</td>
          <td>0.409674</td>
          <td>0.052462</td>
          <td>0.046031</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.476852</td>
          <td>0.380488</td>
          <td>28.970684</td>
          <td>0.938968</td>
          <td>27.770646</td>
          <td>0.366280</td>
          <td>25.794273</td>
          <td>0.112022</td>
          <td>25.028005</td>
          <td>0.108582</td>
          <td>24.221236</td>
          <td>0.119963</td>
          <td>0.058261</td>
          <td>0.057810</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.231987</td>
          <td>0.646768</td>
          <td>27.642662</td>
          <td>0.356810</td>
          <td>27.660636</td>
          <td>0.323011</td>
          <td>26.015718</td>
          <td>0.129815</td>
          <td>25.283108</td>
          <td>0.129850</td>
          <td>25.158358</td>
          <td>0.254149</td>
          <td>0.005592</td>
          <td>0.004065</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.289732</td>
          <td>0.325031</td>
          <td>26.086444</td>
          <td>0.098947</td>
          <td>25.930565</td>
          <td>0.076190</td>
          <td>25.555199</td>
          <td>0.089269</td>
          <td>25.648755</td>
          <td>0.182259</td>
          <td>25.490582</td>
          <td>0.340648</td>
          <td>0.055375</td>
          <td>0.032077</td>
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
          <td>27.788127</td>
          <td>0.988190</td>
          <td>26.408357</td>
          <td>0.141772</td>
          <td>25.367492</td>
          <td>0.050747</td>
          <td>25.009038</td>
          <td>0.060658</td>
          <td>25.047578</td>
          <td>0.118947</td>
          <td>24.861003</td>
          <td>0.223017</td>
          <td>0.100595</td>
          <td>0.098322</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.044597</td>
          <td>0.614908</td>
          <td>26.769124</td>
          <td>0.196326</td>
          <td>26.058554</td>
          <td>0.095489</td>
          <td>25.207849</td>
          <td>0.073937</td>
          <td>25.033158</td>
          <td>0.119935</td>
          <td>24.305431</td>
          <td>0.142219</td>
          <td>0.133435</td>
          <td>0.081786</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.820748</td>
          <td>0.505614</td>
          <td>26.330493</td>
          <td>0.128499</td>
          <td>26.519583</td>
          <td>0.134875</td>
          <td>26.225787</td>
          <td>0.169164</td>
          <td>25.944814</td>
          <td>0.246128</td>
          <td>25.799560</td>
          <td>0.454901</td>
          <td>0.093308</td>
          <td>0.068540</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.275414</td>
          <td>0.335915</td>
          <td>26.311351</td>
          <td>0.127760</td>
          <td>26.025771</td>
          <td>0.088755</td>
          <td>25.826561</td>
          <td>0.121519</td>
          <td>26.049799</td>
          <td>0.271332</td>
          <td>25.078476</td>
          <td>0.260893</td>
          <td>0.088600</td>
          <td>0.087507</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.226791</td>
          <td>0.650900</td>
          <td>26.762782</td>
          <td>0.176126</td>
          <td>26.589012</td>
          <td>0.134430</td>
          <td>26.449424</td>
          <td>0.191621</td>
          <td>25.795562</td>
          <td>0.204603</td>
          <td>25.689293</td>
          <td>0.394778</td>
          <td>0.039714</td>
          <td>0.034617</td>
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
