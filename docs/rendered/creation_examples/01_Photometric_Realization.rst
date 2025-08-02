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

    <pzflow.flow.Flow at 0x7f4186dfd630>



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
    0      23.994413  0.049870  0.029415  
    1      25.391064  0.017372  0.014328  
    2      24.304707  0.027896  0.024786  
    3      25.291103  0.147769  0.082199  
    4      25.096743  0.063102  0.056903  
    ...          ...       ...       ...  
    99995  24.737946  0.159984  0.112534  
    99996  24.224169  0.030240  0.024136  
    99997  25.613836  0.059121  0.056315  
    99998  25.274899  0.078134  0.065905  
    99999  25.699642  0.210823  0.154330  
    
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
          <td>27.382806</td>
          <td>0.716881</td>
          <td>26.621165</td>
          <td>0.153684</td>
          <td>26.018213</td>
          <td>0.080124</td>
          <td>25.300536</td>
          <td>0.069303</td>
          <td>24.745797</td>
          <td>0.081131</td>
          <td>24.100303</td>
          <td>0.103197</td>
          <td>0.049870</td>
          <td>0.029415</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.127489</td>
          <td>1.139560</td>
          <td>27.702408</td>
          <td>0.373769</td>
          <td>26.414599</td>
          <td>0.113454</td>
          <td>26.405565</td>
          <td>0.181235</td>
          <td>25.822377</td>
          <td>0.205599</td>
          <td>25.568223</td>
          <td>0.353193</td>
          <td>0.017372</td>
          <td>0.014328</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.266224</td>
          <td>1.091964</td>
          <td>28.594965</td>
          <td>0.649388</td>
          <td>26.138430</td>
          <td>0.144270</td>
          <td>25.036922</td>
          <td>0.104776</td>
          <td>24.213898</td>
          <td>0.113958</td>
          <td>0.027896</td>
          <td>0.024786</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.785609</td>
          <td>0.930042</td>
          <td>28.275845</td>
          <td>0.573965</td>
          <td>27.471833</td>
          <td>0.277425</td>
          <td>26.336466</td>
          <td>0.170911</td>
          <td>25.352761</td>
          <td>0.137862</td>
          <td>25.249981</td>
          <td>0.273814</td>
          <td>0.147769</td>
          <td>0.082199</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.655560</td>
          <td>0.424941</td>
          <td>26.107555</td>
          <td>0.098466</td>
          <td>25.889101</td>
          <td>0.071485</td>
          <td>25.802491</td>
          <td>0.107804</td>
          <td>25.398102</td>
          <td>0.143355</td>
          <td>25.075525</td>
          <td>0.237318</td>
          <td>0.063102</td>
          <td>0.056903</td>
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
          <td>26.378710</td>
          <td>0.342885</td>
          <td>26.500472</td>
          <td>0.138548</td>
          <td>25.619068</td>
          <td>0.056267</td>
          <td>25.076039</td>
          <td>0.056794</td>
          <td>24.919216</td>
          <td>0.094508</td>
          <td>24.984096</td>
          <td>0.219985</td>
          <td>0.159984</td>
          <td>0.112534</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.588613</td>
          <td>0.403738</td>
          <td>26.511725</td>
          <td>0.139898</td>
          <td>25.954276</td>
          <td>0.075726</td>
          <td>25.244336</td>
          <td>0.065938</td>
          <td>24.704757</td>
          <td>0.078245</td>
          <td>24.306218</td>
          <td>0.123484</td>
          <td>0.030240</td>
          <td>0.024136</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.752169</td>
          <td>0.457136</td>
          <td>26.767347</td>
          <td>0.174085</td>
          <td>26.414637</td>
          <td>0.113458</td>
          <td>25.941673</td>
          <td>0.121701</td>
          <td>25.779249</td>
          <td>0.198290</td>
          <td>25.772063</td>
          <td>0.413702</td>
          <td>0.059121</td>
          <td>0.056315</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.281582</td>
          <td>0.317478</td>
          <td>26.142228</td>
          <td>0.101499</td>
          <td>26.163019</td>
          <td>0.091025</td>
          <td>25.979731</td>
          <td>0.125788</td>
          <td>25.618484</td>
          <td>0.173096</td>
          <td>25.015646</td>
          <td>0.225833</td>
          <td>0.078134</td>
          <td>0.065905</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.977046</td>
          <td>0.539667</td>
          <td>26.504384</td>
          <td>0.139016</td>
          <td>26.552181</td>
          <td>0.127863</td>
          <td>26.276944</td>
          <td>0.162458</td>
          <td>25.980441</td>
          <td>0.234522</td>
          <td>25.512928</td>
          <td>0.338128</td>
          <td>0.210823</td>
          <td>0.154330</td>
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
          <td>26.832077</td>
          <td>0.537092</td>
          <td>26.586651</td>
          <td>0.172299</td>
          <td>25.986218</td>
          <td>0.092128</td>
          <td>25.164392</td>
          <td>0.073245</td>
          <td>24.725999</td>
          <td>0.094268</td>
          <td>23.963923</td>
          <td>0.108791</td>
          <td>0.049870</td>
          <td>0.029415</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.960552</td>
          <td>0.514171</td>
          <td>26.859723</td>
          <td>0.194780</td>
          <td>25.986338</td>
          <td>0.149394</td>
          <td>25.757032</td>
          <td>0.227095</td>
          <td>26.260104</td>
          <td>0.677563</td>
          <td>0.017372</td>
          <td>0.014328</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.257437</td>
          <td>0.721770</td>
          <td>29.010550</td>
          <td>1.035273</td>
          <td>29.077446</td>
          <td>1.001603</td>
          <td>25.800391</td>
          <td>0.127453</td>
          <td>25.086153</td>
          <td>0.128632</td>
          <td>24.381891</td>
          <td>0.155647</td>
          <td>0.027896</td>
          <td>0.024786</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.858252</td>
          <td>1.078332</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.289116</td>
          <td>0.289631</td>
          <td>26.450753</td>
          <td>0.231223</td>
          <td>25.450311</td>
          <td>0.183308</td>
          <td>25.255250</td>
          <td>0.334321</td>
          <td>0.147769</td>
          <td>0.082199</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.064640</td>
          <td>0.636230</td>
          <td>26.109605</td>
          <td>0.114962</td>
          <td>26.045289</td>
          <td>0.097660</td>
          <td>25.692697</td>
          <td>0.117245</td>
          <td>25.402595</td>
          <td>0.170407</td>
          <td>24.898274</td>
          <td>0.242639</td>
          <td>0.063102</td>
          <td>0.056903</td>
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
          <td>27.521138</td>
          <td>0.885991</td>
          <td>26.275324</td>
          <td>0.138602</td>
          <td>25.553678</td>
          <td>0.066483</td>
          <td>25.048266</td>
          <td>0.070000</td>
          <td>25.026172</td>
          <td>0.129352</td>
          <td>25.013425</td>
          <td>0.279256</td>
          <td>0.159984</td>
          <td>0.112534</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.524102</td>
          <td>0.162921</td>
          <td>25.892836</td>
          <td>0.084597</td>
          <td>25.072119</td>
          <td>0.067284</td>
          <td>24.834349</td>
          <td>0.103333</td>
          <td>24.554767</td>
          <td>0.180365</td>
          <td>0.030240</td>
          <td>0.024136</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.710896</td>
          <td>0.493210</td>
          <td>26.929023</td>
          <td>0.230750</td>
          <td>26.555928</td>
          <td>0.151975</td>
          <td>26.188372</td>
          <td>0.179351</td>
          <td>25.970769</td>
          <td>0.273340</td>
          <td>25.148382</td>
          <td>0.297230</td>
          <td>0.059121</td>
          <td>0.056315</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.140081</td>
          <td>0.319322</td>
          <td>26.369978</td>
          <td>0.144674</td>
          <td>26.155697</td>
          <td>0.108136</td>
          <td>25.876152</td>
          <td>0.138189</td>
          <td>25.667973</td>
          <td>0.214213</td>
          <td>25.256118</td>
          <td>0.325881</td>
          <td>0.078134</td>
          <td>0.065905</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.711862</td>
          <td>0.524155</td>
          <td>26.959154</td>
          <td>0.255924</td>
          <td>26.421459</td>
          <td>0.148159</td>
          <td>26.512705</td>
          <td>0.257213</td>
          <td>25.650047</td>
          <td>0.228892</td>
          <td>25.187567</td>
          <td>0.334118</td>
          <td>0.210823</td>
          <td>0.154330</td>
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
          <td>27.260801</td>
          <td>0.667701</td>
          <td>26.953081</td>
          <td>0.207390</td>
          <td>26.058004</td>
          <td>0.084853</td>
          <td>25.159486</td>
          <td>0.062613</td>
          <td>24.789909</td>
          <td>0.086242</td>
          <td>24.099738</td>
          <td>0.105536</td>
          <td>0.049870</td>
          <td>0.029415</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>31.053964</td>
          <td>3.639225</td>
          <td>27.552750</td>
          <td>0.333199</td>
          <td>26.641803</td>
          <td>0.138630</td>
          <td>26.087292</td>
          <td>0.138542</td>
          <td>25.889089</td>
          <td>0.218096</td>
          <td>25.499039</td>
          <td>0.335509</td>
          <td>0.017372</td>
          <td>0.014328</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.755844</td>
          <td>1.592118</td>
          <td>28.141601</td>
          <td>0.524396</td>
          <td>28.489273</td>
          <td>0.607645</td>
          <td>26.111701</td>
          <td>0.142355</td>
          <td>24.955719</td>
          <td>0.098501</td>
          <td>24.168551</td>
          <td>0.110605</td>
          <td>0.027896</td>
          <td>0.024786</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.363250</td>
          <td>2.928962</td>
          <td>27.584143</td>
          <td>0.350227</td>
          <td>26.235907</td>
          <td>0.183849</td>
          <td>25.493048</td>
          <td>0.181145</td>
          <td>24.961335</td>
          <td>0.251563</td>
          <td>0.147769</td>
          <td>0.082199</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.053491</td>
          <td>0.272295</td>
          <td>26.235028</td>
          <td>0.114608</td>
          <td>26.040524</td>
          <td>0.085641</td>
          <td>25.559142</td>
          <td>0.091460</td>
          <td>25.456177</td>
          <td>0.157704</td>
          <td>25.889127</td>
          <td>0.471232</td>
          <td>0.063102</td>
          <td>0.056903</td>
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
          <td>26.270682</td>
          <td>0.135872</td>
          <td>25.464444</td>
          <td>0.060319</td>
          <td>25.073231</td>
          <td>0.070223</td>
          <td>24.700370</td>
          <td>0.095645</td>
          <td>24.564228</td>
          <td>0.189151</td>
          <td>0.159984</td>
          <td>0.112534</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.159711</td>
          <td>1.915628</td>
          <td>26.570980</td>
          <td>0.148474</td>
          <td>26.040694</td>
          <td>0.082560</td>
          <td>25.308879</td>
          <td>0.070565</td>
          <td>24.760799</td>
          <td>0.083046</td>
          <td>24.277458</td>
          <td>0.121690</td>
          <td>0.030240</td>
          <td>0.024136</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.668372</td>
          <td>0.440489</td>
          <td>26.538774</td>
          <td>0.148580</td>
          <td>26.419738</td>
          <td>0.118959</td>
          <td>26.095324</td>
          <td>0.145334</td>
          <td>25.784819</td>
          <td>0.207614</td>
          <td>25.339555</td>
          <td>0.306757</td>
          <td>0.059121</td>
          <td>0.056315</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.474733</td>
          <td>0.385276</td>
          <td>26.036585</td>
          <td>0.098032</td>
          <td>26.144045</td>
          <td>0.095642</td>
          <td>25.873277</td>
          <td>0.122804</td>
          <td>25.951677</td>
          <td>0.243674</td>
          <td>25.022580</td>
          <td>0.242273</td>
          <td>0.078134</td>
          <td>0.065905</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.716232</td>
          <td>0.545386</td>
          <td>27.178152</td>
          <td>0.319903</td>
          <td>26.494455</td>
          <td>0.166227</td>
          <td>26.604341</td>
          <td>0.291672</td>
          <td>26.065280</td>
          <td>0.336927</td>
          <td>25.007124</td>
          <td>0.304173</td>
          <td>0.210823</td>
          <td>0.154330</td>
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
