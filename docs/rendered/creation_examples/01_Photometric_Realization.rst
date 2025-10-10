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

    <pzflow.flow.Flow at 0x7f628297c700>



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
    0      23.994413  0.103777  0.078957  
    1      25.391064  0.060806  0.057147  
    2      24.304707  0.138687  0.092284  
    3      25.291103  0.097043  0.086631  
    4      25.096743  0.009906  0.008201  
    ...          ...       ...       ...  
    99995  24.737946  0.058231  0.040513  
    99996  24.224169  0.165911  0.132576  
    99997  25.613836  0.070687  0.039769  
    99998  25.274899  0.056129  0.029151  
    99999  25.699642  0.016211  0.009363  
    
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
          <td>29.178059</td>
          <td>1.924258</td>
          <td>26.946530</td>
          <td>0.202501</td>
          <td>26.154556</td>
          <td>0.090350</td>
          <td>25.163601</td>
          <td>0.061383</td>
          <td>24.722606</td>
          <td>0.079488</td>
          <td>23.994745</td>
          <td>0.094077</td>
          <td>0.103777</td>
          <td>0.078957</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.140070</td>
          <td>0.606397</td>
          <td>27.378133</td>
          <td>0.288969</td>
          <td>26.590889</td>
          <td>0.132219</td>
          <td>26.199975</td>
          <td>0.152103</td>
          <td>25.742497</td>
          <td>0.192251</td>
          <td>25.364388</td>
          <td>0.300355</td>
          <td>0.060806</td>
          <td>0.057147</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.446425</td>
          <td>0.647241</td>
          <td>27.853183</td>
          <td>0.375770</td>
          <td>26.153367</td>
          <td>0.146136</td>
          <td>25.226113</td>
          <td>0.123552</td>
          <td>24.429419</td>
          <td>0.137377</td>
          <td>0.138687</td>
          <td>0.092284</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.894796</td>
          <td>0.433356</td>
          <td>27.312507</td>
          <td>0.243518</td>
          <td>26.506706</td>
          <td>0.197384</td>
          <td>25.377226</td>
          <td>0.140801</td>
          <td>25.473136</td>
          <td>0.327630</td>
          <td>0.097043</td>
          <td>0.086631</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.788292</td>
          <td>0.469669</td>
          <td>26.039883</td>
          <td>0.092795</td>
          <td>25.956042</td>
          <td>0.075844</td>
          <td>25.581684</td>
          <td>0.088828</td>
          <td>25.530997</td>
          <td>0.160661</td>
          <td>25.070509</td>
          <td>0.236337</td>
          <td>0.009906</td>
          <td>0.008201</td>
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
          <td>26.860869</td>
          <td>0.495680</td>
          <td>26.398703</td>
          <td>0.126888</td>
          <td>25.500445</td>
          <td>0.050642</td>
          <td>25.089707</td>
          <td>0.057487</td>
          <td>24.845208</td>
          <td>0.088557</td>
          <td>25.103193</td>
          <td>0.242800</td>
          <td>0.058231</td>
          <td>0.040513</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.251715</td>
          <td>0.655543</td>
          <td>26.875281</td>
          <td>0.190728</td>
          <td>26.042987</td>
          <td>0.081895</td>
          <td>25.149264</td>
          <td>0.060607</td>
          <td>24.731835</td>
          <td>0.080138</td>
          <td>24.309014</td>
          <td>0.123784</td>
          <td>0.165911</td>
          <td>0.132576</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.722553</td>
          <td>0.447063</td>
          <td>26.623366</td>
          <td>0.153974</td>
          <td>26.396088</td>
          <td>0.111637</td>
          <td>26.360285</td>
          <td>0.174407</td>
          <td>25.594490</td>
          <td>0.169599</td>
          <td>25.570564</td>
          <td>0.353843</td>
          <td>0.070687</td>
          <td>0.039769</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.993500</td>
          <td>0.251496</td>
          <td>26.278774</td>
          <td>0.114341</td>
          <td>26.163522</td>
          <td>0.091065</td>
          <td>25.970421</td>
          <td>0.124776</td>
          <td>25.541583</td>
          <td>0.162120</td>
          <td>25.404774</td>
          <td>0.310246</td>
          <td>0.056129</td>
          <td>0.029151</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.971482</td>
          <td>0.206779</td>
          <td>26.571877</td>
          <td>0.130062</td>
          <td>26.273524</td>
          <td>0.161984</td>
          <td>26.204809</td>
          <td>0.281838</td>
          <td>25.094478</td>
          <td>0.241062</td>
          <td>0.016211</td>
          <td>0.009363</td>
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
          <td>27.341306</td>
          <td>0.774623</td>
          <td>26.895935</td>
          <td>0.227763</td>
          <td>26.028624</td>
          <td>0.097787</td>
          <td>25.149431</td>
          <td>0.073992</td>
          <td>24.699489</td>
          <td>0.094187</td>
          <td>23.859880</td>
          <td>0.101638</td>
          <td>0.103777</td>
          <td>0.078957</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.847585</td>
          <td>1.052933</td>
          <td>27.564975</td>
          <td>0.384708</td>
          <td>26.746616</td>
          <td>0.178890</td>
          <td>26.205205</td>
          <td>0.182016</td>
          <td>25.562250</td>
          <td>0.194976</td>
          <td>25.717419</td>
          <td>0.463093</td>
          <td>0.060806</td>
          <td>0.057147</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.925602</td>
          <td>0.588711</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.651642</td>
          <td>0.385561</td>
          <td>25.891757</td>
          <td>0.143992</td>
          <td>25.309633</td>
          <td>0.162494</td>
          <td>24.265430</td>
          <td>0.146975</td>
          <td>0.138687</td>
          <td>0.092284</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.868672</td>
          <td>0.490491</td>
          <td>27.341715</td>
          <td>0.297274</td>
          <td>26.136182</td>
          <td>0.174480</td>
          <td>25.773707</td>
          <td>0.236214</td>
          <td>25.105021</td>
          <td>0.291589</td>
          <td>0.097043</td>
          <td>0.086631</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.413759</td>
          <td>0.391168</td>
          <td>26.066172</td>
          <td>0.109526</td>
          <td>25.850153</td>
          <td>0.081289</td>
          <td>25.532506</td>
          <td>0.100708</td>
          <td>25.241732</td>
          <td>0.146805</td>
          <td>24.904118</td>
          <td>0.241037</td>
          <td>0.009906</td>
          <td>0.008201</td>
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
          <td>26.501261</td>
          <td>0.160613</td>
          <td>25.411229</td>
          <td>0.055581</td>
          <td>25.059922</td>
          <td>0.066976</td>
          <td>25.046777</td>
          <td>0.125070</td>
          <td>24.591457</td>
          <td>0.187147</td>
          <td>0.058231</td>
          <td>0.040513</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.296367</td>
          <td>0.374821</td>
          <td>26.624459</td>
          <td>0.188385</td>
          <td>25.979065</td>
          <td>0.097738</td>
          <td>25.249882</td>
          <td>0.084536</td>
          <td>24.747733</td>
          <td>0.102559</td>
          <td>24.047694</td>
          <td>0.125057</td>
          <td>0.165911</td>
          <td>0.132576</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.599715</td>
          <td>0.453929</td>
          <td>26.540576</td>
          <td>0.166460</td>
          <td>26.408299</td>
          <td>0.133816</td>
          <td>26.348117</td>
          <td>0.205159</td>
          <td>26.062092</td>
          <td>0.294269</td>
          <td>25.688139</td>
          <td>0.452752</td>
          <td>0.070687</td>
          <td>0.039769</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.776156</td>
          <td>0.515993</td>
          <td>26.010628</td>
          <td>0.104954</td>
          <td>26.189367</td>
          <td>0.110180</td>
          <td>25.773486</td>
          <td>0.125075</td>
          <td>26.025027</td>
          <td>0.284466</td>
          <td>25.231726</td>
          <td>0.316436</td>
          <td>0.056129</td>
          <td>0.029151</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.025865</td>
          <td>0.614889</td>
          <td>26.647650</td>
          <td>0.180643</td>
          <td>26.505714</td>
          <td>0.144055</td>
          <td>26.030030</td>
          <td>0.155054</td>
          <td>25.791846</td>
          <td>0.233684</td>
          <td>25.234656</td>
          <td>0.315359</td>
          <td>0.016211</td>
          <td>0.009363</td>
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
          <td>27.091318</td>
          <td>0.621483</td>
          <td>27.287435</td>
          <td>0.291827</td>
          <td>26.189577</td>
          <td>0.103255</td>
          <td>25.195972</td>
          <td>0.070415</td>
          <td>24.626574</td>
          <td>0.080981</td>
          <td>23.820434</td>
          <td>0.089800</td>
          <td>0.103777</td>
          <td>0.078957</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.046008</td>
          <td>0.228391</td>
          <td>26.634348</td>
          <td>0.143491</td>
          <td>25.994691</td>
          <td>0.133508</td>
          <td>25.889191</td>
          <td>0.226882</td>
          <td>25.775567</td>
          <td>0.431999</td>
          <td>0.060806</td>
          <td>0.057147</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.526614</td>
          <td>2.334284</td>
          <td>29.413108</td>
          <td>1.294388</td>
          <td>28.337063</td>
          <td>0.613942</td>
          <td>26.134989</td>
          <td>0.168552</td>
          <td>24.908892</td>
          <td>0.109417</td>
          <td>24.021337</td>
          <td>0.113080</td>
          <td>0.138687</td>
          <td>0.092284</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>32.521163</td>
          <td>3.983193</td>
          <td>27.062037</td>
          <td>0.218040</td>
          <td>26.206454</td>
          <td>0.169816</td>
          <td>25.413900</td>
          <td>0.160656</td>
          <td>24.895754</td>
          <td>0.226096</td>
          <td>0.097043</td>
          <td>0.086631</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.856485</td>
          <td>0.224791</td>
          <td>26.099732</td>
          <td>0.097889</td>
          <td>25.846723</td>
          <td>0.068931</td>
          <td>25.704483</td>
          <td>0.099061</td>
          <td>25.382702</td>
          <td>0.141623</td>
          <td>25.118980</td>
          <td>0.246248</td>
          <td>0.009906</td>
          <td>0.008201</td>
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
          <td>26.463188</td>
          <td>0.137996</td>
          <td>25.517269</td>
          <td>0.053147</td>
          <td>24.989565</td>
          <td>0.054468</td>
          <td>24.923953</td>
          <td>0.098071</td>
          <td>24.616455</td>
          <td>0.166749</td>
          <td>0.058231</td>
          <td>0.040513</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.858066</td>
          <td>1.093902</td>
          <td>26.816972</td>
          <td>0.221844</td>
          <td>26.135139</td>
          <td>0.112299</td>
          <td>25.299454</td>
          <td>0.088527</td>
          <td>24.963674</td>
          <td>0.124090</td>
          <td>24.329686</td>
          <td>0.159809</td>
          <td>0.165911</td>
          <td>0.132576</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.342751</td>
          <td>0.342318</td>
          <td>26.596437</td>
          <td>0.155976</td>
          <td>26.292517</td>
          <td>0.106364</td>
          <td>26.156471</td>
          <td>0.153006</td>
          <td>25.787224</td>
          <td>0.207831</td>
          <td>26.449140</td>
          <td>0.700152</td>
          <td>0.070687</td>
          <td>0.039769</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.520278</td>
          <td>0.796229</td>
          <td>26.185282</td>
          <td>0.107813</td>
          <td>26.122301</td>
          <td>0.090161</td>
          <td>25.933878</td>
          <td>0.124209</td>
          <td>25.735175</td>
          <td>0.195924</td>
          <td>25.057742</td>
          <td>0.239916</td>
          <td>0.056129</td>
          <td>0.029151</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.750309</td>
          <td>0.457141</td>
          <td>26.769619</td>
          <td>0.174769</td>
          <td>26.367291</td>
          <td>0.109124</td>
          <td>26.370352</td>
          <td>0.176328</td>
          <td>25.722804</td>
          <td>0.189515</td>
          <td>24.794618</td>
          <td>0.188109</td>
          <td>0.016211</td>
          <td>0.009363</td>
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
