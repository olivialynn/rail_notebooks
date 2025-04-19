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

    <pzflow.flow.Flow at 0x7f9f731ddfc0>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>26.266872</td>
          <td>0.313775</td>
          <td>26.880018</td>
          <td>0.191491</td>
          <td>26.129108</td>
          <td>0.088350</td>
          <td>25.190924</td>
          <td>0.062888</td>
          <td>24.740616</td>
          <td>0.080761</td>
          <td>24.015682</td>
          <td>0.095822</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.102864</td>
          <td>0.506274</td>
          <td>26.625117</td>
          <td>0.136188</td>
          <td>26.543487</td>
          <td>0.203576</td>
          <td>25.536433</td>
          <td>0.161409</td>
          <td>25.999845</td>
          <td>0.491133</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.422016</td>
          <td>0.735994</td>
          <td>27.954752</td>
          <td>0.453441</td>
          <td>28.013617</td>
          <td>0.425180</td>
          <td>26.190457</td>
          <td>0.150866</td>
          <td>25.127926</td>
          <td>0.113440</td>
          <td>24.414426</td>
          <td>0.135611</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.504839</td>
          <td>1.249226</td>
          <td>27.265067</td>
          <td>0.234162</td>
          <td>26.070972</td>
          <td>0.136121</td>
          <td>25.459782</td>
          <td>0.151158</td>
          <td>25.099899</td>
          <td>0.242142</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.761989</td>
          <td>0.207634</td>
          <td>26.003519</td>
          <td>0.089880</td>
          <td>25.868557</td>
          <td>0.070197</td>
          <td>25.754796</td>
          <td>0.103401</td>
          <td>25.363128</td>
          <td>0.139101</td>
          <td>25.142313</td>
          <td>0.250744</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>28.053059</td>
          <td>1.091789</td>
          <td>26.397552</td>
          <td>0.126761</td>
          <td>25.507837</td>
          <td>0.050976</td>
          <td>25.144121</td>
          <td>0.060331</td>
          <td>24.844642</td>
          <td>0.088513</td>
          <td>25.295129</td>
          <td>0.284032</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.030406</td>
          <td>0.560859</td>
          <td>27.092455</td>
          <td>0.228705</td>
          <td>26.045556</td>
          <td>0.082081</td>
          <td>25.277505</td>
          <td>0.067904</td>
          <td>24.790605</td>
          <td>0.084400</td>
          <td>24.154129</td>
          <td>0.108169</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.103672</td>
          <td>0.590985</td>
          <td>27.003438</td>
          <td>0.212378</td>
          <td>26.302933</td>
          <td>0.102911</td>
          <td>26.419053</td>
          <td>0.183316</td>
          <td>26.037471</td>
          <td>0.245825</td>
          <td>24.977734</td>
          <td>0.218823</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.739101</td>
          <td>0.452669</td>
          <td>26.110236</td>
          <td>0.098697</td>
          <td>26.204593</td>
          <td>0.094411</td>
          <td>25.956369</td>
          <td>0.123264</td>
          <td>25.914253</td>
          <td>0.221991</td>
          <td>24.826059</td>
          <td>0.192707</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.897720</td>
          <td>0.509316</td>
          <td>27.102701</td>
          <td>0.230656</td>
          <td>26.566055</td>
          <td>0.129408</td>
          <td>25.977552</td>
          <td>0.125550</td>
          <td>26.530935</td>
          <td>0.365430</td>
          <td>25.563677</td>
          <td>0.351933</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>27.385888</td>
          <td>0.784958</td>
          <td>26.602916</td>
          <td>0.173838</td>
          <td>26.051817</td>
          <td>0.097038</td>
          <td>25.178756</td>
          <td>0.073746</td>
          <td>24.794006</td>
          <td>0.099498</td>
          <td>23.937585</td>
          <td>0.105706</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.294772</td>
          <td>3.040284</td>
          <td>27.451605</td>
          <td>0.348888</td>
          <td>26.491371</td>
          <td>0.142237</td>
          <td>26.838294</td>
          <td>0.303652</td>
          <td>25.566138</td>
          <td>0.193476</td>
          <td>25.183619</td>
          <td>0.302628</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.021091</td>
          <td>1.169925</td>
          <td>27.692356</td>
          <td>0.427809</td>
          <td>27.278669</td>
          <td>0.281046</td>
          <td>26.003287</td>
          <td>0.154902</td>
          <td>25.075551</td>
          <td>0.129988</td>
          <td>23.969796</td>
          <td>0.111225</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.565406</td>
          <td>0.459105</td>
          <td>27.498892</td>
          <td>0.382096</td>
          <td>27.269382</td>
          <td>0.290694</td>
          <td>26.502618</td>
          <td>0.246386</td>
          <td>25.462992</td>
          <td>0.189119</td>
          <td>25.773162</td>
          <td>0.506414</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.707567</td>
          <td>0.488579</td>
          <td>26.284598</td>
          <td>0.132391</td>
          <td>25.985472</td>
          <td>0.091578</td>
          <td>25.763354</td>
          <td>0.123172</td>
          <td>25.965140</td>
          <td>0.269366</td>
          <td>25.226927</td>
          <td>0.313348</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>27.130356</td>
          <td>0.669385</td>
          <td>26.518395</td>
          <td>0.164781</td>
          <td>25.374255</td>
          <td>0.054475</td>
          <td>25.107186</td>
          <td>0.070757</td>
          <td>24.778903</td>
          <td>0.100269</td>
          <td>25.041765</td>
          <td>0.275208</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.760725</td>
          <td>0.995940</td>
          <td>27.006973</td>
          <td>0.244628</td>
          <td>26.038812</td>
          <td>0.096339</td>
          <td>25.221643</td>
          <td>0.076927</td>
          <td>24.735867</td>
          <td>0.094949</td>
          <td>24.349299</td>
          <td>0.151640</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.676363</td>
          <td>0.481241</td>
          <td>26.811154</td>
          <td>0.209479</td>
          <td>26.191071</td>
          <td>0.111010</td>
          <td>26.053172</td>
          <td>0.160114</td>
          <td>25.748619</td>
          <td>0.228073</td>
          <td>25.324086</td>
          <td>0.342428</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.574920</td>
          <td>0.451548</td>
          <td>26.094206</td>
          <td>0.115415</td>
          <td>26.184623</td>
          <td>0.112450</td>
          <td>25.973846</td>
          <td>0.152438</td>
          <td>25.824728</td>
          <td>0.247162</td>
          <td>24.816398</td>
          <td>0.231029</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.611740</td>
          <td>0.176682</td>
          <td>26.570601</td>
          <td>0.153724</td>
          <td>26.309248</td>
          <td>0.198384</td>
          <td>25.644955</td>
          <td>0.208658</td>
          <td>26.176365</td>
          <td>0.644301</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
          <td>26.907936</td>
          <td>0.513187</td>
          <td>26.763289</td>
          <td>0.173505</td>
          <td>25.826881</td>
          <td>0.067663</td>
          <td>25.228768</td>
          <td>0.065043</td>
          <td>24.636839</td>
          <td>0.073697</td>
          <td>24.027959</td>
          <td>0.096873</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.622283</td>
          <td>0.414526</td>
          <td>27.034259</td>
          <td>0.218077</td>
          <td>26.443679</td>
          <td>0.116473</td>
          <td>26.302389</td>
          <td>0.166183</td>
          <td>26.082402</td>
          <td>0.255295</td>
          <td>25.393796</td>
          <td>0.307806</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.694388</td>
          <td>0.356965</td>
          <td>25.965803</td>
          <td>0.135207</td>
          <td>25.039639</td>
          <td>0.113918</td>
          <td>24.202977</td>
          <td>0.122763</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.177845</td>
          <td>0.709298</td>
          <td>28.574560</td>
          <td>0.821673</td>
          <td>27.294109</td>
          <td>0.295579</td>
          <td>26.084711</td>
          <td>0.173057</td>
          <td>25.600432</td>
          <td>0.211531</td>
          <td>24.864914</td>
          <td>0.248030</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.757560</td>
          <td>0.207061</td>
          <td>26.294747</td>
          <td>0.116083</td>
          <td>25.802097</td>
          <td>0.066280</td>
          <td>25.594958</td>
          <td>0.090006</td>
          <td>25.686084</td>
          <td>0.183558</td>
          <td>24.838828</td>
          <td>0.195066</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>27.038900</td>
          <td>0.590466</td>
          <td>26.609155</td>
          <td>0.162676</td>
          <td>25.353628</td>
          <td>0.048145</td>
          <td>25.099725</td>
          <td>0.063040</td>
          <td>24.933047</td>
          <td>0.103473</td>
          <td>24.927032</td>
          <td>0.226622</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.596403</td>
          <td>1.475825</td>
          <td>26.552347</td>
          <td>0.146918</td>
          <td>25.966049</td>
          <td>0.077797</td>
          <td>25.197406</td>
          <td>0.064366</td>
          <td>24.659793</td>
          <td>0.076455</td>
          <td>24.251403</td>
          <td>0.119752</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.127485</td>
          <td>0.289545</td>
          <td>26.709298</td>
          <td>0.172700</td>
          <td>26.304392</td>
          <td>0.108186</td>
          <td>26.342945</td>
          <td>0.180574</td>
          <td>25.716331</td>
          <td>0.197074</td>
          <td>25.758944</td>
          <td>0.428020</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.217137</td>
          <td>0.324923</td>
          <td>26.182023</td>
          <td>0.116170</td>
          <td>26.025386</td>
          <td>0.090467</td>
          <td>26.031363</td>
          <td>0.147947</td>
          <td>25.418326</td>
          <td>0.163108</td>
          <td>25.176371</td>
          <td>0.287789</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.917198</td>
          <td>0.204081</td>
          <td>26.832620</td>
          <td>0.169010</td>
          <td>26.666718</td>
          <td>0.234481</td>
          <td>25.827203</td>
          <td>0.214208</td>
          <td>25.509216</td>
          <td>0.349581</td>
          <td>0.059611</td>
          <td>0.049181</td>
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
