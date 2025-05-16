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

    <pzflow.flow.Flow at 0x7f3138772470>



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
          <td>26.766299</td>
          <td>0.462006</td>
          <td>26.687772</td>
          <td>0.162687</td>
          <td>26.181683</td>
          <td>0.092530</td>
          <td>25.214155</td>
          <td>0.064197</td>
          <td>24.788991</td>
          <td>0.084280</td>
          <td>24.024075</td>
          <td>0.096530</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.943053</td>
          <td>2.591627</td>
          <td>27.793279</td>
          <td>0.401007</td>
          <td>26.383726</td>
          <td>0.110440</td>
          <td>26.663154</td>
          <td>0.224964</td>
          <td>25.842148</td>
          <td>0.209031</td>
          <td>25.500363</td>
          <td>0.334782</td>
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
          <td>27.281414</td>
          <td>0.237349</td>
          <td>26.082084</td>
          <td>0.137433</td>
          <td>24.957276</td>
          <td>0.097718</td>
          <td>24.294311</td>
          <td>0.122214</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.910170</td>
          <td>1.003414</td>
          <td>29.070900</td>
          <td>0.972245</td>
          <td>27.163872</td>
          <td>0.215276</td>
          <td>26.306133</td>
          <td>0.166554</td>
          <td>25.678454</td>
          <td>0.182127</td>
          <td>25.161099</td>
          <td>0.254641</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.121301</td>
          <td>0.279107</td>
          <td>25.989303</td>
          <td>0.088764</td>
          <td>26.159195</td>
          <td>0.090719</td>
          <td>25.815442</td>
          <td>0.109031</td>
          <td>25.374027</td>
          <td>0.140414</td>
          <td>24.748432</td>
          <td>0.180474</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.263784</td>
          <td>0.112859</td>
          <td>25.516682</td>
          <td>0.051378</td>
          <td>25.032774</td>
          <td>0.054654</td>
          <td>24.744425</td>
          <td>0.081033</td>
          <td>24.510884</td>
          <td>0.147360</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.039587</td>
          <td>0.564568</td>
          <td>26.494741</td>
          <td>0.137865</td>
          <td>25.935685</td>
          <td>0.074491</td>
          <td>25.323965</td>
          <td>0.070755</td>
          <td>24.800723</td>
          <td>0.085156</td>
          <td>24.608528</td>
          <td>0.160220</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.765219</td>
          <td>0.461633</td>
          <td>26.516472</td>
          <td>0.140471</td>
          <td>26.413648</td>
          <td>0.113360</td>
          <td>26.052888</td>
          <td>0.134011</td>
          <td>26.185025</td>
          <td>0.277351</td>
          <td>25.558782</td>
          <td>0.350581</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.303817</td>
          <td>0.323147</td>
          <td>26.414685</td>
          <td>0.128655</td>
          <td>26.172374</td>
          <td>0.091777</td>
          <td>25.840342</td>
          <td>0.111426</td>
          <td>25.735679</td>
          <td>0.191149</td>
          <td>24.775221</td>
          <td>0.184613</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.048260</td>
          <td>0.568090</td>
          <td>26.620985</td>
          <td>0.153660</td>
          <td>26.544376</td>
          <td>0.127001</td>
          <td>26.190336</td>
          <td>0.150850</td>
          <td>25.717628</td>
          <td>0.188260</td>
          <td>25.858653</td>
          <td>0.441877</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.709450</td>
          <td>0.190229</td>
          <td>26.065036</td>
          <td>0.098169</td>
          <td>25.325716</td>
          <td>0.083958</td>
          <td>24.541092</td>
          <td>0.079661</td>
          <td>23.871395</td>
          <td>0.099759</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.174811</td>
          <td>0.324379</td>
          <td>27.620949</td>
          <td>0.398067</td>
          <td>26.612404</td>
          <td>0.157805</td>
          <td>26.445049</td>
          <td>0.220107</td>
          <td>25.654813</td>
          <td>0.208428</td>
          <td>25.077601</td>
          <td>0.277799</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.038072</td>
          <td>0.628441</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.296919</td>
          <td>0.610239</td>
          <td>25.792836</td>
          <td>0.129227</td>
          <td>25.049147</td>
          <td>0.127050</td>
          <td>24.323462</td>
          <td>0.151038</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.122190</td>
          <td>0.606712</td>
          <td>27.466067</td>
          <td>0.340150</td>
          <td>26.193021</td>
          <td>0.190348</td>
          <td>25.656605</td>
          <td>0.222423</td>
          <td>25.106994</td>
          <td>0.302993</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.781752</td>
          <td>0.516014</td>
          <td>26.200569</td>
          <td>0.123107</td>
          <td>25.807971</td>
          <td>0.078326</td>
          <td>25.929224</td>
          <td>0.142163</td>
          <td>25.642707</td>
          <td>0.206349</td>
          <td>24.712005</td>
          <td>0.205467</td>
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
          <td>27.096406</td>
          <td>0.653907</td>
          <td>26.479615</td>
          <td>0.159419</td>
          <td>25.523010</td>
          <td>0.062156</td>
          <td>25.098733</td>
          <td>0.070230</td>
          <td>24.895947</td>
          <td>0.111067</td>
          <td>25.014953</td>
          <td>0.269269</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.542016</td>
          <td>1.527922</td>
          <td>26.397706</td>
          <td>0.146433</td>
          <td>26.202792</td>
          <td>0.111195</td>
          <td>25.040550</td>
          <td>0.065542</td>
          <td>24.803061</td>
          <td>0.100708</td>
          <td>24.411823</td>
          <td>0.159976</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.294559</td>
          <td>0.359581</td>
          <td>26.698907</td>
          <td>0.190646</td>
          <td>26.409737</td>
          <td>0.134216</td>
          <td>26.303115</td>
          <td>0.197901</td>
          <td>25.636062</td>
          <td>0.207654</td>
          <td>25.268908</td>
          <td>0.327789</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.788080</td>
          <td>0.528751</td>
          <td>26.078135</td>
          <td>0.113813</td>
          <td>25.927843</td>
          <td>0.089810</td>
          <td>25.823438</td>
          <td>0.133930</td>
          <td>25.752773</td>
          <td>0.232913</td>
          <td>24.646205</td>
          <td>0.200453</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.456177</td>
          <td>0.826294</td>
          <td>26.947857</td>
          <td>0.234144</td>
          <td>26.306001</td>
          <td>0.122352</td>
          <td>26.197736</td>
          <td>0.180570</td>
          <td>25.566470</td>
          <td>0.195360</td>
          <td>25.338657</td>
          <td>0.345510</td>
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
          <td>28.654011</td>
          <td>1.509183</td>
          <td>26.587150</td>
          <td>0.149285</td>
          <td>26.060318</td>
          <td>0.083167</td>
          <td>25.246159</td>
          <td>0.066053</td>
          <td>24.598057</td>
          <td>0.071212</td>
          <td>24.045865</td>
          <td>0.098406</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.781478</td>
          <td>2.446623</td>
          <td>27.394589</td>
          <td>0.293052</td>
          <td>26.828407</td>
          <td>0.162314</td>
          <td>26.046000</td>
          <td>0.133346</td>
          <td>25.858518</td>
          <td>0.212102</td>
          <td>24.871871</td>
          <td>0.200467</td>
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
          <td>28.944645</td>
          <td>0.871125</td>
          <td>26.224283</td>
          <td>0.168765</td>
          <td>25.015090</td>
          <td>0.111506</td>
          <td>24.364599</td>
          <td>0.141178</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.575266</td>
          <td>1.592983</td>
          <td>28.742731</td>
          <td>0.914142</td>
          <td>28.030116</td>
          <td>0.521131</td>
          <td>26.461004</td>
          <td>0.237254</td>
          <td>25.691166</td>
          <td>0.228129</td>
          <td>25.324619</td>
          <td>0.358914</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.267140</td>
          <td>0.314122</td>
          <td>26.371834</td>
          <td>0.124119</td>
          <td>26.005667</td>
          <td>0.079355</td>
          <td>25.768986</td>
          <td>0.104849</td>
          <td>25.335908</td>
          <td>0.136061</td>
          <td>25.128688</td>
          <td>0.248294</td>
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
          <td>27.198334</td>
          <td>0.660151</td>
          <td>26.367799</td>
          <td>0.132229</td>
          <td>25.452607</td>
          <td>0.052566</td>
          <td>25.191530</td>
          <td>0.068382</td>
          <td>25.030363</td>
          <td>0.112650</td>
          <td>25.000327</td>
          <td>0.240795</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.732404</td>
          <td>1.578466</td>
          <td>26.816223</td>
          <td>0.183968</td>
          <td>26.130652</td>
          <td>0.089943</td>
          <td>25.150210</td>
          <td>0.061728</td>
          <td>24.748557</td>
          <td>0.082685</td>
          <td>24.045600</td>
          <td>0.100064</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.183051</td>
          <td>0.642290</td>
          <td>26.826980</td>
          <td>0.190781</td>
          <td>26.479273</td>
          <td>0.125969</td>
          <td>26.204167</td>
          <td>0.160464</td>
          <td>26.042064</td>
          <td>0.258277</td>
          <td>25.321024</td>
          <td>0.303838</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.056527</td>
          <td>0.285694</td>
          <td>26.311631</td>
          <td>0.129986</td>
          <td>26.111662</td>
          <td>0.097586</td>
          <td>25.921224</td>
          <td>0.134556</td>
          <td>25.625734</td>
          <td>0.194463</td>
          <td>25.736495</td>
          <td>0.446174</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.189612</td>
          <td>0.255783</td>
          <td>26.450907</td>
          <td>0.121697</td>
          <td>26.367135</td>
          <td>0.182465</td>
          <td>26.524837</td>
          <td>0.376451</td>
          <td>25.755998</td>
          <td>0.423249</td>
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
