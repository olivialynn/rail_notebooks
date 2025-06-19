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

    <pzflow.flow.Flow at 0x7f612aca50f0>



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
    0      23.994413  0.142008  0.090947  
    1      25.391064  0.000837  0.000541  
    2      24.304707  0.074986  0.046179  
    3      25.291103  0.077091  0.045773  
    4      25.096743  0.062620  0.039975  
    ...          ...       ...       ...  
    99995  24.737946  0.094581  0.081196  
    99996  24.224169  0.167594  0.158855  
    99997  25.613836  0.091752  0.063325  
    99998  25.274899  0.150087  0.148323  
    99999  25.699642  0.014218  0.012908  
    
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
          <td>29.390061</td>
          <td>2.103006</td>
          <td>26.530515</td>
          <td>0.142179</td>
          <td>26.040274</td>
          <td>0.081699</td>
          <td>25.166852</td>
          <td>0.061560</td>
          <td>24.715056</td>
          <td>0.078960</td>
          <td>23.922902</td>
          <td>0.088320</td>
          <td>0.142008</td>
          <td>0.090947</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.879682</td>
          <td>0.428408</td>
          <td>26.572380</td>
          <td>0.130119</td>
          <td>26.105957</td>
          <td>0.140292</td>
          <td>26.380343</td>
          <td>0.324508</td>
          <td>25.457472</td>
          <td>0.323574</td>
          <td>0.000837</td>
          <td>0.000541</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.261186</td>
          <td>2.687771</td>
          <td>28.856154</td>
          <td>0.774752</td>
          <td>25.885406</td>
          <td>0.115889</td>
          <td>25.240249</td>
          <td>0.125077</td>
          <td>24.141022</td>
          <td>0.106938</td>
          <td>0.074986</td>
          <td>0.046179</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.274509</td>
          <td>0.665926</td>
          <td>29.665621</td>
          <td>1.361640</td>
          <td>28.444480</td>
          <td>0.584233</td>
          <td>26.555457</td>
          <td>0.205629</td>
          <td>25.473889</td>
          <td>0.152998</td>
          <td>26.130547</td>
          <td>0.540511</td>
          <td>0.077091</td>
          <td>0.045773</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.927711</td>
          <td>0.238253</td>
          <td>26.083494</td>
          <td>0.096412</td>
          <td>26.070618</td>
          <td>0.083914</td>
          <td>25.668868</td>
          <td>0.095901</td>
          <td>25.484482</td>
          <td>0.154393</td>
          <td>24.970628</td>
          <td>0.217531</td>
          <td>0.062620</td>
          <td>0.039975</td>
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
          <td>27.590245</td>
          <td>0.822014</td>
          <td>26.671448</td>
          <td>0.160436</td>
          <td>25.437385</td>
          <td>0.047885</td>
          <td>25.050262</td>
          <td>0.055509</td>
          <td>24.890524</td>
          <td>0.092156</td>
          <td>24.885878</td>
          <td>0.202647</td>
          <td>0.094581</td>
          <td>0.081196</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.440488</td>
          <td>2.146303</td>
          <td>26.821851</td>
          <td>0.182314</td>
          <td>26.105250</td>
          <td>0.086514</td>
          <td>25.148903</td>
          <td>0.060588</td>
          <td>24.899374</td>
          <td>0.092876</td>
          <td>24.124625</td>
          <td>0.105416</td>
          <td>0.167594</td>
          <td>0.158855</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.443572</td>
          <td>0.360802</td>
          <td>26.553736</td>
          <td>0.145046</td>
          <td>26.320332</td>
          <td>0.104490</td>
          <td>26.157738</td>
          <td>0.146686</td>
          <td>25.776130</td>
          <td>0.197771</td>
          <td>25.009893</td>
          <td>0.224756</td>
          <td>0.091752</td>
          <td>0.063325</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.199767</td>
          <td>0.297353</td>
          <td>26.292224</td>
          <td>0.115687</td>
          <td>26.149047</td>
          <td>0.089914</td>
          <td>26.063798</td>
          <td>0.135280</td>
          <td>25.390552</td>
          <td>0.142427</td>
          <td>25.616689</td>
          <td>0.366861</td>
          <td>0.150087</td>
          <td>0.148323</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.453376</td>
          <td>0.363578</td>
          <td>26.840187</td>
          <td>0.185162</td>
          <td>26.488573</td>
          <td>0.120997</td>
          <td>26.324712</td>
          <td>0.169210</td>
          <td>25.815212</td>
          <td>0.204368</td>
          <td>25.669421</td>
          <td>0.382235</td>
          <td>0.014218</td>
          <td>0.012908</td>
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
          <td>26.281423</td>
          <td>0.364067</td>
          <td>27.085566</td>
          <td>0.270218</td>
          <td>25.962378</td>
          <td>0.093917</td>
          <td>25.189749</td>
          <td>0.078100</td>
          <td>24.555773</td>
          <td>0.084495</td>
          <td>24.137482</td>
          <td>0.131761</td>
          <td>0.142008</td>
          <td>0.090947</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.542120</td>
          <td>0.374426</td>
          <td>26.467363</td>
          <td>0.139293</td>
          <td>26.542286</td>
          <td>0.238535</td>
          <td>26.337271</td>
          <td>0.362545</td>
          <td>26.512859</td>
          <td>0.801681</td>
          <td>0.000837</td>
          <td>0.000541</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.588131</td>
          <td>0.801326</td>
          <td>27.614179</td>
          <td>0.364081</td>
          <td>26.110341</td>
          <td>0.168146</td>
          <td>24.947632</td>
          <td>0.115261</td>
          <td>24.294328</td>
          <td>0.145934</td>
          <td>0.074986</td>
          <td>0.046179</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.680355</td>
          <td>0.953114</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.470838</td>
          <td>0.325298</td>
          <td>26.459587</td>
          <td>0.225688</td>
          <td>25.758659</td>
          <td>0.230121</td>
          <td>26.078182</td>
          <td>0.603154</td>
          <td>0.077091</td>
          <td>0.045773</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.410460</td>
          <td>0.392534</td>
          <td>26.192259</td>
          <td>0.123202</td>
          <td>26.070147</td>
          <td>0.099527</td>
          <td>25.760548</td>
          <td>0.124001</td>
          <td>25.477878</td>
          <td>0.181150</td>
          <td>25.266135</td>
          <td>0.326001</td>
          <td>0.062620</td>
          <td>0.039975</td>
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
          <td>26.194153</td>
          <td>0.125232</td>
          <td>25.300038</td>
          <td>0.051249</td>
          <td>25.062528</td>
          <td>0.068352</td>
          <td>24.990080</td>
          <td>0.121114</td>
          <td>24.606590</td>
          <td>0.192805</td>
          <td>0.094581</td>
          <td>0.081196</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.933934</td>
          <td>0.606660</td>
          <td>26.983559</td>
          <td>0.256624</td>
          <td>25.956473</td>
          <td>0.096989</td>
          <td>25.241868</td>
          <td>0.085002</td>
          <td>24.894661</td>
          <td>0.117993</td>
          <td>24.189562</td>
          <td>0.143097</td>
          <td>0.167594</td>
          <td>0.158855</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.110320</td>
          <td>0.660081</td>
          <td>26.647524</td>
          <td>0.183807</td>
          <td>26.385898</td>
          <td>0.132500</td>
          <td>26.318570</td>
          <td>0.202049</td>
          <td>26.160138</td>
          <td>0.321125</td>
          <td>27.937263</td>
          <td>1.778979</td>
          <td>0.091752</td>
          <td>0.063325</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.219182</td>
          <td>0.732287</td>
          <td>26.254543</td>
          <td>0.137328</td>
          <td>25.941478</td>
          <td>0.094510</td>
          <td>26.077103</td>
          <td>0.173118</td>
          <td>25.936706</td>
          <td>0.280820</td>
          <td>26.646926</td>
          <td>0.919898</td>
          <td>0.150087</td>
          <td>0.148323</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.886031</td>
          <td>1.070920</td>
          <td>26.614341</td>
          <td>0.175622</td>
          <td>26.384624</td>
          <td>0.129766</td>
          <td>26.562718</td>
          <td>0.242738</td>
          <td>25.422509</td>
          <td>0.171397</td>
          <td>26.082726</td>
          <td>0.598712</td>
          <td>0.014218</td>
          <td>0.012908</td>
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
          <td>26.730763</td>
          <td>0.496333</td>
          <td>27.225072</td>
          <td>0.290635</td>
          <td>25.903498</td>
          <td>0.084947</td>
          <td>25.199546</td>
          <td>0.074890</td>
          <td>24.680990</td>
          <td>0.089859</td>
          <td>24.015494</td>
          <td>0.112816</td>
          <td>0.142008</td>
          <td>0.090947</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.675409</td>
          <td>0.365984</td>
          <td>26.696873</td>
          <td>0.144877</td>
          <td>26.124262</td>
          <td>0.142523</td>
          <td>25.788284</td>
          <td>0.199803</td>
          <td>25.125691</td>
          <td>0.247343</td>
          <td>0.000837</td>
          <td>0.000541</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.408860</td>
          <td>2.153239</td>
          <td>28.627217</td>
          <td>0.756534</td>
          <td>28.475383</td>
          <td>0.621372</td>
          <td>26.061549</td>
          <td>0.142073</td>
          <td>24.946454</td>
          <td>0.101695</td>
          <td>24.142893</td>
          <td>0.112723</td>
          <td>0.074986</td>
          <td>0.046179</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.439750</td>
          <td>3.089624</td>
          <td>27.547252</td>
          <td>0.344439</td>
          <td>27.338828</td>
          <td>0.261017</td>
          <td>25.987330</td>
          <td>0.133483</td>
          <td>25.401755</td>
          <td>0.151177</td>
          <td>25.689997</td>
          <td>0.406868</td>
          <td>0.077091</td>
          <td>0.045773</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.973527</td>
          <td>0.253295</td>
          <td>26.202262</td>
          <td>0.110345</td>
          <td>25.891849</td>
          <td>0.074299</td>
          <td>25.669120</td>
          <td>0.099596</td>
          <td>25.554137</td>
          <td>0.169664</td>
          <td>24.804798</td>
          <td>0.196166</td>
          <td>0.062620</td>
          <td>0.039975</td>
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
          <td>27.127348</td>
          <td>0.634453</td>
          <td>26.299287</td>
          <td>0.126434</td>
          <td>25.400972</td>
          <td>0.051060</td>
          <td>25.107805</td>
          <td>0.064608</td>
          <td>24.898616</td>
          <td>0.102069</td>
          <td>25.208502</td>
          <td>0.289970</td>
          <td>0.094581</td>
          <td>0.081196</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.991888</td>
          <td>1.201205</td>
          <td>26.715943</td>
          <td>0.210123</td>
          <td>26.000137</td>
          <td>0.103301</td>
          <td>25.188367</td>
          <td>0.083199</td>
          <td>24.961491</td>
          <td>0.128146</td>
          <td>24.084583</td>
          <td>0.134021</td>
          <td>0.167594</td>
          <td>0.158855</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.304040</td>
          <td>0.339577</td>
          <td>26.651665</td>
          <td>0.168451</td>
          <td>26.288299</td>
          <td>0.109709</td>
          <td>26.027261</td>
          <td>0.141887</td>
          <td>26.070188</td>
          <td>0.271282</td>
          <td>26.115419</td>
          <td>0.570830</td>
          <td>0.091752</td>
          <td>0.063325</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.332429</td>
          <td>0.385933</td>
          <td>26.061691</td>
          <td>0.116477</td>
          <td>26.281643</td>
          <td>0.127466</td>
          <td>25.686348</td>
          <td>0.124066</td>
          <td>25.401217</td>
          <td>0.180491</td>
          <td>25.596485</td>
          <td>0.446147</td>
          <td>0.150087</td>
          <td>0.148323</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.774400</td>
          <td>0.465503</td>
          <td>26.444736</td>
          <td>0.132324</td>
          <td>26.578893</td>
          <td>0.131177</td>
          <td>26.300179</td>
          <td>0.166133</td>
          <td>25.717036</td>
          <td>0.188618</td>
          <td>25.055891</td>
          <td>0.234068</td>
          <td>0.014218</td>
          <td>0.012908</td>
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
