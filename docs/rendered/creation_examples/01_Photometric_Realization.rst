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

    <pzflow.flow.Flow at 0x7f0783eceef0>



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
    0      23.994413  0.024385  0.019841  
    1      25.391064  0.071486  0.070938  
    2      24.304707  0.085952  0.055667  
    3      25.291103  0.150592  0.141702  
    4      25.096743  0.084053  0.071618  
    ...          ...       ...       ...  
    99995  24.737946  0.090370  0.055446  
    99996  24.224169  0.024265  0.021187  
    99997  25.613836  0.072526  0.070511  
    99998  25.274899  0.042995  0.034293  
    99999  25.699642  0.066706  0.042936  
    
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
          <td>28.244410</td>
          <td>1.216939</td>
          <td>26.852068</td>
          <td>0.187029</td>
          <td>25.950212</td>
          <td>0.075454</td>
          <td>25.098769</td>
          <td>0.057951</td>
          <td>24.718681</td>
          <td>0.079213</td>
          <td>24.120626</td>
          <td>0.105048</td>
          <td>0.024385</td>
          <td>0.019841</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.933805</td>
          <td>0.446339</td>
          <td>26.604065</td>
          <td>0.133734</td>
          <td>26.354432</td>
          <td>0.173542</td>
          <td>26.286103</td>
          <td>0.300952</td>
          <td>25.679038</td>
          <td>0.385096</td>
          <td>0.071486</td>
          <td>0.070938</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.775575</td>
          <td>0.924283</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.840686</td>
          <td>0.766890</td>
          <td>25.895091</td>
          <td>0.116870</td>
          <td>25.225820</td>
          <td>0.123521</td>
          <td>24.061155</td>
          <td>0.099720</td>
          <td>0.085952</td>
          <td>0.055667</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.808750</td>
          <td>0.943412</td>
          <td>28.103364</td>
          <td>0.506461</td>
          <td>27.215155</td>
          <td>0.224668</td>
          <td>26.296943</td>
          <td>0.165254</td>
          <td>25.347965</td>
          <td>0.137293</td>
          <td>25.191432</td>
          <td>0.261046</td>
          <td>0.150592</td>
          <td>0.141702</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.425776</td>
          <td>0.355809</td>
          <td>26.077327</td>
          <td>0.095893</td>
          <td>25.897918</td>
          <td>0.072045</td>
          <td>25.683469</td>
          <td>0.097138</td>
          <td>25.575868</td>
          <td>0.166931</td>
          <td>24.963090</td>
          <td>0.216168</td>
          <td>0.084053</td>
          <td>0.071618</td>
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
          <td>26.985390</td>
          <td>0.542940</td>
          <td>26.585989</td>
          <td>0.149120</td>
          <td>25.460850</td>
          <td>0.048893</td>
          <td>25.060829</td>
          <td>0.056032</td>
          <td>24.869551</td>
          <td>0.090473</td>
          <td>24.668394</td>
          <td>0.168614</td>
          <td>0.090370</td>
          <td>0.055446</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.297633</td>
          <td>0.676580</td>
          <td>27.016683</td>
          <td>0.214738</td>
          <td>26.056208</td>
          <td>0.082855</td>
          <td>25.276531</td>
          <td>0.067845</td>
          <td>24.964719</td>
          <td>0.098357</td>
          <td>24.256843</td>
          <td>0.118299</td>
          <td>0.024265</td>
          <td>0.021187</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.337247</td>
          <td>0.695117</td>
          <td>26.465375</td>
          <td>0.134417</td>
          <td>26.312919</td>
          <td>0.103815</td>
          <td>26.549652</td>
          <td>0.204631</td>
          <td>25.648616</td>
          <td>0.177581</td>
          <td>26.012561</td>
          <td>0.495775</td>
          <td>0.072526</td>
          <td>0.070511</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.874236</td>
          <td>0.500592</td>
          <td>26.245247</td>
          <td>0.111051</td>
          <td>26.049365</td>
          <td>0.082357</td>
          <td>25.990288</td>
          <td>0.126944</td>
          <td>26.189445</td>
          <td>0.278348</td>
          <td>25.803807</td>
          <td>0.423855</td>
          <td>0.042995</td>
          <td>0.034293</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.445985</td>
          <td>1.356778</td>
          <td>26.480466</td>
          <td>0.136179</td>
          <td>26.585618</td>
          <td>0.131618</td>
          <td>26.382478</td>
          <td>0.177723</td>
          <td>26.557592</td>
          <td>0.373111</td>
          <td>25.087769</td>
          <td>0.239731</td>
          <td>0.066706</td>
          <td>0.042936</td>
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
          <td>27.065914</td>
          <td>0.256214</td>
          <td>25.979898</td>
          <td>0.091252</td>
          <td>25.214410</td>
          <td>0.076237</td>
          <td>24.769939</td>
          <td>0.097582</td>
          <td>23.926065</td>
          <td>0.104824</td>
          <td>0.024385</td>
          <td>0.019841</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.073212</td>
          <td>0.641909</td>
          <td>27.584235</td>
          <td>0.392120</td>
          <td>26.442066</td>
          <td>0.138580</td>
          <td>26.123242</td>
          <td>0.170682</td>
          <td>25.519254</td>
          <td>0.188988</td>
          <td>25.667705</td>
          <td>0.448169</td>
          <td>0.071486</td>
          <td>0.070938</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.169752</td>
          <td>0.326948</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.770741</td>
          <td>0.836599</td>
          <td>26.010111</td>
          <td>0.155053</td>
          <td>24.941581</td>
          <td>0.115167</td>
          <td>24.240746</td>
          <td>0.139989</td>
          <td>0.085952</td>
          <td>0.055667</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.674515</td>
          <td>0.878279</td>
          <td>28.911497</td>
          <td>0.947392</td>
          <td>26.213242</td>
          <td>0.193742</td>
          <td>25.930608</td>
          <td>0.278760</td>
          <td>24.607144</td>
          <td>0.201021</td>
          <td>0.150592</td>
          <td>0.141702</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.080193</td>
          <td>0.305017</td>
          <td>26.390420</td>
          <td>0.147609</td>
          <td>25.994390</td>
          <td>0.094163</td>
          <td>25.900950</td>
          <td>0.141584</td>
          <td>25.329455</td>
          <td>0.161385</td>
          <td>24.959229</td>
          <td>0.257115</td>
          <td>0.084053</td>
          <td>0.071618</td>
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
          <td>27.399472</td>
          <td>0.800388</td>
          <td>26.219225</td>
          <td>0.127178</td>
          <td>25.529036</td>
          <td>0.062343</td>
          <td>25.001398</td>
          <td>0.064276</td>
          <td>24.635028</td>
          <td>0.088170</td>
          <td>24.432957</td>
          <td>0.165264</td>
          <td>0.090370</td>
          <td>0.055446</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.403349</td>
          <td>0.794750</td>
          <td>26.679811</td>
          <td>0.185810</td>
          <td>25.875629</td>
          <td>0.083259</td>
          <td>25.204487</td>
          <td>0.075578</td>
          <td>24.880790</td>
          <td>0.107530</td>
          <td>24.098126</td>
          <td>0.121787</td>
          <td>0.024265</td>
          <td>0.021187</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.281936</td>
          <td>0.740031</td>
          <td>26.822303</td>
          <td>0.212224</td>
          <td>26.502831</td>
          <td>0.146043</td>
          <td>26.801800</td>
          <td>0.299627</td>
          <td>26.379253</td>
          <td>0.380333</td>
          <td>25.247851</td>
          <td>0.323641</td>
          <td>0.072526</td>
          <td>0.070511</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.206142</td>
          <td>0.333654</td>
          <td>26.446318</td>
          <td>0.152790</td>
          <td>26.109729</td>
          <td>0.102608</td>
          <td>25.840731</td>
          <td>0.132353</td>
          <td>25.520912</td>
          <td>0.187111</td>
          <td>26.659010</td>
          <td>0.883748</td>
          <td>0.042995</td>
          <td>0.034293</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.091886</td>
          <td>0.647779</td>
          <td>26.854102</td>
          <td>0.216708</td>
          <td>26.863786</td>
          <td>0.197275</td>
          <td>26.279533</td>
          <td>0.193585</td>
          <td>25.911098</td>
          <td>0.260199</td>
          <td>26.185570</td>
          <td>0.648701</td>
          <td>0.066706</td>
          <td>0.042936</td>
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
          <td>29.011178</td>
          <td>1.791873</td>
          <td>26.521016</td>
          <td>0.141824</td>
          <td>25.978922</td>
          <td>0.077914</td>
          <td>25.169568</td>
          <td>0.062147</td>
          <td>24.664880</td>
          <td>0.076045</td>
          <td>23.818215</td>
          <td>0.081101</td>
          <td>0.024385</td>
          <td>0.019841</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.383879</td>
          <td>1.349357</td>
          <td>27.069300</td>
          <td>0.236584</td>
          <td>26.656755</td>
          <td>0.149067</td>
          <td>26.187959</td>
          <td>0.160769</td>
          <td>25.610703</td>
          <td>0.182971</td>
          <td>25.095978</td>
          <td>0.256916</td>
          <td>0.071486</td>
          <td>0.070938</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.779017</td>
          <td>1.393631</td>
          <td>25.974616</td>
          <td>0.134021</td>
          <td>25.047355</td>
          <td>0.112872</td>
          <td>24.157886</td>
          <td>0.116124</td>
          <td>0.085952</td>
          <td>0.055667</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.521483</td>
          <td>0.388283</td>
          <td>27.318596</td>
          <td>0.301949</td>
          <td>26.192552</td>
          <td>0.189934</td>
          <td>25.449072</td>
          <td>0.186588</td>
          <td>25.616571</td>
          <td>0.449967</td>
          <td>0.150592</td>
          <td>0.141702</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.523309</td>
          <td>0.402581</td>
          <td>26.051379</td>
          <td>0.100207</td>
          <td>26.142812</td>
          <td>0.096518</td>
          <td>25.684838</td>
          <td>0.105316</td>
          <td>25.495176</td>
          <td>0.167795</td>
          <td>25.286511</td>
          <td>0.303285</td>
          <td>0.084053</td>
          <td>0.071618</td>
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
          <td>27.126506</td>
          <td>0.624956</td>
          <td>26.236130</td>
          <td>0.117080</td>
          <td>25.386492</td>
          <td>0.049132</td>
          <td>25.137580</td>
          <td>0.064589</td>
          <td>24.862187</td>
          <td>0.096399</td>
          <td>24.685815</td>
          <td>0.183537</td>
          <td>0.090370</td>
          <td>0.055446</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.183617</td>
          <td>0.627649</td>
          <td>27.110972</td>
          <td>0.233573</td>
          <td>25.888726</td>
          <td>0.071967</td>
          <td>25.050389</td>
          <td>0.055930</td>
          <td>24.903285</td>
          <td>0.093850</td>
          <td>24.195214</td>
          <td>0.112930</td>
          <td>0.024265</td>
          <td>0.021187</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.681990</td>
          <td>0.450991</td>
          <td>26.519663</td>
          <td>0.148907</td>
          <td>26.435054</td>
          <td>0.123161</td>
          <td>26.168813</td>
          <td>0.158244</td>
          <td>25.770517</td>
          <td>0.209410</td>
          <td>25.959693</td>
          <td>0.504518</td>
          <td>0.072526</td>
          <td>0.070511</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.858436</td>
          <td>0.500643</td>
          <td>26.355194</td>
          <td>0.124321</td>
          <td>26.137600</td>
          <td>0.090824</td>
          <td>26.091603</td>
          <td>0.141465</td>
          <td>25.709646</td>
          <td>0.190638</td>
          <td>25.138387</td>
          <td>0.254860</td>
          <td>0.042995</td>
          <td>0.034293</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.449341</td>
          <td>0.765837</td>
          <td>26.840111</td>
          <td>0.191572</td>
          <td>26.401224</td>
          <td>0.116758</td>
          <td>26.001466</td>
          <td>0.133679</td>
          <td>25.946916</td>
          <td>0.237002</td>
          <td>25.434370</td>
          <td>0.330048</td>
          <td>0.066706</td>
          <td>0.042936</td>
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
