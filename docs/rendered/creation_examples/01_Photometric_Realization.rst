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

    <pzflow.flow.Flow at 0x7fec50641a80>



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
    0      23.994413  0.011752  0.009137  
    1      25.391064  0.083416  0.067999  
    2      24.304707  0.017905  0.016720  
    3      25.291103  0.008547  0.008256  
    4      25.096743  0.091809  0.071369  
    ...          ...       ...       ...  
    99995  24.737946  0.131665  0.099333  
    99996  24.224169  0.009546  0.008023  
    99997  25.613836  0.113038  0.108674  
    99998  25.274899  0.027364  0.022603  
    99999  25.699642  0.035173  0.024985  
    
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
          <td>27.317916</td>
          <td>0.686026</td>
          <td>26.725761</td>
          <td>0.168039</td>
          <td>26.191908</td>
          <td>0.093365</td>
          <td>25.221699</td>
          <td>0.064628</td>
          <td>24.729678</td>
          <td>0.079985</td>
          <td>23.932337</td>
          <td>0.089056</td>
          <td>0.011752</td>
          <td>0.009137</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.263974</td>
          <td>1.230159</td>
          <td>28.302989</td>
          <td>0.585187</td>
          <td>26.669505</td>
          <td>0.141502</td>
          <td>26.336488</td>
          <td>0.170914</td>
          <td>25.619171</td>
          <td>0.173197</td>
          <td>25.641386</td>
          <td>0.373995</td>
          <td>0.083416</td>
          <td>0.067999</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.239289</td>
          <td>1.074962</td>
          <td>28.408361</td>
          <td>0.569352</td>
          <td>26.002748</td>
          <td>0.128322</td>
          <td>25.117835</td>
          <td>0.112446</td>
          <td>24.424115</td>
          <td>0.136750</td>
          <td>0.017905</td>
          <td>0.016720</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.557434</td>
          <td>1.285447</td>
          <td>27.591084</td>
          <td>0.305457</td>
          <td>26.154766</td>
          <td>0.146312</td>
          <td>25.627053</td>
          <td>0.174360</td>
          <td>24.636637</td>
          <td>0.164112</td>
          <td>0.008547</td>
          <td>0.008256</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.324345</td>
          <td>0.328457</td>
          <td>26.218501</td>
          <td>0.108492</td>
          <td>25.972587</td>
          <td>0.076961</td>
          <td>25.417588</td>
          <td>0.076862</td>
          <td>25.240575</td>
          <td>0.125112</td>
          <td>24.835490</td>
          <td>0.194244</td>
          <td>0.091809</td>
          <td>0.071369</td>
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
          <td>26.442798</td>
          <td>0.360584</td>
          <td>26.347358</td>
          <td>0.121364</td>
          <td>25.529865</td>
          <td>0.051983</td>
          <td>24.981083</td>
          <td>0.052202</td>
          <td>24.990044</td>
          <td>0.100564</td>
          <td>25.006901</td>
          <td>0.224198</td>
          <td>0.131665</td>
          <td>0.099333</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.795386</td>
          <td>0.472162</td>
          <td>26.691284</td>
          <td>0.163175</td>
          <td>26.250998</td>
          <td>0.098335</td>
          <td>25.188580</td>
          <td>0.062758</td>
          <td>24.840584</td>
          <td>0.088197</td>
          <td>24.224379</td>
          <td>0.115003</td>
          <td>0.009546</td>
          <td>0.008023</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.083783</td>
          <td>0.582688</td>
          <td>26.366099</td>
          <td>0.123353</td>
          <td>26.239422</td>
          <td>0.097341</td>
          <td>26.251330</td>
          <td>0.158941</td>
          <td>25.920618</td>
          <td>0.223169</td>
          <td>26.716866</td>
          <td>0.809129</td>
          <td>0.113038</td>
          <td>0.108674</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.272457</td>
          <td>0.315176</td>
          <td>26.241083</td>
          <td>0.110649</td>
          <td>26.233445</td>
          <td>0.096832</td>
          <td>25.993102</td>
          <td>0.127254</td>
          <td>25.705686</td>
          <td>0.186371</td>
          <td>25.371439</td>
          <td>0.302062</td>
          <td>0.027364</td>
          <td>0.022603</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.061062</td>
          <td>0.573318</td>
          <td>27.106712</td>
          <td>0.231423</td>
          <td>26.715079</td>
          <td>0.147161</td>
          <td>26.398169</td>
          <td>0.180103</td>
          <td>25.757303</td>
          <td>0.194663</td>
          <td>25.715391</td>
          <td>0.396073</td>
          <td>0.035173</td>
          <td>0.024985</td>
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
          <td>27.240279</td>
          <td>0.712651</td>
          <td>26.563278</td>
          <td>0.168130</td>
          <td>26.028801</td>
          <td>0.095132</td>
          <td>25.173020</td>
          <td>0.073399</td>
          <td>24.673652</td>
          <td>0.089556</td>
          <td>24.042341</td>
          <td>0.115859</td>
          <td>0.011752</td>
          <td>0.009137</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.663379</td>
          <td>1.630974</td>
          <td>27.696703</td>
          <td>0.428213</td>
          <td>26.424040</td>
          <td>0.136756</td>
          <td>26.096619</td>
          <td>0.167250</td>
          <td>25.920807</td>
          <td>0.264411</td>
          <td>26.414724</td>
          <td>0.762897</td>
          <td>0.083416</td>
          <td>0.067999</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.296012</td>
          <td>2.137476</td>
          <td>28.869477</td>
          <td>0.949974</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.131423</td>
          <td>0.169147</td>
          <td>24.883342</td>
          <td>0.107690</td>
          <td>24.182626</td>
          <td>0.130941</td>
          <td>0.017905</td>
          <td>0.016720</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.518375</td>
          <td>0.367631</td>
          <td>27.632341</td>
          <td>0.365120</td>
          <td>26.074138</td>
          <td>0.160957</td>
          <td>25.600633</td>
          <td>0.199172</td>
          <td>25.908268</td>
          <td>0.528006</td>
          <td>0.008547</td>
          <td>0.008256</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.078146</td>
          <td>0.304976</td>
          <td>26.054774</td>
          <td>0.110626</td>
          <td>26.001223</td>
          <td>0.094932</td>
          <td>25.700858</td>
          <td>0.119336</td>
          <td>25.378641</td>
          <td>0.168647</td>
          <td>24.856411</td>
          <td>0.236746</td>
          <td>0.091809</td>
          <td>0.071369</td>
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
          <td>28.133751</td>
          <td>1.258565</td>
          <td>26.542786</td>
          <td>0.171686</td>
          <td>25.460292</td>
          <td>0.060184</td>
          <td>25.168458</td>
          <td>0.076511</td>
          <td>24.848560</td>
          <td>0.109040</td>
          <td>25.282923</td>
          <td>0.341179</td>
          <td>0.131665</td>
          <td>0.099333</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.355437</td>
          <td>1.387648</td>
          <td>27.140711</td>
          <td>0.272022</td>
          <td>26.125222</td>
          <td>0.103505</td>
          <td>25.212101</td>
          <td>0.075969</td>
          <td>24.695325</td>
          <td>0.091268</td>
          <td>24.291220</td>
          <td>0.143691</td>
          <td>0.009546</td>
          <td>0.008023</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.543488</td>
          <td>0.888269</td>
          <td>26.469261</td>
          <td>0.160707</td>
          <td>26.281834</td>
          <td>0.123431</td>
          <td>26.321271</td>
          <td>0.206437</td>
          <td>25.529908</td>
          <td>0.194978</td>
          <td>25.116056</td>
          <td>0.297581</td>
          <td>0.113038</td>
          <td>0.108674</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.982842</td>
          <td>0.278437</td>
          <td>26.051173</td>
          <td>0.108283</td>
          <td>26.129612</td>
          <td>0.104098</td>
          <td>26.201436</td>
          <td>0.179710</td>
          <td>25.541857</td>
          <td>0.189903</td>
          <td>25.924134</td>
          <td>0.535005</td>
          <td>0.027364</td>
          <td>0.022603</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.545031</td>
          <td>0.433325</td>
          <td>26.629290</td>
          <td>0.178254</td>
          <td>26.404703</td>
          <td>0.132369</td>
          <td>26.245492</td>
          <td>0.186723</td>
          <td>25.920725</td>
          <td>0.260451</td>
          <td>25.423022</td>
          <td>0.366825</td>
          <td>0.035173</td>
          <td>0.024985</td>
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
          <td>30.096967</td>
          <td>2.733350</td>
          <td>26.642020</td>
          <td>0.156651</td>
          <td>25.894484</td>
          <td>0.071935</td>
          <td>25.339710</td>
          <td>0.071863</td>
          <td>24.762130</td>
          <td>0.082432</td>
          <td>23.901867</td>
          <td>0.086836</td>
          <td>0.011752</td>
          <td>0.009137</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.529883</td>
          <td>0.821050</td>
          <td>27.353936</td>
          <td>0.300410</td>
          <td>26.679821</td>
          <td>0.153281</td>
          <td>26.571144</td>
          <td>0.223944</td>
          <td>25.846236</td>
          <td>0.224701</td>
          <td>27.198961</td>
          <td>1.143640</td>
          <td>0.083416</td>
          <td>0.067999</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.404096</td>
          <td>1.938628</td>
          <td>29.918486</td>
          <td>1.444182</td>
          <td>26.002509</td>
          <td>0.128836</td>
          <td>24.914494</td>
          <td>0.094500</td>
          <td>24.253104</td>
          <td>0.118409</td>
          <td>0.017905</td>
          <td>0.016720</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.605371</td>
          <td>3.208076</td>
          <td>28.395351</td>
          <td>0.625025</td>
          <td>27.455435</td>
          <td>0.273996</td>
          <td>26.264559</td>
          <td>0.160906</td>
          <td>25.547073</td>
          <td>0.163033</td>
          <td>25.001941</td>
          <td>0.223486</td>
          <td>0.008547</td>
          <td>0.008256</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.673655</td>
          <td>0.453443</td>
          <td>26.024425</td>
          <td>0.098514</td>
          <td>26.038567</td>
          <td>0.088725</td>
          <td>25.871962</td>
          <td>0.124906</td>
          <td>25.274407</td>
          <td>0.139877</td>
          <td>24.858110</td>
          <td>0.215031</td>
          <td>0.091809</td>
          <td>0.071369</td>
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
          <td>31.099685</td>
          <td>3.804607</td>
          <td>26.432883</td>
          <td>0.149507</td>
          <td>25.413195</td>
          <td>0.054832</td>
          <td>24.999630</td>
          <td>0.062493</td>
          <td>24.746826</td>
          <td>0.094835</td>
          <td>24.922897</td>
          <td>0.243165</td>
          <td>0.131665</td>
          <td>0.099333</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.284982</td>
          <td>0.671119</td>
          <td>26.995262</td>
          <td>0.211116</td>
          <td>26.185312</td>
          <td>0.092924</td>
          <td>25.270657</td>
          <td>0.067569</td>
          <td>24.637662</td>
          <td>0.073820</td>
          <td>24.244447</td>
          <td>0.117157</td>
          <td>0.009546</td>
          <td>0.008023</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.441056</td>
          <td>0.394617</td>
          <td>26.765756</td>
          <td>0.196347</td>
          <td>26.597512</td>
          <td>0.153027</td>
          <td>26.233108</td>
          <td>0.180878</td>
          <td>26.064950</td>
          <td>0.287130</td>
          <td>24.898565</td>
          <td>0.235680</td>
          <td>0.113038</td>
          <td>0.108674</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.812709</td>
          <td>0.480684</td>
          <td>26.314658</td>
          <td>0.118829</td>
          <td>26.126678</td>
          <td>0.088914</td>
          <td>25.677008</td>
          <td>0.097452</td>
          <td>25.995195</td>
          <td>0.239299</td>
          <td>24.907007</td>
          <td>0.208009</td>
          <td>0.027364</td>
          <td>0.022603</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.034376</td>
          <td>1.086200</td>
          <td>26.689027</td>
          <td>0.164583</td>
          <td>26.735385</td>
          <td>0.151586</td>
          <td>26.262268</td>
          <td>0.162489</td>
          <td>25.590597</td>
          <td>0.171090</td>
          <td>25.275663</td>
          <td>0.282947</td>
          <td>0.035173</td>
          <td>0.024985</td>
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
