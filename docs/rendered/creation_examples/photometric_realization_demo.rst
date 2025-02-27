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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f04e59142e0>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.646199</td>
          <td>0.157012</td>
          <td>25.920946</td>
          <td>0.073527</td>
          <td>25.279461</td>
          <td>0.068022</td>
          <td>24.674928</td>
          <td>0.076211</td>
          <td>24.077032</td>
          <td>0.101116</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.674281</td>
          <td>1.524358</td>
          <td>27.466024</td>
          <td>0.310127</td>
          <td>26.519440</td>
          <td>0.124284</td>
          <td>26.194884</td>
          <td>0.151440</td>
          <td>26.119352</td>
          <td>0.262903</td>
          <td>25.295442</td>
          <td>0.284104</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.181021</td>
          <td>1.174638</td>
          <td>28.048790</td>
          <td>0.486447</td>
          <td>28.345655</td>
          <td>0.544207</td>
          <td>26.050862</td>
          <td>0.133777</td>
          <td>25.011572</td>
          <td>0.102478</td>
          <td>24.003585</td>
          <td>0.094810</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.371794</td>
          <td>0.711576</td>
          <td>27.159498</td>
          <td>0.241740</td>
          <td>27.235892</td>
          <td>0.228569</td>
          <td>26.075736</td>
          <td>0.136682</td>
          <td>25.520433</td>
          <td>0.159217</td>
          <td>25.198361</td>
          <td>0.262529</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.603331</td>
          <td>0.408323</td>
          <td>26.051816</td>
          <td>0.093772</td>
          <td>25.915133</td>
          <td>0.073150</td>
          <td>25.809379</td>
          <td>0.108455</td>
          <td>25.269001</td>
          <td>0.128233</td>
          <td>24.652405</td>
          <td>0.166333</td>
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
          <td>27.559176</td>
          <td>0.805637</td>
          <td>26.425536</td>
          <td>0.129868</td>
          <td>25.437957</td>
          <td>0.047909</td>
          <td>25.146214</td>
          <td>0.060443</td>
          <td>24.826689</td>
          <td>0.087125</td>
          <td>24.575486</td>
          <td>0.155756</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.843029</td>
          <td>2.501197</td>
          <td>26.714726</td>
          <td>0.166468</td>
          <td>25.992978</td>
          <td>0.078359</td>
          <td>25.255496</td>
          <td>0.066593</td>
          <td>25.019625</td>
          <td>0.103203</td>
          <td>24.205178</td>
          <td>0.113095</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.114749</td>
          <td>0.595644</td>
          <td>26.734663</td>
          <td>0.169317</td>
          <td>26.338647</td>
          <td>0.106177</td>
          <td>26.393683</td>
          <td>0.179420</td>
          <td>25.972410</td>
          <td>0.232968</td>
          <td>25.629910</td>
          <td>0.370666</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.801851</td>
          <td>0.474444</td>
          <td>26.046009</td>
          <td>0.093295</td>
          <td>26.185794</td>
          <td>0.092865</td>
          <td>25.822865</td>
          <td>0.109739</td>
          <td>25.750076</td>
          <td>0.193482</td>
          <td>25.082987</td>
          <td>0.238786</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.190111</td>
          <td>0.295053</td>
          <td>26.570233</td>
          <td>0.147117</td>
          <td>26.649127</td>
          <td>0.139039</td>
          <td>26.138085</td>
          <td>0.144228</td>
          <td>26.051708</td>
          <td>0.248722</td>
          <td>25.604766</td>
          <td>0.363458</td>
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
          <td>28.517654</td>
          <td>1.506874</td>
          <td>26.426515</td>
          <td>0.149547</td>
          <td>26.086777</td>
          <td>0.100057</td>
          <td>25.204533</td>
          <td>0.075445</td>
          <td>24.722196</td>
          <td>0.093426</td>
          <td>23.934470</td>
          <td>0.105419</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.414842</td>
          <td>0.338919</td>
          <td>26.537345</td>
          <td>0.147973</td>
          <td>26.137408</td>
          <td>0.169877</td>
          <td>25.979957</td>
          <td>0.272605</td>
          <td>25.927157</td>
          <td>0.535315</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.374496</td>
          <td>0.789082</td>
          <td>29.407940</td>
          <td>1.310474</td>
          <td>27.976185</td>
          <td>0.483795</td>
          <td>26.054236</td>
          <td>0.161799</td>
          <td>25.069082</td>
          <td>0.129262</td>
          <td>24.165478</td>
          <td>0.131825</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.282573</td>
          <td>0.762352</td>
          <td>27.687631</td>
          <td>0.441531</td>
          <td>27.250512</td>
          <td>0.286296</td>
          <td>26.243604</td>
          <td>0.198628</td>
          <td>25.526742</td>
          <td>0.199545</td>
          <td>24.918013</td>
          <td>0.259958</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.746739</td>
          <td>0.229190</td>
          <td>26.198034</td>
          <td>0.122836</td>
          <td>26.058862</td>
          <td>0.097671</td>
          <td>25.740222</td>
          <td>0.120722</td>
          <td>25.192696</td>
          <td>0.140751</td>
          <td>25.123474</td>
          <td>0.288347</td>
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
          <td>28.456776</td>
          <td>1.475026</td>
          <td>26.320973</td>
          <td>0.139140</td>
          <td>25.505830</td>
          <td>0.061217</td>
          <td>24.992817</td>
          <td>0.063944</td>
          <td>24.850822</td>
          <td>0.106778</td>
          <td>24.354637</td>
          <td>0.154925</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.202441</td>
          <td>0.332468</td>
          <td>26.533470</td>
          <td>0.164466</td>
          <td>26.048493</td>
          <td>0.097160</td>
          <td>25.063370</td>
          <td>0.066880</td>
          <td>24.813375</td>
          <td>0.101622</td>
          <td>24.111356</td>
          <td>0.123499</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.707209</td>
          <td>0.492371</td>
          <td>26.700752</td>
          <td>0.190943</td>
          <td>26.258297</td>
          <td>0.117704</td>
          <td>26.499808</td>
          <td>0.233194</td>
          <td>25.794906</td>
          <td>0.236984</td>
          <td>25.236111</td>
          <td>0.319345</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.386064</td>
          <td>0.391002</td>
          <td>26.223428</td>
          <td>0.129100</td>
          <td>26.126315</td>
          <td>0.106872</td>
          <td>25.973230</td>
          <td>0.152358</td>
          <td>25.627127</td>
          <td>0.209793</td>
          <td>25.458646</td>
          <td>0.386944</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.641090</td>
          <td>0.467923</td>
          <td>26.685940</td>
          <td>0.188124</td>
          <td>26.534123</td>
          <td>0.148990</td>
          <td>26.294298</td>
          <td>0.195905</td>
          <td>26.293590</td>
          <td>0.353497</td>
          <td>25.731548</td>
          <td>0.467341</td>
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
          <td>26.689512</td>
          <td>0.436073</td>
          <td>27.142382</td>
          <td>0.238376</td>
          <td>26.158879</td>
          <td>0.090706</td>
          <td>25.097665</td>
          <td>0.057903</td>
          <td>24.804006</td>
          <td>0.085413</td>
          <td>24.038014</td>
          <td>0.097731</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.524851</td>
          <td>3.131895</td>
          <td>28.264805</td>
          <td>0.569829</td>
          <td>26.826937</td>
          <td>0.162110</td>
          <td>26.204692</td>
          <td>0.152868</td>
          <td>25.853939</td>
          <td>0.211292</td>
          <td>25.309920</td>
          <td>0.287711</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.986619</td>
          <td>0.493576</td>
          <td>28.129832</td>
          <td>0.497475</td>
          <td>26.337836</td>
          <td>0.185830</td>
          <td>24.893336</td>
          <td>0.100250</td>
          <td>24.316470</td>
          <td>0.135438</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.109196</td>
          <td>0.279606</td>
          <td>27.335841</td>
          <td>0.305661</td>
          <td>26.801237</td>
          <td>0.312890</td>
          <td>25.945354</td>
          <td>0.281019</td>
          <td>25.956422</td>
          <td>0.576641</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.122852</td>
          <td>0.279712</td>
          <td>26.013901</td>
          <td>0.090815</td>
          <td>26.065505</td>
          <td>0.083656</td>
          <td>25.543098</td>
          <td>0.085990</td>
          <td>25.576094</td>
          <td>0.167193</td>
          <td>25.133149</td>
          <td>0.249207</td>
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
          <td>27.953890</td>
          <td>1.069316</td>
          <td>26.334772</td>
          <td>0.128507</td>
          <td>25.447200</td>
          <td>0.052315</td>
          <td>25.020317</td>
          <td>0.058754</td>
          <td>24.836736</td>
          <td>0.095100</td>
          <td>24.861298</td>
          <td>0.214556</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.375708</td>
          <td>0.719762</td>
          <td>26.532937</td>
          <td>0.144488</td>
          <td>26.042040</td>
          <td>0.083192</td>
          <td>25.201633</td>
          <td>0.064607</td>
          <td>24.912851</td>
          <td>0.095541</td>
          <td>24.137678</td>
          <td>0.108456</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.593673</td>
          <td>0.844827</td>
          <td>26.857400</td>
          <td>0.195730</td>
          <td>26.628390</td>
          <td>0.143289</td>
          <td>25.957658</td>
          <td>0.129802</td>
          <td>25.857030</td>
          <td>0.221690</td>
          <td>25.903581</td>
          <td>0.477265</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.571336</td>
          <td>0.427959</td>
          <td>26.213798</td>
          <td>0.119421</td>
          <td>26.073579</td>
          <td>0.094379</td>
          <td>25.706082</td>
          <td>0.111634</td>
          <td>25.811606</td>
          <td>0.227155</td>
          <td>25.962334</td>
          <td>0.527594</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.601637</td>
          <td>0.417591</td>
          <td>26.776828</td>
          <td>0.181327</td>
          <td>26.610453</td>
          <td>0.139714</td>
          <td>26.438458</td>
          <td>0.193790</td>
          <td>25.851765</td>
          <td>0.218641</td>
          <td>26.438678</td>
          <td>0.693332</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
