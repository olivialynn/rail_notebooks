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

    <pzflow.flow.Flow at 0x7f6501e68e80>



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
    0      23.994413  0.057429  0.040391  
    1      25.391064  0.015362  0.013653  
    2      24.304707  0.008037  0.004136  
    3      25.291103  0.079480  0.076744  
    4      25.096743  0.007100  0.004146  
    ...          ...       ...       ...  
    99995  24.737946  0.128194  0.110940  
    99996  24.224169  0.167377  0.158782  
    99997  25.613836  0.120680  0.065993  
    99998  25.274899  0.011891  0.010778  
    99999  25.699642  0.010922  0.008883  
    
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
          <td>26.758377</td>
          <td>0.172764</td>
          <td>26.078334</td>
          <td>0.084487</td>
          <td>25.133649</td>
          <td>0.059773</td>
          <td>24.685709</td>
          <td>0.076940</td>
          <td>23.885451</td>
          <td>0.085456</td>
          <td>0.057429</td>
          <td>0.040391</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.675238</td>
          <td>0.431345</td>
          <td>27.522401</td>
          <td>0.324394</td>
          <td>26.463076</td>
          <td>0.118345</td>
          <td>26.122255</td>
          <td>0.142276</td>
          <td>26.199481</td>
          <td>0.280624</td>
          <td>25.413859</td>
          <td>0.312509</td>
          <td>0.015362</td>
          <td>0.013653</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.168873</td>
          <td>1.031253</td>
          <td>27.619456</td>
          <td>0.312478</td>
          <td>26.044407</td>
          <td>0.133032</td>
          <td>25.109794</td>
          <td>0.111661</td>
          <td>24.095016</td>
          <td>0.102721</td>
          <td>0.008037</td>
          <td>0.004136</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.877432</td>
          <td>0.427676</td>
          <td>26.821040</td>
          <td>0.161147</td>
          <td>26.263018</td>
          <td>0.160537</td>
          <td>25.505711</td>
          <td>0.157225</td>
          <td>25.515664</td>
          <td>0.338861</td>
          <td>0.079480</td>
          <td>0.076744</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.520364</td>
          <td>0.383031</td>
          <td>26.174115</td>
          <td>0.104368</td>
          <td>25.840088</td>
          <td>0.068450</td>
          <td>25.628564</td>
          <td>0.092566</td>
          <td>25.288795</td>
          <td>0.130449</td>
          <td>24.949530</td>
          <td>0.213736</td>
          <td>0.007100</td>
          <td>0.004146</td>
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
          <td>28.508567</td>
          <td>1.401777</td>
          <td>26.466242</td>
          <td>0.134517</td>
          <td>25.425789</td>
          <td>0.047394</td>
          <td>25.185982</td>
          <td>0.062613</td>
          <td>24.842693</td>
          <td>0.088361</td>
          <td>24.594068</td>
          <td>0.158252</td>
          <td>0.128194</td>
          <td>0.110940</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.505824</td>
          <td>0.139188</td>
          <td>26.102899</td>
          <td>0.086335</td>
          <td>25.146014</td>
          <td>0.060432</td>
          <td>24.993782</td>
          <td>0.100894</td>
          <td>24.311305</td>
          <td>0.124030</td>
          <td>0.167377</td>
          <td>0.158782</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.571777</td>
          <td>1.447958</td>
          <td>26.619849</td>
          <td>0.153511</td>
          <td>26.386616</td>
          <td>0.110719</td>
          <td>26.660915</td>
          <td>0.224546</td>
          <td>25.689775</td>
          <td>0.183881</td>
          <td>25.465489</td>
          <td>0.325644</td>
          <td>0.120680</td>
          <td>0.065993</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.419362</td>
          <td>0.354024</td>
          <td>26.083495</td>
          <td>0.096412</td>
          <td>26.079971</td>
          <td>0.084609</td>
          <td>25.969827</td>
          <td>0.124712</td>
          <td>25.906891</td>
          <td>0.220635</td>
          <td>26.024623</td>
          <td>0.500211</td>
          <td>0.011891</td>
          <td>0.010778</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.119224</td>
          <td>0.597534</td>
          <td>26.850421</td>
          <td>0.186769</td>
          <td>26.554617</td>
          <td>0.128133</td>
          <td>26.572490</td>
          <td>0.208583</td>
          <td>25.646057</td>
          <td>0.177196</td>
          <td>25.747209</td>
          <td>0.405893</td>
          <td>0.010922</td>
          <td>0.008883</td>
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
          <td>26.973803</td>
          <td>0.595520</td>
          <td>26.615110</td>
          <td>0.176918</td>
          <td>26.029412</td>
          <td>0.095942</td>
          <td>25.193665</td>
          <td>0.075373</td>
          <td>24.591580</td>
          <td>0.083982</td>
          <td>24.001425</td>
          <td>0.112712</td>
          <td>0.057429</td>
          <td>0.040391</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.784252</td>
          <td>0.450974</td>
          <td>26.880432</td>
          <td>0.198175</td>
          <td>25.808311</td>
          <td>0.128115</td>
          <td>25.565983</td>
          <td>0.193540</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.015362</td>
          <td>0.013653</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.980602</td>
          <td>0.476357</td>
          <td>25.977290</td>
          <td>0.148129</td>
          <td>25.284449</td>
          <td>0.152266</td>
          <td>24.315657</td>
          <td>0.146724</td>
          <td>0.008037</td>
          <td>0.004136</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.222638</td>
          <td>1.305697</td>
          <td>27.778200</td>
          <td>0.455794</td>
          <td>27.906713</td>
          <td>0.458566</td>
          <td>26.332655</td>
          <td>0.204388</td>
          <td>25.525908</td>
          <td>0.190670</td>
          <td>25.060520</td>
          <td>0.279256</td>
          <td>0.079480</td>
          <td>0.076744</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.189940</td>
          <td>0.328270</td>
          <td>26.075813</td>
          <td>0.110433</td>
          <td>26.030567</td>
          <td>0.095254</td>
          <td>25.630300</td>
          <td>0.109677</td>
          <td>25.373704</td>
          <td>0.164338</td>
          <td>25.066958</td>
          <td>0.275376</td>
          <td>0.007100</td>
          <td>0.004146</td>
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
          <td>26.290214</td>
          <td>0.366754</td>
          <td>26.096248</td>
          <td>0.117208</td>
          <td>25.407840</td>
          <td>0.057594</td>
          <td>24.971914</td>
          <td>0.064469</td>
          <td>24.956178</td>
          <td>0.120047</td>
          <td>24.371324</td>
          <td>0.161216</td>
          <td>0.128194</td>
          <td>0.110940</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.540587</td>
          <td>0.177350</td>
          <td>26.137612</td>
          <td>0.113610</td>
          <td>25.184300</td>
          <td>0.080786</td>
          <td>24.610742</td>
          <td>0.092050</td>
          <td>24.346175</td>
          <td>0.163624</td>
          <td>0.167377</td>
          <td>0.158782</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.680671</td>
          <td>0.190719</td>
          <td>26.420363</td>
          <td>0.137885</td>
          <td>26.260353</td>
          <td>0.194360</td>
          <td>26.203071</td>
          <td>0.335353</td>
          <td>25.535646</td>
          <td>0.410399</td>
          <td>0.120680</td>
          <td>0.065993</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.377618</td>
          <td>0.780892</td>
          <td>26.368658</td>
          <td>0.142349</td>
          <td>26.079021</td>
          <td>0.099420</td>
          <td>25.903488</td>
          <td>0.139057</td>
          <td>25.621293</td>
          <td>0.202695</td>
          <td>25.792314</td>
          <td>0.484908</td>
          <td>0.011891</td>
          <td>0.010778</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.997331</td>
          <td>0.602559</td>
          <td>26.728133</td>
          <td>0.193297</td>
          <td>26.775164</td>
          <td>0.181268</td>
          <td>26.297082</td>
          <td>0.194480</td>
          <td>26.447482</td>
          <td>0.395079</td>
          <td>25.732493</td>
          <td>0.463707</td>
          <td>0.010922</td>
          <td>0.008883</td>
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
          <td>27.832568</td>
          <td>0.972407</td>
          <td>26.478243</td>
          <td>0.139721</td>
          <td>26.161281</td>
          <td>0.093870</td>
          <td>25.189558</td>
          <td>0.064998</td>
          <td>24.665028</td>
          <td>0.078039</td>
          <td>24.020082</td>
          <td>0.099466</td>
          <td>0.057429</td>
          <td>0.040391</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.315571</td>
          <td>0.275318</td>
          <td>26.626838</td>
          <td>0.136775</td>
          <td>26.223687</td>
          <td>0.155681</td>
          <td>25.394104</td>
          <td>0.143264</td>
          <td>25.375687</td>
          <td>0.303919</td>
          <td>0.015362</td>
          <td>0.013653</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.679797</td>
          <td>0.758509</td>
          <td>28.985669</td>
          <td>0.843073</td>
          <td>26.223593</td>
          <td>0.155301</td>
          <td>24.972435</td>
          <td>0.099080</td>
          <td>24.490715</td>
          <td>0.144909</td>
          <td>0.008037</td>
          <td>0.004136</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.009089</td>
          <td>0.980750</td>
          <td>27.691310</td>
          <td>0.354410</td>
          <td>26.269945</td>
          <td>0.174520</td>
          <td>25.343694</td>
          <td>0.147430</td>
          <td>25.295951</td>
          <td>0.305650</td>
          <td>0.079480</td>
          <td>0.076744</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.313157</td>
          <td>0.325647</td>
          <td>26.163280</td>
          <td>0.103426</td>
          <td>26.079019</td>
          <td>0.084577</td>
          <td>25.826304</td>
          <td>0.110122</td>
          <td>25.509845</td>
          <td>0.157852</td>
          <td>24.944578</td>
          <td>0.212950</td>
          <td>0.007100</td>
          <td>0.004146</td>
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
          <td>26.454060</td>
          <td>0.153369</td>
          <td>25.317427</td>
          <td>0.050800</td>
          <td>25.152676</td>
          <td>0.072201</td>
          <td>25.065085</td>
          <td>0.126241</td>
          <td>24.966235</td>
          <td>0.254054</td>
          <td>0.128194</td>
          <td>0.110940</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.707777</td>
          <td>0.208627</td>
          <td>26.138109</td>
          <td>0.116470</td>
          <td>25.196312</td>
          <td>0.083752</td>
          <td>24.645490</td>
          <td>0.097265</td>
          <td>24.367763</td>
          <td>0.170782</td>
          <td>0.167377</td>
          <td>0.158782</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.496166</td>
          <td>0.403263</td>
          <td>26.954003</td>
          <td>0.223586</td>
          <td>26.340207</td>
          <td>0.118677</td>
          <td>26.040456</td>
          <td>0.148494</td>
          <td>25.607974</td>
          <td>0.190856</td>
          <td>25.252897</td>
          <td>0.304921</td>
          <td>0.120680</td>
          <td>0.065993</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.679736</td>
          <td>0.194032</td>
          <td>26.174771</td>
          <td>0.104586</td>
          <td>26.119829</td>
          <td>0.087785</td>
          <td>25.829795</td>
          <td>0.110607</td>
          <td>25.823883</td>
          <td>0.206201</td>
          <td>25.047479</td>
          <td>0.232273</td>
          <td>0.011891</td>
          <td>0.010778</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.583963</td>
          <td>0.402626</td>
          <td>26.489780</td>
          <td>0.137434</td>
          <td>26.516863</td>
          <td>0.124171</td>
          <td>26.212788</td>
          <td>0.153995</td>
          <td>26.034573</td>
          <td>0.245548</td>
          <td>26.501526</td>
          <td>0.702108</td>
          <td>0.010922</td>
          <td>0.008883</td>
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
