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

    <pzflow.flow.Flow at 0x7f31dd9051b0>



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
    0      23.994413  0.162079  0.095662  
    1      25.391064  0.090771  0.048558  
    2      24.304707  0.337454  0.255903  
    3      25.291103  0.002558  0.001510  
    4      25.096743  0.056912  0.051876  
    ...          ...       ...       ...  
    99995  24.737946  0.128347  0.082136  
    99996  24.224169  0.035850  0.026665  
    99997  25.613836  0.077536  0.059093  
    99998  25.274899  0.147560  0.083350  
    99999  25.699642  0.041009  0.027920  
    
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
          <td>29.145608</td>
          <td>1.897395</td>
          <td>26.793871</td>
          <td>0.178046</td>
          <td>26.064270</td>
          <td>0.083446</td>
          <td>25.165749</td>
          <td>0.061500</td>
          <td>24.684615</td>
          <td>0.076866</td>
          <td>23.935636</td>
          <td>0.089315</td>
          <td>0.162079</td>
          <td>0.095662</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.466346</td>
          <td>0.310207</td>
          <td>26.619260</td>
          <td>0.135501</td>
          <td>26.498909</td>
          <td>0.196093</td>
          <td>26.936669</td>
          <td>0.497560</td>
          <td>25.096048</td>
          <td>0.241374</td>
          <td>0.090771</td>
          <td>0.048558</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.296957</td>
          <td>1.252622</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.590185</td>
          <td>1.209708</td>
          <td>26.003765</td>
          <td>0.128435</td>
          <td>24.987437</td>
          <td>0.100335</td>
          <td>24.296524</td>
          <td>0.122449</td>
          <td>0.337454</td>
          <td>0.255903</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.532571</td>
          <td>0.291408</td>
          <td>26.370332</td>
          <td>0.175901</td>
          <td>25.646699</td>
          <td>0.177292</td>
          <td>26.529637</td>
          <td>0.714815</td>
          <td>0.002558</td>
          <td>0.001510</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.975952</td>
          <td>0.247901</td>
          <td>26.214947</td>
          <td>0.108156</td>
          <td>26.055354</td>
          <td>0.082793</td>
          <td>25.747465</td>
          <td>0.102740</td>
          <td>25.582565</td>
          <td>0.167886</td>
          <td>24.990234</td>
          <td>0.221112</td>
          <td>0.056912</td>
          <td>0.051876</td>
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
          <td>27.093999</td>
          <td>0.586939</td>
          <td>26.301608</td>
          <td>0.116635</td>
          <td>25.520848</td>
          <td>0.051568</td>
          <td>25.096711</td>
          <td>0.057846</td>
          <td>25.088824</td>
          <td>0.109636</td>
          <td>24.829469</td>
          <td>0.193262</td>
          <td>0.128347</td>
          <td>0.082136</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.171370</td>
          <td>0.619891</td>
          <td>26.773991</td>
          <td>0.175070</td>
          <td>26.047495</td>
          <td>0.082221</td>
          <td>25.112771</td>
          <td>0.058676</td>
          <td>24.872862</td>
          <td>0.090737</td>
          <td>24.336659</td>
          <td>0.126787</td>
          <td>0.035850</td>
          <td>0.026665</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.419769</td>
          <td>0.354137</td>
          <td>26.574485</td>
          <td>0.147655</td>
          <td>26.438218</td>
          <td>0.115812</td>
          <td>26.066778</td>
          <td>0.135629</td>
          <td>25.896574</td>
          <td>0.218748</td>
          <td>26.190200</td>
          <td>0.564288</td>
          <td>0.077536</td>
          <td>0.059093</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.530834</td>
          <td>0.386149</td>
          <td>26.517821</td>
          <td>0.140634</td>
          <td>26.023790</td>
          <td>0.080520</td>
          <td>25.804704</td>
          <td>0.108013</td>
          <td>25.689076</td>
          <td>0.183772</td>
          <td>25.017749</td>
          <td>0.226228</td>
          <td>0.147560</td>
          <td>0.083350</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.596446</td>
          <td>0.406173</td>
          <td>27.264083</td>
          <td>0.263403</td>
          <td>26.639881</td>
          <td>0.137935</td>
          <td>26.318950</td>
          <td>0.168382</td>
          <td>26.164738</td>
          <td>0.272815</td>
          <td>25.332379</td>
          <td>0.292712</td>
          <td>0.041009</td>
          <td>0.027920</td>
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
          <td>27.277479</td>
          <td>0.754938</td>
          <td>26.478240</td>
          <td>0.164255</td>
          <td>26.137991</td>
          <td>0.110652</td>
          <td>25.201281</td>
          <td>0.079749</td>
          <td>24.567029</td>
          <td>0.086226</td>
          <td>24.145315</td>
          <td>0.134045</td>
          <td>0.162079</td>
          <td>0.095662</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.189663</td>
          <td>0.695662</td>
          <td>27.353168</td>
          <td>0.327361</td>
          <td>26.594622</td>
          <td>0.158078</td>
          <td>26.636829</td>
          <td>0.262237</td>
          <td>25.625745</td>
          <td>0.206814</td>
          <td>25.881464</td>
          <td>0.525514</td>
          <td>0.090771</td>
          <td>0.048558</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.059715</td>
          <td>0.656482</td>
          <td>28.301868</td>
          <td>0.722490</td>
          <td>25.805045</td>
          <td>0.161506</td>
          <td>25.009122</td>
          <td>0.151020</td>
          <td>24.265520</td>
          <td>0.177269</td>
          <td>0.337454</td>
          <td>0.255903</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.255381</td>
          <td>0.719788</td>
          <td>27.669164</td>
          <td>0.413010</td>
          <td>27.471079</td>
          <td>0.321422</td>
          <td>26.258680</td>
          <td>0.188224</td>
          <td>25.619406</td>
          <td>0.202292</td>
          <td>25.101243</td>
          <td>0.283116</td>
          <td>0.002558</td>
          <td>0.001510</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.321533</td>
          <td>0.366536</td>
          <td>26.112409</td>
          <td>0.115021</td>
          <td>25.943135</td>
          <td>0.089090</td>
          <td>25.730973</td>
          <td>0.120945</td>
          <td>25.523379</td>
          <td>0.188381</td>
          <td>25.167251</td>
          <td>0.301426</td>
          <td>0.056912</td>
          <td>0.051876</td>
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
          <td>27.818092</td>
          <td>1.048793</td>
          <td>26.815724</td>
          <td>0.214852</td>
          <td>25.580337</td>
          <td>0.066511</td>
          <td>24.951734</td>
          <td>0.062746</td>
          <td>24.792746</td>
          <td>0.103199</td>
          <td>24.602312</td>
          <td>0.194395</td>
          <td>0.128347</td>
          <td>0.082136</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.008818</td>
          <td>0.608592</td>
          <td>26.926369</td>
          <td>0.228712</td>
          <td>26.187865</td>
          <td>0.109668</td>
          <td>25.211795</td>
          <td>0.076197</td>
          <td>24.900022</td>
          <td>0.109527</td>
          <td>24.472396</td>
          <td>0.168321</td>
          <td>0.035850</td>
          <td>0.026665</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.229465</td>
          <td>0.713929</td>
          <td>26.966895</td>
          <td>0.239039</td>
          <td>26.630850</td>
          <td>0.162775</td>
          <td>26.627940</td>
          <td>0.259908</td>
          <td>26.179967</td>
          <td>0.324832</td>
          <td>25.935216</td>
          <td>0.545653</td>
          <td>0.077536</td>
          <td>0.059093</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.016237</td>
          <td>0.295085</td>
          <td>26.316742</td>
          <td>0.141763</td>
          <td>26.180095</td>
          <td>0.113643</td>
          <td>25.725821</td>
          <td>0.124920</td>
          <td>25.601725</td>
          <td>0.208256</td>
          <td>25.430079</td>
          <td>0.383501</td>
          <td>0.147560</td>
          <td>0.083350</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.885852</td>
          <td>0.557849</td>
          <td>27.555501</td>
          <td>0.379602</td>
          <td>26.438567</td>
          <td>0.136435</td>
          <td>26.238168</td>
          <td>0.185759</td>
          <td>25.508242</td>
          <td>0.184949</td>
          <td>26.483841</td>
          <td>0.789146</td>
          <td>0.041009</td>
          <td>0.027920</td>
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

.. parsed-literal::

    




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
          <td>25.898539</td>
          <td>0.264915</td>
          <td>26.886890</td>
          <td>0.225985</td>
          <td>25.890596</td>
          <td>0.086537</td>
          <td>25.328306</td>
          <td>0.086529</td>
          <td>24.721149</td>
          <td>0.095899</td>
          <td>24.110723</td>
          <td>0.126310</td>
          <td>0.162079</td>
          <td>0.095662</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.560529</td>
          <td>0.411591</td>
          <td>27.754961</td>
          <td>0.409504</td>
          <td>26.575636</td>
          <td>0.139213</td>
          <td>26.133264</td>
          <td>0.153632</td>
          <td>25.992570</td>
          <td>0.251945</td>
          <td>25.878723</td>
          <td>0.475494</td>
          <td>0.090771</td>
          <td>0.048558</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.576202</td>
          <td>0.543716</td>
          <td>26.746576</td>
          <td>0.267067</td>
          <td>25.959580</td>
          <td>0.223868</td>
          <td>24.611228</td>
          <td>0.130596</td>
          <td>24.116100</td>
          <td>0.190112</td>
          <td>0.337454</td>
          <td>0.255903</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.349369</td>
          <td>2.068319</td>
          <td>28.478018</td>
          <td>0.661563</td>
          <td>27.903468</td>
          <td>0.390736</td>
          <td>26.141094</td>
          <td>0.144610</td>
          <td>25.552980</td>
          <td>0.163714</td>
          <td>25.371605</td>
          <td>0.302120</td>
          <td>0.002558</td>
          <td>0.001510</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.273099</td>
          <td>0.323183</td>
          <td>26.052803</td>
          <td>0.097069</td>
          <td>26.116279</td>
          <td>0.090812</td>
          <td>25.450155</td>
          <td>0.082398</td>
          <td>25.396350</td>
          <td>0.148657</td>
          <td>25.019931</td>
          <td>0.235383</td>
          <td>0.056912</td>
          <td>0.051876</td>
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
          <td>26.684918</td>
          <td>0.472088</td>
          <td>26.574293</td>
          <td>0.165646</td>
          <td>25.460542</td>
          <td>0.055974</td>
          <td>25.034782</td>
          <td>0.063055</td>
          <td>24.930012</td>
          <td>0.109007</td>
          <td>24.512954</td>
          <td>0.168897</td>
          <td>0.128347</td>
          <td>0.082136</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.751822</td>
          <td>0.916747</td>
          <td>26.875057</td>
          <td>0.192826</td>
          <td>26.046993</td>
          <td>0.083296</td>
          <td>25.188048</td>
          <td>0.063624</td>
          <td>24.700730</td>
          <td>0.079022</td>
          <td>24.266223</td>
          <td>0.120918</td>
          <td>0.035850</td>
          <td>0.026665</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.536814</td>
          <td>0.402714</td>
          <td>26.978834</td>
          <td>0.218749</td>
          <td>26.356486</td>
          <td>0.114508</td>
          <td>26.343489</td>
          <td>0.182751</td>
          <td>26.247079</td>
          <td>0.308174</td>
          <td>25.311475</td>
          <td>0.304797</td>
          <td>0.077536</td>
          <td>0.059093</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.985074</td>
          <td>0.596534</td>
          <td>26.359978</td>
          <td>0.140898</td>
          <td>26.054012</td>
          <td>0.096907</td>
          <td>25.891011</td>
          <td>0.137026</td>
          <td>25.799710</td>
          <td>0.234361</td>
          <td>25.033065</td>
          <td>0.266985</td>
          <td>0.147560</td>
          <td>0.083350</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.015041</td>
          <td>0.559903</td>
          <td>27.015701</td>
          <td>0.217485</td>
          <td>26.429141</td>
          <td>0.116785</td>
          <td>26.358768</td>
          <td>0.177104</td>
          <td>26.907667</td>
          <td>0.493762</td>
          <td>25.621821</td>
          <td>0.373943</td>
          <td>0.041009</td>
          <td>0.027920</td>
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
