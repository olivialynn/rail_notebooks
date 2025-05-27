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

    <pzflow.flow.Flow at 0x7f72ecd48d00>



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
    0      23.994413  0.183595  0.150560  
    1      25.391064  0.025078  0.023607  
    2      24.304707  0.018229  0.013905  
    3      25.291103  0.048824  0.046515  
    4      25.096743  0.005488  0.003627  
    ...          ...       ...       ...  
    99995  24.737946  0.005325  0.004949  
    99996  24.224169  0.083398  0.081778  
    99997  25.613836  0.119800  0.080097  
    99998  25.274899  0.028734  0.027458  
    99999  25.699642  0.010735  0.007655  
    
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
          <td>27.172187</td>
          <td>0.620246</td>
          <td>26.795872</td>
          <td>0.178348</td>
          <td>25.923332</td>
          <td>0.073682</td>
          <td>25.061020</td>
          <td>0.056042</td>
          <td>24.650738</td>
          <td>0.074599</td>
          <td>24.043438</td>
          <td>0.098183</td>
          <td>0.183595</td>
          <td>0.150560</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.599085</td>
          <td>0.344704</td>
          <td>26.610804</td>
          <td>0.134515</td>
          <td>26.248588</td>
          <td>0.158569</td>
          <td>26.284300</td>
          <td>0.300516</td>
          <td>24.967385</td>
          <td>0.216943</td>
          <td>0.025078</td>
          <td>0.023607</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.768594</td>
          <td>0.803757</td>
          <td>28.351990</td>
          <td>0.546708</td>
          <td>26.063271</td>
          <td>0.135219</td>
          <td>25.252194</td>
          <td>0.126379</td>
          <td>24.247821</td>
          <td>0.117374</td>
          <td>0.018229</td>
          <td>0.013905</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.467789</td>
          <td>0.656883</td>
          <td>27.623739</td>
          <td>0.313550</td>
          <td>26.527826</td>
          <td>0.200918</td>
          <td>25.445267</td>
          <td>0.149287</td>
          <td>25.396686</td>
          <td>0.308243</td>
          <td>0.048824</td>
          <td>0.046515</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.551720</td>
          <td>0.392432</td>
          <td>26.261096</td>
          <td>0.112595</td>
          <td>25.952324</td>
          <td>0.075595</td>
          <td>25.438388</td>
          <td>0.078288</td>
          <td>25.014886</td>
          <td>0.102776</td>
          <td>25.477656</td>
          <td>0.328808</td>
          <td>0.005488</td>
          <td>0.003627</td>
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
          <td>26.610030</td>
          <td>0.410424</td>
          <td>26.402408</td>
          <td>0.127295</td>
          <td>25.401448</td>
          <td>0.046381</td>
          <td>25.080431</td>
          <td>0.057016</td>
          <td>24.743693</td>
          <td>0.080980</td>
          <td>25.188683</td>
          <td>0.260460</td>
          <td>0.005325</td>
          <td>0.004949</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.342550</td>
          <td>0.697625</td>
          <td>26.642395</td>
          <td>0.156502</td>
          <td>26.131912</td>
          <td>0.088568</td>
          <td>25.128475</td>
          <td>0.059499</td>
          <td>24.759518</td>
          <td>0.082119</td>
          <td>24.359826</td>
          <td>0.129357</td>
          <td>0.083398</td>
          <td>0.081778</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.144968</td>
          <td>0.608494</td>
          <td>26.626577</td>
          <td>0.154398</td>
          <td>26.249064</td>
          <td>0.098168</td>
          <td>26.420119</td>
          <td>0.183481</td>
          <td>25.820604</td>
          <td>0.205294</td>
          <td>25.464566</td>
          <td>0.325405</td>
          <td>0.119800</td>
          <td>0.080097</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.966653</td>
          <td>0.246014</td>
          <td>26.263947</td>
          <td>0.112875</td>
          <td>26.124319</td>
          <td>0.087979</td>
          <td>25.960327</td>
          <td>0.123688</td>
          <td>25.672705</td>
          <td>0.181243</td>
          <td>24.939616</td>
          <td>0.211973</td>
          <td>0.028734</td>
          <td>0.027458</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.194965</td>
          <td>0.630209</td>
          <td>26.598671</td>
          <td>0.150750</td>
          <td>26.541757</td>
          <td>0.126713</td>
          <td>26.366690</td>
          <td>0.175358</td>
          <td>26.123452</td>
          <td>0.263785</td>
          <td>26.659446</td>
          <td>0.779347</td>
          <td>0.010735</td>
          <td>0.007655</td>
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
          <td>26.719249</td>
          <td>0.206920</td>
          <td>26.061394</td>
          <td>0.106752</td>
          <td>25.064062</td>
          <td>0.072971</td>
          <td>24.553683</td>
          <td>0.087922</td>
          <td>24.155614</td>
          <td>0.139549</td>
          <td>0.183595</td>
          <td>0.150560</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.189145</td>
          <td>2.046919</td>
          <td>27.115202</td>
          <td>0.266820</td>
          <td>26.946350</td>
          <td>0.209695</td>
          <td>26.381373</td>
          <td>0.209082</td>
          <td>25.835116</td>
          <td>0.242510</td>
          <td>24.735223</td>
          <td>0.209840</td>
          <td>0.025078</td>
          <td>0.023607</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.930330</td>
          <td>0.985785</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.189910</td>
          <td>0.177741</td>
          <td>25.041151</td>
          <td>0.123532</td>
          <td>24.662377</td>
          <td>0.197189</td>
          <td>0.018229</td>
          <td>0.013905</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.795103</td>
          <td>0.523498</td>
          <td>32.080966</td>
          <td>3.621768</td>
          <td>27.143682</td>
          <td>0.248312</td>
          <td>26.241426</td>
          <td>0.186923</td>
          <td>25.232277</td>
          <td>0.146676</td>
          <td>25.145185</td>
          <td>0.295479</td>
          <td>0.048824</td>
          <td>0.046515</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.835619</td>
          <td>0.246564</td>
          <td>26.176249</td>
          <td>0.120507</td>
          <td>25.882697</td>
          <td>0.083637</td>
          <td>25.787645</td>
          <td>0.125758</td>
          <td>25.159320</td>
          <td>0.136721</td>
          <td>25.424609</td>
          <td>0.366246</td>
          <td>0.005488</td>
          <td>0.003627</td>
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
          <td>28.020218</td>
          <td>1.156641</td>
          <td>26.442100</td>
          <td>0.151566</td>
          <td>25.427169</td>
          <td>0.055895</td>
          <td>25.099649</td>
          <td>0.068767</td>
          <td>24.923524</td>
          <td>0.111429</td>
          <td>25.044200</td>
          <td>0.270318</td>
          <td>0.005325</td>
          <td>0.004949</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.643499</td>
          <td>0.183507</td>
          <td>26.044079</td>
          <td>0.098590</td>
          <td>25.224884</td>
          <td>0.078649</td>
          <td>25.007958</td>
          <td>0.122638</td>
          <td>24.228930</td>
          <td>0.139300</td>
          <td>0.083398</td>
          <td>0.081778</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.024544</td>
          <td>0.626980</td>
          <td>26.764356</td>
          <td>0.205150</td>
          <td>26.298734</td>
          <td>0.124491</td>
          <td>26.251516</td>
          <td>0.193504</td>
          <td>25.686143</td>
          <td>0.220910</td>
          <td>25.219182</td>
          <td>0.321332</td>
          <td>0.119800</td>
          <td>0.080097</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.263508</td>
          <td>0.348516</td>
          <td>26.173009</td>
          <td>0.120449</td>
          <td>25.999387</td>
          <td>0.092921</td>
          <td>26.089815</td>
          <td>0.163527</td>
          <td>26.059429</td>
          <td>0.291394</td>
          <td>24.826020</td>
          <td>0.226478</td>
          <td>0.028734</td>
          <td>0.027458</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.092141</td>
          <td>0.303683</td>
          <td>26.755777</td>
          <td>0.197836</td>
          <td>26.383551</td>
          <td>0.129603</td>
          <td>26.180880</td>
          <td>0.176279</td>
          <td>26.109065</td>
          <td>0.302612</td>
          <td>25.743779</td>
          <td>0.467624</td>
          <td>0.010735</td>
          <td>0.007655</td>
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
          <td>28.219666</td>
          <td>1.365432</td>
          <td>26.424249</td>
          <td>0.165755</td>
          <td>26.305437</td>
          <td>0.136025</td>
          <td>25.201958</td>
          <td>0.085067</td>
          <td>24.650442</td>
          <td>0.098701</td>
          <td>23.896790</td>
          <td>0.115029</td>
          <td>0.183595</td>
          <td>0.150560</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.737552</td>
          <td>1.577243</td>
          <td>27.354609</td>
          <td>0.285341</td>
          <td>26.679926</td>
          <td>0.143901</td>
          <td>26.396608</td>
          <td>0.181327</td>
          <td>25.410967</td>
          <td>0.146089</td>
          <td>25.298053</td>
          <td>0.286890</td>
          <td>0.025078</td>
          <td>0.023607</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.011604</td>
          <td>0.425840</td>
          <td>25.924783</td>
          <td>0.120372</td>
          <td>25.150334</td>
          <td>0.116084</td>
          <td>24.216386</td>
          <td>0.114625</td>
          <td>0.018229</td>
          <td>0.013905</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.858600</td>
          <td>0.867953</td>
          <td>27.208184</td>
          <td>0.229779</td>
          <td>26.131143</td>
          <td>0.147845</td>
          <td>25.314822</td>
          <td>0.137402</td>
          <td>25.613003</td>
          <td>0.376035</td>
          <td>0.048824</td>
          <td>0.046515</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.111587</td>
          <td>0.276970</td>
          <td>26.074647</td>
          <td>0.095692</td>
          <td>25.940741</td>
          <td>0.074847</td>
          <td>25.663342</td>
          <td>0.095467</td>
          <td>25.318607</td>
          <td>0.133895</td>
          <td>25.103499</td>
          <td>0.242931</td>
          <td>0.005488</td>
          <td>0.003627</td>
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
          <td>26.978098</td>
          <td>0.540190</td>
          <td>26.524393</td>
          <td>0.141475</td>
          <td>25.406912</td>
          <td>0.046623</td>
          <td>25.013818</td>
          <td>0.053762</td>
          <td>24.890081</td>
          <td>0.092154</td>
          <td>24.708495</td>
          <td>0.174526</td>
          <td>0.005325</td>
          <td>0.004949</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.538376</td>
          <td>1.473867</td>
          <td>27.330250</td>
          <td>0.297630</td>
          <td>26.047076</td>
          <td>0.089487</td>
          <td>25.240085</td>
          <td>0.071840</td>
          <td>24.722131</td>
          <td>0.086508</td>
          <td>24.296155</td>
          <td>0.133486</td>
          <td>0.083398</td>
          <td>0.081778</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.718707</td>
          <td>0.480426</td>
          <td>26.858067</td>
          <td>0.208330</td>
          <td>26.535116</td>
          <td>0.142032</td>
          <td>25.866188</td>
          <td>0.129241</td>
          <td>25.736983</td>
          <td>0.214938</td>
          <td>25.833128</td>
          <td>0.482686</td>
          <td>0.119800</td>
          <td>0.080097</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.756497</td>
          <td>0.461517</td>
          <td>26.224485</td>
          <td>0.110063</td>
          <td>26.016439</td>
          <td>0.080858</td>
          <td>25.610612</td>
          <td>0.092140</td>
          <td>26.015485</td>
          <td>0.243825</td>
          <td>25.574327</td>
          <td>0.358402</td>
          <td>0.028734</td>
          <td>0.027458</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.841616</td>
          <td>0.489007</td>
          <td>26.623393</td>
          <td>0.154131</td>
          <td>26.342181</td>
          <td>0.106630</td>
          <td>26.102533</td>
          <td>0.140049</td>
          <td>25.913071</td>
          <td>0.222020</td>
          <td>25.476208</td>
          <td>0.328794</td>
          <td>0.010735</td>
          <td>0.007655</td>
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
