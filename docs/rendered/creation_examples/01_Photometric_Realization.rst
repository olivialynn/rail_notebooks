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

    <pzflow.flow.Flow at 0x7f0edc3a5c30>



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
    0      23.994413  0.022082  0.019409  
    1      25.391064  0.012415  0.012328  
    2      24.304707  0.143810  0.111213  
    3      25.291103  0.083403  0.045779  
    4      25.096743  0.127494  0.072922  
    ...          ...       ...       ...  
    99995  24.737946  0.078304  0.048682  
    99996  24.224169  0.110401  0.058552  
    99997  25.613836  0.005362  0.004959  
    99998  25.274899  0.243118  0.200642  
    99999  25.699642  0.050858  0.032257  
    
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
          <td>27.053824</td>
          <td>0.570358</td>
          <td>26.937923</td>
          <td>0.201045</td>
          <td>26.011118</td>
          <td>0.079624</td>
          <td>25.250926</td>
          <td>0.066324</td>
          <td>24.698563</td>
          <td>0.077818</td>
          <td>24.018610</td>
          <td>0.096069</td>
          <td>0.022082</td>
          <td>0.019409</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.150582</td>
          <td>1.154619</td>
          <td>27.293728</td>
          <td>0.269849</td>
          <td>26.769547</td>
          <td>0.154203</td>
          <td>26.533206</td>
          <td>0.201828</td>
          <td>25.925327</td>
          <td>0.224045</td>
          <td>25.466905</td>
          <td>0.326011</td>
          <td>0.012415</td>
          <td>0.012328</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.823768</td>
          <td>0.952152</td>
          <td>29.029751</td>
          <td>0.948096</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.875455</td>
          <td>0.114889</td>
          <td>25.126266</td>
          <td>0.113276</td>
          <td>24.403644</td>
          <td>0.134354</td>
          <td>0.143810</td>
          <td>0.111213</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.151682</td>
          <td>1.020747</td>
          <td>27.126334</td>
          <td>0.208628</td>
          <td>26.172334</td>
          <td>0.148537</td>
          <td>25.353198</td>
          <td>0.137914</td>
          <td>25.185572</td>
          <td>0.259798</td>
          <td>0.083403</td>
          <td>0.045779</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.003016</td>
          <td>0.253465</td>
          <td>25.989075</td>
          <td>0.088747</td>
          <td>25.976384</td>
          <td>0.077219</td>
          <td>25.631551</td>
          <td>0.092810</td>
          <td>25.295586</td>
          <td>0.131218</td>
          <td>25.008221</td>
          <td>0.224444</td>
          <td>0.127494</td>
          <td>0.072922</td>
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
          <td>26.433857</td>
          <td>0.358069</td>
          <td>26.398912</td>
          <td>0.126911</td>
          <td>25.424256</td>
          <td>0.047330</td>
          <td>24.984974</td>
          <td>0.052383</td>
          <td>24.707901</td>
          <td>0.078462</td>
          <td>24.637843</td>
          <td>0.164281</td>
          <td>0.078304</td>
          <td>0.048682</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.475037</td>
          <td>0.369776</td>
          <td>26.817290</td>
          <td>0.181612</td>
          <td>26.115119</td>
          <td>0.087269</td>
          <td>25.344036</td>
          <td>0.072023</td>
          <td>24.915668</td>
          <td>0.094214</td>
          <td>24.175551</td>
          <td>0.110211</td>
          <td>0.110401</td>
          <td>0.058552</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.106989</td>
          <td>0.592377</td>
          <td>26.883817</td>
          <td>0.192104</td>
          <td>26.482986</td>
          <td>0.120411</td>
          <td>26.328422</td>
          <td>0.169746</td>
          <td>25.923178</td>
          <td>0.223645</td>
          <td>25.017773</td>
          <td>0.226232</td>
          <td>0.005362</td>
          <td>0.004959</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.483545</td>
          <td>0.372234</td>
          <td>26.146654</td>
          <td>0.101893</td>
          <td>26.090879</td>
          <td>0.085426</td>
          <td>26.015727</td>
          <td>0.129772</td>
          <td>25.510538</td>
          <td>0.157875</td>
          <td>26.042996</td>
          <td>0.507027</td>
          <td>0.243118</td>
          <td>0.200642</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.466949</td>
          <td>0.367451</td>
          <td>26.639663</td>
          <td>0.156137</td>
          <td>26.415672</td>
          <td>0.113560</td>
          <td>26.361342</td>
          <td>0.174564</td>
          <td>25.836901</td>
          <td>0.208115</td>
          <td>25.967045</td>
          <td>0.479317</td>
          <td>0.050858</td>
          <td>0.032257</td>
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
          <td>31.388449</td>
          <td>4.092236</td>
          <td>27.063448</td>
          <td>0.255651</td>
          <td>25.863617</td>
          <td>0.082358</td>
          <td>25.197052</td>
          <td>0.075060</td>
          <td>24.591053</td>
          <td>0.083368</td>
          <td>23.793440</td>
          <td>0.093305</td>
          <td>0.022082</td>
          <td>0.019409</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.416610</td>
          <td>1.432273</td>
          <td>27.213147</td>
          <td>0.288534</td>
          <td>26.749530</td>
          <td>0.177405</td>
          <td>26.317770</td>
          <td>0.197930</td>
          <td>26.142390</td>
          <td>0.310869</td>
          <td>25.163527</td>
          <td>0.297858</td>
          <td>0.012415</td>
          <td>0.012328</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.676082</td>
          <td>0.432873</td>
          <td>27.653448</td>
          <td>0.388923</td>
          <td>25.952826</td>
          <td>0.153015</td>
          <td>25.034781</td>
          <td>0.129341</td>
          <td>24.182302</td>
          <td>0.137968</td>
          <td>0.143810</td>
          <td>0.111213</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.045970</td>
          <td>1.936191</td>
          <td>28.289893</td>
          <td>0.656929</td>
          <td>27.667351</td>
          <td>0.380195</td>
          <td>26.672852</td>
          <td>0.269409</td>
          <td>25.472621</td>
          <td>0.181359</td>
          <td>25.305111</td>
          <td>0.338050</td>
          <td>0.083403</td>
          <td>0.045779</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.344792</td>
          <td>0.779578</td>
          <td>25.895327</td>
          <td>0.097353</td>
          <td>25.887524</td>
          <td>0.087000</td>
          <td>25.728926</td>
          <td>0.123868</td>
          <td>25.264798</td>
          <td>0.154915</td>
          <td>25.500539</td>
          <td>0.400981</td>
          <td>0.127494</td>
          <td>0.072922</td>
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
          <td>28.048582</td>
          <td>1.183425</td>
          <td>26.515529</td>
          <td>0.163402</td>
          <td>25.539299</td>
          <td>0.062628</td>
          <td>25.123581</td>
          <td>0.071283</td>
          <td>24.940872</td>
          <td>0.114725</td>
          <td>24.746290</td>
          <td>0.214332</td>
          <td>0.078304</td>
          <td>0.048682</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.788039</td>
          <td>0.526814</td>
          <td>26.780674</td>
          <td>0.206488</td>
          <td>25.932063</td>
          <td>0.089625</td>
          <td>25.097933</td>
          <td>0.070522</td>
          <td>24.843095</td>
          <td>0.106553</td>
          <td>24.347513</td>
          <td>0.154703</td>
          <td>0.110401</td>
          <td>0.058552</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.269914</td>
          <td>2.114374</td>
          <td>26.637989</td>
          <td>0.179093</td>
          <td>26.556122</td>
          <td>0.150356</td>
          <td>26.198986</td>
          <td>0.178970</td>
          <td>25.797534</td>
          <td>0.234675</td>
          <td>26.563012</td>
          <td>0.828204</td>
          <td>0.005362</td>
          <td>0.004959</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.137177</td>
          <td>0.348670</td>
          <td>26.515057</td>
          <td>0.183272</td>
          <td>26.097816</td>
          <td>0.116623</td>
          <td>25.803737</td>
          <td>0.147579</td>
          <td>25.625693</td>
          <td>0.233172</td>
          <td>25.638350</td>
          <td>0.489842</td>
          <td>0.243118</td>
          <td>0.200642</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.181602</td>
          <td>0.327457</td>
          <td>26.739844</td>
          <td>0.196191</td>
          <td>26.395968</td>
          <td>0.131759</td>
          <td>26.013743</td>
          <td>0.153764</td>
          <td>25.678321</td>
          <td>0.213758</td>
          <td>25.855992</td>
          <td>0.510759</td>
          <td>0.050858</td>
          <td>0.032257</td>
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
          <td>26.803271</td>
          <td>0.476578</td>
          <td>26.660978</td>
          <td>0.159793</td>
          <td>26.126231</td>
          <td>0.088644</td>
          <td>25.208700</td>
          <td>0.064285</td>
          <td>24.617289</td>
          <td>0.072852</td>
          <td>24.082804</td>
          <td>0.102246</td>
          <td>0.022082</td>
          <td>0.019409</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.588656</td>
          <td>2.276536</td>
          <td>27.734744</td>
          <td>0.383888</td>
          <td>26.677119</td>
          <td>0.142724</td>
          <td>26.260781</td>
          <td>0.160571</td>
          <td>26.302254</td>
          <td>0.305458</td>
          <td>25.231980</td>
          <td>0.270371</td>
          <td>0.012415</td>
          <td>0.012328</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>31.783762</td>
          <td>4.496903</td>
          <td>29.361837</td>
          <td>1.277558</td>
          <td>32.753255</td>
          <td>4.171891</td>
          <td>26.150231</td>
          <td>0.175345</td>
          <td>25.181204</td>
          <td>0.142221</td>
          <td>24.365248</td>
          <td>0.156337</td>
          <td>0.143810</td>
          <td>0.111213</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.764058</td>
          <td>1.474695</td>
          <td>27.676221</td>
          <td>0.344158</td>
          <td>26.539367</td>
          <td>0.214726</td>
          <td>25.549545</td>
          <td>0.172485</td>
          <td>25.110742</td>
          <td>0.258180</td>
          <td>0.083403</td>
          <td>0.045779</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.198962</td>
          <td>0.322571</td>
          <td>26.276898</td>
          <td>0.127273</td>
          <td>25.968143</td>
          <td>0.086875</td>
          <td>25.586189</td>
          <td>0.101543</td>
          <td>25.364428</td>
          <td>0.157260</td>
          <td>24.698677</td>
          <td>0.195825</td>
          <td>0.127494</td>
          <td>0.072922</td>
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
          <td>26.886882</td>
          <td>0.521598</td>
          <td>26.420157</td>
          <td>0.135414</td>
          <td>25.361309</td>
          <td>0.047276</td>
          <td>25.083191</td>
          <td>0.060523</td>
          <td>24.793419</td>
          <td>0.089314</td>
          <td>24.694852</td>
          <td>0.182033</td>
          <td>0.078304</td>
          <td>0.048682</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.612562</td>
          <td>0.875386</td>
          <td>26.931343</td>
          <td>0.216210</td>
          <td>26.039543</td>
          <td>0.089666</td>
          <td>24.986822</td>
          <td>0.057915</td>
          <td>24.762661</td>
          <td>0.090424</td>
          <td>24.006022</td>
          <td>0.104632</td>
          <td>0.110401</td>
          <td>0.058552</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.317959</td>
          <td>0.686180</td>
          <td>27.034215</td>
          <td>0.217965</td>
          <td>26.508248</td>
          <td>0.123127</td>
          <td>26.246762</td>
          <td>0.158380</td>
          <td>25.957555</td>
          <td>0.230198</td>
          <td>26.237029</td>
          <td>0.583685</td>
          <td>0.005362</td>
          <td>0.004959</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.259601</td>
          <td>0.416165</td>
          <td>26.355821</td>
          <td>0.177174</td>
          <td>26.176841</td>
          <td>0.139480</td>
          <td>25.893190</td>
          <td>0.178050</td>
          <td>25.252571</td>
          <td>0.189908</td>
          <td>25.227301</td>
          <td>0.395971</td>
          <td>0.243118</td>
          <td>0.200642</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.962847</td>
          <td>0.249111</td>
          <td>26.668217</td>
          <td>0.163265</td>
          <td>26.543862</td>
          <td>0.129976</td>
          <td>26.197872</td>
          <td>0.155593</td>
          <td>26.140682</td>
          <td>0.273513</td>
          <td>25.094752</td>
          <td>0.246800</td>
          <td>0.050858</td>
          <td>0.032257</td>
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
