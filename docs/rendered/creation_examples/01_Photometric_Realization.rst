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

    <pzflow.flow.Flow at 0x7f402aae0880>



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
    0      23.994413  0.160081  0.100545  
    1      25.391064  0.181790  0.135174  
    2      24.304707  0.064402  0.052881  
    3      25.291103  0.090203  0.058477  
    4      25.096743  0.022666  0.015715  
    ...          ...       ...       ...  
    99995  24.737946  0.138890  0.079570  
    99996  24.224169  0.062526  0.038298  
    99997  25.613836  0.128858  0.092579  
    99998  25.274899  0.044749  0.041155  
    99999  25.699642  0.082544  0.042915  
    
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
          <td>27.315522</td>
          <td>0.684907</td>
          <td>26.482092</td>
          <td>0.136370</td>
          <td>26.107668</td>
          <td>0.086698</td>
          <td>25.248646</td>
          <td>0.066190</td>
          <td>24.779848</td>
          <td>0.083604</td>
          <td>23.868743</td>
          <td>0.084207</td>
          <td>0.160081</td>
          <td>0.100545</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>32.988429</td>
          <td>5.538892</td>
          <td>27.425694</td>
          <td>0.300257</td>
          <td>26.934845</td>
          <td>0.177540</td>
          <td>26.190089</td>
          <td>0.150819</td>
          <td>25.607903</td>
          <td>0.171546</td>
          <td>25.201523</td>
          <td>0.263208</td>
          <td>0.181790</td>
          <td>0.135174</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.708650</td>
          <td>1.392561</td>
          <td>28.608055</td>
          <td>0.655299</td>
          <td>25.985111</td>
          <td>0.126376</td>
          <td>24.942737</td>
          <td>0.096479</td>
          <td>24.401179</td>
          <td>0.134068</td>
          <td>0.064402</td>
          <td>0.052881</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.093620</td>
          <td>0.985740</td>
          <td>27.721888</td>
          <td>0.338996</td>
          <td>26.202940</td>
          <td>0.152490</td>
          <td>26.077318</td>
          <td>0.254009</td>
          <td>25.493230</td>
          <td>0.332896</td>
          <td>0.090203</td>
          <td>0.058477</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.896742</td>
          <td>0.232239</td>
          <td>26.140528</td>
          <td>0.101348</td>
          <td>26.017428</td>
          <td>0.080069</td>
          <td>25.586939</td>
          <td>0.089240</td>
          <td>25.392581</td>
          <td>0.142676</td>
          <td>25.170404</td>
          <td>0.256591</td>
          <td>0.022666</td>
          <td>0.015715</td>
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
          <td>27.164974</td>
          <td>0.617116</td>
          <td>26.204690</td>
          <td>0.107192</td>
          <td>25.475728</td>
          <td>0.049543</td>
          <td>25.057966</td>
          <td>0.055890</td>
          <td>24.843187</td>
          <td>0.088399</td>
          <td>24.815429</td>
          <td>0.190988</td>
          <td>0.138890</td>
          <td>0.079570</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.578783</td>
          <td>0.815946</td>
          <td>26.388598</td>
          <td>0.125782</td>
          <td>26.096418</td>
          <td>0.085844</td>
          <td>25.226398</td>
          <td>0.064898</td>
          <td>24.926643</td>
          <td>0.095126</td>
          <td>24.296516</td>
          <td>0.122448</td>
          <td>0.062526</td>
          <td>0.038298</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.478736</td>
          <td>0.370843</td>
          <td>27.114159</td>
          <td>0.232854</td>
          <td>26.291330</td>
          <td>0.101871</td>
          <td>26.370906</td>
          <td>0.175987</td>
          <td>26.197768</td>
          <td>0.280234</td>
          <td>25.386381</td>
          <td>0.305707</td>
          <td>0.128858</td>
          <td>0.092579</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.234608</td>
          <td>0.305784</td>
          <td>26.329657</td>
          <td>0.119513</td>
          <td>26.003179</td>
          <td>0.079068</td>
          <td>25.770450</td>
          <td>0.104827</td>
          <td>25.751766</td>
          <td>0.193758</td>
          <td>25.312414</td>
          <td>0.288031</td>
          <td>0.044749</td>
          <td>0.041155</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.504345</td>
          <td>0.378302</td>
          <td>26.659736</td>
          <td>0.158839</td>
          <td>26.660815</td>
          <td>0.140447</td>
          <td>26.588162</td>
          <td>0.211335</td>
          <td>25.752191</td>
          <td>0.193827</td>
          <td>25.307124</td>
          <td>0.286802</td>
          <td>0.082544</td>
          <td>0.042915</td>
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
          <td>26.662942</td>
          <td>0.192176</td>
          <td>26.234963</td>
          <td>0.120469</td>
          <td>25.122198</td>
          <td>0.074418</td>
          <td>24.838603</td>
          <td>0.109463</td>
          <td>24.153685</td>
          <td>0.135100</td>
          <td>0.160081</td>
          <td>0.100545</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.617194</td>
          <td>0.481290</td>
          <td>28.027208</td>
          <td>0.572481</td>
          <td>26.528747</td>
          <td>0.158750</td>
          <td>26.010727</td>
          <td>0.165152</td>
          <td>25.897104</td>
          <td>0.274309</td>
          <td>25.026927</td>
          <td>0.287464</td>
          <td>0.181790</td>
          <td>0.135174</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>37.812842</td>
          <td>9.318595</td>
          <td>27.581422</td>
          <td>0.354438</td>
          <td>26.128943</td>
          <td>0.170604</td>
          <td>25.230750</td>
          <td>0.147057</td>
          <td>24.461674</td>
          <td>0.168176</td>
          <td>0.064402</td>
          <td>0.052881</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.733410</td>
          <td>0.401567</td>
          <td>26.061330</td>
          <td>0.162278</td>
          <td>25.618969</td>
          <td>0.205959</td>
          <td>25.707722</td>
          <td>0.462774</td>
          <td>0.090203</td>
          <td>0.058477</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.336158</td>
          <td>0.368565</td>
          <td>26.185400</td>
          <td>0.121600</td>
          <td>25.894662</td>
          <td>0.084627</td>
          <td>25.865008</td>
          <td>0.134632</td>
          <td>25.477223</td>
          <td>0.179659</td>
          <td>24.885801</td>
          <td>0.237652</td>
          <td>0.022666</td>
          <td>0.015715</td>
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
          <td>27.469590</td>
          <td>0.848240</td>
          <td>26.348278</td>
          <td>0.145050</td>
          <td>25.459532</td>
          <td>0.059977</td>
          <td>24.957226</td>
          <td>0.063288</td>
          <td>24.802021</td>
          <td>0.104411</td>
          <td>24.653151</td>
          <td>0.203590</td>
          <td>0.138890</td>
          <td>0.079570</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.785809</td>
          <td>1.013613</td>
          <td>26.541394</td>
          <td>0.166286</td>
          <td>26.099206</td>
          <td>0.102067</td>
          <td>25.206468</td>
          <td>0.076288</td>
          <td>24.852266</td>
          <td>0.105647</td>
          <td>24.233380</td>
          <td>0.137925</td>
          <td>0.062526</td>
          <td>0.038298</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.950593</td>
          <td>1.134495</td>
          <td>26.653686</td>
          <td>0.188060</td>
          <td>26.468636</td>
          <td>0.145155</td>
          <td>26.231342</td>
          <td>0.191556</td>
          <td>25.829287</td>
          <td>0.250299</td>
          <td>26.248796</td>
          <td>0.694164</td>
          <td>0.128858</td>
          <td>0.092579</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.957276</td>
          <td>0.587833</td>
          <td>26.184859</td>
          <td>0.122082</td>
          <td>26.035779</td>
          <td>0.096280</td>
          <td>26.013157</td>
          <td>0.153706</td>
          <td>25.559236</td>
          <td>0.193467</td>
          <td>25.005528</td>
          <td>0.263470</td>
          <td>0.044749</td>
          <td>0.041155</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.123380</td>
          <td>0.663569</td>
          <td>27.165583</td>
          <td>0.280878</td>
          <td>26.662357</td>
          <td>0.166971</td>
          <td>26.256035</td>
          <td>0.190519</td>
          <td>26.116646</td>
          <td>0.308414</td>
          <td>25.420213</td>
          <td>0.369812</td>
          <td>0.082544</td>
          <td>0.042915</td>
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
          <td>27.581036</td>
          <td>0.904779</td>
          <td>26.671247</td>
          <td>0.189083</td>
          <td>26.049916</td>
          <td>0.099789</td>
          <td>25.297559</td>
          <td>0.084448</td>
          <td>24.581892</td>
          <td>0.085074</td>
          <td>24.205946</td>
          <td>0.137513</td>
          <td>0.160081</td>
          <td>0.100545</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.346585</td>
          <td>2.255341</td>
          <td>27.364744</td>
          <td>0.352589</td>
          <td>26.529584</td>
          <td>0.161508</td>
          <td>26.164623</td>
          <td>0.191351</td>
          <td>26.122659</td>
          <td>0.333895</td>
          <td>25.368837</td>
          <td>0.382850</td>
          <td>0.181790</td>
          <td>0.135174</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.446848</td>
          <td>1.240063</td>
          <td>28.389832</td>
          <td>0.582758</td>
          <td>25.823046</td>
          <td>0.115011</td>
          <td>25.015824</td>
          <td>0.107562</td>
          <td>24.236936</td>
          <td>0.121761</td>
          <td>0.064402</td>
          <td>0.052881</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.511106</td>
          <td>0.340303</td>
          <td>26.879366</td>
          <td>0.181554</td>
          <td>26.276918</td>
          <td>0.174744</td>
          <td>25.567094</td>
          <td>0.177619</td>
          <td>25.123880</td>
          <td>0.264751</td>
          <td>0.090203</td>
          <td>0.058477</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.841563</td>
          <td>0.490121</td>
          <td>26.057347</td>
          <td>0.094649</td>
          <td>25.888408</td>
          <td>0.071812</td>
          <td>25.563008</td>
          <td>0.087854</td>
          <td>25.451502</td>
          <td>0.150843</td>
          <td>25.085410</td>
          <td>0.240468</td>
          <td>0.022666</td>
          <td>0.015715</td>
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
          <td>26.509213</td>
          <td>0.416321</td>
          <td>26.208693</td>
          <td>0.122081</td>
          <td>25.368469</td>
          <td>0.052125</td>
          <td>25.263898</td>
          <td>0.078057</td>
          <td>24.912310</td>
          <td>0.108442</td>
          <td>24.784368</td>
          <td>0.214490</td>
          <td>0.138890</td>
          <td>0.079570</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.174428</td>
          <td>0.633447</td>
          <td>26.560026</td>
          <td>0.150243</td>
          <td>26.008048</td>
          <td>0.082250</td>
          <td>25.197259</td>
          <td>0.065629</td>
          <td>24.922085</td>
          <td>0.098112</td>
          <td>24.313397</td>
          <td>0.128786</td>
          <td>0.062526</td>
          <td>0.038298</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.353656</td>
          <td>0.761305</td>
          <td>26.852809</td>
          <td>0.211466</td>
          <td>26.334740</td>
          <td>0.122136</td>
          <td>26.273764</td>
          <td>0.187432</td>
          <td>26.593823</td>
          <td>0.435290</td>
          <td>26.601821</td>
          <td>0.838287</td>
          <td>0.128858</td>
          <td>0.092579</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.198333</td>
          <td>0.301685</td>
          <td>26.189400</td>
          <td>0.108035</td>
          <td>26.123084</td>
          <td>0.090066</td>
          <td>25.738117</td>
          <td>0.104545</td>
          <td>25.754175</td>
          <td>0.198750</td>
          <td>25.214995</td>
          <td>0.272467</td>
          <td>0.044749</td>
          <td>0.041155</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.820287</td>
          <td>0.975905</td>
          <td>26.646565</td>
          <td>0.164519</td>
          <td>26.511186</td>
          <td>0.130219</td>
          <td>26.321593</td>
          <td>0.178338</td>
          <td>26.093892</td>
          <td>0.270851</td>
          <td>26.286505</td>
          <td>0.632243</td>
          <td>0.082544</td>
          <td>0.042915</td>
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
