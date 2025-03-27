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

    <pzflow.flow.Flow at 0x7f5d64bc9870>



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
          <td>26.648353</td>
          <td>0.157301</td>
          <td>25.970597</td>
          <td>0.076826</td>
          <td>25.366559</td>
          <td>0.073473</td>
          <td>24.741318</td>
          <td>0.080811</td>
          <td>23.927149</td>
          <td>0.088651</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.211948</td>
          <td>0.252397</td>
          <td>26.553570</td>
          <td>0.128016</td>
          <td>26.284034</td>
          <td>0.163444</td>
          <td>25.865787</td>
          <td>0.213202</td>
          <td>25.466901</td>
          <td>0.326010</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.760300</td>
          <td>1.589858</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.347535</td>
          <td>1.771477</td>
          <td>25.897120</td>
          <td>0.117076</td>
          <td>25.150303</td>
          <td>0.115672</td>
          <td>24.362130</td>
          <td>0.129616</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.533461</td>
          <td>0.291618</td>
          <td>26.029572</td>
          <td>0.131337</td>
          <td>25.576665</td>
          <td>0.167044</td>
          <td>24.717049</td>
          <td>0.175735</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.858067</td>
          <td>0.224920</td>
          <td>26.109199</td>
          <td>0.098608</td>
          <td>25.972980</td>
          <td>0.076987</td>
          <td>25.499934</td>
          <td>0.082657</td>
          <td>25.480020</td>
          <td>0.153804</td>
          <td>24.765552</td>
          <td>0.183109</td>
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
          <td>26.703313</td>
          <td>0.440616</td>
          <td>26.447123</td>
          <td>0.132315</td>
          <td>25.427251</td>
          <td>0.047456</td>
          <td>25.106493</td>
          <td>0.058350</td>
          <td>24.911546</td>
          <td>0.093874</td>
          <td>24.769496</td>
          <td>0.183721</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.564407</td>
          <td>0.808379</td>
          <td>26.940439</td>
          <td>0.201469</td>
          <td>25.789892</td>
          <td>0.065473</td>
          <td>25.189608</td>
          <td>0.062815</td>
          <td>24.706414</td>
          <td>0.078360</td>
          <td>24.128928</td>
          <td>0.105813</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.180558</td>
          <td>0.623894</td>
          <td>26.657494</td>
          <td>0.158535</td>
          <td>26.546303</td>
          <td>0.127213</td>
          <td>26.191180</td>
          <td>0.150960</td>
          <td>25.643043</td>
          <td>0.176743</td>
          <td>26.178725</td>
          <td>0.559654</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.162740</td>
          <td>0.616148</td>
          <td>26.357989</td>
          <td>0.122488</td>
          <td>26.075942</td>
          <td>0.084309</td>
          <td>25.852425</td>
          <td>0.112606</td>
          <td>26.169588</td>
          <td>0.273893</td>
          <td>26.319446</td>
          <td>0.618524</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.728673</td>
          <td>0.449130</td>
          <td>26.638199</td>
          <td>0.155941</td>
          <td>26.659802</td>
          <td>0.140324</td>
          <td>26.232759</td>
          <td>0.156436</td>
          <td>26.051830</td>
          <td>0.248746</td>
          <td>25.853064</td>
          <td>0.440013</td>
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
          <td>27.103274</td>
          <td>0.648787</td>
          <td>26.766842</td>
          <td>0.199637</td>
          <td>25.960494</td>
          <td>0.089561</td>
          <td>25.141186</td>
          <td>0.071337</td>
          <td>24.914606</td>
          <td>0.110560</td>
          <td>24.074141</td>
          <td>0.119066</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.161450</td>
          <td>0.320952</td>
          <td>27.707483</td>
          <td>0.425344</td>
          <td>26.852199</td>
          <td>0.193434</td>
          <td>26.369958</td>
          <td>0.206731</td>
          <td>25.615951</td>
          <td>0.201750</td>
          <td>25.290711</td>
          <td>0.329642</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.267894</td>
          <td>0.650344</td>
          <td>27.496654</td>
          <td>0.334700</td>
          <td>26.060181</td>
          <td>0.162622</td>
          <td>24.881858</td>
          <td>0.109852</td>
          <td>24.127536</td>
          <td>0.127567</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.124226</td>
          <td>0.607584</td>
          <td>27.448892</td>
          <td>0.335561</td>
          <td>26.330574</td>
          <td>0.213637</td>
          <td>25.337931</td>
          <td>0.170107</td>
          <td>26.044946</td>
          <td>0.615779</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.437771</td>
          <td>0.398495</td>
          <td>25.962210</td>
          <td>0.100029</td>
          <td>25.888037</td>
          <td>0.084055</td>
          <td>25.857096</td>
          <td>0.133588</td>
          <td>25.653025</td>
          <td>0.208140</td>
          <td>24.933181</td>
          <td>0.246896</td>
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
          <td>27.801348</td>
          <td>1.029425</td>
          <td>26.453997</td>
          <td>0.155967</td>
          <td>25.450049</td>
          <td>0.058263</td>
          <td>25.104927</td>
          <td>0.070616</td>
          <td>24.863412</td>
          <td>0.107958</td>
          <td>24.519543</td>
          <td>0.178297</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.820023</td>
          <td>0.209481</td>
          <td>25.986124</td>
          <td>0.091985</td>
          <td>25.145769</td>
          <td>0.071939</td>
          <td>24.923539</td>
          <td>0.111887</td>
          <td>24.236779</td>
          <td>0.137653</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.110823</td>
          <td>1.224285</td>
          <td>26.522593</td>
          <td>0.164184</td>
          <td>26.549649</td>
          <td>0.151394</td>
          <td>26.105155</td>
          <td>0.167374</td>
          <td>25.806531</td>
          <td>0.239270</td>
          <td>26.536060</td>
          <td>0.821774</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.011625</td>
          <td>0.290935</td>
          <td>26.065036</td>
          <td>0.112523</td>
          <td>25.896893</td>
          <td>0.087398</td>
          <td>26.071923</td>
          <td>0.165769</td>
          <td>25.728591</td>
          <td>0.228292</td>
          <td>25.652499</td>
          <td>0.448738</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.836620</td>
          <td>0.540357</td>
          <td>27.214993</td>
          <td>0.291260</td>
          <td>26.649068</td>
          <td>0.164389</td>
          <td>25.934458</td>
          <td>0.144218</td>
          <td>26.425648</td>
          <td>0.391807</td>
          <td>27.130304</td>
          <td>1.174619</td>
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
          <td>26.638334</td>
          <td>0.419433</td>
          <td>26.831143</td>
          <td>0.183772</td>
          <td>25.996565</td>
          <td>0.078618</td>
          <td>25.242615</td>
          <td>0.065846</td>
          <td>24.878887</td>
          <td>0.091231</td>
          <td>23.987142</td>
          <td>0.093464</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.943140</td>
          <td>0.526813</td>
          <td>27.528135</td>
          <td>0.326119</td>
          <td>26.594831</td>
          <td>0.132794</td>
          <td>26.186510</td>
          <td>0.150502</td>
          <td>25.858642</td>
          <td>0.212124</td>
          <td>26.027877</td>
          <td>0.501826</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.434925</td>
          <td>0.679359</td>
          <td>28.110770</td>
          <td>0.490510</td>
          <td>26.101859</td>
          <td>0.152003</td>
          <td>25.007316</td>
          <td>0.110753</td>
          <td>24.412539</td>
          <td>0.147122</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.128530</td>
          <td>0.685943</td>
          <td>27.453859</td>
          <td>0.367878</td>
          <td>27.642763</td>
          <td>0.389307</td>
          <td>25.876709</td>
          <td>0.144865</td>
          <td>25.454979</td>
          <td>0.187202</td>
          <td>25.601170</td>
          <td>0.444075</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.457340</td>
          <td>0.365025</td>
          <td>26.038669</td>
          <td>0.092811</td>
          <td>25.856489</td>
          <td>0.069551</td>
          <td>25.604259</td>
          <td>0.090745</td>
          <td>25.316996</td>
          <td>0.133857</td>
          <td>25.076012</td>
          <td>0.237744</td>
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
          <td>26.393663</td>
          <td>0.135215</td>
          <td>25.396551</td>
          <td>0.050015</td>
          <td>25.047342</td>
          <td>0.060179</td>
          <td>24.791940</td>
          <td>0.091431</td>
          <td>24.841409</td>
          <td>0.211021</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.991678</td>
          <td>0.550586</td>
          <td>26.893701</td>
          <td>0.196385</td>
          <td>25.926346</td>
          <td>0.075116</td>
          <td>25.099463</td>
          <td>0.059011</td>
          <td>24.916805</td>
          <td>0.095873</td>
          <td>24.227469</td>
          <td>0.117286</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.954529</td>
          <td>0.546221</td>
          <td>26.848154</td>
          <td>0.194213</td>
          <td>26.497394</td>
          <td>0.127963</td>
          <td>26.219019</td>
          <td>0.162512</td>
          <td>26.352595</td>
          <td>0.331761</td>
          <td>25.756774</td>
          <td>0.427314</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.595045</td>
          <td>0.435728</td>
          <td>26.086561</td>
          <td>0.106901</td>
          <td>26.221816</td>
          <td>0.107461</td>
          <td>26.236376</td>
          <td>0.176252</td>
          <td>25.757444</td>
          <td>0.217149</td>
          <td>25.648342</td>
          <td>0.417276</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.776960</td>
          <td>0.476617</td>
          <td>26.975675</td>
          <td>0.214307</td>
          <td>26.497983</td>
          <td>0.126771</td>
          <td>26.467905</td>
          <td>0.198651</td>
          <td>26.587738</td>
          <td>0.395250</td>
          <td>25.618482</td>
          <td>0.380753</td>
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
