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

    <pzflow.flow.Flow at 0x7f87e5867550>



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
    0      23.994413  0.080006  0.078452  
    1      25.391064  0.102903  0.061458  
    2      24.304707  0.002805  0.002373  
    3      25.291103  0.088467  0.065611  
    4      25.096743  0.037557  0.034872  
    ...          ...       ...       ...  
    99995  24.737946  0.180940  0.140723  
    99996  24.224169  0.022911  0.022091  
    99997  25.613836  0.176783  0.156653  
    99998  25.274899  0.175460  0.101629  
    99999  25.699642  0.000900  0.000537  
    
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
          <td>26.637733</td>
          <td>0.419208</td>
          <td>26.594174</td>
          <td>0.150170</td>
          <td>26.012024</td>
          <td>0.079688</td>
          <td>25.306766</td>
          <td>0.069686</td>
          <td>24.769523</td>
          <td>0.082846</td>
          <td>24.021472</td>
          <td>0.096310</td>
          <td>0.080006</td>
          <td>0.078452</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.858639</td>
          <td>1.666223</td>
          <td>27.307719</td>
          <td>0.272939</td>
          <td>26.861372</td>
          <td>0.166789</td>
          <td>26.438268</td>
          <td>0.186319</td>
          <td>25.909996</td>
          <td>0.221206</td>
          <td>25.393227</td>
          <td>0.307390</td>
          <td>0.102903</td>
          <td>0.061458</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.296545</td>
          <td>0.525118</td>
          <td>26.179497</td>
          <td>0.149454</td>
          <td>25.056482</td>
          <td>0.106583</td>
          <td>24.460131</td>
          <td>0.141064</td>
          <td>0.002805</td>
          <td>0.002373</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.192569</td>
          <td>1.045842</td>
          <td>27.154388</td>
          <td>0.213578</td>
          <td>26.493667</td>
          <td>0.195230</td>
          <td>25.729121</td>
          <td>0.190095</td>
          <td>25.905876</td>
          <td>0.457888</td>
          <td>0.088467</td>
          <td>0.065611</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.841927</td>
          <td>0.221928</td>
          <td>26.036137</td>
          <td>0.092491</td>
          <td>26.003263</td>
          <td>0.079074</td>
          <td>25.705482</td>
          <td>0.099031</td>
          <td>25.274520</td>
          <td>0.128847</td>
          <td>24.837871</td>
          <td>0.194634</td>
          <td>0.037557</td>
          <td>0.034872</td>
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
          <td>28.537418</td>
          <td>1.422766</td>
          <td>26.185313</td>
          <td>0.105394</td>
          <td>25.419738</td>
          <td>0.047140</td>
          <td>25.048171</td>
          <td>0.055406</td>
          <td>24.757358</td>
          <td>0.081962</td>
          <td>24.815979</td>
          <td>0.191077</td>
          <td>0.180940</td>
          <td>0.140723</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.119305</td>
          <td>1.134250</td>
          <td>26.945535</td>
          <td>0.202333</td>
          <td>26.124391</td>
          <td>0.087984</td>
          <td>25.226880</td>
          <td>0.064925</td>
          <td>24.807501</td>
          <td>0.085665</td>
          <td>24.233064</td>
          <td>0.115876</td>
          <td>0.022911</td>
          <td>0.022091</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.381460</td>
          <td>0.343629</td>
          <td>27.054689</td>
          <td>0.221644</td>
          <td>26.601750</td>
          <td>0.133467</td>
          <td>26.527434</td>
          <td>0.200852</td>
          <td>25.925329</td>
          <td>0.224045</td>
          <td>26.219004</td>
          <td>0.576052</td>
          <td>0.176783</td>
          <td>0.156653</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.283631</td>
          <td>0.317997</td>
          <td>26.307281</td>
          <td>0.117212</td>
          <td>26.075424</td>
          <td>0.084271</td>
          <td>25.655031</td>
          <td>0.094743</td>
          <td>25.579580</td>
          <td>0.167460</td>
          <td>25.774210</td>
          <td>0.414382</td>
          <td>0.175460</td>
          <td>0.101629</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.218382</td>
          <td>0.640574</td>
          <td>26.683901</td>
          <td>0.162150</td>
          <td>26.396118</td>
          <td>0.111640</td>
          <td>26.620426</td>
          <td>0.217104</td>
          <td>25.915474</td>
          <td>0.222217</td>
          <td>25.467399</td>
          <td>0.326139</td>
          <td>0.000900</td>
          <td>0.000537</td>
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
          <td>26.276433</td>
          <td>0.356482</td>
          <td>26.986849</td>
          <td>0.244041</td>
          <td>25.978360</td>
          <td>0.092903</td>
          <td>25.267460</td>
          <td>0.081508</td>
          <td>24.779453</td>
          <td>0.100308</td>
          <td>24.022504</td>
          <td>0.116285</td>
          <td>0.080006</td>
          <td>0.078452</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.406358</td>
          <td>0.343146</td>
          <td>26.567952</td>
          <td>0.155436</td>
          <td>26.444099</td>
          <td>0.225054</td>
          <td>25.518869</td>
          <td>0.190158</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.102903</td>
          <td>0.061458</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.443562</td>
          <td>0.721810</td>
          <td>28.516217</td>
          <td>0.697868</td>
          <td>25.802093</td>
          <td>0.127336</td>
          <td>24.967250</td>
          <td>0.115748</td>
          <td>24.349638</td>
          <td>0.151049</td>
          <td>0.002805</td>
          <td>0.002373</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.910829</td>
          <td>3.644069</td>
          <td>28.190790</td>
          <td>0.615339</td>
          <td>27.379029</td>
          <td>0.304151</td>
          <td>26.374367</td>
          <td>0.211616</td>
          <td>25.840757</td>
          <td>0.247824</td>
          <td>24.815628</td>
          <td>0.228383</td>
          <td>0.088467</td>
          <td>0.065611</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.928139</td>
          <td>0.266771</td>
          <td>25.887019</td>
          <td>0.094002</td>
          <td>25.983555</td>
          <td>0.091800</td>
          <td>25.705862</td>
          <td>0.117663</td>
          <td>25.333942</td>
          <td>0.159524</td>
          <td>24.894062</td>
          <td>0.240010</td>
          <td>0.037557</td>
          <td>0.034872</td>
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
          <td>27.099517</td>
          <td>0.679996</td>
          <td>26.345923</td>
          <td>0.150062</td>
          <td>25.454493</td>
          <td>0.062198</td>
          <td>25.114777</td>
          <td>0.075874</td>
          <td>24.824070</td>
          <td>0.110798</td>
          <td>24.713922</td>
          <td>0.222772</td>
          <td>0.180940</td>
          <td>0.140723</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.145911</td>
          <td>0.668827</td>
          <td>26.649979</td>
          <td>0.181177</td>
          <td>26.004881</td>
          <td>0.093281</td>
          <td>25.201212</td>
          <td>0.075357</td>
          <td>24.736118</td>
          <td>0.094734</td>
          <td>24.300083</td>
          <td>0.145003</td>
          <td>0.022911</td>
          <td>0.022091</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.922574</td>
          <td>0.281705</td>
          <td>26.671025</td>
          <td>0.198605</td>
          <td>26.460848</td>
          <td>0.150777</td>
          <td>26.376856</td>
          <td>0.226285</td>
          <td>25.744986</td>
          <td>0.243729</td>
          <td>27.619359</td>
          <td>1.589149</td>
          <td>0.176783</td>
          <td>0.156653</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.826280</td>
          <td>0.555257</td>
          <td>26.173290</td>
          <td>0.127342</td>
          <td>26.248675</td>
          <td>0.122840</td>
          <td>25.884185</td>
          <td>0.145916</td>
          <td>25.817449</td>
          <td>0.253375</td>
          <td>24.800237</td>
          <td>0.235330</td>
          <td>0.175460</td>
          <td>0.101629</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.988893</td>
          <td>0.598854</td>
          <td>26.653636</td>
          <td>0.181467</td>
          <td>26.778998</td>
          <td>0.181799</td>
          <td>26.335410</td>
          <td>0.200781</td>
          <td>25.921504</td>
          <td>0.259852</td>
          <td>25.265401</td>
          <td>0.323006</td>
          <td>0.000900</td>
          <td>0.000537</td>
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
          <td>27.073430</td>
          <td>0.604985</td>
          <td>26.846371</td>
          <td>0.198819</td>
          <td>26.158827</td>
          <td>0.098100</td>
          <td>25.181285</td>
          <td>0.067751</td>
          <td>24.819110</td>
          <td>0.093622</td>
          <td>23.907772</td>
          <td>0.094553</td>
          <td>0.080006</td>
          <td>0.078452</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.960907</td>
          <td>1.078372</td>
          <td>27.404626</td>
          <td>0.316604</td>
          <td>26.393382</td>
          <td>0.121441</td>
          <td>26.134596</td>
          <td>0.157199</td>
          <td>25.852831</td>
          <td>0.229072</td>
          <td>25.523691</td>
          <td>0.369683</td>
          <td>0.102903</td>
          <td>0.061458</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.235375</td>
          <td>0.648206</td>
          <td>28.089905</td>
          <td>0.501499</td>
          <td>27.601668</td>
          <td>0.308086</td>
          <td>26.025231</td>
          <td>0.130857</td>
          <td>25.108239</td>
          <td>0.111519</td>
          <td>24.232525</td>
          <td>0.115833</td>
          <td>0.002805</td>
          <td>0.002373</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.953641</td>
          <td>0.193994</td>
          <td>26.027775</td>
          <td>0.141737</td>
          <td>25.751459</td>
          <td>0.208207</td>
          <td>25.698344</td>
          <td>0.418609</td>
          <td>0.088467</td>
          <td>0.065611</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.165187</td>
          <td>0.292444</td>
          <td>26.011717</td>
          <td>0.091920</td>
          <td>26.013504</td>
          <td>0.081211</td>
          <td>25.836917</td>
          <td>0.113141</td>
          <td>25.415619</td>
          <td>0.148046</td>
          <td>25.308651</td>
          <td>0.292001</td>
          <td>0.037557</td>
          <td>0.034872</td>
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
          <td>26.979473</td>
          <td>0.634244</td>
          <td>26.470253</td>
          <td>0.170012</td>
          <td>25.357983</td>
          <td>0.058312</td>
          <td>25.151921</td>
          <td>0.080102</td>
          <td>24.956987</td>
          <td>0.126950</td>
          <td>24.511754</td>
          <td>0.191924</td>
          <td>0.180940</td>
          <td>0.140723</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.777834</td>
          <td>0.467896</td>
          <td>26.656056</td>
          <td>0.159257</td>
          <td>26.009135</td>
          <td>0.080033</td>
          <td>25.110987</td>
          <td>0.059010</td>
          <td>24.755566</td>
          <td>0.082396</td>
          <td>24.305840</td>
          <td>0.124313</td>
          <td>0.022911</td>
          <td>0.022091</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.805838</td>
          <td>0.566066</td>
          <td>26.690144</td>
          <td>0.207258</td>
          <td>26.504155</td>
          <td>0.161163</td>
          <td>26.106015</td>
          <td>0.185790</td>
          <td>25.531762</td>
          <td>0.210124</td>
          <td>27.732615</td>
          <td>1.705030</td>
          <td>0.176783</td>
          <td>0.156653</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.352429</td>
          <td>0.386383</td>
          <td>26.189100</td>
          <td>0.127544</td>
          <td>26.144096</td>
          <td>0.110599</td>
          <td>25.720548</td>
          <td>0.124860</td>
          <td>26.054969</td>
          <td>0.303285</td>
          <td>25.227619</td>
          <td>0.328373</td>
          <td>0.175460</td>
          <td>0.101629</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.451599</td>
          <td>0.750651</td>
          <td>26.558443</td>
          <td>0.145635</td>
          <td>26.327159</td>
          <td>0.105116</td>
          <td>26.316788</td>
          <td>0.168074</td>
          <td>26.073107</td>
          <td>0.253134</td>
          <td>26.309394</td>
          <td>0.614174</td>
          <td>0.000900</td>
          <td>0.000537</td>
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
