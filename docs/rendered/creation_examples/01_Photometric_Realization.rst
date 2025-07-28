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

    <pzflow.flow.Flow at 0x7fcb1a881ba0>



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
    0      23.994413  0.184535  0.133793  
    1      25.391064  0.040904  0.039143  
    2      24.304707  0.015764  0.010589  
    3      25.291103  0.136684  0.089738  
    4      25.096743  0.113258  0.098119  
    ...          ...       ...       ...  
    99995  24.737946  0.186611  0.152871  
    99996  24.224169  0.199165  0.120374  
    99997  25.613836  0.083094  0.055113  
    99998  25.274899  0.071036  0.064720  
    99999  25.699642  0.075928  0.051158  
    
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
          <td>29.668459</td>
          <td>2.345393</td>
          <td>26.833198</td>
          <td>0.184072</td>
          <td>26.064675</td>
          <td>0.083476</td>
          <td>25.267236</td>
          <td>0.067289</td>
          <td>24.641566</td>
          <td>0.073996</td>
          <td>23.921477</td>
          <td>0.088209</td>
          <td>0.184535</td>
          <td>0.133793</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.274460</td>
          <td>0.265643</td>
          <td>26.477396</td>
          <td>0.119828</td>
          <td>25.995937</td>
          <td>0.127567</td>
          <td>25.891072</td>
          <td>0.217747</td>
          <td>25.424145</td>
          <td>0.315089</td>
          <td>0.040904</td>
          <td>0.039143</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.925800</td>
          <td>0.519904</td>
          <td>29.517427</td>
          <td>1.257846</td>
          <td>27.803243</td>
          <td>0.361404</td>
          <td>25.902825</td>
          <td>0.117659</td>
          <td>25.084629</td>
          <td>0.109236</td>
          <td>24.237706</td>
          <td>0.116345</td>
          <td>0.015764</td>
          <td>0.010589</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.006716</td>
          <td>0.471450</td>
          <td>27.561698</td>
          <td>0.298329</td>
          <td>26.163590</td>
          <td>0.147426</td>
          <td>25.613549</td>
          <td>0.172371</td>
          <td>25.533726</td>
          <td>0.343729</td>
          <td>0.136684</td>
          <td>0.089738</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.326639</td>
          <td>0.329056</td>
          <td>26.187164</td>
          <td>0.105564</td>
          <td>25.914461</td>
          <td>0.073107</td>
          <td>25.953791</td>
          <td>0.122988</td>
          <td>25.744865</td>
          <td>0.192635</td>
          <td>25.156918</td>
          <td>0.253769</td>
          <td>0.113258</td>
          <td>0.098119</td>
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
          <td>26.850298</td>
          <td>0.491822</td>
          <td>26.405947</td>
          <td>0.127686</td>
          <td>25.488683</td>
          <td>0.050116</td>
          <td>25.018365</td>
          <td>0.053959</td>
          <td>24.804545</td>
          <td>0.085443</td>
          <td>24.729440</td>
          <td>0.177592</td>
          <td>0.186611</td>
          <td>0.152871</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.672535</td>
          <td>0.866458</td>
          <td>26.694454</td>
          <td>0.163617</td>
          <td>26.099486</td>
          <td>0.086076</td>
          <td>25.196367</td>
          <td>0.063193</td>
          <td>24.845149</td>
          <td>0.088552</td>
          <td>24.285468</td>
          <td>0.121279</td>
          <td>0.199165</td>
          <td>0.120374</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.221836</td>
          <td>1.960715</td>
          <td>27.040267</td>
          <td>0.219001</td>
          <td>26.340676</td>
          <td>0.106365</td>
          <td>26.247509</td>
          <td>0.158423</td>
          <td>25.898057</td>
          <td>0.219018</td>
          <td>25.719140</td>
          <td>0.397220</td>
          <td>0.083094</td>
          <td>0.055113</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.340984</td>
          <td>0.696884</td>
          <td>26.169776</td>
          <td>0.103973</td>
          <td>26.093386</td>
          <td>0.085615</td>
          <td>25.945118</td>
          <td>0.122065</td>
          <td>25.891698</td>
          <td>0.217861</td>
          <td>25.328656</td>
          <td>0.291834</td>
          <td>0.071036</td>
          <td>0.064720</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.875596</td>
          <td>0.501094</td>
          <td>27.173800</td>
          <td>0.244605</td>
          <td>26.558680</td>
          <td>0.128584</td>
          <td>26.355596</td>
          <td>0.173714</td>
          <td>26.016827</td>
          <td>0.241679</td>
          <td>26.030740</td>
          <td>0.502472</td>
          <td>0.075928</td>
          <td>0.051158</td>
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
          <td>26.856121</td>
          <td>0.573184</td>
          <td>26.775314</td>
          <td>0.215595</td>
          <td>25.905701</td>
          <td>0.092531</td>
          <td>25.403960</td>
          <td>0.097755</td>
          <td>24.706959</td>
          <td>0.099917</td>
          <td>24.152851</td>
          <td>0.138292</td>
          <td>0.184535</td>
          <td>0.133793</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.562336</td>
          <td>0.382004</td>
          <td>26.728988</td>
          <td>0.175168</td>
          <td>26.266427</td>
          <td>0.190482</td>
          <td>25.609073</td>
          <td>0.201586</td>
          <td>26.165159</td>
          <td>0.636895</td>
          <td>0.040904</td>
          <td>0.039143</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.212578</td>
          <td>0.699515</td>
          <td>28.859033</td>
          <td>0.943655</td>
          <td>27.933035</td>
          <td>0.459896</td>
          <td>25.817965</td>
          <td>0.129176</td>
          <td>25.033133</td>
          <td>0.122641</td>
          <td>24.201127</td>
          <td>0.132998</td>
          <td>0.015764</td>
          <td>0.010589</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.273575</td>
          <td>0.662707</td>
          <td>27.358267</td>
          <td>0.305481</td>
          <td>26.591083</td>
          <td>0.258902</td>
          <td>25.405000</td>
          <td>0.175964</td>
          <td>25.494096</td>
          <td>0.401920</td>
          <td>0.136684</td>
          <td>0.089738</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.174132</td>
          <td>0.332509</td>
          <td>26.163666</td>
          <td>0.123179</td>
          <td>26.146820</td>
          <td>0.109382</td>
          <td>25.511371</td>
          <td>0.102655</td>
          <td>25.462332</td>
          <td>0.183577</td>
          <td>24.947588</td>
          <td>0.258715</td>
          <td>0.113258</td>
          <td>0.098119</td>
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
          <td>26.657211</td>
          <td>0.499055</td>
          <td>26.128004</td>
          <td>0.125284</td>
          <td>25.458827</td>
          <td>0.062953</td>
          <td>25.082934</td>
          <td>0.074395</td>
          <td>24.923441</td>
          <td>0.121781</td>
          <td>24.737925</td>
          <td>0.229050</td>
          <td>0.186611</td>
          <td>0.152871</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.864174</td>
          <td>1.103153</td>
          <td>26.515996</td>
          <td>0.173706</td>
          <td>25.906211</td>
          <td>0.092791</td>
          <td>25.213494</td>
          <td>0.082887</td>
          <td>24.952595</td>
          <td>0.124063</td>
          <td>24.357526</td>
          <td>0.165208</td>
          <td>0.199165</td>
          <td>0.120374</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.644088</td>
          <td>0.470971</td>
          <td>26.786797</td>
          <td>0.205926</td>
          <td>26.442583</td>
          <td>0.138591</td>
          <td>26.030767</td>
          <td>0.157677</td>
          <td>26.118787</td>
          <td>0.309549</td>
          <td>25.881271</td>
          <td>0.524983</td>
          <td>0.083094</td>
          <td>0.055113</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.845158</td>
          <td>0.545563</td>
          <td>26.539686</td>
          <td>0.166981</td>
          <td>25.992420</td>
          <td>0.093545</td>
          <td>25.840799</td>
          <td>0.133764</td>
          <td>25.807304</td>
          <td>0.240023</td>
          <td>24.943412</td>
          <td>0.252626</td>
          <td>0.071036</td>
          <td>0.064720</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.637991</td>
          <td>0.468051</td>
          <td>27.204718</td>
          <td>0.289811</td>
          <td>26.601246</td>
          <td>0.158423</td>
          <td>26.030955</td>
          <td>0.157296</td>
          <td>26.294209</td>
          <td>0.354927</td>
          <td>26.237214</td>
          <td>0.674095</td>
          <td>0.075928</td>
          <td>0.051158</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.693661</td>
          <td>0.204614</td>
          <td>26.082458</td>
          <td>0.109968</td>
          <td>25.156001</td>
          <td>0.080068</td>
          <td>24.792683</td>
          <td>0.109635</td>
          <td>24.017910</td>
          <td>0.125304</td>
          <td>0.184535</td>
          <td>0.133793</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.086009</td>
          <td>1.862236</td>
          <td>27.294253</td>
          <td>0.274632</td>
          <td>26.560560</td>
          <td>0.131526</td>
          <td>26.073721</td>
          <td>0.139473</td>
          <td>25.575743</td>
          <td>0.170391</td>
          <td>24.992006</td>
          <td>0.226122</td>
          <td>0.040904</td>
          <td>0.039143</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.301904</td>
          <td>1.257329</td>
          <td>29.282865</td>
          <td>1.104083</td>
          <td>28.301333</td>
          <td>0.528025</td>
          <td>26.037668</td>
          <td>0.132594</td>
          <td>24.965400</td>
          <td>0.098656</td>
          <td>24.355964</td>
          <td>0.129249</td>
          <td>0.015764</td>
          <td>0.010589</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.792153</td>
          <td>1.712390</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.151112</td>
          <td>0.245640</td>
          <td>26.093558</td>
          <td>0.161888</td>
          <td>25.420115</td>
          <td>0.169234</td>
          <td>27.588953</td>
          <td>1.480473</td>
          <td>0.136684</td>
          <td>0.089738</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.232429</td>
          <td>0.332792</td>
          <td>26.034887</td>
          <td>0.103793</td>
          <td>25.840425</td>
          <td>0.078228</td>
          <td>25.746487</td>
          <td>0.117735</td>
          <td>25.306026</td>
          <td>0.150711</td>
          <td>25.914022</td>
          <td>0.516981</td>
          <td>0.113258</td>
          <td>0.098119</td>
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
          <td>27.477873</td>
          <td>0.894208</td>
          <td>26.412610</td>
          <td>0.165110</td>
          <td>25.437616</td>
          <td>0.063999</td>
          <td>25.100763</td>
          <td>0.078351</td>
          <td>24.636865</td>
          <td>0.098190</td>
          <td>24.558661</td>
          <td>0.204033</td>
          <td>0.186611</td>
          <td>0.152871</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.145659</td>
          <td>2.986076</td>
          <td>26.737901</td>
          <td>0.212923</td>
          <td>26.109991</td>
          <td>0.112948</td>
          <td>25.211431</td>
          <td>0.084301</td>
          <td>24.791018</td>
          <td>0.109776</td>
          <td>24.119017</td>
          <td>0.137114</td>
          <td>0.199165</td>
          <td>0.120374</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.597773</td>
          <td>0.422572</td>
          <td>26.887506</td>
          <td>0.203076</td>
          <td>26.436964</td>
          <td>0.123088</td>
          <td>26.123367</td>
          <td>0.151847</td>
          <td>25.490874</td>
          <td>0.164984</td>
          <td>25.415644</td>
          <td>0.331919</td>
          <td>0.083094</td>
          <td>0.055113</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.272565</td>
          <td>0.327297</td>
          <td>26.443436</td>
          <td>0.138743</td>
          <td>25.983769</td>
          <td>0.082472</td>
          <td>26.163494</td>
          <td>0.156571</td>
          <td>25.743983</td>
          <td>0.203646</td>
          <td>24.928639</td>
          <td>0.222583</td>
          <td>0.071036</td>
          <td>0.064720</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.138015</td>
          <td>0.248124</td>
          <td>26.783543</td>
          <td>0.164389</td>
          <td>26.577781</td>
          <td>0.220942</td>
          <td>25.578444</td>
          <td>0.176168</td>
          <td>25.213879</td>
          <td>0.279874</td>
          <td>0.075928</td>
          <td>0.051158</td>
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
