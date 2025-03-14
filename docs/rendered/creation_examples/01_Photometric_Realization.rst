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

    <pzflow.flow.Flow at 0x7f05c6e9fc70>



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
          <td>26.604861</td>
          <td>0.408802</td>
          <td>26.940512</td>
          <td>0.201482</td>
          <td>25.937736</td>
          <td>0.074627</td>
          <td>25.231795</td>
          <td>0.065209</td>
          <td>24.617194</td>
          <td>0.072419</td>
          <td>23.992821</td>
          <td>0.093918</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.036707</td>
          <td>1.081451</td>
          <td>28.353807</td>
          <td>0.606641</td>
          <td>26.682345</td>
          <td>0.143076</td>
          <td>26.042831</td>
          <td>0.132851</td>
          <td>25.442085</td>
          <td>0.148880</td>
          <td>25.720350</td>
          <td>0.397590</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.033591</td>
          <td>0.868861</td>
          <td>25.770745</td>
          <td>0.104854</td>
          <td>25.151676</td>
          <td>0.115811</td>
          <td>24.090340</td>
          <td>0.102301</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.475877</td>
          <td>1.229516</td>
          <td>27.378200</td>
          <td>0.257025</td>
          <td>26.495814</td>
          <td>0.195584</td>
          <td>25.482361</td>
          <td>0.154113</td>
          <td>25.580177</td>
          <td>0.356523</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.492115</td>
          <td>0.374724</td>
          <td>26.268125</td>
          <td>0.113286</td>
          <td>25.862325</td>
          <td>0.069811</td>
          <td>25.708705</td>
          <td>0.099311</td>
          <td>25.257446</td>
          <td>0.126955</td>
          <td>24.866030</td>
          <td>0.199298</td>
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
          <td>28.766438</td>
          <td>1.594580</td>
          <td>26.356055</td>
          <td>0.122283</td>
          <td>25.416919</td>
          <td>0.047022</td>
          <td>25.017563</td>
          <td>0.053921</td>
          <td>25.052814</td>
          <td>0.106242</td>
          <td>24.820878</td>
          <td>0.191868</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.489210</td>
          <td>0.373878</td>
          <td>26.991247</td>
          <td>0.210226</td>
          <td>26.264913</td>
          <td>0.099541</td>
          <td>25.183868</td>
          <td>0.062496</td>
          <td>24.831899</td>
          <td>0.087526</td>
          <td>24.375409</td>
          <td>0.131114</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.615981</td>
          <td>0.835747</td>
          <td>26.752391</td>
          <td>0.171888</td>
          <td>26.443342</td>
          <td>0.116330</td>
          <td>26.340443</td>
          <td>0.171490</td>
          <td>25.880290</td>
          <td>0.215798</td>
          <td>25.005391</td>
          <td>0.223917</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.389651</td>
          <td>0.345853</td>
          <td>26.123530</td>
          <td>0.099852</td>
          <td>26.004641</td>
          <td>0.079170</td>
          <td>25.773743</td>
          <td>0.105129</td>
          <td>25.676228</td>
          <td>0.181785</td>
          <td>25.393245</td>
          <td>0.307394</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.378461</td>
          <td>0.342818</td>
          <td>26.705153</td>
          <td>0.165116</td>
          <td>26.571430</td>
          <td>0.130012</td>
          <td>26.402970</td>
          <td>0.180837</td>
          <td>25.859617</td>
          <td>0.212106</td>
          <td>25.632341</td>
          <td>0.371369</td>
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
          <td>26.186615</td>
          <td>0.327387</td>
          <td>26.487772</td>
          <td>0.157599</td>
          <td>25.919784</td>
          <td>0.086410</td>
          <td>25.175682</td>
          <td>0.073546</td>
          <td>24.699511</td>
          <td>0.091583</td>
          <td>24.044103</td>
          <td>0.115996</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.051545</td>
          <td>0.625920</td>
          <td>27.234983</td>
          <td>0.293593</td>
          <td>26.749726</td>
          <td>0.177385</td>
          <td>26.219154</td>
          <td>0.182080</td>
          <td>25.843603</td>
          <td>0.243806</td>
          <td>25.863926</td>
          <td>0.511150</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.609258</td>
          <td>3.185148</td>
          <td>28.655655</td>
          <td>0.779115</td>
          <td>26.027592</td>
          <td>0.158157</td>
          <td>24.918782</td>
          <td>0.113446</td>
          <td>24.154278</td>
          <td>0.130554</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.836146</td>
          <td>1.076258</td>
          <td>28.198838</td>
          <td>0.640172</td>
          <td>27.554658</td>
          <td>0.364677</td>
          <td>26.615045</td>
          <td>0.270144</td>
          <td>25.638815</td>
          <td>0.219154</td>
          <td>25.264750</td>
          <td>0.343533</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.434056</td>
          <td>0.397357</td>
          <td>26.181734</td>
          <td>0.121112</td>
          <td>26.094774</td>
          <td>0.100793</td>
          <td>25.764531</td>
          <td>0.123297</td>
          <td>25.243783</td>
          <td>0.147075</td>
          <td>25.208545</td>
          <td>0.308773</td>
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
          <td>26.299340</td>
          <td>0.136571</td>
          <td>25.534499</td>
          <td>0.062792</td>
          <td>25.009640</td>
          <td>0.064904</td>
          <td>24.707935</td>
          <td>0.094221</td>
          <td>24.679590</td>
          <td>0.204053</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.313041</td>
          <td>1.359668</td>
          <td>27.250250</td>
          <td>0.298200</td>
          <td>25.890743</td>
          <td>0.084582</td>
          <td>25.259534</td>
          <td>0.079544</td>
          <td>24.744131</td>
          <td>0.095639</td>
          <td>24.292967</td>
          <td>0.144479</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.114244</td>
          <td>0.658752</td>
          <td>26.419304</td>
          <td>0.150310</td>
          <td>26.527399</td>
          <td>0.148530</td>
          <td>26.183780</td>
          <td>0.178937</td>
          <td>26.044753</td>
          <td>0.290653</td>
          <td>27.561845</td>
          <td>1.480557</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.439776</td>
          <td>0.407497</td>
          <td>26.164905</td>
          <td>0.122721</td>
          <td>26.342773</td>
          <td>0.129010</td>
          <td>26.144538</td>
          <td>0.176329</td>
          <td>25.893473</td>
          <td>0.261498</td>
          <td>25.167842</td>
          <td>0.307691</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.161379</td>
          <td>0.323116</td>
          <td>26.702763</td>
          <td>0.190811</td>
          <td>26.396987</td>
          <td>0.132387</td>
          <td>26.261631</td>
          <td>0.190588</td>
          <td>25.838655</td>
          <td>0.245060</td>
          <td>25.920778</td>
          <td>0.537307</td>
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
          <td>27.915676</td>
          <td>1.006799</td>
          <td>26.410665</td>
          <td>0.128223</td>
          <td>25.982597</td>
          <td>0.077654</td>
          <td>25.069479</td>
          <td>0.056472</td>
          <td>24.639897</td>
          <td>0.073897</td>
          <td>24.046682</td>
          <td>0.098476</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.719808</td>
          <td>0.446390</td>
          <td>27.734144</td>
          <td>0.383378</td>
          <td>26.370421</td>
          <td>0.109268</td>
          <td>26.383319</td>
          <td>0.178021</td>
          <td>26.142898</td>
          <td>0.268240</td>
          <td>25.126949</td>
          <td>0.247825</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.528861</td>
          <td>0.724045</td>
          <td>inf</td>
          <td>inf</td>
          <td>25.779978</td>
          <td>0.115081</td>
          <td>25.033096</td>
          <td>0.113270</td>
          <td>24.674446</td>
          <td>0.183936</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.052897</td>
          <td>0.576077</td>
          <td>27.274756</td>
          <td>0.291002</td>
          <td>25.800570</td>
          <td>0.135666</td>
          <td>25.611945</td>
          <td>0.213574</td>
          <td>24.731378</td>
          <td>0.222096</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.878089</td>
          <td>0.984853</td>
          <td>26.139163</td>
          <td>0.101352</td>
          <td>25.922397</td>
          <td>0.073727</td>
          <td>25.928948</td>
          <td>0.120540</td>
          <td>25.386432</td>
          <td>0.142120</td>
          <td>25.301777</td>
          <td>0.285952</td>
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
          <td>26.332402</td>
          <td>0.128244</td>
          <td>25.469168</td>
          <td>0.053345</td>
          <td>25.128069</td>
          <td>0.064644</td>
          <td>24.993837</td>
          <td>0.109118</td>
          <td>24.916702</td>
          <td>0.224686</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.765903</td>
          <td>2.443696</td>
          <td>26.835193</td>
          <td>0.186939</td>
          <td>25.955790</td>
          <td>0.077095</td>
          <td>25.112582</td>
          <td>0.059701</td>
          <td>24.856499</td>
          <td>0.090927</td>
          <td>24.141697</td>
          <td>0.108837</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.221123</td>
          <td>1.227660</td>
          <td>26.808807</td>
          <td>0.187879</td>
          <td>26.291773</td>
          <td>0.107000</td>
          <td>26.018177</td>
          <td>0.136773</td>
          <td>25.800193</td>
          <td>0.211429</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.738019</td>
          <td>0.220050</td>
          <td>26.044899</td>
          <td>0.103082</td>
          <td>26.072711</td>
          <td>0.094307</td>
          <td>25.884079</td>
          <td>0.130303</td>
          <td>25.343642</td>
          <td>0.153016</td>
          <td>25.207758</td>
          <td>0.295172</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.287555</td>
          <td>0.326964</td>
          <td>26.769255</td>
          <td>0.180169</td>
          <td>26.699524</td>
          <td>0.150838</td>
          <td>26.356462</td>
          <td>0.180823</td>
          <td>25.730784</td>
          <td>0.197586</td>
          <td>25.699075</td>
          <td>0.405206</td>
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
