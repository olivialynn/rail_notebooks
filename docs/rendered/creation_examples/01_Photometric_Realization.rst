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

    <pzflow.flow.Flow at 0x7f2a8ad57ee0>



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
          <td>27.489663</td>
          <td>0.769801</td>
          <td>26.671236</td>
          <td>0.160407</td>
          <td>25.864881</td>
          <td>0.069969</td>
          <td>25.054178</td>
          <td>0.055702</td>
          <td>24.603053</td>
          <td>0.071518</td>
          <td>23.872792</td>
          <td>0.084508</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.637732</td>
          <td>0.419208</td>
          <td>27.573339</td>
          <td>0.337767</td>
          <td>26.507300</td>
          <td>0.122981</td>
          <td>26.325115</td>
          <td>0.169268</td>
          <td>26.264354</td>
          <td>0.295730</td>
          <td>25.297722</td>
          <td>0.284629</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.363443</td>
          <td>0.707572</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.212474</td>
          <td>0.970992</td>
          <td>25.956707</td>
          <td>0.123300</td>
          <td>25.086816</td>
          <td>0.109444</td>
          <td>24.491888</td>
          <td>0.144973</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.960286</td>
          <td>1.033904</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.086313</td>
          <td>0.201747</td>
          <td>26.234344</td>
          <td>0.156648</td>
          <td>25.404752</td>
          <td>0.144178</td>
          <td>25.132844</td>
          <td>0.248801</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.404891</td>
          <td>0.350024</td>
          <td>26.079230</td>
          <td>0.096053</td>
          <td>26.060437</td>
          <td>0.083165</td>
          <td>25.801655</td>
          <td>0.107726</td>
          <td>25.300545</td>
          <td>0.131782</td>
          <td>25.006112</td>
          <td>0.224051</td>
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
          <td>26.653310</td>
          <td>0.424214</td>
          <td>26.377488</td>
          <td>0.124577</td>
          <td>25.345289</td>
          <td>0.044126</td>
          <td>25.162819</td>
          <td>0.061340</td>
          <td>24.997093</td>
          <td>0.101187</td>
          <td>24.738720</td>
          <td>0.178995</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.767070</td>
          <td>0.919419</td>
          <td>26.665575</td>
          <td>0.159633</td>
          <td>26.123798</td>
          <td>0.087938</td>
          <td>25.129793</td>
          <td>0.059569</td>
          <td>24.638788</td>
          <td>0.073815</td>
          <td>24.252619</td>
          <td>0.117865</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.666430</td>
          <td>0.863107</td>
          <td>27.016489</td>
          <td>0.214704</td>
          <td>26.396307</td>
          <td>0.111659</td>
          <td>26.312227</td>
          <td>0.167421</td>
          <td>25.991125</td>
          <td>0.236604</td>
          <td>25.184148</td>
          <td>0.259495</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.367758</td>
          <td>0.339936</td>
          <td>26.108137</td>
          <td>0.098516</td>
          <td>26.260684</td>
          <td>0.099173</td>
          <td>26.073482</td>
          <td>0.136416</td>
          <td>25.631685</td>
          <td>0.175048</td>
          <td>24.648077</td>
          <td>0.165721</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.658811</td>
          <td>0.858938</td>
          <td>26.581369</td>
          <td>0.148530</td>
          <td>26.732949</td>
          <td>0.149438</td>
          <td>26.275886</td>
          <td>0.162311</td>
          <td>25.847182</td>
          <td>0.209912</td>
          <td>25.345117</td>
          <td>0.295733</td>
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
          <td>28.718479</td>
          <td>1.660907</td>
          <td>26.820594</td>
          <td>0.208832</td>
          <td>26.074183</td>
          <td>0.098959</td>
          <td>25.251732</td>
          <td>0.078655</td>
          <td>24.709477</td>
          <td>0.092388</td>
          <td>23.890542</td>
          <td>0.101446</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.301871</td>
          <td>0.309796</td>
          <td>26.738495</td>
          <td>0.175703</td>
          <td>26.734304</td>
          <td>0.279207</td>
          <td>25.945918</td>
          <td>0.265147</td>
          <td>24.828621</td>
          <td>0.226429</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.865958</td>
          <td>1.650501</td>
          <td>28.422512</td>
          <td>0.666010</td>
          <td>25.866395</td>
          <td>0.137708</td>
          <td>24.968931</td>
          <td>0.118506</td>
          <td>24.166596</td>
          <td>0.131953</td>
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
          <td>27.031885</td>
          <td>0.239449</td>
          <td>26.267570</td>
          <td>0.202665</td>
          <td>25.444190</td>
          <td>0.186141</td>
          <td>24.866507</td>
          <td>0.249207</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.088310</td>
          <td>0.302765</td>
          <td>26.143602</td>
          <td>0.117168</td>
          <td>26.109688</td>
          <td>0.102117</td>
          <td>25.733700</td>
          <td>0.120040</td>
          <td>25.467544</td>
          <td>0.178030</td>
          <td>24.888992</td>
          <td>0.238064</td>
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
          <td>26.443357</td>
          <td>0.154554</td>
          <td>25.462103</td>
          <td>0.058889</td>
          <td>25.065276</td>
          <td>0.068181</td>
          <td>24.987967</td>
          <td>0.120328</td>
          <td>24.834826</td>
          <td>0.232230</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.750275</td>
          <td>0.197588</td>
          <td>26.082324</td>
          <td>0.100083</td>
          <td>25.204949</td>
          <td>0.075802</td>
          <td>24.752128</td>
          <td>0.096313</td>
          <td>24.304549</td>
          <td>0.145925</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.212233</td>
          <td>0.337041</td>
          <td>26.884752</td>
          <td>0.222731</td>
          <td>26.282318</td>
          <td>0.120188</td>
          <td>26.052916</td>
          <td>0.160079</td>
          <td>26.106451</td>
          <td>0.305447</td>
          <td>25.349923</td>
          <td>0.349472</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.342340</td>
          <td>0.776648</td>
          <td>26.081697</td>
          <td>0.114166</td>
          <td>25.875653</td>
          <td>0.085779</td>
          <td>26.063114</td>
          <td>0.164529</td>
          <td>26.330425</td>
          <td>0.370820</td>
          <td>24.980124</td>
          <td>0.264337</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.668545</td>
          <td>0.944515</td>
          <td>26.312459</td>
          <td>0.136786</td>
          <td>26.344885</td>
          <td>0.126549</td>
          <td>26.454994</td>
          <td>0.224083</td>
          <td>25.912758</td>
          <td>0.260427</td>
          <td>26.134568</td>
          <td>0.625799</td>
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
          <td>27.383031</td>
          <td>0.717039</td>
          <td>26.828387</td>
          <td>0.183345</td>
          <td>26.032461</td>
          <td>0.081149</td>
          <td>25.305505</td>
          <td>0.069618</td>
          <td>24.596298</td>
          <td>0.071102</td>
          <td>24.107674</td>
          <td>0.103879</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.930004</td>
          <td>0.521788</td>
          <td>27.851648</td>
          <td>0.419653</td>
          <td>26.855989</td>
          <td>0.166178</td>
          <td>26.126581</td>
          <td>0.142946</td>
          <td>26.005010</td>
          <td>0.239546</td>
          <td>25.731586</td>
          <td>0.401392</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.480283</td>
          <td>0.390977</td>
          <td>29.838760</td>
          <td>1.549689</td>
          <td>28.791162</td>
          <td>0.789155</td>
          <td>25.997603</td>
          <td>0.138969</td>
          <td>24.987692</td>
          <td>0.108872</td>
          <td>24.225240</td>
          <td>0.125157</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.261533</td>
          <td>0.750212</td>
          <td>27.656891</td>
          <td>0.430153</td>
          <td>27.402117</td>
          <td>0.322286</td>
          <td>26.459756</td>
          <td>0.237009</td>
          <td>25.609451</td>
          <td>0.213130</td>
          <td>25.228370</td>
          <td>0.332694</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.003644</td>
          <td>0.253828</td>
          <td>25.973417</td>
          <td>0.087642</td>
          <td>26.008733</td>
          <td>0.079570</td>
          <td>25.672148</td>
          <td>0.096321</td>
          <td>25.268308</td>
          <td>0.128335</td>
          <td>25.292161</td>
          <td>0.283735</td>
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
          <td>28.281686</td>
          <td>1.285854</td>
          <td>26.495225</td>
          <td>0.147565</td>
          <td>25.495356</td>
          <td>0.054599</td>
          <td>25.100644</td>
          <td>0.063092</td>
          <td>24.967572</td>
          <td>0.106644</td>
          <td>24.440797</td>
          <td>0.150270</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.031568</td>
          <td>1.086447</td>
          <td>26.673843</td>
          <td>0.163017</td>
          <td>25.981112</td>
          <td>0.078839</td>
          <td>25.223456</td>
          <td>0.065869</td>
          <td>24.858706</td>
          <td>0.091104</td>
          <td>24.132456</td>
          <td>0.107963</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.044160</td>
          <td>0.582506</td>
          <td>26.573640</td>
          <td>0.153833</td>
          <td>26.385939</td>
          <td>0.116158</td>
          <td>26.108406</td>
          <td>0.147824</td>
          <td>25.919681</td>
          <td>0.233521</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.628331</td>
          <td>0.446827</td>
          <td>26.110304</td>
          <td>0.109138</td>
          <td>26.091848</td>
          <td>0.095904</td>
          <td>26.174694</td>
          <td>0.167247</td>
          <td>25.445256</td>
          <td>0.166896</td>
          <td>26.298263</td>
          <td>0.669384</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.730443</td>
          <td>0.460338</td>
          <td>26.633216</td>
          <td>0.160489</td>
          <td>26.413382</td>
          <td>0.117792</td>
          <td>26.287041</td>
          <td>0.170475</td>
          <td>26.292114</td>
          <td>0.313314</td>
          <td>29.237701</td>
          <td>2.765411</td>
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
