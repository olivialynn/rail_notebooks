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

    <pzflow.flow.Flow at 0x7fd031716ef0>



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
    0      23.994413  0.139964  0.105850  
    1      25.391064  0.029368  0.016871  
    2      24.304707  0.072139  0.044052  
    3      25.291103  0.108785  0.079961  
    4      25.096743  0.084019  0.063656  
    ...          ...       ...       ...  
    99995  24.737946  0.065413  0.051929  
    99996  24.224169  0.048921  0.034361  
    99997  25.613836  0.084607  0.067062  
    99998  25.274899  0.046989  0.030278  
    99999  25.699642  0.143239  0.077984  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.856397</td>
          <td>0.187714</td>
          <td>25.940193</td>
          <td>0.074789</td>
          <td>25.129385</td>
          <td>0.059547</td>
          <td>24.641113</td>
          <td>0.073967</td>
          <td>23.903830</td>
          <td>0.086850</td>
          <td>0.139964</td>
          <td>0.105850</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.192559</td>
          <td>1.182276</td>
          <td>27.292395</td>
          <td>0.269556</td>
          <td>26.576525</td>
          <td>0.130586</td>
          <td>26.177489</td>
          <td>0.149196</td>
          <td>25.983001</td>
          <td>0.235019</td>
          <td>24.981652</td>
          <td>0.219538</td>
          <td>0.029368</td>
          <td>0.016871</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.225506</td>
          <td>1.788889</td>
          <td>27.801765</td>
          <td>0.360986</td>
          <td>26.129625</td>
          <td>0.143181</td>
          <td>25.075612</td>
          <td>0.108379</td>
          <td>24.238123</td>
          <td>0.116388</td>
          <td>0.072139</td>
          <td>0.044052</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.506065</td>
          <td>1.250065</td>
          <td>27.837967</td>
          <td>0.371343</td>
          <td>26.076202</td>
          <td>0.136737</td>
          <td>25.657449</td>
          <td>0.178916</td>
          <td>25.250751</td>
          <td>0.273985</td>
          <td>0.108785</td>
          <td>0.079961</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.084305</td>
          <td>0.270851</td>
          <td>26.005751</td>
          <td>0.090056</td>
          <td>25.947405</td>
          <td>0.075267</td>
          <td>25.657899</td>
          <td>0.094982</td>
          <td>25.397223</td>
          <td>0.143247</td>
          <td>25.424231</td>
          <td>0.315111</td>
          <td>0.084019</td>
          <td>0.063656</td>
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
          <td>27.509075</td>
          <td>0.779696</td>
          <td>26.228153</td>
          <td>0.109409</td>
          <td>25.368055</td>
          <td>0.045026</td>
          <td>25.093400</td>
          <td>0.057676</td>
          <td>24.887400</td>
          <td>0.091904</td>
          <td>24.936744</td>
          <td>0.211465</td>
          <td>0.065413</td>
          <td>0.051929</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.182888</td>
          <td>0.624912</td>
          <td>26.811202</td>
          <td>0.180679</td>
          <td>26.004295</td>
          <td>0.079146</td>
          <td>25.182051</td>
          <td>0.062395</td>
          <td>24.858922</td>
          <td>0.089631</td>
          <td>23.994030</td>
          <td>0.094018</td>
          <td>0.048921</td>
          <td>0.034361</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.004461</td>
          <td>0.253765</td>
          <td>26.502900</td>
          <td>0.138838</td>
          <td>26.420562</td>
          <td>0.114045</td>
          <td>26.142620</td>
          <td>0.144791</td>
          <td>26.060403</td>
          <td>0.250506</td>
          <td>26.010761</td>
          <td>0.495116</td>
          <td>0.084607</td>
          <td>0.067062</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.304323</td>
          <td>0.323277</td>
          <td>26.075189</td>
          <td>0.095713</td>
          <td>26.073566</td>
          <td>0.084133</td>
          <td>25.691993</td>
          <td>0.097866</td>
          <td>25.577671</td>
          <td>0.167188</td>
          <td>25.994388</td>
          <td>0.489151</td>
          <td>0.046989</td>
          <td>0.030278</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.253605</td>
          <td>0.310467</td>
          <td>26.451301</td>
          <td>0.132793</td>
          <td>26.520408</td>
          <td>0.124388</td>
          <td>26.443580</td>
          <td>0.187157</td>
          <td>26.025546</td>
          <td>0.243422</td>
          <td>25.329912</td>
          <td>0.292130</td>
          <td>0.143239</td>
          <td>0.077984</td>
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
          <td>26.552861</td>
          <td>0.174004</td>
          <td>26.126303</td>
          <td>0.108821</td>
          <td>25.164820</td>
          <td>0.076704</td>
          <td>24.512192</td>
          <td>0.081624</td>
          <td>23.883433</td>
          <td>0.106063</td>
          <td>0.139964</td>
          <td>0.105850</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.273236</td>
          <td>0.303194</td>
          <td>26.624798</td>
          <td>0.159752</td>
          <td>26.201936</td>
          <td>0.179751</td>
          <td>25.408968</td>
          <td>0.169651</td>
          <td>25.502329</td>
          <td>0.389718</td>
          <td>0.029368</td>
          <td>0.016871</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.546114</td>
          <td>0.344821</td>
          <td>26.002851</td>
          <td>0.153239</td>
          <td>24.821271</td>
          <td>0.103124</td>
          <td>24.283168</td>
          <td>0.144394</td>
          <td>0.072139</td>
          <td>0.044052</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.827289</td>
          <td>0.476384</td>
          <td>27.235098</td>
          <td>0.273210</td>
          <td>26.147749</td>
          <td>0.176558</td>
          <td>25.549431</td>
          <td>0.196291</td>
          <td>25.003724</td>
          <td>0.269109</td>
          <td>0.108785</td>
          <td>0.079961</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.126608</td>
          <td>0.117345</td>
          <td>25.986126</td>
          <td>0.093315</td>
          <td>25.904203</td>
          <td>0.141724</td>
          <td>25.252457</td>
          <td>0.150830</td>
          <td>25.863807</td>
          <td>0.519183</td>
          <td>0.084019</td>
          <td>0.063656</td>
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
          <td>27.709113</td>
          <td>0.969111</td>
          <td>26.277048</td>
          <td>0.132867</td>
          <td>25.493044</td>
          <td>0.059961</td>
          <td>25.183579</td>
          <td>0.074971</td>
          <td>24.962445</td>
          <td>0.116610</td>
          <td>24.578724</td>
          <td>0.185739</td>
          <td>0.065413</td>
          <td>0.051929</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.539339</td>
          <td>0.432278</td>
          <td>26.864938</td>
          <td>0.217825</td>
          <td>26.103710</td>
          <td>0.102163</td>
          <td>25.154440</td>
          <td>0.072632</td>
          <td>24.855403</td>
          <td>0.105620</td>
          <td>24.287568</td>
          <td>0.144077</td>
          <td>0.048921</td>
          <td>0.034361</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.025496</td>
          <td>0.291724</td>
          <td>26.466076</td>
          <td>0.157367</td>
          <td>26.192082</td>
          <td>0.111842</td>
          <td>26.252636</td>
          <td>0.190917</td>
          <td>25.729388</td>
          <td>0.225871</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.084607</td>
          <td>0.067062</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.145714</td>
          <td>0.318062</td>
          <td>26.277134</td>
          <td>0.132118</td>
          <td>26.186033</td>
          <td>0.109700</td>
          <td>25.990563</td>
          <td>0.150609</td>
          <td>25.754383</td>
          <td>0.227545</td>
          <td>25.920787</td>
          <td>0.535135</td>
          <td>0.046989</td>
          <td>0.030278</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.486046</td>
          <td>0.857882</td>
          <td>26.769950</td>
          <td>0.207694</td>
          <td>26.414587</td>
          <td>0.138807</td>
          <td>26.685359</td>
          <td>0.279452</td>
          <td>25.745963</td>
          <td>0.234088</td>
          <td>26.142255</td>
          <td>0.646119</td>
          <td>0.143239</td>
          <td>0.077984</td>
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
          <td>26.978403</td>
          <td>0.598375</td>
          <td>26.653057</td>
          <td>0.183053</td>
          <td>25.953787</td>
          <td>0.089966</td>
          <td>25.145457</td>
          <td>0.072383</td>
          <td>24.627280</td>
          <td>0.086846</td>
          <td>24.051296</td>
          <td>0.117959</td>
          <td>0.139964</td>
          <td>0.105850</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.188611</td>
          <td>0.249161</td>
          <td>26.322278</td>
          <td>0.105476</td>
          <td>26.229517</td>
          <td>0.157239</td>
          <td>25.611698</td>
          <td>0.173388</td>
          <td>25.275298</td>
          <td>0.281579</td>
          <td>0.029368</td>
          <td>0.016871</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.054981</td>
          <td>1.116415</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.214357</td>
          <td>0.234439</td>
          <td>25.813439</td>
          <td>0.114156</td>
          <td>24.997003</td>
          <td>0.105902</td>
          <td>24.259157</td>
          <td>0.124244</td>
          <td>0.072139</td>
          <td>0.044052</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.816758</td>
          <td>0.888937</td>
          <td>27.113572</td>
          <td>0.228947</td>
          <td>26.102790</td>
          <td>0.156427</td>
          <td>25.329220</td>
          <td>0.150342</td>
          <td>26.237646</td>
          <td>0.639084</td>
          <td>0.108785</td>
          <td>0.079961</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.330316</td>
          <td>0.345019</td>
          <td>26.130332</td>
          <td>0.106753</td>
          <td>25.940550</td>
          <td>0.080257</td>
          <td>25.941228</td>
          <td>0.130749</td>
          <td>25.380170</td>
          <td>0.151126</td>
          <td>25.071161</td>
          <td>0.253052</td>
          <td>0.084019</td>
          <td>0.063656</td>
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
          <td>26.773021</td>
          <td>0.477001</td>
          <td>26.515385</td>
          <td>0.145855</td>
          <td>25.427025</td>
          <td>0.049666</td>
          <td>24.994239</td>
          <td>0.055406</td>
          <td>25.000530</td>
          <td>0.106153</td>
          <td>25.188743</td>
          <td>0.272039</td>
          <td>0.065413</td>
          <td>0.051929</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.724799</td>
          <td>0.171278</td>
          <td>26.066434</td>
          <td>0.085613</td>
          <td>25.159556</td>
          <td>0.062716</td>
          <td>24.822047</td>
          <td>0.088848</td>
          <td>24.018013</td>
          <td>0.098402</td>
          <td>0.048921</td>
          <td>0.034361</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.123206</td>
          <td>0.624561</td>
          <td>26.850658</td>
          <td>0.198621</td>
          <td>26.466216</td>
          <td>0.127554</td>
          <td>26.287254</td>
          <td>0.176488</td>
          <td>25.614108</td>
          <td>0.185032</td>
          <td>25.948828</td>
          <td>0.504028</td>
          <td>0.084607</td>
          <td>0.067062</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.865495</td>
          <td>0.229415</td>
          <td>25.959210</td>
          <td>0.088022</td>
          <td>26.108333</td>
          <td>0.088569</td>
          <td>25.842449</td>
          <td>0.114064</td>
          <td>25.653189</td>
          <td>0.181862</td>
          <td>25.189337</td>
          <td>0.265861</td>
          <td>0.046989</td>
          <td>0.030278</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.356925</td>
          <td>0.371193</td>
          <td>26.787566</td>
          <td>0.200876</td>
          <td>26.682926</td>
          <td>0.165369</td>
          <td>26.169711</td>
          <td>0.172177</td>
          <td>25.523247</td>
          <td>0.184147</td>
          <td>25.250253</td>
          <td>0.315075</td>
          <td>0.143239</td>
          <td>0.077984</td>
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
