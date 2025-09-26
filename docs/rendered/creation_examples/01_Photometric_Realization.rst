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

    <pzflow.flow.Flow at 0x7f5964e40eb0>



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
    0      23.994413  0.088976  0.085463  
    1      25.391064  0.202561  0.152820  
    2      24.304707  0.145709  0.143207  
    3      25.291103  0.073865  0.059589  
    4      25.096743  0.028510  0.018054  
    ...          ...       ...       ...  
    99995  24.737946  0.121692  0.099815  
    99996  24.224169  0.016617  0.013191  
    99997  25.613836  0.017955  0.009619  
    99998  25.274899  0.205053  0.146403  
    99999  25.699642  0.038278  0.020964  
    
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
          <td>28.249425</td>
          <td>1.220320</td>
          <td>26.843244</td>
          <td>0.185641</td>
          <td>26.088187</td>
          <td>0.085224</td>
          <td>25.037294</td>
          <td>0.054874</td>
          <td>24.723861</td>
          <td>0.079576</td>
          <td>24.019106</td>
          <td>0.096110</td>
          <td>0.088976</td>
          <td>0.085463</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.324914</td>
          <td>0.689307</td>
          <td>27.453911</td>
          <td>0.307133</td>
          <td>26.618628</td>
          <td>0.135427</td>
          <td>26.562658</td>
          <td>0.206873</td>
          <td>25.451466</td>
          <td>0.150083</td>
          <td>25.365459</td>
          <td>0.300614</td>
          <td>0.202561</td>
          <td>0.152820</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>30.247636</td>
          <td>2.871505</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.194086</td>
          <td>0.486976</td>
          <td>25.869608</td>
          <td>0.114305</td>
          <td>25.156435</td>
          <td>0.116292</td>
          <td>24.547601</td>
          <td>0.152078</td>
          <td>0.145709</td>
          <td>0.143207</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.418824</td>
          <td>0.734424</td>
          <td>29.695556</td>
          <td>1.383115</td>
          <td>27.322549</td>
          <td>0.245541</td>
          <td>26.260065</td>
          <td>0.160132</td>
          <td>25.697674</td>
          <td>0.185113</td>
          <td>25.437079</td>
          <td>0.318359</td>
          <td>0.073865</td>
          <td>0.059589</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.220392</td>
          <td>0.302319</td>
          <td>26.147768</td>
          <td>0.101992</td>
          <td>25.983119</td>
          <td>0.077680</td>
          <td>25.685430</td>
          <td>0.097305</td>
          <td>25.482240</td>
          <td>0.154097</td>
          <td>25.366855</td>
          <td>0.300952</td>
          <td>0.028510</td>
          <td>0.018054</td>
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
          <td>28.178709</td>
          <td>1.173111</td>
          <td>26.153008</td>
          <td>0.102460</td>
          <td>25.373267</td>
          <td>0.045235</td>
          <td>25.004826</td>
          <td>0.053314</td>
          <td>24.983865</td>
          <td>0.100021</td>
          <td>24.748884</td>
          <td>0.180543</td>
          <td>0.121692</td>
          <td>0.099815</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.164325</td>
          <td>0.288986</td>
          <td>26.767049</td>
          <td>0.174041</td>
          <td>26.162634</td>
          <td>0.090994</td>
          <td>25.078125</td>
          <td>0.056899</td>
          <td>24.888164</td>
          <td>0.091965</td>
          <td>24.157619</td>
          <td>0.108499</td>
          <td>0.016617</td>
          <td>0.013191</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.689965</td>
          <td>0.436188</td>
          <td>26.708533</td>
          <td>0.165592</td>
          <td>26.567963</td>
          <td>0.129622</td>
          <td>26.097450</td>
          <td>0.139267</td>
          <td>25.988553</td>
          <td>0.236101</td>
          <td>25.448727</td>
          <td>0.321329</td>
          <td>0.017955</td>
          <td>0.009619</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.259578</td>
          <td>0.311953</td>
          <td>26.350549</td>
          <td>0.121700</td>
          <td>26.140325</td>
          <td>0.089226</td>
          <td>25.800860</td>
          <td>0.107651</td>
          <td>25.612366</td>
          <td>0.172198</td>
          <td>24.905235</td>
          <td>0.205963</td>
          <td>0.205053</td>
          <td>0.146403</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.996410</td>
          <td>0.547285</td>
          <td>27.060193</td>
          <td>0.222661</td>
          <td>26.556843</td>
          <td>0.128380</td>
          <td>26.591575</td>
          <td>0.211939</td>
          <td>26.172941</td>
          <td>0.274641</td>
          <td>25.174227</td>
          <td>0.257396</td>
          <td>0.038278</td>
          <td>0.020964</td>
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
          <td>27.336597</td>
          <td>0.770966</td>
          <td>26.678880</td>
          <td>0.189491</td>
          <td>25.930752</td>
          <td>0.089483</td>
          <td>25.225892</td>
          <td>0.078928</td>
          <td>24.732224</td>
          <td>0.096658</td>
          <td>23.824430</td>
          <td>0.098247</td>
          <td>0.088976</td>
          <td>0.085463</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.150996</td>
          <td>1.303978</td>
          <td>27.594043</td>
          <td>0.421462</td>
          <td>26.271485</td>
          <td>0.129535</td>
          <td>26.691383</td>
          <td>0.295945</td>
          <td>25.922768</td>
          <td>0.284848</td>
          <td>26.139142</td>
          <td>0.674062</td>
          <td>0.202561</td>
          <td>0.152820</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.958600</td>
          <td>0.539364</td>
          <td>28.861360</td>
          <td>0.917524</td>
          <td>26.004642</td>
          <td>0.162090</td>
          <td>24.886877</td>
          <td>0.115255</td>
          <td>24.131877</td>
          <td>0.133866</td>
          <td>0.145709</td>
          <td>0.143207</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.133715</td>
          <td>1.241160</td>
          <td>28.402863</td>
          <td>0.709639</td>
          <td>27.489968</td>
          <td>0.330767</td>
          <td>26.384322</td>
          <td>0.212315</td>
          <td>25.552268</td>
          <td>0.193971</td>
          <td>25.249488</td>
          <td>0.323453</td>
          <td>0.073865</td>
          <td>0.059589</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.310566</td>
          <td>0.361431</td>
          <td>26.167803</td>
          <td>0.119826</td>
          <td>25.988133</td>
          <td>0.091938</td>
          <td>25.543874</td>
          <td>0.101889</td>
          <td>25.690159</td>
          <td>0.215021</td>
          <td>25.224400</td>
          <td>0.313177</td>
          <td>0.028510</td>
          <td>0.018054</td>
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
          <td>27.344316</td>
          <td>0.781692</td>
          <td>26.183198</td>
          <td>0.125688</td>
          <td>25.420075</td>
          <td>0.057858</td>
          <td>25.105023</td>
          <td>0.072062</td>
          <td>24.885604</td>
          <td>0.112206</td>
          <td>24.846173</td>
          <td>0.238845</td>
          <td>0.121692</td>
          <td>0.099815</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.416994</td>
          <td>2.242148</td>
          <td>26.399864</td>
          <td>0.146260</td>
          <td>26.046749</td>
          <td>0.096679</td>
          <td>25.062557</td>
          <td>0.066592</td>
          <td>24.839008</td>
          <td>0.103571</td>
          <td>24.436361</td>
          <td>0.162803</td>
          <td>0.016617</td>
          <td>0.013191</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.393437</td>
          <td>1.415465</td>
          <td>26.566199</td>
          <td>0.168597</td>
          <td>26.204315</td>
          <td>0.110957</td>
          <td>26.293509</td>
          <td>0.193966</td>
          <td>26.132359</td>
          <td>0.308434</td>
          <td>25.462606</td>
          <td>0.377471</td>
          <td>0.017955</td>
          <td>0.009619</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.455799</td>
          <td>0.431256</td>
          <td>26.237404</td>
          <td>0.138554</td>
          <td>26.205184</td>
          <td>0.122122</td>
          <td>25.859806</td>
          <td>0.147611</td>
          <td>26.015572</td>
          <td>0.306539</td>
          <td>25.108765</td>
          <td>0.311886</td>
          <td>0.205053</td>
          <td>0.146403</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.413951</td>
          <td>0.391997</td>
          <td>27.108557</td>
          <td>0.265647</td>
          <td>26.914850</td>
          <td>0.204474</td>
          <td>26.427524</td>
          <td>0.217556</td>
          <td>25.838674</td>
          <td>0.243500</td>
          <td>26.271966</td>
          <td>0.684377</td>
          <td>0.038278</td>
          <td>0.020964</td>
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
          <td>28.126917</td>
          <td>1.189042</td>
          <td>26.708297</td>
          <td>0.179255</td>
          <td>25.966536</td>
          <td>0.084109</td>
          <td>25.283709</td>
          <td>0.075364</td>
          <td>24.710456</td>
          <td>0.086389</td>
          <td>23.974234</td>
          <td>0.101800</td>
          <td>0.088976</td>
          <td>0.085463</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.075786</td>
          <td>0.696160</td>
          <td>27.768018</td>
          <td>0.498369</td>
          <td>26.585008</td>
          <td>0.177392</td>
          <td>26.184542</td>
          <td>0.204011</td>
          <td>25.981215</td>
          <td>0.311628</td>
          <td>26.103921</td>
          <td>0.682923</td>
          <td>0.202561</td>
          <td>0.152820</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.478088</td>
          <td>0.374154</td>
          <td>30.234002</td>
          <td>1.885161</td>
          <td>25.645270</td>
          <td>0.118294</td>
          <td>25.122107</td>
          <td>0.140583</td>
          <td>24.254801</td>
          <td>0.148048</td>
          <td>0.145709</td>
          <td>0.143207</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.592163</td>
          <td>0.742995</td>
          <td>27.792407</td>
          <td>0.377112</td>
          <td>26.200820</td>
          <td>0.161404</td>
          <td>25.214362</td>
          <td>0.129429</td>
          <td>24.979992</td>
          <td>0.231938</td>
          <td>0.073865</td>
          <td>0.059589</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.072895</td>
          <td>0.269675</td>
          <td>26.057840</td>
          <td>0.094898</td>
          <td>25.988977</td>
          <td>0.078688</td>
          <td>25.826291</td>
          <td>0.110952</td>
          <td>25.427974</td>
          <td>0.148192</td>
          <td>25.344161</td>
          <td>0.297670</td>
          <td>0.028510</td>
          <td>0.018054</td>
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
          <td>26.846277</td>
          <td>0.534773</td>
          <td>26.390652</td>
          <td>0.142749</td>
          <td>25.375175</td>
          <td>0.052411</td>
          <td>25.016394</td>
          <td>0.062685</td>
          <td>24.915552</td>
          <td>0.108698</td>
          <td>24.749318</td>
          <td>0.208226</td>
          <td>0.121692</td>
          <td>0.099815</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.671738</td>
          <td>0.430986</td>
          <td>26.799048</td>
          <td>0.179285</td>
          <td>26.042456</td>
          <td>0.082108</td>
          <td>25.229385</td>
          <td>0.065281</td>
          <td>24.549293</td>
          <td>0.068405</td>
          <td>24.299855</td>
          <td>0.123189</td>
          <td>0.016617</td>
          <td>0.013191</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.833251</td>
          <td>0.486448</td>
          <td>26.792393</td>
          <td>0.178242</td>
          <td>26.208933</td>
          <td>0.095038</td>
          <td>26.318997</td>
          <td>0.168871</td>
          <td>26.302697</td>
          <td>0.305771</td>
          <td>25.480680</td>
          <td>0.330468</td>
          <td>0.017955</td>
          <td>0.009619</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.052705</td>
          <td>0.683372</td>
          <td>26.243429</td>
          <td>0.144901</td>
          <td>26.100677</td>
          <td>0.116427</td>
          <td>26.021961</td>
          <td>0.177027</td>
          <td>25.809451</td>
          <td>0.270086</td>
          <td>24.963020</td>
          <td>0.288885</td>
          <td>0.205053</td>
          <td>0.146403</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.449260</td>
          <td>0.365275</td>
          <td>26.463410</td>
          <td>0.135655</td>
          <td>26.430479</td>
          <td>0.116495</td>
          <td>26.094154</td>
          <td>0.140704</td>
          <td>26.078339</td>
          <td>0.257253</td>
          <td>26.827140</td>
          <td>0.876748</td>
          <td>0.038278</td>
          <td>0.020964</td>
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
