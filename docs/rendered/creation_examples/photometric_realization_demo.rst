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

    <pzflow.flow.Flow at 0x7f96be080a90>



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
          <td>27.706186</td>
          <td>0.885077</td>
          <td>27.080265</td>
          <td>0.226404</td>
          <td>26.017088</td>
          <td>0.080045</td>
          <td>25.131189</td>
          <td>0.059643</td>
          <td>24.845108</td>
          <td>0.088549</td>
          <td>23.928309</td>
          <td>0.088741</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.649697</td>
          <td>0.358693</td>
          <td>26.703886</td>
          <td>0.145752</td>
          <td>26.272091</td>
          <td>0.161786</td>
          <td>26.023408</td>
          <td>0.242993</td>
          <td>25.179561</td>
          <td>0.258522</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.857205</td>
          <td>1.665098</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.242741</td>
          <td>0.504807</td>
          <td>25.993852</td>
          <td>0.127337</td>
          <td>25.033214</td>
          <td>0.104437</td>
          <td>24.451461</td>
          <td>0.140014</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.418921</td>
          <td>1.191248</td>
          <td>27.631923</td>
          <td>0.315607</td>
          <td>26.347698</td>
          <td>0.172552</td>
          <td>25.417412</td>
          <td>0.145757</td>
          <td>24.952596</td>
          <td>0.214283</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.987445</td>
          <td>0.250250</td>
          <td>26.074608</td>
          <td>0.095664</td>
          <td>26.080679</td>
          <td>0.084662</td>
          <td>25.653964</td>
          <td>0.094655</td>
          <td>25.473307</td>
          <td>0.152922</td>
          <td>25.863829</td>
          <td>0.443610</td>
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
          <td>27.001603</td>
          <td>0.549342</td>
          <td>26.158202</td>
          <td>0.102927</td>
          <td>25.453469</td>
          <td>0.048573</td>
          <td>25.108154</td>
          <td>0.058436</td>
          <td>24.915811</td>
          <td>0.094226</td>
          <td>24.908543</td>
          <td>0.206534</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.945823</td>
          <td>0.527558</td>
          <td>26.728800</td>
          <td>0.168474</td>
          <td>26.082277</td>
          <td>0.084781</td>
          <td>25.286156</td>
          <td>0.068426</td>
          <td>24.745863</td>
          <td>0.081136</td>
          <td>24.358947</td>
          <td>0.129259</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.487083</td>
          <td>0.768492</td>
          <td>26.425398</td>
          <td>0.129853</td>
          <td>26.343106</td>
          <td>0.106591</td>
          <td>26.548934</td>
          <td>0.204508</td>
          <td>25.979644</td>
          <td>0.234368</td>
          <td>25.752913</td>
          <td>0.407674</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.716669</td>
          <td>0.445084</td>
          <td>26.334837</td>
          <td>0.120052</td>
          <td>26.174984</td>
          <td>0.091987</td>
          <td>26.096373</td>
          <td>0.139138</td>
          <td>25.752060</td>
          <td>0.193806</td>
          <td>25.143300</td>
          <td>0.250948</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.220716</td>
          <td>0.641614</td>
          <td>26.372548</td>
          <td>0.124045</td>
          <td>26.518840</td>
          <td>0.124219</td>
          <td>26.053749</td>
          <td>0.134111</td>
          <td>25.992823</td>
          <td>0.236936</td>
          <td>24.836006</td>
          <td>0.194328</td>
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
          <td>26.150552</td>
          <td>0.318135</td>
          <td>27.015716</td>
          <td>0.245528</td>
          <td>25.957394</td>
          <td>0.089317</td>
          <td>25.146666</td>
          <td>0.071683</td>
          <td>24.641688</td>
          <td>0.087043</td>
          <td>23.826049</td>
          <td>0.095872</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.938873</td>
          <td>0.505811</td>
          <td>26.596430</td>
          <td>0.155663</td>
          <td>26.001438</td>
          <td>0.151246</td>
          <td>25.736594</td>
          <td>0.223141</td>
          <td>25.055561</td>
          <td>0.272868</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.275648</td>
          <td>2.872128</td>
          <td>28.233042</td>
          <td>0.583243</td>
          <td>26.036639</td>
          <td>0.159385</td>
          <td>25.140070</td>
          <td>0.137440</td>
          <td>24.352519</td>
          <td>0.154846</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.555911</td>
          <td>1.580558</td>
          <td>28.948384</td>
          <td>1.038158</td>
          <td>27.124430</td>
          <td>0.258377</td>
          <td>26.092007</td>
          <td>0.174754</td>
          <td>25.892323</td>
          <td>0.270057</td>
          <td>26.747864</td>
          <td>0.976450</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.024188</td>
          <td>0.614075</td>
          <td>26.238773</td>
          <td>0.127248</td>
          <td>25.991287</td>
          <td>0.092047</td>
          <td>25.764227</td>
          <td>0.123265</td>
          <td>25.828895</td>
          <td>0.240894</td>
          <td>24.955637</td>
          <td>0.251495</td>
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
          <td>26.838693</td>
          <td>0.544920</td>
          <td>26.194631</td>
          <td>0.124755</td>
          <td>25.481495</td>
          <td>0.059910</td>
          <td>25.017227</td>
          <td>0.065341</td>
          <td>24.689649</td>
          <td>0.092720</td>
          <td>24.400920</td>
          <td>0.161179</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.852688</td>
          <td>0.215272</td>
          <td>26.119985</td>
          <td>0.103437</td>
          <td>25.202045</td>
          <td>0.075607</td>
          <td>24.653966</td>
          <td>0.088357</td>
          <td>24.099013</td>
          <td>0.122183</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.838688</td>
          <td>0.542105</td>
          <td>26.871545</td>
          <td>0.220298</td>
          <td>26.549727</td>
          <td>0.151404</td>
          <td>26.711262</td>
          <td>0.277348</td>
          <td>25.905121</td>
          <td>0.259467</td>
          <td>27.616208</td>
          <td>1.521271</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.686808</td>
          <td>0.966098</td>
          <td>26.206110</td>
          <td>0.127180</td>
          <td>25.898894</td>
          <td>0.087552</td>
          <td>25.861608</td>
          <td>0.138416</td>
          <td>25.538061</td>
          <td>0.194688</td>
          <td>25.077203</td>
          <td>0.286038</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.712744</td>
          <td>0.493516</td>
          <td>26.614363</td>
          <td>0.177076</td>
          <td>26.528288</td>
          <td>0.148245</td>
          <td>26.223570</td>
          <td>0.184561</td>
          <td>26.689976</td>
          <td>0.478827</td>
          <td>25.359996</td>
          <td>0.351364</td>
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
          <td>27.562531</td>
          <td>0.807449</td>
          <td>26.816613</td>
          <td>0.181528</td>
          <td>25.956890</td>
          <td>0.075911</td>
          <td>25.181873</td>
          <td>0.062394</td>
          <td>24.693064</td>
          <td>0.077451</td>
          <td>24.199666</td>
          <td>0.112568</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.736789</td>
          <td>2.406719</td>
          <td>27.365649</td>
          <td>0.286284</td>
          <td>26.563877</td>
          <td>0.129285</td>
          <td>26.154265</td>
          <td>0.146391</td>
          <td>25.706462</td>
          <td>0.186662</td>
          <td>25.159627</td>
          <td>0.254567</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.404434</td>
          <td>1.374658</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.338904</td>
          <td>1.839882</td>
          <td>26.110340</td>
          <td>0.153112</td>
          <td>24.809112</td>
          <td>0.093112</td>
          <td>24.567475</td>
          <td>0.167973</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.058621</td>
          <td>0.578436</td>
          <td>27.473465</td>
          <td>0.341047</td>
          <td>26.379091</td>
          <td>0.221676</td>
          <td>25.462072</td>
          <td>0.188326</td>
          <td>25.224313</td>
          <td>0.331626</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.043950</td>
          <td>0.262332</td>
          <td>26.011352</td>
          <td>0.090612</td>
          <td>25.993436</td>
          <td>0.078503</td>
          <td>25.634712</td>
          <td>0.093207</td>
          <td>25.370857</td>
          <td>0.140226</td>
          <td>24.916653</td>
          <td>0.208234</td>
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
          <td>26.690009</td>
          <td>0.457617</td>
          <td>26.493051</td>
          <td>0.147290</td>
          <td>25.375768</td>
          <td>0.049101</td>
          <td>25.030149</td>
          <td>0.059268</td>
          <td>24.961274</td>
          <td>0.106059</td>
          <td>24.994419</td>
          <td>0.239624</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.705013</td>
          <td>0.891712</td>
          <td>26.882833</td>
          <td>0.194598</td>
          <td>26.181925</td>
          <td>0.094088</td>
          <td>25.201170</td>
          <td>0.064581</td>
          <td>24.894535</td>
          <td>0.094017</td>
          <td>24.295397</td>
          <td>0.124416</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.471399</td>
          <td>0.780463</td>
          <td>26.524591</td>
          <td>0.147498</td>
          <td>26.255615</td>
          <td>0.103670</td>
          <td>26.442817</td>
          <td>0.196459</td>
          <td>25.820914</td>
          <td>0.215119</td>
          <td>28.088581</td>
          <td>1.775034</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.110593</td>
          <td>0.298412</td>
          <td>26.220888</td>
          <td>0.120159</td>
          <td>26.181020</td>
          <td>0.103697</td>
          <td>26.026714</td>
          <td>0.147357</td>
          <td>25.853588</td>
          <td>0.235195</td>
          <td>25.322585</td>
          <td>0.323601</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.108635</td>
          <td>0.606219</td>
          <td>27.176462</td>
          <td>0.253040</td>
          <td>26.407382</td>
          <td>0.117178</td>
          <td>26.327537</td>
          <td>0.176443</td>
          <td>26.351094</td>
          <td>0.328390</td>
          <td>25.804500</td>
          <td>0.439136</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
