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

    <pzflow.flow.Flow at 0x7fb16c5e9600>



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
    0      23.994413  0.048858  0.033015  
    1      25.391064  0.066095  0.062192  
    2      24.304707  0.100134  0.062612  
    3      25.291103  0.031081  0.030694  
    4      25.096743  0.182746  0.146457  
    ...          ...       ...       ...  
    99995  24.737946  0.050134  0.029700  
    99996  24.224169  0.028198  0.027887  
    99997  25.613836  0.211504  0.131470  
    99998  25.274899  0.019921  0.010344  
    99999  25.699642  0.119469  0.110149  
    
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
          <td>27.112170</td>
          <td>0.594557</td>
          <td>26.856813</td>
          <td>0.187780</td>
          <td>26.021924</td>
          <td>0.080387</td>
          <td>25.138534</td>
          <td>0.060033</td>
          <td>24.627875</td>
          <td>0.073106</td>
          <td>23.961052</td>
          <td>0.091333</td>
          <td>0.048858</td>
          <td>0.033015</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.778300</td>
          <td>0.466175</td>
          <td>27.392780</td>
          <td>0.292405</td>
          <td>26.742152</td>
          <td>0.150623</td>
          <td>26.141810</td>
          <td>0.144691</td>
          <td>26.352042</td>
          <td>0.317272</td>
          <td>25.440999</td>
          <td>0.319356</td>
          <td>0.066095</td>
          <td>0.062192</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.706991</td>
          <td>0.701238</td>
          <td>26.077445</td>
          <td>0.136884</td>
          <td>24.934688</td>
          <td>0.095800</td>
          <td>24.427181</td>
          <td>0.137112</td>
          <td>0.100134</td>
          <td>0.062612</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.979322</td>
          <td>1.045630</td>
          <td>28.118214</td>
          <td>0.512017</td>
          <td>27.731386</td>
          <td>0.341550</td>
          <td>26.155311</td>
          <td>0.146380</td>
          <td>25.617705</td>
          <td>0.172981</td>
          <td>25.150623</td>
          <td>0.252461</td>
          <td>0.031081</td>
          <td>0.030694</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.193754</td>
          <td>0.295919</td>
          <td>26.129947</td>
          <td>0.100415</td>
          <td>25.939615</td>
          <td>0.074751</td>
          <td>25.695300</td>
          <td>0.098151</td>
          <td>25.656598</td>
          <td>0.178787</td>
          <td>25.196773</td>
          <td>0.262188</td>
          <td>0.182746</td>
          <td>0.146457</td>
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
          <td>26.487830</td>
          <td>0.137046</td>
          <td>25.398884</td>
          <td>0.046275</td>
          <td>25.118784</td>
          <td>0.058990</td>
          <td>24.806802</td>
          <td>0.085613</td>
          <td>24.820998</td>
          <td>0.191887</td>
          <td>0.050134</td>
          <td>0.029700</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.271801</td>
          <td>0.315012</td>
          <td>26.851574</td>
          <td>0.186952</td>
          <td>26.131997</td>
          <td>0.088575</td>
          <td>25.137323</td>
          <td>0.059968</td>
          <td>24.922811</td>
          <td>0.094807</td>
          <td>24.252034</td>
          <td>0.117805</td>
          <td>0.028198</td>
          <td>0.027887</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.232547</td>
          <td>0.305279</td>
          <td>26.798062</td>
          <td>0.178679</td>
          <td>26.436683</td>
          <td>0.115658</td>
          <td>26.623843</td>
          <td>0.217723</td>
          <td>25.983943</td>
          <td>0.235203</td>
          <td>24.977418</td>
          <td>0.218765</td>
          <td>0.211504</td>
          <td>0.131470</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.882460</td>
          <td>0.229512</td>
          <td>26.288085</td>
          <td>0.115271</td>
          <td>26.095942</td>
          <td>0.085808</td>
          <td>25.689710</td>
          <td>0.097671</td>
          <td>25.360966</td>
          <td>0.138842</td>
          <td>25.371330</td>
          <td>0.302036</td>
          <td>0.019921</td>
          <td>0.010344</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.587890</td>
          <td>0.403514</td>
          <td>26.559969</td>
          <td>0.145825</td>
          <td>26.364334</td>
          <td>0.108586</td>
          <td>26.352721</td>
          <td>0.173290</td>
          <td>25.456809</td>
          <td>0.150773</td>
          <td>25.930842</td>
          <td>0.466540</td>
          <td>0.119469</td>
          <td>0.110149</td>
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
          <td>26.676529</td>
          <td>0.185959</td>
          <td>26.154663</td>
          <td>0.106800</td>
          <td>25.134508</td>
          <td>0.071351</td>
          <td>24.686328</td>
          <td>0.091060</td>
          <td>24.107662</td>
          <td>0.123314</td>
          <td>0.048858</td>
          <td>0.033015</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.671754</td>
          <td>0.418342</td>
          <td>26.498012</td>
          <td>0.144971</td>
          <td>26.094571</td>
          <td>0.166041</td>
          <td>26.377954</td>
          <td>0.378847</td>
          <td>24.885032</td>
          <td>0.240394</td>
          <td>0.066095</td>
          <td>0.062192</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.125992</td>
          <td>0.668217</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.950450</td>
          <td>0.474924</td>
          <td>25.813292</td>
          <td>0.131637</td>
          <td>24.904502</td>
          <td>0.112128</td>
          <td>24.292461</td>
          <td>0.147185</td>
          <td>0.100134</td>
          <td>0.062612</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.834661</td>
          <td>1.610923</td>
          <td>27.866838</td>
          <td>0.438494</td>
          <td>26.111906</td>
          <td>0.166730</td>
          <td>25.199050</td>
          <td>0.141924</td>
          <td>25.054203</td>
          <td>0.273346</td>
          <td>0.031081</td>
          <td>0.030694</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.138851</td>
          <td>0.334555</td>
          <td>26.149330</td>
          <td>0.127068</td>
          <td>25.864558</td>
          <td>0.089640</td>
          <td>25.693129</td>
          <td>0.126350</td>
          <td>25.920785</td>
          <td>0.280985</td>
          <td>25.275075</td>
          <td>0.352068</td>
          <td>0.182746</td>
          <td>0.146457</td>
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
          <td>26.587546</td>
          <td>0.448234</td>
          <td>26.551168</td>
          <td>0.167189</td>
          <td>25.417617</td>
          <td>0.055742</td>
          <td>25.144079</td>
          <td>0.071947</td>
          <td>24.821052</td>
          <td>0.102464</td>
          <td>24.472582</td>
          <td>0.168743</td>
          <td>0.050134</td>
          <td>0.029700</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.429407</td>
          <td>0.808756</td>
          <td>26.405399</td>
          <td>0.147204</td>
          <td>25.945502</td>
          <td>0.088622</td>
          <td>25.285236</td>
          <td>0.081238</td>
          <td>24.752694</td>
          <td>0.096213</td>
          <td>24.491473</td>
          <td>0.170954</td>
          <td>0.028198</td>
          <td>0.027887</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.067353</td>
          <td>0.318174</td>
          <td>26.912108</td>
          <td>0.244272</td>
          <td>26.318763</td>
          <td>0.134382</td>
          <td>26.308516</td>
          <td>0.215298</td>
          <td>25.524422</td>
          <td>0.204314</td>
          <td>25.971910</td>
          <td>0.597846</td>
          <td>0.211504</td>
          <td>0.131470</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.950597</td>
          <td>0.271001</td>
          <td>26.006658</td>
          <td>0.104038</td>
          <td>26.209146</td>
          <td>0.111442</td>
          <td>25.695450</td>
          <td>0.116173</td>
          <td>25.688696</td>
          <td>0.214539</td>
          <td>24.936552</td>
          <td>0.247701</td>
          <td>0.019921</td>
          <td>0.010344</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.495716</td>
          <td>0.428584</td>
          <td>26.705375</td>
          <td>0.196778</td>
          <td>26.630364</td>
          <td>0.167042</td>
          <td>27.157592</td>
          <td>0.405995</td>
          <td>25.490622</td>
          <td>0.189142</td>
          <td>25.443244</td>
          <td>0.386370</td>
          <td>0.119469</td>
          <td>0.110149</td>
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
          <td>26.535052</td>
          <td>0.145558</td>
          <td>26.041028</td>
          <td>0.083664</td>
          <td>25.311554</td>
          <td>0.071706</td>
          <td>24.521438</td>
          <td>0.068095</td>
          <td>23.948141</td>
          <td>0.092488</td>
          <td>0.048858</td>
          <td>0.033015</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.840592</td>
          <td>0.987078</td>
          <td>27.636777</td>
          <td>0.369982</td>
          <td>26.663657</td>
          <td>0.148282</td>
          <td>26.065557</td>
          <td>0.143065</td>
          <td>25.874670</td>
          <td>0.225821</td>
          <td>25.235754</td>
          <td>0.284744</td>
          <td>0.066095</td>
          <td>0.062192</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.211350</td>
          <td>0.582465</td>
          <td>28.240287</td>
          <td>0.540881</td>
          <td>26.010435</td>
          <td>0.141006</td>
          <td>25.025641</td>
          <td>0.112914</td>
          <td>24.399531</td>
          <td>0.145989</td>
          <td>0.100134</td>
          <td>0.062612</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.626848</td>
          <td>2.134472</td>
          <td>27.321067</td>
          <td>0.248198</td>
          <td>26.599704</td>
          <td>0.216122</td>
          <td>25.560894</td>
          <td>0.166871</td>
          <td>24.961779</td>
          <td>0.218672</td>
          <td>0.031081</td>
          <td>0.030694</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.711633</td>
          <td>0.527135</td>
          <td>26.300413</td>
          <td>0.148290</td>
          <td>25.980047</td>
          <td>0.101886</td>
          <td>25.733445</td>
          <td>0.134426</td>
          <td>24.996441</td>
          <td>0.132573</td>
          <td>24.535142</td>
          <td>0.197542</td>
          <td>0.182746</td>
          <td>0.146457</td>
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
          <td>26.507743</td>
          <td>0.384576</td>
          <td>26.279911</td>
          <td>0.116689</td>
          <td>25.520136</td>
          <td>0.052720</td>
          <td>25.027211</td>
          <td>0.055696</td>
          <td>24.878972</td>
          <td>0.093294</td>
          <td>24.565327</td>
          <td>0.157950</td>
          <td>0.050134</td>
          <td>0.029700</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.915007</td>
          <td>0.518976</td>
          <td>26.742820</td>
          <td>0.172016</td>
          <td>26.028889</td>
          <td>0.081747</td>
          <td>25.177168</td>
          <td>0.062827</td>
          <td>24.811568</td>
          <td>0.086889</td>
          <td>23.992044</td>
          <td>0.094891</td>
          <td>0.028198</td>
          <td>0.027887</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.195002</td>
          <td>0.747952</td>
          <td>26.839076</td>
          <td>0.237443</td>
          <td>26.226174</td>
          <td>0.128544</td>
          <td>26.358403</td>
          <td>0.232393</td>
          <td>25.548727</td>
          <td>0.215809</td>
          <td>25.212858</td>
          <td>0.349293</td>
          <td>0.211504</td>
          <td>0.131470</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.172078</td>
          <td>0.291430</td>
          <td>26.401521</td>
          <td>0.127569</td>
          <td>26.120690</td>
          <td>0.087998</td>
          <td>25.914480</td>
          <td>0.119280</td>
          <td>25.493249</td>
          <td>0.156074</td>
          <td>25.736816</td>
          <td>0.403913</td>
          <td>0.019921</td>
          <td>0.010344</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.661842</td>
          <td>1.610635</td>
          <td>26.489507</td>
          <td>0.156554</td>
          <td>26.583258</td>
          <td>0.152498</td>
          <td>25.979157</td>
          <td>0.146961</td>
          <td>25.896253</td>
          <td>0.252345</td>
          <td>25.586581</td>
          <td>0.411656</td>
          <td>0.119469</td>
          <td>0.110149</td>
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
