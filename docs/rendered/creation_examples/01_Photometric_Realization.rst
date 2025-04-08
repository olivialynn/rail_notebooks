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

    <pzflow.flow.Flow at 0x7f35c87f0220>



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
          <td>27.700859</td>
          <td>0.882113</td>
          <td>26.670359</td>
          <td>0.160287</td>
          <td>26.042646</td>
          <td>0.081870</td>
          <td>25.114524</td>
          <td>0.058767</td>
          <td>24.607608</td>
          <td>0.071807</td>
          <td>24.094314</td>
          <td>0.102658</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.475783</td>
          <td>0.762778</td>
          <td>27.318803</td>
          <td>0.275409</td>
          <td>26.725386</td>
          <td>0.148470</td>
          <td>26.080261</td>
          <td>0.137217</td>
          <td>26.040693</td>
          <td>0.246478</td>
          <td>24.931988</td>
          <td>0.210626</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.894629</td>
          <td>0.994072</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.020751</td>
          <td>0.427496</td>
          <td>26.094743</td>
          <td>0.138942</td>
          <td>24.943222</td>
          <td>0.096520</td>
          <td>24.455747</td>
          <td>0.140532</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.825461</td>
          <td>0.482851</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.158955</td>
          <td>0.214394</td>
          <td>25.894468</td>
          <td>0.116807</td>
          <td>25.628548</td>
          <td>0.174582</td>
          <td>25.310680</td>
          <td>0.287628</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.888623</td>
          <td>0.230685</td>
          <td>26.112966</td>
          <td>0.098933</td>
          <td>25.893089</td>
          <td>0.071737</td>
          <td>25.624114</td>
          <td>0.092205</td>
          <td>25.604002</td>
          <td>0.170977</td>
          <td>25.350732</td>
          <td>0.297074</td>
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
          <td>26.568592</td>
          <td>0.397570</td>
          <td>26.341006</td>
          <td>0.120696</td>
          <td>25.391960</td>
          <td>0.045992</td>
          <td>25.122855</td>
          <td>0.059203</td>
          <td>25.022548</td>
          <td>0.103467</td>
          <td>24.461404</td>
          <td>0.141218</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.615627</td>
          <td>3.217085</td>
          <td>26.508798</td>
          <td>0.139545</td>
          <td>26.220362</td>
          <td>0.095727</td>
          <td>25.202675</td>
          <td>0.063547</td>
          <td>24.684929</td>
          <td>0.076887</td>
          <td>24.075707</td>
          <td>0.100999</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.141309</td>
          <td>0.606927</td>
          <td>26.529995</td>
          <td>0.142115</td>
          <td>26.587122</td>
          <td>0.131789</td>
          <td>26.166867</td>
          <td>0.147841</td>
          <td>26.211929</td>
          <td>0.283469</td>
          <td>25.414320</td>
          <td>0.312625</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.136269</td>
          <td>0.282510</td>
          <td>26.126427</td>
          <td>0.100106</td>
          <td>26.163457</td>
          <td>0.091060</td>
          <td>25.997208</td>
          <td>0.127708</td>
          <td>25.981272</td>
          <td>0.234683</td>
          <td>25.521767</td>
          <td>0.340499</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.527310</td>
          <td>0.385098</td>
          <td>26.788354</td>
          <td>0.177215</td>
          <td>26.649150</td>
          <td>0.139042</td>
          <td>26.286667</td>
          <td>0.163812</td>
          <td>25.771865</td>
          <td>0.197063</td>
          <td>25.520560</td>
          <td>0.340175</td>
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
          <td>26.652762</td>
          <td>0.468966</td>
          <td>26.922902</td>
          <td>0.227405</td>
          <td>26.050023</td>
          <td>0.096886</td>
          <td>25.198164</td>
          <td>0.075022</td>
          <td>24.632635</td>
          <td>0.086353</td>
          <td>23.996372</td>
          <td>0.111272</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.230332</td>
          <td>1.298877</td>
          <td>26.875191</td>
          <td>0.218603</td>
          <td>27.062919</td>
          <td>0.230676</td>
          <td>26.212945</td>
          <td>0.181125</td>
          <td>26.682135</td>
          <td>0.472088</td>
          <td>25.407921</td>
          <td>0.361551</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.708812</td>
          <td>1.668527</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.374621</td>
          <td>1.205316</td>
          <td>26.065984</td>
          <td>0.163430</td>
          <td>24.907363</td>
          <td>0.112323</td>
          <td>24.184480</td>
          <td>0.134008</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.585881</td>
          <td>0.829672</td>
          <td>27.187518</td>
          <td>0.272031</td>
          <td>26.582501</td>
          <td>0.263066</td>
          <td>25.579552</td>
          <td>0.208577</td>
          <td>25.477882</td>
          <td>0.405548</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.165414</td>
          <td>0.321991</td>
          <td>26.262787</td>
          <td>0.129919</td>
          <td>25.803271</td>
          <td>0.078001</td>
          <td>25.535237</td>
          <td>0.100957</td>
          <td>25.539985</td>
          <td>0.189279</td>
          <td>25.700260</td>
          <td>0.452624</td>
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
          <td>26.175994</td>
          <td>0.122756</td>
          <td>25.529466</td>
          <td>0.062513</td>
          <td>25.091821</td>
          <td>0.069802</td>
          <td>24.720056</td>
          <td>0.095228</td>
          <td>24.696143</td>
          <td>0.206902</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.318857</td>
          <td>0.752800</td>
          <td>26.773576</td>
          <td>0.201491</td>
          <td>26.068907</td>
          <td>0.098914</td>
          <td>25.318277</td>
          <td>0.083772</td>
          <td>24.788426</td>
          <td>0.099426</td>
          <td>24.329681</td>
          <td>0.149109</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.011396</td>
          <td>0.287086</td>
          <td>26.426400</td>
          <td>0.151227</td>
          <td>26.317927</td>
          <td>0.123961</td>
          <td>26.034365</td>
          <td>0.157560</td>
          <td>25.693523</td>
          <td>0.217861</td>
          <td>26.002975</td>
          <td>0.571435</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.092443</td>
          <td>0.310428</td>
          <td>25.947371</td>
          <td>0.101547</td>
          <td>25.993661</td>
          <td>0.095154</td>
          <td>25.853468</td>
          <td>0.137447</td>
          <td>25.861207</td>
          <td>0.254680</td>
          <td>24.970882</td>
          <td>0.262349</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.639952</td>
          <td>0.467526</td>
          <td>26.825478</td>
          <td>0.211501</td>
          <td>26.453295</td>
          <td>0.138981</td>
          <td>26.097488</td>
          <td>0.165826</td>
          <td>25.628139</td>
          <td>0.205742</td>
          <td>25.159599</td>
          <td>0.299597</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.831515</td>
          <td>0.183830</td>
          <td>26.038723</td>
          <td>0.081598</td>
          <td>25.247547</td>
          <td>0.066135</td>
          <td>24.805878</td>
          <td>0.085554</td>
          <td>23.849371</td>
          <td>0.082793</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.158691</td>
          <td>0.287849</td>
          <td>28.371293</td>
          <td>0.614560</td>
          <td>26.725162</td>
          <td>0.148579</td>
          <td>26.195736</td>
          <td>0.151698</td>
          <td>25.743780</td>
          <td>0.192633</td>
          <td>25.272116</td>
          <td>0.279036</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.557556</td>
          <td>0.738100</td>
          <td>29.769051</td>
          <td>1.399647</td>
          <td>26.013018</td>
          <td>0.140827</td>
          <td>24.951711</td>
          <td>0.105504</td>
          <td>24.429936</td>
          <td>0.149337</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.569652</td>
          <td>1.588667</td>
          <td>28.366540</td>
          <td>0.716264</td>
          <td>27.190871</td>
          <td>0.271874</td>
          <td>26.180754</td>
          <td>0.187723</td>
          <td>25.279986</td>
          <td>0.161351</td>
          <td>25.625684</td>
          <td>0.452362</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.539654</td>
          <td>0.796041</td>
          <td>26.069995</td>
          <td>0.095396</td>
          <td>25.971311</td>
          <td>0.076984</td>
          <td>25.779254</td>
          <td>0.105794</td>
          <td>25.657066</td>
          <td>0.179103</td>
          <td>25.169051</td>
          <td>0.256659</td>
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
          <td>27.211946</td>
          <td>0.666368</td>
          <td>26.466748</td>
          <td>0.143999</td>
          <td>25.520341</td>
          <td>0.055824</td>
          <td>25.130111</td>
          <td>0.064761</td>
          <td>24.806589</td>
          <td>0.092616</td>
          <td>24.981694</td>
          <td>0.237118</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.362050</td>
          <td>0.713168</td>
          <td>26.742107</td>
          <td>0.172770</td>
          <td>25.958571</td>
          <td>0.077285</td>
          <td>25.173213</td>
          <td>0.063000</td>
          <td>24.856887</td>
          <td>0.090958</td>
          <td>24.451948</td>
          <td>0.142443</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.867902</td>
          <td>0.234247</td>
          <td>26.718383</td>
          <td>0.174037</td>
          <td>26.470862</td>
          <td>0.125054</td>
          <td>26.261819</td>
          <td>0.168552</td>
          <td>26.216277</td>
          <td>0.297523</td>
          <td>25.360570</td>
          <td>0.313617</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.125851</td>
          <td>0.302090</td>
          <td>26.336283</td>
          <td>0.132784</td>
          <td>26.077767</td>
          <td>0.094727</td>
          <td>25.874910</td>
          <td>0.129273</td>
          <td>25.411101</td>
          <td>0.162105</td>
          <td>24.822285</td>
          <td>0.215130</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.176795</td>
          <td>0.299291</td>
          <td>26.755073</td>
          <td>0.178017</td>
          <td>26.576957</td>
          <td>0.135735</td>
          <td>26.104500</td>
          <td>0.145830</td>
          <td>25.837610</td>
          <td>0.216077</td>
          <td>25.742353</td>
          <td>0.418865</td>
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
