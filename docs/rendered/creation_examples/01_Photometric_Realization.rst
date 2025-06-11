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

    <pzflow.flow.Flow at 0x7fe7ac5fb220>



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
    0      23.994413  0.223189  0.156196  
    1      25.391064  0.049014  0.047987  
    2      24.304707  0.108236  0.057548  
    3      25.291103  0.031786  0.030086  
    4      25.096743  0.089654  0.074142  
    ...          ...       ...       ...  
    99995  24.737946  0.079824  0.066999  
    99996  24.224169  0.050991  0.025949  
    99997  25.613836  0.063285  0.037457  
    99998  25.274899  0.058053  0.053152  
    99999  25.699642  0.165158  0.138483  
    
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
          <td>27.731026</td>
          <td>0.898987</td>
          <td>26.924676</td>
          <td>0.198821</td>
          <td>25.998085</td>
          <td>0.078713</td>
          <td>25.097361</td>
          <td>0.057879</td>
          <td>24.564537</td>
          <td>0.069122</td>
          <td>23.882704</td>
          <td>0.085249</td>
          <td>0.223189</td>
          <td>0.156196</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.300011</td>
          <td>0.322170</td>
          <td>28.023631</td>
          <td>0.477434</td>
          <td>26.531858</td>
          <td>0.125630</td>
          <td>26.070800</td>
          <td>0.136101</td>
          <td>25.697781</td>
          <td>0.185130</td>
          <td>25.757363</td>
          <td>0.409068</td>
          <td>0.049014</td>
          <td>0.047987</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.688869</td>
          <td>0.762802</td>
          <td>27.816816</td>
          <td>0.365262</td>
          <td>25.761922</td>
          <td>0.104048</td>
          <td>25.138549</td>
          <td>0.114494</td>
          <td>24.125115</td>
          <td>0.105461</td>
          <td>0.108236</td>
          <td>0.057548</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.866207</td>
          <td>0.379594</td>
          <td>26.086455</td>
          <td>0.137952</td>
          <td>25.629890</td>
          <td>0.174781</td>
          <td>24.791317</td>
          <td>0.187141</td>
          <td>0.031786</td>
          <td>0.030086</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.916150</td>
          <td>0.235992</td>
          <td>25.958092</td>
          <td>0.086362</td>
          <td>25.942198</td>
          <td>0.074922</td>
          <td>25.671385</td>
          <td>0.096113</td>
          <td>26.092653</td>
          <td>0.257222</td>
          <td>25.270900</td>
          <td>0.278507</td>
          <td>0.089654</td>
          <td>0.074142</td>
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
          <td>27.631038</td>
          <td>0.843852</td>
          <td>26.215463</td>
          <td>0.108204</td>
          <td>25.453757</td>
          <td>0.048586</td>
          <td>25.129162</td>
          <td>0.059536</td>
          <td>24.976230</td>
          <td>0.099355</td>
          <td>24.855619</td>
          <td>0.197562</td>
          <td>0.079824</td>
          <td>0.066999</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.673880</td>
          <td>2.350190</td>
          <td>26.657464</td>
          <td>0.158531</td>
          <td>25.963665</td>
          <td>0.076356</td>
          <td>25.164338</td>
          <td>0.061423</td>
          <td>24.968877</td>
          <td>0.098716</td>
          <td>24.311585</td>
          <td>0.124060</td>
          <td>0.050991</td>
          <td>0.025949</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.497500</td>
          <td>0.138193</td>
          <td>26.392560</td>
          <td>0.111294</td>
          <td>26.121346</td>
          <td>0.142164</td>
          <td>26.415797</td>
          <td>0.333773</td>
          <td>26.573420</td>
          <td>0.736147</td>
          <td>0.063285</td>
          <td>0.037457</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.676178</td>
          <td>0.431653</td>
          <td>26.143702</td>
          <td>0.101630</td>
          <td>26.100353</td>
          <td>0.086142</td>
          <td>25.866629</td>
          <td>0.114009</td>
          <td>25.421126</td>
          <td>0.146223</td>
          <td>25.506897</td>
          <td>0.336519</td>
          <td>0.058053</td>
          <td>0.053152</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.839244</td>
          <td>0.487813</td>
          <td>26.503532</td>
          <td>0.138914</td>
          <td>26.396492</td>
          <td>0.111677</td>
          <td>26.122747</td>
          <td>0.142336</td>
          <td>25.745836</td>
          <td>0.192792</td>
          <td>25.882733</td>
          <td>0.449984</td>
          <td>0.165158</td>
          <td>0.138483</td>
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
          <td>26.203272</td>
          <td>0.358549</td>
          <td>27.078148</td>
          <td>0.283821</td>
          <td>25.909612</td>
          <td>0.095707</td>
          <td>25.269959</td>
          <td>0.089660</td>
          <td>24.940849</td>
          <td>0.126240</td>
          <td>23.911623</td>
          <td>0.115698</td>
          <td>0.223189</td>
          <td>0.156196</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.742558</td>
          <td>0.439455</td>
          <td>26.729819</td>
          <td>0.175719</td>
          <td>26.050033</td>
          <td>0.158905</td>
          <td>26.887150</td>
          <td>0.552324</td>
          <td>25.565368</td>
          <td>0.411311</td>
          <td>0.049014</td>
          <td>0.047987</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.169605</td>
          <td>0.689151</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.559182</td>
          <td>0.352392</td>
          <td>26.202232</td>
          <td>0.183929</td>
          <td>24.768624</td>
          <td>0.099740</td>
          <td>24.068660</td>
          <td>0.121516</td>
          <td>0.108236</td>
          <td>0.057548</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.167559</td>
          <td>0.597947</td>
          <td>26.959918</td>
          <td>0.212336</td>
          <td>26.008620</td>
          <td>0.152645</td>
          <td>25.691491</td>
          <td>0.215530</td>
          <td>24.919688</td>
          <td>0.244851</td>
          <td>0.031786</td>
          <td>0.030086</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.493240</td>
          <td>0.421963</td>
          <td>26.253548</td>
          <td>0.131453</td>
          <td>25.988613</td>
          <td>0.093891</td>
          <td>25.736943</td>
          <td>0.123139</td>
          <td>25.708449</td>
          <td>0.222619</td>
          <td>25.481337</td>
          <td>0.390633</td>
          <td>0.089654</td>
          <td>0.074142</td>
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
          <td>27.182070</td>
          <td>0.692285</td>
          <td>26.430397</td>
          <td>0.152462</td>
          <td>25.404785</td>
          <td>0.055808</td>
          <td>25.228549</td>
          <td>0.078530</td>
          <td>25.156692</td>
          <td>0.138853</td>
          <td>24.641177</td>
          <td>0.197020</td>
          <td>0.079824</td>
          <td>0.066999</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>31.243289</td>
          <td>3.954079</td>
          <td>26.903171</td>
          <td>0.224763</td>
          <td>25.911018</td>
          <td>0.086221</td>
          <td>25.513467</td>
          <td>0.099582</td>
          <td>24.898761</td>
          <td>0.109640</td>
          <td>23.960959</td>
          <td>0.108495</td>
          <td>0.050991</td>
          <td>0.025949</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.041891</td>
          <td>0.625059</td>
          <td>26.824962</td>
          <td>0.211236</td>
          <td>26.565314</td>
          <td>0.152886</td>
          <td>26.270128</td>
          <td>0.191776</td>
          <td>25.951176</td>
          <td>0.268486</td>
          <td>25.405294</td>
          <td>0.363749</td>
          <td>0.063285</td>
          <td>0.037457</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.013770</td>
          <td>0.287156</td>
          <td>26.087177</td>
          <td>0.112569</td>
          <td>26.057007</td>
          <td>0.098500</td>
          <td>25.843277</td>
          <td>0.133365</td>
          <td>25.911423</td>
          <td>0.260233</td>
          <td>25.645929</td>
          <td>0.438316</td>
          <td>0.058053</td>
          <td>0.053152</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.625599</td>
          <td>0.482118</td>
          <td>26.312404</td>
          <td>0.144687</td>
          <td>26.267299</td>
          <td>0.125917</td>
          <td>26.112147</td>
          <td>0.178820</td>
          <td>25.423925</td>
          <td>0.184030</td>
          <td>25.276595</td>
          <td>0.348652</td>
          <td>0.165158</td>
          <td>0.138483</td>
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
          <td>28.572590</td>
          <td>1.670972</td>
          <td>26.243535</td>
          <td>0.149612</td>
          <td>25.970610</td>
          <td>0.107637</td>
          <td>25.123941</td>
          <td>0.084232</td>
          <td>24.648034</td>
          <td>0.104276</td>
          <td>24.103214</td>
          <td>0.145651</td>
          <td>0.223189</td>
          <td>0.156196</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.278919</td>
          <td>0.273352</td>
          <td>26.533192</td>
          <td>0.129675</td>
          <td>26.529921</td>
          <td>0.207603</td>
          <td>25.592150</td>
          <td>0.174403</td>
          <td>25.457094</td>
          <td>0.333008</td>
          <td>0.049014</td>
          <td>0.047987</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.158748</td>
          <td>1.081069</td>
          <td>28.661286</td>
          <td>0.728675</td>
          <td>25.969782</td>
          <td>0.136843</td>
          <td>24.936498</td>
          <td>0.104983</td>
          <td>24.581903</td>
          <td>0.171482</td>
          <td>0.108236</td>
          <td>0.057548</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.150968</td>
          <td>0.529315</td>
          <td>27.385760</td>
          <td>0.261732</td>
          <td>26.410374</td>
          <td>0.184353</td>
          <td>25.300445</td>
          <td>0.133446</td>
          <td>25.458834</td>
          <td>0.327859</td>
          <td>0.031786</td>
          <td>0.030086</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.040657</td>
          <td>0.276247</td>
          <td>26.134496</td>
          <td>0.108479</td>
          <td>25.857164</td>
          <td>0.075628</td>
          <td>25.696666</td>
          <td>0.107254</td>
          <td>25.741490</td>
          <td>0.208111</td>
          <td>24.591440</td>
          <td>0.171802</td>
          <td>0.089654</td>
          <td>0.074142</td>
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
          <td>27.131581</td>
          <td>0.626583</td>
          <td>26.304832</td>
          <td>0.124109</td>
          <td>25.468561</td>
          <td>0.052768</td>
          <td>25.098157</td>
          <td>0.062280</td>
          <td>24.800913</td>
          <td>0.091220</td>
          <td>24.862343</td>
          <td>0.212604</td>
          <td>0.079824</td>
          <td>0.066999</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.923364</td>
          <td>0.525519</td>
          <td>26.774453</td>
          <td>0.178343</td>
          <td>26.104225</td>
          <td>0.088328</td>
          <td>25.200530</td>
          <td>0.064895</td>
          <td>24.847190</td>
          <td>0.090649</td>
          <td>24.301950</td>
          <td>0.125782</td>
          <td>0.050991</td>
          <td>0.025949</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.675863</td>
          <td>0.440813</td>
          <td>26.980861</td>
          <td>0.214566</td>
          <td>26.395415</td>
          <td>0.115531</td>
          <td>26.290969</td>
          <td>0.170390</td>
          <td>25.951265</td>
          <td>0.236617</td>
          <td>27.375943</td>
          <td>1.231923</td>
          <td>0.063285</td>
          <td>0.037457</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.220006</td>
          <td>0.310121</td>
          <td>26.091200</td>
          <td>0.100533</td>
          <td>26.049207</td>
          <td>0.085750</td>
          <td>25.835902</td>
          <td>0.115753</td>
          <td>25.744665</td>
          <td>0.200178</td>
          <td>25.228125</td>
          <td>0.279608</td>
          <td>0.058053</td>
          <td>0.053152</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.962985</td>
          <td>0.617923</td>
          <td>26.462863</td>
          <td>0.165544</td>
          <td>26.708781</td>
          <td>0.184982</td>
          <td>26.663474</td>
          <td>0.284346</td>
          <td>25.984645</td>
          <td>0.294417</td>
          <td>26.253381</td>
          <td>0.717943</td>
          <td>0.165158</td>
          <td>0.138483</td>
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
