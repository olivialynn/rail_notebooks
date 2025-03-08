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

    <pzflow.flow.Flow at 0x7f16a734c340>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.777169</td>
          <td>0.175542</td>
          <td>25.926321</td>
          <td>0.073877</td>
          <td>25.193627</td>
          <td>0.063039</td>
          <td>24.784064</td>
          <td>0.083915</td>
          <td>23.970819</td>
          <td>0.092121</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.075012</td>
          <td>0.579058</td>
          <td>27.416873</td>
          <td>0.298134</td>
          <td>26.629956</td>
          <td>0.136758</td>
          <td>26.258664</td>
          <td>0.159941</td>
          <td>26.295660</td>
          <td>0.303271</td>
          <td>25.051657</td>
          <td>0.232679</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.938459</td>
          <td>0.524732</td>
          <td>30.422489</td>
          <td>1.950616</td>
          <td>27.995934</td>
          <td>0.419485</td>
          <td>25.923564</td>
          <td>0.119800</td>
          <td>25.020541</td>
          <td>0.103285</td>
          <td>24.229365</td>
          <td>0.115504</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.836931</td>
          <td>0.414664</td>
          <td>27.465468</td>
          <td>0.275995</td>
          <td>25.950633</td>
          <td>0.122651</td>
          <td>25.398652</td>
          <td>0.143423</td>
          <td>25.138031</td>
          <td>0.249864</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.300279</td>
          <td>0.322239</td>
          <td>26.111806</td>
          <td>0.098833</td>
          <td>25.986427</td>
          <td>0.077907</td>
          <td>25.742648</td>
          <td>0.102308</td>
          <td>25.447485</td>
          <td>0.149572</td>
          <td>24.960016</td>
          <td>0.215614</td>
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
          <td>27.236471</td>
          <td>0.648666</td>
          <td>26.380358</td>
          <td>0.124887</td>
          <td>25.371770</td>
          <td>0.045175</td>
          <td>25.069428</td>
          <td>0.056462</td>
          <td>24.787769</td>
          <td>0.084189</td>
          <td>24.813451</td>
          <td>0.190670</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.447506</td>
          <td>0.748609</td>
          <td>26.619038</td>
          <td>0.153405</td>
          <td>26.002992</td>
          <td>0.079055</td>
          <td>25.213130</td>
          <td>0.064139</td>
          <td>24.770873</td>
          <td>0.082945</td>
          <td>24.232277</td>
          <td>0.115797</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.917068</td>
          <td>1.712313</td>
          <td>26.721861</td>
          <td>0.167482</td>
          <td>26.323513</td>
          <td>0.104781</td>
          <td>26.268157</td>
          <td>0.161243</td>
          <td>25.717824</td>
          <td>0.188291</td>
          <td>25.413095</td>
          <td>0.312318</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.475893</td>
          <td>0.370022</td>
          <td>26.179081</td>
          <td>0.104822</td>
          <td>26.013827</td>
          <td>0.079815</td>
          <td>26.041250</td>
          <td>0.132670</td>
          <td>25.348580</td>
          <td>0.137366</td>
          <td>24.958938</td>
          <td>0.215420</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.914895</td>
          <td>0.515772</td>
          <td>26.896904</td>
          <td>0.194233</td>
          <td>26.308902</td>
          <td>0.103450</td>
          <td>26.425675</td>
          <td>0.184346</td>
          <td>25.848112</td>
          <td>0.210076</td>
          <td>25.515988</td>
          <td>0.338947</td>
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
          <td>28.356482</td>
          <td>1.388257</td>
          <td>26.576984</td>
          <td>0.170050</td>
          <td>26.090536</td>
          <td>0.100387</td>
          <td>25.191860</td>
          <td>0.074605</td>
          <td>24.608225</td>
          <td>0.084517</td>
          <td>24.037552</td>
          <td>0.115336</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.628851</td>
          <td>0.460712</td>
          <td>27.739187</td>
          <td>0.435714</td>
          <td>26.589107</td>
          <td>0.154690</td>
          <td>26.128835</td>
          <td>0.168642</td>
          <td>26.260342</td>
          <td>0.341342</td>
          <td>25.737918</td>
          <td>0.465554</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.431037</td>
          <td>0.349480</td>
          <td>28.850936</td>
          <td>0.883465</td>
          <td>25.932131</td>
          <td>0.145728</td>
          <td>25.028083</td>
          <td>0.124752</td>
          <td>24.157325</td>
          <td>0.130899</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.023752</td>
          <td>0.565681</td>
          <td>26.949803</td>
          <td>0.223710</td>
          <td>26.494334</td>
          <td>0.244712</td>
          <td>25.857355</td>
          <td>0.262461</td>
          <td>24.903696</td>
          <td>0.256929</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.900275</td>
          <td>0.260024</td>
          <td>26.288182</td>
          <td>0.132801</td>
          <td>25.882090</td>
          <td>0.083616</td>
          <td>25.743756</td>
          <td>0.121094</td>
          <td>25.258203</td>
          <td>0.148908</td>
          <td>24.740468</td>
          <td>0.210420</td>
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
          <td>26.862426</td>
          <td>0.554336</td>
          <td>26.534963</td>
          <td>0.167122</td>
          <td>25.463411</td>
          <td>0.058957</td>
          <td>25.131453</td>
          <td>0.072292</td>
          <td>24.838689</td>
          <td>0.105652</td>
          <td>24.965536</td>
          <td>0.258618</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.355087</td>
          <td>1.389852</td>
          <td>26.725844</td>
          <td>0.193571</td>
          <td>26.081795</td>
          <td>0.100037</td>
          <td>25.271515</td>
          <td>0.080389</td>
          <td>24.768490</td>
          <td>0.097704</td>
          <td>24.462788</td>
          <td>0.167085</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.634981</td>
          <td>0.926521</td>
          <td>26.679273</td>
          <td>0.187516</td>
          <td>26.288718</td>
          <td>0.120858</td>
          <td>26.045013</td>
          <td>0.159001</td>
          <td>26.059625</td>
          <td>0.294161</td>
          <td>25.820460</td>
          <td>0.500443</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.326993</td>
          <td>0.373507</td>
          <td>26.294577</td>
          <td>0.137278</td>
          <td>26.186180</td>
          <td>0.112602</td>
          <td>25.842854</td>
          <td>0.136194</td>
          <td>25.790060</td>
          <td>0.240202</td>
          <td>24.962988</td>
          <td>0.260661</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.007254</td>
          <td>1.153807</td>
          <td>26.654181</td>
          <td>0.183146</td>
          <td>26.742098</td>
          <td>0.177922</td>
          <td>26.076170</td>
          <td>0.162839</td>
          <td>25.903005</td>
          <td>0.258356</td>
          <td>25.275944</td>
          <td>0.328787</td>
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
          <td>26.682264</td>
          <td>0.433684</td>
          <td>26.718688</td>
          <td>0.167049</td>
          <td>25.929170</td>
          <td>0.074073</td>
          <td>25.059830</td>
          <td>0.055990</td>
          <td>24.606617</td>
          <td>0.071754</td>
          <td>23.969115</td>
          <td>0.091995</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.729807</td>
          <td>0.898720</td>
          <td>27.101498</td>
          <td>0.230604</td>
          <td>26.720206</td>
          <td>0.147948</td>
          <td>26.223671</td>
          <td>0.155374</td>
          <td>26.103959</td>
          <td>0.259843</td>
          <td>25.447401</td>
          <td>0.321276</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.119928</td>
          <td>1.931235</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.899641</td>
          <td>0.418456</td>
          <td>25.705761</td>
          <td>0.107868</td>
          <td>25.165597</td>
          <td>0.127093</td>
          <td>24.389595</td>
          <td>0.144248</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.301601</td>
          <td>0.770371</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.656239</td>
          <td>0.393382</td>
          <td>25.988821</td>
          <td>0.159479</td>
          <td>25.502980</td>
          <td>0.194932</td>
          <td>25.656287</td>
          <td>0.462881</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.719747</td>
          <td>0.446495</td>
          <td>25.939055</td>
          <td>0.085033</td>
          <td>26.031936</td>
          <td>0.081216</td>
          <td>25.591471</td>
          <td>0.089730</td>
          <td>25.610135</td>
          <td>0.172107</td>
          <td>24.867764</td>
          <td>0.199871</td>
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
          <td>26.175897</td>
          <td>0.306985</td>
          <td>26.219745</td>
          <td>0.116305</td>
          <td>25.556964</td>
          <td>0.057667</td>
          <td>25.183810</td>
          <td>0.067916</td>
          <td>24.926305</td>
          <td>0.102865</td>
          <td>24.781423</td>
          <td>0.200679</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.758475</td>
          <td>0.921966</td>
          <td>26.849220</td>
          <td>0.189165</td>
          <td>26.098968</td>
          <td>0.087470</td>
          <td>25.247671</td>
          <td>0.067298</td>
          <td>24.827494</td>
          <td>0.088637</td>
          <td>24.299518</td>
          <td>0.124861</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.294118</td>
          <td>0.693252</td>
          <td>26.686562</td>
          <td>0.169395</td>
          <td>26.526383</td>
          <td>0.131215</td>
          <td>26.357298</td>
          <td>0.182782</td>
          <td>26.104295</td>
          <td>0.271738</td>
          <td>25.631516</td>
          <td>0.388145</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.418364</td>
          <td>0.380530</td>
          <td>26.220517</td>
          <td>0.120120</td>
          <td>26.186309</td>
          <td>0.104177</td>
          <td>25.979675</td>
          <td>0.141514</td>
          <td>25.669170</td>
          <td>0.201693</td>
          <td>26.359254</td>
          <td>0.697869</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.848499</td>
          <td>0.192637</td>
          <td>26.467968</td>
          <td>0.123513</td>
          <td>26.262716</td>
          <td>0.166981</td>
          <td>25.700864</td>
          <td>0.192672</td>
          <td>25.147740</td>
          <td>0.261527</td>
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
