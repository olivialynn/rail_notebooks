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

    <pzflow.flow.Flow at 0x7fc025bee8c0>



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
          <td>26.925113</td>
          <td>0.519643</td>
          <td>26.510696</td>
          <td>0.139774</td>
          <td>26.051195</td>
          <td>0.082490</td>
          <td>25.143922</td>
          <td>0.060320</td>
          <td>24.511885</td>
          <td>0.065973</td>
          <td>23.970277</td>
          <td>0.092077</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.218480</td>
          <td>1.199536</td>
          <td>27.451698</td>
          <td>0.306589</td>
          <td>27.064932</td>
          <td>0.198156</td>
          <td>26.567581</td>
          <td>0.207728</td>
          <td>26.183556</td>
          <td>0.277020</td>
          <td>26.810001</td>
          <td>0.859045</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.513628</td>
          <td>0.782030</td>
          <td>28.443115</td>
          <td>0.645756</td>
          <td>27.762634</td>
          <td>0.350067</td>
          <td>26.165074</td>
          <td>0.147614</td>
          <td>24.947193</td>
          <td>0.096857</td>
          <td>24.195946</td>
          <td>0.112189</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.715082</td>
          <td>0.776106</td>
          <td>27.414609</td>
          <td>0.264795</td>
          <td>26.254652</td>
          <td>0.159393</td>
          <td>25.426295</td>
          <td>0.146874</td>
          <td>24.902626</td>
          <td>0.205513</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.611679</td>
          <td>0.410943</td>
          <td>26.016784</td>
          <td>0.090933</td>
          <td>25.755574</td>
          <td>0.063511</td>
          <td>25.723965</td>
          <td>0.100648</td>
          <td>25.593679</td>
          <td>0.169482</td>
          <td>24.928012</td>
          <td>0.209927</td>
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
          <td>28.633713</td>
          <td>1.493902</td>
          <td>26.335399</td>
          <td>0.120110</td>
          <td>25.500007</td>
          <td>0.050622</td>
          <td>25.065749</td>
          <td>0.056277</td>
          <td>24.897142</td>
          <td>0.092694</td>
          <td>24.725160</td>
          <td>0.176949</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.524530</td>
          <td>0.787637</td>
          <td>26.726595</td>
          <td>0.168158</td>
          <td>26.021940</td>
          <td>0.080388</td>
          <td>25.206662</td>
          <td>0.063772</td>
          <td>24.769807</td>
          <td>0.082867</td>
          <td>24.188294</td>
          <td>0.111443</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.128729</td>
          <td>0.601563</td>
          <td>26.762781</td>
          <td>0.173412</td>
          <td>26.305952</td>
          <td>0.103184</td>
          <td>26.271204</td>
          <td>0.161664</td>
          <td>25.814591</td>
          <td>0.204262</td>
          <td>25.269296</td>
          <td>0.278145</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.696878</td>
          <td>0.438477</td>
          <td>26.224010</td>
          <td>0.109014</td>
          <td>26.036252</td>
          <td>0.081410</td>
          <td>25.926200</td>
          <td>0.120075</td>
          <td>25.449495</td>
          <td>0.149830</td>
          <td>25.111956</td>
          <td>0.244560</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.749519</td>
          <td>0.171469</td>
          <td>26.526827</td>
          <td>0.125083</td>
          <td>26.347968</td>
          <td>0.172591</td>
          <td>25.670279</td>
          <td>0.180871</td>
          <td>25.517018</td>
          <td>0.339224</td>
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
          <td>27.787676</td>
          <td>1.010010</td>
          <td>26.717118</td>
          <td>0.191462</td>
          <td>25.857909</td>
          <td>0.081826</td>
          <td>25.136463</td>
          <td>0.071039</td>
          <td>24.704013</td>
          <td>0.091946</td>
          <td>23.836202</td>
          <td>0.096729</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.006075</td>
          <td>0.243629</td>
          <td>26.453156</td>
          <td>0.137629</td>
          <td>26.251402</td>
          <td>0.187112</td>
          <td>25.641285</td>
          <td>0.206080</td>
          <td>25.284925</td>
          <td>0.328131</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.772623</td>
          <td>2.413612</td>
          <td>27.440057</td>
          <td>0.319983</td>
          <td>25.984517</td>
          <td>0.152431</td>
          <td>24.842124</td>
          <td>0.106107</td>
          <td>24.252593</td>
          <td>0.142115</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.819359</td>
          <td>0.553396</td>
          <td>28.145145</td>
          <td>0.616594</td>
          <td>27.625060</td>
          <td>0.385217</td>
          <td>26.352805</td>
          <td>0.217635</td>
          <td>25.419900</td>
          <td>0.182357</td>
          <td>25.247304</td>
          <td>0.338832</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.601242</td>
          <td>0.451293</td>
          <td>26.135507</td>
          <td>0.116347</td>
          <td>25.860698</td>
          <td>0.082054</td>
          <td>25.714154</td>
          <td>0.118018</td>
          <td>25.170504</td>
          <td>0.138084</td>
          <td>24.794109</td>
          <td>0.220050</td>
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
          <td>30.813324</td>
          <td>3.550801</td>
          <td>26.427988</td>
          <td>0.152534</td>
          <td>25.345520</td>
          <td>0.053104</td>
          <td>25.267495</td>
          <td>0.081518</td>
          <td>24.660522</td>
          <td>0.090378</td>
          <td>24.439869</td>
          <td>0.166624</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.983153</td>
          <td>1.134805</td>
          <td>27.148378</td>
          <td>0.274624</td>
          <td>26.089267</td>
          <td>0.100694</td>
          <td>25.365547</td>
          <td>0.087332</td>
          <td>24.681810</td>
          <td>0.090547</td>
          <td>24.124708</td>
          <td>0.124938</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.270086</td>
          <td>0.352751</td>
          <td>26.713844</td>
          <td>0.193059</td>
          <td>26.704290</td>
          <td>0.172761</td>
          <td>26.012289</td>
          <td>0.154611</td>
          <td>25.913639</td>
          <td>0.261281</td>
          <td>25.635590</td>
          <td>0.435810</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.913293</td>
          <td>0.268668</td>
          <td>26.201748</td>
          <td>0.126701</td>
          <td>26.113045</td>
          <td>0.105640</td>
          <td>25.805647</td>
          <td>0.131886</td>
          <td>25.463610</td>
          <td>0.182832</td>
          <td>25.330358</td>
          <td>0.350075</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.479994</td>
          <td>0.414277</td>
          <td>26.608619</td>
          <td>0.176216</td>
          <td>26.434955</td>
          <td>0.136800</td>
          <td>26.313001</td>
          <td>0.199010</td>
          <td>25.967941</td>
          <td>0.272419</td>
          <td>inf</td>
          <td>inf</td>
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
          <td>27.826469</td>
          <td>0.953789</td>
          <td>26.582668</td>
          <td>0.148712</td>
          <td>26.083835</td>
          <td>0.084909</td>
          <td>25.240479</td>
          <td>0.065722</td>
          <td>24.626480</td>
          <td>0.073025</td>
          <td>23.866718</td>
          <td>0.084068</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.224199</td>
          <td>1.203861</td>
          <td>27.792524</td>
          <td>0.401061</td>
          <td>26.771505</td>
          <td>0.154604</td>
          <td>26.240925</td>
          <td>0.157685</td>
          <td>25.803929</td>
          <td>0.202626</td>
          <td>26.348206</td>
          <td>0.631597</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.456168</td>
          <td>1.412045</td>
          <td>27.409120</td>
          <td>0.316370</td>
          <td>28.113265</td>
          <td>0.491417</td>
          <td>26.551645</td>
          <td>0.222323</td>
          <td>24.936952</td>
          <td>0.104151</td>
          <td>24.370525</td>
          <td>0.141900</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.934388</td>
          <td>0.528867</td>
          <td>27.243173</td>
          <td>0.283666</td>
          <td>26.900172</td>
          <td>0.338497</td>
          <td>25.317459</td>
          <td>0.166591</td>
          <td>25.251154</td>
          <td>0.338749</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.271485</td>
          <td>0.315213</td>
          <td>26.189996</td>
          <td>0.105956</td>
          <td>25.977486</td>
          <td>0.077405</td>
          <td>25.700321</td>
          <td>0.098731</td>
          <td>25.582512</td>
          <td>0.168109</td>
          <td>24.920101</td>
          <td>0.208835</td>
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
          <td>27.284350</td>
          <td>0.700151</td>
          <td>26.523538</td>
          <td>0.151193</td>
          <td>25.433058</td>
          <td>0.051662</td>
          <td>25.052018</td>
          <td>0.060429</td>
          <td>24.727316</td>
          <td>0.086379</td>
          <td>24.568773</td>
          <td>0.167644</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.339718</td>
          <td>0.335940</td>
          <td>27.388076</td>
          <td>0.295149</td>
          <td>25.929339</td>
          <td>0.075315</td>
          <td>25.237493</td>
          <td>0.066693</td>
          <td>24.742627</td>
          <td>0.082253</td>
          <td>24.093564</td>
          <td>0.104355</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.844941</td>
          <td>0.504255</td>
          <td>26.641625</td>
          <td>0.163034</td>
          <td>26.329618</td>
          <td>0.110594</td>
          <td>26.421680</td>
          <td>0.192994</td>
          <td>25.797124</td>
          <td>0.210887</td>
          <td>24.879217</td>
          <td>0.211502</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.560547</td>
          <td>0.424462</td>
          <td>26.370316</td>
          <td>0.136742</td>
          <td>26.055459</td>
          <td>0.092889</td>
          <td>25.658069</td>
          <td>0.107053</td>
          <td>26.332999</td>
          <td>0.346533</td>
          <td>24.614339</td>
          <td>0.180623</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.577442</td>
          <td>0.409936</td>
          <td>26.703632</td>
          <td>0.170412</td>
          <td>26.489749</td>
          <td>0.125869</td>
          <td>26.262662</td>
          <td>0.166973</td>
          <td>25.813548</td>
          <td>0.211780</td>
          <td>27.413427</td>
          <td>1.260555</td>
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
