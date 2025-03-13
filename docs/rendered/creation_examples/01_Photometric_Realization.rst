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

    <pzflow.flow.Flow at 0x7f0e945eeb60>



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
          <td>27.060966</td>
          <td>0.573279</td>
          <td>26.574999</td>
          <td>0.147720</td>
          <td>25.944147</td>
          <td>0.075051</td>
          <td>25.121990</td>
          <td>0.059158</td>
          <td>24.719817</td>
          <td>0.079292</td>
          <td>24.023812</td>
          <td>0.096508</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.759906</td>
          <td>0.459798</td>
          <td>27.547681</td>
          <td>0.330973</td>
          <td>26.404720</td>
          <td>0.112481</td>
          <td>26.214764</td>
          <td>0.154043</td>
          <td>25.752402</td>
          <td>0.193862</td>
          <td>25.320664</td>
          <td>0.289957</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.447974</td>
          <td>2.152754</td>
          <td>29.064710</td>
          <td>0.968588</td>
          <td>28.258679</td>
          <td>0.510758</td>
          <td>25.837044</td>
          <td>0.111106</td>
          <td>25.101147</td>
          <td>0.110822</td>
          <td>24.279469</td>
          <td>0.120649</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.823084</td>
          <td>0.951753</td>
          <td>29.270180</td>
          <td>1.094474</td>
          <td>27.261148</td>
          <td>0.233403</td>
          <td>26.374880</td>
          <td>0.176581</td>
          <td>25.485116</td>
          <td>0.154477</td>
          <td>25.266285</td>
          <td>0.277466</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.479429</td>
          <td>0.371043</td>
          <td>26.186308</td>
          <td>0.105486</td>
          <td>25.872010</td>
          <td>0.070412</td>
          <td>25.447752</td>
          <td>0.078937</td>
          <td>25.559361</td>
          <td>0.164598</td>
          <td>24.969374</td>
          <td>0.217303</td>
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
          <td>26.646848</td>
          <td>0.422131</td>
          <td>26.341543</td>
          <td>0.120753</td>
          <td>25.440990</td>
          <td>0.048038</td>
          <td>25.118623</td>
          <td>0.058981</td>
          <td>24.735958</td>
          <td>0.080430</td>
          <td>24.772147</td>
          <td>0.184133</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.639596</td>
          <td>0.419804</td>
          <td>26.928319</td>
          <td>0.199430</td>
          <td>25.984831</td>
          <td>0.077798</td>
          <td>25.253132</td>
          <td>0.066453</td>
          <td>24.905798</td>
          <td>0.093401</td>
          <td>24.415464</td>
          <td>0.135732</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.858380</td>
          <td>0.494769</td>
          <td>26.446859</td>
          <td>0.132284</td>
          <td>26.468195</td>
          <td>0.118873</td>
          <td>26.007682</td>
          <td>0.128872</td>
          <td>25.947433</td>
          <td>0.228196</td>
          <td>26.455315</td>
          <td>0.679612</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.936600</td>
          <td>0.524021</td>
          <td>26.261863</td>
          <td>0.112670</td>
          <td>26.065931</td>
          <td>0.083568</td>
          <td>25.894429</td>
          <td>0.116803</td>
          <td>25.553199</td>
          <td>0.163735</td>
          <td>25.471569</td>
          <td>0.327222</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.273590</td>
          <td>0.665505</td>
          <td>26.554343</td>
          <td>0.145122</td>
          <td>26.497851</td>
          <td>0.121976</td>
          <td>26.200151</td>
          <td>0.152126</td>
          <td>25.705841</td>
          <td>0.186395</td>
          <td>25.846838</td>
          <td>0.437943</td>
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
          <td>27.196731</td>
          <td>0.691787</td>
          <td>27.073778</td>
          <td>0.257512</td>
          <td>26.096000</td>
          <td>0.100869</td>
          <td>25.209469</td>
          <td>0.075775</td>
          <td>24.721448</td>
          <td>0.093365</td>
          <td>23.885403</td>
          <td>0.100990</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.315265</td>
          <td>1.358808</td>
          <td>27.084533</td>
          <td>0.259833</td>
          <td>26.489608</td>
          <td>0.142022</td>
          <td>26.372083</td>
          <td>0.207099</td>
          <td>25.998613</td>
          <td>0.276771</td>
          <td>25.590791</td>
          <td>0.416508</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.889056</td>
          <td>0.495804</td>
          <td>27.859561</td>
          <td>0.443309</td>
          <td>25.942790</td>
          <td>0.147069</td>
          <td>25.067501</td>
          <td>0.129086</td>
          <td>24.332317</td>
          <td>0.152189</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.091179</td>
          <td>1.868175</td>
          <td>27.264881</td>
          <td>0.289640</td>
          <td>26.405102</td>
          <td>0.227309</td>
          <td>25.571351</td>
          <td>0.207151</td>
          <td>25.088425</td>
          <td>0.298505</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.807831</td>
          <td>0.525940</td>
          <td>26.100081</td>
          <td>0.112816</td>
          <td>25.897339</td>
          <td>0.084747</td>
          <td>25.704912</td>
          <td>0.117073</td>
          <td>25.476923</td>
          <td>0.179451</td>
          <td>24.676846</td>
          <td>0.199496</td>
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
          <td>26.355294</td>
          <td>0.143310</td>
          <td>25.513396</td>
          <td>0.061629</td>
          <td>25.156407</td>
          <td>0.073904</td>
          <td>24.931070</td>
          <td>0.114518</td>
          <td>24.718639</td>
          <td>0.210832</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.228393</td>
          <td>0.708534</td>
          <td>26.640198</td>
          <td>0.180071</td>
          <td>26.149957</td>
          <td>0.106184</td>
          <td>25.162272</td>
          <td>0.072996</td>
          <td>24.793889</td>
          <td>0.099903</td>
          <td>24.134904</td>
          <td>0.126047</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.907253</td>
          <td>0.569541</td>
          <td>26.962509</td>
          <td>0.237550</td>
          <td>26.446336</td>
          <td>0.138523</td>
          <td>26.241445</td>
          <td>0.187882</td>
          <td>25.667288</td>
          <td>0.213146</td>
          <td>26.574707</td>
          <td>0.842457</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.241711</td>
          <td>0.349414</td>
          <td>26.126296</td>
          <td>0.118678</td>
          <td>26.196099</td>
          <td>0.113580</td>
          <td>25.881520</td>
          <td>0.140811</td>
          <td>25.731613</td>
          <td>0.228865</td>
          <td>25.773097</td>
          <td>0.491059</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.241564</td>
          <td>0.344280</td>
          <td>26.761551</td>
          <td>0.200480</td>
          <td>26.369557</td>
          <td>0.129282</td>
          <td>26.191070</td>
          <td>0.179553</td>
          <td>26.627309</td>
          <td>0.456903</td>
          <td>25.453224</td>
          <td>0.377930</td>
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
          <td>26.863523</td>
          <td>0.188867</td>
          <td>26.125088</td>
          <td>0.088050</td>
          <td>25.274379</td>
          <td>0.067726</td>
          <td>24.755199</td>
          <td>0.081817</td>
          <td>23.995750</td>
          <td>0.094173</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.742099</td>
          <td>1.576467</td>
          <td>27.789084</td>
          <td>0.400001</td>
          <td>26.593281</td>
          <td>0.132616</td>
          <td>26.212370</td>
          <td>0.153877</td>
          <td>25.978785</td>
          <td>0.234410</td>
          <td>26.123061</td>
          <td>0.538020</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.302557</td>
          <td>0.710296</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.610292</td>
          <td>0.699508</td>
          <td>26.030343</td>
          <td>0.142944</td>
          <td>25.075593</td>
          <td>0.117540</td>
          <td>24.325561</td>
          <td>0.136505</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.633466</td>
          <td>0.481884</td>
          <td>28.094062</td>
          <td>0.593206</td>
          <td>27.606618</td>
          <td>0.378549</td>
          <td>26.202458</td>
          <td>0.191192</td>
          <td>25.597318</td>
          <td>0.210981</td>
          <td>24.933709</td>
          <td>0.262420</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.530328</td>
          <td>0.386333</td>
          <td>25.945686</td>
          <td>0.085531</td>
          <td>26.026525</td>
          <td>0.080830</td>
          <td>25.789377</td>
          <td>0.106734</td>
          <td>25.643770</td>
          <td>0.177095</td>
          <td>24.792902</td>
          <td>0.187658</td>
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
          <td>28.920956</td>
          <td>1.766101</td>
          <td>26.392209</td>
          <td>0.135045</td>
          <td>25.431422</td>
          <td>0.051587</td>
          <td>25.072368</td>
          <td>0.061530</td>
          <td>24.984665</td>
          <td>0.108248</td>
          <td>24.783678</td>
          <td>0.201059</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.225267</td>
          <td>0.649499</td>
          <td>27.344136</td>
          <td>0.284863</td>
          <td>25.991776</td>
          <td>0.079584</td>
          <td>25.135971</td>
          <td>0.060953</td>
          <td>24.790505</td>
          <td>0.085798</td>
          <td>24.393646</td>
          <td>0.135459</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.871405</td>
          <td>1.003656</td>
          <td>26.661452</td>
          <td>0.165813</td>
          <td>26.573866</td>
          <td>0.136711</td>
          <td>26.438048</td>
          <td>0.195672</td>
          <td>26.307315</td>
          <td>0.320030</td>
          <td>26.019411</td>
          <td>0.519876</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.418174</td>
          <td>0.380474</td>
          <td>26.083295</td>
          <td>0.106597</td>
          <td>26.198603</td>
          <td>0.105303</td>
          <td>25.928528</td>
          <td>0.135407</td>
          <td>25.686617</td>
          <td>0.204666</td>
          <td>25.650984</td>
          <td>0.418120</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.522651</td>
          <td>0.802753</td>
          <td>27.021527</td>
          <td>0.222648</td>
          <td>26.554295</td>
          <td>0.133103</td>
          <td>26.416737</td>
          <td>0.190275</td>
          <td>25.788399</td>
          <td>0.207371</td>
          <td>25.839005</td>
          <td>0.450731</td>
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
