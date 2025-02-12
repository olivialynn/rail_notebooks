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

    <pzflow.flow.Flow at 0x7f334a7ab6d0>



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
    0      1.398945  27.667538  26.723339  26.032640  25.178589  24.695959   
    1      2.285624  28.786999  27.476589  26.640173  26.259747  25.865671   
    2      1.495130  30.011343  29.789326  28.200378  26.014816  25.030161   
    3      0.842595  29.306242  28.721798  27.353014  26.256908  25.529823   
    4      1.588960  26.273870  26.115385  25.950439  25.687403  25.466604   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270809  26.371513  25.436861  25.077417  24.852785   
    99996  1.481047  27.478111  26.735254  26.042774  25.204937  24.825092   
    99997  2.023549  26.990149  26.714739  26.377953  26.250345  25.917372   
    99998  1.548203  26.367432  26.206882  26.087980  25.876928  25.715893   
    99999  1.739491  26.881981  26.773064  26.553120  26.319618  25.955980   
    
                   y     major     minor  
    0      23.994417  0.003319  0.002869  
    1      25.391062  0.008733  0.007945  
    2      24.304695  0.103938  0.052162  
    3      25.291105  0.147522  0.143359  
    4      25.096741  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737953  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613838  0.073146  0.047825  
    99998  25.274897  0.100551  0.094662  
    99999  25.699638  0.059611  0.049181  
    
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
          <td>1.398945</td>
          <td>27.029563</td>
          <td>0.560519</td>
          <td>26.723601</td>
          <td>0.167730</td>
          <td>25.912471</td>
          <td>0.072978</td>
          <td>25.277678</td>
          <td>0.067914</td>
          <td>24.770457</td>
          <td>0.082914</td>
          <td>23.991683</td>
          <td>0.093825</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.165343</td>
          <td>0.529968</td>
          <td>26.642974</td>
          <td>0.138303</td>
          <td>26.554587</td>
          <td>0.205479</td>
          <td>25.984546</td>
          <td>0.235320</td>
          <td>28.885913</td>
          <td>2.404695</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.269714</td>
          <td>0.235064</td>
          <td>26.088964</td>
          <td>0.138251</td>
          <td>25.191909</td>
          <td>0.119936</td>
          <td>24.310006</td>
          <td>0.123890</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>26.641350</td>
          <td>0.420366</td>
          <td>30.221938</td>
          <td>1.786010</td>
          <td>27.623360</td>
          <td>0.313455</td>
          <td>26.106857</td>
          <td>0.140401</td>
          <td>25.768106</td>
          <td>0.196441</td>
          <td>25.069929</td>
          <td>0.236223</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.381662</td>
          <td>0.343684</td>
          <td>26.064100</td>
          <td>0.094787</td>
          <td>26.021024</td>
          <td>0.080323</td>
          <td>25.614449</td>
          <td>0.091425</td>
          <td>25.210894</td>
          <td>0.121930</td>
          <td>25.423726</td>
          <td>0.314984</td>
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
          <td>28.309632</td>
          <td>1.261312</td>
          <td>26.177821</td>
          <td>0.104707</td>
          <td>25.482166</td>
          <td>0.049827</td>
          <td>25.090705</td>
          <td>0.057538</td>
          <td>24.798386</td>
          <td>0.084980</td>
          <td>24.618263</td>
          <td>0.161558</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.698944</td>
          <td>0.439163</td>
          <td>26.468816</td>
          <td>0.134817</td>
          <td>26.189557</td>
          <td>0.093173</td>
          <td>25.172853</td>
          <td>0.061889</td>
          <td>24.913296</td>
          <td>0.094018</td>
          <td>24.218106</td>
          <td>0.114377</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.534209</td>
          <td>0.387159</td>
          <td>26.966467</td>
          <td>0.205913</td>
          <td>26.289169</td>
          <td>0.101679</td>
          <td>26.384381</td>
          <td>0.178010</td>
          <td>26.138842</td>
          <td>0.267120</td>
          <td>25.193182</td>
          <td>0.261420</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.808803</td>
          <td>0.476907</td>
          <td>26.037528</td>
          <td>0.092604</td>
          <td>26.120423</td>
          <td>0.087677</td>
          <td>25.954823</td>
          <td>0.123098</td>
          <td>25.832514</td>
          <td>0.207352</td>
          <td>25.171163</td>
          <td>0.256750</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.277346</td>
          <td>0.667226</td>
          <td>27.099432</td>
          <td>0.230032</td>
          <td>26.417325</td>
          <td>0.113724</td>
          <td>26.554107</td>
          <td>0.205397</td>
          <td>26.531805</td>
          <td>0.365679</td>
          <td>29.302808</td>
          <td>2.783114</td>
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
          <td>1.398945</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.760879</td>
          <td>0.198640</td>
          <td>25.914656</td>
          <td>0.086021</td>
          <td>25.104985</td>
          <td>0.069088</td>
          <td>24.814425</td>
          <td>0.101293</td>
          <td>24.098544</td>
          <td>0.121617</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.068662</td>
          <td>0.256480</td>
          <td>26.553603</td>
          <td>0.150053</td>
          <td>26.425198</td>
          <td>0.216497</td>
          <td>26.622158</td>
          <td>0.451326</td>
          <td>25.686926</td>
          <td>0.448052</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.343748</td>
          <td>1.393061</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.671651</td>
          <td>0.383903</td>
          <td>26.673119</td>
          <td>0.271336</td>
          <td>24.859256</td>
          <td>0.107706</td>
          <td>24.131546</td>
          <td>0.128011</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>26.649848</td>
          <td>0.488925</td>
          <td>27.662286</td>
          <td>0.433135</td>
          <td>27.491197</td>
          <td>0.346961</td>
          <td>26.273197</td>
          <td>0.203624</td>
          <td>25.637307</td>
          <td>0.218879</td>
          <td>25.086423</td>
          <td>0.298025</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.955806</td>
          <td>0.272055</td>
          <td>25.980848</td>
          <td>0.101672</td>
          <td>25.972188</td>
          <td>0.090516</td>
          <td>25.816601</td>
          <td>0.128990</td>
          <td>25.698611</td>
          <td>0.216218</td>
          <td>24.993313</td>
          <td>0.259384</td>
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
          <td>26.950100</td>
          <td>0.590204</td>
          <td>26.341436</td>
          <td>0.141613</td>
          <td>25.452287</td>
          <td>0.058379</td>
          <td>25.019391</td>
          <td>0.065467</td>
          <td>25.034992</td>
          <td>0.125340</td>
          <td>24.831586</td>
          <td>0.231607</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.791038</td>
          <td>1.014233</td>
          <td>26.526754</td>
          <td>0.163528</td>
          <td>26.189875</td>
          <td>0.109950</td>
          <td>25.168279</td>
          <td>0.073385</td>
          <td>24.730349</td>
          <td>0.094490</td>
          <td>24.167084</td>
          <td>0.129609</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.763693</td>
          <td>0.201318</td>
          <td>26.767509</td>
          <td>0.182276</td>
          <td>25.947101</td>
          <td>0.146201</td>
          <td>26.287333</td>
          <td>0.352627</td>
          <td>25.934921</td>
          <td>0.544110</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.318862</td>
          <td>0.371151</td>
          <td>26.321264</td>
          <td>0.140470</td>
          <td>26.065920</td>
          <td>0.101375</td>
          <td>25.766606</td>
          <td>0.127504</td>
          <td>25.661169</td>
          <td>0.215842</td>
          <td>24.557159</td>
          <td>0.185969</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.317503</td>
          <td>0.365402</td>
          <td>27.176291</td>
          <td>0.282290</td>
          <td>26.461779</td>
          <td>0.140001</td>
          <td>26.341827</td>
          <td>0.203884</td>
          <td>25.858106</td>
          <td>0.249013</td>
          <td>27.557516</td>
          <td>1.474953</td>
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
          <td>1.398945</td>
          <td>26.571244</td>
          <td>0.398414</td>
          <td>26.448942</td>
          <td>0.132537</td>
          <td>26.090301</td>
          <td>0.085394</td>
          <td>25.162027</td>
          <td>0.061306</td>
          <td>24.609372</td>
          <td>0.071929</td>
          <td>24.033328</td>
          <td>0.097330</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.608534</td>
          <td>0.832156</td>
          <td>27.035777</td>
          <td>0.218353</td>
          <td>26.715295</td>
          <td>0.147325</td>
          <td>26.451820</td>
          <td>0.188644</td>
          <td>25.603991</td>
          <td>0.171132</td>
          <td>26.261636</td>
          <td>0.594273</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.364456</td>
          <td>0.647164</td>
          <td>28.097526</td>
          <td>0.485717</td>
          <td>25.769413</td>
          <td>0.114026</td>
          <td>25.011540</td>
          <td>0.111162</td>
          <td>24.462273</td>
          <td>0.153536</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.677704</td>
          <td>0.877623</td>
          <td>27.180262</td>
          <td>0.269536</td>
          <td>26.152531</td>
          <td>0.183299</td>
          <td>25.886595</td>
          <td>0.267911</td>
          <td>25.297182</td>
          <td>0.351267</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.108022</td>
          <td>0.276369</td>
          <td>26.171226</td>
          <td>0.104233</td>
          <td>25.986188</td>
          <td>0.078002</td>
          <td>25.642753</td>
          <td>0.093867</td>
          <td>25.566801</td>
          <td>0.165874</td>
          <td>25.023902</td>
          <td>0.227704</td>
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
          <td>26.875554</td>
          <td>0.524978</td>
          <td>26.268107</td>
          <td>0.121294</td>
          <td>25.414017</td>
          <td>0.050796</td>
          <td>25.106349</td>
          <td>0.063412</td>
          <td>24.842186</td>
          <td>0.095556</td>
          <td>24.943389</td>
          <td>0.229718</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>30.115619</td>
          <td>2.761435</td>
          <td>26.802542</td>
          <td>0.181852</td>
          <td>25.976850</td>
          <td>0.078543</td>
          <td>25.171480</td>
          <td>0.062903</td>
          <td>25.018582</td>
          <td>0.104813</td>
          <td>24.328838</td>
          <td>0.128075</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.267717</td>
          <td>0.323950</td>
          <td>26.925393</td>
          <td>0.207218</td>
          <td>26.186298</td>
          <td>0.097563</td>
          <td>26.124688</td>
          <td>0.149905</td>
          <td>26.551960</td>
          <td>0.387864</td>
          <td>25.642013</td>
          <td>0.391310</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.298780</td>
          <td>0.346588</td>
          <td>26.235938</td>
          <td>0.121738</td>
          <td>25.966037</td>
          <td>0.085865</td>
          <td>25.828225</td>
          <td>0.124147</td>
          <td>25.537314</td>
          <td>0.180472</td>
          <td>25.440659</td>
          <td>0.355257</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.282630</td>
          <td>0.684028</td>
          <td>26.570153</td>
          <td>0.152062</td>
          <td>26.371755</td>
          <td>0.113599</td>
          <td>26.239534</td>
          <td>0.163713</td>
          <td>26.015213</td>
          <td>0.250306</td>
          <td>25.444111</td>
          <td>0.332057</td>
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
