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

    <pzflow.flow.Flow at 0x7f200c079c90>



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
          <td>26.605482</td>
          <td>0.408997</td>
          <td>26.675329</td>
          <td>0.160969</td>
          <td>26.136972</td>
          <td>0.088964</td>
          <td>25.260962</td>
          <td>0.066916</td>
          <td>24.665190</td>
          <td>0.075558</td>
          <td>24.157129</td>
          <td>0.108453</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.811717</td>
          <td>0.945134</td>
          <td>27.206792</td>
          <td>0.251331</td>
          <td>26.856550</td>
          <td>0.166105</td>
          <td>26.342639</td>
          <td>0.171811</td>
          <td>25.809452</td>
          <td>0.203383</td>
          <td>25.040741</td>
          <td>0.230584</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.190144</td>
          <td>0.628090</td>
          <td>29.470264</td>
          <td>1.225716</td>
          <td>29.233318</td>
          <td>0.983360</td>
          <td>25.969339</td>
          <td>0.124659</td>
          <td>24.972596</td>
          <td>0.099039</td>
          <td>24.351553</td>
          <td>0.128434</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.054053</td>
          <td>1.092419</td>
          <td>28.986964</td>
          <td>0.923389</td>
          <td>27.171126</td>
          <td>0.216582</td>
          <td>26.074236</td>
          <td>0.136505</td>
          <td>25.692750</td>
          <td>0.184344</td>
          <td>25.219401</td>
          <td>0.267078</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.954424</td>
          <td>0.243553</td>
          <td>26.210110</td>
          <td>0.107700</td>
          <td>25.972881</td>
          <td>0.076981</td>
          <td>25.899875</td>
          <td>0.117357</td>
          <td>25.457654</td>
          <td>0.150883</td>
          <td>24.925842</td>
          <td>0.209546</td>
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
          <td>26.166730</td>
          <td>0.103697</td>
          <td>25.458371</td>
          <td>0.048785</td>
          <td>25.006649</td>
          <td>0.053401</td>
          <td>24.836649</td>
          <td>0.087892</td>
          <td>24.800305</td>
          <td>0.188567</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.527704</td>
          <td>2.221841</td>
          <td>26.888791</td>
          <td>0.192911</td>
          <td>26.055159</td>
          <td>0.082779</td>
          <td>25.221813</td>
          <td>0.064634</td>
          <td>24.790275</td>
          <td>0.084375</td>
          <td>24.364303</td>
          <td>0.129860</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.761270</td>
          <td>0.916111</td>
          <td>26.729814</td>
          <td>0.168620</td>
          <td>26.264926</td>
          <td>0.099542</td>
          <td>26.065031</td>
          <td>0.135425</td>
          <td>25.916592</td>
          <td>0.222423</td>
          <td>25.454274</td>
          <td>0.322752</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.148749</td>
          <td>0.285375</td>
          <td>26.129545</td>
          <td>0.100379</td>
          <td>25.983759</td>
          <td>0.077724</td>
          <td>25.942689</td>
          <td>0.121808</td>
          <td>25.675986</td>
          <td>0.181747</td>
          <td>25.607186</td>
          <td>0.364146</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.062503</td>
          <td>0.573909</td>
          <td>26.824083</td>
          <td>0.182659</td>
          <td>26.545301</td>
          <td>0.127102</td>
          <td>26.344582</td>
          <td>0.172095</td>
          <td>26.090319</td>
          <td>0.256730</td>
          <td>25.894724</td>
          <td>0.454066</td>
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
          <td>26.477212</td>
          <td>0.410655</td>
          <td>26.739762</td>
          <td>0.195146</td>
          <td>25.949596</td>
          <td>0.088707</td>
          <td>25.124312</td>
          <td>0.070280</td>
          <td>24.807849</td>
          <td>0.100712</td>
          <td>23.939138</td>
          <td>0.105850</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.210548</td>
          <td>0.698398</td>
          <td>27.391719</td>
          <td>0.332774</td>
          <td>26.740221</td>
          <td>0.175961</td>
          <td>26.438516</td>
          <td>0.218913</td>
          <td>25.562693</td>
          <td>0.192915</td>
          <td>28.833107</td>
          <td>2.527747</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>32.126844</td>
          <td>4.836118</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.133337</td>
          <td>0.173076</td>
          <td>25.111906</td>
          <td>0.134139</td>
          <td>24.336581</td>
          <td>0.152746</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.071833</td>
          <td>0.247462</td>
          <td>26.138240</td>
          <td>0.181739</td>
          <td>25.732075</td>
          <td>0.236783</td>
          <td>26.299032</td>
          <td>0.733082</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.470973</td>
          <td>0.408785</td>
          <td>26.198042</td>
          <td>0.122837</td>
          <td>25.925281</td>
          <td>0.086857</td>
          <td>25.862529</td>
          <td>0.134216</td>
          <td>25.319957</td>
          <td>0.157000</td>
          <td>25.506195</td>
          <td>0.390321</td>
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
          <td>26.834351</td>
          <td>0.543210</td>
          <td>26.180327</td>
          <td>0.123218</td>
          <td>25.350413</td>
          <td>0.053335</td>
          <td>24.988821</td>
          <td>0.063718</td>
          <td>24.889147</td>
          <td>0.110410</td>
          <td>24.744007</td>
          <td>0.215346</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.415218</td>
          <td>0.148650</td>
          <td>26.003197</td>
          <td>0.093374</td>
          <td>25.463575</td>
          <td>0.095188</td>
          <td>24.960067</td>
          <td>0.115504</td>
          <td>24.014135</td>
          <td>0.113490</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.533724</td>
          <td>0.165748</td>
          <td>26.451707</td>
          <td>0.139166</td>
          <td>26.404531</td>
          <td>0.215443</td>
          <td>26.146283</td>
          <td>0.315341</td>
          <td>25.694365</td>
          <td>0.455585</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.655535</td>
          <td>0.479612</td>
          <td>26.472846</td>
          <td>0.159963</td>
          <td>26.233214</td>
          <td>0.117309</td>
          <td>25.812203</td>
          <td>0.132636</td>
          <td>25.638275</td>
          <td>0.211757</td>
          <td>25.022331</td>
          <td>0.273587</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.591919</td>
          <td>0.450984</td>
          <td>26.712117</td>
          <td>0.192320</td>
          <td>26.376852</td>
          <td>0.130101</td>
          <td>26.195412</td>
          <td>0.180215</td>
          <td>25.757805</td>
          <td>0.229223</td>
          <td>25.779195</td>
          <td>0.484237</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.742440</td>
          <td>0.170459</td>
          <td>26.053180</td>
          <td>0.082645</td>
          <td>25.233326</td>
          <td>0.065306</td>
          <td>24.745519</td>
          <td>0.081122</td>
          <td>23.988831</td>
          <td>0.093603</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.863451</td>
          <td>0.423448</td>
          <td>26.592374</td>
          <td>0.132513</td>
          <td>26.322827</td>
          <td>0.169102</td>
          <td>25.971400</td>
          <td>0.232981</td>
          <td>25.339747</td>
          <td>0.294722</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.145014</td>
          <td>1.067128</td>
          <td>29.957607</td>
          <td>1.539425</td>
          <td>26.119216</td>
          <td>0.154281</td>
          <td>25.077837</td>
          <td>0.117769</td>
          <td>24.314612</td>
          <td>0.135221</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.756958</td>
          <td>0.424975</td>
          <td>26.569690</td>
          <td>0.259435</td>
          <td>25.427457</td>
          <td>0.182897</td>
          <td>25.812385</td>
          <td>0.519610</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.188078</td>
          <td>0.294836</td>
          <td>26.089804</td>
          <td>0.097066</td>
          <td>26.242361</td>
          <td>0.097731</td>
          <td>25.906423</td>
          <td>0.118203</td>
          <td>25.395172</td>
          <td>0.143193</td>
          <td>25.543081</td>
          <td>0.346732</td>
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
          <td>27.280519</td>
          <td>0.698334</td>
          <td>26.574047</td>
          <td>0.157872</td>
          <td>25.411226</td>
          <td>0.050671</td>
          <td>25.026393</td>
          <td>0.059071</td>
          <td>24.949832</td>
          <td>0.105003</td>
          <td>24.988062</td>
          <td>0.238369</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.314780</td>
          <td>0.690680</td>
          <td>26.746142</td>
          <td>0.173363</td>
          <td>25.862807</td>
          <td>0.071011</td>
          <td>25.149899</td>
          <td>0.061711</td>
          <td>24.775685</td>
          <td>0.084685</td>
          <td>24.267631</td>
          <td>0.121453</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.682941</td>
          <td>0.446925</td>
          <td>26.820787</td>
          <td>0.189787</td>
          <td>26.381037</td>
          <td>0.115663</td>
          <td>26.539556</td>
          <td>0.213053</td>
          <td>25.902260</td>
          <td>0.230176</td>
          <td>25.938761</td>
          <td>0.489903</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.732640</td>
          <td>0.483098</td>
          <td>26.335534</td>
          <td>0.132699</td>
          <td>25.909151</td>
          <td>0.081667</td>
          <td>25.760160</td>
          <td>0.117018</td>
          <td>25.958301</td>
          <td>0.256371</td>
          <td>25.359160</td>
          <td>0.333138</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.560663</td>
          <td>0.404696</td>
          <td>26.690596</td>
          <td>0.168533</td>
          <td>26.681862</td>
          <td>0.148568</td>
          <td>26.237049</td>
          <td>0.163366</td>
          <td>25.775782</td>
          <td>0.205191</td>
          <td>inf</td>
          <td>inf</td>
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
