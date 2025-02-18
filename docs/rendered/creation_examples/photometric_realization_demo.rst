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

    <pzflow.flow.Flow at 0x7fcffe2243d0>



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
          <td>inf</td>
          <td>inf</td>
          <td>26.782198</td>
          <td>0.176293</td>
          <td>26.150916</td>
          <td>0.090061</td>
          <td>25.139577</td>
          <td>0.060088</td>
          <td>24.646429</td>
          <td>0.074315</td>
          <td>23.917089</td>
          <td>0.087869</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.440521</td>
          <td>0.745137</td>
          <td>28.795422</td>
          <td>0.817867</td>
          <td>26.483169</td>
          <td>0.120430</td>
          <td>26.317525</td>
          <td>0.168178</td>
          <td>25.824767</td>
          <td>0.206011</td>
          <td>25.843174</td>
          <td>0.436729</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>26.478029</td>
          <td>0.370639</td>
          <td>29.196977</td>
          <td>1.048569</td>
          <td>28.135781</td>
          <td>0.466271</td>
          <td>26.027344</td>
          <td>0.131084</td>
          <td>25.022390</td>
          <td>0.103453</td>
          <td>24.183113</td>
          <td>0.110940</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>26.843508</td>
          <td>0.489356</td>
          <td>28.119467</td>
          <td>0.512488</td>
          <td>27.311575</td>
          <td>0.243331</td>
          <td>25.939293</td>
          <td>0.121449</td>
          <td>25.716839</td>
          <td>0.188134</td>
          <td>25.482545</td>
          <td>0.330087</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.499469</td>
          <td>0.376872</td>
          <td>26.020270</td>
          <td>0.091212</td>
          <td>26.078602</td>
          <td>0.084507</td>
          <td>25.692153</td>
          <td>0.097880</td>
          <td>25.446445</td>
          <td>0.149438</td>
          <td>25.331818</td>
          <td>0.292580</td>
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
          <td>26.337883</td>
          <td>0.332001</td>
          <td>26.548217</td>
          <td>0.144360</td>
          <td>25.396192</td>
          <td>0.046165</td>
          <td>25.076164</td>
          <td>0.056800</td>
          <td>24.848800</td>
          <td>0.088837</td>
          <td>24.677157</td>
          <td>0.169877</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.117203</td>
          <td>0.596679</td>
          <td>26.709353</td>
          <td>0.165708</td>
          <td>25.901486</td>
          <td>0.072272</td>
          <td>25.227441</td>
          <td>0.064958</td>
          <td>24.802796</td>
          <td>0.085311</td>
          <td>24.151531</td>
          <td>0.107924</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.616144</td>
          <td>0.412350</td>
          <td>27.170340</td>
          <td>0.243910</td>
          <td>26.255042</td>
          <td>0.098684</td>
          <td>26.435262</td>
          <td>0.185846</td>
          <td>25.983840</td>
          <td>0.235183</td>
          <td>25.669775</td>
          <td>0.382340</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.771444</td>
          <td>0.463790</td>
          <td>26.292102</td>
          <td>0.115675</td>
          <td>26.092032</td>
          <td>0.085513</td>
          <td>25.897019</td>
          <td>0.117066</td>
          <td>25.513490</td>
          <td>0.158275</td>
          <td>25.178483</td>
          <td>0.258294</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.063465</td>
          <td>0.574304</td>
          <td>26.744034</td>
          <td>0.170671</td>
          <td>26.834709</td>
          <td>0.163039</td>
          <td>26.335540</td>
          <td>0.170777</td>
          <td>25.838426</td>
          <td>0.208381</td>
          <td>25.391005</td>
          <td>0.306843</td>
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
          <td>26.557031</td>
          <td>0.436388</td>
          <td>26.692414</td>
          <td>0.187516</td>
          <td>25.975115</td>
          <td>0.090719</td>
          <td>25.136499</td>
          <td>0.071041</td>
          <td>24.597826</td>
          <td>0.083746</td>
          <td>24.072081</td>
          <td>0.118853</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.107919</td>
          <td>0.650957</td>
          <td>27.731926</td>
          <td>0.433321</td>
          <td>26.718704</td>
          <td>0.172775</td>
          <td>26.040670</td>
          <td>0.156416</td>
          <td>25.639266</td>
          <td>0.205732</td>
          <td>25.613902</td>
          <td>0.423920</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.501726</td>
          <td>0.761933</td>
          <td>27.473676</td>
          <td>0.328656</td>
          <td>26.121545</td>
          <td>0.171350</td>
          <td>24.951831</td>
          <td>0.116757</td>
          <td>24.145551</td>
          <td>0.129572</td>
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
          <td>27.173976</td>
          <td>0.269048</td>
          <td>26.190559</td>
          <td>0.189953</td>
          <td>25.347598</td>
          <td>0.171511</td>
          <td>24.937568</td>
          <td>0.264146</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.874590</td>
          <td>0.254621</td>
          <td>26.122015</td>
          <td>0.114990</td>
          <td>25.938340</td>
          <td>0.087861</td>
          <td>25.551715</td>
          <td>0.102424</td>
          <td>25.996846</td>
          <td>0.276404</td>
          <td>25.099040</td>
          <td>0.282703</td>
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
          <td>27.634694</td>
          <td>0.930461</td>
          <td>26.585451</td>
          <td>0.174449</td>
          <td>25.473100</td>
          <td>0.059466</td>
          <td>25.178544</td>
          <td>0.075364</td>
          <td>24.844878</td>
          <td>0.106225</td>
          <td>24.545577</td>
          <td>0.182272</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.635027</td>
          <td>0.464039</td>
          <td>26.511941</td>
          <td>0.161475</td>
          <td>26.102723</td>
          <td>0.101887</td>
          <td>25.156180</td>
          <td>0.072604</td>
          <td>24.852315</td>
          <td>0.105142</td>
          <td>24.367593</td>
          <td>0.154036</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.388408</td>
          <td>0.386817</td>
          <td>26.716137</td>
          <td>0.193432</td>
          <td>26.368382</td>
          <td>0.129501</td>
          <td>26.588843</td>
          <td>0.250959</td>
          <td>25.767061</td>
          <td>0.231587</td>
          <td>26.068557</td>
          <td>0.598743</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.570818</td>
          <td>0.450156</td>
          <td>26.246050</td>
          <td>0.131649</td>
          <td>26.139467</td>
          <td>0.108106</td>
          <td>25.666533</td>
          <td>0.116895</td>
          <td>25.764789</td>
          <td>0.235240</td>
          <td>25.335938</td>
          <td>0.351615</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.032475</td>
          <td>0.621312</td>
          <td>26.932477</td>
          <td>0.231183</td>
          <td>26.511164</td>
          <td>0.146080</td>
          <td>26.095289</td>
          <td>0.165516</td>
          <td>25.515012</td>
          <td>0.187067</td>
          <td>26.707257</td>
          <td>0.914157</td>
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
          <td>29.947304</td>
          <td>2.595581</td>
          <td>26.640003</td>
          <td>0.156199</td>
          <td>26.062902</td>
          <td>0.083357</td>
          <td>25.219161</td>
          <td>0.064492</td>
          <td>24.900906</td>
          <td>0.093013</td>
          <td>23.977487</td>
          <td>0.092675</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.637588</td>
          <td>0.355563</td>
          <td>26.588242</td>
          <td>0.132040</td>
          <td>26.310416</td>
          <td>0.167324</td>
          <td>25.752336</td>
          <td>0.194027</td>
          <td>26.694067</td>
          <td>0.797795</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.926840</td>
          <td>1.617270</td>
          <td>28.329623</td>
          <td>0.575220</td>
          <td>25.765509</td>
          <td>0.113639</td>
          <td>25.047280</td>
          <td>0.114678</td>
          <td>24.240890</td>
          <td>0.126867</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.889558</td>
          <td>0.511810</td>
          <td>27.372920</td>
          <td>0.314869</td>
          <td>26.406436</td>
          <td>0.226770</td>
          <td>25.838391</td>
          <td>0.257564</td>
          <td>26.428486</td>
          <td>0.796383</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.694007</td>
          <td>0.437896</td>
          <td>26.099033</td>
          <td>0.097854</td>
          <td>26.010297</td>
          <td>0.079680</td>
          <td>25.644732</td>
          <td>0.094031</td>
          <td>25.497060</td>
          <td>0.156281</td>
          <td>24.864610</td>
          <td>0.199342</td>
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
          <td>26.954087</td>
          <td>0.555727</td>
          <td>26.291707</td>
          <td>0.123803</td>
          <td>25.384261</td>
          <td>0.049472</td>
          <td>25.247698</td>
          <td>0.071867</td>
          <td>24.730530</td>
          <td>0.086624</td>
          <td>24.765416</td>
          <td>0.197998</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.029755</td>
          <td>1.085301</td>
          <td>26.767037</td>
          <td>0.176464</td>
          <td>26.014441</td>
          <td>0.081192</td>
          <td>25.175295</td>
          <td>0.063116</td>
          <td>24.813753</td>
          <td>0.087572</td>
          <td>24.157228</td>
          <td>0.110323</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>28.208179</td>
          <td>1.218914</td>
          <td>26.685791</td>
          <td>0.169284</td>
          <td>26.541992</td>
          <td>0.132998</td>
          <td>26.360341</td>
          <td>0.183253</td>
          <td>26.368297</td>
          <td>0.335914</td>
          <td>25.245198</td>
          <td>0.285828</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.868761</td>
          <td>0.533905</td>
          <td>26.164270</td>
          <td>0.114390</td>
          <td>25.979408</td>
          <td>0.086881</td>
          <td>26.103991</td>
          <td>0.157451</td>
          <td>25.481380</td>
          <td>0.172107</td>
          <td>25.289252</td>
          <td>0.315114</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.121275</td>
          <td>0.611640</td>
          <td>26.625797</td>
          <td>0.159475</td>
          <td>26.586730</td>
          <td>0.136885</td>
          <td>26.642910</td>
          <td>0.229903</td>
          <td>25.637159</td>
          <td>0.182581</td>
          <td>25.880257</td>
          <td>0.464916</td>
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
