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

    <pzflow.flow.Flow at 0x7f4387c08b20>



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
          <td>26.383088</td>
          <td>0.344070</td>
          <td>26.860675</td>
          <td>0.188393</td>
          <td>25.794405</td>
          <td>0.065735</td>
          <td>25.157237</td>
          <td>0.061037</td>
          <td>24.670529</td>
          <td>0.075915</td>
          <td>24.012779</td>
          <td>0.095578</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.865822</td>
          <td>0.976896</td>
          <td>27.983272</td>
          <td>0.463255</td>
          <td>26.828990</td>
          <td>0.162245</td>
          <td>26.026450</td>
          <td>0.130982</td>
          <td>26.160410</td>
          <td>0.271856</td>
          <td>25.916185</td>
          <td>0.461445</td>
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
          <td>27.906082</td>
          <td>0.391505</td>
          <td>26.134443</td>
          <td>0.143776</td>
          <td>24.882103</td>
          <td>0.091477</td>
          <td>24.269909</td>
          <td>0.119650</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>29.642242</td>
          <td>2.322234</td>
          <td>27.655204</td>
          <td>0.360244</td>
          <td>27.298622</td>
          <td>0.240745</td>
          <td>26.563736</td>
          <td>0.207060</td>
          <td>25.548287</td>
          <td>0.163051</td>
          <td>25.544629</td>
          <td>0.346697</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.339191</td>
          <td>0.332345</td>
          <td>26.080546</td>
          <td>0.096164</td>
          <td>26.021697</td>
          <td>0.080371</td>
          <td>25.747864</td>
          <td>0.102776</td>
          <td>25.680441</td>
          <td>0.182434</td>
          <td>25.305646</td>
          <td>0.286459</td>
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
          <td>27.192072</td>
          <td>0.628937</td>
          <td>26.260466</td>
          <td>0.112533</td>
          <td>25.480122</td>
          <td>0.049736</td>
          <td>25.016546</td>
          <td>0.053872</td>
          <td>24.854729</td>
          <td>0.089301</td>
          <td>24.992778</td>
          <td>0.221581</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.654396</td>
          <td>0.856528</td>
          <td>26.772250</td>
          <td>0.174811</td>
          <td>25.994828</td>
          <td>0.078487</td>
          <td>25.307309</td>
          <td>0.069720</td>
          <td>24.818123</td>
          <td>0.086470</td>
          <td>24.268284</td>
          <td>0.119481</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.612041</td>
          <td>0.833635</td>
          <td>26.506865</td>
          <td>0.139313</td>
          <td>26.253136</td>
          <td>0.098519</td>
          <td>25.987475</td>
          <td>0.126635</td>
          <td>25.838572</td>
          <td>0.208406</td>
          <td>25.741804</td>
          <td>0.404210</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.669920</td>
          <td>0.429607</td>
          <td>26.289194</td>
          <td>0.115383</td>
          <td>26.053434</td>
          <td>0.082653</td>
          <td>25.887659</td>
          <td>0.116116</td>
          <td>25.715375</td>
          <td>0.187902</td>
          <td>25.593402</td>
          <td>0.360239</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.184853</td>
          <td>0.293807</td>
          <td>26.749870</td>
          <td>0.171520</td>
          <td>26.801722</td>
          <td>0.158508</td>
          <td>26.327922</td>
          <td>0.169673</td>
          <td>26.333002</td>
          <td>0.312483</td>
          <td>27.211823</td>
          <td>1.096800</td>
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
          <td>26.483062</td>
          <td>0.412498</td>
          <td>26.537342</td>
          <td>0.164408</td>
          <td>25.975367</td>
          <td>0.090739</td>
          <td>25.179803</td>
          <td>0.073814</td>
          <td>24.728694</td>
          <td>0.093960</td>
          <td>23.859109</td>
          <td>0.098691</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.144403</td>
          <td>0.272834</td>
          <td>26.607713</td>
          <td>0.157173</td>
          <td>26.143083</td>
          <td>0.170699</td>
          <td>25.625046</td>
          <td>0.203295</td>
          <td>25.066988</td>
          <td>0.275415</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.305591</td>
          <td>1.239772</td>
          <td>27.975672</td>
          <td>0.483611</td>
          <td>26.005071</td>
          <td>0.155139</td>
          <td>25.122946</td>
          <td>0.135424</td>
          <td>24.180404</td>
          <td>0.133537</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>26.558522</td>
          <td>0.456740</td>
          <td>28.566576</td>
          <td>0.819404</td>
          <td>26.854748</td>
          <td>0.206655</td>
          <td>26.234305</td>
          <td>0.197082</td>
          <td>25.498943</td>
          <td>0.194936</td>
          <td>25.123169</td>
          <td>0.306950</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.533551</td>
          <td>0.428776</td>
          <td>26.150281</td>
          <td>0.117850</td>
          <td>26.076613</td>
          <td>0.099202</td>
          <td>25.530794</td>
          <td>0.100565</td>
          <td>25.749116</td>
          <td>0.225500</td>
          <td>25.525647</td>
          <td>0.396228</td>
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
          <td>26.371651</td>
          <td>0.145339</td>
          <td>25.435253</td>
          <td>0.057503</td>
          <td>25.047949</td>
          <td>0.067143</td>
          <td>24.887998</td>
          <td>0.110299</td>
          <td>24.596881</td>
          <td>0.190345</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.977516</td>
          <td>2.748372</td>
          <td>26.647621</td>
          <td>0.181206</td>
          <td>25.928874</td>
          <td>0.087469</td>
          <td>25.254598</td>
          <td>0.079198</td>
          <td>24.689366</td>
          <td>0.091150</td>
          <td>24.294965</td>
          <td>0.144727</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.320898</td>
          <td>0.757564</td>
          <td>26.728720</td>
          <td>0.195491</td>
          <td>26.256242</td>
          <td>0.117494</td>
          <td>26.362342</td>
          <td>0.207980</td>
          <td>25.868252</td>
          <td>0.251743</td>
          <td>25.258926</td>
          <td>0.325199</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.782694</td>
          <td>0.241435</td>
          <td>26.310683</td>
          <td>0.139196</td>
          <td>26.195434</td>
          <td>0.113514</td>
          <td>26.055326</td>
          <td>0.163440</td>
          <td>25.819210</td>
          <td>0.246042</td>
          <td>25.172267</td>
          <td>0.308784</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.179749</td>
          <td>0.687890</td>
          <td>26.591647</td>
          <td>0.173697</td>
          <td>26.613395</td>
          <td>0.159458</td>
          <td>26.460776</td>
          <td>0.225163</td>
          <td>25.807637</td>
          <td>0.238872</td>
          <td>25.725609</td>
          <td>0.465269</td>
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
          <td>26.996542</td>
          <td>0.547378</td>
          <td>26.761467</td>
          <td>0.173237</td>
          <td>25.948286</td>
          <td>0.075336</td>
          <td>25.107417</td>
          <td>0.058406</td>
          <td>24.638829</td>
          <td>0.073827</td>
          <td>24.036189</td>
          <td>0.097575</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.705245</td>
          <td>0.374868</td>
          <td>26.748238</td>
          <td>0.151551</td>
          <td>26.424681</td>
          <td>0.184367</td>
          <td>25.683727</td>
          <td>0.183109</td>
          <td>25.318336</td>
          <td>0.289675</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.466371</td>
          <td>1.419476</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.390526</td>
          <td>1.139260</td>
          <td>26.060636</td>
          <td>0.146718</td>
          <td>25.066533</td>
          <td>0.116617</td>
          <td>24.422410</td>
          <td>0.148375</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>26.860296</td>
          <td>0.568614</td>
          <td>29.965694</td>
          <td>1.763272</td>
          <td>28.121916</td>
          <td>0.557031</td>
          <td>25.942758</td>
          <td>0.153315</td>
          <td>25.265188</td>
          <td>0.159324</td>
          <td>25.350783</td>
          <td>0.366337</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.251862</td>
          <td>0.310313</td>
          <td>26.174241</td>
          <td>0.104508</td>
          <td>25.845611</td>
          <td>0.068884</td>
          <td>25.694259</td>
          <td>0.098207</td>
          <td>25.610222</td>
          <td>0.172120</td>
          <td>25.004455</td>
          <td>0.224056</td>
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
          <td>26.734028</td>
          <td>0.472937</td>
          <td>26.219112</td>
          <td>0.116241</td>
          <td>25.411587</td>
          <td>0.050687</td>
          <td>25.037721</td>
          <td>0.059668</td>
          <td>24.752913</td>
          <td>0.088347</td>
          <td>24.835919</td>
          <td>0.210055</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.112926</td>
          <td>1.881209</td>
          <td>26.692119</td>
          <td>0.165577</td>
          <td>26.089368</td>
          <td>0.086734</td>
          <td>25.156495</td>
          <td>0.062073</td>
          <td>24.799959</td>
          <td>0.086515</td>
          <td>24.151597</td>
          <td>0.109782</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.460090</td>
          <td>0.376823</td>
          <td>27.001086</td>
          <td>0.220728</td>
          <td>26.382984</td>
          <td>0.115859</td>
          <td>26.058652</td>
          <td>0.141630</td>
          <td>26.081077</td>
          <td>0.266645</td>
          <td>25.313858</td>
          <td>0.302095</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.450806</td>
          <td>0.390205</td>
          <td>26.245442</td>
          <td>0.122746</td>
          <td>25.944401</td>
          <td>0.084244</td>
          <td>25.918761</td>
          <td>0.134270</td>
          <td>26.399056</td>
          <td>0.364972</td>
          <td>25.371391</td>
          <td>0.336381</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.945088</td>
          <td>0.539301</td>
          <td>26.761508</td>
          <td>0.178991</td>
          <td>26.504365</td>
          <td>0.127474</td>
          <td>26.194889</td>
          <td>0.157586</td>
          <td>25.872787</td>
          <td>0.222501</td>
          <td>25.932766</td>
          <td>0.483488</td>
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
