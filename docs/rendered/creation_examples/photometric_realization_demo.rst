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

    <pzflow.flow.Flow at 0x7f2ebd191c30>



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
          <td>26.269644</td>
          <td>0.314470</td>
          <td>26.653618</td>
          <td>0.158011</td>
          <td>26.036880</td>
          <td>0.081455</td>
          <td>25.187343</td>
          <td>0.062689</td>
          <td>24.691670</td>
          <td>0.077346</td>
          <td>23.917402</td>
          <td>0.087894</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.777167</td>
          <td>0.396060</td>
          <td>26.788115</td>
          <td>0.156674</td>
          <td>26.218417</td>
          <td>0.154526</td>
          <td>26.369727</td>
          <td>0.321778</td>
          <td>25.690480</td>
          <td>0.388523</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.448586</td>
          <td>1.358633</td>
          <td>29.606671</td>
          <td>1.319844</td>
          <td>27.596343</td>
          <td>0.306748</td>
          <td>25.912293</td>
          <td>0.118632</td>
          <td>25.126157</td>
          <td>0.113265</td>
          <td>24.502789</td>
          <td>0.146338</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>32.069372</td>
          <td>4.628605</td>
          <td>29.093228</td>
          <td>0.985506</td>
          <td>27.409762</td>
          <td>0.263748</td>
          <td>26.212667</td>
          <td>0.153767</td>
          <td>25.428061</td>
          <td>0.147097</td>
          <td>25.241766</td>
          <td>0.271990</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.346804</td>
          <td>0.334353</td>
          <td>26.145879</td>
          <td>0.101824</td>
          <td>25.938869</td>
          <td>0.074701</td>
          <td>25.765593</td>
          <td>0.104383</td>
          <td>25.341483</td>
          <td>0.136527</td>
          <td>24.554659</td>
          <td>0.153001</td>
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
          <td>26.462661</td>
          <td>0.366224</td>
          <td>26.535565</td>
          <td>0.142798</td>
          <td>25.425128</td>
          <td>0.047366</td>
          <td>25.083427</td>
          <td>0.057168</td>
          <td>24.854830</td>
          <td>0.089309</td>
          <td>24.742886</td>
          <td>0.179628</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.480008</td>
          <td>0.371210</td>
          <td>26.760078</td>
          <td>0.173014</td>
          <td>26.059929</td>
          <td>0.083127</td>
          <td>25.221100</td>
          <td>0.064594</td>
          <td>24.834948</td>
          <td>0.087761</td>
          <td>24.029599</td>
          <td>0.096999</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.766388</td>
          <td>0.462037</td>
          <td>26.579980</td>
          <td>0.148353</td>
          <td>26.317575</td>
          <td>0.104238</td>
          <td>26.199340</td>
          <td>0.152020</td>
          <td>25.875710</td>
          <td>0.214975</td>
          <td>25.079014</td>
          <td>0.238004</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.760916</td>
          <td>0.460146</td>
          <td>26.024949</td>
          <td>0.091587</td>
          <td>26.245783</td>
          <td>0.097886</td>
          <td>25.713186</td>
          <td>0.099702</td>
          <td>25.692356</td>
          <td>0.184282</td>
          <td>25.283933</td>
          <td>0.281467</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.343335</td>
          <td>0.333437</td>
          <td>26.488041</td>
          <td>0.137071</td>
          <td>26.287568</td>
          <td>0.101536</td>
          <td>26.492228</td>
          <td>0.194994</td>
          <td>25.561070</td>
          <td>0.164838</td>
          <td>25.994526</td>
          <td>0.489201</td>
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
          <td>27.720541</td>
          <td>0.969875</td>
          <td>26.985684</td>
          <td>0.239527</td>
          <td>26.415133</td>
          <td>0.133157</td>
          <td>25.208748</td>
          <td>0.075726</td>
          <td>24.563793</td>
          <td>0.081272</td>
          <td>24.017288</td>
          <td>0.113319</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.321444</td>
          <td>0.314681</td>
          <td>26.837620</td>
          <td>0.191072</td>
          <td>26.122353</td>
          <td>0.167713</td>
          <td>26.247852</td>
          <td>0.337990</td>
          <td>26.164805</td>
          <td>0.634049</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>30.004296</td>
          <td>2.787604</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.079376</td>
          <td>0.522019</td>
          <td>26.355608</td>
          <td>0.208758</td>
          <td>25.247599</td>
          <td>0.150759</td>
          <td>24.654150</td>
          <td>0.199998</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.628859</td>
          <td>1.636913</td>
          <td>29.008208</td>
          <td>1.075367</td>
          <td>27.409307</td>
          <td>0.325185</td>
          <td>26.139407</td>
          <td>0.181919</td>
          <td>25.520012</td>
          <td>0.198420</td>
          <td>24.920360</td>
          <td>0.260457</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.679576</td>
          <td>0.216768</td>
          <td>25.999760</td>
          <td>0.103366</td>
          <td>25.899258</td>
          <td>0.084890</td>
          <td>25.612830</td>
          <td>0.108044</td>
          <td>25.625746</td>
          <td>0.203437</td>
          <td>25.254762</td>
          <td>0.320387</td>
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
          <td>26.554382</td>
          <td>0.169906</td>
          <td>25.450137</td>
          <td>0.058268</td>
          <td>25.093770</td>
          <td>0.069922</td>
          <td>24.846177</td>
          <td>0.106345</td>
          <td>24.568954</td>
          <td>0.185910</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.163065</td>
          <td>1.254750</td>
          <td>26.657075</td>
          <td>0.182661</td>
          <td>26.094313</td>
          <td>0.101140</td>
          <td>25.149095</td>
          <td>0.072151</td>
          <td>24.820878</td>
          <td>0.102291</td>
          <td>24.322190</td>
          <td>0.148153</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.937615</td>
          <td>1.110286</td>
          <td>26.450770</td>
          <td>0.154416</td>
          <td>26.360812</td>
          <td>0.128655</td>
          <td>26.311159</td>
          <td>0.199243</td>
          <td>26.500749</td>
          <td>0.416088</td>
          <td>25.075373</td>
          <td>0.280625</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.213588</td>
          <td>0.341765</td>
          <td>26.058806</td>
          <td>0.111915</td>
          <td>26.074208</td>
          <td>0.102113</td>
          <td>26.014368</td>
          <td>0.157820</td>
          <td>25.891004</td>
          <td>0.260970</td>
          <td>25.161990</td>
          <td>0.306252</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.125651</td>
          <td>0.314052</td>
          <td>26.584431</td>
          <td>0.172636</td>
          <td>26.450275</td>
          <td>0.138620</td>
          <td>26.424460</td>
          <td>0.218462</td>
          <td>25.938483</td>
          <td>0.265958</td>
          <td>25.866581</td>
          <td>0.516480</td>
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
          <td>27.621287</td>
          <td>0.838652</td>
          <td>26.624826</td>
          <td>0.154184</td>
          <td>25.935138</td>
          <td>0.074465</td>
          <td>25.167455</td>
          <td>0.061601</td>
          <td>24.640765</td>
          <td>0.073954</td>
          <td>24.100509</td>
          <td>0.103230</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.702807</td>
          <td>0.883610</td>
          <td>27.512642</td>
          <td>0.322124</td>
          <td>26.980827</td>
          <td>0.184759</td>
          <td>26.231115</td>
          <td>0.156367</td>
          <td>25.552647</td>
          <td>0.163809</td>
          <td>26.276070</td>
          <td>0.600379</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.878632</td>
          <td>0.455387</td>
          <td>28.824579</td>
          <td>0.806542</td>
          <td>25.869384</td>
          <td>0.124380</td>
          <td>25.079626</td>
          <td>0.117953</td>
          <td>24.449898</td>
          <td>0.151916</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>28.634675</td>
          <td>1.638969</td>
          <td>28.012180</td>
          <td>0.559507</td>
          <td>27.582534</td>
          <td>0.371520</td>
          <td>26.808403</td>
          <td>0.314687</td>
          <td>25.964406</td>
          <td>0.285389</td>
          <td>25.641262</td>
          <td>0.457692</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.341906</td>
          <td>0.333355</td>
          <td>26.095352</td>
          <td>0.097539</td>
          <td>26.010778</td>
          <td>0.079714</td>
          <td>25.913104</td>
          <td>0.118891</td>
          <td>25.381284</td>
          <td>0.141491</td>
          <td>24.860301</td>
          <td>0.198621</td>
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
          <td>27.870096</td>
          <td>1.017601</td>
          <td>26.328185</td>
          <td>0.127777</td>
          <td>25.437295</td>
          <td>0.051857</td>
          <td>25.105406</td>
          <td>0.063359</td>
          <td>24.840207</td>
          <td>0.095390</td>
          <td>24.987317</td>
          <td>0.238223</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.780763</td>
          <td>0.934770</td>
          <td>26.991782</td>
          <td>0.213203</td>
          <td>25.917693</td>
          <td>0.074543</td>
          <td>25.304282</td>
          <td>0.070757</td>
          <td>24.702054</td>
          <td>0.079362</td>
          <td>24.290748</td>
          <td>0.123915</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.513452</td>
          <td>0.392717</td>
          <td>26.780291</td>
          <td>0.183408</td>
          <td>26.367035</td>
          <td>0.114261</td>
          <td>26.329778</td>
          <td>0.178571</td>
          <td>26.234556</td>
          <td>0.301928</td>
          <td>25.369433</td>
          <td>0.315846</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.403203</td>
          <td>0.376077</td>
          <td>26.151174</td>
          <td>0.113094</td>
          <td>26.010868</td>
          <td>0.089320</td>
          <td>25.961351</td>
          <td>0.139297</td>
          <td>25.511748</td>
          <td>0.176603</td>
          <td>25.019993</td>
          <td>0.253372</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.455150</td>
          <td>0.768024</td>
          <td>26.645006</td>
          <td>0.162112</td>
          <td>26.456795</td>
          <td>0.122321</td>
          <td>26.228652</td>
          <td>0.162199</td>
          <td>26.059200</td>
          <td>0.259499</td>
          <td>26.025431</td>
          <td>0.517692</td>
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
