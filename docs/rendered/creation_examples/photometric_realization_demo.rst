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

    <pzflow.flow.Flow at 0x7f930b4b09a0>



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
          <td>27.713117</td>
          <td>0.888944</td>
          <td>26.429579</td>
          <td>0.130323</td>
          <td>26.033408</td>
          <td>0.081206</td>
          <td>25.193558</td>
          <td>0.063035</td>
          <td>24.733088</td>
          <td>0.080226</td>
          <td>23.890580</td>
          <td>0.085843</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.016832</td>
          <td>1.068964</td>
          <td>27.258261</td>
          <td>0.262153</td>
          <td>26.643389</td>
          <td>0.138353</td>
          <td>26.075131</td>
          <td>0.136611</td>
          <td>25.663613</td>
          <td>0.179853</td>
          <td>25.155203</td>
          <td>0.253412</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.211931</td>
          <td>0.637706</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.933860</td>
          <td>0.399985</td>
          <td>26.139065</td>
          <td>0.144349</td>
          <td>24.903689</td>
          <td>0.093228</td>
          <td>24.461822</td>
          <td>0.141269</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.108110</td>
          <td>0.592848</td>
          <td>27.703575</td>
          <td>0.374109</td>
          <td>27.006910</td>
          <td>0.188703</td>
          <td>26.388997</td>
          <td>0.178708</td>
          <td>25.664505</td>
          <td>0.179989</td>
          <td>26.008415</td>
          <td>0.494258</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.683461</td>
          <td>0.434044</td>
          <td>26.140452</td>
          <td>0.101342</td>
          <td>26.004134</td>
          <td>0.079135</td>
          <td>25.609175</td>
          <td>0.091002</td>
          <td>25.199771</td>
          <td>0.120758</td>
          <td>24.876907</td>
          <td>0.201127</td>
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
          <td>28.638428</td>
          <td>1.497427</td>
          <td>26.185944</td>
          <td>0.105452</td>
          <td>25.460244</td>
          <td>0.048866</td>
          <td>25.149948</td>
          <td>0.060644</td>
          <td>24.958078</td>
          <td>0.097786</td>
          <td>25.073647</td>
          <td>0.236950</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.618863</td>
          <td>0.837294</td>
          <td>26.481239</td>
          <td>0.136269</td>
          <td>26.032108</td>
          <td>0.081113</td>
          <td>25.151147</td>
          <td>0.060708</td>
          <td>24.720550</td>
          <td>0.079343</td>
          <td>24.388470</td>
          <td>0.132603</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>27.476651</td>
          <td>0.763216</td>
          <td>26.814889</td>
          <td>0.181243</td>
          <td>26.502920</td>
          <td>0.122514</td>
          <td>26.263536</td>
          <td>0.160608</td>
          <td>25.978869</td>
          <td>0.234217</td>
          <td>25.655481</td>
          <td>0.378119</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>26.129545</td>
          <td>0.280977</td>
          <td>26.286699</td>
          <td>0.115132</td>
          <td>26.060791</td>
          <td>0.083191</td>
          <td>25.805899</td>
          <td>0.108126</td>
          <td>25.669087</td>
          <td>0.180689</td>
          <td>25.752461</td>
          <td>0.407533</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.005858</td>
          <td>0.551032</td>
          <td>26.691259</td>
          <td>0.163171</td>
          <td>26.399891</td>
          <td>0.112008</td>
          <td>26.198563</td>
          <td>0.151919</td>
          <td>26.214708</td>
          <td>0.284107</td>
          <td>25.935420</td>
          <td>0.468140</td>
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
          <td>26.748706</td>
          <td>0.503543</td>
          <td>26.849120</td>
          <td>0.213868</td>
          <td>26.087798</td>
          <td>0.100147</td>
          <td>25.265573</td>
          <td>0.079622</td>
          <td>24.769509</td>
          <td>0.097385</td>
          <td>24.052699</td>
          <td>0.116867</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.460948</td>
          <td>0.351460</td>
          <td>26.538033</td>
          <td>0.148061</td>
          <td>26.703079</td>
          <td>0.272212</td>
          <td>25.667986</td>
          <td>0.210736</td>
          <td>24.870706</td>
          <td>0.234465</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>28.502173</td>
          <td>1.509805</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.158767</td>
          <td>0.552999</td>
          <td>26.346178</td>
          <td>0.207117</td>
          <td>24.924709</td>
          <td>0.114033</td>
          <td>24.410507</td>
          <td>0.162715</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.597635</td>
          <td>0.412308</td>
          <td>26.841976</td>
          <td>0.204455</td>
          <td>26.393159</td>
          <td>0.225066</td>
          <td>25.378237</td>
          <td>0.176032</td>
          <td>24.912319</td>
          <td>0.258749</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.926695</td>
          <td>0.265688</td>
          <td>26.320079</td>
          <td>0.136507</td>
          <td>25.965150</td>
          <td>0.089957</td>
          <td>25.525814</td>
          <td>0.100127</td>
          <td>25.316683</td>
          <td>0.156561</td>
          <td>24.950837</td>
          <td>0.250506</td>
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
          <td>26.927716</td>
          <td>0.580884</td>
          <td>26.473564</td>
          <td>0.158598</td>
          <td>25.501963</td>
          <td>0.061007</td>
          <td>25.087281</td>
          <td>0.069522</td>
          <td>24.859103</td>
          <td>0.107553</td>
          <td>24.852390</td>
          <td>0.235630</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.768142</td>
          <td>1.702849</td>
          <td>26.786349</td>
          <td>0.203660</td>
          <td>26.178027</td>
          <td>0.108819</td>
          <td>25.220920</td>
          <td>0.076878</td>
          <td>24.792184</td>
          <td>0.099754</td>
          <td>24.156843</td>
          <td>0.128465</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.698534</td>
          <td>0.190586</td>
          <td>26.630766</td>
          <td>0.162274</td>
          <td>25.878640</td>
          <td>0.137832</td>
          <td>26.096878</td>
          <td>0.303109</td>
          <td>25.171753</td>
          <td>0.303318</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.945745</td>
          <td>0.275845</td>
          <td>26.043668</td>
          <td>0.110449</td>
          <td>25.917612</td>
          <td>0.089005</td>
          <td>25.792691</td>
          <td>0.130416</td>
          <td>26.018225</td>
          <td>0.289400</td>
          <td>25.748058</td>
          <td>0.482020</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.534841</td>
          <td>0.431947</td>
          <td>26.521695</td>
          <td>0.163663</td>
          <td>26.610256</td>
          <td>0.159030</td>
          <td>26.698388</td>
          <td>0.273741</td>
          <td>26.213114</td>
          <td>0.331741</td>
          <td>29.006588</td>
          <td>2.696004</td>
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
          <td>27.094823</td>
          <td>0.587325</td>
          <td>26.819278</td>
          <td>0.181938</td>
          <td>25.978802</td>
          <td>0.077395</td>
          <td>25.206471</td>
          <td>0.063770</td>
          <td>24.872892</td>
          <td>0.090751</td>
          <td>23.992408</td>
          <td>0.093897</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.481871</td>
          <td>0.766230</td>
          <td>27.680090</td>
          <td>0.367590</td>
          <td>26.580014</td>
          <td>0.131103</td>
          <td>26.621636</td>
          <td>0.217528</td>
          <td>25.987186</td>
          <td>0.236044</td>
          <td>25.214501</td>
          <td>0.266256</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495130</td>
          <td>27.970083</td>
          <td>1.081404</td>
          <td>30.111581</td>
          <td>1.763003</td>
          <td>28.501203</td>
          <td>0.649073</td>
          <td>26.364283</td>
          <td>0.190026</td>
          <td>24.961231</td>
          <td>0.106385</td>
          <td>24.358115</td>
          <td>0.140391</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842595</td>
          <td>27.885081</td>
          <td>1.105247</td>
          <td>28.966121</td>
          <td>1.046816</td>
          <td>27.623454</td>
          <td>0.383528</td>
          <td>26.541772</td>
          <td>0.253568</td>
          <td>25.475841</td>
          <td>0.190526</td>
          <td>25.735024</td>
          <td>0.490847</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.086933</td>
          <td>0.584461</td>
          <td>26.067604</td>
          <td>0.095196</td>
          <td>26.003972</td>
          <td>0.079237</td>
          <td>25.755412</td>
          <td>0.103611</td>
          <td>25.283975</td>
          <td>0.130088</td>
          <td>24.985263</td>
          <td>0.220507</td>
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
          <td>26.271035</td>
          <td>0.121603</td>
          <td>25.440462</td>
          <td>0.052003</td>
          <td>25.094186</td>
          <td>0.062732</td>
          <td>24.798054</td>
          <td>0.091924</td>
          <td>24.902074</td>
          <td>0.221970</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.064726</td>
          <td>0.580203</td>
          <td>26.669873</td>
          <td>0.162466</td>
          <td>25.856902</td>
          <td>0.070641</td>
          <td>25.248296</td>
          <td>0.067335</td>
          <td>25.051499</td>
          <td>0.107872</td>
          <td>24.144650</td>
          <td>0.109118</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023549</td>
          <td>26.384305</td>
          <td>0.355180</td>
          <td>26.593232</td>
          <td>0.156433</td>
          <td>26.459166</td>
          <td>0.123791</td>
          <td>26.364266</td>
          <td>0.183863</td>
          <td>25.911211</td>
          <td>0.231889</td>
          <td>25.794864</td>
          <td>0.439847</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548203</td>
          <td>25.964499</td>
          <td>0.265139</td>
          <td>26.179744</td>
          <td>0.115940</td>
          <td>25.968548</td>
          <td>0.086055</td>
          <td>25.835892</td>
          <td>0.124976</td>
          <td>25.568689</td>
          <td>0.185327</td>
          <td>24.848151</td>
          <td>0.219819</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.488814</td>
          <td>0.382880</td>
          <td>26.638771</td>
          <td>0.161252</td>
          <td>26.430334</td>
          <td>0.119541</td>
          <td>26.465798</td>
          <td>0.198300</td>
          <td>26.746650</td>
          <td>0.446223</td>
          <td>25.362553</td>
          <td>0.311171</td>
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
