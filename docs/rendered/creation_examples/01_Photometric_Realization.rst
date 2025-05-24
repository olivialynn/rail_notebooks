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

    <pzflow.flow.Flow at 0x7fc9e15cd660>



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
    0      23.994413  0.120863  0.074396  
    1      25.391064  0.013962  0.008544  
    2      24.304707  0.033546  0.017045  
    3      25.291103  0.048072  0.041103  
    4      25.096743  0.109928  0.080023  
    ...          ...       ...       ...  
    99995  24.737946  0.006988  0.004270  
    99996  24.224169  0.060633  0.041273  
    99997  25.613836  0.026372  0.023981  
    99998  25.274899  0.008591  0.008404  
    99999  25.699642  0.329881  0.189104  
    
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
          <td>28.487575</td>
          <td>1.386601</td>
          <td>26.601525</td>
          <td>0.151120</td>
          <td>26.094000</td>
          <td>0.085661</td>
          <td>25.244900</td>
          <td>0.065971</td>
          <td>24.637526</td>
          <td>0.073732</td>
          <td>23.976811</td>
          <td>0.092607</td>
          <td>0.120863</td>
          <td>0.074396</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.571156</td>
          <td>0.398355</td>
          <td>27.243202</td>
          <td>0.258945</td>
          <td>26.628149</td>
          <td>0.136545</td>
          <td>26.053529</td>
          <td>0.134086</td>
          <td>25.733457</td>
          <td>0.190791</td>
          <td>25.331545</td>
          <td>0.292515</td>
          <td>0.013962</td>
          <td>0.008544</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.907620</td>
          <td>0.513030</td>
          <td>32.825409</td>
          <td>4.179925</td>
          <td>27.191622</td>
          <td>0.220313</td>
          <td>25.927247</td>
          <td>0.120185</td>
          <td>25.008771</td>
          <td>0.102227</td>
          <td>24.272663</td>
          <td>0.119937</td>
          <td>0.033546</td>
          <td>0.017045</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.917392</td>
          <td>0.884095</td>
          <td>27.299232</td>
          <td>0.240867</td>
          <td>26.220782</td>
          <td>0.154840</td>
          <td>25.325384</td>
          <td>0.134642</td>
          <td>25.341487</td>
          <td>0.294870</td>
          <td>0.048072</td>
          <td>0.041103</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.195461</td>
          <td>0.296326</td>
          <td>25.930354</td>
          <td>0.084281</td>
          <td>26.070714</td>
          <td>0.083921</td>
          <td>25.808962</td>
          <td>0.108415</td>
          <td>25.478428</td>
          <td>0.153594</td>
          <td>25.083431</td>
          <td>0.238874</td>
          <td>0.109928</td>
          <td>0.080023</td>
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
          <td>27.180601</td>
          <td>0.623912</td>
          <td>26.388481</td>
          <td>0.125769</td>
          <td>25.515418</td>
          <td>0.051320</td>
          <td>25.030222</td>
          <td>0.054530</td>
          <td>24.692082</td>
          <td>0.077374</td>
          <td>24.829630</td>
          <td>0.193288</td>
          <td>0.006988</td>
          <td>0.004270</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.468362</td>
          <td>0.367857</td>
          <td>26.848107</td>
          <td>0.186405</td>
          <td>26.089039</td>
          <td>0.085288</td>
          <td>25.225057</td>
          <td>0.064821</td>
          <td>24.897800</td>
          <td>0.092747</td>
          <td>24.132620</td>
          <td>0.106155</td>
          <td>0.060633</td>
          <td>0.041273</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.020933</td>
          <td>1.071534</td>
          <td>26.671883</td>
          <td>0.160496</td>
          <td>26.320259</td>
          <td>0.104483</td>
          <td>26.109123</td>
          <td>0.140675</td>
          <td>26.199414</td>
          <td>0.280609</td>
          <td>25.449122</td>
          <td>0.321430</td>
          <td>0.026372</td>
          <td>0.023981</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.706658</td>
          <td>0.441732</td>
          <td>26.305049</td>
          <td>0.116984</td>
          <td>26.301135</td>
          <td>0.102750</td>
          <td>25.797791</td>
          <td>0.107363</td>
          <td>25.941346</td>
          <td>0.227046</td>
          <td>25.626092</td>
          <td>0.369564</td>
          <td>0.008591</td>
          <td>0.008404</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.413409</td>
          <td>0.352374</td>
          <td>26.651995</td>
          <td>0.157792</td>
          <td>26.538949</td>
          <td>0.126405</td>
          <td>26.202880</td>
          <td>0.152482</td>
          <td>25.869420</td>
          <td>0.213850</td>
          <td>25.774870</td>
          <td>0.414591</td>
          <td>0.329881</td>
          <td>0.189104</td>
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
          <td>27.374277</td>
          <td>0.793773</td>
          <td>26.538357</td>
          <td>0.169370</td>
          <td>26.075212</td>
          <td>0.102354</td>
          <td>25.164078</td>
          <td>0.075335</td>
          <td>24.706650</td>
          <td>0.095241</td>
          <td>24.071949</td>
          <td>0.122879</td>
          <td>0.120863</td>
          <td>0.074396</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.657985</td>
          <td>0.933591</td>
          <td>27.426741</td>
          <td>0.342179</td>
          <td>27.163104</td>
          <td>0.250607</td>
          <td>26.158979</td>
          <td>0.173059</td>
          <td>25.717747</td>
          <td>0.219714</td>
          <td>25.172338</td>
          <td>0.299959</td>
          <td>0.013962</td>
          <td>0.008544</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.689517</td>
          <td>3.126191</td>
          <td>25.960784</td>
          <td>0.146379</td>
          <td>25.130255</td>
          <td>0.133638</td>
          <td>24.163084</td>
          <td>0.128926</td>
          <td>0.033546</td>
          <td>0.017045</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.475300</td>
          <td>0.740747</td>
          <td>27.367064</td>
          <td>0.297561</td>
          <td>26.340604</td>
          <td>0.203012</td>
          <td>25.671906</td>
          <td>0.212744</td>
          <td>25.060852</td>
          <td>0.275751</td>
          <td>0.048072</td>
          <td>0.041103</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.681776</td>
          <td>0.221953</td>
          <td>26.199465</td>
          <td>0.126337</td>
          <td>26.041972</td>
          <td>0.099170</td>
          <td>25.785261</td>
          <td>0.129446</td>
          <td>25.554163</td>
          <td>0.197151</td>
          <td>25.202980</td>
          <td>0.316161</td>
          <td>0.109928</td>
          <td>0.080023</td>
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
          <td>26.469560</td>
          <td>0.408277</td>
          <td>26.438046</td>
          <td>0.151043</td>
          <td>25.429398</td>
          <td>0.056007</td>
          <td>24.979110</td>
          <td>0.061804</td>
          <td>24.817360</td>
          <td>0.101562</td>
          <td>24.705835</td>
          <td>0.204359</td>
          <td>0.006988</td>
          <td>0.004270</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.212251</td>
          <td>0.336180</td>
          <td>26.748378</td>
          <td>0.198105</td>
          <td>26.019522</td>
          <td>0.095184</td>
          <td>25.279780</td>
          <td>0.081388</td>
          <td>24.991857</td>
          <td>0.119316</td>
          <td>24.244623</td>
          <td>0.139270</td>
          <td>0.060633</td>
          <td>0.041273</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.875776</td>
          <td>0.553122</td>
          <td>27.290454</td>
          <td>0.307462</td>
          <td>26.389626</td>
          <td>0.130526</td>
          <td>26.227375</td>
          <td>0.183703</td>
          <td>25.735045</td>
          <td>0.223259</td>
          <td>25.545989</td>
          <td>0.403138</td>
          <td>0.026372</td>
          <td>0.023981</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.765541</td>
          <td>0.509882</td>
          <td>26.263311</td>
          <td>0.129965</td>
          <td>25.930794</td>
          <td>0.087270</td>
          <td>25.873441</td>
          <td>0.135472</td>
          <td>25.550164</td>
          <td>0.190891</td>
          <td>25.753457</td>
          <td>0.470998</td>
          <td>0.008591</td>
          <td>0.008404</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.883065</td>
          <td>1.180824</td>
          <td>26.973683</td>
          <td>0.279457</td>
          <td>26.575389</td>
          <td>0.184032</td>
          <td>26.077614</td>
          <td>0.195184</td>
          <td>25.750761</td>
          <td>0.270030</td>
          <td>27.474908</td>
          <td>1.571632</td>
          <td>0.329881</td>
          <td>0.189104</td>
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
          <td>27.062968</td>
          <td>0.614861</td>
          <td>26.712357</td>
          <td>0.183706</td>
          <td>26.104518</td>
          <td>0.097298</td>
          <td>25.152657</td>
          <td>0.068845</td>
          <td>24.792017</td>
          <td>0.095097</td>
          <td>24.127316</td>
          <td>0.119273</td>
          <td>0.120863</td>
          <td>0.074396</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.152104</td>
          <td>0.612175</td>
          <td>27.386426</td>
          <td>0.291331</td>
          <td>26.432389</td>
          <td>0.115434</td>
          <td>25.963369</td>
          <td>0.124249</td>
          <td>26.060945</td>
          <td>0.251043</td>
          <td>25.779831</td>
          <td>0.416852</td>
          <td>0.013962</td>
          <td>0.008544</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.161895</td>
          <td>0.619035</td>
          <td>28.953055</td>
          <td>0.909406</td>
          <td>27.651960</td>
          <td>0.323463</td>
          <td>25.974230</td>
          <td>0.126427</td>
          <td>24.973342</td>
          <td>0.100046</td>
          <td>24.539391</td>
          <td>0.152473</td>
          <td>0.033546</td>
          <td>0.017045</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.061637</td>
          <td>0.582206</td>
          <td>29.060645</td>
          <td>0.981668</td>
          <td>27.185134</td>
          <td>0.224677</td>
          <td>26.305624</td>
          <td>0.171015</td>
          <td>25.761423</td>
          <td>0.200336</td>
          <td>25.237296</td>
          <td>0.277970</td>
          <td>0.048072</td>
          <td>0.041103</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.302872</td>
          <td>0.346825</td>
          <td>26.183557</td>
          <td>0.115917</td>
          <td>25.773233</td>
          <td>0.072133</td>
          <td>25.615342</td>
          <td>0.102689</td>
          <td>25.366538</td>
          <td>0.155429</td>
          <td>25.517263</td>
          <td>0.375759</td>
          <td>0.109928</td>
          <td>0.080023</td>
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
          <td>26.345712</td>
          <td>0.334159</td>
          <td>26.310995</td>
          <td>0.117637</td>
          <td>25.388020</td>
          <td>0.045853</td>
          <td>25.101415</td>
          <td>0.058116</td>
          <td>24.828387</td>
          <td>0.087295</td>
          <td>24.952123</td>
          <td>0.214295</td>
          <td>0.006988</td>
          <td>0.004270</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.783281</td>
          <td>1.629203</td>
          <td>26.793175</td>
          <td>0.183261</td>
          <td>26.224392</td>
          <td>0.099492</td>
          <td>25.155496</td>
          <td>0.063252</td>
          <td>24.820083</td>
          <td>0.089717</td>
          <td>24.136653</td>
          <td>0.110458</td>
          <td>0.060633</td>
          <td>0.041273</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.963292</td>
          <td>0.536934</td>
          <td>26.820167</td>
          <td>0.183361</td>
          <td>26.395731</td>
          <td>0.112555</td>
          <td>26.189272</td>
          <td>0.152041</td>
          <td>25.843477</td>
          <td>0.210972</td>
          <td>25.150966</td>
          <td>0.254638</td>
          <td>0.026372</td>
          <td>0.023981</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.609781</td>
          <td>0.410589</td>
          <td>26.220427</td>
          <td>0.108766</td>
          <td>26.023673</td>
          <td>0.080591</td>
          <td>25.982332</td>
          <td>0.126200</td>
          <td>26.103506</td>
          <td>0.259755</td>
          <td>25.298920</td>
          <td>0.285172</td>
          <td>0.008591</td>
          <td>0.008404</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.861548</td>
          <td>0.326842</td>
          <td>27.060940</td>
          <td>0.341478</td>
          <td>26.244327</td>
          <td>0.160821</td>
          <td>26.811091</td>
          <td>0.406676</td>
          <td>26.919364</td>
          <td>0.736098</td>
          <td>25.421316</td>
          <td>0.494255</td>
          <td>0.329881</td>
          <td>0.189104</td>
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
