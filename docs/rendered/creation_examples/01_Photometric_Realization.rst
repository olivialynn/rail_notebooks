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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.11.14/x64/lib/python3.11/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f6d97edf650>



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
    0      23.994413  0.034168  0.018738  
    1      25.391064  0.067197  0.065029  
    2      24.304707  0.050649  0.030007  
    3      25.291103  0.106078  0.085090  
    4      25.096743  0.048506  0.042381  
    ...          ...       ...       ...  
    99995  24.737946  0.002945  0.002488  
    99996  24.224169  0.058839  0.055760  
    99997  25.613836  0.107146  0.082949  
    99998  25.274899  0.062133  0.040149  
    99999  25.699642  0.023770  0.014319  
    
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

    Inserting handle into data store.  input: None, error_model
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
          <td>26.893970</td>
          <td>0.507915</td>
          <td>26.640811</td>
          <td>0.156290</td>
          <td>26.053912</td>
          <td>0.082688</td>
          <td>25.169583</td>
          <td>0.061709</td>
          <td>24.722388</td>
          <td>0.079472</td>
          <td>24.025727</td>
          <td>0.096670</td>
          <td>0.034168</td>
          <td>0.018738</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.552877</td>
          <td>2.243792</td>
          <td>27.133238</td>
          <td>0.236557</td>
          <td>26.499298</td>
          <td>0.122130</td>
          <td>26.088542</td>
          <td>0.138201</td>
          <td>25.793195</td>
          <td>0.200627</td>
          <td>25.511346</td>
          <td>0.337706</td>
          <td>0.067197</td>
          <td>0.065029</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.041880</td>
          <td>0.955175</td>
          <td>28.335897</td>
          <td>0.540372</td>
          <td>25.938962</td>
          <td>0.121414</td>
          <td>24.924790</td>
          <td>0.094972</td>
          <td>24.412907</td>
          <td>0.135433</td>
          <td>0.050649</td>
          <td>0.030007</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.055251</td>
          <td>0.570941</td>
          <td>28.406377</td>
          <td>0.629446</td>
          <td>27.131732</td>
          <td>0.209572</td>
          <td>26.323984</td>
          <td>0.169105</td>
          <td>25.322629</td>
          <td>0.134322</td>
          <td>25.050147</td>
          <td>0.232388</td>
          <td>0.106078</td>
          <td>0.085090</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.644758</td>
          <td>0.421460</td>
          <td>25.970955</td>
          <td>0.087344</td>
          <td>25.806646</td>
          <td>0.066452</td>
          <td>25.682183</td>
          <td>0.097028</td>
          <td>25.370482</td>
          <td>0.139985</td>
          <td>24.973984</td>
          <td>0.218140</td>
          <td>0.048506</td>
          <td>0.042381</td>
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
          <td>26.753951</td>
          <td>0.457748</td>
          <td>26.404594</td>
          <td>0.127536</td>
          <td>25.421434</td>
          <td>0.047211</td>
          <td>25.057359</td>
          <td>0.055860</td>
          <td>24.924907</td>
          <td>0.094982</td>
          <td>24.647494</td>
          <td>0.165638</td>
          <td>0.002945</td>
          <td>0.002488</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.137594</td>
          <td>0.282813</td>
          <td>26.794637</td>
          <td>0.178161</td>
          <td>26.057179</td>
          <td>0.082926</td>
          <td>25.277134</td>
          <td>0.067882</td>
          <td>24.791782</td>
          <td>0.084487</td>
          <td>24.349094</td>
          <td>0.128161</td>
          <td>0.058839</td>
          <td>0.055760</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>25.801035</td>
          <td>0.214508</td>
          <td>26.589197</td>
          <td>0.149531</td>
          <td>26.431249</td>
          <td>0.115112</td>
          <td>26.188253</td>
          <td>0.150581</td>
          <td>26.180395</td>
          <td>0.276310</td>
          <td>25.463433</td>
          <td>0.325112</td>
          <td>0.107146</td>
          <td>0.082949</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.973065</td>
          <td>0.247314</td>
          <td>26.350952</td>
          <td>0.121743</td>
          <td>26.095421</td>
          <td>0.085768</td>
          <td>25.998679</td>
          <td>0.127870</td>
          <td>25.848970</td>
          <td>0.210227</td>
          <td>25.216311</td>
          <td>0.266405</td>
          <td>0.062133</td>
          <td>0.040149</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.940721</td>
          <td>0.525599</td>
          <td>27.127421</td>
          <td>0.235423</td>
          <td>26.496341</td>
          <td>0.121816</td>
          <td>26.196156</td>
          <td>0.151606</td>
          <td>25.924884</td>
          <td>0.223962</td>
          <td>26.578232</td>
          <td>0.738519</td>
          <td>0.023770</td>
          <td>0.014319</td>
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
          <td>26.824156</td>
          <td>0.532974</td>
          <td>26.437943</td>
          <td>0.151358</td>
          <td>26.027280</td>
          <td>0.095214</td>
          <td>25.229668</td>
          <td>0.077343</td>
          <td>24.782489</td>
          <td>0.098750</td>
          <td>23.969642</td>
          <td>0.108990</td>
          <td>0.034168</td>
          <td>0.018738</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.224452</td>
          <td>0.294612</td>
          <td>26.849467</td>
          <td>0.195678</td>
          <td>26.332471</td>
          <td>0.203213</td>
          <td>25.728958</td>
          <td>0.224775</td>
          <td>25.282940</td>
          <td>0.332032</td>
          <td>0.067197</td>
          <td>0.065029</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.959405</td>
          <td>0.588580</td>
          <td>29.894899</td>
          <td>1.659889</td>
          <td>27.818053</td>
          <td>0.423497</td>
          <td>26.355688</td>
          <td>0.205411</td>
          <td>25.027717</td>
          <td>0.122701</td>
          <td>24.536556</td>
          <td>0.178188</td>
          <td>0.050649</td>
          <td>0.030007</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.495463</td>
          <td>0.763002</td>
          <td>27.438685</td>
          <td>0.322003</td>
          <td>26.615276</td>
          <td>0.260860</td>
          <td>25.444235</td>
          <td>0.179691</td>
          <td>25.200276</td>
          <td>0.315494</td>
          <td>0.106078</td>
          <td>0.085090</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.099417</td>
          <td>0.306886</td>
          <td>26.185504</td>
          <td>0.122233</td>
          <td>25.881057</td>
          <td>0.084101</td>
          <td>25.735318</td>
          <td>0.121032</td>
          <td>25.433074</td>
          <td>0.174014</td>
          <td>24.778069</td>
          <td>0.218544</td>
          <td>0.048506</td>
          <td>0.042381</td>
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
          <td>26.281494</td>
          <td>0.352836</td>
          <td>26.257829</td>
          <td>0.129325</td>
          <td>25.396862</td>
          <td>0.054408</td>
          <td>25.022501</td>
          <td>0.064221</td>
          <td>24.745364</td>
          <td>0.095344</td>
          <td>24.778817</td>
          <td>0.217195</td>
          <td>0.002945</td>
          <td>0.002488</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.442400</td>
          <td>0.402781</td>
          <td>26.472328</td>
          <td>0.157048</td>
          <td>26.104903</td>
          <td>0.102783</td>
          <td>25.170099</td>
          <td>0.074031</td>
          <td>24.704012</td>
          <td>0.092967</td>
          <td>24.225833</td>
          <td>0.137308</td>
          <td>0.058839</td>
          <td>0.055760</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.345830</td>
          <td>0.777905</td>
          <td>26.850912</td>
          <td>0.219812</td>
          <td>26.315382</td>
          <td>0.125837</td>
          <td>26.091562</td>
          <td>0.168365</td>
          <td>25.748570</td>
          <td>0.231844</td>
          <td>25.082199</td>
          <td>0.286873</td>
          <td>0.107146</td>
          <td>0.082949</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.018567</td>
          <td>0.288024</td>
          <td>26.505792</td>
          <td>0.161339</td>
          <td>25.982608</td>
          <td>0.092163</td>
          <td>26.075898</td>
          <td>0.162662</td>
          <td>25.706742</td>
          <td>0.219525</td>
          <td>25.919678</td>
          <td>0.536492</td>
          <td>0.062133</td>
          <td>0.040149</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.223120</td>
          <td>0.337281</td>
          <td>26.946794</td>
          <td>0.232203</td>
          <td>26.337081</td>
          <td>0.124613</td>
          <td>26.170977</td>
          <td>0.174981</td>
          <td>25.802549</td>
          <td>0.235920</td>
          <td>25.169052</td>
          <td>0.299407</td>
          <td>0.023770</td>
          <td>0.014319</td>
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
          <td>26.924089</td>
          <td>0.200409</td>
          <td>26.039551</td>
          <td>0.082487</td>
          <td>25.144141</td>
          <td>0.060988</td>
          <td>24.661435</td>
          <td>0.076083</td>
          <td>23.881125</td>
          <td>0.086039</td>
          <td>0.034168</td>
          <td>0.018738</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.318302</td>
          <td>1.298352</td>
          <td>27.293273</td>
          <td>0.282224</td>
          <td>26.296944</td>
          <td>0.108246</td>
          <td>26.681539</td>
          <td>0.241394</td>
          <td>26.479910</td>
          <td>0.369049</td>
          <td>24.896020</td>
          <td>0.215996</td>
          <td>0.067197</td>
          <td>0.065029</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.948608</td>
          <td>0.412704</td>
          <td>25.931480</td>
          <td>0.123524</td>
          <td>25.119816</td>
          <td>0.115223</td>
          <td>24.091054</td>
          <td>0.104817</td>
          <td>0.050649</td>
          <td>0.030007</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.413410</td>
          <td>0.777089</td>
          <td>28.378412</td>
          <td>0.666815</td>
          <td>27.725151</td>
          <td>0.375282</td>
          <td>26.196240</td>
          <td>0.169732</td>
          <td>26.003582</td>
          <td>0.265166</td>
          <td>25.673758</td>
          <td>0.424093</td>
          <td>0.106078</td>
          <td>0.085090</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.888503</td>
          <td>0.514021</td>
          <td>26.116649</td>
          <td>0.101641</td>
          <td>25.974871</td>
          <td>0.079271</td>
          <td>25.747562</td>
          <td>0.105732</td>
          <td>25.626870</td>
          <td>0.179000</td>
          <td>25.549447</td>
          <td>0.356976</td>
          <td>0.048506</td>
          <td>0.042381</td>
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
          <td>27.188961</td>
          <td>0.627606</td>
          <td>26.149999</td>
          <td>0.102200</td>
          <td>25.431194</td>
          <td>0.047627</td>
          <td>25.077244</td>
          <td>0.056861</td>
          <td>24.937722</td>
          <td>0.096065</td>
          <td>24.495815</td>
          <td>0.145479</td>
          <td>0.002945</td>
          <td>0.002488</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.237617</td>
          <td>0.314990</td>
          <td>26.610101</td>
          <td>0.157859</td>
          <td>25.826776</td>
          <td>0.070627</td>
          <td>25.200452</td>
          <td>0.066356</td>
          <td>24.968059</td>
          <td>0.102933</td>
          <td>24.223979</td>
          <td>0.120122</td>
          <td>0.058839</td>
          <td>0.055760</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.516168</td>
          <td>0.409195</td>
          <td>26.663981</td>
          <td>0.175135</td>
          <td>26.468787</td>
          <td>0.132609</td>
          <td>26.152424</td>
          <td>0.163368</td>
          <td>25.962242</td>
          <td>0.256144</td>
          <td>25.320396</td>
          <td>0.321701</td>
          <td>0.107146</td>
          <td>0.082949</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.239468</td>
          <td>0.314031</td>
          <td>26.234893</td>
          <td>0.113498</td>
          <td>26.161552</td>
          <td>0.094206</td>
          <td>25.981670</td>
          <td>0.130724</td>
          <td>25.510442</td>
          <td>0.163419</td>
          <td>26.091085</td>
          <td>0.541519</td>
          <td>0.062133</td>
          <td>0.040149</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.565850</td>
          <td>0.811291</td>
          <td>26.486228</td>
          <td>0.137465</td>
          <td>26.653615</td>
          <td>0.140292</td>
          <td>26.225116</td>
          <td>0.156245</td>
          <td>26.283526</td>
          <td>0.301758</td>
          <td>25.931177</td>
          <td>0.468807</td>
          <td>0.023770</td>
          <td>0.014319</td>
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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_24_0.png


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




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_25_0.png


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
