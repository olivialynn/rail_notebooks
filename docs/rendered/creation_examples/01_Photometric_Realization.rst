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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f388b3f66b0>



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
    0      23.994413  0.108286  0.084443  
    1      25.391064  0.189455  0.172073  
    2      24.304707  0.325755  0.209772  
    3      25.291103  0.180916  0.159800  
    4      25.096743  0.064801  0.040584  
    ...          ...       ...       ...  
    99995  24.737946  0.043661  0.028862  
    99996  24.224169  0.117940  0.083574  
    99997  25.613836  0.096824  0.082460  
    99998  25.274899  0.101130  0.070281  
    99999  25.699642  0.080992  0.055866  
    
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
          <td>29.021081</td>
          <td>1.795627</td>
          <td>26.997796</td>
          <td>0.211379</td>
          <td>25.990625</td>
          <td>0.078197</td>
          <td>25.165831</td>
          <td>0.061504</td>
          <td>24.611328</td>
          <td>0.072044</td>
          <td>23.950875</td>
          <td>0.090520</td>
          <td>0.108286</td>
          <td>0.084443</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.562229</td>
          <td>0.334811</td>
          <td>26.766490</td>
          <td>0.153799</td>
          <td>25.939564</td>
          <td>0.121478</td>
          <td>25.759555</td>
          <td>0.195033</td>
          <td>25.302789</td>
          <td>0.285798</td>
          <td>0.189455</td>
          <td>0.172073</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.758110</td>
          <td>0.914312</td>
          <td>28.195599</td>
          <td>0.541747</td>
          <td>28.077870</td>
          <td>0.446410</td>
          <td>25.854461</td>
          <td>0.112806</td>
          <td>25.005520</td>
          <td>0.101936</td>
          <td>24.298995</td>
          <td>0.122712</td>
          <td>0.325755</td>
          <td>0.209772</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.968293</td>
          <td>1.753146</td>
          <td>29.206347</td>
          <td>1.054381</td>
          <td>27.157071</td>
          <td>0.214057</td>
          <td>26.040432</td>
          <td>0.132576</td>
          <td>25.699710</td>
          <td>0.185432</td>
          <td>24.920874</td>
          <td>0.208677</td>
          <td>0.180916</td>
          <td>0.159800</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.019029</td>
          <td>0.256809</td>
          <td>26.274637</td>
          <td>0.113930</td>
          <td>25.956855</td>
          <td>0.075898</td>
          <td>25.800958</td>
          <td>0.107660</td>
          <td>25.506847</td>
          <td>0.157378</td>
          <td>24.945525</td>
          <td>0.213022</td>
          <td>0.064801</td>
          <td>0.040584</td>
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
          <td>26.865305</td>
          <td>0.497306</td>
          <td>26.193299</td>
          <td>0.106132</td>
          <td>25.444401</td>
          <td>0.048184</td>
          <td>25.115681</td>
          <td>0.058828</td>
          <td>24.845261</td>
          <td>0.088561</td>
          <td>24.736275</td>
          <td>0.178625</td>
          <td>0.043661</td>
          <td>0.028862</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>31.500404</td>
          <td>4.070121</td>
          <td>26.894727</td>
          <td>0.193877</td>
          <td>26.031113</td>
          <td>0.081042</td>
          <td>25.160161</td>
          <td>0.061196</td>
          <td>24.978606</td>
          <td>0.099562</td>
          <td>24.223482</td>
          <td>0.114913</td>
          <td>0.117940</td>
          <td>0.083574</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.288250</td>
          <td>0.319169</td>
          <td>26.940912</td>
          <td>0.201549</td>
          <td>26.406200</td>
          <td>0.112626</td>
          <td>26.208490</td>
          <td>0.153217</td>
          <td>26.130633</td>
          <td>0.265336</td>
          <td>25.731133</td>
          <td>0.400906</td>
          <td>0.096824</td>
          <td>0.082460</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.388382</td>
          <td>0.345508</td>
          <td>26.118740</td>
          <td>0.099435</td>
          <td>26.107925</td>
          <td>0.086718</td>
          <td>25.834325</td>
          <td>0.110842</td>
          <td>25.509014</td>
          <td>0.157670</td>
          <td>25.605918</td>
          <td>0.363786</td>
          <td>0.101130</td>
          <td>0.070281</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.996662</td>
          <td>0.547385</td>
          <td>26.659642</td>
          <td>0.158826</td>
          <td>26.598706</td>
          <td>0.133116</td>
          <td>26.487788</td>
          <td>0.194266</td>
          <td>25.868945</td>
          <td>0.213765</td>
          <td>25.480520</td>
          <td>0.329556</td>
          <td>0.080992</td>
          <td>0.055866</td>
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
          <td>27.237075</td>
          <td>0.724013</td>
          <td>26.545309</td>
          <td>0.170090</td>
          <td>25.914385</td>
          <td>0.088718</td>
          <td>25.124395</td>
          <td>0.072596</td>
          <td>24.642257</td>
          <td>0.089835</td>
          <td>23.850007</td>
          <td>0.101070</td>
          <td>0.108286</td>
          <td>0.084443</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.270106</td>
          <td>1.389869</td>
          <td>27.117832</td>
          <td>0.290401</td>
          <td>26.518735</td>
          <td>0.160554</td>
          <td>26.441168</td>
          <td>0.241824</td>
          <td>25.618306</td>
          <td>0.222298</td>
          <td>26.164400</td>
          <td>0.686962</td>
          <td>0.189455</td>
          <td>0.172073</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.420536</td>
          <td>0.762229</td>
          <td>25.824486</td>
          <td>0.158780</td>
          <td>24.986315</td>
          <td>0.143283</td>
          <td>24.082230</td>
          <td>0.146612</td>
          <td>0.325755</td>
          <td>0.209772</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.968584</td>
          <td>1.927048</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.366535</td>
          <td>0.320901</td>
          <td>26.474264</td>
          <td>0.246091</td>
          <td>25.449002</td>
          <td>0.191062</td>
          <td>25.100664</td>
          <td>0.307991</td>
          <td>0.180916</td>
          <td>0.159800</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.090483</td>
          <td>0.305299</td>
          <td>26.186866</td>
          <td>0.122687</td>
          <td>25.937818</td>
          <td>0.088660</td>
          <td>26.086201</td>
          <td>0.164201</td>
          <td>25.631234</td>
          <td>0.206231</td>
          <td>25.349745</td>
          <td>0.348469</td>
          <td>0.064801</td>
          <td>0.040584</td>
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
          <td>28.064216</td>
          <td>1.188281</td>
          <td>26.160817</td>
          <td>0.119389</td>
          <td>25.409125</td>
          <td>0.055261</td>
          <td>25.083354</td>
          <td>0.068105</td>
          <td>24.881482</td>
          <td>0.107903</td>
          <td>24.962921</td>
          <td>0.254046</td>
          <td>0.043661</td>
          <td>0.028862</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.610438</td>
          <td>0.923180</td>
          <td>26.621328</td>
          <td>0.181921</td>
          <td>26.033067</td>
          <td>0.098773</td>
          <td>25.074273</td>
          <td>0.069678</td>
          <td>24.747277</td>
          <td>0.098821</td>
          <td>24.145857</td>
          <td>0.131175</td>
          <td>0.117940</td>
          <td>0.083574</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.253452</td>
          <td>0.730176</td>
          <td>26.678520</td>
          <td>0.189685</td>
          <td>26.528615</td>
          <td>0.150729</td>
          <td>26.227683</td>
          <td>0.188307</td>
          <td>25.911905</td>
          <td>0.264319</td>
          <td>25.443096</td>
          <td>0.380699</td>
          <td>0.096824</td>
          <td>0.082460</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.107405</td>
          <td>0.312771</td>
          <td>26.299726</td>
          <td>0.137113</td>
          <td>26.081125</td>
          <td>0.102084</td>
          <td>25.705690</td>
          <td>0.120160</td>
          <td>25.774487</td>
          <td>0.235723</td>
          <td>25.326952</td>
          <td>0.347107</td>
          <td>0.101130</td>
          <td>0.070281</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.338017</td>
          <td>0.372836</td>
          <td>26.748838</td>
          <td>0.199404</td>
          <td>26.593411</td>
          <td>0.157693</td>
          <td>26.282945</td>
          <td>0.195223</td>
          <td>25.997722</td>
          <td>0.280670</td>
          <td>28.072964</td>
          <td>1.885289</td>
          <td>0.080992</td>
          <td>0.055866</td>
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
          <td>30.900801</td>
          <td>3.577719</td>
          <td>26.561485</td>
          <td>0.160877</td>
          <td>26.009556</td>
          <td>0.089061</td>
          <td>25.183945</td>
          <td>0.070408</td>
          <td>24.684991</td>
          <td>0.086122</td>
          <td>23.973088</td>
          <td>0.103732</td>
          <td>0.108286</td>
          <td>0.084443</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.777020</td>
          <td>1.087917</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.846632</td>
          <td>0.222598</td>
          <td>26.102631</td>
          <td>0.191881</td>
          <td>25.647744</td>
          <td>0.239295</td>
          <td>25.236959</td>
          <td>0.363373</td>
          <td>0.189455</td>
          <td>0.172073</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.089347</td>
          <td>0.233055</td>
          <td>25.315342</td>
          <td>0.221904</td>
          <td>24.476237</td>
          <td>0.239960</td>
          <td>0.325755</td>
          <td>0.209772</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>25.948557</td>
          <td>0.296152</td>
          <td>28.661240</td>
          <td>0.904975</td>
          <td>27.161905</td>
          <td>0.281358</td>
          <td>26.102340</td>
          <td>0.186907</td>
          <td>25.240061</td>
          <td>0.165711</td>
          <td>24.907375</td>
          <td>0.272553</td>
          <td>0.180916</td>
          <td>0.159800</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.654629</td>
          <td>0.194660</td>
          <td>26.188481</td>
          <td>0.109219</td>
          <td>25.994075</td>
          <td>0.081483</td>
          <td>25.655351</td>
          <td>0.098608</td>
          <td>25.620059</td>
          <td>0.179782</td>
          <td>25.259553</td>
          <td>0.286147</td>
          <td>0.064801</td>
          <td>0.040584</td>
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
          <td>27.118061</td>
          <td>0.603165</td>
          <td>26.259855</td>
          <td>0.114254</td>
          <td>25.403750</td>
          <td>0.047342</td>
          <td>25.009908</td>
          <td>0.054602</td>
          <td>24.916522</td>
          <td>0.096016</td>
          <td>24.675184</td>
          <td>0.172726</td>
          <td>0.043661</td>
          <td>0.028862</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.630626</td>
          <td>2.401192</td>
          <td>26.771309</td>
          <td>0.193880</td>
          <td>25.927116</td>
          <td>0.083665</td>
          <td>25.277391</td>
          <td>0.077277</td>
          <td>24.802709</td>
          <td>0.096473</td>
          <td>24.008917</td>
          <td>0.108138</td>
          <td>0.117940</td>
          <td>0.083574</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.680404</td>
          <td>0.460038</td>
          <td>26.408721</td>
          <td>0.139392</td>
          <td>26.420787</td>
          <td>0.125778</td>
          <td>26.296011</td>
          <td>0.182451</td>
          <td>25.920270</td>
          <td>0.244819</td>
          <td>27.296310</td>
          <td>1.228644</td>
          <td>0.096824</td>
          <td>0.082460</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.129799</td>
          <td>0.298597</td>
          <td>26.146305</td>
          <td>0.110435</td>
          <td>26.062016</td>
          <td>0.091371</td>
          <td>25.901891</td>
          <td>0.129342</td>
          <td>26.042957</td>
          <td>0.269122</td>
          <td>25.810800</td>
          <td>0.462647</td>
          <td>0.101130</td>
          <td>0.070281</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.006817</td>
          <td>1.093656</td>
          <td>26.996168</td>
          <td>0.222089</td>
          <td>26.558925</td>
          <td>0.136599</td>
          <td>26.311733</td>
          <td>0.178051</td>
          <td>25.940141</td>
          <td>0.240262</td>
          <td>25.251104</td>
          <td>0.290574</td>
          <td>0.080992</td>
          <td>0.055866</td>
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
