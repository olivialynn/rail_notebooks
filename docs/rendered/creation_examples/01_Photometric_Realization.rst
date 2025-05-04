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

    <pzflow.flow.Flow at 0x7f1bceae3e20>



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
          <td>27.794899</td>
          <td>0.935395</td>
          <td>26.702416</td>
          <td>0.164731</td>
          <td>26.016653</td>
          <td>0.080014</td>
          <td>25.094903</td>
          <td>0.057753</td>
          <td>24.641610</td>
          <td>0.073999</td>
          <td>24.038555</td>
          <td>0.097764</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.081156</td>
          <td>0.226572</td>
          <td>26.528357</td>
          <td>0.125249</td>
          <td>26.203465</td>
          <td>0.152559</td>
          <td>25.711045</td>
          <td>0.187216</td>
          <td>26.139913</td>
          <td>0.544193</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.840693</td>
          <td>0.415859</td>
          <td>27.589069</td>
          <td>0.304963</td>
          <td>26.061013</td>
          <td>0.134955</td>
          <td>25.095406</td>
          <td>0.110268</td>
          <td>24.300253</td>
          <td>0.122846</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.443478</td>
          <td>0.271102</td>
          <td>26.262375</td>
          <td>0.160449</td>
          <td>25.235866</td>
          <td>0.124602</td>
          <td>25.913879</td>
          <td>0.460647</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.723146</td>
          <td>0.447263</td>
          <td>26.186117</td>
          <td>0.105468</td>
          <td>25.961718</td>
          <td>0.076225</td>
          <td>25.762599</td>
          <td>0.104110</td>
          <td>25.389330</td>
          <td>0.142277</td>
          <td>24.860170</td>
          <td>0.198319</td>
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
          <td>26.279905</td>
          <td>0.317054</td>
          <td>26.208063</td>
          <td>0.107508</td>
          <td>25.505296</td>
          <td>0.050861</td>
          <td>25.054275</td>
          <td>0.055707</td>
          <td>24.693186</td>
          <td>0.077450</td>
          <td>24.657135</td>
          <td>0.167005</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.750018</td>
          <td>0.456399</td>
          <td>26.631462</td>
          <td>0.155045</td>
          <td>25.972672</td>
          <td>0.076967</td>
          <td>25.180561</td>
          <td>0.062313</td>
          <td>24.642471</td>
          <td>0.074056</td>
          <td>24.071447</td>
          <td>0.100623</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.284089</td>
          <td>0.670325</td>
          <td>26.727927</td>
          <td>0.168349</td>
          <td>26.359821</td>
          <td>0.108159</td>
          <td>26.599386</td>
          <td>0.213326</td>
          <td>26.204968</td>
          <td>0.281875</td>
          <td>26.362750</td>
          <td>0.637537</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.608367</td>
          <td>0.409902</td>
          <td>26.287245</td>
          <td>0.115187</td>
          <td>26.164122</td>
          <td>0.091113</td>
          <td>25.937334</td>
          <td>0.121243</td>
          <td>25.975905</td>
          <td>0.233644</td>
          <td>24.954291</td>
          <td>0.214587</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.736181</td>
          <td>0.451676</td>
          <td>26.612360</td>
          <td>0.152530</td>
          <td>26.543221</td>
          <td>0.126874</td>
          <td>26.341811</td>
          <td>0.171690</td>
          <td>25.605468</td>
          <td>0.171191</td>
          <td>25.757430</td>
          <td>0.409089</td>
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
          <td>25.902883</td>
          <td>0.260519</td>
          <td>26.508666</td>
          <td>0.160436</td>
          <td>26.068937</td>
          <td>0.098506</td>
          <td>25.282025</td>
          <td>0.080786</td>
          <td>24.696054</td>
          <td>0.091305</td>
          <td>23.891302</td>
          <td>0.101513</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.372633</td>
          <td>0.327774</td>
          <td>26.765703</td>
          <td>0.179804</td>
          <td>26.239172</td>
          <td>0.185189</td>
          <td>26.156187</td>
          <td>0.314236</td>
          <td>27.161397</td>
          <td>1.187617</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.482367</td>
          <td>0.418439</td>
          <td>27.750587</td>
          <td>0.447102</td>
          <td>28.690335</td>
          <td>0.797008</td>
          <td>26.047935</td>
          <td>0.160931</td>
          <td>25.192933</td>
          <td>0.143843</td>
          <td>24.253711</td>
          <td>0.142251</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.662493</td>
          <td>0.970383</td>
          <td>28.316213</td>
          <td>0.694005</td>
          <td>27.273699</td>
          <td>0.291709</td>
          <td>26.310408</td>
          <td>0.210068</td>
          <td>25.831391</td>
          <td>0.256945</td>
          <td>25.492305</td>
          <td>0.410061</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.052137</td>
          <td>0.294094</td>
          <td>26.148067</td>
          <td>0.117624</td>
          <td>26.112575</td>
          <td>0.102376</td>
          <td>25.714586</td>
          <td>0.118062</td>
          <td>25.590108</td>
          <td>0.197440</td>
          <td>24.532875</td>
          <td>0.176665</td>
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
          <td>28.234486</td>
          <td>1.314405</td>
          <td>26.524872</td>
          <td>0.165693</td>
          <td>25.382005</td>
          <td>0.054851</td>
          <td>25.070018</td>
          <td>0.068468</td>
          <td>24.891321</td>
          <td>0.110619</td>
          <td>24.694671</td>
          <td>0.206647</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.089860</td>
          <td>0.644390</td>
          <td>26.418332</td>
          <td>0.149048</td>
          <td>25.880796</td>
          <td>0.083844</td>
          <td>25.213978</td>
          <td>0.076408</td>
          <td>24.904079</td>
          <td>0.110004</td>
          <td>23.931817</td>
          <td>0.105625</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.548838</td>
          <td>0.437345</td>
          <td>26.599963</td>
          <td>0.175347</td>
          <td>26.551555</td>
          <td>0.151641</td>
          <td>26.105854</td>
          <td>0.167474</td>
          <td>26.420314</td>
          <td>0.391138</td>
          <td>25.311476</td>
          <td>0.339034</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.511768</td>
          <td>0.430503</td>
          <td>26.336585</td>
          <td>0.142334</td>
          <td>26.150753</td>
          <td>0.109177</td>
          <td>25.856877</td>
          <td>0.137852</td>
          <td>25.571678</td>
          <td>0.200268</td>
          <td>25.532755</td>
          <td>0.409683</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.552141</td>
          <td>0.437647</td>
          <td>26.930811</td>
          <td>0.230864</td>
          <td>26.947364</td>
          <td>0.211487</td>
          <td>26.245979</td>
          <td>0.188088</td>
          <td>25.700366</td>
          <td>0.218537</td>
          <td>25.270461</td>
          <td>0.327358</td>
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
          <td>28.691072</td>
          <td>1.537124</td>
          <td>26.626640</td>
          <td>0.154423</td>
          <td>26.067951</td>
          <td>0.083728</td>
          <td>25.151688</td>
          <td>0.060746</td>
          <td>24.687481</td>
          <td>0.077070</td>
          <td>23.932196</td>
          <td>0.089057</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.484244</td>
          <td>0.314912</td>
          <td>26.690648</td>
          <td>0.144236</td>
          <td>26.039704</td>
          <td>0.132622</td>
          <td>25.641190</td>
          <td>0.176627</td>
          <td>25.850496</td>
          <td>0.439530</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.270346</td>
          <td>0.694961</td>
          <td>28.245337</td>
          <td>0.595310</td>
          <td>27.638192</td>
          <td>0.341521</td>
          <td>26.043130</td>
          <td>0.144526</td>
          <td>24.999559</td>
          <td>0.110006</td>
          <td>24.205229</td>
          <td>0.123003</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.356857</td>
          <td>0.798778</td>
          <td>28.026867</td>
          <td>0.565441</td>
          <td>27.357255</td>
          <td>0.310950</td>
          <td>26.812652</td>
          <td>0.315757</td>
          <td>25.684168</td>
          <td>0.226808</td>
          <td>24.528139</td>
          <td>0.187310</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.619246</td>
          <td>0.413683</td>
          <td>26.155109</td>
          <td>0.102775</td>
          <td>26.029410</td>
          <td>0.081035</td>
          <td>25.959753</td>
          <td>0.123808</td>
          <td>25.272777</td>
          <td>0.128833</td>
          <td>24.795300</td>
          <td>0.188038</td>
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
          <td>26.346630</td>
          <td>0.129832</td>
          <td>25.447313</td>
          <td>0.052320</td>
          <td>25.036676</td>
          <td>0.059612</td>
          <td>24.912281</td>
          <td>0.101610</td>
          <td>24.402232</td>
          <td>0.145374</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.351067</td>
          <td>1.299027</td>
          <td>26.765204</td>
          <td>0.176190</td>
          <td>25.936357</td>
          <td>0.075783</td>
          <td>25.251809</td>
          <td>0.067545</td>
          <td>24.692342</td>
          <td>0.078684</td>
          <td>24.036657</td>
          <td>0.099283</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.357852</td>
          <td>1.322077</td>
          <td>26.818119</td>
          <td>0.189361</td>
          <td>26.480485</td>
          <td>0.126101</td>
          <td>26.118835</td>
          <td>0.149154</td>
          <td>26.008571</td>
          <td>0.251280</td>
          <td>26.491488</td>
          <td>0.724282</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.800978</td>
          <td>0.508106</td>
          <td>26.214124</td>
          <td>0.119455</td>
          <td>26.229971</td>
          <td>0.108229</td>
          <td>25.878710</td>
          <td>0.129699</td>
          <td>25.470765</td>
          <td>0.170560</td>
          <td>24.737713</td>
          <td>0.200428</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.729719</td>
          <td>0.460088</td>
          <td>26.600222</td>
          <td>0.156027</td>
          <td>26.616925</td>
          <td>0.140496</td>
          <td>26.406786</td>
          <td>0.188684</td>
          <td>25.607426</td>
          <td>0.178040</td>
          <td>25.136215</td>
          <td>0.259073</td>
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
