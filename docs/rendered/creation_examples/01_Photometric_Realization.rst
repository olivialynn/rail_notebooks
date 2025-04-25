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

    <pzflow.flow.Flow at 0x7f24931b3c10>



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
          <td>27.370460</td>
          <td>0.710936</td>
          <td>26.415873</td>
          <td>0.128787</td>
          <td>26.050815</td>
          <td>0.082462</td>
          <td>25.279266</td>
          <td>0.068010</td>
          <td>24.612505</td>
          <td>0.072119</td>
          <td>23.844416</td>
          <td>0.082421</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.093392</td>
          <td>1.117531</td>
          <td>27.390615</td>
          <td>0.291895</td>
          <td>26.658280</td>
          <td>0.140140</td>
          <td>26.310226</td>
          <td>0.167136</td>
          <td>26.472076</td>
          <td>0.348943</td>
          <td>25.183482</td>
          <td>0.259353</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.323826</td>
          <td>0.245799</td>
          <td>25.947883</td>
          <td>0.122359</td>
          <td>25.239841</td>
          <td>0.125032</td>
          <td>24.380818</td>
          <td>0.131729</td>
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
          <td>27.257606</td>
          <td>0.232720</td>
          <td>26.245985</td>
          <td>0.158216</td>
          <td>25.531429</td>
          <td>0.160720</td>
          <td>25.915625</td>
          <td>0.461251</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.904086</td>
          <td>0.233653</td>
          <td>26.180212</td>
          <td>0.104925</td>
          <td>25.978015</td>
          <td>0.077331</td>
          <td>25.824386</td>
          <td>0.109885</td>
          <td>25.546106</td>
          <td>0.162747</td>
          <td>25.499412</td>
          <td>0.334530</td>
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
          <td>27.135088</td>
          <td>0.604270</td>
          <td>26.517971</td>
          <td>0.140652</td>
          <td>25.387708</td>
          <td>0.045819</td>
          <td>25.075990</td>
          <td>0.056791</td>
          <td>24.973172</td>
          <td>0.099089</td>
          <td>24.789349</td>
          <td>0.186830</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.479324</td>
          <td>0.764565</td>
          <td>26.496362</td>
          <td>0.138058</td>
          <td>26.061024</td>
          <td>0.083208</td>
          <td>25.144915</td>
          <td>0.060374</td>
          <td>24.776172</td>
          <td>0.083333</td>
          <td>24.344246</td>
          <td>0.127624</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.241072</td>
          <td>0.650736</td>
          <td>26.568174</td>
          <td>0.146857</td>
          <td>26.378199</td>
          <td>0.109909</td>
          <td>26.001016</td>
          <td>0.128130</td>
          <td>26.735485</td>
          <td>0.427887</td>
          <td>25.516549</td>
          <td>0.339098</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.467421</td>
          <td>0.367586</td>
          <td>26.352139</td>
          <td>0.121868</td>
          <td>26.305038</td>
          <td>0.103101</td>
          <td>26.014068</td>
          <td>0.129586</td>
          <td>25.549060</td>
          <td>0.163158</td>
          <td>25.112776</td>
          <td>0.244725</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>28.303511</td>
          <td>1.257111</td>
          <td>26.925362</td>
          <td>0.198936</td>
          <td>26.554685</td>
          <td>0.128140</td>
          <td>26.477472</td>
          <td>0.192586</td>
          <td>26.024049</td>
          <td>0.243122</td>
          <td>25.764097</td>
          <td>0.411185</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.939852</td>
          <td>0.230622</td>
          <td>25.897290</td>
          <td>0.084715</td>
          <td>25.280500</td>
          <td>0.080677</td>
          <td>24.747605</td>
          <td>0.095533</td>
          <td>24.007992</td>
          <td>0.112405</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.237755</td>
          <td>1.304059</td>
          <td>27.844071</td>
          <td>0.471488</td>
          <td>27.252503</td>
          <td>0.269569</td>
          <td>26.209308</td>
          <td>0.180568</td>
          <td>25.848211</td>
          <td>0.244733</td>
          <td>25.070225</td>
          <td>0.276140</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.454890</td>
          <td>0.680971</td>
          <td>25.967749</td>
          <td>0.150254</td>
          <td>25.047169</td>
          <td>0.126833</td>
          <td>24.006682</td>
          <td>0.114858</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.206432</td>
          <td>0.724638</td>
          <td>28.478270</td>
          <td>0.773528</td>
          <td>27.388831</td>
          <td>0.319927</td>
          <td>26.292411</td>
          <td>0.206928</td>
          <td>25.809036</td>
          <td>0.252278</td>
          <td>25.062540</td>
          <td>0.292345</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.940213</td>
          <td>0.268628</td>
          <td>26.215968</td>
          <td>0.124760</td>
          <td>25.830878</td>
          <td>0.079925</td>
          <td>25.872912</td>
          <td>0.135425</td>
          <td>25.545062</td>
          <td>0.190091</td>
          <td>25.056816</td>
          <td>0.273177</td>
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
          <td>26.225010</td>
          <td>0.128079</td>
          <td>25.430519</td>
          <td>0.057263</td>
          <td>25.231778</td>
          <td>0.078990</td>
          <td>24.932923</td>
          <td>0.114703</td>
          <td>24.866700</td>
          <td>0.238432</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.618569</td>
          <td>0.458352</td>
          <td>26.581574</td>
          <td>0.171338</td>
          <td>26.030060</td>
          <td>0.095602</td>
          <td>25.241256</td>
          <td>0.078271</td>
          <td>24.864361</td>
          <td>0.106255</td>
          <td>24.467135</td>
          <td>0.167705</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.190234</td>
          <td>0.331227</td>
          <td>27.371904</td>
          <td>0.330960</td>
          <td>26.575664</td>
          <td>0.154807</td>
          <td>26.250478</td>
          <td>0.189320</td>
          <td>26.196312</td>
          <td>0.328160</td>
          <td>25.583667</td>
          <td>0.418926</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.731404</td>
          <td>0.992584</td>
          <td>26.379844</td>
          <td>0.147723</td>
          <td>25.853374</td>
          <td>0.084113</td>
          <td>26.126850</td>
          <td>0.173701</td>
          <td>26.017720</td>
          <td>0.289282</td>
          <td>25.985144</td>
          <td>0.573040</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.476613</td>
          <td>0.413207</td>
          <td>27.114358</td>
          <td>0.268439</td>
          <td>26.627843</td>
          <td>0.161438</td>
          <td>26.409038</td>
          <td>0.215672</td>
          <td>27.345346</td>
          <td>0.759834</td>
          <td>25.328891</td>
          <td>0.342859</td>
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
          <td>28.091555</td>
          <td>1.116417</td>
          <td>26.649613</td>
          <td>0.157488</td>
          <td>25.931580</td>
          <td>0.074231</td>
          <td>25.257268</td>
          <td>0.066707</td>
          <td>24.674654</td>
          <td>0.076202</td>
          <td>23.888218</td>
          <td>0.085676</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.471535</td>
          <td>0.761013</td>
          <td>27.939673</td>
          <td>0.448635</td>
          <td>26.641580</td>
          <td>0.138265</td>
          <td>26.537877</td>
          <td>0.202813</td>
          <td>25.723189</td>
          <td>0.189317</td>
          <td>25.020677</td>
          <td>0.226989</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.148907</td>
          <td>0.639285</td>
          <td>28.367932</td>
          <td>0.648725</td>
          <td>28.254467</td>
          <td>0.544947</td>
          <td>25.751228</td>
          <td>0.112233</td>
          <td>25.066934</td>
          <td>0.116657</td>
          <td>24.235617</td>
          <td>0.126288</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.323874</td>
          <td>0.781737</td>
          <td>27.852358</td>
          <td>0.497986</td>
          <td>27.265477</td>
          <td>0.288830</td>
          <td>26.061117</td>
          <td>0.169620</td>
          <td>25.417021</td>
          <td>0.181289</td>
          <td>25.502461</td>
          <td>0.411949</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.563548</td>
          <td>0.396369</td>
          <td>26.043299</td>
          <td>0.093188</td>
          <td>25.895322</td>
          <td>0.071982</td>
          <td>25.953962</td>
          <td>0.123188</td>
          <td>25.270431</td>
          <td>0.128572</td>
          <td>25.055532</td>
          <td>0.233751</td>
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
          <td>29.017843</td>
          <td>1.844599</td>
          <td>26.091343</td>
          <td>0.103991</td>
          <td>25.517095</td>
          <td>0.055663</td>
          <td>24.967290</td>
          <td>0.056053</td>
          <td>24.796249</td>
          <td>0.091778</td>
          <td>25.008368</td>
          <td>0.242397</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.044438</td>
          <td>0.571859</td>
          <td>26.506677</td>
          <td>0.141261</td>
          <td>25.994237</td>
          <td>0.079757</td>
          <td>25.206307</td>
          <td>0.064876</td>
          <td>24.783639</td>
          <td>0.085281</td>
          <td>24.026155</td>
          <td>0.098374</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.503862</td>
          <td>0.389820</td>
          <td>26.417086</td>
          <td>0.134461</td>
          <td>26.430068</td>
          <td>0.120702</td>
          <td>26.446672</td>
          <td>0.197096</td>
          <td>25.837176</td>
          <td>0.218056</td>
          <td>24.983804</td>
          <td>0.230736</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.751272</td>
          <td>0.222485</td>
          <td>26.085762</td>
          <td>0.106827</td>
          <td>26.101944</td>
          <td>0.096758</td>
          <td>25.921474</td>
          <td>0.134585</td>
          <td>25.380885</td>
          <td>0.157973</td>
          <td>24.988090</td>
          <td>0.246816</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>30.071855</td>
          <td>2.737668</td>
          <td>26.709735</td>
          <td>0.171298</td>
          <td>26.620516</td>
          <td>0.140931</td>
          <td>26.193361</td>
          <td>0.157380</td>
          <td>26.030078</td>
          <td>0.253379</td>
          <td>26.614958</td>
          <td>0.780081</td>
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
