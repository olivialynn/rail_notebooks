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

    <pzflow.flow.Flow at 0x7f359c56c430>



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
    0      23.994413  0.137228  0.130736  
    1      25.391064  0.059881  0.049777  
    2      24.304707  0.047975  0.030565  
    3      25.291103  0.125717  0.101529  
    4      25.096743  0.047339  0.045496  
    ...          ...       ...       ...  
    99995  24.737946  0.131803  0.095542  
    99996  24.224169  0.177722  0.166959  
    99997  25.613836  0.160130  0.087671  
    99998  25.274899  0.001648  0.001565  
    99999  25.699642  0.134787  0.093770  
    
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
          <td>28.876637</td>
          <td>1.680364</td>
          <td>27.178175</td>
          <td>0.245488</td>
          <td>26.045520</td>
          <td>0.082078</td>
          <td>25.217393</td>
          <td>0.064382</td>
          <td>24.720116</td>
          <td>0.079313</td>
          <td>24.023216</td>
          <td>0.096458</td>
          <td>0.137228</td>
          <td>0.130736</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.847984</td>
          <td>1.657874</td>
          <td>27.867200</td>
          <td>0.424357</td>
          <td>26.572524</td>
          <td>0.130135</td>
          <td>26.288406</td>
          <td>0.164055</td>
          <td>26.141613</td>
          <td>0.267724</td>
          <td>25.103117</td>
          <td>0.242785</td>
          <td>0.059881</td>
          <td>0.049777</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.950239</td>
          <td>1.027748</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.151702</td>
          <td>0.471853</td>
          <td>25.991816</td>
          <td>0.127112</td>
          <td>25.002938</td>
          <td>0.101706</td>
          <td>24.307908</td>
          <td>0.123665</td>
          <td>0.047975</td>
          <td>0.030565</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.594149</td>
          <td>0.824087</td>
          <td>28.219986</td>
          <td>0.551388</td>
          <td>28.076261</td>
          <td>0.445868</td>
          <td>26.253848</td>
          <td>0.159284</td>
          <td>25.483801</td>
          <td>0.154303</td>
          <td>25.263695</td>
          <td>0.276883</td>
          <td>0.125717</td>
          <td>0.101529</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.068677</td>
          <td>0.576446</td>
          <td>26.092425</td>
          <td>0.097170</td>
          <td>25.962834</td>
          <td>0.076300</td>
          <td>25.727697</td>
          <td>0.100977</td>
          <td>25.595968</td>
          <td>0.169813</td>
          <td>25.515558</td>
          <td>0.338832</td>
          <td>0.047339</td>
          <td>0.045496</td>
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
          <td>26.574338</td>
          <td>0.399332</td>
          <td>26.452666</td>
          <td>0.132950</td>
          <td>25.468901</td>
          <td>0.049243</td>
          <td>25.066364</td>
          <td>0.056308</td>
          <td>24.968536</td>
          <td>0.098687</td>
          <td>24.559602</td>
          <td>0.153651</td>
          <td>0.131803</td>
          <td>0.095542</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.571901</td>
          <td>0.147328</td>
          <td>26.025563</td>
          <td>0.080646</td>
          <td>25.351128</td>
          <td>0.072477</td>
          <td>24.806333</td>
          <td>0.085577</td>
          <td>24.346420</td>
          <td>0.127864</td>
          <td>0.177722</td>
          <td>0.166959</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.456214</td>
          <td>0.364385</td>
          <td>26.492270</td>
          <td>0.137572</td>
          <td>26.312962</td>
          <td>0.103818</td>
          <td>26.707993</td>
          <td>0.233487</td>
          <td>25.866949</td>
          <td>0.213409</td>
          <td>25.056016</td>
          <td>0.233520</td>
          <td>0.160130</td>
          <td>0.087671</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.163982</td>
          <td>0.288907</td>
          <td>26.072443</td>
          <td>0.095483</td>
          <td>26.055261</td>
          <td>0.082786</td>
          <td>25.980065</td>
          <td>0.125824</td>
          <td>25.610135</td>
          <td>0.171871</td>
          <td>25.054563</td>
          <td>0.233239</td>
          <td>0.001648</td>
          <td>0.001565</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.623930</td>
          <td>0.414812</td>
          <td>26.778598</td>
          <td>0.175755</td>
          <td>26.409708</td>
          <td>0.112971</td>
          <td>26.338085</td>
          <td>0.171147</td>
          <td>25.845794</td>
          <td>0.209669</td>
          <td>25.322816</td>
          <td>0.290462</td>
          <td>0.134787</td>
          <td>0.093770</td>
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
          <td>26.903836</td>
          <td>0.584335</td>
          <td>26.586904</td>
          <td>0.180355</td>
          <td>26.004107</td>
          <td>0.098573</td>
          <td>25.091439</td>
          <td>0.072488</td>
          <td>24.550187</td>
          <td>0.085081</td>
          <td>23.772828</td>
          <td>0.097067</td>
          <td>0.137228</td>
          <td>0.130736</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.976058</td>
          <td>0.597168</td>
          <td>27.156177</td>
          <td>0.277759</td>
          <td>26.668834</td>
          <td>0.167206</td>
          <td>26.548001</td>
          <td>0.242054</td>
          <td>26.438718</td>
          <td>0.395833</td>
          <td>25.062930</td>
          <td>0.277123</td>
          <td>0.059881</td>
          <td>0.049777</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.725457</td>
          <td>0.975567</td>
          <td>27.668429</td>
          <td>0.414551</td>
          <td>27.882655</td>
          <td>0.444626</td>
          <td>26.202045</td>
          <td>0.180402</td>
          <td>25.135901</td>
          <td>0.134698</td>
          <td>24.180654</td>
          <td>0.131309</td>
          <td>0.047975</td>
          <td>0.030565</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>32.564037</td>
          <td>5.286617</td>
          <td>27.728286</td>
          <td>0.446588</td>
          <td>27.335224</td>
          <td>0.299672</td>
          <td>26.305979</td>
          <td>0.204272</td>
          <td>25.684050</td>
          <td>0.222306</td>
          <td>24.649416</td>
          <td>0.203178</td>
          <td>0.125717</td>
          <td>0.101529</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.521705</td>
          <td>0.426877</td>
          <td>26.013622</td>
          <td>0.105279</td>
          <td>25.980647</td>
          <td>0.091827</td>
          <td>25.860123</td>
          <td>0.134887</td>
          <td>25.623560</td>
          <td>0.204412</td>
          <td>25.678209</td>
          <td>0.447893</td>
          <td>0.047339</td>
          <td>0.045496</td>
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
          <td>28.184305</td>
          <td>1.292813</td>
          <td>26.370974</td>
          <td>0.148114</td>
          <td>25.442282</td>
          <td>0.059164</td>
          <td>25.116296</td>
          <td>0.072980</td>
          <td>24.754304</td>
          <td>0.100304</td>
          <td>24.368323</td>
          <td>0.160226</td>
          <td>0.131803</td>
          <td>0.095542</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.612261</td>
          <td>0.189899</td>
          <td>26.010928</td>
          <td>0.102613</td>
          <td>25.360279</td>
          <td>0.095174</td>
          <td>24.929322</td>
          <td>0.122650</td>
          <td>24.023400</td>
          <td>0.125056</td>
          <td>0.177722</td>
          <td>0.166959</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.333180</td>
          <td>0.380906</td>
          <td>26.572589</td>
          <td>0.177456</td>
          <td>26.351252</td>
          <td>0.132734</td>
          <td>26.217422</td>
          <td>0.191574</td>
          <td>26.879457</td>
          <td>0.569819</td>
          <td>25.241549</td>
          <td>0.332872</td>
          <td>0.160130</td>
          <td>0.087671</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.837170</td>
          <td>0.246867</td>
          <td>26.144301</td>
          <td>0.117203</td>
          <td>26.100375</td>
          <td>0.101253</td>
          <td>25.787442</td>
          <td>0.125727</td>
          <td>25.676687</td>
          <td>0.212228</td>
          <td>24.837509</td>
          <td>0.228055</td>
          <td>0.001648</td>
          <td>0.001565</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.713522</td>
          <td>0.504637</td>
          <td>27.071271</td>
          <td>0.266601</td>
          <td>26.399221</td>
          <td>0.137097</td>
          <td>26.030850</td>
          <td>0.162032</td>
          <td>25.665436</td>
          <td>0.219132</td>
          <td>26.220774</td>
          <td>0.682503</td>
          <td>0.134787</td>
          <td>0.093770</td>
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
          <td>26.312454</td>
          <td>0.370371</td>
          <td>26.615025</td>
          <td>0.181178</td>
          <td>25.931134</td>
          <td>0.090450</td>
          <td>25.186818</td>
          <td>0.077080</td>
          <td>24.805912</td>
          <td>0.104173</td>
          <td>23.943821</td>
          <td>0.110221</td>
          <td>0.137228</td>
          <td>0.130736</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.382977</td>
          <td>0.299397</td>
          <td>26.516150</td>
          <td>0.128855</td>
          <td>26.301464</td>
          <td>0.172679</td>
          <td>25.750865</td>
          <td>0.201057</td>
          <td>26.022440</td>
          <td>0.516815</td>
          <td>0.059881</td>
          <td>0.049777</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.727646</td>
          <td>1.421738</td>
          <td>28.879566</td>
          <td>0.799376</td>
          <td>26.220534</td>
          <td>0.158233</td>
          <td>24.809797</td>
          <td>0.087701</td>
          <td>24.255199</td>
          <td>0.120760</td>
          <td>0.047975</td>
          <td>0.030565</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.034256</td>
          <td>1.907628</td>
          <td>31.384511</td>
          <td>2.941383</td>
          <td>28.000924</td>
          <td>0.478934</td>
          <td>26.277309</td>
          <td>0.189049</td>
          <td>25.292994</td>
          <td>0.151675</td>
          <td>25.672878</td>
          <td>0.438791</td>
          <td>0.125717</td>
          <td>0.101529</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.057612</td>
          <td>0.269955</td>
          <td>26.240851</td>
          <td>0.113378</td>
          <td>26.106871</td>
          <td>0.089144</td>
          <td>25.911157</td>
          <td>0.122066</td>
          <td>25.463568</td>
          <td>0.155910</td>
          <td>24.900420</td>
          <td>0.210984</td>
          <td>0.047339</td>
          <td>0.045496</td>
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
          <td>26.417055</td>
          <td>0.389495</td>
          <td>26.511568</td>
          <td>0.159395</td>
          <td>25.405831</td>
          <td>0.054267</td>
          <td>25.012272</td>
          <td>0.062946</td>
          <td>24.766627</td>
          <td>0.096132</td>
          <td>25.113273</td>
          <td>0.283065</td>
          <td>0.131803</td>
          <td>0.095542</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.896798</td>
          <td>0.609406</td>
          <td>27.233129</td>
          <td>0.326848</td>
          <td>25.998398</td>
          <td>0.105543</td>
          <td>25.242442</td>
          <td>0.089355</td>
          <td>24.789205</td>
          <td>0.112884</td>
          <td>24.276854</td>
          <td>0.161769</td>
          <td>0.177722</td>
          <td>0.166959</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.926180</td>
          <td>0.579273</td>
          <td>26.849105</td>
          <td>0.217055</td>
          <td>26.218135</td>
          <td>0.114089</td>
          <td>26.066140</td>
          <td>0.162474</td>
          <td>26.071036</td>
          <td>0.297910</td>
          <td>25.282102</td>
          <td>0.332345</td>
          <td>0.160130</td>
          <td>0.087671</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.073062</td>
          <td>0.268391</td>
          <td>26.150573</td>
          <td>0.102246</td>
          <td>26.103604</td>
          <td>0.086392</td>
          <td>25.779453</td>
          <td>0.105660</td>
          <td>25.619608</td>
          <td>0.173267</td>
          <td>25.018305</td>
          <td>0.226340</td>
          <td>0.001648</td>
          <td>0.001565</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.758685</td>
          <td>0.504649</td>
          <td>27.135502</td>
          <td>0.268918</td>
          <td>26.490092</td>
          <td>0.140850</td>
          <td>26.256213</td>
          <td>0.186207</td>
          <td>27.112130</td>
          <td>0.639228</td>
          <td>25.257291</td>
          <td>0.318334</td>
          <td>0.134787</td>
          <td>0.093770</td>
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
