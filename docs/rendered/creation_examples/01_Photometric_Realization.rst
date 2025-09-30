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

    <pzflow.flow.Flow at 0x7fae5a2b8700>



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
    0      23.994413  0.046661  0.025199  
    1      25.391064  0.034003  0.030413  
    2      24.304707  0.075626  0.048406  
    3      25.291103  0.105279  0.075307  
    4      25.096743  0.030618  0.017372  
    ...          ...       ...       ...  
    99995  24.737946  0.150378  0.141294  
    99996  24.224169  0.100244  0.067783  
    99997  25.613836  0.023464  0.018260  
    99998  25.274899  0.081318  0.051042  
    99999  25.699642  0.223906  0.122727  
    
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
          <td>26.266720</td>
          <td>0.313737</td>
          <td>26.466128</td>
          <td>0.134504</td>
          <td>26.062400</td>
          <td>0.083309</td>
          <td>25.209234</td>
          <td>0.063918</td>
          <td>24.774948</td>
          <td>0.083243</td>
          <td>23.980961</td>
          <td>0.092945</td>
          <td>0.046661</td>
          <td>0.025199</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.229061</td>
          <td>0.645342</td>
          <td>27.212693</td>
          <td>0.252552</td>
          <td>26.734173</td>
          <td>0.149595</td>
          <td>26.325034</td>
          <td>0.169257</td>
          <td>25.850230</td>
          <td>0.210448</td>
          <td>25.134521</td>
          <td>0.249144</td>
          <td>0.034003</td>
          <td>0.030413</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.346006</td>
          <td>1.143230</td>
          <td>27.451140</td>
          <td>0.272798</td>
          <td>26.053100</td>
          <td>0.134036</td>
          <td>24.896738</td>
          <td>0.092661</td>
          <td>24.297408</td>
          <td>0.122543</td>
          <td>0.075626</td>
          <td>0.048406</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.154086</td>
          <td>1.156914</td>
          <td>29.865109</td>
          <td>1.507828</td>
          <td>27.717085</td>
          <td>0.337711</td>
          <td>26.059559</td>
          <td>0.134786</td>
          <td>25.460485</td>
          <td>0.151249</td>
          <td>25.357571</td>
          <td>0.298713</td>
          <td>0.105279</td>
          <td>0.075307</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.884958</td>
          <td>0.229987</td>
          <td>26.167569</td>
          <td>0.103773</td>
          <td>25.895291</td>
          <td>0.071877</td>
          <td>25.728568</td>
          <td>0.101054</td>
          <td>25.771184</td>
          <td>0.196950</td>
          <td>24.985894</td>
          <td>0.220315</td>
          <td>0.030618</td>
          <td>0.017372</td>
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
          <td>27.278121</td>
          <td>0.667582</td>
          <td>26.462633</td>
          <td>0.134099</td>
          <td>25.492604</td>
          <td>0.050291</td>
          <td>24.946690</td>
          <td>0.050632</td>
          <td>24.930909</td>
          <td>0.095483</td>
          <td>24.789284</td>
          <td>0.186820</td>
          <td>0.150378</td>
          <td>0.141294</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.927279</td>
          <td>0.520466</td>
          <td>26.765068</td>
          <td>0.173749</td>
          <td>26.064082</td>
          <td>0.083432</td>
          <td>25.228687</td>
          <td>0.065029</td>
          <td>24.806742</td>
          <td>0.085608</td>
          <td>24.224832</td>
          <td>0.115049</td>
          <td>0.100244</td>
          <td>0.067783</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.662292</td>
          <td>0.427123</td>
          <td>26.620550</td>
          <td>0.153603</td>
          <td>26.277720</td>
          <td>0.100664</td>
          <td>26.327775</td>
          <td>0.169652</td>
          <td>26.527599</td>
          <td>0.364479</td>
          <td>25.507157</td>
          <td>0.336588</td>
          <td>0.023464</td>
          <td>0.018260</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.874849</td>
          <td>0.228070</td>
          <td>26.280105</td>
          <td>0.114474</td>
          <td>26.031202</td>
          <td>0.081048</td>
          <td>25.966052</td>
          <td>0.124304</td>
          <td>25.705550</td>
          <td>0.186349</td>
          <td>25.239242</td>
          <td>0.271431</td>
          <td>0.081318</td>
          <td>0.051042</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.784262</td>
          <td>0.468258</td>
          <td>26.592756</td>
          <td>0.149988</td>
          <td>26.594775</td>
          <td>0.132664</td>
          <td>26.082337</td>
          <td>0.137463</td>
          <td>26.068118</td>
          <td>0.252098</td>
          <td>25.355810</td>
          <td>0.298290</td>
          <td>0.223906</td>
          <td>0.122727</td>
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
          <td>28.699071</td>
          <td>1.648926</td>
          <td>26.933926</td>
          <td>0.230417</td>
          <td>26.026200</td>
          <td>0.095332</td>
          <td>25.192338</td>
          <td>0.075006</td>
          <td>24.563462</td>
          <td>0.081635</td>
          <td>23.901506</td>
          <td>0.102921</td>
          <td>0.046661</td>
          <td>0.025199</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.918186</td>
          <td>0.227193</td>
          <td>26.407264</td>
          <td>0.132712</td>
          <td>26.093447</td>
          <td>0.164175</td>
          <td>25.595827</td>
          <td>0.198995</td>
          <td>25.055985</td>
          <td>0.273819</td>
          <td>0.034003</td>
          <td>0.030413</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.802492</td>
          <td>0.918729</td>
          <td>27.611677</td>
          <td>0.363535</td>
          <td>25.835542</td>
          <td>0.132892</td>
          <td>24.995134</td>
          <td>0.120183</td>
          <td>24.412580</td>
          <td>0.161576</td>
          <td>0.075626</td>
          <td>0.048406</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.362884</td>
          <td>0.785486</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.031361</td>
          <td>0.230608</td>
          <td>26.096472</td>
          <td>0.168630</td>
          <td>25.543383</td>
          <td>0.194854</td>
          <td>25.588024</td>
          <td>0.425912</td>
          <td>0.105279</td>
          <td>0.075307</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.029306</td>
          <td>0.289086</td>
          <td>26.007312</td>
          <td>0.104214</td>
          <td>26.110627</td>
          <td>0.102380</td>
          <td>25.821994</td>
          <td>0.129825</td>
          <td>25.323234</td>
          <td>0.157710</td>
          <td>25.511962</td>
          <td>0.392685</td>
          <td>0.030618</td>
          <td>0.017372</td>
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
          <td>26.220284</td>
          <td>0.132984</td>
          <td>25.513961</td>
          <td>0.064629</td>
          <td>25.067943</td>
          <td>0.071735</td>
          <td>24.971529</td>
          <td>0.124202</td>
          <td>24.802837</td>
          <td>0.236552</td>
          <td>0.150378</td>
          <td>0.141294</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.709194</td>
          <td>0.496741</td>
          <td>27.061486</td>
          <td>0.260184</td>
          <td>25.798502</td>
          <td>0.079564</td>
          <td>25.103768</td>
          <td>0.070781</td>
          <td>24.764758</td>
          <td>0.099353</td>
          <td>24.427534</td>
          <td>0.165411</td>
          <td>0.100244</td>
          <td>0.067783</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.730567</td>
          <td>0.193889</td>
          <td>26.539156</td>
          <td>0.148387</td>
          <td>26.017394</td>
          <td>0.153526</td>
          <td>25.513822</td>
          <td>0.185347</td>
          <td>25.106790</td>
          <td>0.284792</td>
          <td>0.023464</td>
          <td>0.018260</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.997166</td>
          <td>0.608073</td>
          <td>26.243126</td>
          <td>0.129452</td>
          <td>26.082847</td>
          <td>0.101255</td>
          <td>25.766055</td>
          <td>0.125376</td>
          <td>25.936685</td>
          <td>0.266888</td>
          <td>25.013597</td>
          <td>0.267537</td>
          <td>0.081318</td>
          <td>0.051042</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.386894</td>
          <td>0.830866</td>
          <td>27.183414</td>
          <td>0.305457</td>
          <td>26.629042</td>
          <td>0.175881</td>
          <td>26.003598</td>
          <td>0.167050</td>
          <td>26.045288</td>
          <td>0.314135</td>
          <td>25.592947</td>
          <td>0.454609</td>
          <td>0.223906</td>
          <td>0.122727</td>
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
          <td>26.843073</td>
          <td>0.494579</td>
          <td>26.600413</td>
          <td>0.153383</td>
          <td>25.954042</td>
          <td>0.077144</td>
          <td>25.074241</td>
          <td>0.057837</td>
          <td>24.835188</td>
          <td>0.089433</td>
          <td>24.044077</td>
          <td>0.100155</td>
          <td>0.046661</td>
          <td>0.025199</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.124846</td>
          <td>0.237612</td>
          <td>26.890242</td>
          <td>0.173261</td>
          <td>25.974581</td>
          <td>0.127049</td>
          <td>25.710056</td>
          <td>0.189579</td>
          <td>25.857521</td>
          <td>0.447027</td>
          <td>0.034003</td>
          <td>0.030413</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.775523</td>
          <td>0.411410</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.018016</td>
          <td>0.137108</td>
          <td>25.059175</td>
          <td>0.112432</td>
          <td>24.126410</td>
          <td>0.111334</td>
          <td>0.075626</td>
          <td>0.048406</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.433306</td>
          <td>0.687660</td>
          <td>27.374905</td>
          <td>0.281556</td>
          <td>26.373532</td>
          <td>0.195265</td>
          <td>25.415912</td>
          <td>0.160662</td>
          <td>25.195159</td>
          <td>0.288485</td>
          <td>0.105279</td>
          <td>0.075307</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.686344</td>
          <td>0.437168</td>
          <td>26.010552</td>
          <td>0.091094</td>
          <td>25.836022</td>
          <td>0.068780</td>
          <td>25.683385</td>
          <td>0.097983</td>
          <td>25.625178</td>
          <td>0.175489</td>
          <td>24.787899</td>
          <td>0.188161</td>
          <td>0.030618</td>
          <td>0.017372</td>
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
          <td>27.082536</td>
          <td>0.665084</td>
          <td>26.110467</td>
          <td>0.120608</td>
          <td>25.471088</td>
          <td>0.062036</td>
          <td>25.212221</td>
          <td>0.081233</td>
          <td>24.978891</td>
          <td>0.124636</td>
          <td>25.059687</td>
          <td>0.290970</td>
          <td>0.150378</td>
          <td>0.141294</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.161073</td>
          <td>0.305572</td>
          <td>26.794319</td>
          <td>0.192089</td>
          <td>26.174510</td>
          <td>0.100546</td>
          <td>25.195177</td>
          <td>0.069378</td>
          <td>24.837124</td>
          <td>0.096155</td>
          <td>24.112461</td>
          <td>0.114360</td>
          <td>0.100244</td>
          <td>0.067783</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.967763</td>
          <td>0.537880</td>
          <td>26.861957</td>
          <td>0.189537</td>
          <td>26.325164</td>
          <td>0.105558</td>
          <td>26.196447</td>
          <td>0.152574</td>
          <td>26.286614</td>
          <td>0.302720</td>
          <td>26.126634</td>
          <td>0.541744</td>
          <td>0.023464</td>
          <td>0.018260</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.811481</td>
          <td>0.224987</td>
          <td>26.203008</td>
          <td>0.112596</td>
          <td>26.174324</td>
          <td>0.097451</td>
          <td>25.676351</td>
          <td>0.102593</td>
          <td>25.675764</td>
          <td>0.192170</td>
          <td>25.747285</td>
          <td>0.427807</td>
          <td>0.081318</td>
          <td>0.051042</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.751221</td>
          <td>0.552051</td>
          <td>26.492307</td>
          <td>0.178548</td>
          <td>26.489152</td>
          <td>0.161993</td>
          <td>26.070122</td>
          <td>0.183480</td>
          <td>25.716260</td>
          <td>0.249157</td>
          <td>25.676042</td>
          <td>0.499683</td>
          <td>0.223906</td>
          <td>0.122727</td>
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
