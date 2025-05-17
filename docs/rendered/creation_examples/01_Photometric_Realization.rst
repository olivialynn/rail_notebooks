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

    <pzflow.flow.Flow at 0x7fcc442c8e80>



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
          <td>26.588882</td>
          <td>0.403822</td>
          <td>26.603996</td>
          <td>0.151440</td>
          <td>26.056776</td>
          <td>0.082897</td>
          <td>25.239170</td>
          <td>0.065636</td>
          <td>24.787593</td>
          <td>0.084176</td>
          <td>24.013688</td>
          <td>0.095655</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>30.197360</td>
          <td>2.824881</td>
          <td>28.163405</td>
          <td>0.529220</td>
          <td>26.563373</td>
          <td>0.129108</td>
          <td>26.180780</td>
          <td>0.149618</td>
          <td>25.816015</td>
          <td>0.204506</td>
          <td>25.944480</td>
          <td>0.471321</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.411147</td>
          <td>0.730660</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.381544</td>
          <td>0.257730</td>
          <td>26.216869</td>
          <td>0.154322</td>
          <td>25.074471</td>
          <td>0.108271</td>
          <td>24.204815</td>
          <td>0.113060</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.710211</td>
          <td>0.442919</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.493728</td>
          <td>0.282397</td>
          <td>26.769578</td>
          <td>0.245666</td>
          <td>25.379691</td>
          <td>0.141101</td>
          <td>24.888272</td>
          <td>0.203055</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.536531</td>
          <td>0.387855</td>
          <td>26.113304</td>
          <td>0.098963</td>
          <td>25.978598</td>
          <td>0.077370</td>
          <td>25.676956</td>
          <td>0.096584</td>
          <td>25.505378</td>
          <td>0.157180</td>
          <td>24.683947</td>
          <td>0.170861</td>
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
          <td>27.428631</td>
          <td>0.739253</td>
          <td>26.168255</td>
          <td>0.103835</td>
          <td>25.366416</td>
          <td>0.044961</td>
          <td>25.017842</td>
          <td>0.053934</td>
          <td>24.976038</td>
          <td>0.099338</td>
          <td>24.695989</td>
          <td>0.172619</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.566474</td>
          <td>0.809464</td>
          <td>26.763028</td>
          <td>0.173448</td>
          <td>26.052926</td>
          <td>0.082616</td>
          <td>25.290763</td>
          <td>0.068706</td>
          <td>24.797585</td>
          <td>0.084920</td>
          <td>24.198336</td>
          <td>0.112423</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.524852</td>
          <td>0.384365</td>
          <td>27.052785</td>
          <td>0.221294</td>
          <td>26.390807</td>
          <td>0.111124</td>
          <td>26.148922</td>
          <td>0.145578</td>
          <td>26.554551</td>
          <td>0.372228</td>
          <td>25.418385</td>
          <td>0.313642</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.457351</td>
          <td>0.364709</td>
          <td>26.304140</td>
          <td>0.116892</td>
          <td>25.967350</td>
          <td>0.076605</td>
          <td>25.781675</td>
          <td>0.105861</td>
          <td>25.549662</td>
          <td>0.163242</td>
          <td>25.064737</td>
          <td>0.235211</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.984341</td>
          <td>0.542528</td>
          <td>26.587500</td>
          <td>0.149313</td>
          <td>26.556740</td>
          <td>0.128368</td>
          <td>26.429953</td>
          <td>0.185014</td>
          <td>26.453526</td>
          <td>0.343879</td>
          <td>25.943534</td>
          <td>0.470988</td>
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
          <td>27.498519</td>
          <td>0.844345</td>
          <td>26.514437</td>
          <td>0.161228</td>
          <td>26.098117</td>
          <td>0.101056</td>
          <td>25.236923</td>
          <td>0.077634</td>
          <td>24.803566</td>
          <td>0.100335</td>
          <td>24.159860</td>
          <td>0.128258</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.489737</td>
          <td>0.839705</td>
          <td>27.583823</td>
          <td>0.386817</td>
          <td>26.718988</td>
          <td>0.172816</td>
          <td>26.408781</td>
          <td>0.213551</td>
          <td>27.548359</td>
          <td>0.860600</td>
          <td>25.654965</td>
          <td>0.437356</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.604926</td>
          <td>0.459094</td>
          <td>29.715223</td>
          <td>1.534574</td>
          <td>27.993269</td>
          <td>0.489966</td>
          <td>25.792163</td>
          <td>0.129151</td>
          <td>25.093625</td>
          <td>0.132036</td>
          <td>24.149815</td>
          <td>0.130051</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.871496</td>
          <td>0.991482</td>
          <td>26.765955</td>
          <td>0.191801</td>
          <td>26.231483</td>
          <td>0.196614</td>
          <td>25.318702</td>
          <td>0.167345</td>
          <td>25.133344</td>
          <td>0.309463</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.528272</td>
          <td>0.427060</td>
          <td>26.002151</td>
          <td>0.103582</td>
          <td>25.960659</td>
          <td>0.089603</td>
          <td>25.568768</td>
          <td>0.103963</td>
          <td>25.308207</td>
          <td>0.155430</td>
          <td>25.167396</td>
          <td>0.298741</td>
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
          <td>26.812688</td>
          <td>0.534744</td>
          <td>26.265023</td>
          <td>0.132586</td>
          <td>25.334080</td>
          <td>0.052567</td>
          <td>24.951358</td>
          <td>0.061637</td>
          <td>24.764004</td>
          <td>0.098969</td>
          <td>24.573203</td>
          <td>0.186579</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.701502</td>
          <td>0.487589</td>
          <td>26.697114</td>
          <td>0.188942</td>
          <td>26.093669</td>
          <td>0.101083</td>
          <td>25.136296</td>
          <td>0.071339</td>
          <td>24.922551</td>
          <td>0.111790</td>
          <td>24.203158</td>
          <td>0.133716</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.687004</td>
          <td>0.485058</td>
          <td>26.958382</td>
          <td>0.236742</td>
          <td>26.452911</td>
          <td>0.139310</td>
          <td>25.992799</td>
          <td>0.152050</td>
          <td>25.914941</td>
          <td>0.261559</td>
          <td>26.107029</td>
          <td>0.615211</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.619449</td>
          <td>0.210892</td>
          <td>26.280584</td>
          <td>0.135632</td>
          <td>26.083984</td>
          <td>0.102990</td>
          <td>25.806274</td>
          <td>0.131958</td>
          <td>25.524605</td>
          <td>0.192494</td>
          <td>25.999307</td>
          <td>0.578868</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.656617</td>
          <td>0.473377</td>
          <td>26.885581</td>
          <td>0.222361</td>
          <td>26.562910</td>
          <td>0.152714</td>
          <td>26.578891</td>
          <td>0.248254</td>
          <td>25.203622</td>
          <td>0.143440</td>
          <td>25.551732</td>
          <td>0.407806</td>
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
          <td>27.788249</td>
          <td>0.931621</td>
          <td>26.874766</td>
          <td>0.190666</td>
          <td>25.820360</td>
          <td>0.067273</td>
          <td>25.129189</td>
          <td>0.059545</td>
          <td>24.604759</td>
          <td>0.071636</td>
          <td>23.854427</td>
          <td>0.083163</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.678046</td>
          <td>0.869899</td>
          <td>27.411764</td>
          <td>0.297135</td>
          <td>26.589272</td>
          <td>0.132158</td>
          <td>26.271726</td>
          <td>0.161892</td>
          <td>25.730865</td>
          <td>0.190547</td>
          <td>25.701504</td>
          <td>0.392189</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.548548</td>
          <td>0.733667</td>
          <td>28.681571</td>
          <td>0.733938</td>
          <td>25.966424</td>
          <td>0.135279</td>
          <td>25.077616</td>
          <td>0.117747</td>
          <td>24.125037</td>
          <td>0.114719</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.495527</td>
          <td>1.403335</td>
          <td>27.730001</td>
          <td>0.416321</td>
          <td>26.135458</td>
          <td>0.180669</td>
          <td>25.389120</td>
          <td>0.177054</td>
          <td>25.030783</td>
          <td>0.283982</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.933628</td>
          <td>0.523313</td>
          <td>25.960813</td>
          <td>0.086676</td>
          <td>26.024893</td>
          <td>0.080713</td>
          <td>25.666399</td>
          <td>0.095836</td>
          <td>25.383490</td>
          <td>0.141760</td>
          <td>25.424431</td>
          <td>0.315584</td>
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
          <td>26.419215</td>
          <td>0.138227</td>
          <td>25.421508</td>
          <td>0.051135</td>
          <td>25.066295</td>
          <td>0.061199</td>
          <td>24.944949</td>
          <td>0.104556</td>
          <td>24.827678</td>
          <td>0.208612</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.878476</td>
          <td>0.507008</td>
          <td>27.099612</td>
          <td>0.233192</td>
          <td>25.941884</td>
          <td>0.076154</td>
          <td>25.091707</td>
          <td>0.058606</td>
          <td>24.770288</td>
          <td>0.084284</td>
          <td>24.110985</td>
          <td>0.105956</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.603536</td>
          <td>0.850168</td>
          <td>26.624512</td>
          <td>0.160671</td>
          <td>26.062448</td>
          <td>0.087505</td>
          <td>26.229322</td>
          <td>0.163947</td>
          <td>25.948985</td>
          <td>0.239247</td>
          <td>25.095232</td>
          <td>0.252945</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.065558</td>
          <td>0.287785</td>
          <td>26.208724</td>
          <td>0.118896</td>
          <td>25.975729</td>
          <td>0.086600</td>
          <td>25.792510</td>
          <td>0.120357</td>
          <td>25.447286</td>
          <td>0.167185</td>
          <td>25.128527</td>
          <td>0.276846</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.981639</td>
          <td>0.553741</td>
          <td>26.600538</td>
          <td>0.156069</td>
          <td>26.543696</td>
          <td>0.131889</td>
          <td>26.324663</td>
          <td>0.176013</td>
          <td>26.270702</td>
          <td>0.307991</td>
          <td>26.715542</td>
          <td>0.832782</td>
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
