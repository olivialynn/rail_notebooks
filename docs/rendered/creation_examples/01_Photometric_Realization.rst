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

    <pzflow.flow.Flow at 0x7f0a3203a890>



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
    0      23.994413  0.129468  0.095741  
    1      25.391064  0.076442  0.063499  
    2      24.304707  0.058220  0.052608  
    3      25.291103  0.111235  0.074377  
    4      25.096743  0.078887  0.046789  
    ...          ...       ...       ...  
    99995  24.737946  0.113959  0.082712  
    99996  24.224169  0.081689  0.065304  
    99997  25.613836  0.102185  0.065578  
    99998  25.274899  0.066076  0.058446  
    99999  25.699642  0.103537  0.093816  
    
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
          <td>27.179676</td>
          <td>0.623508</td>
          <td>26.777918</td>
          <td>0.175654</td>
          <td>26.049923</td>
          <td>0.082397</td>
          <td>25.135276</td>
          <td>0.059860</td>
          <td>24.794882</td>
          <td>0.084718</td>
          <td>23.959450</td>
          <td>0.091205</td>
          <td>0.129468</td>
          <td>0.095741</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.064903</td>
          <td>0.223534</td>
          <td>26.385880</td>
          <td>0.110648</td>
          <td>26.230528</td>
          <td>0.156137</td>
          <td>26.208986</td>
          <td>0.282794</td>
          <td>25.396448</td>
          <td>0.308184</td>
          <td>0.076442</td>
          <td>0.063499</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.093060</td>
          <td>0.586547</td>
          <td>29.191917</td>
          <td>1.045439</td>
          <td>27.804311</td>
          <td>0.361707</td>
          <td>25.779940</td>
          <td>0.105701</td>
          <td>25.202954</td>
          <td>0.121092</td>
          <td>24.255358</td>
          <td>0.118146</td>
          <td>0.058220</td>
          <td>0.052608</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.145279</td>
          <td>0.522267</td>
          <td>27.382499</td>
          <td>0.257932</td>
          <td>26.231102</td>
          <td>0.156214</td>
          <td>25.695106</td>
          <td>0.184711</td>
          <td>24.871730</td>
          <td>0.200255</td>
          <td>0.111235</td>
          <td>0.074377</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.451468</td>
          <td>0.363037</td>
          <td>26.210495</td>
          <td>0.107736</td>
          <td>25.964116</td>
          <td>0.076387</td>
          <td>25.647744</td>
          <td>0.094139</td>
          <td>25.453255</td>
          <td>0.150314</td>
          <td>25.002687</td>
          <td>0.223414</td>
          <td>0.078887</td>
          <td>0.046789</td>
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
          <td>27.212083</td>
          <td>0.637773</td>
          <td>26.292684</td>
          <td>0.115733</td>
          <td>25.419704</td>
          <td>0.047139</td>
          <td>25.094746</td>
          <td>0.057745</td>
          <td>24.863249</td>
          <td>0.089973</td>
          <td>24.750572</td>
          <td>0.180802</td>
          <td>0.113959</td>
          <td>0.082712</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.172460</td>
          <td>0.620365</td>
          <td>26.702889</td>
          <td>0.164797</td>
          <td>26.082507</td>
          <td>0.084798</td>
          <td>25.178305</td>
          <td>0.062189</td>
          <td>24.894189</td>
          <td>0.092454</td>
          <td>24.078197</td>
          <td>0.101220</td>
          <td>0.081689</td>
          <td>0.065304</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.042618</td>
          <td>0.565797</td>
          <td>26.503013</td>
          <td>0.138852</td>
          <td>26.475670</td>
          <td>0.119648</td>
          <td>26.201268</td>
          <td>0.152272</td>
          <td>25.957261</td>
          <td>0.230063</td>
          <td>25.472071</td>
          <td>0.327353</td>
          <td>0.102185</td>
          <td>0.065578</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.488673</td>
          <td>0.373722</td>
          <td>26.161839</td>
          <td>0.103255</td>
          <td>26.133699</td>
          <td>0.088708</td>
          <td>25.797538</td>
          <td>0.107339</td>
          <td>25.501176</td>
          <td>0.156616</td>
          <td>25.739853</td>
          <td>0.403605</td>
          <td>0.066076</td>
          <td>0.058446</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.190654</td>
          <td>0.628314</td>
          <td>27.253453</td>
          <td>0.261125</td>
          <td>26.330303</td>
          <td>0.105405</td>
          <td>26.274852</td>
          <td>0.162168</td>
          <td>26.117593</td>
          <td>0.262525</td>
          <td>24.821734</td>
          <td>0.192006</td>
          <td>0.103537</td>
          <td>0.093816</td>
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
          <td>26.450894</td>
          <td>0.413955</td>
          <td>26.863364</td>
          <td>0.224376</td>
          <td>25.943687</td>
          <td>0.092048</td>
          <td>25.234503</td>
          <td>0.080935</td>
          <td>24.752402</td>
          <td>0.100051</td>
          <td>24.252428</td>
          <td>0.144950</td>
          <td>0.129468</td>
          <td>0.095741</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.006357</td>
          <td>0.247067</td>
          <td>26.608221</td>
          <td>0.159756</td>
          <td>25.901091</td>
          <td>0.141056</td>
          <td>25.740707</td>
          <td>0.227377</td>
          <td>25.556991</td>
          <td>0.411831</td>
          <td>0.076442</td>
          <td>0.063499</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.037053</td>
          <td>1.925352</td>
          <td>27.845313</td>
          <td>0.475604</td>
          <td>28.736367</td>
          <td>0.813996</td>
          <td>26.056794</td>
          <td>0.160217</td>
          <td>24.991984</td>
          <td>0.119484</td>
          <td>24.236998</td>
          <td>0.138540</td>
          <td>0.058220</td>
          <td>0.052608</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.560438</td>
          <td>0.388820</td>
          <td>27.028782</td>
          <td>0.230507</td>
          <td>26.367769</td>
          <td>0.212374</td>
          <td>25.553450</td>
          <td>0.196850</td>
          <td>24.866317</td>
          <td>0.240301</td>
          <td>0.111235</td>
          <td>0.074377</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.822531</td>
          <td>0.246378</td>
          <td>26.026289</td>
          <td>0.107101</td>
          <td>25.992109</td>
          <td>0.093388</td>
          <td>25.834361</td>
          <td>0.132832</td>
          <td>25.783335</td>
          <td>0.235007</td>
          <td>25.119141</td>
          <td>0.291087</td>
          <td>0.078887</td>
          <td>0.046789</td>
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
          <td>26.286925</td>
          <td>0.136511</td>
          <td>25.339283</td>
          <td>0.053431</td>
          <td>25.195848</td>
          <td>0.077448</td>
          <td>24.938369</td>
          <td>0.116572</td>
          <td>24.810239</td>
          <td>0.230099</td>
          <td>0.113959</td>
          <td>0.082712</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.281128</td>
          <td>0.357145</td>
          <td>26.622237</td>
          <td>0.179541</td>
          <td>26.170588</td>
          <td>0.109637</td>
          <td>25.363718</td>
          <td>0.088474</td>
          <td>24.635317</td>
          <td>0.088154</td>
          <td>24.652783</td>
          <td>0.198975</td>
          <td>0.081689</td>
          <td>0.065304</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.720047</td>
          <td>0.500779</td>
          <td>26.930418</td>
          <td>0.233623</td>
          <td>26.303986</td>
          <td>0.123879</td>
          <td>26.226496</td>
          <td>0.187670</td>
          <td>25.999843</td>
          <td>0.283284</td>
          <td>25.642071</td>
          <td>0.442456</td>
          <td>0.102185</td>
          <td>0.065578</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.879756</td>
          <td>0.258019</td>
          <td>26.245457</td>
          <td>0.129444</td>
          <td>26.164302</td>
          <td>0.108476</td>
          <td>26.295564</td>
          <td>0.196700</td>
          <td>25.557067</td>
          <td>0.194380</td>
          <td>25.349356</td>
          <td>0.349401</td>
          <td>0.066076</td>
          <td>0.058446</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.706400</td>
          <td>0.498253</td>
          <td>26.461567</td>
          <td>0.158529</td>
          <td>26.604184</td>
          <td>0.161643</td>
          <td>26.414222</td>
          <td>0.221345</td>
          <td>25.966689</td>
          <td>0.277753</td>
          <td>25.287266</td>
          <td>0.338600</td>
          <td>0.103537</td>
          <td>0.093816</td>
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
          <td>28.580939</td>
          <td>1.546418</td>
          <td>26.817891</td>
          <td>0.206074</td>
          <td>26.207869</td>
          <td>0.109797</td>
          <td>25.172243</td>
          <td>0.072328</td>
          <td>24.743208</td>
          <td>0.093934</td>
          <td>24.162061</td>
          <td>0.126810</td>
          <td>0.129468</td>
          <td>0.095741</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.544454</td>
          <td>0.346806</td>
          <td>26.731570</td>
          <td>0.158671</td>
          <td>26.164925</td>
          <td>0.157359</td>
          <td>26.134832</td>
          <td>0.282126</td>
          <td>26.062856</td>
          <td>0.542958</td>
          <td>0.076442</td>
          <td>0.063499</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.125870</td>
          <td>1.740778</td>
          <td>28.295350</td>
          <td>0.542357</td>
          <td>25.773204</td>
          <td>0.109569</td>
          <td>24.889932</td>
          <td>0.095875</td>
          <td>24.320261</td>
          <td>0.130237</td>
          <td>0.058220</td>
          <td>0.052608</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>31.143746</td>
          <td>3.807902</td>
          <td>28.733903</td>
          <td>0.842198</td>
          <td>27.608138</td>
          <td>0.341126</td>
          <td>26.225058</td>
          <td>0.173215</td>
          <td>25.661036</td>
          <td>0.198857</td>
          <td>25.194665</td>
          <td>0.289944</td>
          <td>0.111235</td>
          <td>0.074377</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.098115</td>
          <td>0.283603</td>
          <td>26.157740</td>
          <td>0.107779</td>
          <td>25.848391</td>
          <td>0.072770</td>
          <td>25.836025</td>
          <td>0.117337</td>
          <td>25.829405</td>
          <td>0.217567</td>
          <td>24.829532</td>
          <td>0.203780</td>
          <td>0.078887</td>
          <td>0.046789</td>
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
          <td>28.049744</td>
          <td>1.151555</td>
          <td>26.284461</td>
          <td>0.127310</td>
          <td>25.379614</td>
          <td>0.051249</td>
          <td>25.045404</td>
          <td>0.062584</td>
          <td>25.015905</td>
          <td>0.115613</td>
          <td>24.676822</td>
          <td>0.190968</td>
          <td>0.113959</td>
          <td>0.082712</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.006625</td>
          <td>1.097495</td>
          <td>26.723240</td>
          <td>0.177746</td>
          <td>26.088421</td>
          <td>0.091337</td>
          <td>25.268721</td>
          <td>0.072463</td>
          <td>24.760703</td>
          <td>0.088084</td>
          <td>24.259822</td>
          <td>0.127281</td>
          <td>0.081689</td>
          <td>0.065304</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.084258</td>
          <td>0.613731</td>
          <td>26.620743</td>
          <td>0.165870</td>
          <td>26.158979</td>
          <td>0.099216</td>
          <td>26.174023</td>
          <td>0.162987</td>
          <td>25.988405</td>
          <td>0.256741</td>
          <td>25.910974</td>
          <td>0.497267</td>
          <td>0.102185</td>
          <td>0.065578</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.430094</td>
          <td>0.368389</td>
          <td>26.107577</td>
          <td>0.102860</td>
          <td>25.924120</td>
          <td>0.077553</td>
          <td>25.714172</td>
          <td>0.105153</td>
          <td>25.341541</td>
          <td>0.143411</td>
          <td>25.436989</td>
          <td>0.333667</td>
          <td>0.066076</td>
          <td>0.058446</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.492938</td>
          <td>0.403689</td>
          <td>27.089994</td>
          <td>0.251408</td>
          <td>26.480809</td>
          <td>0.134887</td>
          <td>26.116155</td>
          <td>0.159472</td>
          <td>26.143481</td>
          <td>0.298617</td>
          <td>25.326436</td>
          <td>0.325299</td>
          <td>0.103537</td>
          <td>0.093816</td>
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
