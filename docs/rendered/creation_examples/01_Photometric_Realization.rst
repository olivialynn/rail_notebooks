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

    <pzflow.flow.Flow at 0x7f0c3d47e9e0>



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
    0      23.994413  0.000873  0.000497  
    1      25.391064  0.022481  0.016427  
    2      24.304707  0.278039  0.177455  
    3      25.291103  0.084920  0.066781  
    4      25.096743  0.143649  0.102054  
    ...          ...       ...       ...  
    99995  24.737946  0.135607  0.077789  
    99996  24.224169  0.084176  0.075639  
    99997  25.613836  0.095783  0.076203  
    99998  25.274899  0.087536  0.059718  
    99999  25.699642  0.064570  0.059486  
    
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
          <td>27.094107</td>
          <td>0.586983</td>
          <td>26.584524</td>
          <td>0.148932</td>
          <td>26.097606</td>
          <td>0.085934</td>
          <td>25.207488</td>
          <td>0.063819</td>
          <td>24.697327</td>
          <td>0.077733</td>
          <td>23.942253</td>
          <td>0.089836</td>
          <td>0.000873</td>
          <td>0.000497</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.654049</td>
          <td>0.359918</td>
          <td>26.627619</td>
          <td>0.136483</td>
          <td>26.120643</td>
          <td>0.142078</td>
          <td>25.605637</td>
          <td>0.171215</td>
          <td>25.684907</td>
          <td>0.386851</td>
          <td>0.022481</td>
          <td>0.016427</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.304749</td>
          <td>1.257960</td>
          <td>28.276950</td>
          <td>0.574419</td>
          <td>27.641833</td>
          <td>0.318113</td>
          <td>26.070257</td>
          <td>0.136037</td>
          <td>25.111652</td>
          <td>0.111842</td>
          <td>24.477622</td>
          <td>0.143204</td>
          <td>0.278039</td>
          <td>0.177455</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.423474</td>
          <td>1.340770</td>
          <td>29.973486</td>
          <td>1.590165</td>
          <td>27.250029</td>
          <td>0.231264</td>
          <td>26.478067</td>
          <td>0.192682</td>
          <td>25.734520</td>
          <td>0.190962</td>
          <td>25.512521</td>
          <td>0.338020</td>
          <td>0.084920</td>
          <td>0.066781</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.671198</td>
          <td>0.192421</td>
          <td>26.070013</td>
          <td>0.095280</td>
          <td>25.896079</td>
          <td>0.071927</td>
          <td>25.621205</td>
          <td>0.091970</td>
          <td>25.479591</td>
          <td>0.153747</td>
          <td>25.422916</td>
          <td>0.314780</td>
          <td>0.143649</td>
          <td>0.102054</td>
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
          <td>27.129674</td>
          <td>0.601965</td>
          <td>26.310228</td>
          <td>0.117512</td>
          <td>25.479944</td>
          <td>0.049729</td>
          <td>25.035593</td>
          <td>0.054791</td>
          <td>25.031963</td>
          <td>0.104323</td>
          <td>24.640940</td>
          <td>0.164715</td>
          <td>0.135607</td>
          <td>0.077789</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.939376</td>
          <td>1.730047</td>
          <td>26.624755</td>
          <td>0.154157</td>
          <td>26.094503</td>
          <td>0.085699</td>
          <td>25.345019</td>
          <td>0.072086</td>
          <td>24.813782</td>
          <td>0.086141</td>
          <td>24.292221</td>
          <td>0.121992</td>
          <td>0.084176</td>
          <td>0.075639</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.952011</td>
          <td>1.028832</td>
          <td>26.751132</td>
          <td>0.171704</td>
          <td>26.391871</td>
          <td>0.111228</td>
          <td>26.457572</td>
          <td>0.189381</td>
          <td>25.931846</td>
          <td>0.225262</td>
          <td>26.252563</td>
          <td>0.589990</td>
          <td>0.095783</td>
          <td>0.076203</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.281620</td>
          <td>0.669189</td>
          <td>26.354117</td>
          <td>0.122078</td>
          <td>26.226547</td>
          <td>0.096248</td>
          <td>25.750842</td>
          <td>0.103044</td>
          <td>26.133995</td>
          <td>0.266065</td>
          <td>26.833142</td>
          <td>0.871755</td>
          <td>0.087536</td>
          <td>0.059718</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.524209</td>
          <td>0.384174</td>
          <td>26.809519</td>
          <td>0.180421</td>
          <td>26.399738</td>
          <td>0.111993</td>
          <td>26.366878</td>
          <td>0.175386</td>
          <td>26.178255</td>
          <td>0.275830</td>
          <td>25.822714</td>
          <td>0.429999</td>
          <td>0.064570</td>
          <td>0.059486</td>
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
          <td>26.101227</td>
          <td>0.305840</td>
          <td>26.517247</td>
          <td>0.161611</td>
          <td>26.074616</td>
          <td>0.098994</td>
          <td>25.313952</td>
          <td>0.083089</td>
          <td>24.853552</td>
          <td>0.104817</td>
          <td>23.905256</td>
          <td>0.102757</td>
          <td>0.000873</td>
          <td>0.000497</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>28.023517</td>
          <td>1.159499</td>
          <td>28.381079</td>
          <td>0.692553</td>
          <td>26.375038</td>
          <td>0.128781</td>
          <td>26.385566</td>
          <td>0.209672</td>
          <td>26.142378</td>
          <td>0.311092</td>
          <td>25.218163</td>
          <td>0.311440</td>
          <td>0.022481</td>
          <td>0.016427</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.193606</td>
          <td>0.630108</td>
          <td>26.173038</td>
          <td>0.203722</td>
          <td>24.872120</td>
          <td>0.124018</td>
          <td>24.226612</td>
          <td>0.158420</td>
          <td>0.278039</td>
          <td>0.177455</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.320895</td>
          <td>1.374779</td>
          <td>28.397287</td>
          <td>0.709120</td>
          <td>27.381069</td>
          <td>0.304444</td>
          <td>26.440209</td>
          <td>0.223391</td>
          <td>25.771523</td>
          <td>0.233904</td>
          <td>25.255650</td>
          <td>0.326363</td>
          <td>0.084920</td>
          <td>0.066781</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.877184</td>
          <td>0.570574</td>
          <td>26.279661</td>
          <td>0.137782</td>
          <td>26.001063</td>
          <td>0.097560</td>
          <td>25.725867</td>
          <td>0.125422</td>
          <td>25.462535</td>
          <td>0.185943</td>
          <td>25.415647</td>
          <td>0.380567</td>
          <td>0.143649</td>
          <td>0.102054</td>
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
          <td>28.675242</td>
          <td>1.654079</td>
          <td>26.257955</td>
          <td>0.133980</td>
          <td>25.445700</td>
          <td>0.059137</td>
          <td>25.078323</td>
          <td>0.070317</td>
          <td>24.745954</td>
          <td>0.099233</td>
          <td>24.767424</td>
          <td>0.223575</td>
          <td>0.135607</td>
          <td>0.077789</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.790427</td>
          <td>1.022916</td>
          <td>26.757375</td>
          <td>0.201733</td>
          <td>26.049409</td>
          <td>0.098917</td>
          <td>25.155222</td>
          <td>0.073851</td>
          <td>24.674980</td>
          <td>0.091562</td>
          <td>24.813582</td>
          <td>0.228246</td>
          <td>0.084176</td>
          <td>0.075639</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.531663</td>
          <td>0.435137</td>
          <td>26.659091</td>
          <td>0.186287</td>
          <td>26.119633</td>
          <td>0.105554</td>
          <td>26.028249</td>
          <td>0.158648</td>
          <td>25.820989</td>
          <td>0.244885</td>
          <td>25.480695</td>
          <td>0.391262</td>
          <td>0.095783</td>
          <td>0.076203</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.300451</td>
          <td>0.362697</td>
          <td>26.496778</td>
          <td>0.161444</td>
          <td>25.946624</td>
          <td>0.090148</td>
          <td>25.908942</td>
          <td>0.142328</td>
          <td>25.747847</td>
          <td>0.229199</td>
          <td>25.131204</td>
          <td>0.295219</td>
          <td>0.087536</td>
          <td>0.059718</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.611252</td>
          <td>0.458447</td>
          <td>26.622268</td>
          <td>0.178715</td>
          <td>26.940814</td>
          <td>0.210911</td>
          <td>26.709767</td>
          <td>0.277049</td>
          <td>25.770264</td>
          <td>0.232233</td>
          <td>26.263757</td>
          <td>0.685853</td>
          <td>0.064570</td>
          <td>0.059486</td>
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
          <td>28.314102</td>
          <td>1.264388</td>
          <td>26.919887</td>
          <td>0.198024</td>
          <td>26.055743</td>
          <td>0.082822</td>
          <td>25.194254</td>
          <td>0.063075</td>
          <td>24.710554</td>
          <td>0.078647</td>
          <td>24.020827</td>
          <td>0.096256</td>
          <td>0.000873</td>
          <td>0.000497</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.511654</td>
          <td>0.783136</td>
          <td>27.978403</td>
          <td>0.463359</td>
          <td>26.475894</td>
          <td>0.120293</td>
          <td>26.401336</td>
          <td>0.181547</td>
          <td>25.714535</td>
          <td>0.188715</td>
          <td>25.636498</td>
          <td>0.374376</td>
          <td>0.022481</td>
          <td>0.016427</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.365660</td>
          <td>1.580299</td>
          <td>28.381273</td>
          <td>0.834751</td>
          <td>28.379218</td>
          <td>0.781239</td>
          <td>26.259760</td>
          <td>0.244821</td>
          <td>25.115805</td>
          <td>0.171219</td>
          <td>24.445984</td>
          <td>0.213534</td>
          <td>0.278039</td>
          <td>0.177455</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.055220</td>
          <td>1.130903</td>
          <td>28.611805</td>
          <td>0.760652</td>
          <td>27.661545</td>
          <td>0.345161</td>
          <td>25.817380</td>
          <td>0.117832</td>
          <td>25.487756</td>
          <td>0.166225</td>
          <td>25.550259</td>
          <td>0.372564</td>
          <td>0.084920</td>
          <td>0.066781</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.616591</td>
          <td>0.207335</td>
          <td>25.891552</td>
          <td>0.094941</td>
          <td>25.960888</td>
          <td>0.090567</td>
          <td>25.746728</td>
          <td>0.122705</td>
          <td>25.708721</td>
          <td>0.220223</td>
          <td>24.823020</td>
          <td>0.227694</td>
          <td>0.143649</td>
          <td>0.102054</td>
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
          <td>26.197645</td>
          <td>0.325336</td>
          <td>26.368044</td>
          <td>0.139411</td>
          <td>25.561741</td>
          <td>0.061523</td>
          <td>24.961591</td>
          <td>0.059383</td>
          <td>24.815719</td>
          <td>0.099103</td>
          <td>24.814768</td>
          <td>0.218797</td>
          <td>0.135607</td>
          <td>0.077789</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.982805</td>
          <td>0.567699</td>
          <td>26.461617</td>
          <td>0.143512</td>
          <td>25.924243</td>
          <td>0.079920</td>
          <td>25.212671</td>
          <td>0.069762</td>
          <td>24.977020</td>
          <td>0.107655</td>
          <td>24.215916</td>
          <td>0.123913</td>
          <td>0.084176</td>
          <td>0.075639</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.544311</td>
          <td>0.155678</td>
          <td>26.319701</td>
          <td>0.114427</td>
          <td>26.335001</td>
          <td>0.187284</td>
          <td>26.241443</td>
          <td>0.315724</td>
          <td>26.474880</td>
          <td>0.740951</td>
          <td>0.095783</td>
          <td>0.076203</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.462440</td>
          <td>0.382656</td>
          <td>26.027384</td>
          <td>0.097603</td>
          <td>26.144241</td>
          <td>0.096052</td>
          <td>25.790323</td>
          <td>0.114741</td>
          <td>26.082844</td>
          <td>0.272346</td>
          <td>25.485440</td>
          <td>0.353187</td>
          <td>0.087536</td>
          <td>0.059718</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.172509</td>
          <td>0.300361</td>
          <td>26.438244</td>
          <td>0.137025</td>
          <td>26.597111</td>
          <td>0.139587</td>
          <td>26.319570</td>
          <td>0.177174</td>
          <td>25.877756</td>
          <td>0.225710</td>
          <td>25.365530</td>
          <td>0.315099</td>
          <td>0.064570</td>
          <td>0.059486</td>
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
