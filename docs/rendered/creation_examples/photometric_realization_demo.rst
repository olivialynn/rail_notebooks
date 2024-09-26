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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f137d5feb30>



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
    0      0.890625  27.370831  26.712660  26.025223  25.327185  25.016500   
    1      1.978239  29.557047  28.361183  27.587227  27.238544  26.628105   
    2      0.974287  26.566013  25.937716  24.787411  23.872454  23.139563   
    3      1.317978  29.042736  28.274597  27.501110  26.648792  26.091452   
    4      1.386366  26.292624  25.774778  25.429960  24.806530  24.367950   
    ...         ...        ...        ...        ...        ...        ...   
    99995  2.147172  26.550978  26.349937  26.135286  26.082020  25.911032   
    99996  1.457508  27.362209  27.036276  26.823141  26.420132  26.110037   
    99997  1.372993  27.736042  27.271955  26.887583  26.416138  26.043432   
    99998  0.855022  28.044554  27.327116  26.599014  25.862329  25.592169   
    99999  1.723768  27.049067  26.526747  26.094597  25.642973  25.197958   
    
                   y     major     minor  
    0      24.926819  0.003319  0.002869  
    1      26.248560  0.008733  0.007945  
    2      22.832047  0.103938  0.052162  
    3      25.346504  0.147522  0.143359  
    4      23.700008  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  25.558136  0.086491  0.071701  
    99996  25.524906  0.044537  0.022302  
    99997  25.456163  0.073146  0.047825  
    99998  25.506388  0.100551  0.094662  
    99999  24.900501  0.059611  0.049181  
    
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
          <td>0.890625</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.674711</td>
          <td>0.160884</td>
          <td>26.039919</td>
          <td>0.081674</td>
          <td>25.367118</td>
          <td>0.073509</td>
          <td>24.985663</td>
          <td>0.100179</td>
          <td>25.273939</td>
          <td>0.279195</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.152553</td>
          <td>1.155909</td>
          <td>28.232883</td>
          <td>0.556539</td>
          <td>27.021777</td>
          <td>0.191085</td>
          <td>26.759578</td>
          <td>0.243651</td>
          <td>27.424573</td>
          <td>0.703161</td>
          <td>27.340922</td>
          <td>1.180636</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.703366</td>
          <td>0.440634</td>
          <td>26.074227</td>
          <td>0.095632</td>
          <td>24.836704</td>
          <td>0.028167</td>
          <td>23.889285</td>
          <td>0.020048</td>
          <td>23.154278</td>
          <td>0.020023</td>
          <td>22.841395</td>
          <td>0.033898</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.668793</td>
          <td>0.864403</td>
          <td>28.863934</td>
          <td>0.854650</td>
          <td>27.221379</td>
          <td>0.225832</td>
          <td>27.044901</td>
          <td>0.307277</td>
          <td>26.128004</td>
          <td>0.264767</td>
          <td>25.289927</td>
          <td>0.282838</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.087415</td>
          <td>0.271537</td>
          <td>25.690378</td>
          <td>0.068201</td>
          <td>25.526973</td>
          <td>0.051849</td>
          <td>24.830568</td>
          <td>0.045671</td>
          <td>24.390830</td>
          <td>0.059258</td>
          <td>23.817838</td>
          <td>0.080511</td>
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
          <td>2.147172</td>
          <td>26.477166</td>
          <td>0.370389</td>
          <td>26.388980</td>
          <td>0.125824</td>
          <td>26.169686</td>
          <td>0.091560</td>
          <td>25.821203</td>
          <td>0.109580</td>
          <td>26.107139</td>
          <td>0.260291</td>
          <td>26.691480</td>
          <td>0.795869</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.843219</td>
          <td>0.489251</td>
          <td>26.658116</td>
          <td>0.158619</td>
          <td>26.513422</td>
          <td>0.123637</td>
          <td>26.173306</td>
          <td>0.148661</td>
          <td>26.341627</td>
          <td>0.314645</td>
          <td>25.466854</td>
          <td>0.325998</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.970882</td>
          <td>1.040422</td>
          <td>26.880357</td>
          <td>0.191545</td>
          <td>27.127412</td>
          <td>0.208816</td>
          <td>26.647001</td>
          <td>0.221963</td>
          <td>25.975015</td>
          <td>0.233472</td>
          <td>25.249295</td>
          <td>0.273661</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.778414</td>
          <td>0.466215</td>
          <td>27.336095</td>
          <td>0.279302</td>
          <td>26.669458</td>
          <td>0.141497</td>
          <td>25.846840</td>
          <td>0.112059</td>
          <td>25.547734</td>
          <td>0.162974</td>
          <td>25.710507</td>
          <td>0.394583</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.595184</td>
          <td>1.465242</td>
          <td>26.507378</td>
          <td>0.139375</td>
          <td>26.004401</td>
          <td>0.079154</td>
          <td>25.768244</td>
          <td>0.104625</td>
          <td>25.289454</td>
          <td>0.130524</td>
          <td>24.774440</td>
          <td>0.184491</td>
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
          <td>0.890625</td>
          <td>26.936493</td>
          <td>0.576981</td>
          <td>26.784077</td>
          <td>0.202545</td>
          <td>25.964360</td>
          <td>0.089866</td>
          <td>25.364127</td>
          <td>0.086846</td>
          <td>25.352058</td>
          <td>0.161317</td>
          <td>25.049388</td>
          <td>0.271447</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.423153</td>
          <td>0.804376</td>
          <td>27.891846</td>
          <td>0.488546</td>
          <td>27.402415</td>
          <td>0.304314</td>
          <td>28.113228</td>
          <td>0.775978</td>
          <td>26.273790</td>
          <td>0.344984</td>
          <td>26.670927</td>
          <td>0.887206</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.934684</td>
          <td>0.584260</td>
          <td>26.069276</td>
          <td>0.112019</td>
          <td>24.759567</td>
          <td>0.031663</td>
          <td>23.864652</td>
          <td>0.023693</td>
          <td>23.124199</td>
          <td>0.023372</td>
          <td>22.839877</td>
          <td>0.041017</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.963592</td>
          <td>0.541666</td>
          <td>27.778468</td>
          <td>0.433319</td>
          <td>26.575796</td>
          <td>0.261629</td>
          <td>25.663668</td>
          <td>0.223732</td>
          <td>26.137060</td>
          <td>0.656599</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.209962</td>
          <td>0.333573</td>
          <td>26.033391</td>
          <td>0.106446</td>
          <td>25.397543</td>
          <td>0.054460</td>
          <td>24.845858</td>
          <td>0.054929</td>
          <td>24.282938</td>
          <td>0.063424</td>
          <td>23.750588</td>
          <td>0.089755</td>
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
          <td>2.147172</td>
          <td>26.032398</td>
          <td>0.293677</td>
          <td>26.191353</td>
          <td>0.124401</td>
          <td>26.108099</td>
          <td>0.104098</td>
          <td>25.986462</td>
          <td>0.152485</td>
          <td>25.777479</td>
          <td>0.235404</td>
          <td>25.496665</td>
          <td>0.394756</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.205393</td>
          <td>0.333246</td>
          <td>27.103444</td>
          <td>0.264757</td>
          <td>26.890463</td>
          <td>0.200519</td>
          <td>26.259496</td>
          <td>0.189140</td>
          <td>26.242870</td>
          <td>0.337863</td>
          <td>24.628814</td>
          <td>0.192322</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.514155</td>
          <td>1.512561</td>
          <td>27.417021</td>
          <td>0.342983</td>
          <td>27.137702</td>
          <td>0.248282</td>
          <td>26.187498</td>
          <td>0.179502</td>
          <td>25.996748</td>
          <td>0.279578</td>
          <td>24.877004</td>
          <td>0.238572</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>28.486952</td>
          <td>1.504338</td>
          <td>26.875513</td>
          <td>0.224581</td>
          <td>26.945737</td>
          <td>0.215533</td>
          <td>25.824534</td>
          <td>0.134057</td>
          <td>25.334763</td>
          <td>0.163876</td>
          <td>25.659404</td>
          <td>0.451079</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.596263</td>
          <td>0.903128</td>
          <td>26.734170</td>
          <td>0.195922</td>
          <td>26.215325</td>
          <td>0.113074</td>
          <td>25.892731</td>
          <td>0.139128</td>
          <td>25.211554</td>
          <td>0.144422</td>
          <td>24.810974</td>
          <td>0.225276</td>
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
          <td>0.890625</td>
          <td>26.924076</td>
          <td>0.519288</td>
          <td>26.760320</td>
          <td>0.173069</td>
          <td>26.119581</td>
          <td>0.087624</td>
          <td>25.469219</td>
          <td>0.080459</td>
          <td>25.032405</td>
          <td>0.104377</td>
          <td>24.905624</td>
          <td>0.206056</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.520593</td>
          <td>1.411049</td>
          <td>28.487925</td>
          <td>0.666493</td>
          <td>27.145500</td>
          <td>0.212189</td>
          <td>27.482794</td>
          <td>0.432980</td>
          <td>26.633216</td>
          <td>0.395967</td>
          <td>26.420936</td>
          <td>0.664268</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.296708</td>
          <td>0.338755</td>
          <td>25.952171</td>
          <td>0.092336</td>
          <td>24.815369</td>
          <td>0.030011</td>
          <td>23.882044</td>
          <td>0.021663</td>
          <td>23.154463</td>
          <td>0.021694</td>
          <td>22.837338</td>
          <td>0.036795</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.783271</td>
          <td>0.473097</td>
          <td>27.575073</td>
          <td>0.369364</td>
          <td>26.499444</td>
          <td>0.244897</td>
          <td>25.544845</td>
          <td>0.201911</td>
          <td>25.351958</td>
          <td>0.366674</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.469019</td>
          <td>0.368366</td>
          <td>25.603057</td>
          <td>0.063210</td>
          <td>25.423792</td>
          <td>0.047378</td>
          <td>24.857981</td>
          <td>0.046867</td>
          <td>24.413940</td>
          <td>0.060573</td>
          <td>23.744654</td>
          <td>0.075585</td>
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
          <td>2.147172</td>
          <td>29.328901</td>
          <td>2.104997</td>
          <td>26.388166</td>
          <td>0.134575</td>
          <td>26.094101</td>
          <td>0.092701</td>
          <td>25.840844</td>
          <td>0.120957</td>
          <td>25.645082</td>
          <td>0.190969</td>
          <td>25.336812</td>
          <td>0.316465</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.351049</td>
          <td>0.286460</td>
          <td>26.930827</td>
          <td>0.179771</td>
          <td>26.450424</td>
          <td>0.191391</td>
          <td>26.550898</td>
          <td>0.376618</td>
          <td>25.755896</td>
          <td>0.414745</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.797375</td>
          <td>0.959626</td>
          <td>26.897169</td>
          <td>0.202377</td>
          <td>27.621356</td>
          <td>0.327145</td>
          <td>26.231097</td>
          <td>0.164196</td>
          <td>25.744411</td>
          <td>0.201779</td>
          <td>25.267570</td>
          <td>0.291042</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.775015</td>
          <td>0.498487</td>
          <td>27.332556</td>
          <td>0.305517</td>
          <td>26.398120</td>
          <td>0.125283</td>
          <td>25.666685</td>
          <td>0.107861</td>
          <td>25.944891</td>
          <td>0.253567</td>
          <td>28.855175</td>
          <td>2.497968</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.628548</td>
          <td>0.159851</td>
          <td>26.120460</td>
          <td>0.091167</td>
          <td>25.463544</td>
          <td>0.083395</td>
          <td>25.412866</td>
          <td>0.150812</td>
          <td>24.772286</td>
          <td>0.191442</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
