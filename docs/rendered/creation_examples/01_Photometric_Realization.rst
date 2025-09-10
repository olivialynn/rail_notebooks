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

    <pzflow.flow.Flow at 0x7ffb19072ad0>



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
    0      23.994413  0.095365  0.062376  
    1      25.391064  0.069767  0.037766  
    2      24.304707  0.073229  0.071941  
    3      25.291103  0.056929  0.042993  
    4      25.096743  0.118856  0.067069  
    ...          ...       ...       ...  
    99995  24.737946  0.070132  0.068275  
    99996  24.224169  0.131527  0.067129  
    99997  25.613836  0.147404  0.081710  
    99998  25.274899  0.082798  0.048557  
    99999  25.699642  0.099810  0.067017  
    
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
          <td>27.208384</td>
          <td>0.636133</td>
          <td>26.787315</td>
          <td>0.177059</td>
          <td>26.053670</td>
          <td>0.082670</td>
          <td>25.105474</td>
          <td>0.058297</td>
          <td>24.638887</td>
          <td>0.073821</td>
          <td>24.000960</td>
          <td>0.094592</td>
          <td>0.095365</td>
          <td>0.062376</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.760447</td>
          <td>0.459985</td>
          <td>26.950834</td>
          <td>0.203233</td>
          <td>26.686952</td>
          <td>0.143644</td>
          <td>26.562037</td>
          <td>0.206766</td>
          <td>25.962631</td>
          <td>0.231089</td>
          <td>25.409169</td>
          <td>0.311339</td>
          <td>0.069767</td>
          <td>0.037766</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.730222</td>
          <td>1.566815</td>
          <td>27.417956</td>
          <td>0.298394</td>
          <td>28.590188</td>
          <td>0.647241</td>
          <td>26.022638</td>
          <td>0.130551</td>
          <td>24.861250</td>
          <td>0.089815</td>
          <td>24.391285</td>
          <td>0.132926</td>
          <td>0.073229</td>
          <td>0.071941</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.811690</td>
          <td>2.473030</td>
          <td>28.961917</td>
          <td>0.909116</td>
          <td>27.920583</td>
          <td>0.395913</td>
          <td>26.099403</td>
          <td>0.139501</td>
          <td>25.612511</td>
          <td>0.172219</td>
          <td>25.468972</td>
          <td>0.326547</td>
          <td>0.056929</td>
          <td>0.042993</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.486878</td>
          <td>0.373201</td>
          <td>26.076350</td>
          <td>0.095811</td>
          <td>25.890615</td>
          <td>0.071581</td>
          <td>25.723954</td>
          <td>0.100647</td>
          <td>25.659705</td>
          <td>0.179258</td>
          <td>24.856671</td>
          <td>0.197737</td>
          <td>0.118856</td>
          <td>0.067069</td>
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
          <td>27.692356</td>
          <td>0.877393</td>
          <td>26.357671</td>
          <td>0.122455</td>
          <td>25.456615</td>
          <td>0.048709</td>
          <td>25.024510</td>
          <td>0.054254</td>
          <td>25.013101</td>
          <td>0.102615</td>
          <td>24.466010</td>
          <td>0.141780</td>
          <td>0.070132</td>
          <td>0.068275</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.646253</td>
          <td>1.503286</td>
          <td>26.431368</td>
          <td>0.130525</td>
          <td>26.046649</td>
          <td>0.082160</td>
          <td>25.187966</td>
          <td>0.062724</td>
          <td>24.764286</td>
          <td>0.082465</td>
          <td>24.244659</td>
          <td>0.117052</td>
          <td>0.131527</td>
          <td>0.067129</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.943555</td>
          <td>1.023664</td>
          <td>26.482774</td>
          <td>0.136450</td>
          <td>26.454927</td>
          <td>0.117509</td>
          <td>26.210595</td>
          <td>0.153494</td>
          <td>26.331497</td>
          <td>0.312107</td>
          <td>25.821203</td>
          <td>0.429505</td>
          <td>0.147404</td>
          <td>0.081710</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.263026</td>
          <td>0.312813</td>
          <td>26.290782</td>
          <td>0.115542</td>
          <td>26.145656</td>
          <td>0.089646</td>
          <td>25.800814</td>
          <td>0.107647</td>
          <td>26.441408</td>
          <td>0.340605</td>
          <td>25.215712</td>
          <td>0.266275</td>
          <td>0.082798</td>
          <td>0.048557</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.356831</td>
          <td>0.704414</td>
          <td>27.300734</td>
          <td>0.271392</td>
          <td>26.648434</td>
          <td>0.138956</td>
          <td>26.513446</td>
          <td>0.198506</td>
          <td>25.954722</td>
          <td>0.229579</td>
          <td>25.844448</td>
          <td>0.437151</td>
          <td>0.099810</td>
          <td>0.067017</td>
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
          <td>29.677894</td>
          <td>2.489429</td>
          <td>26.779408</td>
          <td>0.205532</td>
          <td>26.117755</td>
          <td>0.105037</td>
          <td>25.240892</td>
          <td>0.079672</td>
          <td>24.872053</td>
          <td>0.108833</td>
          <td>24.079667</td>
          <td>0.122285</td>
          <td>0.095365</td>
          <td>0.062376</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>25.971776</td>
          <td>0.277594</td>
          <td>27.652845</td>
          <td>0.411281</td>
          <td>26.966960</td>
          <td>0.215079</td>
          <td>26.265804</td>
          <td>0.191356</td>
          <td>26.077674</td>
          <td>0.297856</td>
          <td>25.730119</td>
          <td>0.467048</td>
          <td>0.069767</td>
          <td>0.037766</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.147185</td>
          <td>0.675781</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.421741</td>
          <td>0.313997</td>
          <td>26.127039</td>
          <td>0.171345</td>
          <td>24.963331</td>
          <td>0.117383</td>
          <td>24.640376</td>
          <td>0.196787</td>
          <td>0.073229</td>
          <td>0.071941</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.179707</td>
          <td>0.255981</td>
          <td>26.051522</td>
          <td>0.159210</td>
          <td>25.475376</td>
          <td>0.180643</td>
          <td>25.810036</td>
          <td>0.494798</td>
          <td>0.056929</td>
          <td>0.042993</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.090716</td>
          <td>0.309804</td>
          <td>26.137615</td>
          <td>0.119750</td>
          <td>25.845569</td>
          <td>0.083456</td>
          <td>25.818251</td>
          <td>0.133200</td>
          <td>25.045329</td>
          <td>0.127653</td>
          <td>25.784427</td>
          <td>0.494781</td>
          <td>0.118856</td>
          <td>0.067069</td>
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
          <td>27.072158</td>
          <td>0.641079</td>
          <td>26.270445</td>
          <td>0.132620</td>
          <td>25.410563</td>
          <td>0.055975</td>
          <td>24.997061</td>
          <td>0.063852</td>
          <td>24.723875</td>
          <td>0.095074</td>
          <td>24.794258</td>
          <td>0.223464</td>
          <td>0.070132</td>
          <td>0.068275</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.535265</td>
          <td>0.169289</td>
          <td>25.962838</td>
          <td>0.092978</td>
          <td>25.079177</td>
          <td>0.070064</td>
          <td>24.853682</td>
          <td>0.108581</td>
          <td>24.209637</td>
          <td>0.138760</td>
          <td>0.131527</td>
          <td>0.067129</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.748242</td>
          <td>1.010258</td>
          <td>26.561866</td>
          <td>0.174716</td>
          <td>26.372907</td>
          <td>0.134267</td>
          <td>26.126099</td>
          <td>0.176047</td>
          <td>26.180885</td>
          <td>0.333878</td>
          <td>25.067163</td>
          <td>0.287522</td>
          <td>0.147404</td>
          <td>0.081710</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.882225</td>
          <td>0.560310</td>
          <td>26.076820</td>
          <td>0.112055</td>
          <td>26.034769</td>
          <td>0.097077</td>
          <td>25.957906</td>
          <td>0.147950</td>
          <td>25.533994</td>
          <td>0.191061</td>
          <td>25.590409</td>
          <td>0.422056</td>
          <td>0.082798</td>
          <td>0.048557</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.521734</td>
          <td>0.868330</td>
          <td>26.334361</td>
          <td>0.141118</td>
          <td>26.819615</td>
          <td>0.192522</td>
          <td>26.819711</td>
          <td>0.305928</td>
          <td>26.148392</td>
          <td>0.319076</td>
          <td>26.742888</td>
          <td>0.943949</td>
          <td>0.099810</td>
          <td>0.067017</td>
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
          <td>26.641291</td>
          <td>0.167397</td>
          <td>26.068650</td>
          <td>0.090773</td>
          <td>25.149751</td>
          <td>0.065989</td>
          <td>24.702026</td>
          <td>0.084591</td>
          <td>24.086522</td>
          <td>0.110735</td>
          <td>0.095365</td>
          <td>0.062376</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.775021</td>
          <td>2.469170</td>
          <td>27.617449</td>
          <td>0.360934</td>
          <td>26.575377</td>
          <td>0.135760</td>
          <td>26.105727</td>
          <td>0.146217</td>
          <td>26.442785</td>
          <td>0.353559</td>
          <td>25.289258</td>
          <td>0.293815</td>
          <td>0.069767</td>
          <td>0.037766</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.460408</td>
          <td>0.782238</td>
          <td>28.498693</td>
          <td>0.702100</td>
          <td>27.663260</td>
          <td>0.343655</td>
          <td>26.249782</td>
          <td>0.169873</td>
          <td>25.014798</td>
          <td>0.109819</td>
          <td>24.413518</td>
          <td>0.145025</td>
          <td>0.073229</td>
          <td>0.071941</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.342219</td>
          <td>0.615852</td>
          <td>27.612064</td>
          <td>0.320182</td>
          <td>26.581218</td>
          <td>0.217202</td>
          <td>25.704807</td>
          <td>0.192280</td>
          <td>25.091748</td>
          <td>0.248444</td>
          <td>0.056929</td>
          <td>0.042993</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.728209</td>
          <td>0.217585</td>
          <td>26.068171</td>
          <td>0.104745</td>
          <td>25.853073</td>
          <td>0.077321</td>
          <td>25.664902</td>
          <td>0.107107</td>
          <td>25.478674</td>
          <td>0.170855</td>
          <td>25.163594</td>
          <td>0.283411</td>
          <td>0.118856</td>
          <td>0.067069</td>
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
          <td>26.396414</td>
          <td>0.133470</td>
          <td>25.461786</td>
          <td>0.052056</td>
          <td>24.959254</td>
          <td>0.054626</td>
          <td>24.999291</td>
          <td>0.107731</td>
          <td>24.546783</td>
          <td>0.161638</td>
          <td>0.070132</td>
          <td>0.068275</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.781360</td>
          <td>0.195832</td>
          <td>26.049175</td>
          <td>0.093287</td>
          <td>25.185434</td>
          <td>0.071346</td>
          <td>25.024133</td>
          <td>0.117234</td>
          <td>24.282839</td>
          <td>0.137360</td>
          <td>0.131527</td>
          <td>0.067129</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.642970</td>
          <td>0.464452</td>
          <td>26.723181</td>
          <td>0.191704</td>
          <td>26.132326</td>
          <td>0.103607</td>
          <td>27.309683</td>
          <td>0.436654</td>
          <td>25.430481</td>
          <td>0.171629</td>
          <td>25.382600</td>
          <td>0.352697</td>
          <td>0.147404</td>
          <td>0.081710</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.854427</td>
          <td>0.510608</td>
          <td>26.134322</td>
          <td>0.106035</td>
          <td>26.219097</td>
          <td>0.101331</td>
          <td>25.726277</td>
          <td>0.107150</td>
          <td>25.332898</td>
          <td>0.143460</td>
          <td>25.319687</td>
          <td>0.306149</td>
          <td>0.082798</td>
          <td>0.048557</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.283391</td>
          <td>0.336567</td>
          <td>26.644583</td>
          <td>0.169066</td>
          <td>26.506942</td>
          <td>0.134134</td>
          <td>26.617414</td>
          <td>0.236304</td>
          <td>26.075759</td>
          <td>0.275360</td>
          <td>26.110003</td>
          <td>0.574014</td>
          <td>0.099810</td>
          <td>0.067017</td>
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
