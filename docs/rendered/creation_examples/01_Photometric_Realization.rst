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

    <pzflow.flow.Flow at 0x7fa0fcfe93c0>



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
    0      23.994413  0.088465  0.052761  
    1      25.391064  0.309014  0.279123  
    2      24.304707  0.053432  0.027194  
    3      25.291103  0.067913  0.051263  
    4      25.096743  0.161671  0.091641  
    ...          ...       ...       ...  
    99995  24.737946  0.114350  0.080768  
    99996  24.224169  0.008070  0.006444  
    99997  25.613836  0.026130  0.018875  
    99998  25.274899  0.124291  0.074944  
    99999  25.699642  0.017436  0.015064  
    
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
          <td>26.804745</td>
          <td>0.475468</td>
          <td>26.581738</td>
          <td>0.148577</td>
          <td>25.951458</td>
          <td>0.075537</td>
          <td>25.147291</td>
          <td>0.060501</td>
          <td>24.590692</td>
          <td>0.070741</td>
          <td>23.960369</td>
          <td>0.091279</td>
          <td>0.088465</td>
          <td>0.052761</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.897739</td>
          <td>0.995937</td>
          <td>27.773117</td>
          <td>0.394825</td>
          <td>26.690956</td>
          <td>0.144140</td>
          <td>26.477164</td>
          <td>0.192536</td>
          <td>26.195738</td>
          <td>0.279773</td>
          <td>25.319568</td>
          <td>0.289701</td>
          <td>0.309014</td>
          <td>0.279123</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.022866</td>
          <td>0.557826</td>
          <td>28.219790</td>
          <td>0.551310</td>
          <td>28.352888</td>
          <td>0.547063</td>
          <td>26.337675</td>
          <td>0.171087</td>
          <td>25.059630</td>
          <td>0.106877</td>
          <td>24.348292</td>
          <td>0.128072</td>
          <td>0.053432</td>
          <td>0.027194</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.504555</td>
          <td>0.284884</td>
          <td>26.014731</td>
          <td>0.129661</td>
          <td>25.495431</td>
          <td>0.155847</td>
          <td>25.182154</td>
          <td>0.259072</td>
          <td>0.067913</td>
          <td>0.051263</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.205825</td>
          <td>0.298805</td>
          <td>26.074554</td>
          <td>0.095660</td>
          <td>26.009718</td>
          <td>0.079526</td>
          <td>25.783487</td>
          <td>0.106029</td>
          <td>25.509632</td>
          <td>0.157753</td>
          <td>24.762833</td>
          <td>0.182688</td>
          <td>0.161671</td>
          <td>0.091641</td>
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
          <td>27.414589</td>
          <td>0.732346</td>
          <td>26.166700</td>
          <td>0.103694</td>
          <td>25.388258</td>
          <td>0.045841</td>
          <td>25.015334</td>
          <td>0.053814</td>
          <td>24.758001</td>
          <td>0.082009</td>
          <td>24.624006</td>
          <td>0.162352</td>
          <td>0.114350</td>
          <td>0.080768</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.695067</td>
          <td>0.437877</td>
          <td>26.837899</td>
          <td>0.184805</td>
          <td>26.039623</td>
          <td>0.081652</td>
          <td>25.286521</td>
          <td>0.068448</td>
          <td>24.872590</td>
          <td>0.090715</td>
          <td>24.269931</td>
          <td>0.119653</td>
          <td>0.008070</td>
          <td>0.006444</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.643009</td>
          <td>0.420898</td>
          <td>26.646479</td>
          <td>0.157049</td>
          <td>26.220643</td>
          <td>0.095751</td>
          <td>26.509136</td>
          <td>0.197788</td>
          <td>25.978044</td>
          <td>0.234058</td>
          <td>25.041078</td>
          <td>0.230649</td>
          <td>0.026130</td>
          <td>0.018875</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.517197</td>
          <td>0.382093</td>
          <td>26.039615</td>
          <td>0.092773</td>
          <td>26.158647</td>
          <td>0.090676</td>
          <td>25.919001</td>
          <td>0.119326</td>
          <td>25.546645</td>
          <td>0.162822</td>
          <td>25.362495</td>
          <td>0.299899</td>
          <td>0.124291</td>
          <td>0.074944</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.978401</td>
          <td>1.045061</td>
          <td>27.004140</td>
          <td>0.212502</td>
          <td>26.477427</td>
          <td>0.119831</td>
          <td>26.372057</td>
          <td>0.176159</td>
          <td>26.242413</td>
          <td>0.290544</td>
          <td>25.854264</td>
          <td>0.440413</td>
          <td>0.017436</td>
          <td>0.015064</td>
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
          <td>26.711808</td>
          <td>0.495598</td>
          <td>26.922586</td>
          <td>0.230790</td>
          <td>25.964409</td>
          <td>0.091472</td>
          <td>25.283580</td>
          <td>0.082395</td>
          <td>24.840253</td>
          <td>0.105444</td>
          <td>24.093393</td>
          <td>0.123264</td>
          <td>0.088465</td>
          <td>0.052761</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.070627</td>
          <td>0.731554</td>
          <td>26.709819</td>
          <td>0.232652</td>
          <td>26.456185</td>
          <td>0.172588</td>
          <td>26.789048</td>
          <td>0.361265</td>
          <td>26.530513</td>
          <td>0.511834</td>
          <td>25.683590</td>
          <td>0.544381</td>
          <td>0.309014</td>
          <td>0.279123</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.245376</td>
          <td>0.344374</td>
          <td>27.876183</td>
          <td>0.485053</td>
          <td>28.087805</td>
          <td>0.518192</td>
          <td>25.958635</td>
          <td>0.146653</td>
          <td>24.980197</td>
          <td>0.117763</td>
          <td>24.186660</td>
          <td>0.132072</td>
          <td>0.053432</td>
          <td>0.027194</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>32.601527</td>
          <td>5.297510</td>
          <td>28.931619</td>
          <td>0.993475</td>
          <td>27.240835</td>
          <td>0.269989</td>
          <td>26.719008</td>
          <td>0.278926</td>
          <td>25.197249</td>
          <td>0.142949</td>
          <td>25.066595</td>
          <td>0.278464</td>
          <td>0.067913</td>
          <td>0.051263</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.541047</td>
          <td>0.447095</td>
          <td>26.355693</td>
          <td>0.147730</td>
          <td>25.855767</td>
          <td>0.086286</td>
          <td>25.754576</td>
          <td>0.129210</td>
          <td>25.315290</td>
          <td>0.164863</td>
          <td>25.063895</td>
          <td>0.289261</td>
          <td>0.161671</td>
          <td>0.091641</td>
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
          <td>26.468509</td>
          <td>0.159476</td>
          <td>25.512757</td>
          <td>0.062292</td>
          <td>25.104818</td>
          <td>0.071434</td>
          <td>24.860903</td>
          <td>0.108922</td>
          <td>25.129872</td>
          <td>0.298657</td>
          <td>0.114350</td>
          <td>0.080768</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>25.971183</td>
          <td>0.275439</td>
          <td>26.607178</td>
          <td>0.174490</td>
          <td>26.113783</td>
          <td>0.102466</td>
          <td>25.250601</td>
          <td>0.078589</td>
          <td>24.784236</td>
          <td>0.098665</td>
          <td>24.022998</td>
          <td>0.113901</td>
          <td>0.008070</td>
          <td>0.006444</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.765108</td>
          <td>0.510212</td>
          <td>26.350844</td>
          <td>0.140348</td>
          <td>26.489687</td>
          <td>0.142244</td>
          <td>26.465897</td>
          <td>0.224293</td>
          <td>25.683204</td>
          <td>0.213743</td>
          <td>25.386004</td>
          <td>0.355893</td>
          <td>0.026130</td>
          <td>0.018875</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.244830</td>
          <td>0.350992</td>
          <td>26.084441</td>
          <td>0.114743</td>
          <td>26.004250</td>
          <td>0.096323</td>
          <td>25.930765</td>
          <td>0.147338</td>
          <td>26.148676</td>
          <td>0.322176</td>
          <td>25.044416</td>
          <td>0.279309</td>
          <td>0.124291</td>
          <td>0.074944</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.748353</td>
          <td>0.986830</td>
          <td>26.749813</td>
          <td>0.196950</td>
          <td>26.508648</td>
          <td>0.144462</td>
          <td>26.206941</td>
          <td>0.180327</td>
          <td>25.844755</td>
          <td>0.244190</td>
          <td>26.913338</td>
          <td>1.029600</td>
          <td>0.017436</td>
          <td>0.015064</td>
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
          <td>26.397541</td>
          <td>0.362910</td>
          <td>26.512172</td>
          <td>0.148134</td>
          <td>26.154935</td>
          <td>0.096573</td>
          <td>25.201545</td>
          <td>0.068088</td>
          <td>24.581683</td>
          <td>0.075020</td>
          <td>23.941774</td>
          <td>0.096185</td>
          <td>0.088465</td>
          <td>0.052761</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.118096</td>
          <td>0.787872</td>
          <td>27.057650</td>
          <td>0.341675</td>
          <td>26.120169</td>
          <td>0.254733</td>
          <td>26.084938</td>
          <td>0.435461</td>
          <td>24.814551</td>
          <td>0.335710</td>
          <td>0.309014</td>
          <td>0.279123</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.681815</td>
          <td>0.374587</td>
          <td>29.304132</td>
          <td>1.042598</td>
          <td>26.165374</td>
          <td>0.151272</td>
          <td>24.995687</td>
          <td>0.103469</td>
          <td>24.302396</td>
          <td>0.126094</td>
          <td>0.053432</td>
          <td>0.027194</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.392597</td>
          <td>0.739886</td>
          <td>28.649347</td>
          <td>0.766235</td>
          <td>27.289365</td>
          <td>0.249599</td>
          <td>26.313864</td>
          <td>0.175781</td>
          <td>25.555370</td>
          <td>0.171625</td>
          <td>25.411680</td>
          <td>0.325977</td>
          <td>0.067913</td>
          <td>0.051263</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.301020</td>
          <td>0.364193</td>
          <td>26.013970</td>
          <td>0.106825</td>
          <td>26.085515</td>
          <td>0.102211</td>
          <td>25.560260</td>
          <td>0.105547</td>
          <td>25.569878</td>
          <td>0.198263</td>
          <td>25.962966</td>
          <td>0.559693</td>
          <td>0.161671</td>
          <td>0.091641</td>
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
          <td>26.949498</td>
          <td>0.566597</td>
          <td>26.465582</td>
          <td>0.148640</td>
          <td>25.359264</td>
          <td>0.050262</td>
          <td>25.097722</td>
          <td>0.065459</td>
          <td>24.793734</td>
          <td>0.095076</td>
          <td>24.951975</td>
          <td>0.239936</td>
          <td>0.114350</td>
          <td>0.080768</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.669082</td>
          <td>0.864874</td>
          <td>26.517096</td>
          <td>0.140633</td>
          <td>26.039390</td>
          <td>0.081695</td>
          <td>25.137854</td>
          <td>0.060043</td>
          <td>24.927353</td>
          <td>0.095255</td>
          <td>24.167717</td>
          <td>0.109542</td>
          <td>0.008070</td>
          <td>0.006444</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.332821</td>
          <td>0.332125</td>
          <td>26.545164</td>
          <td>0.144841</td>
          <td>26.164966</td>
          <td>0.091823</td>
          <td>26.444057</td>
          <td>0.188560</td>
          <td>25.692542</td>
          <td>0.185557</td>
          <td>25.018450</td>
          <td>0.227922</td>
          <td>0.026130</td>
          <td>0.018875</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.996902</td>
          <td>0.588428</td>
          <td>26.484611</td>
          <td>0.151927</td>
          <td>26.214804</td>
          <td>0.107636</td>
          <td>25.846519</td>
          <td>0.127131</td>
          <td>26.625957</td>
          <td>0.437609</td>
          <td>25.187558</td>
          <td>0.292528</td>
          <td>0.124291</td>
          <td>0.074944</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.889070</td>
          <td>0.507144</td>
          <td>26.977960</td>
          <td>0.208521</td>
          <td>26.680380</td>
          <td>0.143338</td>
          <td>26.242218</td>
          <td>0.158288</td>
          <td>25.865066</td>
          <td>0.213802</td>
          <td>25.808774</td>
          <td>0.426841</td>
          <td>0.017436</td>
          <td>0.015064</td>
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
