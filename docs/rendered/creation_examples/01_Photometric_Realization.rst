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

    <pzflow.flow.Flow at 0x7fc99a9e8fa0>



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
    0      23.994413  0.025846  0.018693  
    1      25.391064  0.111933  0.109895  
    2      24.304707  0.215880  0.141358  
    3      25.291103  0.009847  0.009592  
    4      25.096743  0.004110  0.004064  
    ...          ...       ...       ...  
    99995  24.737946  0.017260  0.012025  
    99996  24.224169  0.009489  0.008027  
    99997  25.613836  0.018972  0.015002  
    99998  25.274899  0.022133  0.014871  
    99999  25.699642  0.077039  0.038538  
    
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
          <td>27.049035</td>
          <td>0.568405</td>
          <td>26.376741</td>
          <td>0.124496</td>
          <td>25.978417</td>
          <td>0.077358</td>
          <td>25.124389</td>
          <td>0.059284</td>
          <td>24.710640</td>
          <td>0.078652</td>
          <td>24.082891</td>
          <td>0.101636</td>
          <td>0.025846</td>
          <td>0.018693</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.649322</td>
          <td>2.328482</td>
          <td>28.358869</td>
          <td>0.608809</td>
          <td>26.792882</td>
          <td>0.157314</td>
          <td>26.352768</td>
          <td>0.173297</td>
          <td>25.724753</td>
          <td>0.189395</td>
          <td>26.230381</td>
          <td>0.580749</td>
          <td>0.111933</td>
          <td>0.109895</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.702948</td>
          <td>0.883274</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.451655</td>
          <td>0.587223</td>
          <td>26.138888</td>
          <td>0.144327</td>
          <td>24.874490</td>
          <td>0.090867</td>
          <td>24.305920</td>
          <td>0.123452</td>
          <td>0.215880</td>
          <td>0.141358</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.336249</td>
          <td>1.279663</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.994733</td>
          <td>0.186773</td>
          <td>26.507021</td>
          <td>0.197436</td>
          <td>25.384147</td>
          <td>0.141643</td>
          <td>25.170452</td>
          <td>0.256601</td>
          <td>0.009847</td>
          <td>0.009592</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.392162</td>
          <td>0.346538</td>
          <td>26.124313</td>
          <td>0.099921</td>
          <td>25.968438</td>
          <td>0.076679</td>
          <td>25.792080</td>
          <td>0.106828</td>
          <td>25.386859</td>
          <td>0.141974</td>
          <td>25.065782</td>
          <td>0.235415</td>
          <td>0.004110</td>
          <td>0.004064</td>
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
          <td>26.570048</td>
          <td>0.398016</td>
          <td>26.283356</td>
          <td>0.114798</td>
          <td>25.376787</td>
          <td>0.045377</td>
          <td>25.067341</td>
          <td>0.056357</td>
          <td>24.857627</td>
          <td>0.089529</td>
          <td>24.646087</td>
          <td>0.165440</td>
          <td>0.017260</td>
          <td>0.012025</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.548955</td>
          <td>0.391596</td>
          <td>26.630050</td>
          <td>0.154858</td>
          <td>25.961843</td>
          <td>0.076234</td>
          <td>25.257703</td>
          <td>0.066723</td>
          <td>24.922077</td>
          <td>0.094746</td>
          <td>24.137520</td>
          <td>0.106611</td>
          <td>0.009489</td>
          <td>0.008027</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.734136</td>
          <td>0.450982</td>
          <td>26.697484</td>
          <td>0.164040</td>
          <td>26.407877</td>
          <td>0.112791</td>
          <td>26.246223</td>
          <td>0.158249</td>
          <td>26.227377</td>
          <td>0.287035</td>
          <td>25.720736</td>
          <td>0.397709</td>
          <td>0.018972</td>
          <td>0.015002</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.150529</td>
          <td>0.285786</td>
          <td>26.181679</td>
          <td>0.105060</td>
          <td>26.068227</td>
          <td>0.083738</td>
          <td>25.792341</td>
          <td>0.106853</td>
          <td>25.685129</td>
          <td>0.183159</td>
          <td>25.232407</td>
          <td>0.269925</td>
          <td>0.022133</td>
          <td>0.014871</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.576275</td>
          <td>0.399927</td>
          <td>26.515026</td>
          <td>0.140296</td>
          <td>26.647863</td>
          <td>0.138887</td>
          <td>25.872350</td>
          <td>0.114578</td>
          <td>25.818490</td>
          <td>0.204930</td>
          <td>26.814427</td>
          <td>0.861466</td>
          <td>0.077039</td>
          <td>0.038538</td>
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
          <td>28.958102</td>
          <td>1.854095</td>
          <td>26.870901</td>
          <td>0.218102</td>
          <td>26.096531</td>
          <td>0.101087</td>
          <td>25.074920</td>
          <td>0.067393</td>
          <td>24.778041</td>
          <td>0.098283</td>
          <td>23.939792</td>
          <td>0.106094</td>
          <td>0.025846</td>
          <td>0.018693</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.788669</td>
          <td>2.604330</td>
          <td>27.593872</td>
          <td>0.402399</td>
          <td>26.691880</td>
          <td>0.175536</td>
          <td>26.149414</td>
          <td>0.178611</td>
          <td>26.286285</td>
          <td>0.361046</td>
          <td>25.566506</td>
          <td>0.423686</td>
          <td>0.111933</td>
          <td>0.109895</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.714654</td>
          <td>0.524267</td>
          <td>28.320552</td>
          <td>0.712629</td>
          <td>28.608756</td>
          <td>0.800762</td>
          <td>26.135914</td>
          <td>0.187449</td>
          <td>25.094090</td>
          <td>0.142613</td>
          <td>24.311148</td>
          <td>0.161566</td>
          <td>0.215880</td>
          <td>0.141358</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.506824</td>
          <td>0.420133</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.037576</td>
          <td>0.225895</td>
          <td>25.742484</td>
          <td>0.120956</td>
          <td>25.433919</td>
          <td>0.173015</td>
          <td>25.892140</td>
          <td>0.521863</td>
          <td>0.009847</td>
          <td>0.009592</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.339650</td>
          <td>0.369261</td>
          <td>26.258331</td>
          <td>0.129385</td>
          <td>25.984941</td>
          <td>0.091508</td>
          <td>25.639698</td>
          <td>0.110573</td>
          <td>26.209900</td>
          <td>0.327921</td>
          <td>25.091039</td>
          <td>0.280796</td>
          <td>0.004110</td>
          <td>0.004064</td>
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
          <td>26.298362</td>
          <td>0.134021</td>
          <td>25.423425</td>
          <td>0.055747</td>
          <td>25.196553</td>
          <td>0.074971</td>
          <td>25.125462</td>
          <td>0.132870</td>
          <td>24.145017</td>
          <td>0.126712</td>
          <td>0.017260</td>
          <td>0.012025</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.717932</td>
          <td>0.492312</td>
          <td>26.966626</td>
          <td>0.235833</td>
          <td>25.971767</td>
          <td>0.090474</td>
          <td>25.212027</td>
          <td>0.075964</td>
          <td>24.765183</td>
          <td>0.097039</td>
          <td>24.396205</td>
          <td>0.157233</td>
          <td>0.009489</td>
          <td>0.008027</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.066202</td>
          <td>1.187505</td>
          <td>26.960105</td>
          <td>0.234711</td>
          <td>26.269673</td>
          <td>0.117490</td>
          <td>26.522204</td>
          <td>0.234835</td>
          <td>25.718056</td>
          <td>0.219883</td>
          <td>25.749923</td>
          <td>0.470064</td>
          <td>0.018972</td>
          <td>0.015002</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.462105</td>
          <td>0.406245</td>
          <td>26.070599</td>
          <td>0.110041</td>
          <td>26.192122</td>
          <td>0.109838</td>
          <td>25.774656</td>
          <td>0.124494</td>
          <td>25.777489</td>
          <td>0.231056</td>
          <td>25.240568</td>
          <td>0.317033</td>
          <td>0.022133</td>
          <td>0.014871</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.155012</td>
          <td>0.677315</td>
          <td>26.611067</td>
          <td>0.176946</td>
          <td>26.646360</td>
          <td>0.164385</td>
          <td>26.425732</td>
          <td>0.219199</td>
          <td>25.476033</td>
          <td>0.181413</td>
          <td>25.061204</td>
          <td>0.277315</td>
          <td>0.077039</td>
          <td>0.038538</td>
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
          <td>27.373940</td>
          <td>0.715214</td>
          <td>26.530991</td>
          <td>0.143069</td>
          <td>26.043256</td>
          <td>0.082481</td>
          <td>25.169075</td>
          <td>0.062133</td>
          <td>24.779775</td>
          <td>0.084176</td>
          <td>23.911751</td>
          <td>0.088085</td>
          <td>0.025846</td>
          <td>0.018693</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.925996</td>
          <td>0.494138</td>
          <td>26.876837</td>
          <td>0.194045</td>
          <td>26.212793</td>
          <td>0.177811</td>
          <td>26.466608</td>
          <td>0.394517</td>
          <td>25.616024</td>
          <td>0.417708</td>
          <td>0.111933</td>
          <td>0.109895</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.846032</td>
          <td>0.594954</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.797310</td>
          <td>2.455776</td>
          <td>25.940764</td>
          <td>0.166440</td>
          <td>24.971635</td>
          <td>0.134414</td>
          <td>24.311734</td>
          <td>0.169368</td>
          <td>0.215880</td>
          <td>0.141358</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>30.376558</td>
          <td>2.992699</td>
          <td>27.882198</td>
          <td>0.429642</td>
          <td>27.349269</td>
          <td>0.251298</td>
          <td>26.252763</td>
          <td>0.159345</td>
          <td>25.562788</td>
          <td>0.165286</td>
          <td>25.937715</td>
          <td>0.469476</td>
          <td>0.009847</td>
          <td>0.009592</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.202699</td>
          <td>0.298098</td>
          <td>26.380224</td>
          <td>0.124897</td>
          <td>25.984760</td>
          <td>0.077810</td>
          <td>25.716802</td>
          <td>0.100042</td>
          <td>25.231701</td>
          <td>0.124181</td>
          <td>25.333919</td>
          <td>0.293139</td>
          <td>0.004110</td>
          <td>0.004064</td>
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
          <td>26.579524</td>
          <td>0.401657</td>
          <td>26.263165</td>
          <td>0.113089</td>
          <td>25.503417</td>
          <td>0.050930</td>
          <td>25.080317</td>
          <td>0.057192</td>
          <td>24.887915</td>
          <td>0.092221</td>
          <td>24.626022</td>
          <td>0.163125</td>
          <td>0.017260</td>
          <td>0.012025</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.735184</td>
          <td>0.451618</td>
          <td>26.631964</td>
          <td>0.155249</td>
          <td>26.140013</td>
          <td>0.089296</td>
          <td>25.287130</td>
          <td>0.068561</td>
          <td>24.799662</td>
          <td>0.085166</td>
          <td>24.314490</td>
          <td>0.124507</td>
          <td>0.009489</td>
          <td>0.008027</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>29.082966</td>
          <td>1.848484</td>
          <td>26.418885</td>
          <td>0.129561</td>
          <td>26.346177</td>
          <td>0.107300</td>
          <td>25.825742</td>
          <td>0.110471</td>
          <td>25.846272</td>
          <td>0.210544</td>
          <td>25.693019</td>
          <td>0.390699</td>
          <td>0.018972</td>
          <td>0.015002</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.016566</td>
          <td>0.257089</td>
          <td>26.090616</td>
          <td>0.097421</td>
          <td>26.031607</td>
          <td>0.081469</td>
          <td>25.947696</td>
          <td>0.122951</td>
          <td>26.041540</td>
          <td>0.247760</td>
          <td>25.157277</td>
          <td>0.255030</td>
          <td>0.022133</td>
          <td>0.014871</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.645243</td>
          <td>0.433932</td>
          <td>26.658746</td>
          <td>0.165192</td>
          <td>26.492793</td>
          <td>0.127234</td>
          <td>26.480397</td>
          <td>0.202402</td>
          <td>26.005564</td>
          <td>0.250250</td>
          <td>25.672660</td>
          <td>0.400029</td>
          <td>0.077039</td>
          <td>0.038538</td>
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
