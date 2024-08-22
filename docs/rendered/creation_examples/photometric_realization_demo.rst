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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7fc18ae2e620>



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
          <td>25.883131</td>
          <td>0.229639</td>
          <td>26.889432</td>
          <td>0.193015</td>
          <td>25.973870</td>
          <td>0.077048</td>
          <td>25.316198</td>
          <td>0.070271</td>
          <td>24.823986</td>
          <td>0.086918</td>
          <td>25.280281</td>
          <td>0.280635</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.673939</td>
          <td>0.365562</td>
          <td>27.978200</td>
          <td>0.413836</td>
          <td>27.314463</td>
          <td>0.380140</td>
          <td>26.540845</td>
          <td>0.368270</td>
          <td>25.980890</td>
          <td>0.484277</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.990162</td>
          <td>0.250809</td>
          <td>25.880368</td>
          <td>0.080652</td>
          <td>24.799119</td>
          <td>0.027257</td>
          <td>23.881554</td>
          <td>0.019917</td>
          <td>23.132394</td>
          <td>0.019656</td>
          <td>22.843447</td>
          <td>0.033959</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.620415</td>
          <td>0.350542</td>
          <td>27.444468</td>
          <td>0.271320</td>
          <td>26.490867</td>
          <td>0.194771</td>
          <td>26.345446</td>
          <td>0.315606</td>
          <td>25.364364</td>
          <td>0.300349</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.732810</td>
          <td>0.450531</td>
          <td>25.976630</td>
          <td>0.087781</td>
          <td>25.334566</td>
          <td>0.043708</td>
          <td>24.865600</td>
          <td>0.047114</td>
          <td>24.397439</td>
          <td>0.059607</td>
          <td>23.596174</td>
          <td>0.066181</td>
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
          <td>26.544005</td>
          <td>0.390102</td>
          <td>26.308461</td>
          <td>0.117332</td>
          <td>26.200469</td>
          <td>0.094070</td>
          <td>26.267763</td>
          <td>0.161189</td>
          <td>25.605327</td>
          <td>0.171170</td>
          <td>25.205049</td>
          <td>0.263968</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.957350</td>
          <td>0.532004</td>
          <td>27.142024</td>
          <td>0.238280</td>
          <td>26.877469</td>
          <td>0.169092</td>
          <td>26.299252</td>
          <td>0.165580</td>
          <td>26.106806</td>
          <td>0.260220</td>
          <td>25.351330</td>
          <td>0.297217</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.037638</td>
          <td>0.563779</td>
          <td>27.252767</td>
          <td>0.260978</td>
          <td>26.937320</td>
          <td>0.177913</td>
          <td>26.494198</td>
          <td>0.195318</td>
          <td>26.233862</td>
          <td>0.288544</td>
          <td>25.673102</td>
          <td>0.383328</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.723595</td>
          <td>0.894811</td>
          <td>26.970144</td>
          <td>0.206548</td>
          <td>26.691615</td>
          <td>0.144222</td>
          <td>25.699342</td>
          <td>0.098499</td>
          <td>25.603585</td>
          <td>0.170917</td>
          <td>27.215854</td>
          <td>1.099365</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.524478</td>
          <td>0.384254</td>
          <td>26.653086</td>
          <td>0.157939</td>
          <td>26.022732</td>
          <td>0.080445</td>
          <td>25.694036</td>
          <td>0.098042</td>
          <td>25.306755</td>
          <td>0.132492</td>
          <td>25.599294</td>
          <td>0.361905</td>
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
          <td>27.421709</td>
          <td>0.803530</td>
          <td>26.694126</td>
          <td>0.187787</td>
          <td>26.023519</td>
          <td>0.094659</td>
          <td>25.249999</td>
          <td>0.078535</td>
          <td>24.826304</td>
          <td>0.102352</td>
          <td>24.708080</td>
          <td>0.204728</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.500310</td>
          <td>1.494021</td>
          <td>31.354430</td>
          <td>2.924318</td>
          <td>28.058296</td>
          <td>0.504620</td>
          <td>26.571499</td>
          <td>0.244408</td>
          <td>27.097183</td>
          <td>0.637064</td>
          <td>26.492395</td>
          <td>0.791189</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.526465</td>
          <td>0.432715</td>
          <td>26.079504</td>
          <td>0.113020</td>
          <td>24.783452</td>
          <td>0.032335</td>
          <td>23.874952</td>
          <td>0.023905</td>
          <td>23.118651</td>
          <td>0.023261</td>
          <td>22.816120</td>
          <td>0.040163</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.812619</td>
          <td>0.956623</td>
          <td>28.180357</td>
          <td>0.582539</td>
          <td>26.512555</td>
          <td>0.248409</td>
          <td>26.789337</td>
          <td>0.540262</td>
          <td>25.663245</td>
          <td>0.466751</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.341552</td>
          <td>0.369884</td>
          <td>25.804732</td>
          <td>0.087130</td>
          <td>25.396654</td>
          <td>0.054417</td>
          <td>24.803645</td>
          <td>0.052910</td>
          <td>24.368916</td>
          <td>0.068441</td>
          <td>23.626478</td>
          <td>0.080464</td>
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
          <td>30.684303</td>
          <td>3.426977</td>
          <td>26.082864</td>
          <td>0.113217</td>
          <td>25.987599</td>
          <td>0.093668</td>
          <td>25.899126</td>
          <td>0.141461</td>
          <td>25.752972</td>
          <td>0.230677</td>
          <td>24.966149</td>
          <td>0.258748</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.148187</td>
          <td>0.274581</td>
          <td>27.571187</td>
          <td>0.349253</td>
          <td>26.408281</td>
          <td>0.214296</td>
          <td>26.382621</td>
          <td>0.377001</td>
          <td>25.459504</td>
          <td>0.377760</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.917946</td>
          <td>0.573913</td>
          <td>27.051962</td>
          <td>0.255688</td>
          <td>27.114628</td>
          <td>0.243611</td>
          <td>26.473172</td>
          <td>0.228103</td>
          <td>26.244774</td>
          <td>0.341000</td>
          <td>25.003901</td>
          <td>0.264772</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.406473</td>
          <td>0.345326</td>
          <td>26.570007</td>
          <td>0.156881</td>
          <td>25.735606</td>
          <td>0.124124</td>
          <td>25.562803</td>
          <td>0.198781</td>
          <td>26.169908</td>
          <td>0.652618</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.435992</td>
          <td>0.152106</td>
          <td>26.181740</td>
          <td>0.109810</td>
          <td>25.453848</td>
          <td>0.094954</td>
          <td>25.215293</td>
          <td>0.144887</td>
          <td>25.500089</td>
          <td>0.391907</td>
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
          <td>27.465484</td>
          <td>0.757648</td>
          <td>26.853948</td>
          <td>0.187347</td>
          <td>25.931312</td>
          <td>0.074214</td>
          <td>25.328384</td>
          <td>0.071043</td>
          <td>24.883734</td>
          <td>0.091620</td>
          <td>25.187072</td>
          <td>0.260149</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.740822</td>
          <td>0.385367</td>
          <td>27.709185</td>
          <td>0.335892</td>
          <td>27.541810</td>
          <td>0.452736</td>
          <td>26.554605</td>
          <td>0.372555</td>
          <td>26.043457</td>
          <td>0.507616</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.092494</td>
          <td>0.614564</td>
          <td>25.833767</td>
          <td>0.083212</td>
          <td>24.814128</td>
          <td>0.029979</td>
          <td>23.868054</td>
          <td>0.021406</td>
          <td>23.157644</td>
          <td>0.021753</td>
          <td>22.871776</td>
          <td>0.037933</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.156300</td>
          <td>0.699026</td>
          <td>28.216871</td>
          <td>0.646575</td>
          <td>27.913310</td>
          <td>0.478092</td>
          <td>26.865079</td>
          <td>0.329216</td>
          <td>26.294648</td>
          <td>0.371042</td>
          <td>25.092526</td>
          <td>0.298490</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.277349</td>
          <td>0.316691</td>
          <td>25.717670</td>
          <td>0.069953</td>
          <td>25.444353</td>
          <td>0.048251</td>
          <td>24.830398</td>
          <td>0.045734</td>
          <td>24.385068</td>
          <td>0.059041</td>
          <td>23.814370</td>
          <td>0.080384</td>
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
          <td>26.957999</td>
          <td>0.557294</td>
          <td>26.705934</td>
          <td>0.176631</td>
          <td>26.192312</td>
          <td>0.101040</td>
          <td>26.180811</td>
          <td>0.162125</td>
          <td>26.204037</td>
          <td>0.302741</td>
          <td>25.912734</td>
          <td>0.493193</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.052142</td>
          <td>0.575017</td>
          <td>26.978626</td>
          <td>0.210873</td>
          <td>26.756427</td>
          <td>0.154948</td>
          <td>27.244371</td>
          <td>0.365493</td>
          <td>25.810324</td>
          <td>0.206751</td>
          <td>25.440250</td>
          <td>0.324152</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.270630</td>
          <td>0.324700</td>
          <td>27.442129</td>
          <td>0.316310</td>
          <td>27.164679</td>
          <td>0.225636</td>
          <td>26.628874</td>
          <td>0.229493</td>
          <td>26.191210</td>
          <td>0.291572</td>
          <td>25.273976</td>
          <td>0.292550</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.002643</td>
          <td>0.587827</td>
          <td>27.533721</td>
          <td>0.358349</td>
          <td>26.803660</td>
          <td>0.177435</td>
          <td>26.301152</td>
          <td>0.186190</td>
          <td>25.640813</td>
          <td>0.196945</td>
          <td>27.415269</td>
          <td>1.323754</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.877377</td>
          <td>0.513323</td>
          <td>26.574796</td>
          <td>0.152668</td>
          <td>26.023702</td>
          <td>0.083724</td>
          <td>25.662867</td>
          <td>0.099365</td>
          <td>25.198490</td>
          <td>0.125348</td>
          <td>24.842880</td>
          <td>0.203151</td>
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
