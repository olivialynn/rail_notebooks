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

    <pzflow.flow.Flow at 0x7f7bc48eefb0>



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
          <td>28.490024</td>
          <td>1.388368</td>
          <td>26.501482</td>
          <td>0.138668</td>
          <td>25.928346</td>
          <td>0.074010</td>
          <td>25.342454</td>
          <td>0.071923</td>
          <td>25.083317</td>
          <td>0.109111</td>
          <td>25.086214</td>
          <td>0.239423</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.196676</td>
          <td>1.048383</td>
          <td>28.135883</td>
          <td>0.466306</td>
          <td>27.511182</td>
          <td>0.442016</td>
          <td>26.799455</td>
          <td>0.449129</td>
          <td>25.533286</td>
          <td>0.343610</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.534187</td>
          <td>0.387152</td>
          <td>25.869587</td>
          <td>0.079889</td>
          <td>24.812427</td>
          <td>0.027576</td>
          <td>23.848630</td>
          <td>0.019370</td>
          <td>23.167292</td>
          <td>0.020246</td>
          <td>22.846511</td>
          <td>0.034051</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.470591</td>
          <td>0.311262</td>
          <td>27.367646</td>
          <td>0.254811</td>
          <td>26.954857</td>
          <td>0.285783</td>
          <td>26.149460</td>
          <td>0.269442</td>
          <td>25.305672</td>
          <td>0.286465</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>27.969058</td>
          <td>1.039298</td>
          <td>25.807223</td>
          <td>0.075615</td>
          <td>25.446729</td>
          <td>0.048283</td>
          <td>24.846503</td>
          <td>0.046322</td>
          <td>24.397072</td>
          <td>0.059588</td>
          <td>23.692814</td>
          <td>0.072092</td>
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
          <td>26.122298</td>
          <td>0.279333</td>
          <td>26.414237</td>
          <td>0.128605</td>
          <td>26.052962</td>
          <td>0.082618</td>
          <td>26.187297</td>
          <td>0.150458</td>
          <td>25.735947</td>
          <td>0.191192</td>
          <td>25.581113</td>
          <td>0.356785</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>29.158672</td>
          <td>1.908192</td>
          <td>27.129427</td>
          <td>0.235813</td>
          <td>27.116331</td>
          <td>0.206888</td>
          <td>26.195122</td>
          <td>0.151471</td>
          <td>26.194807</td>
          <td>0.279562</td>
          <td>25.701587</td>
          <td>0.391875</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.284088</td>
          <td>0.670325</td>
          <td>26.916915</td>
          <td>0.197529</td>
          <td>26.688491</td>
          <td>0.143835</td>
          <td>26.378154</td>
          <td>0.177073</td>
          <td>25.580365</td>
          <td>0.167572</td>
          <td>25.680235</td>
          <td>0.385453</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.137678</td>
          <td>0.605376</td>
          <td>26.864450</td>
          <td>0.188994</td>
          <td>26.488975</td>
          <td>0.121040</td>
          <td>25.865653</td>
          <td>0.113912</td>
          <td>25.622435</td>
          <td>0.173678</td>
          <td>25.264056</td>
          <td>0.276964</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.059271</td>
          <td>0.265387</td>
          <td>26.620836</td>
          <td>0.153641</td>
          <td>25.982554</td>
          <td>0.077641</td>
          <td>25.811244</td>
          <td>0.108632</td>
          <td>25.085088</td>
          <td>0.109279</td>
          <td>25.014473</td>
          <td>0.225613</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>27.045820</td>
          <td>0.251678</td>
          <td>25.992145</td>
          <td>0.092087</td>
          <td>25.443741</td>
          <td>0.093142</td>
          <td>25.150832</td>
          <td>0.135717</td>
          <td>25.131645</td>
          <td>0.290167</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.524139</td>
          <td>1.376827</td>
          <td>27.538364</td>
          <td>0.339119</td>
          <td>27.071635</td>
          <td>0.365320</td>
          <td>26.084821</td>
          <td>0.296754</td>
          <td>26.590957</td>
          <td>0.843295</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.481701</td>
          <td>0.845737</td>
          <td>26.065671</td>
          <td>0.111668</td>
          <td>24.797696</td>
          <td>0.032743</td>
          <td>23.914833</td>
          <td>0.024743</td>
          <td>23.149983</td>
          <td>0.023897</td>
          <td>22.833380</td>
          <td>0.040781</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>25.917257</td>
          <td>0.276552</td>
          <td>27.646172</td>
          <td>0.427866</td>
          <td>27.091122</td>
          <td>0.251417</td>
          <td>26.519853</td>
          <td>0.249903</td>
          <td>26.290691</td>
          <td>0.371072</td>
          <td>25.298842</td>
          <td>0.352878</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.900579</td>
          <td>0.260088</td>
          <td>25.638961</td>
          <td>0.075300</td>
          <td>25.481319</td>
          <td>0.058661</td>
          <td>24.727621</td>
          <td>0.049458</td>
          <td>24.497441</td>
          <td>0.076675</td>
          <td>23.606541</td>
          <td>0.079061</td>
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
          <td>26.329284</td>
          <td>0.371528</td>
          <td>26.505280</td>
          <td>0.162949</td>
          <td>26.166505</td>
          <td>0.109547</td>
          <td>26.347815</td>
          <td>0.207136</td>
          <td>25.754726</td>
          <td>0.231012</td>
          <td>25.561094</td>
          <td>0.414790</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.163298</td>
          <td>0.277971</td>
          <td>27.028798</td>
          <td>0.225078</td>
          <td>26.139711</td>
          <td>0.170888</td>
          <td>26.132865</td>
          <td>0.309543</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>31.031066</td>
          <td>3.754099</td>
          <td>27.416395</td>
          <td>0.342813</td>
          <td>26.950306</td>
          <td>0.212562</td>
          <td>26.554167</td>
          <td>0.243901</td>
          <td>25.483021</td>
          <td>0.182558</td>
          <td>24.999089</td>
          <td>0.263734</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.191815</td>
          <td>0.290987</td>
          <td>26.512810</td>
          <td>0.149377</td>
          <td>25.833828</td>
          <td>0.135137</td>
          <td>25.642091</td>
          <td>0.212433</td>
          <td>25.843277</td>
          <td>0.517107</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.517008</td>
          <td>0.163010</td>
          <td>26.076281</td>
          <td>0.100140</td>
          <td>25.756279</td>
          <td>0.123640</td>
          <td>25.129613</td>
          <td>0.134574</td>
          <td>24.482607</td>
          <td>0.170922</td>
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
          <td>28.071001</td>
          <td>1.103263</td>
          <td>26.495444</td>
          <td>0.137964</td>
          <td>26.199761</td>
          <td>0.094024</td>
          <td>25.273862</td>
          <td>0.067695</td>
          <td>25.164826</td>
          <td>0.117159</td>
          <td>24.921318</td>
          <td>0.208782</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.699672</td>
          <td>1.544128</td>
          <td>28.426165</td>
          <td>0.638607</td>
          <td>27.138055</td>
          <td>0.210873</td>
          <td>29.019697</td>
          <td>1.199150</td>
          <td>27.258859</td>
          <td>0.627797</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.107171</td>
          <td>0.291212</td>
          <td>25.898534</td>
          <td>0.088089</td>
          <td>24.757202</td>
          <td>0.028520</td>
          <td>23.907755</td>
          <td>0.022146</td>
          <td>23.149420</td>
          <td>0.021601</td>
          <td>22.813787</td>
          <td>0.036037</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.925911</td>
          <td>0.525608</td>
          <td>26.780645</td>
          <td>0.193525</td>
          <td>27.243566</td>
          <td>0.441549</td>
          <td>25.731973</td>
          <td>0.235969</td>
          <td>25.414603</td>
          <td>0.384987</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.836540</td>
          <td>0.221143</td>
          <td>25.825874</td>
          <td>0.076964</td>
          <td>25.455513</td>
          <td>0.048732</td>
          <td>24.769207</td>
          <td>0.043316</td>
          <td>24.427173</td>
          <td>0.061288</td>
          <td>23.809345</td>
          <td>0.080029</td>
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
          <td>26.867469</td>
          <td>0.521889</td>
          <td>26.580682</td>
          <td>0.158769</td>
          <td>26.206927</td>
          <td>0.102341</td>
          <td>25.832029</td>
          <td>0.120034</td>
          <td>25.444776</td>
          <td>0.161112</td>
          <td>25.240315</td>
          <td>0.292884</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.067886</td>
          <td>1.109544</td>
          <td>27.018729</td>
          <td>0.218047</td>
          <td>26.679989</td>
          <td>0.145109</td>
          <td>26.260322</td>
          <td>0.162881</td>
          <td>26.139688</td>
          <td>0.271413</td>
          <td>25.048288</td>
          <td>0.235799</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.649835</td>
          <td>1.535472</td>
          <td>27.787815</td>
          <td>0.414495</td>
          <td>26.712235</td>
          <td>0.153987</td>
          <td>26.644518</td>
          <td>0.232487</td>
          <td>26.637406</td>
          <td>0.414229</td>
          <td>25.084923</td>
          <td>0.250813</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.222190</td>
          <td>0.279501</td>
          <td>26.656569</td>
          <td>0.156535</td>
          <td>25.724171</td>
          <td>0.113408</td>
          <td>25.639474</td>
          <td>0.196724</td>
          <td>25.506238</td>
          <td>0.373946</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.558725</td>
          <td>1.461200</td>
          <td>26.482634</td>
          <td>0.141051</td>
          <td>26.200307</td>
          <td>0.097787</td>
          <td>25.492180</td>
          <td>0.085526</td>
          <td>25.231634</td>
          <td>0.129000</td>
          <td>24.837501</td>
          <td>0.202236</td>
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
