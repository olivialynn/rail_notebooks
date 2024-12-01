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

    <pzflow.flow.Flow at 0x7f0b6c562a40>



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
          <td>28.460746</td>
          <td>1.367325</td>
          <td>26.826642</td>
          <td>0.183054</td>
          <td>26.033057</td>
          <td>0.081181</td>
          <td>25.188570</td>
          <td>0.062757</td>
          <td>25.232024</td>
          <td>0.124187</td>
          <td>25.037454</td>
          <td>0.229957</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.184122</td>
          <td>0.625451</td>
          <td>28.234375</td>
          <td>0.557137</td>
          <td>28.094813</td>
          <td>0.452148</td>
          <td>27.851642</td>
          <td>0.568080</td>
          <td>27.280716</td>
          <td>0.636970</td>
          <td>27.721297</td>
          <td>1.447042</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.637686</td>
          <td>0.419193</td>
          <td>26.060934</td>
          <td>0.094524</td>
          <td>24.804454</td>
          <td>0.027384</td>
          <td>23.859226</td>
          <td>0.019544</td>
          <td>23.161080</td>
          <td>0.020139</td>
          <td>22.829333</td>
          <td>0.033539</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.581001</td>
          <td>1.454757</td>
          <td>29.078517</td>
          <td>0.976757</td>
          <td>27.465831</td>
          <td>0.276076</td>
          <td>26.466770</td>
          <td>0.190856</td>
          <td>25.925074</td>
          <td>0.223998</td>
          <td>25.806619</td>
          <td>0.424764</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.197835</td>
          <td>0.296892</td>
          <td>25.733857</td>
          <td>0.070872</td>
          <td>25.462716</td>
          <td>0.048974</td>
          <td>24.772845</td>
          <td>0.043391</td>
          <td>24.289788</td>
          <td>0.054176</td>
          <td>23.688159</td>
          <td>0.071796</td>
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
          <td>26.847956</td>
          <td>0.490970</td>
          <td>26.521661</td>
          <td>0.141099</td>
          <td>26.189949</td>
          <td>0.093205</td>
          <td>26.119268</td>
          <td>0.141910</td>
          <td>25.793532</td>
          <td>0.200684</td>
          <td>25.914890</td>
          <td>0.460997</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.776741</td>
          <td>0.175479</td>
          <td>26.881879</td>
          <td>0.169728</td>
          <td>26.459228</td>
          <td>0.189646</td>
          <td>25.696484</td>
          <td>0.184927</td>
          <td>25.771900</td>
          <td>0.413650</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.230220</td>
          <td>0.645861</td>
          <td>27.535992</td>
          <td>0.327917</td>
          <td>26.813882</td>
          <td>0.160165</td>
          <td>26.159126</td>
          <td>0.146861</td>
          <td>25.947994</td>
          <td>0.228302</td>
          <td>25.328543</td>
          <td>0.291808</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.048077</td>
          <td>0.220429</td>
          <td>26.766982</td>
          <td>0.153864</td>
          <td>25.954131</td>
          <td>0.123024</td>
          <td>25.766742</td>
          <td>0.196216</td>
          <td>25.028654</td>
          <td>0.228285</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.936353</td>
          <td>0.523927</td>
          <td>26.415366</td>
          <td>0.128731</td>
          <td>26.023501</td>
          <td>0.080499</td>
          <td>25.683209</td>
          <td>0.097115</td>
          <td>25.260912</td>
          <td>0.127337</td>
          <td>24.727009</td>
          <td>0.177227</td>
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
          <td>26.667684</td>
          <td>0.474215</td>
          <td>26.990083</td>
          <td>0.240398</td>
          <td>26.133201</td>
          <td>0.104206</td>
          <td>25.221482</td>
          <td>0.076583</td>
          <td>25.141142</td>
          <td>0.134587</td>
          <td>24.777220</td>
          <td>0.216908</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.525216</td>
          <td>1.512685</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.424515</td>
          <td>0.309752</td>
          <td>26.952095</td>
          <td>0.332510</td>
          <td>27.949874</td>
          <td>1.098408</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.652378</td>
          <td>0.215325</td>
          <td>25.934412</td>
          <td>0.099581</td>
          <td>24.799952</td>
          <td>0.032808</td>
          <td>23.835068</td>
          <td>0.023098</td>
          <td>23.139953</td>
          <td>0.023691</td>
          <td>22.810980</td>
          <td>0.039981</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.658608</td>
          <td>0.431928</td>
          <td>27.341024</td>
          <td>0.307935</td>
          <td>27.062424</td>
          <td>0.385602</td>
          <td>25.784594</td>
          <td>0.247261</td>
          <td>25.038177</td>
          <td>0.286649</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.445029</td>
          <td>0.400726</td>
          <td>25.957046</td>
          <td>0.099578</td>
          <td>25.312241</td>
          <td>0.050490</td>
          <td>24.921404</td>
          <td>0.058737</td>
          <td>24.299207</td>
          <td>0.064344</td>
          <td>23.709823</td>
          <td>0.086594</td>
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
          <td>26.380654</td>
          <td>0.386637</td>
          <td>26.324515</td>
          <td>0.139565</td>
          <td>26.266392</td>
          <td>0.119503</td>
          <td>26.140863</td>
          <td>0.173960</td>
          <td>25.689790</td>
          <td>0.218881</td>
          <td>25.332109</td>
          <td>0.347218</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.304423</td>
          <td>0.745611</td>
          <td>26.882206</td>
          <td>0.220630</td>
          <td>26.911764</td>
          <td>0.204134</td>
          <td>26.264270</td>
          <td>0.189903</td>
          <td>26.164212</td>
          <td>0.317397</td>
          <td>25.233877</td>
          <td>0.316226</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.579471</td>
          <td>0.895023</td>
          <td>28.324914</td>
          <td>0.671867</td>
          <td>27.361260</td>
          <td>0.297819</td>
          <td>26.913216</td>
          <td>0.326217</td>
          <td>25.802250</td>
          <td>0.238426</td>
          <td>26.275758</td>
          <td>0.691409</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.768271</td>
          <td>0.205370</td>
          <td>26.626750</td>
          <td>0.164672</td>
          <td>25.896609</td>
          <td>0.142653</td>
          <td>25.920554</td>
          <td>0.267344</td>
          <td>25.985073</td>
          <td>0.573011</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>25.995754</td>
          <td>0.282933</td>
          <td>26.504665</td>
          <td>0.161303</td>
          <td>26.117967</td>
          <td>0.103860</td>
          <td>25.861023</td>
          <td>0.135374</td>
          <td>25.200319</td>
          <td>0.143032</td>
          <td>25.272697</td>
          <td>0.327940</td>
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
          <td>27.334057</td>
          <td>0.693659</td>
          <td>26.724079</td>
          <td>0.167817</td>
          <td>26.112582</td>
          <td>0.087086</td>
          <td>25.474586</td>
          <td>0.080840</td>
          <td>25.108134</td>
          <td>0.111513</td>
          <td>24.856648</td>
          <td>0.197759</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.247793</td>
          <td>0.654106</td>
          <td>29.207138</td>
          <td>1.055451</td>
          <td>27.306388</td>
          <td>0.242508</td>
          <td>31.960537</td>
          <td>3.738105</td>
          <td>26.402006</td>
          <td>0.330424</td>
          <td>25.920501</td>
          <td>0.463329</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.331193</td>
          <td>0.348087</td>
          <td>25.838149</td>
          <td>0.083534</td>
          <td>24.775462</td>
          <td>0.028980</td>
          <td>23.877121</td>
          <td>0.021572</td>
          <td>23.170528</td>
          <td>0.021994</td>
          <td>22.845871</td>
          <td>0.037074</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.544684</td>
          <td>1.569535</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.377338</td>
          <td>0.315981</td>
          <td>26.415511</td>
          <td>0.228484</td>
          <td>25.622237</td>
          <td>0.215416</td>
          <td>25.512119</td>
          <td>0.415006</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.289092</td>
          <td>0.319668</td>
          <td>25.647263</td>
          <td>0.065731</td>
          <td>25.394634</td>
          <td>0.046168</td>
          <td>24.806829</td>
          <td>0.044787</td>
          <td>24.444428</td>
          <td>0.062233</td>
          <td>23.986190</td>
          <td>0.093511</td>
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
          <td>26.666350</td>
          <td>0.449549</td>
          <td>26.422804</td>
          <td>0.138655</td>
          <td>26.213555</td>
          <td>0.102937</td>
          <td>26.156839</td>
          <td>0.158838</td>
          <td>25.678730</td>
          <td>0.196458</td>
          <td>26.106301</td>
          <td>0.567887</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.536936</td>
          <td>0.391912</td>
          <td>27.142750</td>
          <td>0.241651</td>
          <td>26.720163</td>
          <td>0.150205</td>
          <td>26.327068</td>
          <td>0.172411</td>
          <td>26.735258</td>
          <td>0.433939</td>
          <td>25.766865</td>
          <td>0.418238</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.549350</td>
          <td>0.821100</td>
          <td>27.060062</td>
          <td>0.231800</td>
          <td>27.280431</td>
          <td>0.248290</td>
          <td>26.409723</td>
          <td>0.191058</td>
          <td>25.896911</td>
          <td>0.229157</td>
          <td>26.392400</td>
          <td>0.677186</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.509407</td>
          <td>0.828726</td>
          <td>27.335848</td>
          <td>0.306324</td>
          <td>26.592698</td>
          <td>0.148194</td>
          <td>25.873524</td>
          <td>0.129118</td>
          <td>25.567114</td>
          <td>0.185080</td>
          <td>25.701180</td>
          <td>0.434408</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.971086</td>
          <td>0.549542</td>
          <td>26.614662</td>
          <td>0.157965</td>
          <td>26.069711</td>
          <td>0.087187</td>
          <td>25.745759</td>
          <td>0.106841</td>
          <td>25.192197</td>
          <td>0.124666</td>
          <td>25.155216</td>
          <td>0.263130</td>
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
