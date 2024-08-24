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

    <pzflow.flow.Flow at 0x7f8a44583940>



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
          <td>27.051954</td>
          <td>0.221141</td>
          <td>26.042307</td>
          <td>0.081846</td>
          <td>25.260234</td>
          <td>0.066873</td>
          <td>24.996851</td>
          <td>0.101166</td>
          <td>24.705645</td>
          <td>0.174041</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.970523</td>
          <td>0.914004</td>
          <td>28.361262</td>
          <td>0.550384</td>
          <td>26.540417</td>
          <td>0.203052</td>
          <td>25.718559</td>
          <td>0.188408</td>
          <td>25.864237</td>
          <td>0.443747</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.455103</td>
          <td>0.364069</td>
          <td>25.885373</td>
          <td>0.081008</td>
          <td>24.811107</td>
          <td>0.027544</td>
          <td>23.870737</td>
          <td>0.019736</td>
          <td>23.143034</td>
          <td>0.019834</td>
          <td>22.858988</td>
          <td>0.034428</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>28.105299</td>
          <td>1.125196</td>
          <td>27.877296</td>
          <td>0.427631</td>
          <td>27.118985</td>
          <td>0.207349</td>
          <td>26.928543</td>
          <td>0.279755</td>
          <td>25.949149</td>
          <td>0.228521</td>
          <td>25.020198</td>
          <td>0.226688</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.561836</td>
          <td>0.395506</td>
          <td>25.870686</td>
          <td>0.079967</td>
          <td>25.464889</td>
          <td>0.049068</td>
          <td>24.830757</td>
          <td>0.045679</td>
          <td>24.340374</td>
          <td>0.056664</td>
          <td>23.731871</td>
          <td>0.074625</td>
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
          <td>27.439109</td>
          <td>0.744436</td>
          <td>26.252509</td>
          <td>0.111756</td>
          <td>26.149696</td>
          <td>0.089965</td>
          <td>25.790996</td>
          <td>0.106727</td>
          <td>26.219912</td>
          <td>0.285307</td>
          <td>25.761332</td>
          <td>0.410315</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.574811</td>
          <td>0.813851</td>
          <td>27.108928</td>
          <td>0.231848</td>
          <td>27.136949</td>
          <td>0.210489</td>
          <td>26.891095</td>
          <td>0.271369</td>
          <td>26.052916</td>
          <td>0.248969</td>
          <td>25.570242</td>
          <td>0.353754</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>28.118502</td>
          <td>1.133730</td>
          <td>27.450027</td>
          <td>0.306179</td>
          <td>27.331199</td>
          <td>0.247295</td>
          <td>26.236297</td>
          <td>0.156910</td>
          <td>25.624078</td>
          <td>0.173920</td>
          <td>26.314483</td>
          <td>0.616372</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.467794</td>
          <td>0.758756</td>
          <td>27.398039</td>
          <td>0.293647</td>
          <td>26.525453</td>
          <td>0.124934</td>
          <td>25.752311</td>
          <td>0.103177</td>
          <td>25.558842</td>
          <td>0.164526</td>
          <td>25.821369</td>
          <td>0.429559</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.950888</td>
          <td>0.529507</td>
          <td>26.387602</td>
          <td>0.125673</td>
          <td>26.148489</td>
          <td>0.089869</td>
          <td>25.618919</td>
          <td>0.091785</td>
          <td>25.105862</td>
          <td>0.111278</td>
          <td>25.163648</td>
          <td>0.255174</td>
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
          <td>28.266518</td>
          <td>1.324118</td>
          <td>26.787468</td>
          <td>0.203121</td>
          <td>25.860130</td>
          <td>0.081987</td>
          <td>25.204098</td>
          <td>0.075416</td>
          <td>25.007271</td>
          <td>0.119849</td>
          <td>25.024764</td>
          <td>0.266054</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>29.304642</td>
          <td>1.223727</td>
          <td>27.986346</td>
          <td>0.478440</td>
          <td>27.340392</td>
          <td>0.449073</td>
          <td>26.463241</td>
          <td>0.399873</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.584636</td>
          <td>0.902609</td>
          <td>25.958127</td>
          <td>0.101667</td>
          <td>24.790479</td>
          <td>0.032536</td>
          <td>23.918090</td>
          <td>0.024813</td>
          <td>23.152226</td>
          <td>0.023943</td>
          <td>22.918778</td>
          <td>0.043986</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.908198</td>
          <td>0.520257</td>
          <td>27.624917</td>
          <td>0.385175</td>
          <td>26.589980</td>
          <td>0.264678</td>
          <td>25.736469</td>
          <td>0.237644</td>
          <td>25.518813</td>
          <td>0.418464</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.463780</td>
          <td>0.406538</td>
          <td>25.735658</td>
          <td>0.081995</td>
          <td>25.405660</td>
          <td>0.054853</td>
          <td>24.816051</td>
          <td>0.053496</td>
          <td>24.328377</td>
          <td>0.066028</td>
          <td>23.717152</td>
          <td>0.087154</td>
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
          <td>26.661332</td>
          <td>0.478430</td>
          <td>26.327532</td>
          <td>0.139928</td>
          <td>26.364201</td>
          <td>0.130082</td>
          <td>26.063176</td>
          <td>0.162826</td>
          <td>26.021731</td>
          <td>0.287457</td>
          <td>25.228322</td>
          <td>0.319805</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.507564</td>
          <td>0.421447</td>
          <td>26.509608</td>
          <td>0.161153</td>
          <td>26.910810</td>
          <td>0.203971</td>
          <td>25.957795</td>
          <td>0.146270</td>
          <td>26.084456</td>
          <td>0.297745</td>
          <td>25.352850</td>
          <td>0.347514</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.257249</td>
          <td>0.726084</td>
          <td>27.461572</td>
          <td>0.355218</td>
          <td>26.968845</td>
          <td>0.215876</td>
          <td>26.546328</td>
          <td>0.242330</td>
          <td>26.301294</td>
          <td>0.356514</td>
          <td>25.483017</td>
          <td>0.387729</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.913128</td>
          <td>0.578621</td>
          <td>27.095656</td>
          <td>0.269165</td>
          <td>26.501842</td>
          <td>0.147977</td>
          <td>25.932104</td>
          <td>0.147074</td>
          <td>25.860046</td>
          <td>0.254437</td>
          <td>25.044601</td>
          <td>0.278581</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.652721</td>
          <td>0.182920</td>
          <td>26.057768</td>
          <td>0.098529</td>
          <td>25.640740</td>
          <td>0.111819</td>
          <td>25.037727</td>
          <td>0.124286</td>
          <td>24.926055</td>
          <td>0.247757</td>
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
          <td>27.050992</td>
          <td>0.569245</td>
          <td>26.699742</td>
          <td>0.164374</td>
          <td>25.950004</td>
          <td>0.075450</td>
          <td>25.222840</td>
          <td>0.064702</td>
          <td>24.972775</td>
          <td>0.099067</td>
          <td>25.112187</td>
          <td>0.244638</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.386162</td>
          <td>0.258935</td>
          <td>27.018207</td>
          <td>0.301036</td>
          <td>26.323760</td>
          <td>0.310448</td>
          <td>26.282356</td>
          <td>0.603052</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.003443</td>
          <td>0.267744</td>
          <td>25.853183</td>
          <td>0.084646</td>
          <td>24.752957</td>
          <td>0.028414</td>
          <td>23.855086</td>
          <td>0.021170</td>
          <td>23.133343</td>
          <td>0.021306</td>
          <td>22.876201</td>
          <td>0.038081</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.154888</td>
          <td>0.698356</td>
          <td>28.179112</td>
          <td>0.629802</td>
          <td>31.860053</td>
          <td>3.354123</td>
          <td>26.956244</td>
          <td>0.353790</td>
          <td>25.849043</td>
          <td>0.259819</td>
          <td>25.583210</td>
          <td>0.438083</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.308649</td>
          <td>0.324678</td>
          <td>25.777810</td>
          <td>0.073769</td>
          <td>25.324336</td>
          <td>0.043375</td>
          <td>24.848999</td>
          <td>0.046495</td>
          <td>24.387546</td>
          <td>0.059171</td>
          <td>23.687734</td>
          <td>0.071876</td>
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
          <td>26.248432</td>
          <td>0.325263</td>
          <td>26.194854</td>
          <td>0.113814</td>
          <td>26.190679</td>
          <td>0.100896</td>
          <td>26.184090</td>
          <td>0.162580</td>
          <td>25.711348</td>
          <td>0.201916</td>
          <td>26.055447</td>
          <td>0.547469</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.596998</td>
          <td>0.348582</td>
          <td>26.719980</td>
          <td>0.150181</td>
          <td>26.519142</td>
          <td>0.202779</td>
          <td>26.440749</td>
          <td>0.345493</td>
          <td>25.114864</td>
          <td>0.249105</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.505820</td>
          <td>0.798235</td>
          <td>27.472757</td>
          <td>0.324124</td>
          <td>26.901998</td>
          <td>0.181007</td>
          <td>26.101591</td>
          <td>0.146961</td>
          <td>25.740465</td>
          <td>0.201111</td>
          <td>25.151093</td>
          <td>0.264781</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.242271</td>
          <td>0.694404</td>
          <td>27.575187</td>
          <td>0.370155</td>
          <td>26.560060</td>
          <td>0.144094</td>
          <td>25.645477</td>
          <td>0.105881</td>
          <td>25.551192</td>
          <td>0.182605</td>
          <td>25.238448</td>
          <td>0.302549</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.582570</td>
          <td>1.478884</td>
          <td>26.236671</td>
          <td>0.113998</td>
          <td>26.156829</td>
          <td>0.094127</td>
          <td>25.542225</td>
          <td>0.089378</td>
          <td>25.364007</td>
          <td>0.144613</td>
          <td>25.215693</td>
          <td>0.276420</td>
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
