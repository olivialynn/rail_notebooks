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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f94fcca7eb0>



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
          <td>27.309995</td>
          <td>0.682326</td>
          <td>26.784857</td>
          <td>0.176691</td>
          <td>26.046704</td>
          <td>0.082164</td>
          <td>25.377043</td>
          <td>0.074157</td>
          <td>24.962982</td>
          <td>0.098208</td>
          <td>24.863365</td>
          <td>0.198853</td>
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
          <td>27.503295</td>
          <td>0.284593</td>
          <td>27.007950</td>
          <td>0.298292</td>
          <td>26.630800</td>
          <td>0.394902</td>
          <td>25.893700</td>
          <td>0.453716</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>25.929164</td>
          <td>0.238539</td>
          <td>25.945809</td>
          <td>0.085434</td>
          <td>24.771707</td>
          <td>0.026613</td>
          <td>23.878317</td>
          <td>0.019863</td>
          <td>23.121429</td>
          <td>0.019475</td>
          <td>22.798714</td>
          <td>0.032646</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.983254</td>
          <td>1.048062</td>
          <td>27.819406</td>
          <td>0.409136</td>
          <td>27.225039</td>
          <td>0.226520</td>
          <td>27.011575</td>
          <td>0.299163</td>
          <td>25.994269</td>
          <td>0.237219</td>
          <td>25.589993</td>
          <td>0.359278</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.333986</td>
          <td>0.330977</td>
          <td>25.829635</td>
          <td>0.077125</td>
          <td>25.470684</td>
          <td>0.049321</td>
          <td>24.806985</td>
          <td>0.044725</td>
          <td>24.366413</td>
          <td>0.057989</td>
          <td>23.761620</td>
          <td>0.076613</td>
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
          <td>26.122844</td>
          <td>0.279456</td>
          <td>26.468816</td>
          <td>0.134817</td>
          <td>26.213003</td>
          <td>0.095111</td>
          <td>26.046392</td>
          <td>0.133261</td>
          <td>26.111975</td>
          <td>0.261322</td>
          <td>25.292488</td>
          <td>0.283425</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.174340</td>
          <td>0.621182</td>
          <td>26.619512</td>
          <td>0.153467</td>
          <td>26.758076</td>
          <td>0.152694</td>
          <td>26.315769</td>
          <td>0.167927</td>
          <td>25.992270</td>
          <td>0.236828</td>
          <td>26.060439</td>
          <td>0.513566</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.111265</td>
          <td>0.509411</td>
          <td>26.950704</td>
          <td>0.179943</td>
          <td>26.585875</td>
          <td>0.210931</td>
          <td>26.091938</td>
          <td>0.257071</td>
          <td>24.764664</td>
          <td>0.182972</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.254235</td>
          <td>0.656685</td>
          <td>27.295603</td>
          <td>0.270261</td>
          <td>26.410395</td>
          <td>0.113039</td>
          <td>26.015518</td>
          <td>0.129749</td>
          <td>25.661573</td>
          <td>0.179542</td>
          <td>25.480127</td>
          <td>0.329454</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.935871</td>
          <td>1.018982</td>
          <td>26.512381</td>
          <td>0.139977</td>
          <td>26.076008</td>
          <td>0.084314</td>
          <td>25.740748</td>
          <td>0.102138</td>
          <td>25.040007</td>
          <td>0.105059</td>
          <td>24.900655</td>
          <td>0.205174</td>
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
          <td>28.107864</td>
          <td>1.214819</td>
          <td>26.913024</td>
          <td>0.225548</td>
          <td>26.210535</td>
          <td>0.111486</td>
          <td>25.340822</td>
          <td>0.085082</td>
          <td>25.297098</td>
          <td>0.153909</td>
          <td>24.867789</td>
          <td>0.233854</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.064511</td>
          <td>0.554283</td>
          <td>28.215682</td>
          <td>0.565793</td>
          <td>28.604010</td>
          <td>1.053944</td>
          <td>26.977481</td>
          <td>0.585570</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.378326</td>
          <td>0.386291</td>
          <td>25.833993</td>
          <td>0.091195</td>
          <td>24.774581</td>
          <td>0.032084</td>
          <td>23.881811</td>
          <td>0.024047</td>
          <td>23.096361</td>
          <td>0.022820</td>
          <td>22.828803</td>
          <td>0.040617</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.624448</td>
          <td>0.479798</td>
          <td>27.648438</td>
          <td>0.428604</td>
          <td>27.966608</td>
          <td>0.498862</td>
          <td>26.810358</td>
          <td>0.316235</td>
          <td>26.213165</td>
          <td>0.349209</td>
          <td>24.758974</td>
          <td>0.228032</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.088027</td>
          <td>0.302696</td>
          <td>25.781552</td>
          <td>0.085373</td>
          <td>25.550468</td>
          <td>0.062370</td>
          <td>24.821327</td>
          <td>0.053747</td>
          <td>24.484457</td>
          <td>0.075801</td>
          <td>23.783054</td>
          <td>0.092352</td>
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
          <td>26.069045</td>
          <td>0.302451</td>
          <td>26.438809</td>
          <td>0.153954</td>
          <td>25.893677</td>
          <td>0.086245</td>
          <td>26.228540</td>
          <td>0.187369</td>
          <td>25.712512</td>
          <td>0.223059</td>
          <td>25.185431</td>
          <td>0.309031</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.612462</td>
          <td>1.581527</td>
          <td>27.373745</td>
          <td>0.329125</td>
          <td>26.925431</td>
          <td>0.206485</td>
          <td>26.510106</td>
          <td>0.233224</td>
          <td>26.364942</td>
          <td>0.371849</td>
          <td>25.743716</td>
          <td>0.469202</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.878683</td>
          <td>0.557982</td>
          <td>26.893394</td>
          <td>0.224336</td>
          <td>26.921493</td>
          <td>0.207502</td>
          <td>26.962481</td>
          <td>0.339208</td>
          <td>25.880570</td>
          <td>0.254300</td>
          <td>25.929726</td>
          <td>0.542067</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.449176</td>
          <td>0.357115</td>
          <td>26.618888</td>
          <td>0.163571</td>
          <td>25.908422</td>
          <td>0.144110</td>
          <td>25.917087</td>
          <td>0.266589</td>
          <td>26.396363</td>
          <td>0.760771</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.151479</td>
          <td>0.674726</td>
          <td>26.830520</td>
          <td>0.212393</td>
          <td>26.234713</td>
          <td>0.114999</td>
          <td>25.649152</td>
          <td>0.112642</td>
          <td>25.464431</td>
          <td>0.179232</td>
          <td>24.757342</td>
          <td>0.215440</td>
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
          <td>26.750110</td>
          <td>0.456466</td>
          <td>26.421819</td>
          <td>0.129466</td>
          <td>26.077958</td>
          <td>0.084470</td>
          <td>25.427429</td>
          <td>0.077544</td>
          <td>25.171536</td>
          <td>0.117845</td>
          <td>24.932044</td>
          <td>0.210663</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.552606</td>
          <td>1.434421</td>
          <td>27.711132</td>
          <td>0.376589</td>
          <td>27.156034</td>
          <td>0.214064</td>
          <td>27.909517</td>
          <td>0.592499</td>
          <td>26.937854</td>
          <td>0.498390</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.114073</td>
          <td>0.292835</td>
          <td>25.890642</td>
          <td>0.087480</td>
          <td>24.763471</td>
          <td>0.028677</td>
          <td>23.895920</td>
          <td>0.021922</td>
          <td>23.113369</td>
          <td>0.020946</td>
          <td>22.870599</td>
          <td>0.037893</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.717943</td>
          <td>0.512879</td>
          <td>28.008888</td>
          <td>0.558183</td>
          <td>28.044589</td>
          <td>0.526668</td>
          <td>26.463813</td>
          <td>0.237805</td>
          <td>26.193889</td>
          <td>0.342846</td>
          <td>24.997357</td>
          <td>0.276387</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.512490</td>
          <td>0.381031</td>
          <td>25.693667</td>
          <td>0.068485</td>
          <td>25.538027</td>
          <td>0.052436</td>
          <td>24.810870</td>
          <td>0.044948</td>
          <td>24.404897</td>
          <td>0.060089</td>
          <td>23.734730</td>
          <td>0.074925</td>
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
          <td>26.148126</td>
          <td>0.300227</td>
          <td>26.218307</td>
          <td>0.116160</td>
          <td>26.172367</td>
          <td>0.099290</td>
          <td>26.091805</td>
          <td>0.150232</td>
          <td>26.112522</td>
          <td>0.281191</td>
          <td>25.101028</td>
          <td>0.261558</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.681387</td>
          <td>0.437672</td>
          <td>27.018199</td>
          <td>0.217951</td>
          <td>26.818558</td>
          <td>0.163401</td>
          <td>26.467331</td>
          <td>0.194137</td>
          <td>25.804361</td>
          <td>0.205721</td>
          <td>25.372858</td>
          <td>0.307167</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.926728</td>
          <td>1.037350</td>
          <td>27.385417</td>
          <td>0.302271</td>
          <td>27.219914</td>
          <td>0.236204</td>
          <td>27.266547</td>
          <td>0.383208</td>
          <td>26.256166</td>
          <td>0.307210</td>
          <td>27.501734</td>
          <td>1.330264</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.960423</td>
          <td>1.092295</td>
          <td>27.461473</td>
          <td>0.338540</td>
          <td>26.514129</td>
          <td>0.138504</td>
          <td>25.730568</td>
          <td>0.114042</td>
          <td>25.881755</td>
          <td>0.240732</td>
          <td>25.895196</td>
          <td>0.502255</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.657068</td>
          <td>0.435572</td>
          <td>26.526025</td>
          <td>0.146413</td>
          <td>26.119717</td>
          <td>0.091107</td>
          <td>25.536646</td>
          <td>0.088941</td>
          <td>25.235195</td>
          <td>0.129398</td>
          <td>24.757400</td>
          <td>0.189054</td>
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
