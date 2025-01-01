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

    <pzflow.flow.Flow at 0x7f2bba795f90>



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
          <td>26.823717</td>
          <td>0.182602</td>
          <td>25.996018</td>
          <td>0.078570</td>
          <td>25.128902</td>
          <td>0.059522</td>
          <td>24.936008</td>
          <td>0.095911</td>
          <td>24.809981</td>
          <td>0.190113</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.641764</td>
          <td>0.849657</td>
          <td>27.852647</td>
          <td>0.419674</td>
          <td>28.013008</td>
          <td>0.424983</td>
          <td>27.573127</td>
          <td>0.463118</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.262190</td>
          <td>0.594035</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.959796</td>
          <td>0.532951</td>
          <td>25.867043</td>
          <td>0.079710</td>
          <td>24.802719</td>
          <td>0.027343</td>
          <td>23.884707</td>
          <td>0.019970</td>
          <td>23.133253</td>
          <td>0.019670</td>
          <td>22.856126</td>
          <td>0.034341</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.552075</td>
          <td>0.392540</td>
          <td>27.772858</td>
          <td>0.394746</td>
          <td>27.027191</td>
          <td>0.191959</td>
          <td>26.906185</td>
          <td>0.274721</td>
          <td>25.992275</td>
          <td>0.236829</td>
          <td>25.227739</td>
          <td>0.268900</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.981056</td>
          <td>0.541238</td>
          <td>25.838339</td>
          <td>0.077719</td>
          <td>25.468727</td>
          <td>0.049236</td>
          <td>24.909099</td>
          <td>0.048969</td>
          <td>24.295292</td>
          <td>0.054441</td>
          <td>23.653702</td>
          <td>0.069640</td>
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
          <td>26.621885</td>
          <td>0.414164</td>
          <td>26.404688</td>
          <td>0.127547</td>
          <td>26.189931</td>
          <td>0.093203</td>
          <td>26.013947</td>
          <td>0.129573</td>
          <td>25.687695</td>
          <td>0.183557</td>
          <td>25.572980</td>
          <td>0.354515</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.838139</td>
          <td>0.487413</td>
          <td>27.020318</td>
          <td>0.215390</td>
          <td>26.875919</td>
          <td>0.168869</td>
          <td>26.308239</td>
          <td>0.166853</td>
          <td>26.057896</td>
          <td>0.249990</td>
          <td>25.297178</td>
          <td>0.284503</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.545006</td>
          <td>0.390403</td>
          <td>27.192028</td>
          <td>0.248301</td>
          <td>26.641376</td>
          <td>0.138113</td>
          <td>26.438886</td>
          <td>0.186416</td>
          <td>26.434843</td>
          <td>0.338842</td>
          <td>25.299814</td>
          <td>0.285111</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.976875</td>
          <td>1.044118</td>
          <td>26.876551</td>
          <td>0.190932</td>
          <td>26.677707</td>
          <td>0.142506</td>
          <td>26.032675</td>
          <td>0.131690</td>
          <td>25.419202</td>
          <td>0.145981</td>
          <td>25.475550</td>
          <td>0.328258</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.654294</td>
          <td>0.856472</td>
          <td>26.808402</td>
          <td>0.180251</td>
          <td>25.976978</td>
          <td>0.077260</td>
          <td>25.656837</td>
          <td>0.094894</td>
          <td>25.231515</td>
          <td>0.124133</td>
          <td>25.145422</td>
          <td>0.251385</td>
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
          <td>28.537796</td>
          <td>1.522019</td>
          <td>26.981696</td>
          <td>0.238740</td>
          <td>26.059624</td>
          <td>0.097705</td>
          <td>25.200233</td>
          <td>0.075159</td>
          <td>24.892784</td>
          <td>0.108475</td>
          <td>24.888907</td>
          <td>0.237973</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.072121</td>
          <td>0.634974</td>
          <td>28.344956</td>
          <td>0.675161</td>
          <td>28.444149</td>
          <td>0.664412</td>
          <td>27.766070</td>
          <td>0.612581</td>
          <td>26.966120</td>
          <td>0.580850</td>
          <td>26.818622</td>
          <td>0.972112</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.860117</td>
          <td>0.553882</td>
          <td>25.832361</td>
          <td>0.091064</td>
          <td>24.803977</td>
          <td>0.032925</td>
          <td>23.851415</td>
          <td>0.023425</td>
          <td>23.132660</td>
          <td>0.023543</td>
          <td>22.922591</td>
          <td>0.044135</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.367041</td>
          <td>0.344662</td>
          <td>27.325418</td>
          <td>0.304106</td>
          <td>26.276956</td>
          <td>0.204266</td>
          <td>26.431121</td>
          <td>0.413596</td>
          <td>25.947029</td>
          <td>0.574489</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.068301</td>
          <td>0.297941</td>
          <td>25.712675</td>
          <td>0.080353</td>
          <td>25.435477</td>
          <td>0.056324</td>
          <td>24.741784</td>
          <td>0.050084</td>
          <td>24.350577</td>
          <td>0.067339</td>
          <td>23.771817</td>
          <td>0.091445</td>
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
          <td>26.502931</td>
          <td>0.424669</td>
          <td>26.323248</td>
          <td>0.139413</td>
          <td>26.156168</td>
          <td>0.108563</td>
          <td>25.947039</td>
          <td>0.147413</td>
          <td>25.949919</td>
          <td>0.271188</td>
          <td>25.263128</td>
          <td>0.328783</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.723873</td>
          <td>0.973976</td>
          <td>27.259855</td>
          <td>0.300511</td>
          <td>26.849399</td>
          <td>0.193713</td>
          <td>26.412327</td>
          <td>0.215021</td>
          <td>26.665667</td>
          <td>0.467887</td>
          <td>26.474135</td>
          <td>0.784162</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.764257</td>
          <td>0.513491</td>
          <td>28.229592</td>
          <td>0.628941</td>
          <td>27.156528</td>
          <td>0.252153</td>
          <td>26.548262</td>
          <td>0.242717</td>
          <td>25.677810</td>
          <td>0.215026</td>
          <td>26.716724</td>
          <td>0.921388</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.360055</td>
          <td>0.383216</td>
          <td>28.745782</td>
          <td>0.897058</td>
          <td>26.389700</td>
          <td>0.134353</td>
          <td>25.774654</td>
          <td>0.128396</td>
          <td>26.052270</td>
          <td>0.297453</td>
          <td>25.546042</td>
          <td>0.413874</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.150383</td>
          <td>0.674219</td>
          <td>26.224407</td>
          <td>0.126767</td>
          <td>26.113577</td>
          <td>0.103462</td>
          <td>25.614784</td>
          <td>0.109316</td>
          <td>25.494204</td>
          <td>0.183806</td>
          <td>24.643940</td>
          <td>0.195915</td>
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
          <td>27.394864</td>
          <td>0.722771</td>
          <td>26.603978</td>
          <td>0.151455</td>
          <td>26.074863</td>
          <td>0.084240</td>
          <td>25.263041</td>
          <td>0.067049</td>
          <td>24.982375</td>
          <td>0.099904</td>
          <td>24.869386</td>
          <td>0.199887</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.592079</td>
          <td>1.463494</td>
          <td>27.695262</td>
          <td>0.371965</td>
          <td>28.085818</td>
          <td>0.449459</td>
          <td>27.975355</td>
          <td>0.620669</td>
          <td>26.362790</td>
          <td>0.320278</td>
          <td>26.293160</td>
          <td>0.607669</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.068419</td>
          <td>0.604233</td>
          <td>25.996302</td>
          <td>0.095979</td>
          <td>24.759523</td>
          <td>0.028578</td>
          <td>23.892022</td>
          <td>0.021849</td>
          <td>23.129412</td>
          <td>0.021235</td>
          <td>22.861606</td>
          <td>0.037593</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.722727</td>
          <td>1.004441</td>
          <td>28.801440</td>
          <td>0.947933</td>
          <td>27.790144</td>
          <td>0.435829</td>
          <td>26.641790</td>
          <td>0.275147</td>
          <td>25.907141</td>
          <td>0.272432</td>
          <td>25.111881</td>
          <td>0.303169</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.855652</td>
          <td>0.494181</td>
          <td>25.674953</td>
          <td>0.067361</td>
          <td>25.489709</td>
          <td>0.050234</td>
          <td>24.902621</td>
          <td>0.048762</td>
          <td>24.337473</td>
          <td>0.056599</td>
          <td>23.829827</td>
          <td>0.081488</td>
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
          <td>27.256227</td>
          <td>0.686886</td>
          <td>26.201283</td>
          <td>0.114452</td>
          <td>26.302949</td>
          <td>0.111298</td>
          <td>25.684476</td>
          <td>0.105547</td>
          <td>26.245597</td>
          <td>0.312992</td>
          <td>25.920391</td>
          <td>0.495993</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.888394</td>
          <td>0.510715</td>
          <td>27.223327</td>
          <td>0.258188</td>
          <td>27.200900</td>
          <td>0.225507</td>
          <td>26.594961</td>
          <td>0.216057</td>
          <td>27.316886</td>
          <td>0.661663</td>
          <td>25.349828</td>
          <td>0.301541</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.158306</td>
          <td>0.251357</td>
          <td>27.369286</td>
          <td>0.267035</td>
          <td>26.098644</td>
          <td>0.146589</td>
          <td>26.432696</td>
          <td>0.353415</td>
          <td>25.733890</td>
          <td>0.419925</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.847556</td>
          <td>0.525727</td>
          <td>26.899226</td>
          <td>0.214268</td>
          <td>26.542776</td>
          <td>0.141966</td>
          <td>25.754823</td>
          <td>0.116476</td>
          <td>25.600561</td>
          <td>0.190381</td>
          <td>25.867083</td>
          <td>0.491932</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.068515</td>
          <td>1.121470</td>
          <td>26.394560</td>
          <td>0.130732</td>
          <td>26.135098</td>
          <td>0.092347</td>
          <td>25.556817</td>
          <td>0.090532</td>
          <td>25.085093</td>
          <td>0.113581</td>
          <td>25.011825</td>
          <td>0.233860</td>
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
