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

    <pzflow.flow.Flow at 0x7f7a603ab100>



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
          <td>28.341868</td>
          <td>1.283556</td>
          <td>26.473057</td>
          <td>0.135311</td>
          <td>26.084070</td>
          <td>0.084915</td>
          <td>25.305461</td>
          <td>0.069606</td>
          <td>24.864442</td>
          <td>0.090068</td>
          <td>24.747265</td>
          <td>0.180296</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>28.979106</td>
          <td>1.761815</td>
          <td>27.318633</td>
          <td>0.275371</td>
          <td>27.697367</td>
          <td>0.332478</td>
          <td>26.798833</td>
          <td>0.251648</td>
          <td>26.846832</td>
          <td>0.465404</td>
          <td>25.616547</td>
          <td>0.366821</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.550109</td>
          <td>0.391945</td>
          <td>25.907244</td>
          <td>0.082584</td>
          <td>24.787749</td>
          <td>0.026988</td>
          <td>23.863093</td>
          <td>0.019608</td>
          <td>23.163102</td>
          <td>0.020174</td>
          <td>22.812862</td>
          <td>0.033056</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.870322</td>
          <td>0.425368</td>
          <td>27.183704</td>
          <td>0.218865</td>
          <td>26.372809</td>
          <td>0.176271</td>
          <td>25.967502</td>
          <td>0.232024</td>
          <td>25.023216</td>
          <td>0.227257</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.383607</td>
          <td>0.344211</td>
          <td>25.711933</td>
          <td>0.069513</td>
          <td>25.458030</td>
          <td>0.048770</td>
          <td>24.815501</td>
          <td>0.045065</td>
          <td>24.271747</td>
          <td>0.053315</td>
          <td>23.598045</td>
          <td>0.066291</td>
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
          <td>25.971454</td>
          <td>0.246987</td>
          <td>26.445843</td>
          <td>0.132168</td>
          <td>26.302606</td>
          <td>0.102882</td>
          <td>26.033790</td>
          <td>0.131817</td>
          <td>25.811055</td>
          <td>0.203657</td>
          <td>25.048095</td>
          <td>0.231994</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>26.876936</td>
          <td>0.501589</td>
          <td>27.118111</td>
          <td>0.233617</td>
          <td>26.759626</td>
          <td>0.152897</td>
          <td>26.348475</td>
          <td>0.172666</td>
          <td>26.120079</td>
          <td>0.263059</td>
          <td>25.032163</td>
          <td>0.228950</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.758080</td>
          <td>0.459169</td>
          <td>27.597963</td>
          <td>0.344399</td>
          <td>26.930865</td>
          <td>0.176942</td>
          <td>26.585896</td>
          <td>0.210935</td>
          <td>26.154982</td>
          <td>0.270657</td>
          <td>25.215948</td>
          <td>0.266327</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>27.635246</td>
          <td>0.846126</td>
          <td>27.532646</td>
          <td>0.327047</td>
          <td>26.837281</td>
          <td>0.163398</td>
          <td>25.887609</td>
          <td>0.116111</td>
          <td>25.696220</td>
          <td>0.184886</td>
          <td>25.992661</td>
          <td>0.488525</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.454762</td>
          <td>0.133191</td>
          <td>26.125895</td>
          <td>0.088101</td>
          <td>25.538103</td>
          <td>0.085485</td>
          <td>25.357992</td>
          <td>0.138486</td>
          <td>25.273370</td>
          <td>0.279066</td>
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
          <td>27.101474</td>
          <td>0.647978</td>
          <td>26.832581</td>
          <td>0.210935</td>
          <td>26.141166</td>
          <td>0.104934</td>
          <td>25.380787</td>
          <td>0.088129</td>
          <td>25.088930</td>
          <td>0.128645</td>
          <td>24.546961</td>
          <td>0.178730</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.171989</td>
          <td>0.680291</td>
          <td>27.893694</td>
          <td>0.489215</td>
          <td>26.873555</td>
          <td>0.196942</td>
          <td>27.664664</td>
          <td>0.570026</td>
          <td>25.776383</td>
          <td>0.230633</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.531106</td>
          <td>0.434240</td>
          <td>26.134467</td>
          <td>0.118552</td>
          <td>24.811051</td>
          <td>0.033130</td>
          <td>23.884705</td>
          <td>0.024107</td>
          <td>23.157447</td>
          <td>0.024051</td>
          <td>22.829927</td>
          <td>0.040657</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.348598</td>
          <td>0.339683</td>
          <td>27.729023</td>
          <td>0.417304</td>
          <td>26.556809</td>
          <td>0.257595</td>
          <td>26.318663</td>
          <td>0.379238</td>
          <td>24.897858</td>
          <td>0.255703</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.330224</td>
          <td>0.366631</td>
          <td>25.856495</td>
          <td>0.091181</td>
          <td>25.414448</td>
          <td>0.055283</td>
          <td>24.755258</td>
          <td>0.050686</td>
          <td>24.456410</td>
          <td>0.073946</td>
          <td>23.670386</td>
          <td>0.083639</td>
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
          <td>26.475864</td>
          <td>0.415994</td>
          <td>26.379961</td>
          <td>0.146380</td>
          <td>26.236504</td>
          <td>0.116437</td>
          <td>26.344054</td>
          <td>0.206484</td>
          <td>25.983051</td>
          <td>0.278591</td>
          <td>26.493219</td>
          <td>0.804252</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.772860</td>
          <td>1.003239</td>
          <td>26.590573</td>
          <td>0.172652</td>
          <td>26.740384</td>
          <td>0.176661</td>
          <td>26.402950</td>
          <td>0.213345</td>
          <td>25.923465</td>
          <td>0.261288</td>
          <td>24.769617</td>
          <td>0.216415</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.928248</td>
          <td>0.578150</td>
          <td>27.548141</td>
          <td>0.380050</td>
          <td>26.899920</td>
          <td>0.203785</td>
          <td>26.698585</td>
          <td>0.274506</td>
          <td>26.457399</td>
          <td>0.402483</td>
          <td>24.692879</td>
          <td>0.204685</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.969912</td>
          <td>0.242819</td>
          <td>26.555515</td>
          <td>0.154947</td>
          <td>25.800718</td>
          <td>0.131325</td>
          <td>25.708732</td>
          <td>0.224559</td>
          <td>25.644483</td>
          <td>0.446032</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.638521</td>
          <td>0.467026</td>
          <td>26.641974</td>
          <td>0.181265</td>
          <td>26.142704</td>
          <td>0.106131</td>
          <td>25.521269</td>
          <td>0.100735</td>
          <td>25.267920</td>
          <td>0.151585</td>
          <td>24.583304</td>
          <td>0.186152</td>
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
          <td>26.955888</td>
          <td>0.531478</td>
          <td>26.522781</td>
          <td>0.141251</td>
          <td>26.272661</td>
          <td>0.100232</td>
          <td>25.221717</td>
          <td>0.064638</td>
          <td>24.883573</td>
          <td>0.091607</td>
          <td>25.134320</td>
          <td>0.249134</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>27.670786</td>
          <td>0.865906</td>
          <td>28.384093</td>
          <td>0.620109</td>
          <td>27.985114</td>
          <td>0.416373</td>
          <td>27.171657</td>
          <td>0.340202</td>
          <td>27.192967</td>
          <td>0.599364</td>
          <td>26.309884</td>
          <td>0.614866</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.103918</td>
          <td>0.290449</td>
          <td>25.991672</td>
          <td>0.095590</td>
          <td>24.845957</td>
          <td>0.030828</td>
          <td>23.847694</td>
          <td>0.021037</td>
          <td>23.138779</td>
          <td>0.021405</td>
          <td>22.843132</td>
          <td>0.036984</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.249073</td>
          <td>0.744018</td>
          <td>28.721478</td>
          <td>0.902101</td>
          <td>27.911993</td>
          <td>0.477624</td>
          <td>27.001333</td>
          <td>0.366508</td>
          <td>26.720162</td>
          <td>0.512151</td>
          <td>25.490797</td>
          <td>0.408282</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.708500</td>
          <td>0.442721</td>
          <td>25.880715</td>
          <td>0.080776</td>
          <td>25.360907</td>
          <td>0.044806</td>
          <td>24.757915</td>
          <td>0.042884</td>
          <td>24.315912</td>
          <td>0.055527</td>
          <td>23.849027</td>
          <td>0.082879</td>
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
          <td>26.833806</td>
          <td>0.509179</td>
          <td>26.399983</td>
          <td>0.135954</td>
          <td>26.129437</td>
          <td>0.095622</td>
          <td>26.427017</td>
          <td>0.199723</td>
          <td>25.630987</td>
          <td>0.188712</td>
          <td>24.856633</td>
          <td>0.213722</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.097469</td>
          <td>0.593865</td>
          <td>27.129316</td>
          <td>0.238987</td>
          <td>26.823734</td>
          <td>0.164124</td>
          <td>26.334538</td>
          <td>0.173509</td>
          <td>26.423539</td>
          <td>0.340833</td>
          <td>25.066125</td>
          <td>0.239300</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.656219</td>
          <td>0.879075</td>
          <td>27.026760</td>
          <td>0.225488</td>
          <td>27.189428</td>
          <td>0.230317</td>
          <td>26.579151</td>
          <td>0.220205</td>
          <td>26.308730</td>
          <td>0.320391</td>
          <td>24.944159</td>
          <td>0.223266</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.078926</td>
          <td>0.248648</td>
          <td>26.811319</td>
          <td>0.178591</td>
          <td>26.015427</td>
          <td>0.145935</td>
          <td>25.711907</td>
          <td>0.209047</td>
          <td>25.019676</td>
          <td>0.253306</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.801094</td>
          <td>0.185086</td>
          <td>26.077726</td>
          <td>0.087804</td>
          <td>25.507173</td>
          <td>0.086663</td>
          <td>25.176457</td>
          <td>0.122975</td>
          <td>25.378270</td>
          <td>0.315105</td>
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
