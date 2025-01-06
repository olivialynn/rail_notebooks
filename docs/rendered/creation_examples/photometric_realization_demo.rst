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

    <pzflow.flow.Flow at 0x7f17a8636650>



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
          <td>27.355878</td>
          <td>0.703959</td>
          <td>26.536322</td>
          <td>0.142891</td>
          <td>26.089050</td>
          <td>0.085288</td>
          <td>25.284748</td>
          <td>0.068341</td>
          <td>24.975847</td>
          <td>0.099321</td>
          <td>25.601556</td>
          <td>0.362546</td>
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
          <td>28.747022</td>
          <td>0.720464</td>
          <td>26.795503</td>
          <td>0.250960</td>
          <td>28.282998</td>
          <td>1.195824</td>
          <td>28.280713</td>
          <td>1.884538</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.222521</td>
          <td>0.302836</td>
          <td>25.942761</td>
          <td>0.085206</td>
          <td>24.808948</td>
          <td>0.027492</td>
          <td>23.843303</td>
          <td>0.019283</td>
          <td>23.122287</td>
          <td>0.019489</td>
          <td>22.852531</td>
          <td>0.034233</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.530457</td>
          <td>0.326479</td>
          <td>27.552171</td>
          <td>0.296050</td>
          <td>26.558863</td>
          <td>0.206217</td>
          <td>26.014916</td>
          <td>0.241298</td>
          <td>25.142314</td>
          <td>0.250744</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>25.751568</td>
          <td>0.205834</td>
          <td>25.832110</td>
          <td>0.077293</td>
          <td>25.482114</td>
          <td>0.049825</td>
          <td>24.744523</td>
          <td>0.042314</td>
          <td>24.370568</td>
          <td>0.058203</td>
          <td>23.756169</td>
          <td>0.076245</td>
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
          <td>26.930617</td>
          <td>0.521737</td>
          <td>26.334669</td>
          <td>0.120034</td>
          <td>26.169890</td>
          <td>0.091576</td>
          <td>25.945077</td>
          <td>0.122061</td>
          <td>26.443132</td>
          <td>0.341069</td>
          <td>25.772879</td>
          <td>0.413960</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.512520</td>
          <td>1.404644</td>
          <td>26.967103</td>
          <td>0.206023</td>
          <td>26.793812</td>
          <td>0.157440</td>
          <td>26.453184</td>
          <td>0.188681</td>
          <td>26.629858</td>
          <td>0.394615</td>
          <td>25.123296</td>
          <td>0.246854</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.684385</td>
          <td>0.434348</td>
          <td>27.708355</td>
          <td>0.375503</td>
          <td>26.706255</td>
          <td>0.146049</td>
          <td>26.013776</td>
          <td>0.129553</td>
          <td>26.151489</td>
          <td>0.269888</td>
          <td>25.620815</td>
          <td>0.368045</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.900224</td>
          <td>0.510253</td>
          <td>27.580621</td>
          <td>0.339717</td>
          <td>26.721560</td>
          <td>0.147983</td>
          <td>25.987536</td>
          <td>0.126641</td>
          <td>25.809036</td>
          <td>0.203312</td>
          <td>25.703187</td>
          <td>0.392359</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>27.054848</td>
          <td>0.570776</td>
          <td>26.416673</td>
          <td>0.128877</td>
          <td>26.048754</td>
          <td>0.082312</td>
          <td>25.638762</td>
          <td>0.093399</td>
          <td>25.362595</td>
          <td>0.139037</td>
          <td>24.674794</td>
          <td>0.169535</td>
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
          <td>29.010939</td>
          <td>1.896325</td>
          <td>26.700133</td>
          <td>0.188741</td>
          <td>25.919839</td>
          <td>0.086414</td>
          <td>25.355611</td>
          <td>0.086198</td>
          <td>25.053959</td>
          <td>0.124805</td>
          <td>24.797737</td>
          <td>0.220646</td>
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
          <td>27.573741</td>
          <td>0.348715</td>
          <td>26.814117</td>
          <td>0.297808</td>
          <td>27.342803</td>
          <td>0.752955</td>
          <td>26.867219</td>
          <td>1.001116</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>26.109464</td>
          <td>0.312686</td>
          <td>26.054480</td>
          <td>0.110585</td>
          <td>24.801578</td>
          <td>0.032855</td>
          <td>23.906825</td>
          <td>0.024572</td>
          <td>23.063860</td>
          <td>0.022193</td>
          <td>22.740986</td>
          <td>0.037581</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>27.703955</td>
          <td>0.995060</td>
          <td>27.766940</td>
          <td>0.468658</td>
          <td>27.516511</td>
          <td>0.353938</td>
          <td>26.458284</td>
          <td>0.237542</td>
          <td>26.802657</td>
          <td>0.545501</td>
          <td>25.994579</td>
          <td>0.594271</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.922459</td>
          <td>0.571338</td>
          <td>25.703055</td>
          <td>0.079675</td>
          <td>25.481302</td>
          <td>0.058660</td>
          <td>24.776083</td>
          <td>0.051632</td>
          <td>24.378830</td>
          <td>0.069044</td>
          <td>23.671633</td>
          <td>0.083731</td>
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
          <td>27.966604</td>
          <td>1.133552</td>
          <td>26.277851</td>
          <td>0.134063</td>
          <td>26.201934</td>
          <td>0.112984</td>
          <td>25.964450</td>
          <td>0.149633</td>
          <td>25.854192</td>
          <td>0.250768</td>
          <td>26.918977</td>
          <td>1.047544</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>28.365867</td>
          <td>1.397643</td>
          <td>27.196526</td>
          <td>0.285555</td>
          <td>26.728951</td>
          <td>0.174955</td>
          <td>26.190119</td>
          <td>0.178361</td>
          <td>25.766348</td>
          <td>0.229579</td>
          <td>25.055918</td>
          <td>0.273978</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>27.359791</td>
          <td>0.777259</td>
          <td>26.922928</td>
          <td>0.229899</td>
          <td>27.093932</td>
          <td>0.239488</td>
          <td>26.197890</td>
          <td>0.181089</td>
          <td>25.556449</td>
          <td>0.194230</td>
          <td>25.119315</td>
          <td>0.290781</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.578935</td>
          <td>0.452914</td>
          <td>27.059653</td>
          <td>0.261374</td>
          <td>26.537490</td>
          <td>0.152573</td>
          <td>25.689060</td>
          <td>0.119207</td>
          <td>25.435532</td>
          <td>0.178537</td>
          <td>25.346414</td>
          <td>0.354521</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>28.400138</td>
          <td>1.426266</td>
          <td>26.428324</td>
          <td>0.151110</td>
          <td>26.072268</td>
          <td>0.099788</td>
          <td>25.746519</td>
          <td>0.122597</td>
          <td>25.309097</td>
          <td>0.157026</td>
          <td>24.846001</td>
          <td>0.231916</td>
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
          <td>28.952694</td>
          <td>1.740753</td>
          <td>26.773518</td>
          <td>0.175018</td>
          <td>26.079288</td>
          <td>0.084569</td>
          <td>25.146215</td>
          <td>0.060452</td>
          <td>25.085162</td>
          <td>0.109301</td>
          <td>25.049285</td>
          <td>0.232252</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1.978239</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.837192</td>
          <td>0.840667</td>
          <td>28.576438</td>
          <td>0.641569</td>
          <td>27.208539</td>
          <td>0.350240</td>
          <td>26.552641</td>
          <td>0.371986</td>
          <td>27.029820</td>
          <td>0.985337</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.974287</td>
          <td>27.308438</td>
          <td>0.713122</td>
          <td>26.011112</td>
          <td>0.097232</td>
          <td>24.807943</td>
          <td>0.029816</td>
          <td>23.853526</td>
          <td>0.021142</td>
          <td>23.122613</td>
          <td>0.021112</td>
          <td>22.790710</td>
          <td>0.035310</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1.317978</td>
          <td>26.735269</td>
          <td>0.519426</td>
          <td>27.727517</td>
          <td>0.453745</td>
          <td>27.687239</td>
          <td>0.402892</td>
          <td>27.871324</td>
          <td>0.693705</td>
          <td>26.951662</td>
          <td>0.605053</td>
          <td>24.818342</td>
          <td>0.238692</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.386366</td>
          <td>26.070868</td>
          <td>0.268150</td>
          <td>25.646821</td>
          <td>0.065706</td>
          <td>25.372936</td>
          <td>0.045287</td>
          <td>24.847309</td>
          <td>0.046425</td>
          <td>24.383273</td>
          <td>0.058947</td>
          <td>23.670870</td>
          <td>0.070811</td>
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
          <td>26.269160</td>
          <td>0.330655</td>
          <td>26.622729</td>
          <td>0.164569</td>
          <td>26.493362</td>
          <td>0.131318</td>
          <td>26.156898</td>
          <td>0.158846</td>
          <td>26.292477</td>
          <td>0.324914</td>
          <td>25.634820</td>
          <td>0.399816</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.457508</td>
          <td>27.081817</td>
          <td>0.587304</td>
          <td>26.746919</td>
          <td>0.173477</td>
          <td>26.625725</td>
          <td>0.138483</td>
          <td>26.440557</td>
          <td>0.189805</td>
          <td>26.038050</td>
          <td>0.249759</td>
          <td>24.878529</td>
          <td>0.204717</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>1.372993</td>
          <td>26.904794</td>
          <td>0.526851</td>
          <td>27.467463</td>
          <td>0.322762</td>
          <td>26.863182</td>
          <td>0.175146</td>
          <td>26.615459</td>
          <td>0.226953</td>
          <td>26.030826</td>
          <td>0.255910</td>
          <td>25.342481</td>
          <td>0.309111</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>0.855022</td>
          <td>26.661676</td>
          <td>0.458175</td>
          <td>27.656137</td>
          <td>0.394144</td>
          <td>26.677211</td>
          <td>0.159323</td>
          <td>26.066012</td>
          <td>0.152412</td>
          <td>25.593657</td>
          <td>0.189276</td>
          <td>25.771030</td>
          <td>0.457928</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.723768</td>
          <td>26.427556</td>
          <td>0.365064</td>
          <td>26.554075</td>
          <td>0.149980</td>
          <td>26.130481</td>
          <td>0.091973</td>
          <td>25.622165</td>
          <td>0.095881</td>
          <td>24.830699</td>
          <td>0.090906</td>
          <td>25.352600</td>
          <td>0.308701</td>
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
